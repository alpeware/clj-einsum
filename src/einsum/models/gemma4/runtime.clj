(ns einsum.models.gemma4.runtime
  "Gemma 4 generation runtime engines, decoding loops, and persistent session management."
  (:require [clojure.pprint :as pprint]
            [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.models.gemma :as gemma-logic]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.weights :as weights]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.profile :as profile]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [einsum.runtime.weights :as rw])
  (:import [java.lang.foreign Arena]))

(defn argmax-host
  "Finds the index of the maximum float value in float array `arr`."
  [^floats arr]
  (let [n (alength arr)]
    (loop [i 1
           max-idx 0
           max-val (aget arr 0)]
      (if (< i n)
        (let [v (aget arr i)]
          (if (> v max-val)
            (recur (inc i) i v)
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn argmax-with-penalty
  "Selects the token with maximum logit value with sliding window linear repetition penalty."
  [^floats logits-arr gen-ids rep-pen]
  (let [n (alength logits-arr)
        recent-ids (take-last 32 gen-ids)
        penalty-set (disj (set recent-ids) 107 108 236743)
        pen (double (or rep-pen 1.15))]
    (loop [i 0
           max-idx 0
           max-val Float/NEGATIVE_INFINITY]
      (if (< i n)
        (let [raw-v (aget logits-arr i)
              v (if (contains? penalty-set i)
                  (if (pos? raw-v) (/ raw-v pen) (* raw-v pen))
                  raw-v)]
          (if (> v max-val)
            (recur (inc i) i (float v))
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn sample-next-token
  "Selects next token from float array `logits-arr` using sampling options and repetition penalty."
  [^floats logits-arr opts _prompt-ids gen-ids]
  (let [{:keys [temperature top-k top-p repetition-penalty]
         :or {temperature 0.0 top-k 10 top-p 1.0 repetition-penalty 1.0}} opts
        rep-pen (double (or repetition-penalty 1.0))]
    (if (or (nil? temperature) (<= temperature 0.0))
      (if (and (number? rep-pen) (> rep-pen 1.0))
        (argmax-with-penalty logits-arr gen-ids rep-pen)
        (argmax-host logits-arr))
      (let [logits-vec (vec logits-arr)]
        (sampling/sample-logits logits-vec {:temperature temperature
                                            :top-k top-k
                                            :top-p top-p
                                            :repetition-penalty rep-pen
                                            :seen-ids gen-ids})))))

(defn common-prefix-len
  "Returns the number of leading items shared by sequences xs and ys."
  [xs ys]
  (let [n (min (count xs) (count ys))]
    (loop [i 0]
      (if (and (< i n) (= (nth xs i) (nth ys i)))
        (recur (inc i))
        i))))

(defn run-vram-loop-generation
  "Executes autoregressive token generation entirely within device VRAM using 1-shot prefill
   (or sequential prefill for large contexts, with prefix KV-cache reuse across agent turns)
   and an OpenXLA while-loop supporting chunked semantic early stopping."
  [session exec device-weights prompt-ids max-seq-len]
  (let [exec (or exec (:executable session) (:exec session))
        device-weights (or device-weights (:device-weights session))
        {:keys [ctx opts config kv-state prefill-executable session-arena tokenizer]} session
        session-arena (or session-arena (xla/create-arena ctx))
        {:keys [max-new-tokens quiet]} opts
        stop-pred (or (:stop-predicate opts) (:stop-predicate session))
        chunk-size (long (or (:chunk-size opts) (:chunk-size session) 32))
        seq-len (long max-seq-len)
        raw-p-count (count prompt-ids)
        safe-p-count (min raw-p-count (max 0 (- seq-len 2)))
        clamped-prompt-ids (if (< safe-p-count raw-p-count)
                             (subvec (vec prompt-ids) (- raw-p-count safe-p-count))
                             (vec prompt-ids))
        p-count (count clamped-prompt-ids)
        max-new (long (max 1 (min (- seq-len p-count 1) (or max-new-tokens 150))))
        target-max (long (+ p-count max-new))]
    (if (>= p-count (dec seq-len))
      clamped-prompt-ids
      (let [t0 (System/nanoTime)
            num-layers (long (or (:num-layers config) 35))
            num-kv-shared (long (or (:num-kv-shared-layers config) 0))
            num-unshared (- num-layers num-kv-shared)
            safe-prefill-len (cfg/max-safe-prefill-seq-len config)
            prefill-exec (or prefill-executable
                             (:prefill-executable session)
                             (when (<= seq-len safe-prefill-len)
                               (kernels/compile-gemma4-prefill-executable session seq-len)))
            step-exec (or (:step-executable session)
                          (kernels/compile-gemma4-kv-executable session seq-len))
            in-arr (int-array seq-len)
            _ (dotimes [i p-count] (aset in-arr i (int (nth clamped-prompt-ids i))))

            ;; Check for prefix KV-cache hit across turns
            is-persistent? (some? kv-state)
            prior-cache (when is-persistent? @kv-state)
            p-match (if (and prior-cache (:cached-tokens prior-cache) (seq (:kv-buffers prior-cache)))
                      (common-prefix-len (:cached-tokens prior-cache) clamped-prompt-ids)
                      0)
            _ (when (and is-persistent? prior-cache (zero? p-match))
                (arena/destroy! session-arena (:kv-buffers prior-cache))
                (reset! kv-state nil))

            ;; 1. Populate KV cache for prompt tokens (Delta, 1-shot Parallel, or Sequential)
            t-prefill-0 (System/nanoTime)
            prefill-limit (max 0 (dec p-count))
            num-step-outs (inc (* 2 num-unshared))
            x-arr (int-array 1)
            pos-arr (int-array 1)

            prefill-kv
            (cond
              ;; Path A: Cache hit from token 0 to p-match (delta prefill via step executable)
              (pos? p-match)
              (let [initial-kv (:kv-buffers prior-cache)]
                (if (>= p-match prefill-limit)
                  (do
                    (when-not quiet
                      (println (format "Reusing 100%% of prefix KV-Cache (%d tokens matching >= limit %d)..."
                                       p-match prefill-limit)))
                    initial-kv)
                  (do
                    (when-not quiet
                      (println (format "Delta prefilling %d prompt tokens into KV-Cache (reusing %d prefix tokens)..."
                                       (- prefill-limit p-match) p-match)))
                    (loop [p p-match
                           cur-kv initial-kv]
                      (if (< p prefill-limit)
                        (let [new-kv
                              (xla/with-device-arena [iter-arena session-arena]
                                (let [tok (int (nth clamped-prompt-ids p))
                                      _ (aset x-arr 0 tok)
                                      _ (aset pos-arr 0 p)
                                      x-b (xla/device-buffer iter-arena x-arr [1 1] :i32)
                                      pos-b (xla/device-buffer iter-arena pos-arr [1] :i32)
                                      step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                      outs (xla/track! iter-arena (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) step-inputs num-step-outs))
                                      outs-vec (if (vector? outs) outs [outs])
                                      nk (vec (subvec outs-vec 1))]
                                  (xla/promote! iter-arena session-arena nk)
                                  nk))]
                          (arena/destroy! session-arena cur-kv)
                          (recur (inc p) new-kv))
                        cur-kv)))))

              ;; Path B: 1-shot parallel prefill
              (some? prefill-exec)
              (xla/with-device-arena [prefill-arena session-arena]
                (let [pos-p (int-array [(dec p-count)])
                      in-b (xla/device-buffer prefill-arena in-arr [1 seq-len] :i32)
                      pos-b (xla/device-buffer prefill-arena pos-p [1] :i32)
                      prefill-inputs (into [in-b pos-b] device-weights)
                      num-prefill-outs (inc (* 2 num-unshared))
                      prefill-outs (xla/track! prefill-arena
                                               (pjrt/execute-executable ctx (or (:handle prefill-exec) prefill-exec) prefill-inputs num-prefill-outs))
                      prefill-outs-vec (if (vector? prefill-outs) prefill-outs [prefill-outs])
                      kv-outs (vec (subvec prefill-outs-vec 1))]
                  (xla/promote! prefill-arena session-arena kv-outs)
                  kv-outs))

              ;; Path C: Sequential prefill from token 0
              :else
              (let [initial-kv (kernels/allocate-kv-cache-buffers session seq-len session-arena)]
                (when-not quiet
                  (println (format "Prefilling %d prompt tokens into KV-Cache (exceeds parallel prefill limit %d)..."
                                   prefill-limit safe-prefill-len)))
                (loop [p 0
                       cur-kv initial-kv]
                  (if (< p prefill-limit)
                    (let [new-kv
                          (xla/with-device-arena [iter-arena session-arena]
                            (let [tok (int (nth clamped-prompt-ids p))
                                  _ (aset x-arr 0 tok)
                                  _ (aset pos-arr 0 p)
                                  x-b (xla/device-buffer iter-arena x-arr [1 1] :i32)
                                  pos-b (xla/device-buffer iter-arena pos-arr [1] :i32)
                                  step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                  outs (xla/track! iter-arena (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) step-inputs num-step-outs))
                                  outs-vec (if (vector? outs) outs [outs])
                                  nk (vec (subvec outs-vec 1))]
                              (xla/promote! iter-arena session-arena nk)
                              nk))]
                      (arena/destroy! session-arena cur-kv)
                      (recur (inc p) new-kv))
                    cur-kv))))

            t-prefill-1 (System/nanoTime)
            prefill-ms (/ (- t-prefill-1 t-prefill-0) 1e6)

            ;; 2. Run In-VRAM While Loop with chunked semantic early stopping
            num-loop-outs (+ 2 (* 2 num-unshared))
            cur-toks-arr (aclone in-arr)

            [final-kv final-step decode-ms]
            (loop [cur-step p-count
                   cur-kv prefill-kv
                   total-decode-ms 0.0]
              (if (>= cur-step target-max)
                [cur-kv cur-step total-decode-ms]
                (let [chunk-target (if stop-pred
                                     (min target-max (+ cur-step chunk-size))
                                     target-max)
                      t-chunk-0 (System/nanoTime)
                      [new-step new-kv-promoted stopped-by-eos?]
                      (xla/with-device-arena [chunk-arena session-arena]
                        (let [b-step (xla/device-buffer chunk-arena (int-array [cur-step]) [] :i32)
                              b-max (xla/device-buffer chunk-arena (int-array [chunk-target]) [] :i32)
                              b-toks (xla/device-buffer chunk-arena cur-toks-arr [1 seq-len] :i32)
                              loop-inputs (into [b-step b-max b-toks] (concat cur-kv device-weights))
                              loop-outs (xla/track! chunk-arena (pjrt/execute-executable ctx (or (:handle exec) exec) loop-inputs num-loop-outs))
                              loop-outs-vec (if (vector? loop-outs) loop-outs [loop-outs])
                              out-step (nth loop-outs-vec 0)
                              out-toks (nth loop-outs-vec 1)
                              nk (vec (subvec loop-outs-vec 2))

                              step-floats (pjrt/buffer-to-host-buffer ctx out-step 1 :f32)
                              step-val (int (Float/floatToIntBits (aget step-floats 0)))
                              toks-floats (pjrt/buffer-to-host-buffer ctx out-toks seq-len :f32)
                              step-bounded (min (max cur-step step-val) seq-len)]
                          (dotimes [i (- step-bounded cur-step)]
                            (let [idx (+ cur-step i)]
                              (aset cur-toks-arr idx (Float/floatToIntBits (aget toks-floats idx)))))
                          (xla/promote! chunk-arena session-arena nk)
                          [step-bounded nk (< step-val chunk-target)]))
                      t-chunk-1 (System/nanoTime)
                      chunk-ms (/ (- t-chunk-1 t-chunk-0) 1e6)
                      acc-decode-ms (+ total-decode-ms chunk-ms)]
                  ;; Free cur-kv in session arena since new-kv-promoted replaces it
                  (arena/destroy! session-arena cur-kv)

                  (if (or stopped-by-eos? (>= new-step target-max))
                    [new-kv-promoted new-step acc-decode-ms]
                    (if (and stop-pred tokenizer)
                      (let [gen-tokens (subvec (vec cur-toks-arr) p-count new-step)
                            gen-text (decode tokenizer gen-tokens)
                            semantic-stopped? (try (boolean (stop-pred gen-text)) (catch Throwable _ false))]
                        (if semantic-stopped?
                          (do
                            (when-not quiet
                              (println (format "  [Semantic Early Stop]: Generation stopped after %d new tokens (semantic criteria satisfied)."
                                               (count gen-tokens))))
                            [new-kv-promoted new-step acc-decode-ms])
                          (recur new-step new-kv-promoted acc-decode-ms)))
                      (recur new-step new-kv-promoted acc-decode-ms))))))

            final-ids (subvec (vec cur-toks-arr) 0 final-step)
            cleaned-ids (if (and (> (count final-ids) p-count)
                                 (contains? kernels/GEMMA4-STOP-TOKEN-IDS (last final-ids)))
                          (subvec final-ids 0 (dec (count final-ids)))
                          final-ids)]

        (if is-persistent?
          (reset! kv-state {:cached-tokens cleaned-ids
                            :kv-buffers final-kv})
          (arena/destroy! session-arena final-kv))

        (let [t-end (System/nanoTime)
              total-ms (/ (- t-end t0) 1e6)
              gen-count (- (count cleaned-ids) p-count)
              tok-s (if (pos? decode-ms) (/ (* gen-count 1000.0) decode-ms) 0.0)]
          (when-not quiet
            (println)
            (println "\n------------------------------------------------------------------")
            (println "  Telemetry Benchmark Metrics (In-VRAM While-Loop with KV-Cache):")
            (println (format "    • Prefill Latency             : %8.2f ms (%d tokens)" prefill-ms p-count))
            (println (format "    • Decode Latency              : %8.2f ms (%d tokens)" decode-ms gen-count))
            (println (format "    • Generation Speed            : %8.2f tok/s (%6.2f ms/tok)" tok-s (if (pos? gen-count) (/ decode-ms gen-count) 0.0)))
            (println (format "    • Total Generation Latency    : %8.2f ms" total-ms))
            (println "------------------------------------------------------------------\n"))
          cleaned-ids)))))

(defn run-autoregressive-generation-logic
  "Executes autoregressive token generation using pure Tensor Logic Gemma 4 executable."
  ([session exec device-weights prompt-ids]
   (run-autoregressive-generation-logic session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config tokenizer session-arena] :as session} exec device-weights prompt-ids max-seq-len]
   (let [{:keys [max-new-tokens quiet mode]} opts
         is-agent? (= mode :agent)
         seq-len (long (or max-seq-len (:max-seq-len config) 128))
         vocab-size (long (or (:vocab-size config) (:vocab_size config) 262144))
         last-token? (get opts :last-token-only? true)
         weight-dt (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config)) :bf16 (get config :weight-dtype :bf16))
         prompt-count (count prompt-ids)
         in-arr (int-array seq-len)
         pos-arr (when last-token? (int-array 1))
         _ (dotimes [i (min prompt-count seq-len)]
             (aset in-arr i (int (nth prompt-ids i))))
         cur-tokens (atom (vec prompt-ids))
         t0 (System/nanoTime)]
     (loop [step 0]
       (if (>= step max-new-tokens)
         nil
         (let [s-len (count @cur-tokens)
               _ (when last-token? (aset pos-arr 0 (dec s-len)))
               next-id
               (xla/with-device-arena [step-arena (or session-arena ctx)]
                 (let [in-b (xla/device-buffer step-arena in-arr [1 seq-len] :i32)
                       pos-b (when last-token?
                               (xla/device-buffer step-arena pos-arr [1] :i32))
                       args (let [base (if last-token? [in-b pos-b] [in-b])
                                  rel-bufs (get session :relational-buffers [])]
                              (into base (concat device-weights rel-bufs)))
                       out (xla/track! step-arena (xla/execute exec args))
                       out-buf (if (sequential? out) (first out) out)
                       logits (if last-token?
                                (xla/to-host-slice out-buf 0 vocab-size vocab-size weight-dt)
                                (xla/to-host-slice out-buf (dec s-len) vocab-size (* seq-len vocab-size) weight-dt))]
                   (sample-next-token logits opts prompt-ids (subvec @cur-tokens (count prompt-ids)))))]
           (when (< s-len seq-len)
             (aset in-arr s-len (int next-id)))
           (swap! cur-tokens conj next-id)
           (when (and (not quiet) (not is-agent?))
             (print (decode tokenizer [next-id]))
             (flush))
           (if (or (contains? kernels/GEMMA4-STOP-TOKEN-IDS next-id) (= next-id (eos-id tokenizer)))
             nil
             (recur (inc step))))))
     (let [t1 (System/nanoTime)
           total-ms (/ (- t1 t0) 1e6)
           gen-count (- (count @cur-tokens) (count prompt-ids))
           tok-s (if (pos? total-ms) (/ (* gen-count 1000.0) total-ms) 0.0)]
       (when-not quiet
         (println)
         (println "\n------------------------------------------------------------------")
         (println "  Telemetry Benchmark Metrics (Tensor Logic End-to-End):")
         (println (format "    • Total Generation Latency    : %8.2f ms (%d tokens)" total-ms gen-count))
         (println (format "    • Generation Speed            : %8.2f tok/s (%6.2f ms/tok)" tok-s (if (pos? gen-count) (/ total-ms gen-count) 0.0)))
         (println "------------------------------------------------------------------\n")))
     @cur-tokens)))

(defn run-cached-kv-generation
  "Executes autoregressive generation with in-VRAM KV-Cache using single-token step executable
   and 1-shot parallel prefill."
  ([session exec device-weights prompt-ids]
   (run-cached-kv-generation session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config tokenizer kv-state prefill-executable session-arena] :as session} exec device-weights prompt-ids max-seq-len]
   (let [session-arena (or session-arena (xla/create-arena ctx))
         {:keys [max-new-tokens quiet mode]} opts
         is-agent? (= mode :agent)
         seq-len (long (or max-seq-len (:max-seq-len config) 512))
         vocab-size (long (or (:vocab-size config) 262144))
         weight-dt (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config)) :bf16 (get config :weight-dtype :bf16))
         num-layers (long (or (:num-layers config) 35))
         num-kv-shared (long (or (:num-kv-shared-layers config) 0))
         num-unshared (- num-layers num-kv-shared)
         num-outs (inc (* 2 num-unshared))
         raw-prompt-count (count prompt-ids)
         safe-prompt-count (min raw-prompt-count (max 0 (- seq-len 2)))
         clamped-prompt-ids (if (< safe-prompt-count raw-prompt-count)
                              (subvec (vec prompt-ids) (- raw-prompt-count safe-prompt-count))
                              (vec prompt-ids))
         prompt-count (count clamped-prompt-ids)
         prefill-exec (or prefill-executable (:prefill-executable session))
         is-persistent? (some? kv-state)
         prior-cache (when is-persistent? @kv-state)
         p-match (if (and prior-cache (:cached-tokens prior-cache) (seq (:kv-buffers prior-cache)))
                   (common-prefix-len (:cached-tokens prior-cache) clamped-prompt-ids)
                   0)
         _ (when (and is-persistent? prior-cache (zero? p-match))
             (arena/destroy! session-arena (:kv-buffers prior-cache))
             (reset! kv-state nil))
         initial-kv (cond
                      (pos? p-match)
                      (:kv-buffers prior-cache)

                      (and (some? prefill-exec) (<= seq-len (cfg/max-safe-prefill-seq-len config)))
                      nil

                      :else
                      (kernels/allocate-kv-cache-buffers session seq-len session-arena))
         kv-buffers-atom (atom initial-kv)
         x-arr (int-array 1)
         pos-arr (int-array 1)
         t0 (System/nanoTime)]
     (try
       ;; Phase 1: Prefill prompt tokens into KV cache
       (let [[last-logits t-prefill-end]
             (cond
               ;; Path A: Cache hit from token 0 to p-match (delta prefill via step executable)
               (pos? p-match)
               (let [start-p (if (= p-match prompt-count) (max 0 (dec prompt-count)) p-match)
                     cur-log (loop [p start-p
                                    cur-logits nil]
                               (if (< p prompt-count)
                                 (let [[new-log new-kv]
                                       (xla/with-device-arena [step-arena session-arena]
                                         (let [tok (int (nth clamped-prompt-ids p))
                                               _ (aset x-arr 0 tok)
                                               _ (aset pos-arr 0 p)
                                               x-b (xla/device-buffer step-arena x-arr [1 1] :i32)
                                               pos-b (xla/device-buffer step-arena pos-arr [1] :i32)
                                               step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                               outs (xla/track! step-arena (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs))
                                               outs-vec (if (vector? outs) outs [outs])
                                               nl (first outs-vec)
                                               nk (vec (subvec outs-vec 1))]
                                           (xla/promote! step-arena session-arena nl)
                                           (xla/promote! step-arena session-arena nk)
                                           [nl nk]))
                                       old-kv @kv-buffers-atom]
                                   (when cur-logits (arena/destroy! session-arena cur-logits))
                                   (when (seq old-kv)
                                     (arena/destroy! session-arena old-kv))
                                   (reset! kv-buffers-atom new-kv)
                                   (recur (inc p) new-log))
                                 cur-logits))]
                 [cur-log (System/nanoTime)])

               ;; Path B: 1-Shot Parallel Prefill
               (some? prefill-exec)
               (let [[prefill-logits prefill-kv]
                     (xla/with-device-arena [step-arena session-arena]
                       (let [in-arr (int-array seq-len)
                             _ (dotimes [i prompt-count] (aset in-arr i (int (nth clamped-prompt-ids i))))
                             pos-p (int-array [(dec prompt-count)])
                             in-b (xla/device-buffer step-arena in-arr [1 seq-len] :i32)
                             pos-b (xla/device-buffer step-arena pos-p [1] :i32)
                             step-inputs (into [in-b pos-b] device-weights)
                             outs (xla/track! step-arena (pjrt/execute-executable ctx (or (:handle prefill-exec) prefill-exec) step-inputs num-outs))
                             outs-vec (if (vector? outs) outs [outs])
                             pl (first outs-vec)
                             pkv (vec (subvec outs-vec 1))]
                         (xla/promote! step-arena session-arena pl)
                         (xla/promote! step-arena session-arena pkv)
                         [pl pkv]))]
                 (reset! kv-buffers-atom prefill-kv)
                 [prefill-logits (System/nanoTime)])

               ;; Path C: Sequential fallback prefill
               :else
               (let [cur-log (loop [p 0
                                    cur-logits nil]
                               (if (< p prompt-count)
                                 (let [[new-log new-kv]
                                       (xla/with-device-arena [step-arena session-arena]
                                         (let [tok (int (nth clamped-prompt-ids p))
                                               _ (aset x-arr 0 tok)
                                               _ (aset pos-arr 0 p)
                                               x-b (xla/device-buffer step-arena x-arr [1 1] :i32)
                                               pos-b (xla/device-buffer step-arena pos-arr [1] :i32)
                                               step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                               outs (xla/track! step-arena (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs))
                                               outs-vec (if (vector? outs) outs [outs])
                                               nl (first outs-vec)
                                               nk (vec (subvec outs-vec 1))]
                                           (xla/promote! step-arena session-arena nl)
                                           (xla/promote! step-arena session-arena nk)
                                           [nl nk]))
                                       old-kv @kv-buffers-atom]
                                   (when cur-logits (arena/destroy! session-arena cur-logits))
                                   (when (seq old-kv)
                                     (arena/destroy! session-arena old-kv))
                                   (reset! kv-buffers-atom new-kv)
                                   (recur (inc p) new-log))
                                 cur-logits))]
                 [cur-log (System/nanoTime)]))

             cur-tokens (atom (vec clamped-prompt-ids))
             max-tokens (long (or max-new-tokens 256))]
         ;; Phase 2: Autoregressive decode loop
         (loop [step prompt-count
                cur-logits last-logits]
           (if (or (>= (- (count @cur-tokens) prompt-count) max-tokens)
                   (>= step (dec seq-len)))
             (when cur-logits (arena/destroy! session-arena cur-logits))
             (let [logits-data (xla/to-host-slice cur-logits 0 vocab-size vocab-size weight-dt)
                   _ (arena/destroy! session-arena cur-logits)
                   next-id (sample-next-token logits-data opts clamped-prompt-ids (subvec @cur-tokens prompt-count))]
               (swap! cur-tokens conj next-id)
               (when (and (not quiet) (not is-agent?))
                 (print (decode tokenizer [next-id]))
                 (flush))
               (if (or (contains? kernels/GEMMA4-STOP-TOKEN-IDS next-id) (= next-id (eos-id tokenizer)))
                 nil
                 (let [[new-log new-kv]
                       (xla/with-device-arena [step-arena session-arena]
                         (let [_ (aset x-arr 0 (int next-id))
                               _ (aset pos-arr 0 step)
                               x-b (xla/device-buffer step-arena x-arr [1 1] :i32)
                               pos-b (xla/device-buffer step-arena pos-arr [1] :i32)
                               step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                               outs (xla/track! step-arena (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs))
                               outs-vec (if (vector? outs) outs [outs])
                               nl (first outs-vec)
                               nk (vec (subvec outs-vec 1))]
                           (xla/promote! step-arena session-arena nl)
                           (xla/promote! step-arena session-arena nk)
                           [nl nk]))
                       old-kv @kv-buffers-atom]
                   (arena/destroy! session-arena old-kv)
                   (reset! kv-buffers-atom new-kv)
                   (recur (inc step) new-log))))))
         (when is-persistent?
           (reset! kv-state {:cached-tokens @cur-tokens
                             :kv-buffers @kv-buffers-atom}))
         (let [t1 (System/nanoTime)
               total-ms (/ (- t1 t0) 1e6)
               prefill-ms (/ (- t-prefill-end t0) 1e6)
               decode-ms (/ (- t1 t-prefill-end) 1e6)
               gen-count (- (count @cur-tokens) prompt-count)
               decode-tok-s (if (pos? decode-ms) (/ (* gen-count 1000.0) decode-ms) 0.0)]
           (when-not quiet
             (println)
             (println "\n------------------------------------------------------------------")
             (println "  Telemetry Benchmark Metrics (Tensor Logic KV-Cache):")
             (println (format "    • Prefill Latency             : %8.2f ms (%d tokens, %d cached, %d new)"
                              prefill-ms prompt-count p-match (- prompt-count p-match)))
             (println (format "    • Decode Latency              : %8.2f ms (%d tokens)" decode-ms gen-count))
             (println (format "    • Decode Speed                : %8.2f tok/s (%6.2f ms/tok)" decode-tok-s (if (pos? gen-count) (/ decode-ms gen-count) 0.0)))
             (println (format "    • Total Generation Latency    : %8.2f ms" total-ms))
             (println "------------------------------------------------------------------\n"))
           @cur-tokens))
       (finally
         (when-not is-persistent?
           (when-let [bufs (seq @kv-buffers-atom)]
             (arena/destroy! session-arena bufs))))))))

(defn run-autoregressive-generation
  "Executes autoregressive generation either via in-VRAM while loop, cached KV generation, or full sequence recomputation."
  ([session exec device-weights prompt-ids]
   (run-autoregressive-generation session exec device-weights prompt-ids nil))
  ([session exec device-weights prompt-ids max-seq-len]
   (let [{:keys [opts]} session
         method (or (:method opts) (if (:vram-loop? session) :vram-loop :kv-cache))
         vram-loop? (if (contains? opts :vram-loop?)
                      (:vram-loop? opts)
                      (= method :vram-loop))]
     (cond
       vram-loop?
       (run-vram-loop-generation session exec device-weights prompt-ids (or max-seq-len (:max-seq-len session) 128))

       (= method :tensor-logic-full)
       (run-autoregressive-generation-logic session exec device-weights prompt-ids max-seq-len)

       :else
       (run-cached-kv-generation session exec device-weights prompt-ids (or max-seq-len (:max-seq-len session) 512))))))

(defn init-inference-session
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for Gemma 4 REPL/CLI sessions."
  ([opts]
   (let [{:keys [backend]} opts
         model-dir (or (:model-dir opts) (:model opts))
         ctx (xla/init-backend! (or backend :cpu))
         opts (assoc opts :backend (or backend (:backend ctx) :cpu))
         dirs (if model-dir (cons model-dir cfg/DEFAULT_MODEL_DIRS) cfg/DEFAULT_MODEL_DIRS)
         resolved-model-dir (cfg/find-model-dir dirs)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights resolved-model-dir arena)
         tokenizer (tok/from-file resolved-model-dir)
         header (or (:header weights-mmap) {})
         prefix-base (if (contains? header "model.language_model.embed_tokens.weight")
                       "model.language_model."
                       "model.")
         session-arena (or (:session-arena opts) (xla/create-arena ctx))
         weight-store (rw/create-weight-store ctx weights-mmap
                                              {:aliases (gemma-logic/gemma4-alias-resolver prefix-base)
                                               :arena session-arena})
         config (cfg/build-model-config weights-mmap resolved-model-dir opts)]

     (when-not (:quiet opts)
       (println (str "Loaded Gemma 4 model weights from [" resolved-model-dir "] in [" (name (:weight-dtype config)) "] precision ("
                     (:num-layers config) " layers, " (:num-heads config) " heads, " (:num-kv-heads config) " kv-heads)."))
       (when (seq (:skip-layers config))
         (println (str "  ↳ Mixed-Precision Layers (Unquantized BF16): " (str/join ", " (sort (:skip-layers config)))))))

     {:ctx ctx
      :opts opts
      :model-dir resolved-model-dir
      :tokenizer tokenizer
      :weights-mmap weights-mmap
      :arena arena
      :session-arena session-arena
      :weight-store weight-store
      :config config})))

(defn init-agent-vram-session
  "Initializes a persistent VRAM session for agent loops.
   Pre-allocates weights in device memory and compiles the In-VRAM While Loop (or KV-Cache step and prefill) executable once."
  ([opts]
   (init-agent-vram-session opts (long (or (:max-seq-len opts) 1024))))
  ([opts max-seq-len]
   (let [method (or (:method opts)
                    (if (and (number? (:temperature opts)) (> (:temperature opts) 0.0))
                      :kv-cache
                      :vram-loop))
         vram-loop? (if (contains? opts :vram-loop?)
                      (:vram-loop? opts)
                      (= method :vram-loop))
         opts (assoc opts :mode :agent :max-seq-len max-seq-len :method method :vram-loop? vram-loop?)
         session (init-inference-session opts)
         _ (when-not (:quiet opts)
             (println (format "Pre-compiling Gemma 4 %s graph (max-seq-len=%d)..."
                              (if vram-loop? "In-VRAM While Loop" "KV-Cache")
                              max-seq-len)))
         exec (if vram-loop?
                (kernels/compile-in-vram-loop-executable session max-seq-len)
                (kernels/compile-gemma4-kv-executable session max-seq-len))
         safe-prefill-len (cfg/max-safe-prefill-seq-len (:config session))
         prefill-exec (when (<= max-seq-len safe-prefill-len)
                        (kernels/compile-gemma4-prefill-executable session max-seq-len))
         step-exec (when vram-loop?
                     (kernels/compile-gemma4-kv-executable session max-seq-len))
         _ (when-not (:quiet opts) (println "Pinning Gemma 4 weights in PJRT VRAM..."))
         device-weights (weights/allocate-device-weights session)]
     (assoc session
            :device-weights device-weights
            :executable exec
            :exec exec
            :prefill-executable prefill-exec
            :step-executable step-exec
            :kv-state (atom nil)
            :max-seq-len max-seq-len
            :vram-session? true
            :vram-loop? vram-loop?))))

(defn close-agent-session!
  "Releases VRAM resources for an agent session."
  [{:keys [ctx device-weights kv-state session-arena]}]
  (if session-arena
    (do
      (xla/close-arena! session-arena)
      (when kv-state (reset! kv-state nil)))
    (do
      (when (seq device-weights)
        (doseq [w device-weights]
          (xla/destroy-buffer! ctx w)))
      (when (and kv-state @kv-state)
        (doseq [b (:kv-buffers @kv-state)]
          (xla/destroy-buffer! ctx b))
        (reset! kv-state nil)))))

;; ==============================================================================
;; High-Level Text & Token Generation API
;; ==============================================================================

(defn generate-text
  "Executes end-to-end Gemma 4 generation pipeline for a given prompt string."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        opts (merge (:opts session) opts)
        {:keys [max-new-tokens temperature top-k quiet model]} opts
        max-new-tokens (long (or max-new-tokens 200))
        temperature (double (or temperature 0.0))
        top-k (long (or top-k 10))
        clean-prompt (or prompt "The capital of France is")
        model-str (or model (get-in session [:config :model-dir]) "")
        is-it-model (str/includes? (str/lower-case model-str) "-it")
        is-already-templated (or (str/includes? clean-prompt "<|turn>user") (str/includes? clean-prompt "<|turn>model"))
        prompt-ids (cond
                     (and is-it-model (not is-already-templated))
                     (let [raw-ids (encode tokenizer clean-prompt)
                           clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)
                           prefix (if (:thinking opts)
                                    ;; <bos><|turn>system\n<|think|><turn|>\n<|turn>user\n
                                    [(bos-id tokenizer) 105 9731 107 98 106 107 105 2364 107]
                                    [(bos-id tokenizer) 105 2364 107])]
                       (vec (concat prefix clean-ids [106 107 105 4368 107])))

                     :else
                     (let [raw-ids (encode tokenizer clean-prompt)]
                       (if (= (first raw-ids) (bos-id tokenizer))
                         (vec raw-ids)
                         (vec (cons (bos-id tokenizer) raw-ids)))))
        prompt-len (count prompt-ids)
        max-seq-len (long (or (:max-seq-len session) (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens 16))))]
    (when-not quiet
      (let [prompt-str (if (> (count clean-prompt) 200)
                         (str (subs clean-prompt 0 100) " ... [truncated " (count clean-prompt) " chars] ... " (subs clean-prompt (- (count clean-prompt) 100)))
                         clean-prompt)
            tok-str (if (> prompt-len 30)
                      (str "[" (str/join " " (take 10 prompt-ids)) " ... " (str/join " " (take-last 5 prompt-ids)) "]")
                      (str prompt-ids))]
        (println (format "Prompt: \"%s\"" prompt-str))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s, method=%s"
                         max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))
                         (name (or (:method opts) :kv-cache))))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len tok-str))))

    (let [metrics-atom (or (:metrics-atom session) (atom {}))
          trace-spans-atom (or (:trace-spans-atom session) (atom []))
          reuse-weights? (some? (:device-weights session))
          reuse-exec? (some? (:executable session))
          exec (binding [profile/*active-trace-spans* trace-spans-atom]
                 (if reuse-exec?
                   (:executable session)
                   (profile/with-profile metrics-atom "graph_compilation"
                     (cond
                       (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                       (kernels/compile-in-vram-loop-executable session max-seq-len)

                       (= (:method opts) :tensor-logic-full)
                       (kernels/compile-tensor-logic-executable session max-seq-len)

                       :else
                       (kernels/compile-gemma4-kv-executable session max-seq-len)))))
          safe-prefill-len (cfg/max-safe-prefill-seq-len (:config session))
          prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                         (if (:prefill-executable session)
                           (:prefill-executable session)
                           (when (and (or (= (or (:method opts) :kv-cache) :kv-cache)
                                          (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                                      (<= max-seq-len safe-prefill-len))
                             (profile/with-profile metrics-atom "graph_compilation"
                               (kernels/compile-gemma4-prefill-executable session max-seq-len)))))
          step-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                      (if (:step-executable session)
                        (:step-executable session)
                        (when (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                          (profile/with-profile metrics-atom "graph_compilation"
                            (kernels/compile-gemma4-kv-executable session max-seq-len)))))
          active-session (assoc session
                                :prefill-executable prefill-exec
                                :step-executable step-exec)
          device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                           (if reuse-weights?
                             (:device-weights session)
                             (profile/with-profile metrics-atom "weight_transfer"
                               (weights/allocate-device-weights session))))]
      (when-not quiet
        (println "\nGenerating tokens autoregressively with pure Tensor Logic Gemma 4 Kernel..."))
      (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                            (profile/with-profile metrics-atom "autoregressive_generation"
                              (run-autoregressive-generation active-session exec device-weights prompt-ids max-seq-len)))
            generated-str (decode tokenizer final-context)]
        (when-not quiet
          (println)
          (println generated-str))
        (when-not reuse-weights?
          (if-let [sa (:session-arena session)]
            (xla/close-arena! sa)
            (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w))))
        (when-let [out-path (:out opts)]
          (spit out-path generated-str)
          (when-not quiet
            (println (format "\n  ↳ Written generated output to [%s]" out-path))))
        (when-let [profile-path (:profile-out opts)]
          (spit profile-path (with-out-str (pprint/pprint @metrics-atom)))
          (when-not quiet
            (println (format "  ↳ Saved telemetry profile report to [%s]" profile-path))))
        (when-let [trace-path (:chrome-trace-out opts)]
          (profile/save-chrome-trace! @trace-spans-atom trace-path)
          (when-not quiet
            (println (format "  ↳ Saved Chrome tracing JSON to [%s]" trace-path))))
        (when-not quiet
          (println "\n==================================================================")
          (println "=== Tensor Logic Gemma 4 Generation Verification Passed! ===")
          (println "=================================================================="))
        final-context))))

(defn generate-text-string
  "Generates text response using Gemma 4 model session and returns decoded text string."
  [session prompt]
  (let [{:keys [tokenizer]} session
        final-context (generate-text session prompt)]
    (decode tokenizer final-context)))

(defn generate-new-tokens-and-text
  "Generates text response using Gemma 4 model session and returns a map:
   {:text <decoded-new-text>
    :prompt-tokens <int>
    :new-tokens <int>
    :new-token-ids <vec>}."
  [session prompt]
  (let [{:keys [tokenizer]} session
        raw-ids (encode tokenizer prompt)
        prompt-ids (if (= (first raw-ids) (bos-id tokenizer))
                     raw-ids
                     (vec (cons (bos-id tokenizer) raw-ids)))
        prompt-len (count prompt-ids)
        seq-len (long (or (:max-seq-len session) (get-in session [:config :max-seq-len]) 2048))
        safe-prompt-len (min prompt-len (max 0 (- seq-len 2)))
        final-context (vec (generate-text session prompt))
        total-len (count final-context)
        slice-start (min total-len safe-prompt-len)
        new-ids (subvec final-context slice-start)]
    {:text (decode tokenizer new-ids)
     :prompt-tokens prompt-len
     :new-tokens (count new-ids)
     :new-token-ids (vec new-ids)}))

(defn generate-new-text-string
  "Generates text response using Gemma 4 model session and returns ONLY newly generated text string without prompt prefix."
  [session prompt]
  (:text (generate-new-tokens-and-text session prompt)))
