(ns einsum.models.gemma4.runtime
  "Gemma 4 generation runtime engines, decoding loops, and persistent session management."
  (:require [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.models.gemma :as gemma-logic]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.weights :as weights]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [decode eos-id]]
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
   (or sequential prefill for large contexts) and an OpenXLA while-loop carrying KV-Cache."
  [session exec device-weights prompt-ids max-seq-len]
  (let [{:keys [ctx opts config kv-state prefill-executable session-arena]} session
        session-arena (or session-arena (xla/create-arena ctx))
        {:keys [max-new-tokens quiet]} opts
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
            in-arr (int-array seq-len)
            _ (dotimes [i p-count] (aset in-arr i (int (nth clamped-prompt-ids i))))

            ;; 1. Populate KV cache for prompt tokens (Parallel or Sequential) in a scoped step arena
            t-prefill-0 (System/nanoTime)
            prefill-kv (xla/with-device-arena [prefill-arena session-arena]
                         (if (some? prefill-exec)
                           ;; Path A: 1-shot parallel prefill
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
                             kv-outs)
                           ;; Path B: Sequential prefill for sequence lengths exceeding VRAM parallel workspace headroom
                           (let [step-exec (or (:step-executable session)
                                               (kernels/compile-gemma4-kv-executable session seq-len))
                                 initial-kv (kernels/allocate-kv-cache-buffers session seq-len prefill-arena)
                                 num-step-outs (inc (* 2 num-unshared))
                                 x-arr (int-array 1)
                                 pos-arr (int-array 1)
                                 prefill-limit (max 0 (dec p-count))]
                             (when-not quiet
                               (println (format "Prefilling %d prompt tokens into KV-Cache (exceeds parallel prefill limit %d)..."
                                                prefill-limit safe-prefill-len)))
                             (let [final-prefill-kv
                                   (loop [p 0
                                          cur-kv initial-kv]
                                     (if (< p prefill-limit)
                                       (let [new-kv
                                             (xla/with-device-arena [iter-arena prefill-arena]
                                               (let [tok (int (nth clamped-prompt-ids p))
                                                     _ (aset x-arr 0 tok)
                                                     _ (aset pos-arr 0 p)
                                                     x-b (xla/device-buffer iter-arena x-arr [1 1] :i32)
                                                     pos-b (xla/device-buffer iter-arena pos-arr [1] :i32)
                                                     step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                                     outs (xla/track! iter-arena (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) step-inputs num-step-outs))
                                                     outs-vec (if (vector? outs) outs [outs])
                                                     nk (vec (subvec outs-vec 1))]
                                                 (xla/promote! iter-arena prefill-arena nk)
                                                 (arena/destroy! prefill-arena cur-kv)
                                                 nk))]
                                         (recur (inc p) new-kv))
                                       cur-kv))]
                               (xla/promote! prefill-arena session-arena final-prefill-kv)
                               final-prefill-kv))))
            t-prefill-1 (System/nanoTime)
            prefill-ms (/ (- t-prefill-1 t-prefill-0) 1e6)

            ;; 2. Run In-VRAM While Loop carrying the KV cache in a scoped loop arena
            is-persistent? (some? kv-state)
            [decode-ms cleaned-ids]
            (xla/with-device-arena [loop-arena session-arena]
              (let [b-step (xla/device-buffer loop-arena (int-array [p-count]) [] :i32)
                    b-max (xla/device-buffer loop-arena (int-array [target-max]) [] :i32)
                    b-toks (xla/device-buffer loop-arena in-arr [1 seq-len] :i32)
                    loop-inputs (into [b-step b-max b-toks] (concat prefill-kv device-weights))
                    num-loop-outs (+ 2 (* 2 num-unshared))

                    t-loop-0 (System/nanoTime)
                    loop-outs (xla/track! loop-arena (pjrt/execute-executable ctx (or (:handle exec) exec) loop-inputs num-loop-outs))
                    t-loop-1 (System/nanoTime)
                    decode-ms (/ (- t-loop-1 t-loop-0) 1e6)

                    loop-outs-vec (if (vector? loop-outs) loop-outs [loop-outs])
                    out-step (nth loop-outs-vec 0)
                    out-toks (nth loop-outs-vec 1)
                    final-kv (vec (subvec loop-outs-vec 2))

                    step-floats (pjrt/buffer-to-host-buffer ctx out-step 1 :f32)
                    step-val (int (Float/floatToIntBits (aget step-floats 0)))
                    toks-floats (pjrt/buffer-to-host-buffer ctx out-toks seq-len :f32)

                    actual-step (min (max p-count step-val) seq-len)
                    final-ids (mapv #(Float/floatToIntBits %) (take actual-step (vec toks-floats)))
                    cleaned-ids (if (and (> (count final-ids) p-count)
                                         (contains? kernels/GEMMA4-STOP-TOKEN-IDS (last final-ids)))
                                  (subvec final-ids 0 (dec (count final-ids)))
                                  final-ids)]
                (if is-persistent?
                  (do
                    (xla/promote! loop-arena session-arena final-kv)
                    (when-let [prior @kv-state]
                      (arena/destroy! session-arena (:kv-buffers prior)))
                    (reset! kv-state {:cached-tokens cleaned-ids
                                      :kv-buffers final-kv}))
                  nil)
                [decode-ms cleaned-ids]))]
        ;; Free prefill-kv once loop has finished
        (arena/destroy! session-arena prefill-kv)

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
         step-exec (when (and vram-loop? (> max-seq-len safe-prefill-len))
                     (kernels/compile-gemma4-kv-executable session max-seq-len))
         _ (when-not (:quiet opts) (println "Pinning Gemma 4 weights in PJRT VRAM..."))
         device-weights (weights/allocate-device-weights session)]
     (assoc session
            :device-weights device-weights
            :executable exec
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
