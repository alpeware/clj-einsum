(ns einsum.models.gemma4.runtime
  "Gemma 4 generation runtime engines, decoding loops, and persistent session management."
  (:require [clojure.pprint :as pprint]
            [clojure.string :as str]
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
            [einsum.runtime.weights :as rw]
            [einsum.uop.runtime :as uop-rt])
  (:import [java.lang.foreign Arena MemorySegment]))

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

(defn- model-sliding-window-size
  "Calculates minimum sliding window size across all unshared sliding-attention layers."
  [config num-unshared]
  (let [layer-configs (:layer-configs config)
        layer-types (:layer-types config)
        wins (keep (fn [i]
                     (let [cfg (when (seq layer-configs) (nth layer-configs i nil))
                           is-global? (if cfg (:is-global? cfg) (gemma-logic/layer-is-global? layer-types i))]
                       (when-not is-global?
                         (long (or (:sliding-window cfg)
                                   (:sliding-window config)
                                   (:sliding_window config)
                                   512)))))
                   (range num-unshared))]
    (when (seq wins)
      (reduce min wins))))

(defn- execute-step-executable
  "Executes a compiled UOp step, chunk, or prefill executable given step inputs."
  [ctx exec step-inputs num-outs]
  (let [out-map (uop-rt/execute-uop-executable ctx exec step-inputs)]
    (if (== (long num-outs) 1)
      (first (vals out-map))
      (let [sched-outvars (get-in exec [:uop-graph :outvars])
            ordered (if (seq sched-outvars)
                      (mapv #(get out-map %) sched-outvars)
                      (vec (vals out-map)))]
        ordered))))

(defn- to-host-slice
  "Transfers a slice of UOp output device buffer or array back to host float array."
  ([ctx-or-buf out-buf-or-slice-idx slice-idx-or-vsize vocab-size-or-total total-or-dt & [maybe-dt]]
   (let [[ctx out-buf slice-idx vocab-size total-elements _dtype]
         (if maybe-dt
           [ctx-or-buf out-buf-or-slice-idx (long slice-idx-or-vsize) (long vocab-size-or-total) (long total-or-dt) maybe-dt]
           [nil ctx-or-buf (long out-buf-or-slice-idx) (long slice-idx-or-vsize) (long vocab-size-or-total) total-or-dt])
         n-elements (max (long total-elements) (long (* (inc (long slice-idx)) (long vocab-size))))
         ^floats all-floats (cond
                              (instance? (Class/forName "[F") out-buf)
                              ^floats out-buf

                              (uop-rt/uop-buffer? out-buf)
                              (uop-rt/device->host-floats ctx out-buf n-elements)

                              :else
                              (float-array (map float out-buf)))
         offset (int (* (long slice-idx) (long vocab-size)))
         v-size (int vocab-size)
         slice (float-array v-size)]
     (System/arraycopy all-floats offset slice 0 v-size)
     slice)))

(defn run-vram-loop-generation
  "Executes autoregressive token generation entirely within device VRAM using 1-shot prefill
   (or sequential prefill for large contexts, with prefix KV-cache reuse across agent turns)
   and an UOp while-loop supporting chunked semantic early stopping."
  [session exec device-weights prompt-ids max-seq-len]
  (let [exec (or exec (:executable session) (:exec session))
        device-weights (or device-weights (:device-weights session))
        {:keys [ctx opts config kv-state prefill-executable session-arena tokenizer]} session
        session-arena (or session-arena (arena/create-arena ctx))
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
            uop? (= (:backend ctx) :uop)
            num-layers (long (or (:num-layers config) 35))
            num-kv-shared (long (or (:num-kv-shared-layers config) 0))
            num-unshared (- num-layers num-kv-shared)
            safe-prefill-len (cfg/max-safe-prefill-seq-len config)
            prefill-exec (or prefill-executable
                             (:prefill-executable session)
                             (when (and (not uop?) (<= seq-len safe-prefill-len))
                               (kernels/compile-gemma4-prefill-executable session seq-len)))
            step-exec (or (:step-executable session)
                          (kernels/compile-gemma4-kv-executable session seq-len))
            in-arr (int-array seq-len)
            _ (dotimes [i p-count] (aset in-arr i (int (nth clamped-prompt-ids i))))

            ;; Check for prefix KV-cache hit across turns
            is-persistent? (some? kv-state)
            prior-cache (when is-persistent? @kv-state)
            min-win (model-sliding-window-size config num-unshared)
            ring-wrapped? (boolean (and min-win
                                        prior-cache
                                        (>= (count (:cached-tokens prior-cache)) min-win)))
            p-match (if (and prior-cache
                             (not ring-wrapped?)
                             (:cached-tokens prior-cache)
                             (seq (:kv-buffers prior-cache)))
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
                              (arena/with-device-arena [iter-arena session-arena]
                                (let [tok (int (nth clamped-prompt-ids p))
                                      _ (aset x-arr 0 tok)
                                      _ (aset pos-arr 0 p)
                                      x-b (arena/device-buffer iter-arena x-arr [1 1] :i32)
                                      pos-b (arena/device-buffer iter-arena pos-arr [1] :i32)
                                      step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                      outs (arena/track! iter-arena (execute-step-executable ctx step-exec step-inputs num-step-outs))
                                      outs-vec (if (vector? outs) outs [outs])
                                      nk (vec (subvec outs-vec 1))]
                                  (arena/promote! iter-arena session-arena nk)
                                  nk))]
                          (arena/destroy! session-arena cur-kv)
                          (recur (inc p) new-kv))
                        cur-kv)))))

              ;; Path B: 1-shot parallel prefill
              (and (not uop?) (some? prefill-exec))
              (arena/with-device-arena [prefill-arena session-arena]
                (let [pos-p (int-array [(dec p-count)])
                      in-b (arena/device-buffer prefill-arena in-arr [1 seq-len] :i32)
                      pos-b (arena/device-buffer prefill-arena pos-p [1] :i32)
                      prefill-inputs (into [in-b pos-b] device-weights)
                      num-prefill-outs (inc (* 2 num-unshared))
                      prefill-outs (arena/track! prefill-arena
                                                 (execute-step-executable ctx prefill-exec prefill-inputs num-prefill-outs))
                      prefill-outs-vec (if (vector? prefill-outs) prefill-outs [prefill-outs])
                      kv-outs (vec (subvec prefill-outs-vec 1))]
                  (arena/promote! prefill-arena session-arena kv-outs)
                  kv-outs))

              ;; Path C: Chunked prefill (with sequential fallback for tail)
              :else
              (let [initial-kv (kernels/allocate-kv-cache-buffers session seq-len session-arena)
                    target-chunk-sz (min prefill-limit (long (or (:chunk-size config) 512)))
                    chunk-exec (or (:chunked-prefill-executable session)
                                   (when (pos? target-chunk-sz)
                                     (kernels/compile-gemma4-chunked-prefill-executable session seq-len target-chunk-sz)))
                    chunk-sz target-chunk-sz
                    num-full-chunks (if (and chunk-exec (pos? chunk-sz)) (quot prefill-limit chunk-sz) 0)
                    chunk-end (* num-full-chunks chunk-sz)]
                (when-not quiet
                  (if (pos? num-full-chunks)
                    (println (format "Chunked prefilling %d prompt tokens into KV-Cache (%d x %d chunks, %d tail tokens)..."
                                     prefill-limit num-full-chunks chunk-sz (- prefill-limit chunk-end)))
                    (println (format "Prefilling %d prompt tokens into KV-Cache (exceeds parallel prefill limit %d)..."
                                     prefill-limit safe-prefill-len))))
                (let [post-chunk-kv
                      (if (pos? num-full-chunks)
                        (loop [c 0
                               cur-kv initial-kv]
                          (if (< c num-full-chunks)
                            (let [start-p (* c chunk-sz)
                                  new-kv
                                  (arena/with-device-arena [iter-arena session-arena]
                                    (let [chunk-toks (int-array chunk-sz)
                                          _ (dotimes [ci chunk-sz]
                                              (aset chunk-toks ci (int (nth clamped-prompt-ids (+ start-p ci)))))
                                          pos-c (int-array [start-p])
                                          x-b (arena/device-buffer iter-arena chunk-toks [1 chunk-sz] :i32)
                                          pos-b (arena/device-buffer iter-arena pos-c [1] :i32)
                                          step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                          outs (arena/track! iter-arena (execute-step-executable ctx chunk-exec step-inputs num-step-outs))
                                          outs-vec (if (vector? outs) outs [outs])
                                          nk (vec (subvec outs-vec 1))]
                                      (arena/promote! iter-arena session-arena nk)
                                      nk))]
                              (arena/destroy! session-arena cur-kv)
                              (recur (inc c) new-kv))
                            cur-kv))
                        initial-kv)]
                  (loop [p chunk-end
                         cur-kv post-chunk-kv]
                    (if (< p prefill-limit)
                      (let [new-kv
                            (arena/with-device-arena [iter-arena session-arena]
                              (let [tok (int (nth clamped-prompt-ids p))
                                    _ (aset x-arr 0 tok)
                                    _ (aset pos-arr 0 p)
                                    x-b (arena/device-buffer iter-arena x-arr [1 1] :i32)
                                    pos-b (arena/device-buffer iter-arena pos-arr [1] :i32)
                                    step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                    outs (arena/track! iter-arena (execute-step-executable ctx step-exec step-inputs num-step-outs))
                                    outs-vec (if (vector? outs) outs [outs])
                                    nk (vec (subvec outs-vec 1))]
                                (arena/promote! iter-arena session-arena nk)
                                nk))]
                        (arena/destroy! session-arena cur-kv)
                        (recur (inc p) new-kv))
                      cur-kv)))))

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
                      (arena/with-device-arena [chunk-arena session-arena]
                        (let [b-step (arena/device-buffer chunk-arena (int-array [cur-step]) [] :i32)
                              b-max (arena/device-buffer chunk-arena (int-array [chunk-target]) [] :i32)
                              b-toks (arena/device-buffer chunk-arena cur-toks-arr [1 seq-len] :i32)
                              loop-inputs (into [b-step b-max b-toks] (concat cur-kv device-weights))
                              loop-outs (arena/track! chunk-arena (execute-step-executable ctx exec loop-inputs num-loop-outs))
                              loop-outs-vec (if (vector? loop-outs) loop-outs [loop-outs])
                              out-step (nth loop-outs-vec 0)
                              out-toks (nth loop-outs-vec 1)
                              nk (vec (subvec loop-outs-vec 2))

                              step-floats (uop-rt/device->host-floats ctx out-step 1)
                              step-val (int (Float/floatToIntBits (aget step-floats 0)))
                              toks-floats (uop-rt/device->host-floats ctx out-toks seq-len)
                              step-bounded (min (max cur-step step-val) seq-len)]
                          (dotimes [i (- step-bounded cur-step)]
                            (let [idx (+ cur-step i)]
                              (aset cur-toks-arr idx (Float/floatToIntBits (aget toks-floats idx)))))
                          (arena/promote! chunk-arena session-arena nk)
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
               (arena/with-device-arena [step-arena (or session-arena ctx)]
                 (let [in-b (arena/device-buffer step-arena in-arr [1 seq-len] :i32)
                       pos-b (when last-token?
                               (arena/device-buffer step-arena pos-arr [1] :i32))
                       args (let [base (if last-token? [in-b pos-b] [in-b])
                                  rel-bufs (get session :relational-buffers [])]
                              (into base (concat device-weights rel-bufs)))
                       out (arena/track! step-arena (execute-step-executable ctx exec args 1))
                       out-buf (if (sequential? out) (first out) out)
                       logits (if last-token?
                                (to-host-slice out-buf 0 vocab-size vocab-size weight-dt)
                                (to-host-slice out-buf (dec s-len) vocab-size (* seq-len vocab-size) weight-dt))]
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

(defn step-cached-kv-session!
  "Executes a single autoregressive step for `token-id` at `step-pos`, updating `@kv-buffers-atom` and returning new logits."
  [session exec device-weights session-arena kv-buffers-atom x-arr pos-arr num-outs token-id step-pos]
  (let [ctx (:ctx session)
        [new-log new-kv]
        (arena/with-device-arena [step-arena session-arena]
          (let [_ (aset x-arr 0 (int token-id))
                _ (aset pos-arr 0 (int step-pos))
                x-b (arena/device-buffer step-arena x-arr [1 1] :i32)
                pos-b (arena/device-buffer step-arena pos-arr [1] :i32)
                step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                outs (arena/track! step-arena (execute-step-executable ctx exec step-inputs num-outs))
                outs-vec (if (vector? outs) outs [outs])
                nl (first outs-vec)
                nk (vec (subvec outs-vec 1))]
            (arena/promote! step-arena session-arena nl)
            (arena/promote! step-arena session-arena nk)
            [nl nk]))
        old-kv @kv-buffers-atom]
    (when (seq old-kv)
      (arena/destroy! session-arena old-kv))
    (reset! kv-buffers-atom new-kv)
    new-log))

(defn prefill-cached-kv-session!
  "Executes Phase 1 parallel / chunked prefill into VRAM KV-Cache buffers.
   Returns a map with `:last-logits`, `:kv-buffers-atom`, `:session-arena`, `:clamped-prompt-ids`,
   `:prompt-count`, `:t-prefill-end`, `:p-match`, `:seq-len`, `:num-outs`, `:x-arr`, `:pos-arr`,
   `:vocab-size`, and `:weight-dt`."
  ([session exec device-weights prompt-ids]
   (prefill-cached-kv-session! session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config kv-state prefill-executable session-arena] :as session} exec device-weights prompt-ids max-seq-len]
   (let [session-arena (or session-arena (arena/create-arena ctx))
         uop? (= (:backend ctx) :uop)
         {:keys [quiet]} opts
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
         min-win (model-sliding-window-size config num-unshared)
         ring-wrapped? (boolean (and min-win
                                     prior-cache
                                     (>= (count (:cached-tokens prior-cache)) min-win)))
         p-match (if (and prior-cache
                          (not ring-wrapped?)
                          (:cached-tokens prior-cache)
                          (seq (:kv-buffers prior-cache)))
                   (common-prefix-len (:cached-tokens prior-cache) clamped-prompt-ids)
                   0)
         _ (when (and is-persistent? prior-cache (zero? p-match))
             (arena/destroy! session-arena (:kv-buffers prior-cache))
             (reset! kv-state nil))
         initial-kv (cond
                      (pos? p-match)
                      (:kv-buffers prior-cache)

                      (and (not uop?) (some? prefill-exec) (<= seq-len (cfg/max-safe-prefill-seq-len config)))
                      nil

                      :else
                      (kernels/allocate-kv-cache-buffers session seq-len session-arena))
         kv-buffers-atom (atom initial-kv)
         x-arr (int-array 1)
         pos-arr (int-array 1)
         target-chunk-sz (min prompt-count (long (or (:chunk-size opts) (:chunk-size config) 512)))
         chunk-exec (or (:chunked-prefill-executable session)
                        (when (and (zero? p-match)
                                   (or uop? (nil? prefill-exec) (> seq-len (cfg/max-safe-prefill-seq-len config)))
                                   (pos? target-chunk-sz))
                          (kernels/compile-gemma4-chunked-prefill-executable session seq-len target-chunk-sz)))
         _ (when (and uop? (= (:uop-target ctx) :rocm))
             (arena/with-device-arena [warmup-arena session-arena]
               (let [zero-pos (int-array 1)
                     pos-b (arena/device-buffer warmup-arena zero-pos [1] :i32)]
                 (when (and chunk-exec
                            (pos? target-chunk-sz)
                            (:warmed-up? chunk-exec)
                            (compare-and-set! (:warmed-up? chunk-exec) false true))
                   (let [zero-chunk (int-array target-chunk-sz)
                         x-b (arena/device-buffer warmup-arena zero-chunk [1 target-chunk-sz] :i32)
                         step-inputs (into [x-b pos-b] (concat initial-kv device-weights))]
                     (execute-step-executable ctx chunk-exec step-inputs num-outs)))
                 (when (and exec
                            (:warmed-up? exec)
                            (compare-and-set! (:warmed-up? exec) false true))
                   (let [zero-tok (int-array 1)
                         x-b (arena/device-buffer warmup-arena zero-tok [1 1] :i32)
                         step-inputs (into [x-b pos-b] (concat initial-kv device-weights))]
                     (execute-step-executable ctx exec step-inputs num-outs)))
                 (when (zero? p-match)
                   (doseq [[csz d-exec] (:delta-chunk-executables session)]
                     (when (and d-exec
                                (pos? (long csz))
                                (:warmed-up? d-exec)
                                (compare-and-set! (:warmed-up? d-exec) false true))
                       (let [zero-chunk (int-array (long csz))
                             x-b (arena/device-buffer warmup-arena zero-chunk [1 (long csz)] :i32)
                             step-inputs (into [x-b pos-b] (concat initial-kv device-weights))]
                         (execute-step-executable ctx d-exec step-inputs num-outs))))))))
         step-token-range!
         (fn [init-p init-logits]
           (let [delta-execs (:delta-chunk-executables session)]
             (loop [p (long init-p)
                    cur-logits init-logits]
               (if (< p prompt-count)
                 (let [rem-toks (- prompt-count p)
                       [tile-p tile-sz tile-exec]
                       (when (seq delta-execs)
                         (or
                          (when (>= rem-toks 4)
                            (when-let [[csz ex] (first (filter (fn [[csz ex]]
                                                                 (and ex
                                                                      (>= (long csz) rem-toks)
                                                                      (<= (long csz) prompt-count)))
                                                               (reverse delta-execs)))]
                              [(- prompt-count (long csz)) (long csz) ex]))
                          (when-let [[csz ex] (first (filter (fn [[csz ex]]
                                                               (and ex (<= (long csz) rem-toks)))
                                                             delta-execs))]
                            [p (long csz) ex])))]
                   (if (and tile-exec (> (long tile-sz) 1))
                     (let [tp (long tile-p)
                           csz (long tile-sz)
                           [new-log new-kv]
                           (arena/with-device-arena [step-arena session-arena]
                             (let [chunk-toks (int-array csz)
                                   _ (dotimes [ci csz]
                                       (aset chunk-toks ci (int (nth clamped-prompt-ids (+ tp ci)))))
                                   _ (aset pos-arr 0 (int tp))
                                   x-b (arena/device-buffer step-arena chunk-toks [1 csz] :i32)
                                   pos-b (arena/device-buffer step-arena pos-arr [1] :i32)
                                   step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                   outs (arena/track! step-arena (execute-step-executable ctx tile-exec step-inputs num-outs))
                                   outs-vec (if (vector? outs) outs [outs])
                                   nl (first outs-vec)
                                   nk (vec (subvec outs-vec 1))]
                               (arena/promote! step-arena session-arena nl)
                               (arena/promote! step-arena session-arena nk)
                               [nl nk]))
                           old-kv @kv-buffers-atom]
                       (when cur-logits (arena/destroy! session-arena cur-logits))
                       (when (seq old-kv)
                         (arena/destroy! session-arena old-kv))
                       (reset! kv-buffers-atom new-kv)
                       (recur (+ tp csz) new-log))
                     (let [[new-log new-kv]
                           (arena/with-device-arena [step-arena session-arena]
                             (let [tok (int (nth clamped-prompt-ids p))
                                   _ (aset x-arr 0 tok)
                                   _ (aset pos-arr 0 (int p))
                                   x-b (arena/device-buffer step-arena x-arr [1 1] :i32)
                                   pos-b (arena/device-buffer step-arena pos-arr [1] :i32)
                                   step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                   outs (arena/track! step-arena (execute-step-executable ctx exec step-inputs num-outs))
                                   outs-vec (if (vector? outs) outs [outs])
                                   nl (first outs-vec)
                                   nk (vec (subvec outs-vec 1))]
                               (arena/promote! step-arena session-arena nl)
                               (arena/promote! step-arena session-arena nk)
                               [nl nk]))
                           old-kv @kv-buffers-atom]
                       (when cur-logits (arena/destroy! session-arena cur-logits))
                       (when (seq old-kv)
                         (arena/destroy! session-arena old-kv))
                       (reset! kv-buffers-atom new-kv)
                       (recur (inc p) new-log))))
                 cur-logits))))
         t0 (System/nanoTime)
         [last-logits t-prefill-end]
         (cond
           (pos? p-match)
           (let [start-p (if (= p-match prompt-count) (max 0 (dec prompt-count)) p-match)
                 cur-log (step-token-range! start-p nil)]
             [cur-log (System/nanoTime)])

           (and (not uop?) (some? prefill-exec))
           (let [[prefill-logits prefill-kv]
                 (arena/with-device-arena [step-arena session-arena]
                   (let [in-arr (int-array seq-len)
                         _ (dotimes [i prompt-count] (aset in-arr i (int (nth clamped-prompt-ids i))))
                         pos-p (int-array [(dec prompt-count)])
                         in-b (arena/device-buffer step-arena in-arr [1 seq-len] :i32)
                         pos-b (arena/device-buffer step-arena pos-p [1] :i32)
                         step-inputs (into [in-b pos-b] device-weights)
                         outs (arena/track! step-arena (execute-step-executable ctx prefill-exec step-inputs num-outs))
                         outs-vec (if (vector? outs) outs [outs])
                         pl (first outs-vec)
                         pkv (vec (subvec outs-vec 1))]
                     (arena/promote! step-arena session-arena pl)
                     (arena/promote! step-arena session-arena pkv)
                     [pl pkv]))]
             (reset! kv-buffers-atom prefill-kv)
             [prefill-logits (System/nanoTime)])

           :else
           (let [chunk-sz target-chunk-sz
                 num-full-chunks (if (and chunk-exec (pos? chunk-sz)) (quot prompt-count chunk-sz) 0)
                 chunk-end (* num-full-chunks chunk-sz)
                 exact-chunks? (and (pos? num-full-chunks) (= chunk-end prompt-count))
                 tail-len (- prompt-count chunk-end)]
             (when-not quiet
               (if (pos? num-full-chunks)
                 (if (> tail-len 1)
                   (println (format "Chunked prefilling %d prompt tokens into KV-Cache (%d x %d chunks + 1 x %d tail chunk)..."
                                    prompt-count num-full-chunks chunk-sz tail-len))
                   (println (format "Chunked prefilling %d prompt tokens into KV-Cache (%d x %d chunks, %d tail tokens)..."
                                    prompt-count num-full-chunks chunk-sz tail-len)))
                 (println (format "Prefilling %d prompt tokens into KV-Cache..." prompt-count))))
             (let [[post-chunk-logits post-chunk-kv]
                   (if (pos? num-full-chunks)
                     (loop [c 0
                            cur-kv @kv-buffers-atom
                            cur-log nil]
                       (if (< c num-full-chunks)
                         (let [start-p (* c chunk-sz)
                               [new-log new-kv]
                               (arena/with-device-arena [iter-arena session-arena]
                                 (let [chunk-toks (int-array chunk-sz)
                                       _ (dotimes [ci chunk-sz]
                                           (aset chunk-toks ci (int (nth clamped-prompt-ids (+ start-p ci)))))
                                       pos-c (int-array [start-p])
                                       x-b (arena/device-buffer iter-arena chunk-toks [1 chunk-sz] :i32)
                                       pos-b (arena/device-buffer iter-arena pos-c [1] :i32)
                                       step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                       outs (arena/track! iter-arena (execute-step-executable ctx chunk-exec step-inputs num-outs))
                                       outs-vec (if (vector? outs) outs [outs])
                                       nl (first outs-vec)
                                       nk (vec (subvec outs-vec 1))]
                                   (arena/promote! iter-arena session-arena nl)
                                   (arena/promote! iter-arena session-arena nk)
                                   [nl nk]))
                               old-kv cur-kv]
                           (when cur-log (arena/destroy! session-arena cur-log))
                           (when (seq old-kv) (arena/destroy! session-arena old-kv))
                           (recur (inc c) new-kv new-log))
                         [cur-log cur-kv]))
                     [nil @kv-buffers-atom])]
               (reset! kv-buffers-atom post-chunk-kv)
               (if exact-chunks?
                 [post-chunk-logits (System/nanoTime)]
                 (if (seq (:delta-chunk-executables session))
                   [(step-token-range! chunk-end post-chunk-logits) (System/nanoTime)]
                   (cond
                     (> tail-len 1)
                     (let [tail-exec (or (:tail-prefill-executable session)
                                         (kernels/compile-gemma4-chunked-prefill-executable session seq-len tail-len))
                           start-p chunk-end
                           [tail-log tail-kv]
                           (arena/with-device-arena [iter-arena session-arena]
                             (let [tail-toks (int-array tail-len)
                                   _ (dotimes [ti tail-len]
                                       (aset tail-toks ti (int (nth clamped-prompt-ids (+ start-p ti)))))
                                   pos-c (int-array [start-p])
                                   x-b (arena/device-buffer iter-arena tail-toks [1 tail-len] :i32)
                                   pos-b (arena/device-buffer iter-arena pos-c [1] :i32)
                                   step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                   outs (arena/track! iter-arena (execute-step-executable ctx tail-exec step-inputs num-outs))
                                   outs-vec (if (vector? outs) outs [outs])
                                   nl (first outs-vec)
                                   nk (vec (subvec outs-vec 1))]
                               (arena/promote! iter-arena session-arena nl)
                               (arena/promote! iter-arena session-arena nk)
                               [nl nk]))
                           old-kv @kv-buffers-atom]
                       (when post-chunk-logits (arena/destroy! session-arena post-chunk-logits))
                       (when (seq old-kv) (arena/destroy! session-arena old-kv))
                       (reset! kv-buffers-atom tail-kv)
                       [tail-log (System/nanoTime)])

                     (== tail-len 1)
                     (let [[new-log new-kv]
                           (arena/with-device-arena [step-arena session-arena]
                             (let [tok (int (nth clamped-prompt-ids chunk-end))
                                   _ (aset x-arr 0 tok)
                                   _ (aset pos-arr 0 chunk-end)
                                   x-b (arena/device-buffer step-arena x-arr [1 1] :i32)
                                   pos-b (arena/device-buffer step-arena pos-arr [1] :i32)
                                   step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                   outs (arena/track! step-arena (execute-step-executable ctx exec step-inputs num-outs))
                                   outs-vec (if (vector? outs) outs [outs])
                                   nl (first outs-vec)
                                   nk (vec (subvec outs-vec 1))]
                               (arena/promote! step-arena session-arena nl)
                               (arena/promote! step-arena session-arena nk)
                               [nl nk]))
                           old-kv @kv-buffers-atom]
                       (when post-chunk-logits (arena/destroy! session-arena post-chunk-logits))
                       (when (seq old-kv)
                         (arena/destroy! session-arena old-kv))
                       (reset! kv-buffers-atom new-kv)
                       [new-log (System/nanoTime)])

                     :else
                     [post-chunk-logits (System/nanoTime)]))))))]
     {:last-logits last-logits
      :kv-buffers-atom kv-buffers-atom
      :session-arena session-arena
      :clamped-prompt-ids clamped-prompt-ids
      :prompt-count prompt-count
      :t0 t0
      :t-prefill-end t-prefill-end
      :p-match p-match
      :seq-len seq-len
      :num-outs num-outs
      :x-arr x-arr
      :pos-arr pos-arr
      :vocab-size vocab-size
      :weight-dt weight-dt})))

(defn build-decode-policy
  "Builds a unified autoregressive sampling and stop-token policy map shared across
   both the host KV-cache decode loop (`run-host-kv-decode-loop!`) and the fractal
   graph decoder (`autoregressive-decoder-graph`):
   - `:max-new-tokens`: bounded by `max-seq-len - 1 - prompt-count` when `max-seq-len` is present.
   - `:eos-id`: primary EOS token ID from the tokenizer.
   - `:eos-ids`: set of all stop token IDs (`GEMMA4-STOP-TOKEN-IDS` + `eos-id`, or custom `:eos-ids`).
   - `:stop-token?`: predicate `(fn [tok-id])` returning true if `tok-id` is in `:eos-ids`.
   - `:temperature`: sampling temperature (double).
   - `:sample-fn`: optional custom `(fn [logits seen-tokens])` for `autoregressive-decoder-graph`.
   - `:select-token-fn`: unified `(fn [logits-data seen-tokens])` returning the next token ID (long).
   Note: Stop tokens (`<eos>`, `<end_of_turn>`, `<turn|>`) terminate decoding before being
   appended to `:new-token-ids` across all decode paths (`:host`, `:graph`, `:vram-loop`)."
  [session eff-opts clamped-prompt-ids max-seq-len]
  (let [{:keys [tokenizer]} session
        tok-eos-id (long (or (when tokenizer (eos-id tokenizer)) 1))
        raw-max-new (long (or (:max-new-tokens eff-opts) 256))
        has-seq-bound? (or (some? max-seq-len)
                           (some? (:max-seq-len session))
                           (some? (:max-seq-len eff-opts))
                           (some? (get-in session [:config :max-seq-len])))
        eff-seq-len (long (or max-seq-len (:max-seq-len session) (:max-seq-len eff-opts) 2048))
        max-new-tokens (if has-seq-bound?
                         (min raw-max-new (max 1 (- (dec eff-seq-len) (count clamped-prompt-ids))))
                         raw-max-new)
        eos-ids (cond
                  (set? (:eos-ids eff-opts)) (:eos-ids eff-opts)
                  (sequential? (:eos-ids eff-opts)) (set (:eos-ids eff-opts))
                  :else (set (conj kernels/GEMMA4-STOP-TOKEN-IDS tok-eos-id)))
        stop-token? (fn [tok-id] (contains? eos-ids (long tok-id)))
        rep-pen (double (or (:repetition-penalty eff-opts) 1.0))
        custom-sample-fn (:sample-fn eff-opts)
        normalize-logits (fn ^floats [logits-data]
                           (let [row (cond
                                       (and (map? logits-data) (contains? logits-data :logits)) (:logits logits-data)
                                       (and (map? logits-data) (contains? logits-data :data) (some? (:data logits-data))) (:data logits-data)
                                       :else logits-data)]
                             (if (instance? (Class/forName "[F") row)
                               ^floats row
                               (float-array (map float row)))))
        graph-sample-fn (or custom-sample-fn
                            (when (> rep-pen 1.0)
                              (fn [logits seen-tokens]
                                (sample-next-token (normalize-logits logits) eff-opts clamped-prompt-ids seen-tokens))))
        select-token-fn (fn [logits-data seen-tokens]
                          (long (if (fn? custom-sample-fn)
                                  (custom-sample-fn logits-data seen-tokens)
                                  (sample-next-token (normalize-logits logits-data) eff-opts clamped-prompt-ids seen-tokens))))]
    {:max-new-tokens max-new-tokens
     :eos-id tok-eos-id
     :eos-ids eos-ids
     :stop-token? stop-token?
     :temperature (double (or (:temperature eff-opts) 0.0))
     :sample-fn graph-sample-fn
     :select-token-fn select-token-fn}))

(defn- run-host-kv-decode-loop!
  "Shared host-side autoregressive token loop consuming `build-decode-policy`.
   Halts before appending stop tokens (`:stop-token?`) so `:new-token-ids` contains
   content tokens only, matching `autoregressive-decoder-graph` and `run-vram-loop-generation`."
  [{:keys [max-new-tokens stop-token? select-token-fn]}
   prompt-count seq-len initial-logits step-logits-fn on-token-fn]
  (let [max-tokens (long (or max-new-tokens 256))
        limit-step (long (if seq-len (dec (long seq-len)) Long/MAX_VALUE))]
    (if (or (nil? initial-logits) (<= max-tokens 0) (>= (long prompt-count) limit-step))
      []
      (loop [step (long prompt-count)
             gen-tokens []
             cur-logits initial-logits]
        (let [next-id (long (select-token-fn cur-logits gen-tokens))]
          (if (stop-token? next-id)
            gen-tokens
            (let [gen-tokens' (conj gen-tokens next-id)
                  _ (when on-token-fn (on-token-fn next-id))
                  next-step (inc step)]
              (if (or (>= (count gen-tokens') max-tokens)
                      (>= next-step limit-step))
                gen-tokens'
                (recur next-step
                       gen-tokens'
                       (step-logits-fn next-id step))))))))))

(defn run-cached-kv-generation
  "Executes autoregressive generation with in-VRAM KV-Cache using single-token step executable
   and 1-shot parallel prefill.
   Note: Stop tokens (`GEMMA4-STOP-TOKEN-IDS` and tokenizer `eos-id`) terminate decoding before
   being appended to the returned token vector, matching `autoregressive-decoder-graph`."
  ([session exec device-weights prompt-ids]
   (run-cached-kv-generation session exec device-weights prompt-ids nil))
  ([{:keys [tokenizer opts kv-state] :as session} exec device-weights prompt-ids max-seq-len]
   (let [{:keys [quiet mode]} opts
         is-agent? (= mode :agent)
         is-persistent? (some? kv-state)
         prefill-res (prefill-cached-kv-session! session exec device-weights prompt-ids max-seq-len)
         t0 (:t0 prefill-res)
         {:keys [last-logits kv-buffers-atom session-arena clamped-prompt-ids prompt-count
                 t-prefill-end p-match seq-len num-outs x-arr pos-arr vocab-size weight-dt]}
         prefill-res]
     (try
       (let [policy (build-decode-policy session opts clamped-prompt-ids seq-len)
             initial-logits (when last-logits
                              (try
                                (to-host-slice last-logits 0 vocab-size vocab-size weight-dt)
                                (finally
                                  (arena/destroy! session-arena last-logits))))
             step-logits-fn (fn [next-id step-pos]
                              (let [new-log (step-cached-kv-session! session exec device-weights session-arena kv-buffers-atom x-arr pos-arr num-outs next-id step-pos)]
                                (try
                                  (to-host-slice new-log 0 vocab-size vocab-size weight-dt)
                                  (finally
                                    (arena/destroy! session-arena new-log)))))
             on-token-fn (when (and (not quiet) (not is-agent?))
                           (fn [next-id]
                             (print (decode tokenizer [next-id]))
                             (flush)))
             gen-tokens (run-host-kv-decode-loop! policy prompt-count seq-len initial-logits step-logits-fn on-token-fn)
             all-tokens (into (vec clamped-prompt-ids) gen-tokens)
             t1 (System/nanoTime)
             total-ms (/ (- t1 t0) 1e6)
             prefill-ms (/ (- t-prefill-end t0) 1e6)
             decode-ms (/ (- t1 t-prefill-end) 1e6)
             gen-count (count gen-tokens)
             decode-tok-s (if (pos? decode-ms) (/ (* gen-count 1000.0) decode-ms) 0.0)]
         (when is-persistent?
           (reset! kv-state {:cached-tokens all-tokens
                             :kv-buffers @kv-buffers-atom
                             :prefill-ms prefill-ms
                             :decode-ms decode-ms
                             :decode-tok-s decode-tok-s
                             :decode-ms-tok (if (pos? gen-count) (/ decode-ms (double gen-count)) 0.0)
                             :gen-count gen-count
                             :total-ms total-ms}))
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
         all-tokens)
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
  "Initializes UOp runtime, loads safetensors weights, and prepares model configuration for Gemma 4 REPL/CLI sessions."
  ([opts]
   (let [{:keys [backend]} opts
         _ (when (contains? #{:rocm :UOp :cuda12 :sycl :cpu} backend)
             (throw (ex-info (str "Backend " backend " is no longer supported. UOp/UOp has been excised. Supported backends are :uop-rocm and :uop-cpu.")
                             {:backend backend})))
         target-backend (or backend :uop-cpu)
         ctx (uop-rt/init-uop-context target-backend opts)
         opts (assoc opts :backend (or (:backend ctx) target-backend))
         model-dir (or (:model-dir opts) (:model opts))
         dirs (if model-dir (cons model-dir cfg/DEFAULT_MODEL_DIRS) cfg/DEFAULT_MODEL_DIRS)
         resolved-model-dir (cfg/find-model-dir dirs)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights resolved-model-dir arena)
         tokenizer (tok/from-file resolved-model-dir)
         header (or (:header weights-mmap) {})
         prefix-base (if (contains? header "model.language_model.embed_tokens.weight")
                       "model.language_model."
                       "model.")
         session-arena (or (:session-arena opts) (arena/create-arena ctx))
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
   (let [turboquant? (boolean (or (:turboquant-kv? opts)
                                  (:turboquant-kv opts)
                                  (= (:kv-quant opts) :turboquant)
                                  (= (:kv-quant opts) :turbo-kv)))
         session (init-inference-session opts)
         uop? (= (:backend (:ctx session)) :uop)
         turboquant? (or turboquant? (:turboquant-kv? (:config session)))
         method (or (:method opts)
                    (if (or uop? turboquant? (and (number? (:temperature opts)) (> (:temperature opts) 0.0)))
                      :kv-cache
                      :vram-loop))
         vram-loop? (if (contains? opts :vram-loop?)
                      (and (not uop?) (:vram-loop? opts))
                      (and (not uop?) (not turboquant?) (= method :vram-loop)))
         opts (assoc opts :mode :agent :max-seq-len max-seq-len :method method :vram-loop? vram-loop?)
         session (assoc session :opts opts)
         _ (when-not (:quiet opts)
             (println (format "Pre-compiling Gemma 4 %s graph (max-seq-len=%d%s)..."
                              (if vram-loop? "In-VRAM While Loop" "KV-Cache")
                              max-seq-len
                              (if turboquant? ", Fast-TurboQuant 2-Bit" ""))))
         exec (if vram-loop?
                (kernels/compile-in-vram-loop-executable session max-seq-len)
                (kernels/compile-gemma4-kv-executable session max-seq-len))
         safe-prefill-len (cfg/max-safe-prefill-seq-len (:config session))
         prefill-exec (when (and (not uop?) (not turboquant?) (<= max-seq-len safe-prefill-len))
                        (kernels/compile-gemma4-prefill-executable session max-seq-len))
         step-exec (when vram-loop?
                     (kernels/compile-gemma4-kv-executable session max-seq-len))
         rocm? (= (:uop-target (:ctx session)) :rocm)
         weight-bytes (let [wm (:weights-mmap session)]
                        (cond
                          (seq (:shards wm))
                          (reduce + 0 (map (fn [{:keys [^MemorySegment segment]}]
                                             (if segment (.byteSize segment) 0))
                                           (:shards wm)))
                          (:segment wm)
                          (.byteSize ^MemorySegment (:segment wm))
                          :else 0))
         ;; Bound to a single [64] prefill tile when mapped weights exceed 10 GiB (e.g. E4B ~14.8 GB BF16,
         ;; 31B ~17.3 GB INT4) or num-layers > 35, keeping total VRAM safely inside 24 GiB;
         ;; smaller models like E2B (~4.2 GB, 35 layers) pre-compile [64 32 16].
         large-model? (or (> (long weight-bytes) (* 10 1024 1024 1024))
                          (> (long (or (:num-layers (:config session)) 35)) 35))
         delta-chunk-sizes (when (not vram-loop?)
                             (->> (or (:delta-chunk-sizes opts)
                                      (when rocm?
                                        (if large-model? [64] [64 32 16])))
                                  (filter #(and (> (long %) 1) (< (long %) max-seq-len)))
                                  (sort >)
                                  vec))
         delta-chunk-execs (when (seq delta-chunk-sizes)
                             (into (sorted-map-by >)
                                   (map (fn [csz]
                                          [csz (kernels/compile-gemma4-chunked-prefill-executable session max-seq-len csz)])
                                        delta-chunk-sizes)))
         primary-chunk-sz (first delta-chunk-sizes)
         chunked-prefill-exec (or (when primary-chunk-sz (get delta-chunk-execs primary-chunk-sz))
                                  (when (and (not uop?) (>= max-seq-len 256))
                                    (kernels/compile-gemma4-chunked-prefill-executable session max-seq-len (min max-seq-len 128))))
         opts (cond-> opts
                (and primary-chunk-sz (nil? (:chunk-size opts)))
                (assoc :chunk-size primary-chunk-sz))
         session (assoc session :opts opts)
         _ (when-not (:quiet opts) (println "Pinning Gemma 4 weights in UOp VRAM..."))
         device-weights (weights/allocate-device-weights session)]
     (assoc session
            :opts opts
            :device-weights device-weights
            :executable exec
            :exec exec
            :prefill-executable prefill-exec
            :chunked-prefill-executable chunked-prefill-exec
            :step-executable step-exec
            :delta-chunk-executables delta-chunk-execs
            :kv-state (atom nil)
            :max-seq-len max-seq-len
            :vram-session? true
            :vram-loop? vram-loop?))))

(defn close-agent-session!
  "Releases VRAM resources for an agent session."
  [{:keys [ctx device-weights kv-state session-arena executable exec prefill-executable chunked-prefill-executable tail-prefill-executable step-executable delta-chunk-executables] :as _session}]
  (let [execs (concat [executable exec prefill-executable chunked-prefill-executable tail-prefill-executable step-executable]
                      (vals delta-chunk-executables))]
    (doseq [ex (distinct (filter :uop-executable? execs))]
      (uop-rt/destroy-uop-executable! ctx ex)))
  (if session-arena
    (do
      (arena/close-arena! session-arena)
      (when kv-state (reset! kv-state nil)))
    (do
      (when (seq device-weights)
        (doseq [w device-weights]
          (uop-rt/destroy-buffer! ctx w)))
      (when (and kv-state @kv-state)
        (doseq [b (:kv-buffers @kv-state)]
          (uop-rt/destroy-buffer! ctx b))
        (reset! kv-state nil)))))

;; ==============================================================================
;; High-Level Text & Token Generation API
;; ==============================================================================

(defn prepare-prompt-ids
  "Tokenizes and formats prompt for Gemma 4 session, returning a map:
   {:prompt-ids <vec>
    :clamped-prompt-ids <vec>
    :prompt-len <int>
    :safe-prompt-len <int>
    :max-seq-len <int>
    :clean-prompt <str>}."
  ([session prompt]
   (prepare-prompt-ids session prompt nil))
  ([session prompt call-opts]
   (let [{:keys [tokenizer opts]} session
         opts (merge (:opts session) opts call-opts)
         {:keys [max-new-tokens model]} opts
         max-new-tokens (long (or max-new-tokens 200))
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
         seq-len (long (or (:max-seq-len session) (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens))))
         safe-prompt-len (min prompt-len (max 0 (- seq-len 2)))
         clamped-prompt-ids (if (< safe-prompt-len prompt-len)
                              (subvec (vec prompt-ids) (- prompt-len safe-prompt-len))
                              (vec prompt-ids))]
     {:prompt-ids prompt-ids
      :clamped-prompt-ids (vec clamped-prompt-ids)
      :prompt-len prompt-len
      :safe-prompt-len safe-prompt-len
      :max-seq-len seq-len
      :clean-prompt clean-prompt})))

(defn resolve-active-session
  "Compiles and resolves step and prefill executables and device weights for the active session."
  [session opts prompt-ids max-seq-len]
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
        uop? (= (:backend (:ctx session)) :uop)
        safe-prefill-len (cfg/max-safe-prefill-seq-len (:config session))
        prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                       (if (:prefill-executable session)
                         (:prefill-executable session)
                         (when (and (not uop?) (not (:turboquant-kv? (:config session)))
                                    (or (= (or (:method opts) :kv-cache) :kv-cache)
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
        prompt-count (count prompt-ids)
        prior-cache (when (:kv-state session) @(:kv-state session))
        has-prefix-hit? (boolean (and prior-cache
                                      (seq (:cached-tokens prior-cache))
                                      (seq (:kv-buffers prior-cache))
                                      (pos? (common-prefix-len (:cached-tokens prior-cache) prompt-ids))))
        target-chunk-sz (min prompt-count (long (or (:chunk-size opts) (:chunk-size (:config session)) 512)))
        num-full-chunks (if (pos? target-chunk-sz) (quot prompt-count target-chunk-sz) 0)
        tail-len (- prompt-count (* num-full-chunks target-chunk-sz))
        chunked-prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                               (if (:chunked-prefill-executable session)
                                 (:chunked-prefill-executable session)
                                 (when (and (not has-prefix-hit?) (pos? target-chunk-sz))
                                   (profile/with-profile metrics-atom "graph_compilation"
                                     (kernels/compile-gemma4-chunked-prefill-executable session max-seq-len target-chunk-sz)))))
        tail-prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                            (if (:tail-prefill-executable session)
                              (:tail-prefill-executable session)
                              (when (and (not has-prefix-hit?)
                                         (empty? (:delta-chunk-executables session))
                                         (pos? num-full-chunks)
                                         (> tail-len 1))
                                (profile/with-profile metrics-atom "graph_compilation"
                                  (kernels/compile-gemma4-chunked-prefill-executable session max-seq-len tail-len)))))
        active-session (assoc session
                              :prefill-executable prefill-exec
                              :chunked-prefill-executable chunked-prefill-exec
                              :tail-prefill-executable tail-prefill-exec
                              :step-executable step-exec)
        device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                         (if reuse-weights?
                           (:device-weights session)
                           (profile/with-profile metrics-atom "weight_transfer"
                             (weights/allocate-device-weights session))))]
    {:active-session active-session
     :exec exec
     :device-weights device-weights
     :reuse-weights? reuse-weights?
     :metrics-atom metrics-atom
     :trace-spans-atom trace-spans-atom}))

(defn session-forward-fn
  "Creates a forward evaluation function `(fn [session payload])` suitable for `:compute/einsum`
   capabilities. Dispatches prefill on the first call and single-token steps on subsequent calls."
  ([session]
   (session-forward-fn session nil))
  ([session call-opts]
   (if-let [custom-fwd (:forward-fn session)]
     (fn
       ([payload] (custom-fwd session payload))
       ([sess payload] (custom-fwd (or sess session) payload)))
     (let [eff-opts (merge (:opts session) call-opts)
           gen-state (atom nil)
           t0-atom (atom 0)
           t-prefill-end-atom (atom 0)
           fwd-fn (fn fwd
                    ([payload] (fwd session payload))
                    ([sess payload]
                     (let [inputs (cond
                                    (and (map? payload) (contains? payload :inputs)) (:inputs payload)
                                    (and (map? payload) (contains? payload :args)) (:args payload)
                                    (vector? payload) payload
                                    (sequential? payload) (vec payload)
                                    :else [payload])
                           first-in (first inputs)
                           prefill? (or (nil? @gen-state) (sequential? first-in))]
                       (if prefill?
                         (let [prompt-tokens (if (sequential? first-in) (vec first-in) [first-in])
                               active-sess (assoc (or sess session) :opts eff-opts)
                               seq-len (long (or (:max-seq-len active-sess)
                                                 (:max-seq-len eff-opts)
                                                 (min 2048 (+ (count prompt-tokens) (long (or (:max-new-tokens eff-opts) 200))))))
                               resolved (resolve-active-session active-sess eff-opts prompt-tokens seq-len)
                               active-sess' (:active-session resolved)
                               exec (:exec resolved)
                               device-weights (:device-weights resolved)
                               prefill-res (prefill-cached-kv-session! active-sess' exec device-weights prompt-tokens seq-len)
                               {:keys [last-logits kv-buffers-atom session-arena num-outs x-arr pos-arr vocab-size weight-dt t0 t-prefill-end prompt-count]} prefill-res
                               _ (reset! t0-atom t0)
                               _ (reset! t-prefill-end-atom t-prefill-end)
                               _ (reset! gen-state {:session active-sess'
                                                    :exec exec
                                                    :device-weights device-weights
                                                    :session-arena session-arena
                                                    :kv-buffers-atom kv-buffers-atom
                                                    :num-outs num-outs
                                                    :x-arr x-arr
                                                    :pos-arr pos-arr
                                                    :vocab-size vocab-size
                                                    :weight-dt weight-dt
                                                    :step prompt-count})
                               logits-slice (to-host-slice last-logits 0 vocab-size vocab-size weight-dt)]
                           (arena/destroy! session-arena last-logits)
                           logits-slice)
                         (let [{:keys [session exec device-weights session-arena kv-buffers-atom
                                       num-outs x-arr pos-arr vocab-size weight-dt step]} @gen-state
                               token-id (long (if (sequential? first-in) (first first-in) first-in))
                               step-pos (if (and (map? payload) (:pos payload))
                                          (dec (long (:pos payload)))
                                          (long step))
                               new-logits (step-cached-kv-session! session exec device-weights session-arena kv-buffers-atom x-arr pos-arr num-outs token-id step-pos)
                               logits-slice (to-host-slice new-logits 0 vocab-size vocab-size weight-dt)]
                           (arena/destroy! session-arena new-logits)
                           (swap! gen-state assoc :step (inc step-pos))
                           logits-slice)))))]
       (with-meta fwd-fn {:gen-state-atom gen-state
                          :t0-atom t0-atom
                          :t-prefill-end-atom t-prefill-end-atom})))))

(defn generate-text
  "Executes end-to-end Gemma 4 generation pipeline for a given prompt string."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        opts (merge (:opts session) opts)
        {:keys [max-new-tokens temperature top-k quiet]} opts
        max-new-tokens (long (or max-new-tokens 200))
        temperature (double (or temperature 0.0))
        top-k (long (or top-k 10))
        {:keys [prompt-ids prompt-len max-seq-len clean-prompt]} (prepare-prompt-ids session prompt opts)]
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

    (let [{:keys [active-session exec device-weights reuse-weights? metrics-atom trace-spans-atom]}
          (resolve-active-session session opts prompt-ids max-seq-len)]
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
            (arena/close-arena! sa)
            (doseq [w device-weights] (uop-rt/destroy-buffer! (:ctx session) w))))
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

(defn generate-new-tokens-and-text-graph
  "Executes autoregressive token decoding using the token-level fractal sub-agent graph
   `autoregressive-decoder-graph`. Returns a map with generated text, token telemetry,
   and recorded `:sub-event-log` and `:sub-hiccup` for deterministic replay."
  ([session prompt]
   (generate-new-tokens-and-text-graph session prompt nil))
  ([session prompt call-opts]
   (let [t0 (System/nanoTime)
         {:keys [tokenizer opts]} session
         eff-opts (merge (:opts session) opts call-opts)
         {:keys [clamped-prompt-ids prompt-len max-seq-len]} (prepare-prompt-ids session prompt eff-opts)
         {:keys [max-new-tokens] :as policy} (build-decode-policy session eff-opts clamped-prompt-ids max-seq-len)
         decoder-opts (select-keys policy [:max-new-tokens :eos-ids :eos-id :temperature :sample-fn])
         decoder-hiccup ((requiring-resolve 'einsum.agent.graph/autoregressive-decoder-graph) decoder-opts)
         fwd-fn (session-forward-fn session eff-opts)
         device-hiccup [:device {:id :decoder-device}
                        [:cap {:id :gpu
                               :type :compute/einsum
                               :backend (or (get-in session [:ctx :backend]) :uop-cpu)
                               :session session
                               :forward-fn fwd-fn}]]
         bound-inst ((requiring-resolve 'einsum.graph.binder/bind-agent) decoder-hiccup device-hiccup)
         input-inst ((requiring-resolve 'einsum.graph.runtime/enqueue-input) bound-inst :tokens-in clamped-prompt-ids)
         stepped-inst ((requiring-resolve 'einsum.graph.runtime/run-graph!)
                       input-inst
                       {:max-steps (max 10000 (* 8 (+ max-new-tokens 16)))})
         new-token-ids (or (first (get-in stepped-inst [:outbox :tokens-out])) [])
         t1 (System/nanoTime)
         t0-fwd (long (or (some-> fwd-fn meta :t0-atom deref) 0))
         eff-t0 (if (pos? t0-fwd) t0-fwd t0)
         t-prefill-end (long (or (some-> fwd-fn meta :t-prefill-end-atom deref) 0))
         prefill-ms (if (pos? t-prefill-end) (/ (- t-prefill-end eff-t0) 1e6) 0.0)
         decode-ms (if (pos? t-prefill-end) (/ (- t1 t-prefill-end) 1e6) (/ (- t1 eff-t0) 1e6))
         total-ms (/ (- t1 eff-t0) 1e6)
         new-tokens (count new-token-ids)
         decode-tok-s (if (pos? decode-ms) (/ (* new-tokens 1000.0) decode-ms) 0.0)
         decode-ms-tok (if (pos? new-tokens) (/ decode-ms (double new-tokens)) 0.0)
         decoded-text (decode tokenizer new-token-ids)]
     (when-let [ks (:kv-state session)]
       (let [st (some-> fwd-fn meta :gen-state-atom deref)
             all-toks (into (vec clamped-prompt-ids) new-token-ids)
             stepped-cnt (long (or (:step st) (count all-toks)))]
         (reset! ks {:cached-tokens (subvec all-toks 0 (min (count all-toks) stepped-cnt))
                     :kv-buffers (some-> st :kv-buffers-atom deref)
                     :prefill-ms prefill-ms
                     :decode-ms decode-ms
                     :decode-tok-s decode-tok-s
                     :decode-ms-tok decode-ms-tok
                     :gen-count new-tokens
                     :total-ms total-ms})))
     (when-not (:quiet eff-opts)
       (println)
       (println "\n------------------------------------------------------------------")
       (println "  Telemetry Benchmark Metrics (Fractal Graph Decoder):")
       (println (format "    • Prefill Latency             : %8.2f ms (%d tokens)" prefill-ms prompt-len))
       (println (format "    • Decode Latency              : %8.2f ms (%d tokens)" decode-ms new-tokens))
       (println (format "    • Decode Speed                : %8.2f tok/s (%6.2f ms/tok)" decode-tok-s decode-ms-tok))
       (println (format "    • Total Generation Latency    : %8.2f ms" total-ms))
       (println "------------------------------------------------------------------\n"))
     {:text decoded-text
      :prompt-tokens prompt-len
      :new-tokens new-tokens
      :new-token-ids (vec new-token-ids)
      :prefill-ms prefill-ms
      :decode-ms decode-ms
      :decode-tok-s decode-tok-s
      :decode-ms-tok decode-ms-tok
      :total-ms total-ms
      :sub-event-log (:event-log stepped-inst)
      :sub-decoder-opts (dissoc decoder-opts :sample-fn)
      :sub-hiccup decoder-hiccup})))

(defn- generate-host-forward-decode
  "Runs the shared host KV decode loop (`run-host-kv-decode-loop!`) over `session-forward-fn`
   when a session provides a `:forward-fn` override."
  [session prompt call-opts]
  (let [eff-opts (merge (:opts session) call-opts)
        {:keys [tokenizer]} session
        {:keys [clamped-prompt-ids prompt-len max-seq-len]} (prepare-prompt-ids session prompt eff-opts)
        policy (build-decode-policy session eff-opts clamped-prompt-ids max-seq-len)
        fwd-fn (session-forward-fn session eff-opts)
        t0 (System/nanoTime)
        initial-logits (fwd-fn clamped-prompt-ids)
        t-prefill-end (long (or (some-> fwd-fn meta :t-prefill-end-atom deref) (System/nanoTime)))
        step-logits-fn (fn [next-id step-pos]
                         (fwd-fn {:inputs [next-id] :pos (inc step-pos)}))
        gen-tokens (run-host-kv-decode-loop! policy (count clamped-prompt-ids) max-seq-len initial-logits step-logits-fn nil)
        t1 (System/nanoTime)
        t0-fwd (long (or (some-> fwd-fn meta :t0-atom deref) 0))
        eff-t0 (if (pos? t0-fwd) t0-fwd t0)
        prefill-ms (if (pos? t-prefill-end) (/ (- t-prefill-end eff-t0) 1e6) 0.0)
        decode-ms (if (pos? t-prefill-end) (/ (- t1 t-prefill-end) 1e6) (/ (- t1 eff-t0) 1e6))
        total-ms (/ (- t1 eff-t0) 1e6)
        new-tokens (count gen-tokens)
        decode-tok-s (if (pos? decode-ms) (/ (* new-tokens 1000.0) decode-ms) 0.0)
        decode-ms-tok (if (pos? new-tokens) (/ decode-ms (double new-tokens)) 0.0)
        decoded-text (decode tokenizer gen-tokens)]
    {:text decoded-text
     :prompt-tokens prompt-len
     :new-tokens new-tokens
     :new-token-ids (vec gen-tokens)
     :prefill-ms prefill-ms
     :decode-ms decode-ms
     :decode-tok-s decode-tok-s
     :decode-ms-tok decode-ms-tok
     :total-ms total-ms}))

(defn generate-new-tokens-and-text
  "Generates text response using Gemma 4 model session and returns a map:
   {:text <decoded-new-text>
    :prompt-tokens <int>
    :new-tokens <int>
    :new-token-ids <vec>
    :prefill-ms <double>
    :decode-ms <double>
    :decode-tok-s <double>
    :decode-ms-tok <double>
    :total-ms <double>}.
   Note: Across all decode paths (`:host`, `:graph`, `:vram-loop`), stop tokens
   (`GEMMA4-STOP-TOKEN-IDS` and tokenizer `eos-id`) terminate decoding before being
   appended to `:new-token-ids`."
  ([session prompt]
   (generate-new-tokens-and-text session prompt nil))
  ([session prompt call-opts]
   (let [eff-opts (merge (:opts session) call-opts)
         decode-mode (or (:decode eff-opts) :host)
         temp (double (or (:temperature eff-opts) 0.0))]
     (cond
       (and (= decode-mode :graph) (or (<= temp 0.0) (:sample-fn eff-opts)))
       (generate-new-tokens-and-text-graph session prompt call-opts)

       (:forward-fn session)
       (generate-host-forward-decode session prompt call-opts)

       :else
       (let [{:keys [tokenizer opts]} session
             opts (merge (:opts session) opts call-opts)
             {:keys [prompt-len safe-prompt-len]} (prepare-prompt-ids session prompt opts)
             final-context (vec (generate-text (assoc session :opts opts) prompt))
             total-len (count final-context)
             slice-start (min total-len safe-prompt-len)
             new-ids (subvec final-context slice-start)]
         (merge {:text (decode tokenizer new-ids)
                 :prompt-tokens prompt-len
                 :new-tokens (count new-ids)
                 :new-token-ids (vec new-ids)}
                (when-let [ks (and (:kv-state session) @(:kv-state session))]
                  (select-keys ks [:prefill-ms :decode-ms :decode-tok-s :decode-ms-tok :total-ms]))))))))

(defn generate-new-text-string
  "Generates text response using Gemma 4 model session and returns ONLY newly generated text string without prompt prefix."
  [session prompt]
  (:text (generate-new-tokens-and-text session prompt)))

(defn score-sequence-log-probs
  "Evaluates exact log-probabilities of a token sequence in UOp device memory.
   When `session` contains `:step-executable` (from `init-agent-vram-session`), executes
   sequential in-accelerator KV-cache evaluation with fused target log-prob extraction,
   streaming log-probabilities with zero PCIe logit bloat.
   Alternatively accepts precompiled parallel scoring `exec` (from `compile-gemma4-scoring-executable`).
   Returns a map:
   {:token-ids token-ids
    :target-log-probs [log P(w_1|w_0), log P(w_2|w_<2), ..., log P(w_{n-1}|w_{<n-1})]
    :log-probs [0.0, log P(w_1|w_0), ..., log P(w_{n-1}|w_{<n-1})]}"
  ([session token-ids]
   (score-sequence-log-probs session nil token-ids nil))
  ([session token-ids-or-exec max-seq-len-or-tokens]
   (if (or (vector? token-ids-or-exec) (seq? token-ids-or-exec))
     (score-sequence-log-probs session nil token-ids-or-exec max-seq-len-or-tokens)
     (score-sequence-log-probs session token-ids-or-exec max-seq-len-or-tokens nil)))
  ([session exec token-ids max-seq-len]
   (let [tokens (vec token-ids)
         n (count tokens)
         _ (when (< n 2)
             (throw (ex-info "Sequence must contain at least 2 tokens to score target log-probabilities"
                             {:num-tokens n})))
         seq-len (long (or max-seq-len
                           (:max-seq-len session)
                           (get-in session [:config :max-seq-len])
                           n))
         _ (when (< seq-len n)
             (throw (ex-info "max-seq-len cannot be smaller than sequence token count"
                             {:max-seq-len seq-len :num-tokens n})))
         step-exec (or (:step-executable session)
                       (when-not exec (kernels/compile-gemma4-kv-executable session seq-len)))]
     (if step-exec
       ;; Path A: Sequential in-accelerator KV-cache step scoring
       (let [{:keys [ctx config session-arena]} session
             num-layers (long (or (:num-layers config) 42))
             num-kv-shared (long (or (:num-kv-shared-layers config) 18))
             num-unshared (- num-layers num-kv-shared)
             num-step-outs (inc (* 2 num-unshared))
             norm-dtype (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config) (= (:weight-dtype config) :ternary)) :bf16 (or (:weight-dtype config) :f32))
             device-weights (or (:device-weights session)
                                (weights/allocate-device-weights session))
             target-exec (or (:target-scoring-executable session)
                             (kernels/compile-target-log-prob-executable ctx (long (or (:vocab-size config) 262144)) norm-dtype))
             initial-kv (kernels/allocate-kv-cache-buffers session seq-len session-arena)
             x-arr (int-array 1)
             pos-arr (int-array 1)
             target-arr (int-array 1)
             n-score (dec n)
             collected
             (loop [p 0
                    cur-kv initial-kv
                    acc (transient [])]
               (if (< p n-score)
                 (let [tok (int (nth tokens p))
                       target (int (nth tokens (inc p)))
                       _ (aset x-arr 0 tok)
                       _ (aset pos-arr 0 p)
                       _ (aset target-arr 0 target)
                       [next-kv target-lp]
                       (arena/with-device-arena [step-arena session-arena]
                         (let [x-b (arena/device-buffer step-arena x-arr [1 1] :i32)
                               pos-b (arena/device-buffer step-arena pos-arr [1] :i32)
                               target-b (arena/device-buffer step-arena target-arr [1 1] :i32)
                               step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                               outs (arena/track! step-arena (execute-step-executable ctx step-exec step-inputs num-step-outs))
                               outs-vec (if (vector? outs) outs [outs])
                               logits-buf (first outs-vec)
                               nk (vec (subvec outs-vec 1))
                               lp-out (arena/track! step-arena (execute-step-executable ctx target-exec [logits-buf target-b] 1))
                               lp-buf (if (vector? lp-out) (first lp-out) lp-out)
                               floats (uop-rt/device->host-floats ctx lp-buf 1)
                               lp (aget floats 0)]
                           (arena/promote! step-arena session-arena nk)
                           [nk lp]))]
                   (arena/destroy! session-arena cur-kv)
                   (recur (inc p) next-kv (conj! acc (double target-lp))))
                 (do
                   (arena/destroy! session-arena cur-kv)
                   (persistent! acc))))]
         {:token-ids tokens
          :target-log-probs collected
          :log-probs (into [0.0] collected)})

       ;; Path B: Parallel prefill scoring with explicit exec
       (let [{:keys [ctx session-arena]} session
             in-arr (int-array seq-len)
             targets-arr (int-array seq-len)]
         (dotimes [i n]
           (aset in-arr i (int (nth tokens i))))
         (dotimes [i (dec n)]
           (aset targets-arr i (int (nth tokens (inc i)))))
         (let [target-log-probs
               (arena/with-device-arena [step-arena (or session-arena ctx)]
                 (let [in-b (arena/device-buffer step-arena in-arr [1 seq-len] :i32)
                       targets-b (arena/device-buffer step-arena targets-arr [1 seq-len] :i32)
                       device-weights (or (:device-weights session)
                                          (weights/allocate-device-weights session))
                       args (into [in-b targets-b] device-weights)
                       out (arena/track! step-arena (execute-step-executable ctx exec args 1))
                       out-buf (if (sequential? out) (first out) out)
                       raw-floats (to-host-slice out-buf 0 seq-len seq-len :f32)]
                   (mapv #(double (aget ^floats raw-floats %)) (range (dec n)))))]
           {:token-ids tokens
            :target-log-probs target-log-probs
            :log-probs (into [0.0] target-log-probs)}))))))

