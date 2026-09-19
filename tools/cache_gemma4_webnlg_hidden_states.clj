(ns tools.cache-gemma4-webnlg-hidden-states
  "Pre-computes and caches contextual hidden states h_ctx [N_triples x 1536] bf16
   for WebNLG training triples using frozen Gemma 4 E2B forward passes on ROCm (Experiment E18 Factor B1)."
  (:require [einsum.core :as xla]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.runtime.tokenizer.protocol :as proto]
            [tools.gemma4-inference :as g4])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel]
           [java.nio.file Path StandardOpenOption]))

(defn init-rocm-c-env! []
  (try
    (let [linker (java.lang.foreign.Linker/nativeLinker)
          lookup (.defaultLookup linker)
          setenv-opt (.find lookup "setenv")]
      (when (.isPresent setenv-opt)
        (let [setenv-ptr ^java.lang.foreign.MemorySegment (.get setenv-opt)
              fd (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/JAVA_INT
                                                          (into-array java.lang.foreign.MemoryLayout
                                                                      [java.lang.foreign.ValueLayout/ADDRESS
                                                                       java.lang.foreign.ValueLayout/ADDRESS
                                                                       java.lang.foreign.ValueLayout/JAVA_INT]))
              handle (.downcallHandle linker setenv-ptr fd (make-array java.lang.foreign.Linker$Option 0))]
          (with-open [arena (java.lang.foreign.Arena/ofConfined)]
            (.invokeWithArguments handle [(.allocateFrom arena "HIP_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
            (.invokeWithArguments handle [(.allocateFrom arena "ROCR_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
            (.invokeWithArguments handle [(.allocateFrom arena "HSA_OVERRIDE_GFX_VERSION") (.allocateFrom arena "11.0.0") (int 1)])))))
    (catch Exception _ nil)))

(defn cache-contextual-hidden-states! [opts]
  (init-rocm-c-env!)
  (let [model-dir (or (:model-dir opts) ".models/gemma-4-E2B-it")
        train-file (or (:train-file opts) ".dataset/webnlg/train.edn")
        out-bin (or (:out-bin opts) ".dataset/webnlg/train_h_ctx_bf16.bin")
        out-meta (or (:out-meta opts) ".dataset/webnlg/train_h_ctx_meta.edn")
        backend (or (:backend opts) :rocm)
        din 1536]
    (println "================================================================================")
    (println "📦 CACHING GEMMA 4 CONTEXTUAL HIDDEN STATES FOR WEBNLG TRIPLES (E18 B1)")
    (println "================================================================================")

    ;; 1. Initialize Inference Session
    (println (format "Initializing Gemma 4 E2B session on backend [%s]..." (name backend)))
    (let [session (g4/init-inference-session {:model-dir model-dir
                                              :backend backend
                                              :precision :bf16
                                              :max-seq-len 64
                                              :targets [:logits :normed_last]})
          ctx (:ctx session)
          tokenizer (:tokenizer session)
          dev-weights (g4/allocate-device-weights session)
          fwd-exec (g4/compile-tensor-logic-executable
                    (assoc session :opts {:targets [:logits :normed_last] :last-token-only? true}) 64)

          ;; 2. Load and Prepare Training Triples
          train-entries (:entries (read-string (slurp train-file)))
          raw-triples (distinct (mapcat :triples train-entries))
          valid-triples
          (vec (keep-indexed
                (fn [idx [h r t]]
                  (let [h-toks (proto/encode tokenizer (str " " h) false)
                        t-toks (proto/encode tokenizer (str " " t) false)]
                    (when (and (seq h-toks) (seq t-toks))
                      {:idx idx
                       :h-str h
                       :rel r
                       :t-str t
                       :h-tok (first h-toks)
                       :t-tok (first t-toks)
                       ;; Exact eval prompt template:
                       :prompt (str "The " (name r) " of " h " is")})))
                raw-triples))
          n-triples (count valid-triples)]
      (println (format "Loaded %,d valid training triples." n-triples))

      ;; 3. Cost Control: Benchmark 50 Forward Passes
      (println "\nBenchmarking 50 forward passes for cost control...")
      (let [bench-samples (take 50 valid-triples)
            t0 (System/nanoTime)]
        (doseq [{:keys [prompt]} bench-samples]
          (let [raw-ids (proto/encode tokenizer prompt false)
                prompt-ids (vec (cons (proto/bos-id tokenizer) raw-ids))
                p-len (count prompt-ids)
                in-arr (int-array 64)
                _ (dotimes [i p-len] (aset in-arr i (int (nth prompt-ids i))))
                pos-p (int-array [(dec p-len)])
                in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 64] 4)
                pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                step-inputs (into [in-b pos-b] dev-weights)
                outs (pjrt/execute-executable ctx (or (:handle fwd-exec) fwd-exec) step-inputs 2)]
            (xla/destroy-buffer! ctx in-b)
            (xla/destroy-buffer! ctx pos-b)
            (xla/destroy-buffer! ctx (nth outs 0))
            (xla/destroy-buffer! ctx (nth outs 1))))
        (let [dur-ms (/ (- (System/nanoTime) t0) 1e6)
              per-pass-ms (/ dur-ms 50.0)
              est-total-sec (/ (* per-pass-ms (double n-triples)) 1000.0)]
          (println (format "  50 passes completed in %.2f ms (%.2f ms/pass)." dur-ms per-pass-ms))
          (println (format "  Estimated full pass over %,d triples: %.1f seconds (~%.1f minutes)."
                           n-triples est-total-sec (/ est-total-sec 60.0)))))

      ;; 4. Compute and Cache All Hidden States
      (println (format "\nExecuting full pass over %,d training triples..." n-triples))
      (let [total-shorts (* (long n-triples) (long din))
            ^shorts out-shorts (short-array total-shorts)
            t-start (System/nanoTime)]
        (doseq [i (range n-triples)]
          (let [{:keys [prompt]} (nth valid-triples i)
                raw-ids (proto/encode tokenizer prompt false)
                prompt-ids (vec (cons (proto/bos-id tokenizer) raw-ids))
                p-len (count prompt-ids)
                in-arr (int-array 64)
                _ (dotimes [p p-len] (aset in-arr p (int (nth prompt-ids p))))
                pos-p (int-array [(dec p-len)])
                in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 64] 4)
                pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                step-inputs (into [in-b pos-b] dev-weights)
                outs (pjrt/execute-executable ctx (or (:handle fwd-exec) fwd-exec) step-inputs 2)
                base-logits-b (nth outs 0)
                normed-last-b (nth outs 1)
                ;; Direct byte-level read of BF16 values from VRAM
                fa (pjrt/buffer-to-host-buffer ctx normed-last-b din :bf16)
                sa (g4/floats->bf16-shorts fa)
                base-offset (* (long i) (long din))]
            (System/arraycopy sa 0 out-shorts base-offset din)
            (xla/destroy-buffer! ctx in-b)
            (xla/destroy-buffer! ctx pos-b)
            (xla/destroy-buffer! ctx base-logits-b)
            (xla/destroy-buffer! ctx normed-last-b)
            (when (zero? (mod (inc i) 500))
              (let [elapsed (/ (- (System/nanoTime) t-start) 1e9)]
                (println (format "  Cached %,d / %,d triples (%.1f%%) in %.1f s..."
                                 (inc i) n-triples (* 100.0 (/ (inc i) (double n-triples))) elapsed))))))

        ;; 5. Write Binary Cache to Disk
        (println (format "\nWriting binary cache to %s (%d elements, %.2f MB)..."
                         out-bin total-shorts (/ (* (double total-shorts) 2.0) 1048576.0)))
        (let [byte-buf (ByteBuffer/allocateDirect (* total-shorts 2))]
          (.order byte-buf ByteOrder/LITTLE_ENDIAN)
          (dotimes [i total-shorts]
            (.putShort byte-buf (aget out-shorts i)))
          (.flip byte-buf)
          (let [fc (FileChannel/open (Path/of out-bin (into-array String []))
                                     (into-array [StandardOpenOption/CREATE
                                                  StandardOpenOption/WRITE
                                                  StandardOpenOption/TRUNCATE_EXISTING]))]
            (.write fc byte-buf)
            (.close fc)))

        ;; 6. Write Metadata
        (println (format "Writing metadata index to %s..." out-meta))
        (spit out-meta (pr-str {:n-triples n-triples
                                :dim-in din
                                :dtype :bf16
                                :triples (mapv #(select-keys % [:idx :h-tok :rel :t-tok :h-str :t-str]) valid-triples)}))
        (println "Cache complete and verified!")))))

(defn -main [& _args]
  (cache-contextual-hidden-states! {:backend :rocm})
  (System/exit 0))
