(ns tools.smollm-inference
  "Top-level runnable integration script for end-to-end SmolLM-135M text generation via pure XLA execution."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [models.smollm :as smollm-logic]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 8
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :method :tensor-logic
   :compare false})

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k, --backend, --method, --compare)."
  [args]
  (loop [remaining (vec args)
         opts DEFAULT_CLI_OPTS]
    (if (empty? remaining)
      opts
      (let [flag (first remaining)
            val (second remaining)]
        (cond
          (and (= flag "--prompt") val)
          (recur (subvec remaining 2) (assoc opts :prompt val))

          (and (= flag "--max-new-tokens") val)
          (recur (subvec remaining 2) (assoc opts :max-new-tokens (Long/parseLong val)))

          (and (= flag "--temperature") val)
          (recur (subvec remaining 2) (assoc opts :temperature (Double/parseDouble val)))

          (and (= flag "--top-k") val)
          (recur (subvec remaining 2) (assoc opts :top-k (Long/parseLong val)))

          (and (= flag "--backend") val)
          (recur (subvec remaining 2) (assoc opts :backend (keyword val)))

          (and (= flag "--method") val)
          (recur (subvec remaining 2) (assoc opts :method (keyword val)))

          (and (= flag "--compare") val)
          (recur (subvec remaining 2) (assoc opts :compare (Boolean/parseBoolean val)))

          :else
          (recur (subvec remaining 1) opts))))))

(defn- prepare-input-tensor [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))

(defn- sample-logits
  "Performs temperature scaling and top-k sampling over logit float array.
   When temp <= 0.0, performs deterministic greedy argmax."
  [logits temp top-k]
  (if (<= temp 0.0)
    (let [n (count logits)]
      (loop [i 0 max-i 0 max-v (aget ^floats logits 0)]
        (if (< i n)
          (let [v (aget ^floats logits i)]
            (if (> v max-v)
              (recur (inc i) i v)
              (recur (inc i) max-i max-v)))
          max-i)))
    (let [indexed (map-indexed vector logits)
          sorted (sort-by second > indexed)
          k-truncated (take (min top-k (count logits)) sorted)
          max-logit (apply max (map second k-truncated))
          exp-logits (map (fn [[idx l]] [idx (Math/exp (/ (- l max-logit) temp))]) k-truncated)
          sum-exp (reduce + 0.0 (map second exp-logits))
          probs (map (fn [[idx e]] [idx (/ e sum-exp)]) exp-logits)
          r (rand)]
      (loop [ps probs accum 0.0]
        (if (seq ps)
          (let [[idx p] (first ps)
                new-accum (+ accum p)]
            (if (<= r new-accum)
              idx
              (recur (rest ps) new-accum)))
          (first (first probs)))))))

(defn -main
  "Runs end-to-end SmolLM-135M text generation pipeline: model loading, tokenization,
   full-model compilation (Tensor Logic or legacy tracer), and autoregressive decoding."
  [& args]
  (let [{:keys [prompt max-new-tokens temperature top-k backend method compare]} (parse-cli-args args)]
    (println "==================================================================")
    (println "    clj-xla SmolLM-135M End-to-End Autoregressive Generation Loop ")
    (println "==================================================================")

    ;; 1. Initialize PJRT runtime for specified backend
    (let [ctx (xla/init-backend! (or backend :cpu))
          model-dir ".models/smollm-135m"
          weights-path (str model-dir "/model.safetensors")]

      ;; 2. Load Tokenizer (BPE with HF tokenizer.json schema)
      (println (str "Loading SmolLM Tokenizer from [" model-dir "]..."))
      (let [tokenizer (tok/from-file model-dir)
            prompt-ids (proto/encode tokenizer prompt)]
        (println (format "Prompt: \"%s\"" prompt))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, method=%s, compare=%s"
                         max-new-tokens temperature top-k method compare))
        (println (format "Encoded Token IDs (%d tokens): %s" (count prompt-ids) prompt-ids))

        ;; 3. Load Safetensors weights header & memory-map 30 layers
        (println (str "Loading Safetensors metadata from [" weights-path "]..."))
        (let [arena (Arena/ofConfined)
              weights-mmap (st/map-safetensors-weights weights-path arena)
              metadata (:header weights-mmap)
              ^floats embed-tokens (st/get-tensor-floats weights-mmap "model.embed_tokens.weight")
              ^floats final-norm-w (st/get-tensor-floats weights-mmap "model.norm.weight")
              ^floats lm-head-w (or (try (st/get-tensor-floats weights-mmap "lm_head.weight")
                                         (catch Exception _ nil))
                                    embed-tokens)
              layers-weights (mapv (fn [i]
                                     (let [kmap (smollm-logic/weight-key-map i)]
                                       {:input-ln-w (st/get-tensor-floats weights-mmap (:input-ln-w kmap))
                                        :q-w (st/get-tensor-floats weights-mmap (:q-w kmap))
                                        :k-w (st/get-tensor-floats weights-mmap (:k-w kmap))
                                        :v-w (st/get-tensor-floats weights-mmap (:v-w kmap))
                                        :o-w (st/get-tensor-floats weights-mmap (:o-w kmap))
                                        :post-attn-ln-w (st/get-tensor-floats weights-mmap (:post-attn-ln-w kmap))
                                        :gate-w (st/get-tensor-floats weights-mmap (:gate-w kmap))
                                        :up-w (st/get-tensor-floats weights-mmap (:up-w kmap))
                                        :down-w (st/get-tensor-floats weights-mmap (:down-w kmap))}))
                                   (range 30))]
          (println (format "Parsed Safetensors header (%d tensors, %d layers loaded)."
                           (count metadata) (count layers-weights)))

          ;; 4. Define model signature and compilation function
          (let [max-seq-len 128
                invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                              [:embed_tokens [:tensor [49152 576] :f32]]
                              [:final_norm_w [:tensor [576] :f32]]
                              [:lm_head_w [:tensor [49152 576] :f32]]]
                             (mapcat (fn [i]
                                       [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                        [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                        [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                        [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                        [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                        [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                        [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                        [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                        [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                                     (range 30)))]
            (println "Lowering declarative Tensor Logic SmolLM-135M AST to StableHLO graph...")
            (let [ast (smollm-logic/smollm-model-ast {:num-layers 30 :max-seq-len max-seq-len})
                  graph (lower/ast->graph "full_smollm_model_logic" invars ast #{:logits})
                  exec (xla/compile-graph ctx graph)]

              (println "Transferring weights to PJRT Device Memory...")
              (let [embed-buf (xla/buffer-from-host-buffer ctx (:client ctx) embed-tokens [49152 576] 11)
                    final-norm-buf (xla/buffer-from-host-buffer ctx (:client ctx) final-norm-w [576] 11)
                    lm-head-buf (xla/buffer-from-host-buffer ctx (:client ctx) lm-head-w [49152 576] 11)
                    layer-bufs (mapv (fn [m]
                                       [(xla/buffer-from-host-buffer ctx (:client ctx) (:input-ln-w m) [576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:q-w m) [576 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:k-w m) [192 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:v-w m) [192 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:o-w m) [576 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:post-attn-ln-w m) [576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:gate-w m) [1536 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:up-w m) [1536 576] 11)
                                        (xla/buffer-from-host-buffer ctx (:client ctx) (:down-w m) [576 1536] 11)])
                                     layers-weights)
                    flat-layer-bufs (vec (apply concat layer-bufs))
                    flat-device-weights (into [embed-buf final-norm-buf lm-head-buf] flat-layer-bufs)]

                (println "Successfully compiled model to native XLA PjRtLoadedExecutable handle.")
                (println "\nGenerating tokens autoregressively...")
                (print prompt)
                (flush)
                (let [cur-tokens (atom (vec prompt-ids))]
                  (dotimes [_ max-new-tokens]
                    (let [s-len (count @cur-tokens)
                          in-arr (prepare-input-tensor @cur-tokens max-seq-len)
                          in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                          args (into [in-b] flat-device-weights)
                          out (xla/execute exec args)
                          l (xla/to-host-slice out (dec s-len) 49152)
                          next-id (sample-logits l temperature top-k)]
                      (swap! cur-tokens conj next-id)
                      (print (proto/decode tokenizer [next-id]))
                      (flush)))
                  (println "\n\n==================================================================")
                  (println "Final Generated Sequence:")
                  (println (proto/decode tokenizer @cur-tokens))
                  (println "==================================================================")
                  (println "=== End-to-End SmolLM-135M Generation Completed! ==="))))))))))

(defn -main-wrapper [& args]
  (apply -main args))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
