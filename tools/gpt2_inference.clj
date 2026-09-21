(ns tools.gpt2-inference
  "End-to-End GPT-2 Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.models.gpt2 :as gpt2-logic]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.protocol :as proto])
  (:import [java.lang.foreign Arena]))

(defn- parse-cli-args [args]
  (loop [cli-args args
         opts {:prompt "The quick brown fox"
               :max-new-tokens 15
               :temperature 0.70
               :top-k 10
               :backend :cpu
               :method :tensor-logic
               :compare false}]
    (if (seq cli-args)
      (let [arg (first cli-args)]
        (cond
          (= arg "--prompt")
          (recur (drop 2 cli-args) (assoc opts :prompt (second cli-args)))

          (= arg "--max-new-tokens")
          (recur (drop 2 cli-args) (assoc opts :max-new-tokens (Integer/parseInt (second cli-args))))

          (= arg "--temperature")
          (recur (drop 2 cli-args) (assoc opts :temperature (Double/parseDouble (second cli-args))))

          (= arg "--top-k")
          (recur (drop 2 cli-args) (assoc opts :top-k (Integer/parseInt (second cli-args))))

          (= arg "--backend")
          (recur (drop 2 cli-args) (assoc opts :backend (keyword (second cli-args))))

          (= arg "--method")
          (recur (drop 2 cli-args) (assoc opts :method (keyword (second cli-args))))

          (= arg "--compare")
          (recur (drop 2 cli-args) (assoc opts :compare (Boolean/parseBoolean (second cli-args))))

          :else
          (recur (rest cli-args) opts)))
      opts)))

(defn- sample-logits
  "Performs temperature scaling and top-k sampling over logit float array.
   When temp <= 0.0, performs deterministic greedy argmax."
  [logits temp top-k]
  (if (<= temp 0.0)
    (first (apply max-key second (map-indexed vector logits)))
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

(defn- bpe-token->str [tok-str]
  (if (string? tok-str)
    (let [bytes-vec (keep (fn [ch]
                            (let [code (int ch)]
                              (cond
                                (= ch \Ġ) (byte 32)
                                (= ch \Ċ) (byte 10)
                                (<= code 255) (byte code)
                                :else nil)))
                          tok-str)]
      (String. (byte-array bytes-vec) "UTF-8"))
    (str tok-str)))

(defn- prepare-input-tensor [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    [(vec padded)]))

(defn -main [& args]
  (let [{:keys [prompt max-new-tokens temperature top-k backend]} (parse-cli-args args)]
    (println "==================================================================")
    (println "      clj-xla GPT-2 End-to-End Autoregressive Generation Loop     ")
    (println "==================================================================")
    (let [ctx (xla/init-backend! (or backend :cpu))
          tokenizer-dir ".models/gpt2"
          safetensors-path ".models/gpt2/model.safetensors"]

      (println (str "Loading GPT-2 Tokenizer from [" tokenizer-dir "]..."))
      (let [tokenizer (bpe/load-bpe-tokenizer (str tokenizer-dir "/vocab.json") (str tokenizer-dir "/merges.txt"))
            id->tok (:vocab tokenizer)
            encoded-tokens (proto/encode tokenizer prompt)]

        (println (str "Prompt: \"" prompt "\""))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d"
                         max-new-tokens temperature top-k))
        (println (str "Encoded Subword Token IDs (" (count encoded-tokens) " tokens): " (vec encoded-tokens)))

        (println (str "Loading Safetensors metadata from [" safetensors-path "]..."))
        (let [arena (Arena/ofAuto)
              weights (st/map-safetensors-weights safetensors-path arena)
              header (:header weights)
              ln-f-g (st/get-tensor-floats weights "ln_f.weight")
              ln-f-b (st/get-tensor-floats weights "ln_f.bias")
              wte-floats (st/get-tensor-floats weights "wte.weight")
              wpe-floats (st/get-tensor-floats weights "wpe.weight")
              num-layers 12
              layer-weights (mapv (fn [i]
                                    {:ln1-g (st/get-tensor-floats weights (format "h.%d.ln_1.weight" i))
                                     :ln1-b (st/get-tensor-floats weights (format "h.%d.ln_1.bias" i))
                                     :c-attn-w (st/get-tensor-floats weights (format "h.%d.attn.c_attn.weight" i))
                                     :c-attn-b (st/get-tensor-floats weights (format "h.%d.attn.c_attn.bias" i))
                                     :c-proj-w (st/get-tensor-floats weights (format "h.%d.attn.c_proj.weight" i))
                                     :c-proj-b (st/get-tensor-floats weights (format "h.%d.attn.c_proj.bias" i))
                                     :ln2-g (st/get-tensor-floats weights (format "h.%d.ln_2.weight" i))
                                     :ln2-b (st/get-tensor-floats weights (format "h.%d.ln_2.bias" i))
                                     :mlp-fc-w (st/get-tensor-floats weights (format "h.%d.mlp.c_fc.weight" i))
                                     :mlp-fc-b (st/get-tensor-floats weights (format "h.%d.mlp.c_fc.bias" i))
                                     :mlp-proj-w (st/get-tensor-floats weights (format "h.%d.mlp.c_proj.weight" i))
                                     :mlp-proj-b (st/get-tensor-floats weights (format "h.%d.mlp.c_proj.bias" i))})
                                  (range num-layers))
              flat-layer-weights (vec (mapcat (fn [m] [(:ln1-g m) (:ln1-b m) (:c-attn-w m) (:c-attn-b m)
                                                       (:c-proj-w m) (:c-proj-b m) (:ln2-g m) (:ln2-b m)
                                                       (:mlp-fc-w m) (:mlp-fc-b m) (:mlp-proj-w m) (:mlp-proj-b m)])
                                              layer-weights))
              max-seq-len 128
              invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                            [:pos_ids [:tensor [1 max-seq-len] :i32]]
                            [:ln_f_g [:tensor [768] :f32]]
                            [:ln_f_b [:tensor [768] :f32]]
                            [:wte [:tensor [50257 768] :f32]]
                            [:wpe [:tensor [1024 768] :f32]]]
                           (mapcat (fn [i]
                                     [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                      [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                      [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                      [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                      [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                      [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                      [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                      [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                      [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                      [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                      [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                      [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                                   (range num-layers)))]

          (println (format "Parsed Safetensors header (%d tensors, %d layers loaded)."
                           (count header) num-layers))

          (println "Lowering & JIT Compiling full GPT-2 via Tensor Logic Hiccup AST...")
          (let [ast (gpt2-logic/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
                graph (lower/ast->graph "full_gpt2_model_logic" invars ast #{:logits})
                exec (xla/compile-graph ctx graph)
                pos-array (int-array (range max-seq-len))
                pos-buf (xla/buffer-from-host-buffer ctx (:client ctx) pos-array [1 max-seq-len] 4)
                ln-f-g-buf (xla/buffer-from-host-buffer ctx (:client ctx) ln-f-g [768] 11)
                ln-f-b-buf (xla/buffer-from-host-buffer ctx (:client ctx) ln-f-b [768] 11)
                wte-buf (xla/buffer-from-host-buffer ctx (:client ctx) wte-floats [50257 768] 11)
                wpe-buf (xla/buffer-from-host-buffer ctx (:client ctx) wpe-floats [1024 768] 11)
                weight-bufs (mapv (fn [idx w]
                                    (let [[_var-name [_kw shape dtype]] (nth invars (+ 6 idx))
                                          dtype-enum (if (= dtype :i32) 4 11)]
                                      (xla/buffer-from-host-buffer ctx (:client ctx) w shape dtype-enum)))
                                  (range (count flat-layer-weights))
                                  flat-layer-weights)
                flat-device-weights (into [pos-buf ln-f-g-buf ln-f-b-buf wte-buf wpe-buf] weight-bufs)]

            (println "Successfully compiled model to native XLA PjRtLoadedExecutable handle.")
            (println "\nGenerating tokens autoregressively...")
            (print prompt)
            (flush)
            (let [cur-tokens (atom encoded-tokens)]
              (dotimes [_ max-new-tokens]
                (let [seq-len (count @cur-tokens)
                      input-tensor (prepare-input-tensor @cur-tokens max-seq-len)
                      input-args (into [input-tensor] flat-device-weights)
                      out (xla/execute exec input-args)
                      logits (xla/to-host-slice out (dec seq-len) 50257 (* max-seq-len 50257))
                      next-id (sample-logits logits temperature top-k)]
                  (swap! cur-tokens conj next-id)
                  (print (bpe-token->str (get id->tok next-id next-id)))
                  (flush)))
              (println "\n\n==================================================================")
              (println "Final Generated Sequence:")
              (println (proto/decode tokenizer @cur-tokens))
              (println "=================================================================="))))))))

(defn -main-wrapper [& args]
  (apply -main args))
