(ns tools.gpt2-inference
  "End-to-End GPT-2 Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [einsum.core :as xla]
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
              num-layers 12
              max-seq-len 128]
          (println (format "Parsed Safetensors header (%d tensors, %d layers configured)."
                           (count (:header weights)) num-layers))

          (println "Lowering & JIT Compiling full GPT-2 via Tensor Logic Hiccup AST & WeightStore...")
          (with-open [session-arena (xla/create-arena ctx)]
            (let [store (xla/create-weight-store ctx weights {:aliases gpt2-logic/gpt2-alias-resolver
                                                              :arena session-arena})
                  ast (gpt2-logic/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
                  gpt2-kernel (xla/compile-kernel ctx "full_gpt2_model_logic" ast
                                                  {:in {:x [:tensor [1 max-seq-len] :i32]
                                                        :pos_ids [:tensor [1 max-seq-len] :i32]}
                                                   :weights store
                                                   :out [:logits]})
                  pos-array (int-array (range max-seq-len))
                  pos-buf (xla/device-buffer session-arena pos-array [1 max-seq-len] :i32)]

              (println "Successfully compiled model to native XLA PjRtLoadedExecutable handle.")
              (println "\nGenerating tokens autoregressively...")
              (print prompt)
              (flush)
              (let [cur-tokens (atom encoded-tokens)]
                (dotimes [_ max-new-tokens]
                  (xla/with-device-arena [_step-arena session-arena]
                    (let [seq-len (count @cur-tokens)
                          input-tensor (prepare-input-tensor @cur-tokens max-seq-len)
                          {:keys [logits]} (gpt2-kernel {:x input-tensor :pos_ids pos-buf})
                          logits-slice (xla/to-host-slice logits (dec seq-len) 50257 (* max-seq-len 50257))
                          next-id (sample-logits logits-slice temperature top-k)]
                      (swap! cur-tokens conj next-id)
                      (print (bpe-token->str (get id->tok next-id next-id)))
                      (flush))))
                (println "\n\n==================================================================")
                (println "Final Generated Sequence:")
                (println (proto/decode tokenizer @cur-tokens))
                (println "==================================================================")))))))))

(defn -main-wrapper [& args]
  (apply -main args))
