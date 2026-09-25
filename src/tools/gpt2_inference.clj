(ns tools.gpt2-inference
  "End-to-End GPT-2 Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [einsum.core :as xla]
            [einsum.models.gpt2 :as gpt2-logic]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.protocol :as proto]
            [tools.cli :as cli])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  (assoc cli/DEFAULT_INFERENCE_OPTS
         :prompt "The quick brown fox"
         :max-new-tokens 15
         :method :tensor-logic
         :compare false))

(def parse-cli-args
  "Parses CLI arguments for GPT-2 inference."
  (fn [args]
    (cli/parse-cli-args args DEFAULT_CLI_OPTS)))

(def find-model-dir
  "Resolves GPT-2 model directory."
  (fn [& [explicit-path]]
    (cli/find-model-dir explicit-path :gpt2)))

(def prepare-input-tensor cli/prepare-input-tensor)

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

(defn -main [& args]
  (let [opts (parse-cli-args args)
        {:keys [prompt max-new-tokens temperature top-k backend model-dir]} opts]
    (println "==================================================================")
    (println "      clj-xla GPT-2 End-to-End Autoregressive Generation Loop     ")
    (println "==================================================================")
    (let [ctx (xla/init-backend! (or backend :cpu))
          tokenizer-dir (find-model-dir model-dir)
          safetensors-path (str tokenizer-dir "/model.safetensors")]

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

              (if (= :interpreter (:backend ctx))
                (println "Successfully lowered model for Pure-JVM Interpreter execution.")
                (println (format "Successfully compiled model to native XLA [%s] executable handle." (name (or (:backend ctx) :cpu)))))
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
                          next-id (sampling/sample-logits logits-slice {:temperature temperature :top-k top-k})]
                      (swap! cur-tokens conj next-id)
                      (print (bpe-token->str (get id->tok next-id next-id)))
                      (flush))))
                (println "\n\n==================================================================")
                (println "Final Generated Sequence:")
                (println (proto/decode tokenizer @cur-tokens))
                (println "==================================================================")))))))))

(defn -main-wrapper [& args]
  (apply -main args))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
