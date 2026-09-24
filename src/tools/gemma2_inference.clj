(ns tools.gemma2-inference
  "End-to-End Gemma 2B Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [einsum.core :as xla]
            [einsum.models.gemma :as gemma]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [tools.cli :as cli])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS cli/DEFAULT_INFERENCE_OPTS)

(def DEFAULT_MODEL_DIRS
  (cli/STANDARD_MODEL_CANDIDATES :gemma-2))

(defn parse-cli-args
  "Parses CLI arguments for Gemma 2 inference."
  [args]
  (cli/parse-cli-args args DEFAULT_CLI_OPTS))

(defn find-model-dir
  "Resolves Gemma 2 model directory from explicit path or candidates."
  ([model-dirs]
   (cli/find-model-dir model-dirs))
  ([]
   (cli/find-model-dir :gemma-2)))

(def prepare-input-tensor cli/prepare-input-tensor)

(defn generate-text
  "Runs full end-to-end text generation on an initialized or discovered Gemma 2 model."
  ([opts]
   (let [{:keys [prompt max-new-tokens temperature top-k backend precision model-dir]} (merge DEFAULT_CLI_OPTS opts)
         ctx (xla/init-backend! (or backend :cpu))
         resolved-dir (cli/find-model-dir model-dir :gemma-2)
         tokenizer (tok/from-file resolved-dir)
         prompt-ids (into [(bos-id tokenizer)] (encode tokenizer prompt))
         prompt-len (count prompt-ids)
         max-seq-len (max 32 (+ prompt-len max-new-tokens 4))
         eos (eos-id tokenizer)]

     (println (str "Loading Safetensors metadata from [" resolved-dir "]..."))
     (let [arena (Arena/ofAuto)
           weights (st/map-safetensors-weights resolved-dir arena)
           header (or (:header weights) {})
           emb-shape (get-in header ["model.embed_tokens.weight" "shape"] [256000 2304])
           q-shape (get-in header ["model.layers.0.self_attn.q_proj.weight" "shape"] [2048 2304])
           k-shape (get-in header ["model.layers.0.self_attn.k_proj.weight" "shape"] [1024 2304])
           gate-shape (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"] [9216 2304])

           vocab-size (nth emb-shape 0 256000)
           hidden-dim (nth emb-shape 1 2304)
           q-dim (nth q-shape 0 2048)
           kv-dim (nth k-shape 0 1024)
           intermediate-dim (nth gate-shape 0 9216)
           num-layers (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
           num-heads (quot q-dim 256)
           num-kv-heads (quot kv-dim 256)
           head-dim 256
           weight-dtype (or precision (get-in header ["model.embed_tokens.weight" :dtype] :bf16))
           config {:vocab-size vocab-size
                   :hidden-dim hidden-dim
                   :intermediate-dim intermediate-dim
                   :num-heads num-heads
                   :num-kv-heads num-kv-heads
                   :head-dim head-dim
                   :num-layers num-layers
                   :max-seq-len max-seq-len
                   :weight-dtype weight-dtype}]

       (println (format "Prompt: \"%s\"" prompt))
       (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s (%d layers)"
                        max-new-tokens temperature top-k (name weight-dtype) num-layers))
       (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))

       (println "Lowering & JIT Compiling full Gemma 2 via Tensor Logic Hiccup AST & WeightStore...")
       (with-open [session-arena (xla/create-arena ctx)]
         (let [store (xla/create-weight-store ctx weights {:aliases (gemma/gemma2-alias-resolver "model.")
                                                           :arena session-arena})
               ast (gemma/gemma2-model-ast config)
               kernel (xla/compile-kernel ctx "full_gemma2_model" ast
                                          {:in {:x [:tensor [1 max-seq-len] :i32]}
                                           :weights store
                                           :out [:logits]})
               cur-tokens (atom (vec prompt-ids))]
           (println "Successfully compiled model to native XLA PjRtLoadedExecutable handle.")
           (println "\nGenerating tokens autoregressively...")
           (print prompt)
           (flush)
           (dotimes [_ max-new-tokens]
             (xla/with-device-arena [_step-arena session-arena]
               (let [seq-len (count @cur-tokens)
                     input-tensor (prepare-input-tensor @cur-tokens max-seq-len)
                     {:keys [logits]} (kernel {:x input-tensor})
                     slice (xla/to-host-slice logits (dec seq-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
                     next-id (sampling/sample-logits slice {:temperature temperature :top-k top-k})]
                 (swap! cur-tokens conj next-id)
                 (print (decode tokenizer [next-id]))
                 (flush)
                 (when (= next-id eos)
                   (println "\nReached EOS token.")
                   (reduced nil)))))
           (println "\n\n==================================================================")
           (println "Final Generated Sequence:")
           (println (decode tokenizer @cur-tokens))
           (println "==================================================================")
           @cur-tokens)))))
  ([session prompt-text]
   (generate-text (assoc (:opts session) :prompt prompt-text :model-dir (:model-dir session)))))

(defn init-inference-session
  "Initializes PJRT runtime and loads safetensors metadata and tokenizer.
   Returns a session map for REPL workflows."
  ([] (init-inference-session {}))
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         ctx (xla/init-backend! (or (:backend opts) :cpu))
         resolved-dir (cli/find-model-dir (:model-dir opts) :gemma-2)
         tokenizer (tok/from-file resolved-dir)]
     {:ctx ctx
      :opts opts
      :model-dir resolved-dir
      :tokenizer tokenizer})))

(defn -main
  "CLI entrypoint for Gemma 2 text generation."
  [& args]
  (let [opts (parse-cli-args args)]
    (println "==================================================================")
    (println "      clj-xla Gemma 2B End-to-End Autoregressive Generation       ")
    (println "==================================================================")
    (generate-text opts)))

(defn -main-wrapper [& args]
  (apply -main args))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
