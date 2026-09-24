(ns tools.smollm-inference
  "End-to-End SmolLM-135M Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [einsum.core :as xla]
            [einsum.models.smollm :as smollm-logic]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto]
            [tools.cli :as cli])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  (assoc cli/DEFAULT_INFERENCE_OPTS
         :max-new-tokens 8
         :precision :f32))

(def DEFAULT_MODEL_DIRS
  (cli/STANDARD_MODEL_CANDIDATES :smollm))

(defn parse-cli-args
  "Parses CLI arguments for SmolLM inference."
  [args]
  (cli/parse-cli-args args DEFAULT_CLI_OPTS))

(defn find-model-dir
  "Resolves SmolLM model directory from explicit path or candidates."
  ([model-dirs]
   (cli/find-model-dir model-dirs))
  ([]
   (cli/find-model-dir :smollm)))

(def prepare-input-tensor cli/prepare-input-tensor)

(defn generate-text
  "Runs full end-to-end text generation on an initialized or discovered SmolLM model."
  ([opts]
   (let [{:keys [prompt max-new-tokens temperature top-k backend precision model-dir]} (merge DEFAULT_CLI_OPTS opts)
         ctx (xla/init-backend! (or backend :cpu))
         resolved-dir (cli/find-model-dir model-dir :smollm)
         tokenizer (tok/from-file resolved-dir)
         prompt-ids (proto/encode tokenizer prompt)
         prompt-len (count prompt-ids)
         max-seq-len (max 32 (+ prompt-len max-new-tokens 4))
         eos (try (proto/eos-id tokenizer) (catch Exception _ nil))]

     (println (str "Loading Safetensors metadata from [" resolved-dir "]..."))
     (let [arena (Arena/ofAuto)
           weights (st/map-safetensors-weights resolved-dir arena)
           header (or (:header weights) {})
           emb-shape (get-in header ["model.embed_tokens.weight" "shape"] [49152 576])
           vocab-size (nth emb-shape 0 49152)
           hidden-dim (nth emb-shape 1 576)
           num-layers (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
           num-layers (if (pos? num-layers) num-layers 30)
           weight-dtype (or precision (get-in header ["model.embed_tokens.weight" :dtype] :f32))
           config {:vocab-size vocab-size
                   :hidden-dim hidden-dim
                   :num-layers num-layers
                   :max-seq-len max-seq-len
                   :weight-dtype weight-dtype}]

       (println (format "Prompt: \"%s\"" prompt))
       (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s (%d layers)"
                        max-new-tokens temperature top-k (name weight-dtype) num-layers))
       (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))

       (println "Lowering & JIT Compiling full SmolLM via Tensor Logic Hiccup AST & WeightStore...")
       (with-open [session-arena (xla/create-arena ctx)]
         (let [store (xla/create-weight-store ctx weights {:aliases (smollm-logic/smollm-alias-resolver weights)
                                                           :arena session-arena})
               ast (smollm-logic/smollm-model-ast config)
               kernel (xla/compile-kernel ctx "full_smollm_model" ast
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
                 (print (proto/decode tokenizer [next-id]))
                 (flush)
                 (when (and eos (= next-id eos))
                   (println "\nReached EOS token.")
                   (reduced nil)))))
           (println "\n\n==================================================================")
           (println "Final Generated Sequence:")
           (println (proto/decode tokenizer @cur-tokens))
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
         resolved-dir (cli/find-model-dir (:model-dir opts) :smollm)
         tokenizer (tok/from-file resolved-dir)]
     {:ctx ctx
      :opts opts
      :model-dir resolved-dir
      :tokenizer tokenizer})))

(defn -main
  "CLI entrypoint for SmolLM text generation."
  [& args]
  (let [opts (parse-cli-args args)]
    (println "==================================================================")
    (println "      clj-xla SmolLM-135M End-to-End Autoregressive Generation    ")
    (println "==================================================================")
    (generate-text opts)))

(defn -main-wrapper [& args]
  (apply -main args))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
