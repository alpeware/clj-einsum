(ns tools.gemma2-inference
  "End-to-End Gemma 2B Autoregressive Generation Loop using clj-xla PJRT backend."
  (:require [clojure.java.io :as io]
            [einsum.core :as xla]
            [einsum.models.gemma :as gemma]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode eos-id]])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 10
   :temperature 0.70
   :top-k 10
   :backend :cpu
   :precision :bf16
   :verbose false})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-2-2b-it" ".models/gemma-2b" ".models/gemma-2-2b" ".models/gemma"])

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k, --backend, --precision, --model-dir, --verbose)."
  [args]
  (loop [cli-args args
         opts DEFAULT_CLI_OPTS]
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

          (= arg "--precision")
          (recur (drop 2 cli-args) (assoc opts :precision (keyword (second cli-args))))

          (= arg "--model-dir")
          (recur (drop 2 cli-args) (assoc opts :model-dir (second cli-args)))

          (= arg "--verbose")
          (recur (rest cli-args) (assoc opts :verbose true))

          :else
          (recur (rest cli-args) opts)))
      opts)))

(defn find-model-dir
  "Searches `model-dirs` for an existing directory containing `.safetensors` files.
   Throws an ExceptionInfo if no model files are found."
  [model-dirs]
  (let [existing (first (filter (fn [d]
                                  (let [f (io/file d)]
                                    (and (.exists f)
                                         (or (.exists (io/file f "model.safetensors"))
                                             (.exists (io/file f "model-00001-of-00002.safetensors"))))))
                                model-dirs))]
    (if existing
      existing
      (throw (ex-info (str "Model directory with safetensors not found in candidates: " (vec model-dirs))
                      {:searched-dirs model-dirs})))))

(defn- prepare-input-tensor
  "Pads token IDs sequence to `max-len` with zero padding."
  [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))

(defn generate-text
  "Runs full end-to-end text generation on an initialized or discovered Gemma 2 model."
  ([opts]
   (let [{:keys [prompt max-new-tokens temperature top-k backend precision model-dir]} (merge DEFAULT_CLI_OPTS opts)
         ctx (xla/init-backend! (or backend :cpu))
         resolved-dir (or model-dir (find-model-dir DEFAULT_MODEL_DIRS))
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
         resolved-dir (or (:model-dir opts) (find-model-dir DEFAULT_MODEL_DIRS))
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
