(ns tools.gemma2-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 2B text generation via pure XLA execution."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [models.gemma :as gemma]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode eos-id]]
            [clojure.java.io :as io])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 10
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :verbose false})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-2-2b-it" ".models/gemma-2b" ".models/gemma-2-2b" ".models/gemma"])

(defn parse-cli-args
  "Parses command-line flags (--prompt, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose)."
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

          (and (= flag "--precision") val)
          (recur (subvec remaining 2) (assoc opts :precision (keyword val)))

          (= flag "--verbose")
          (recur (subvec remaining 1) (assoc opts :verbose true))

          :else
          (recur (subvec remaining 1) opts))))))

(defn find-model-dir
  "Searches `model-dirs` for an existing directory containing `.safetensors` files.
   Throws an ExceptionInfo if no model files are found."
  ([model-dirs]
   (let [existing (first (filter (fn [d]
                                   (let [f (io/file d)]
                                     (and (.exists f)
                                          (or (.exists (io/file f "model.safetensors"))
                                              (.exists (io/file f "model-00001-of-00002.safetensors"))))))
                                 model-dirs))]
     (if existing
       existing
       (throw (ex-info (str "Model directory with safetensors not found in candidates: " (vec model-dirs))
                       {:searched-dirs model-dirs}))))))

(defn load-weight-buffer
  "Loads a single weight tensor from `weights-mmap` into PJRT device memory in specified precision."
  [ctx weights-mmap tensor-name shape weight-dtype weight-enum]
  (let [host-data (if (= weight-dtype :f32)
                    (st/get-tensor-floats weights-mmap tensor-name)
                    (st/get-tensor-bf16-shorts weights-mmap tensor-name))]
    (xla/buffer-from-host-buffer ctx (:client ctx) host-data shape weight-enum)))

(defn init-inference-session
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for REPL/CLI sessions.
   Returns an inference session map."
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         {:keys [backend precision]} opts
         ctx (xla/init-backend! (or backend :cpu))
         model-dir (find-model-dir DEFAULT_MODEL_DIRS)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights model-dir arena)
         tokenizer (tok/from-file model-dir)
         header (or (:header weights-mmap) {})
         emb-shape (get-in header ["model.embed_tokens.weight" "shape"] [256000 2048])
         q-shape (get-in header ["model.layers.0.self_attn.q_proj.weight" "shape"] [2048 2048])
         k-shape (get-in header ["model.layers.0.self_attn.k_proj.weight" "shape"] [256 2048])
         gate-shape (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"] [16384 2048])

         vocab-size (nth emb-shape 0 256000)
         hidden-dim (nth emb-shape 1 2048)
         q-dim (nth q-shape 0 2048)
         kv-dim (nth k-shape 0 256)
         intermediate-dim (nth gate-shape 0 16384)
         num-layers (count (filter #(re-find #"^model\.layers\.\d+\.input_layernorm\.weight$" %) (keys header)))
         num-heads (quot q-dim 256)
         num-kv-heads (quot kv-dim 256)
         head-dim 256
         max-seq-len 128
         kv-cache-shape [1 num-kv-heads max-seq-len head-dim]
         weight-dtype (or precision :bf16)
         weight-enum (if (= weight-dtype :f32) 11 13)]

     (println (str "Loaded Gemma model weights from [" model-dir "] in [" (name weight-dtype) "] precision (" num-layers " layers)."))

     {:ctx ctx
      :opts opts
      :model-dir model-dir
      :tokenizer tokenizer
      :weights-mmap weights-mmap
      :arena arena
      :config {:vocab-size vocab-size
               :hidden-dim hidden-dim
               :q-dim q-dim
               :kv-dim kv-dim
               :intermediate-dim intermediate-dim
               :num-layers num-layers
               :num-heads num-heads
               :num-kv-heads num-kv-heads
               :head-dim head-dim
               :max-seq-len max-seq-len
               :kv-cache-shape kv-cache-shape
               :weight-dtype weight-dtype
               :weight-enum weight-enum}})))

(defn allocate-device-weights
  "Transfers all model layer and embedding weights to PJRT device memory."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim num-layers weight-dtype weight-enum]} config
        load-fn (fn [name shape] (load-weight-buffer ctx weights-mmap name shape weight-dtype weight-enum))
        embed-buf (load-fn "model.embed_tokens.weight" [vocab-size hidden-dim])
        final-norm-buf (load-fn "model.norm.weight" [hidden-dim])
        layer-bufs (mapv (fn [i]
                           (let [kmap (gemma/weight-key-map i)]
                             [(load-fn (:input-ln-w kmap) [hidden-dim])
                              (load-fn (:q-w kmap) [q-dim hidden-dim])
                              (load-fn (:k-w kmap) [kv-dim hidden-dim])
                              (load-fn (:v-w kmap) [kv-dim hidden-dim])
                              (load-fn (:o-w kmap) [hidden-dim q-dim])
                              (load-fn (:post-attn-ln-w kmap) [hidden-dim])
                              (load-fn (:pre-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:post-mlp-ln-w kmap) [hidden-dim])
                              (load-fn (:gate-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:up-w kmap) [intermediate-dim hidden-dim])
                              (load-fn (:down-w kmap) [hidden-dim intermediate-dim])]))
                         (range num-layers))
        flat-layer-bufs (vec (apply concat layer-bufs))]
    (into [embed-buf final-norm-buf] flat-layer-bufs)))

(defn- prepare-input-tensor
  "Pads token IDs sequence to `max-len` with zero padding."
  [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))

(defn compile-executable
  "Compiles full Gemma 2 model into a PJRT executable via Tensor Logic AST."
  [{:keys [ctx config]} max-seq-len]
  (let [{:keys [vocab-size hidden-dim q-dim kv-dim intermediate-dim num-layers weight-dtype]} config
        invars (vec (concat
                     [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] weight-dtype]]
                      [:final_norm_w [:tensor [hidden-dim] weight-dtype]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] weight-dtype]]
                                [(keyword (str "gate_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                [(keyword (str "up_w_" i)) [:tensor [intermediate-dim hidden-dim] weight-dtype]]
                                [(keyword (str "down_w_" i)) [:tensor [hidden-dim intermediate-dim] weight-dtype]]])
                             (range num-layers))))
        ast (gemma/gemma2-model-ast (assoc config :max-seq-len max-seq-len))
        graph (lower/ast->graph "full_gemma2_model" invars ast #{:logits})]
    (xla/compile-graph ctx graph)))

(defn generate-text
  "Top-level REPL/programmatic helper: runs full end-to-end text generation on an initialized session."
  ([session] (generate-text session (or (:prompt (:opts session)) "The capital of France is")))
  ([session prompt-text]
   (let [{:keys [ctx tokenizer config opts]} session
         {:keys [max-new-tokens temperature top-k]} opts
         {:keys [vocab-size weight-dtype]} config
         prompt-ids (into [(bos-id tokenizer)] (encode tokenizer prompt-text))
         prompt-len (count prompt-ids)
         max-seq-len (max 32 (+ prompt-len max-new-tokens 4))
         device-weights (allocate-device-weights session)
         exec (compile-executable session max-seq-len)
         eos (eos-id tokenizer)
         cur-tokens (atom (vec prompt-ids))]
     (println (format "Prompt: \"%s\"" prompt-text))
     (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s"
                      max-new-tokens temperature top-k (name weight-dtype)))
     (println (format "Encoded Token IDs (%d tokens): %s" prompt-len prompt-ids))
     (println "\nGenerating tokens autoregressively...")
     (print prompt-text)
     (flush)
     (dotimes [_ max-new-tokens]
       (let [s-len (count @cur-tokens)
             in-arr (prepare-input-tensor @cur-tokens max-seq-len)
             in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
             in-args (into [in-b] device-weights)
             out (xla/execute exec in-args)
             out-b (if (vector? out) (first out) out)
             l (xla/to-host-slice out-b (dec s-len) vocab-size (* max-seq-len vocab-size) weight-dtype)
             next-id (sampling/sample-logits l {:temperature temperature :top-k top-k})]
         (swap! cur-tokens conj next-id)
         (print (decode tokenizer [next-id]))
         (flush)
         (when (= next-id eos)
           (println "\nReached EOS token.")
           (reduced nil))))
     (println "\n\n==================================================================")
     (println "Final Generated Sequence:")
     (println (decode tokenizer @cur-tokens))
     (println "==================================================================")
     @cur-tokens)))

(defn -main
  "CLI entrypoint."
  [& args]
  (let [opts (parse-cli-args args)]
    (println "==================================================================")
    (println "  clj-xla Gemma 2B Single-Pass Prefill & BF16 KV-Cached Generation ")
    (println "==================================================================")
    (let [session (init-inference-session opts)]
      (generate-text session (:prompt opts)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
