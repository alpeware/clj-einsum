(ns einsum.models.gemma4.config
  "Gemma 4 model configuration parsing, dimension resolution, and discovery."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.models.gemma4 :as gemma4-logic]))

(def DEFAULT_MODEL_DIRS
  [".models/gemma-4-E2B-it"
   ".models/gemma-4-2b-it"
   ".models/gemma-4-E4B-it"
   ".models/gemma-4-4b-it"
   ".models/gemma-4-12B-it"
   ".models/gemma-4-12b-it"
   ".models/gemma-4-31B-it"
   ".models/gemma-4-31b-it"
   ".models/gemma4-2b"
   ".models/gemma-2b"
   ".models/gemma4"])

(defn find-model-dir
  "Finds the first existing model directory from candidate paths."
  [dirs]
  (or (first (filter #(.exists (io/file %)) dirs))
      (first dirs)))

(defn load-model-config
  "Loads JSON configuration map from model directory `config.json`."
  [model-dir]
  (let [f (io/file model-dir "config.json")]
    (if (.exists f)
      (json/read-str (slurp f) :key-fn keyword)
      {})))

(defn resolve-weight-shape
  "Resolves the logical [out-features in-features] shape of a tensor from header,
   supporting unquantized safetensors, EXL3 trellis tensors (.svh/.suh), and v_proj fallback to k_proj."
  [header tensor-name default-shape]
  (or (get-in header [tensor-name "shape"])
      (let [svh-name (str/replace tensor-name #"\.weight$" ".svh")
            suh-name (str/replace tensor-name #"\.weight$" ".suh")]
        (when (contains? header svh-name)
          [(first (get-in header [svh-name "shape"]))
           (first (get-in header [suh-name "shape"]))]))
      (let [k-tensor (str/replace tensor-name #"\.v_proj\." ".k_proj.")
            svh-name (str/replace k-tensor #"\.weight$" ".svh")
            suh-name (str/replace k-tensor #"\.weight$" ".suh")]
        (when (contains? header svh-name)
          [(first (get-in header [svh-name "shape"]))
           (first (get-in header [suh-name "shape"]))]))
      default-shape))

(defn max-safe-prefill-seq-len
  "Returns the maximum sequence length for 1-shot parallel prefill that safely fits in VRAM
   without exceeding device workspace headroom."
  [config]
  (let [use-w4a16? (and (:is-int4 config)
                        (if (some? (:use-w4a16-gemv config))
                          (boolean (:use-w4a16-gemv config))
                          (or (= (:backend config) :rocm) (= (:target config) :rocm))))]
    (cond
      use-w4a16?
      0

      (or (:is-int4 config)
          (:is-ternary config)
          (re-find #"31[bB]" (or (:model-dir config) ""))
          (>= (long (or (:num-layers config) 0)) 60))
      2048

      :else
      8192)))

(defn build-model-config
  "Extracts all dimension and layer configurations for Gemma 4 from safetensors header and JSON config."
  [weights-mmap resolved-model-dir opts]
  (let [header (or (:header weights-mmap) {})
        prefix-base (if (contains? header "model.language_model.embed_tokens.weight")
                      "model.language_model."
                      "model.")
        model-cfg (load-model-config resolved-model-dir)
        text-cfg (or (:text_config model-cfg) model-cfg)
        layer-types-cfg (:layer_types text-cfg)
        num-kv-shared-layers (or (:num_kv_shared_layers text-cfg) 0)

        emb-shape (resolve-weight-shape header (str prefix-base "embed_tokens.weight") [262144 1536])
        emb-pl-shape (resolve-weight-shape header (str prefix-base "embed_tokens_per_layer.weight") [262144 0])
        q0-shape (resolve-weight-shape header (str prefix-base "layers.0.self_attn.q_proj.weight") [2048 1536])
        k0-shape (resolve-weight-shape header (str prefix-base "layers.0.self_attn.k_proj.weight") [256 1536])

        vocab-size (nth emb-shape 0 262144)
        hidden-dim (nth emb-shape 1 1536)
        total-pl-dim (nth emb-pl-shape 1 0)

        layer-pattern (re-pattern (str "^" (java.util.regex.Pattern/quote prefix-base) "layers\\.\\d+\\.input_layernorm\\.weight$"))
        detected-layers (count (filter #(re-find layer-pattern %) (keys header)))
        num-layers (if (pos? detected-layers) detected-layers (or (:num_hidden_layers text-cfg) 35))
        pl-dim (if (pos? num-layers) (quot total-pl-dim num-layers) 0)

        num-heads (or (:num_attention_heads text-cfg) (quot (nth q0-shape 0 2048) 256))
        num-kv-heads (or (:num_key_value_heads text-cfg) (quot (nth k0-shape 0 256) 256))
        head-dim (or (:head_dim text-cfg) 256)
        max-seq-len (long (or (:max-seq-len opts) 16384))
        model-sliding-window (long (or (:sliding_window text-cfg) (:sliding-window text-cfg) 512))

        layer-configs (mapv (fn [i]
                              (let [kmap (gemma4-logic/gemma4-weight-key-map i (str prefix-base "layers."))
                                    l-q-dim (first (resolve-weight-shape header (:q-w kmap) [2048 hidden-dim]))
                                    l-kv-dim (first (resolve-weight-shape header (:k-w kmap) [256 hidden-dim]))
                                    l-head-dim (first (get-in header [(:q-norm-w kmap) "shape"] [head-dim]))
                                    mlp-dim (first (resolve-weight-shape header (:gate-w kmap) [(* 4 hidden-dim) hidden-dim]))
                                    l-type-str (or (get layer-types-cfg i)
                                                   (if (= l-head-dim 512) "full_attention" "sliding_attention"))
                                    is-global? (= l-type-str "full_attention")
                                    rope-prop (if is-global? 0.25 1.0)
                                    theta-base (if is-global? 1000000.0 10000.0)
                                    l-nkv (quot l-kv-dim l-head-dim)]
                                {:idx i
                                 :q-dim l-q-dim
                                 :kv-dim l-kv-dim
                                 :head-dim l-head-dim
                                 :num-heads (quot l-q-dim l-head-dim)
                                 :num-kv-heads l-nkv
                                 :mlp-dim mlp-dim
                                 :is-global? is-global?
                                 :layer-type (if is-global? :full_attention :sliding_attention)
                                 :rope-proportion rope-prop
                                 :theta-base theta-base
                                 :sliding-window (if is-global? nil model-sliding-window)}))
                            (range num-layers))

        precision (:precision opts)
        quant-metadata (get-in weights-mmap [:header "__metadata__" "quantization"])
        is-ternary (boolean (or (= precision :ternary)
                                (= quant-metadata "ternary")
                                (:is-ternary opts)))
        is-int4 (boolean (and (not is-ternary)
                              (or (= precision :int4)
                                  (= quant-metadata "int4")
                                  (and (or (nil? precision) (= precision :auto))
                                       (or (re-find #"31[bB]" (or resolved-model-dir ""))
                                           (re-find #"31[bB]" (or (:model-dir opts) "")))))))
        is-int8 (boolean (and (not is-ternary)
                              (or (= precision :int8)
                                  (= quant-metadata "int8"))))
        group-size (or (:group-size opts)
                       (when-let [gs (or (get-in weights-mmap [:header "__metadata__" "group_size"])
                                         (get-in weights-mmap [:header "__metadata__" "group-size"]))]
                         (let [parsed (if (string? gs) (Long/parseLong gs) (long gs))]
                           (when (pos? parsed) parsed)))
                       (when-let [scale-sh (some (fn [[k v]]
                                                   (when (str/ends-with? k ".scales")
                                                     (get v "shape")))
                                                 header)]
                         (if (= 1 (count scale-sh))
                           nil
                           (when (= 2 (count scale-sh))
                             128)))
                       (when is-int4 128))
        weight-dtype (cond
                       is-ternary :ternary
                       is-int4 :int4
                       is-int8 :int8
                       (and precision (not= precision :auto)) precision
                       quant-metadata (keyword quant-metadata)
                       :else :bf16)
        weight-enum (cond (or is-ternary is-int8 is-int4) 2 (= weight-dtype :f32) 11 :else 13)
        norm-enum (if (= weight-dtype :f32) 11 13)
        detected-skip-layers (when (or is-ternary is-int4 is-int8)
                               (set (filter (fn [i]
                                              (not (contains? header (str prefix-base "layers." i ".self_attn.q_proj.weight.scales"))))
                                            (range num-layers))))
        skip-layers (into (set (or (:skip-layers opts) #{})) (or detected-skip-layers #{}))]

    {:model-dir resolved-model-dir
     :prefix-base prefix-base
     :vocab-size vocab-size
     :hidden-dim hidden-dim
     :total-pl-dim total-pl-dim
     :pl-dim pl-dim
     :num-layers num-layers
     :num-heads num-heads
     :num-kv-heads num-kv-heads
     :head-dim head-dim
     :max-seq-len max-seq-len
     :layer-types layer-types-cfg
     :layer-configs layer-configs
     :sliding-window model-sliding-window
     :num-kv-shared-layers num-kv-shared-layers
     :weight-dtype weight-dtype
     :weight-enum weight-enum
     :is-ternary is-ternary
     :is-int8 is-int8
     :is-int4 is-int4
     :skip-layers skip-layers
     :group-size group-size
     :norm-enum norm-enum
     :backend (or (:backend opts) (:target opts))
     :target (or (:target opts) (:backend opts))
     :use-w4a16-gemv (:use-w4a16-gemv opts)}))
