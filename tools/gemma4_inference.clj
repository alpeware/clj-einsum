(ns tools.gemma4-inference
  "Top-level runnable integration script and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.logic.exl3 :as exl3]
            [einsum.logic.lower :as lower]
            [einsum.models.gemma :as gemma-logic]
            [einsum.quant.ternary :as ternary]
            [einsum.runtime.profile :as profile]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.sampling :as sampling]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode eos-id]])
  (:import [java.lang.foreign Arena]))

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 20
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :method :kv-cache
   :compare false
   :verbose false
   :quiet false
   :thinking false
   :profile true
   :profile-out "scratch/gemma4_profile.edn"
   :chrome-trace-out "scratch/gemma4_chrome_trace.json"})

(def DEFAULT_MODEL_DIRS
  [".models/gemma-4-31b-it-int4"
   ".models/gemma-4-31B-it-int4"
   ".models/gemma-4-31B-it-exl3"
   ".models/gemma-4-31b-it-exl3"
   ".models/gemma-4-12B-it-exl3"
   ".models/gemma-4-E2B-it"
   ".models/gemma-4-E2B"
   ".models/gemma-4-E4B-it"
   ".models/gemma-4-E4B"
   ".models/gemma-4-12B-it"
   ".models/gemma-4-12B"
   ".models/gemma-4-26B-A4B-it"
   ".models/gemma-4-26B-A4B"
   ".models/gemma-4-31B-it"
   ".models/gemma-4-31B"
   ".models/gemma-4"])

(defn- normalize-args
  [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args
  "Parses command-line flags (--prompt, --model/--model-dir, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose, --quiet)."
  [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT_CLI_OPTS]
    (if (empty? remaining)
      opts
      (let [flag (first remaining)
            val (second remaining)]
        (cond
          (and (= flag "--prompt") val)
          (recur (subvec remaining 2) (assoc opts :prompt val))

          (and (= flag "--prompt-file") val)
          (recur (subvec remaining 2) (assoc opts :prompt (slurp val)))

          (and (or (= flag "--model-dir") (= flag "--model") (= flag "--model-name") (= flag "-m")) val)
          (let [dir (if (str/starts-with? val ".models/") val (str ".models/" (last (str/split val #"/"))))]
            (recur (subvec remaining 2) (assoc opts :model-dir dir)))

          (and (= flag "--max-new-tokens") val)
          (recur (subvec remaining 2) (assoc opts :max-new-tokens (Long/parseLong val)))

          (and (or (= flag "--temperature") (= flag "--temp")) val)
          (recur (subvec remaining 2) (assoc opts :temperature (Double/parseDouble val)))

          (and (= flag "--top-k") val)
          (recur (subvec remaining 2) (assoc opts :top-k (Long/parseLong val)))

          (and (= flag "--backend") val)
          (let [clean-kw (keyword (str/replace val #"^:+" ""))]
            (when (= clean-kw :rocm)
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
            (recur (subvec remaining 2) (assoc opts :backend clean-kw)))

          (and (= flag "--precision") val)
          (recur (subvec remaining 2) (assoc opts :precision (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--out") val)
          (recur (subvec remaining 2) (assoc opts :out val))

          (and (= flag "--method") val)
          (recur (subvec remaining 2) (assoc opts :method (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--max-seq-len") val)
          (recur (subvec remaining 2) (assoc opts :max-seq-len (Long/parseLong val)))

          (and (= flag "--mode") val)
          (recur (subvec remaining 2) (assoc opts :mode (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--compare") val)
          (recur (subvec remaining 2) (assoc opts :compare (Boolean/parseBoolean val)))

          (and (or (= flag "--group-size") (= flag "-g")) val)
          (recur (subvec remaining 2) (assoc opts :group-size (Long/parseLong val)))

          (= flag "--verbose")
          (recur (subvec remaining 1) (assoc opts :verbose true))

          (or (= flag "--thinking") (= flag "--think"))
          (recur (subvec remaining 1) (assoc opts :thinking true))

          (= flag "--ternary")
          (recur (subvec remaining 1) (assoc opts :precision :ternary :is-ternary true))

          (= flag "--quiet")
          (do (System/setProperty "clj-xla.quiet" "true")
              (recur (subvec remaining 1) (assoc opts :quiet true)))

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

(defn load-model-config
  "Loads model configuration map from `config.json` if available."
  [model-dir]
  (let [cfg-file (io/file model-dir "config.json")]
    (when (.exists cfg-file)
      (try
        (json/read-str (slurp cfg-file) :key-fn keyword)
        (catch Exception _ nil)))))

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

(defn load-weight-buffer
  "Loads a single weight tensor from `weights-mmap` into PJRT device memory in specified precision."
  ([ctx weights-mmap tensor-name shape weight-dtype weight-enum]
   (load-weight-buffer ctx weights-mmap tensor-name shape weight-dtype weight-enum 0.0))
  ([ctx weights-mmap tensor-name shape _weight-dtype weight-enum default-val]
   (let [header (or (:header weights-mmap) {})
         trellis-name (when (str/ends-with? tensor-name ".weight")
                        (str/replace tensor-name #"\.weight$" ".trellis"))
         actual-trellis-name (when trellis-name
                               (if (contains? header trellis-name)
                                 trellis-name
                                 (let [k-trellis (str/replace trellis-name #"\.v_proj\." ".k_proj.")]
                                   (if (contains? header k-trellis) k-trellis trellis-name))))
         actual-tensor-name (if (or (contains? header tensor-name) (contains? (:tensors weights-mmap) tensor-name))
                              tensor-name
                              (let [k-name (str/replace tensor-name #"\.v_proj\." ".k_proj.")]
                                (if (or (contains? header k-name) (contains? (:tensors weights-mmap) k-name))
                                  k-name
                                  tensor-name)))]
     (cond
       ;; EXL3 trellis quantized tensor
       (and actual-trellis-name (contains? header actual-trellis-name))
       (let [base-name (str/replace actual-trellis-name #"\.trellis$" "")
             suh-name (str base-name ".suh")
             svh-name (str base-name ".svh")
             trellis-slice (st/get-tensor-slice weights-mmap actual-trellis-name)
             suh-slice (st/get-tensor-slice weights-mmap suh-name)
             svh-slice (st/get-tensor-slice weights-mmap svh-name)
             in-features (first (get-in header [suh-name "shape"]))
             out-features (first (get-in header [svh-name "shape"]))
             trellis-shape (get-in header [actual-trellis-name "shape"])
             words-per-tile (last trellis-shape)
             bits (quot words-per-tile 16)
             target-format (if (= weight-enum 11) :f32 :bf16)
             dequant-arr (exl3/dequant-exl3-matrix trellis-slice in-features out-features bits suh-slice svh-slice
                                                   {:as target-format :transpose? true})]
         (xla/buffer-from-host-buffer ctx (:client ctx) dequant-arr shape weight-enum))

       (or (zero? (reduce * 1 shape))
           (not (or (contains? header actual-tensor-name)
                    (contains? (:tensors weights-mmap) actual-tensor-name))))
       (let [num-elements (reduce * 1 shape)
             default-f (float default-val)
             data (if (= weight-enum 11)
                    (let [arr (float-array num-elements)]
                      (java.util.Arrays/fill arr default-f)
                      arr)
                    (let [arr (short-array num-elements)
                          bf-bits (short (bit-shift-right (Float/floatToRawIntBits default-f) 16))]
                      (java.util.Arrays/fill arr bf-bits)
                      arr))]
         (xla/buffer-from-host-buffer ctx (:client ctx) data shape weight-enum))

       :else
       (let [slice (st/get-tensor-slice weights-mmap actual-tensor-name)]
         (xla/buffer-from-host-buffer ctx (:client ctx) slice shape weight-enum))))))

(defn load-linear-projection-buffers
  "Loads a linear projection matrix as either a single unquantized PJRT buffer,
   or if is-int8/is-int4/is-ternary is true, a pair [w-buf scale-buf] with in-graph quantization."
  ([ctx weights-mmap tensor-name shape is-int8 norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 false false norm-enum weight-enum 128))
  ([ctx weights-mmap tensor-name shape is-int8 is-int4 norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 is-int4 false norm-enum weight-enum 128))
  ([ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum 128))
  ([ctx weights-mmap tensor-name [rows cols :as shape] is-int8 is-int4 is-ternary norm-enum weight-enum group-size]
   (if-not (or is-int8 is-int4 is-ternary)
     [(load-weight-buffer ctx weights-mmap tensor-name shape (if (= weight-enum 11) :f32 :bf16) weight-enum 0.0)]
     (let [header (or (:header weights-mmap) {})
           actual-tensor-name (if (or (contains? header tensor-name) (contains? (:tensors weights-mmap) tensor-name))
                                tensor-name
                                (let [k-name (str/replace tensor-name #"\.v_proj\." ".k_proj.")]
                                  (if (or (contains? header k-name) (contains? (:tensors weights-mmap) k-name))
                                    k-name
                                    tensor-name)))
           scale-name (str actual-tensor-name ".scales")
           prequantized? (and (or (contains? header scale-name) (contains? (:tensors weights-mmap) scale-name))
                              (or (contains? header actual-tensor-name) (contains? (:tensors weights-mmap) actual-tensor-name)))]
       (if prequantized?
         (let [w-shape (cond is-ternary [rows (quot cols 4)] is-int4 [rows (quot cols 2)] :else [rows cols])
               w-slice (st/get-tensor-slice weights-mmap actual-tensor-name)
               scale-slice (st/get-tensor-slice weights-mmap scale-name)
               scale-shape (or (get-in header [scale-name "shape"])
                               (get-in (:tensors weights-mmap) [scale-name :info "shape"])
                               (if (and (or is-int4 is-ternary) group-size (zero? (mod cols group-size)))
                                 [rows (quot cols group-size)]
                                 [rows]))
               w-buf (xla/buffer-from-host-buffer ctx (:client ctx) w-slice w-shape 2)
               scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scale-slice scale-shape norm-enum)]
           [w-buf scale-buf])
         (let [trellis-name (when (str/ends-with? tensor-name ".weight")
                              (str/replace tensor-name #"\.weight$" ".trellis"))
               actual-trellis-name (when trellis-name
                                     (if (contains? header trellis-name)
                                       trellis-name
                                       (let [k-trellis (str/replace trellis-name #"\.v_proj\." ".k_proj.")]
                                         (if (contains? header k-trellis) k-trellis trellis-name))))
               scale-format (if (= norm-enum 11) :f32 :bf16)
               raw-arr (cond
                         (and actual-trellis-name (contains? header actual-trellis-name))
                         (let [base-name (str/replace actual-trellis-name #"\.trellis$" "")
                               suh-name (str base-name ".suh")
                               svh-name (str base-name ".svh")
                               trellis-slice (st/get-tensor-slice weights-mmap actual-trellis-name)
                               suh-slice (st/get-tensor-slice weights-mmap suh-name)
                               svh-slice (st/get-tensor-slice weights-mmap svh-name)
                               in-features (first (get-in header [suh-name "shape"]))
                               out-features (first (get-in header [svh-name "shape"]))
                               trellis-shape (get-in header [actual-trellis-name "shape"])
                               words-per-tile (last trellis-shape)
                               bits (quot words-per-tile 16)]
                           (exl3/dequant-exl3-matrix trellis-slice in-features out-features bits suh-slice svh-slice
                                                     {:as :bf16 :transpose? true}))

                         (or (zero? (reduce * 1 shape))
                             (not (or (contains? header actual-tensor-name)
                                      (contains? (:tensors weights-mmap) actual-tensor-name))))
                         (short-array (* rows cols))

                         :else
                         (st/get-tensor-floats weights-mmap actual-tensor-name))]
           (cond
             is-ternary
             (let [{:keys [data scales scale-shape]} (ternary/quantize-weights-per-row-ternary raw-arr rows cols
                                                                                               {:as scale-format
                                                                                                :group-size group-size})
                   w-buf (xla/buffer-from-host-buffer ctx (:client ctx) data [rows (quot cols 4)] 2)
                   scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scales (or scale-shape [rows]) norm-enum)]
               [w-buf scale-buf])

             is-int4
             (let [{:keys [data scales scale-shape]} (exl3/quantize-weights-per-row-int4 raw-arr rows cols
                                                                                         {:as scale-format
                                                                                          :group-size group-size})
                   w-buf (xla/buffer-from-host-buffer ctx (:client ctx) data [rows (quot cols 2)] 2)
                   scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scales (or scale-shape [rows]) norm-enum)]
               [w-buf scale-buf])

             :else
             (let [{:keys [data scales]} (exl3/quantize-weights-per-row-int8 raw-arr rows cols {:as scale-format})
                   w-buf (xla/buffer-from-host-buffer ctx (:client ctx) data shape 2)
                   scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scales [rows] norm-enum)]
               [w-buf scale-buf]))))))))

(defn quantize-bf16-to-int8
  "Quantizes a BF16 short-array to INT8 byte-array with per-tensor symmetric quantization.
   Returns {:data byte-array :scale float}."
  [^shorts bf16-shorts]
  (let [n (alength bf16-shorts)
        ;; Find max absolute value in BF16
        max-abs (loop [i 0 m (float 0.0)]
                  (if (>= i n)
                    m
                    (let [s (int (aget bf16-shorts i))
                          bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))
                          f (Math/abs (Float/intBitsToFloat bits))]
                      (recur (inc i) (max m f)))))
        scale (if (zero? max-abs) 1.0 (/ (double max-abs) 127.0))
        inv-scale (float (/ 1.0 scale))
        result (byte-array n)]
    (dotimes [i n]
      (let [s (int (aget bf16-shorts i))
            bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))
            f (Float/intBitsToFloat bits)
            q (Math/round (* f inv-scale))
            clamped (max -127 (min 127 q))]
        (aset result i (byte clamped))))
    {:data result :scale (float scale)}))

(defn init-inference-session
  "Initializes PJRT runtime, loads safetensors weights, and prepares model configuration for Gemma 4 REPL/CLI sessions."
  ([opts]
   (let [opts (merge DEFAULT_CLI_OPTS opts)
         {:keys [backend precision]} opts
         model-dir (or (:model-dir opts) (:model opts))
         ctx (xla/init-backend! (or backend :cpu))
         dirs (if model-dir (cons model-dir DEFAULT_MODEL_DIRS) DEFAULT_MODEL_DIRS)
         resolved-model-dir (find-model-dir dirs)
         arena (Arena/ofConfined)
         weights-mmap (st/map-safetensors-weights resolved-model-dir arena)
         tokenizer (tok/from-file resolved-model-dir)
         header (or (:header weights-mmap) {})
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
         num-layers (count (filter #(re-find layer-pattern %) (keys header)))
         num-layers (if (pos? num-layers) num-layers (or (:num_hidden_layers text-cfg) 35))
         pl-dim (if (pos? num-layers) (quot total-pl-dim num-layers) 0)

         num-heads (or (:num_attention_heads text-cfg) (quot (nth q0-shape 0 2048) 256))
         num-kv-heads (or (:num_key_value_heads text-cfg) (quot (nth k0-shape 0 256) 256))
         head-dim (or (:head_dim text-cfg) 256)
         max-seq-len (long (or (:max-seq-len opts) 16384))
         model-sliding-window (long (or (:sliding_window text-cfg) (:sliding-window text-cfg) 512))

         layer-configs (mapv (fn [i]
                               (let [kmap (gemma-logic/gemma4-weight-key-map i (str prefix-base "layers."))
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

         quant-metadata (get-in weights-mmap [:header "__metadata__" "quantization"])
         is-ternary (boolean (or (= precision :ternary)
                                 (= quant-metadata "ternary")
                                 (:is-ternary opts)))
         is-int4 (boolean (and (not is-ternary)
                               (or (= precision :int4)
                                   (= quant-metadata "int4")
                                   (and (or (nil? precision) (= precision :auto))
                                        (or (re-find #"31[bB]" (or resolved-model-dir ""))
                                            (re-find #"31[bB]" (or model-dir "")))))))
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
         norm-enum (if (= weight-dtype :f32) 11 13)]

     (when-not (:quiet opts)
       (println (str "Loaded Gemma 4 model weights from [" resolved-model-dir "] in [" (name weight-dtype) "] precision (" num-layers " layers, " num-heads " heads, " num-kv-heads " kv-heads).")))

     {:ctx ctx
      :opts opts
      :model-dir resolved-model-dir
      :tokenizer tokenizer
      :weights-mmap weights-mmap
      :arena arena
      :config {:model-dir resolved-model-dir
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
               :group-size group-size
               :norm-enum norm-enum}})))

(defn build-tensor-logic-invars
  "Constructs EDN SSA signature invars for full Gemma 4 model forward pass."
  [config max-seq-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 is-int4 is-ternary layer-configs last-token-only? group-size]} config
        is-ternary (boolean (or is-ternary (= weight-dtype :ternary) (= (:quant-type config) :ternary)))
        norm-dtype (if (or is-int8 is-int4 is-ternary) :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        scale-shape-fn (fn [rows cols]
                         (cond
                           (and is-int4 group-size (zero? (mod cols group-size)))
                           [rows (quot cols group-size)]
                           (and is-ternary group-size (zero? (mod cols group-size)))
                           [rows (quot cols group-size)]
                           :else
                           [rows]))]
    (vec (concat [[:x [:tensor [1 max-seq-len] :i32]]]
                 (when last-token-only?
                   [[:pos [:tensor [1] :i32]]])
                 [[:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                 (when has-ple?
                   [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] norm-dtype]]
                    [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] norm-dtype]]
                    [:per_layer_projection_norm [:tensor [pl-dim] norm-dtype]]])
                 (mapcat (fn [i]
                           (let [cfg (nth layer-configs i)
                                 q-dim (:q-dim cfg)
                                 kv-dim (:kv-dim cfg)
                                 head-dim (:head-dim cfg)
                                 mlp-dim (:mlp-dim cfg)]
                             (concat
                              [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "layer_scalar_" i)) [:tensor [1] norm-dtype]]]
                              (cond
                                is-ternary
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 4)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                is-int4
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 2)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                is-int8
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor [q-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] weight-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] weight-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] weight-dtype]]])
                              [[(keyword (str "q_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]]
                              (cond
                                is-ternary
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 4)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                is-int4
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 2)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                is-int8
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] weight-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] weight-dtype]]])
                              (when has-ple?
                                [[(keyword (str "per_layer_gate_w_" i)) [:tensor [pl-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "per_layer_proj_w_" i)) [:tensor [hidden-dim pl-dim] norm-dtype]]
                                 [(keyword (str "post_per_layer_norm_w_" i)) [:tensor [hidden-dim] norm-dtype]]]))))
                         (range num-layers))
                 [[:final_norm_w [:tensor [hidden-dim] norm-dtype]]]
                 (when (or (:relational-memory config) (:relational-memory? config))
                   (let [rel-cfg (or (:relational-memory config) {})
                         mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
                         entity-count (long (or (:entity-count rel-cfg) 1000))]
                     [[:w_mem_proj [:tensor [hidden-dim mem-dim] norm-dtype]]
                      [:r_active [:tensor [mem-dim mem-dim] norm-dtype]]
                      [:entity_table [:tensor [entity-count mem-dim] norm-dtype]]
                      [:threshold_const [:tensor [1 1 entity-count] norm-dtype]]
                      [:w_entity_to_vocab [:tensor [entity-count vocab-size] norm-dtype]]]))))))

(defn allocate-device-weights
  "Loads individual weight tensors for Gemma 4 into PJRT device buffers matching build-tensor-logic-invars."
  [{:keys [ctx weights-mmap config]}]
  (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim weight-dtype weight-enum norm-enum layer-configs num-layers is-int8 is-int4 is-ternary group-size]} config
        has-ple? (pos? total-pl-dim)
        load-fn (fn
                  ([name shape enum] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum 0.0))
                  ([name shape enum default-val] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum default-val)))
        load-linear-fn (fn [name shape]
                         (load-linear-projection-buffers ctx weights-mmap name shape is-int8 is-int4 is-ternary norm-enum weight-enum group-size))
        embed-buf (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim] norm-enum)
        ple-bufs (when has-ple?
                   [(load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim] norm-enum)
                    (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim] norm-enum)
                    (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim] norm-enum)])
        layer-bufs (mapcat (fn [i]
                             (let [kmap (gemma-logic/gemma4-weight-key-map i (str prefix-base "layers."))
                                   cfg (nth layer-configs i)
                                   q-dim (:q-dim cfg)
                                   kv-dim (:kv-dim cfg)
                                   head-dim (:head-dim cfg)
                                   mlp-dim (:mlp-dim cfg)]
                               (concat
                                [(load-fn (:input-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:layer-scalar-w kmap) [1] norm-enum 1.0)]
                                (load-linear-fn (:q-w kmap) [q-dim hidden-dim])
                                (load-linear-fn (:k-w kmap) [kv-dim hidden-dim])
                                (load-linear-fn (:v-w kmap) [kv-dim hidden-dim])
                                (load-linear-fn (:o-w kmap) [hidden-dim q-dim])
                                [(load-fn (:q-norm-w kmap) [head-dim] norm-enum 0.0)
                                 (load-fn (:k-norm-w kmap) [head-dim] norm-enum 0.0)
                                 (load-fn (:post-attn-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:pre-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0)
                                 (load-fn (:post-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0)]
                                (load-linear-fn (:gate-w kmap) [mlp-dim hidden-dim])
                                (load-linear-fn (:up-w kmap) [mlp-dim hidden-dim])
                                (load-linear-fn (:down-w kmap) [hidden-dim mlp-dim])
                                (when has-ple?
                                  [(load-fn (:per-layer-gate-w kmap) [pl-dim hidden-dim] norm-enum 0.0)
                                   (load-fn (:per-layer-proj-w kmap) [hidden-dim pl-dim] norm-enum 0.0)
                                   (load-fn (:post-per-layer-norm-w kmap) [hidden-dim] norm-enum 0.0)]))))
                           (range num-layers))
        final-norm-buf (load-fn (str prefix-base "norm.weight") [hidden-dim] norm-enum 0.0)]
    (vec (concat [embed-buf] ple-bufs layer-bufs [final-norm-buf]))))

(def allocate-tensor-logic-weights allocate-device-weights)

(defn floats->bf16-shorts
  "Converts float array to short array of bfloat16 bit-patterns."
  ^shorts [^floats fa]
  (let [n (alength fa)
        sa (short-array n)]
    (dotimes [i n]
      (aset-short sa i (exl3/float->bf16-short (aget fa i))))
    sa))

(defn allocate-relational-buffers
  "Allocates device PJRT buffers for relational memory tensors:
   w_mem_proj, r_active, entity_table, threshold_const, w_entity_to_vocab."
  [{:keys [ctx config]} mem & [{:keys [w-mem-proj w-entity-to-vocab threshold rel-id]}]]
  (let [{:keys [norm-enum is-int8 is-int4 is-ternary weight-dtype]} config
        norm-dt (if (or is-int8 is-int4 is-ternary) :bf16 weight-dtype)
        is-bf16? (= norm-dt :bf16)
        to-dev-buf (fn [data shape]
                     (if is-bf16?
                       (let [sa (if (instance? (Class/forName "[S") data)
                                  ^shorts data
                                  (floats->bf16-shorts ^floats data))]
                         (pjrt/buffer-from-host-buffer ctx (:client ctx) sa shape 13))
                       (pjrt/buffer-from-host-buffer ctx (:client ctx) data shape (or norm-enum 11))))
        rel-cfg (or (:relational-memory config) {})
        hidden-dim (long (or (:hidden-dim config) 1536))
        mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
        entity-count (long (or (:entity-count rel-cfg) 1000))
        vocab-size (long (or (:vocab-size config) 262144))
        rel-idx (long (or rel-id 0))
        w-mem-proj (or w-mem-proj (float-array (* hidden-dim mem-dim)))
        cores ^floats (:cores mem)
        r-active (let [arr (float-array (* mem-dim mem-dim))]
                   (when cores
                     (System/arraycopy cores (int (* rel-idx mem-dim mem-dim)) arr 0 (int (* mem-dim mem-dim))))
                   arr)
        e-table (or (:entity-table mem) (float-array (* entity-count mem-dim)))
        thresh-val (float (or threshold (:threshold rel-cfg) 0.5))
        threshold-arr (let [arr (float-array entity-count)]
                        (java.util.Arrays/fill arr thresh-val)
                        arr)
        w-vocab (or w-entity-to-vocab (float-array (* entity-count vocab-size)))]
    [(to-dev-buf w-mem-proj [hidden-dim mem-dim])
     (to-dev-buf r-active [mem-dim mem-dim])
     (to-dev-buf e-table [entity-count mem-dim])
     (to-dev-buf threshold-arr [1 1 entity-count])
     (to-dev-buf w-vocab [entity-count vocab-size])]))

(defn destroy-relational-buffers!
  "Releases device PJRT buffers for relational memory."
  [ctx buffers]
  (doseq [b buffers]
    (when b (xla/destroy-buffer! ctx b))))

(defn compile-tensor-logic-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [last-token? (get opts :last-token-only? true)
        config-with-len (assoc config :max-seq-len max-seq-len :last-token-only? last-token?)
        invars (build-tensor-logic-invars config-with-len max-seq-len)
        ast (gemma-logic/gemma4-model-ast config-with-len)
        target-heads (or (:targets opts) #{:logits})
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 AST (%d layers, max-seq-len=%d, last-token-only=%s) to StableHLO..."
                             (:num-layers config) max-seq-len (str last-token?))))
        graph (lower/ast->graph "gemma4_tensor_logic" invars ast target-heads)]
    (when-not (:quiet opts)
      (println "Compiling Tensor Logic graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn build-gemma4-kv-invars
  "Constructs EDN SSA signature invars for single-token Gemma 4 forward pass with KV Cache."
  [config max-seq-len]
  (let [{:keys [num-layers num-kv-shared-layers weight-dtype is-int8 is-int4 is-ternary layer-configs layer-types]} config
        is-ternary (boolean (or is-ternary (= weight-dtype :ternary) (= (:quant-type config) :ternary)))
        norm-dtype (if (or is-int8 is-int4 is-ternary) :bf16 weight-dtype)
        num-layers (long (or num-layers 35))
        num-kv-shared (long (or num-kv-shared-layers 0))
        num-unshared (- num-layers num-kv-shared)
        kv-invars (mapcat (fn [i]
                            (let [cfg (if (seq layer-configs) (nth layer-configs i nil) nil)
                                  is-global? (if cfg (:is-global? cfg) (gemma-logic/layer-is-global? layer-types i))
                                  win (when-not is-global? (or (:sliding-window cfg) (:sliding-window config) (:sliding_window config) 512))
                                  seq-l (if win (min max-seq-len win) max-seq-len)
                                  h-dim (or (:head-dim cfg) (if is-global? 512 256))
                                  n-kv (or (:num-kv-heads cfg) 1)]
                              [[(keyword (str "k_cache_in_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]
                               [(keyword (str "v_cache_in_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]]))
                          (range num-unshared))
        weight-invars (subvec (build-tensor-logic-invars (assoc config :last-token-only? true) max-seq-len) 2)]
    (vec (concat [[:x [:tensor [1 1] :i32]]
                  [:pos [:tensor [1] :i32]]]
                 kv-invars
                 weight-invars))))

(defn build-gemma4-kv-outvars
  "Constructs output variable list for Gemma 4 KV Cache step: [:logits k_cache_out_0 v_cache_out_0 ...]."
  [config]
  (let [num-layers (long (or (:num-layers config) 35))
        num-kv-shared (long (or (:num-kv-shared-layers config) 0))
        num-unshared (- num-layers num-kv-shared)
        out-kv-heads (mapcat (fn [i]
                               [(keyword (str "k_cache_out_" i))
                                (keyword (str "v_cache_out_" i))])
                             (range num-unshared))]
    (vec (into [:logits] out-kv-heads))))

(defn build-gemma4-prefill-outvars
  "Constructs output variable list for Gemma 4 prefill: [:logits (k_ro_sl_i | k_ro_i) (v_heads_sl_i | v_heads_i) ...]."
  [config]
  (gemma-logic/gemma4-prefill-outvars config (:max-seq-len config)))

(defn max-safe-prefill-seq-len
  "Returns the maximum sequence length for 1-shot parallel prefill that safely fits in VRAM
   without exceeding device workspace headroom."
  [config]
  (cond
    (or (:is-int4 config)
        (:is-ternary config)
        (re-find #"31[bB]" (or (:model-dir config) ""))
        (>= (long (or (:num-layers config) 0)) 60))
    2048

    :else
    8192))

(defn compile-gemma4-prefill-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR prefill executable that produces
   the next-token logits and initial populated KV cache tensors in a single parallel step."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        invars (build-tensor-logic-invars cfg max-seq-len)
        targets (build-gemma4-prefill-outvars cfg)
        ast (gemma-logic/gemma4-model-ast cfg)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 Prefill (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        graph (lower/ast->graph "gemma4_prefill" invars ast targets)]
    (when-not (:quiet opts)
      (println "Compiling Gemma 4 Prefill graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn allocate-kv-cache-buffers
  "Allocates initial zero-filled device VRAM buffers for KV cache of unshared layers."
  [{:keys [ctx config]} max-seq-len]
  (let [num-layers (long (or (:num-layers config) 35))
        num-kv-shared (long (or (:num-kv-shared-layers config) 0))
        num-unshared (- num-layers num-kv-shared)
        norm-enum (or (:norm-enum config) 13)
        layer-configs (:layer-configs config)
        layer-types (:layer-types config)]
    (vec (mapcat (fn [i]
                   (let [c (if (seq layer-configs) (nth layer-configs i nil) nil)
                         is-global? (if c (:is-global? c) (gemma-logic/layer-is-global? layer-types i))
                         win (when-not is-global? (or (:sliding-window c) (:sliding-window config) (:sliding_window config) 512))
                         seq-l (if win (min max-seq-len win) max-seq-len)
                         n-kv (long (or (:num-kv-heads c) 1))
                         h-dim (long (or (:head-dim c) (if is-global? 512 256)))
                         zeros (float-array (* seq-l n-kv h-dim))]
                     [(pjrt/buffer-from-host-buffer ctx (:client ctx) zeros [1 seq-l n-kv h-dim] norm-enum)
                      (pjrt/buffer-from-host-buffer ctx (:client ctx) zeros [1 seq-l n-kv h-dim] norm-enum)]))
                 (range num-unshared)))))

(defn compile-gemma4-kv-executable
  "Compiles single-step Gemma 4 KV-Cache AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        invars (build-gemma4-kv-invars cfg max-seq-len)
        targets (build-gemma4-kv-outvars cfg)
        ast (gemma-logic/gemma4-kv-model-ast cfg)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 KV Cache Step (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        graph (lower/ast->graph "gemma4_kv_step" invars ast targets)]
    (when-not (:quiet opts)
      (println "Compiling Gemma 4 KV Cache step graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(def GEMMA4-STOP-TOKEN-IDS
  "Special token IDs marking end-of-turn or end-of-generation in Gemma 4."
  #{1 106 49 50})

(defn compile-in-vram-loop-executable
  "Compiles an end-to-end in-VRAM autoregressive generation loop using StableHLO while-loop lowering with loop-carried KV-Cache."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        vocab-size (long (or (:vocab-size cfg) 262144))
        num-layers (long (or (:num-layers cfg) 35))
        num-kv-shared (long (or (:num-kv-shared-layers cfg) 0))
        num-unshared (- num-layers num-kv-shared)
        norm-dtype (if (or (:is-int8 cfg) (:is-int4 cfg) (:is-ternary cfg)) :bf16 (get cfg :weight-dtype :bf16))
        layer-configs (:layer-configs cfg)
        layer-types (:layer-types cfg)

        kv-invars (mapcat (fn [i]
                            (let [c (if (seq layer-configs) (nth layer-configs i nil) nil)
                                  is-global? (if c (:is-global? c) (gemma-logic/layer-is-global? layer-types i))
                                  win (when-not is-global? (or (:sliding-window c) (:sliding-window cfg) (:sliding_window cfg) 512))
                                  seq-l (if win (min max-seq-len win) max-seq-len)
                                  h-dim (or (:head-dim c) (if is-global? 512 256))
                                  n-kv (or (:num-kv-heads c) 1)]
                              [[(keyword (str "init_k_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]
                               [(keyword (str "init_v_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]]))
                          (range num-unshared))
        weight-invars (subvec (build-tensor-logic-invars cfg max-seq-len) 2)
        kv-init-names (mapcat (fn [i] [(keyword (str "init_k_" i)) (keyword (str "init_v_" i))]) (range num-unshared))
        kv-state-names (mapcat (fn [i] [(keyword (str "k_cache_in_" i)) (keyword (str "v_cache_in_" i))]) (range num-unshared))
        kv-out-names (mapcat (fn [i] [(keyword (str "k_cache_out_" i)) (keyword (str "v_cache_out_" i))]) (range num-unshared))
        kv-final-names (mapcat (fn [i] [(keyword (str "final_k_" i)) (keyword (str "final_v_" i))]) (range num-unshared))

        cond-args (into [:cur_step :target_max :cur_toks :cur_stopped] kv-state-names)
        cond-ast [:cond [:cond_out] {:args cond-args}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]

        body-in-args cond-args
        body-out-args (into [:next_step :target_max :next_toks :is_stop] kv-out-names)
        model-ast (gemma-logic/gemma4-kv-model-ast cfg)

        body-ast [:body body-out-args
                  {:args body-in-args}

                  ;; 1. Current position & single token input bindings
                  [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                  [:- [:pos_i32] [:cur_step] [:c_one]]
                  [:reshape [:pos] [:pos_i32] {:shape [1]}]
                  [:dynamic-slice [:cur_tok_2d] [:cur_toks] {:start-indices [0 :pos] :slice-sizes [1 1]}]
                  [:reshape [:x :b :p] [:cur_tok_2d] {:shape [1 1]}]

                  ;; 2. Single-step Gemma 4 forward pass with KV cache
                  model-ast

                  ;; 3. Argmax & loop state update
                  [:reshape [:logits_2d] [:logits] {:shape [1 vocab-size]}]
                  [:convert [:logits_f32] [:logits_2d] {:target-dtype :f32}]
                  [:argmax [:next_tok] [:logits_f32] {:axis 1}]
                  [:reshape [:next_tok_1d] [:next_tok] {:shape [1 1]}]
                  [:dynamic-update-slice [:next_toks] [:cur_toks] [:next_tok_1d] {:start-indices [0 :cur_step]}]
                  [:+ [:next_step] [:cur_step] [:c_one]]
                  [:constant [:eos_c] {:value 1 :type [:tensor [1] :i32] :shape [1]}]
                  [:constant [:eot_c] {:value 106 :type [:tensor [1] :i32] :shape [1]}]
                  [:compare [:c_eos] [:next_tok] [:eos_c] {:direction "EQ"}]
                  [:compare [:c_eot] [:next_tok] [:eot_c] {:direction "EQ"}]
                  [:or [:or_stop] [:c_eos] [:c_eot]]
                  [:reshape [:is_stop] [:or_stop] {:shape []}]]

        init-loop-args (into [:init_step :max_step :init_tokens :false_c] kv-init-names)
        while-out-vars (into [:final_step :final_max :final_tokens :final_stopped] kv-final-names)

        loop-ast [:block {}
                  [:constant [:false_c] {:value false :type [:tensor [] :i1] :shape []}]
                  [:while while-out-vars
                   init-loop-args
                   cond-ast
                   body-ast]]

        loop-invars (vec (concat [[:init_step [:tensor [] :i32]]
                                  [:max_step [:tensor [] :i32]]
                                  [:init_tokens [:tensor [1 max-seq-len] :i32]]]
                                 kv-invars
                                 weight-invars))
        loop-targets (into [:final_step :final_tokens] kv-final-names)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic In-VRAM Gemma 4 KV Loop (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        loop-graph (lower/ast->graph "gemma4_in_vram_kv_loop" loop-invars loop-ast loop-targets)]
    (when-not (:quiet opts)
      (println "Compiling In-VRAM KV Loop to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx loop-graph)))

(defn run-vram-loop-generation
  "Executes autoregressive token generation entirely within device VRAM using 1-shot prefill (or sequential prefill for large contexts) and an OpenXLA while-loop carrying KV-Cache."
  [session exec device-weights prompt-ids max-seq-len]
  (let [{:keys [ctx opts config kv-state prefill-executable]} session
        {:keys [max-new-tokens quiet]} opts
        seq-len (long max-seq-len)
        raw-p-count (count prompt-ids)
        safe-p-count (min raw-p-count (max 0 (- seq-len 2)))
        clamped-prompt-ids (if (< safe-p-count raw-p-count)
                             (subvec (vec prompt-ids) (- raw-p-count safe-p-count))
                             (vec prompt-ids))
        p-count (count clamped-prompt-ids)
        max-new (long (max 1 (min (- seq-len p-count 1) (or max-new-tokens 150))))
        target-max (long (+ p-count max-new))]
    (if (>= p-count (dec seq-len))
      clamped-prompt-ids
      (let [t0 (System/nanoTime)
            num-layers (long (or (:num-layers config) 35))
            num-kv-shared (long (or (:num-kv-shared-layers config) 0))
            num-unshared (- num-layers num-kv-shared)
            safe-prefill-len (max-safe-prefill-seq-len config)
            prefill-exec (or prefill-executable
                             (:prefill-executable session)
                             (when (<= seq-len safe-prefill-len)
                               (compile-gemma4-prefill-executable session seq-len)))
            in-arr (int-array seq-len)
            _ (dotimes [i p-count] (aset in-arr i (int (nth clamped-prompt-ids i))))

            ;; 1. Populate KV cache for prompt tokens (Parallel or Sequential)
            t-prefill-0 (System/nanoTime)
            prefill-kv (if (some? prefill-exec)
                         ;; Path A: 1-shot parallel prefill
                         (let [pos-p (int-array [(dec p-count)])
                               in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
                               pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                               prefill-inputs (into [in-b pos-b] device-weights)
                               num-prefill-outs (inc (* 2 num-unshared))
                               prefill-outs (pjrt/execute-executable ctx (or (:handle prefill-exec) prefill-exec) prefill-inputs num-prefill-outs)
                               _ (xla/destroy-buffer! ctx in-b)
                               _ (xla/destroy-buffer! ctx pos-b)
                               prefill-outs-vec (if (vector? prefill-outs) prefill-outs [prefill-outs])
                               prefill-logits (first prefill-outs-vec)
                               kv-outs (vec (subvec prefill-outs-vec 1))]
                           (xla/destroy-buffer! ctx prefill-logits)
                           kv-outs)
                         ;; Path B: Sequential prefill for sequence lengths exceeding VRAM parallel workspace headroom
                         (let [step-exec (or (:step-executable session)
                                             (compile-gemma4-kv-executable session seq-len))
                               initial-kv (allocate-kv-cache-buffers session seq-len)
                               num-step-outs (inc (* 2 num-unshared))
                               x-arr (int-array 1)
                               pos-arr (int-array 1)
                               prefill-limit (max 0 (dec p-count))]
                           (when-not quiet
                             (println (format "Prefilling %d prompt tokens into KV-Cache (exceeds parallel prefill limit %d)..."
                                              prefill-limit safe-prefill-len)))
                           (loop [p 0
                                  cur-kv initial-kv
                                  cur-log nil]
                             (if (< p prefill-limit)
                               (let [tok (int (nth clamped-prompt-ids p))
                                     _ (aset x-arr 0 tok)
                                     _ (aset pos-arr 0 p)
                                     x-b (xla/buffer-from-host-buffer ctx (:client ctx) x-arr [1 1] 4)
                                     pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                                     step-inputs (into [x-b pos-b] (concat cur-kv device-weights))
                                     outs (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) step-inputs num-step-outs)
                                     _ (xla/destroy-buffer! ctx x-b)
                                     _ (xla/destroy-buffer! ctx pos-b)
                                     outs-vec (if (vector? outs) outs [outs])
                                     new-log (first outs-vec)
                                     new-kv (vec (subvec outs-vec 1))]
                                 (when cur-log (xla/destroy-buffer! ctx cur-log))
                                 (doseq [b cur-kv] (xla/destroy-buffer! ctx b))
                                 (recur (inc p) new-kv new-log))
                               (do
                                 (when cur-log (xla/destroy-buffer! ctx cur-log))
                                 cur-kv)))))
            t-prefill-1 (System/nanoTime)
            prefill-ms (/ (- t-prefill-1 t-prefill-0) 1e6)

            ;; 2. Run In-VRAM While Loop carrying the KV cache
            b-step (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [p-count]) [] 4)
            b-max (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [target-max]) [] 4)
            b-toks (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
            loop-inputs (into [b-step b-max b-toks] (concat prefill-kv device-weights))
            num-loop-outs (+ 2 (* 2 num-unshared))

            t-loop-0 (System/nanoTime)
            loop-outs (pjrt/execute-executable ctx (or (:handle exec) exec) loop-inputs num-loop-outs)
            t-loop-1 (System/nanoTime)
            decode-ms (/ (- t-loop-1 t-loop-0) 1e6)

            _ (xla/destroy-buffer! ctx b-step)
            _ (xla/destroy-buffer! ctx b-max)
            _ (xla/destroy-buffer! ctx b-toks)
            _ (doseq [b prefill-kv] (xla/destroy-buffer! ctx b))

            loop-outs-vec (if (vector? loop-outs) loop-outs [loop-outs])
            out-step (nth loop-outs-vec 0)
            out-toks (nth loop-outs-vec 1)
            final-kv (vec (subvec loop-outs-vec 2))

            step-floats (pjrt/buffer-to-host-buffer ctx out-step 1 :f32)
            step-val (int (Float/floatToIntBits (aget step-floats 0)))
            toks-floats (pjrt/buffer-to-host-buffer ctx out-toks seq-len :f32)
            _ (xla/destroy-buffer! ctx out-step)
            _ (xla/destroy-buffer! ctx out-toks)

            actual-step (min (max p-count step-val) seq-len)
            final-ids (mapv #(Float/floatToIntBits %) (take actual-step (vec toks-floats)))
            cleaned-ids (if (and (> (count final-ids) p-count)
                                 (contains? GEMMA4-STOP-TOKEN-IDS (last final-ids)))
                          (subvec final-ids 0 (dec (count final-ids)))
                          final-ids)

            is-persistent? (some? kv-state)]
        (if is-persistent?
          (do
            (when-let [prior @kv-state]
              (doseq [b (:kv-buffers prior)] (when b (xla/destroy-buffer! ctx b))))
            (reset! kv-state {:cached-tokens cleaned-ids
                              :kv-buffers final-kv}))
          (doseq [b final-kv] (xla/destroy-buffer! ctx b)))

        (let [t-end (System/nanoTime)
              total-ms (/ (- t-end t0) 1e6)
              gen-count (- (count cleaned-ids) p-count)
              tok-s (if (pos? decode-ms) (/ (* gen-count 1000.0) decode-ms) 0.0)]
          (when-not quiet
            (println)
            (println "\n------------------------------------------------------------------")
            (println "  Telemetry Benchmark Metrics (In-VRAM While-Loop with KV-Cache):")
            (println (format "    • Prefill Latency             : %8.2f ms (%d tokens)" prefill-ms p-count))
            (println (format "    • Decode Latency              : %8.2f ms (%d tokens)" decode-ms gen-count))
            (println (format "    • Generation Speed            : %8.2f tok/s (%6.2f ms/tok)" tok-s (if (pos? gen-count) (/ decode-ms gen-count) 0.0)))
            (println (format "    • Total Generation Latency    : %8.2f ms" total-ms))
            (println "------------------------------------------------------------------\n"))
          cleaned-ids)))))

(defn argmax-host
  "Finds the index of the maximum float value in float array `arr`."
  [^floats arr]
  (let [n (alength arr)]
    (loop [i 1
           max-idx 0
           max-val (aget arr 0)]
      (if (< i n)
        (let [v (aget arr i)]
          (if (> v max-val)
            (recur (inc i) i v)
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn argmax-with-penalty
  "Selects the token with maximum logit value with sliding window linear repetition penalty."
  [^floats logits-arr gen-ids rep-pen]
  (let [n (alength logits-arr)
        recent-ids (take-last 32 gen-ids)
        penalty-set (disj (set recent-ids) 107 108 236743)
        pen (double (or rep-pen 1.15))]
    (loop [i 0
           max-idx 0
           max-val Float/NEGATIVE_INFINITY]
      (if (< i n)
        (let [raw-v (aget logits-arr i)
              v (if (contains? penalty-set i)
                  (if (pos? raw-v) (/ raw-v pen) (* raw-v pen))
                  raw-v)]
          (if (> v max-val)
            (recur (inc i) i (float v))
            (recur (inc i) max-idx max-val)))
        max-idx))))

(defn sample-next-token
  "Selects next token from float array `logits-arr` using sampling options and repetition penalty."
  [^floats logits-arr opts _prompt-ids gen-ids]
  (let [{:keys [temperature top-k top-p repetition-penalty]
         :or {temperature 0.0 top-k 10 top-p 1.0 repetition-penalty 1.15}} opts
        rep-pen (double (or repetition-penalty 1.15))]
    (if (or (nil? temperature) (<= temperature 0.0))
      (if (and (number? rep-pen) (> rep-pen 1.0))
        (argmax-with-penalty logits-arr gen-ids rep-pen)
        (argmax-host logits-arr))
      (let [logits-vec (vec logits-arr)]
        (sampling/sample-logits logits-vec {:temperature temperature
                                            :top-k top-k
                                            :top-p top-p
                                            :repetition-penalty rep-pen
                                            :seen-ids gen-ids})))))

(defn run-autoregressive-generation-logic
  "Executes autoregressive token generation using pure Tensor Logic Gemma 4 executable."
  ([session exec device-weights prompt-ids]
   (run-autoregressive-generation-logic session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config tokenizer] :as session} exec device-weights prompt-ids max-seq-len]
   (let [{:keys [max-new-tokens quiet mode]} opts
         is-agent? (= mode :agent)
         seq-len (long (or max-seq-len (:max-seq-len config) 128))
         vocab-size (long (or (:vocab-size config) (:vocab_size config) 262144))
         last-token? (get opts :last-token-only? true)
         weight-dt (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config)) :bf16 (get config :weight-dtype :bf16))
         prompt-count (count prompt-ids)
         in-arr (int-array seq-len)
         pos-arr (when last-token? (int-array 1))
         _ (dotimes [i (min prompt-count seq-len)]
             (aset in-arr i (int (nth prompt-ids i))))
         cur-tokens (atom (vec prompt-ids))
         t0 (System/nanoTime)]
     (loop [step 0]
       (if (>= step max-new-tokens)
         nil
         (let [s-len (count @cur-tokens)
               _ (when last-token? (aset pos-arr 0 (dec s-len)))
               in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
               pos-b (when last-token?
                       (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4))
               args (let [base (if last-token? [in-b pos-b] [in-b])
                          rel-bufs (get session :relational-buffers [])]
                      (into base (concat device-weights rel-bufs)))
               out (xla/execute exec args)
               out-buf (if (sequential? out) (first out) out)
               logits (if last-token?
                        (xla/to-host-slice out-buf 0 vocab-size vocab-size weight-dt)
                        (xla/to-host-slice out-buf (dec s-len) vocab-size (* seq-len vocab-size) weight-dt))
               next-id (sample-next-token logits opts prompt-ids (subvec @cur-tokens (count prompt-ids)))]
           (xla/destroy-buffer! ctx in-b)
           (when pos-b (xla/destroy-buffer! ctx pos-b))
           (if (sequential? out)
             (doseq [b out] (xla/destroy-buffer! ctx b))
             (xla/destroy-buffer! ctx out))
           (when (< s-len seq-len)
             (aset in-arr s-len (int next-id)))
           (swap! cur-tokens conj next-id)
           (when (and (not quiet) (not is-agent?))
             (print (decode tokenizer [next-id]))
             (flush))
           (if (or (contains? GEMMA4-STOP-TOKEN-IDS next-id) (= next-id (eos-id tokenizer)))
             nil
             (recur (inc step))))))
     (let [t1 (System/nanoTime)
           total-ms (/ (- t1 t0) 1e6)
           gen-count (- (count @cur-tokens) (count prompt-ids))
           tok-s (if (pos? total-ms) (/ (* gen-count 1000.0) total-ms) 0.0)]
       (when-not quiet
         (println)
         (println "\n------------------------------------------------------------------")
         (println "  Telemetry Benchmark Metrics (Tensor Logic End-to-End):")
         (println (format "    • Total Generation Latency    : %8.2f ms (%d tokens)" total-ms gen-count))
         (println (format "    • Generation Speed            : %8.2f tok/s (%6.2f ms/tok)" tok-s (if (pos? gen-count) (/ total-ms gen-count) 0.0)))
         (println "------------------------------------------------------------------\n")))
     @cur-tokens)))

(defn common-prefix-len
  "Returns the number of leading items shared by sequences xs and ys."
  [xs ys]
  (let [n (min (count xs) (count ys))]
    (loop [i 0]
      (if (and (< i n) (= (nth xs i) (nth ys i)))
        (recur (inc i))
        i))))

(defn run-cached-kv-generation
  "Executes autoregressive generation with in-VRAM KV-Cache using single-token step executable
   and 1-shot parallel prefill."
  ([session exec device-weights prompt-ids]
   (run-cached-kv-generation session exec device-weights prompt-ids nil))
  ([{:keys [ctx opts config tokenizer kv-state prefill-executable] :as session} exec device-weights prompt-ids max-seq-len]
   (let [{:keys [max-new-tokens quiet mode]} opts
         is-agent? (= mode :agent)
         seq-len (long (or max-seq-len (:max-seq-len config) 512))
         vocab-size (long (or (:vocab-size config) 262144))
         weight-dt (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config)) :bf16 (get config :weight-dtype :bf16))
         num-layers (long (or (:num-layers config) 35))
         num-kv-shared (long (or (:num-kv-shared-layers config) 0))
         num-unshared (- num-layers num-kv-shared)
         num-outs (inc (* 2 num-unshared))
         raw-prompt-count (count prompt-ids)
         safe-prompt-count (min raw-prompt-count (max 0 (- seq-len 2)))
         clamped-prompt-ids (if (< safe-prompt-count raw-prompt-count)
                              (subvec (vec prompt-ids) (- raw-prompt-count safe-prompt-count))
                              (vec prompt-ids))
         prompt-count (count clamped-prompt-ids)
         prefill-exec (or prefill-executable (:prefill-executable session))
         is-persistent? (some? kv-state)
         prior-cache (when is-persistent? @kv-state)
         p-match (if (and prior-cache (:cached-tokens prior-cache) (seq (:kv-buffers prior-cache)))
                   (common-prefix-len (:cached-tokens prior-cache) clamped-prompt-ids)
                   0)
         _ (when (and is-persistent? prior-cache (zero? p-match))
             (doseq [b (:kv-buffers prior-cache)] (xla/destroy-buffer! ctx b))
             (reset! kv-state nil))
         initial-kv (cond
                      (pos? p-match)
                      (:kv-buffers prior-cache)

                      (and (some? prefill-exec) (<= seq-len (max-safe-prefill-seq-len config)))
                      nil

                      :else
                      (allocate-kv-cache-buffers session seq-len))
         kv-buffers-atom (atom initial-kv)
         x-arr (int-array 1)
         pos-arr (int-array 1)
         t0 (System/nanoTime)]
     (try
       ;; Phase 1: Prefill prompt tokens into KV cache
       (let [[last-logits t-prefill-end]
             (cond
               ;; Path A: Cache hit from token 0 to p-match (delta prefill via step executable)
               (pos? p-match)
               (let [start-p (if (= p-match prompt-count) (max 0 (dec prompt-count)) p-match)
                     cur-log (loop [p start-p
                                    cur-logits nil]
                               (if (< p prompt-count)
                                 (let [tok (int (nth clamped-prompt-ids p))
                                       _ (aset x-arr 0 tok)
                                       _ (aset pos-arr 0 p)
                                       x-b (xla/buffer-from-host-buffer ctx (:client ctx) x-arr [1 1] 4)
                                       pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                                       step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                       outs (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs)
                                       _ (xla/destroy-buffer! ctx x-b)
                                       _ (xla/destroy-buffer! ctx pos-b)
                                       outs-vec (if (vector? outs) outs [outs])
                                       new-logits (first outs-vec)
                                       new-kv (vec (subvec outs-vec 1))
                                       old-kv @kv-buffers-atom]
                                   (when cur-logits (xla/destroy-buffer! ctx cur-logits))
                                   (when (not= old-kv (:kv-buffers prior-cache))
                                     (doseq [b old-kv] (xla/destroy-buffer! ctx b)))
                                   (reset! kv-buffers-atom new-kv)
                                   (recur (inc p) new-logits))
                                 cur-logits))]
                 [cur-log (System/nanoTime)])

               ;; Path B: 1-Shot Parallel Prefill
               (some? prefill-exec)
               (let [in-arr (int-array seq-len)
                     _ (dotimes [i prompt-count] (aset in-arr i (int (nth clamped-prompt-ids i))))
                     pos-p (int-array [(dec prompt-count)])
                     in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 seq-len] 4)
                     pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                     step-inputs (into [in-b pos-b] device-weights)
                     outs (pjrt/execute-executable ctx (or (:handle prefill-exec) prefill-exec) step-inputs num-outs)
                     _ (xla/destroy-buffer! ctx in-b)
                     _ (xla/destroy-buffer! ctx pos-b)
                     outs-vec (if (vector? outs) outs [outs])
                     prefill-logits (first outs-vec)
                     prefill-kv (vec (subvec outs-vec 1))]
                 (reset! kv-buffers-atom prefill-kv)
                 [prefill-logits (System/nanoTime)])

               ;; Path C: Sequential fallback prefill
               :else
               (let [cur-log (loop [p 0
                                    cur-logits nil]
                               (if (< p prompt-count)
                                 (let [tok (int (nth clamped-prompt-ids p))
                                       _ (aset x-arr 0 tok)
                                       _ (aset pos-arr 0 p)
                                       x-b (xla/buffer-from-host-buffer ctx (:client ctx) x-arr [1 1] 4)
                                       pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                                       step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                                       outs (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs)
                                       _ (xla/destroy-buffer! ctx x-b)
                                       _ (xla/destroy-buffer! ctx pos-b)
                                       outs-vec (if (vector? outs) outs [outs])
                                       new-logits (first outs-vec)
                                       new-kv (vec (subvec outs-vec 1))
                                       old-kv @kv-buffers-atom]
                                   (when cur-logits (xla/destroy-buffer! ctx cur-logits))
                                   (doseq [b old-kv] (xla/destroy-buffer! ctx b))
                                   (reset! kv-buffers-atom new-kv)
                                   (recur (inc p) new-logits))
                                 cur-logits))]
                 [cur-log (System/nanoTime)]))

             cur-tokens (atom (vec clamped-prompt-ids))
             max-tokens (long (or max-new-tokens 256))]
         ;; Phase 2: Autoregressive decode loop
         (loop [step prompt-count
                cur-logits last-logits]
           (if (or (>= (- (count @cur-tokens) prompt-count) max-tokens)
                   (>= step (dec seq-len)))
             (when cur-logits (xla/destroy-buffer! ctx cur-logits))
             (let [logits-data (xla/to-host-slice cur-logits 0 vocab-size vocab-size weight-dt)
                   _ (xla/destroy-buffer! ctx cur-logits)
                   next-id (sample-next-token logits-data opts clamped-prompt-ids (subvec @cur-tokens prompt-count))]
               (swap! cur-tokens conj next-id)
               (when (and (not quiet) (not is-agent?))
                 (print (decode tokenizer [next-id]))
                 (flush))
               (if (or (contains? GEMMA4-STOP-TOKEN-IDS next-id) (= next-id (eos-id tokenizer)))
                 nil
                 (let [_ (aset x-arr 0 (int next-id))
                       _ (aset pos-arr 0 step)
                       x-b (xla/buffer-from-host-buffer ctx (:client ctx) x-arr [1 1] 4)
                       pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                       step-inputs (into [x-b pos-b] (concat @kv-buffers-atom device-weights))
                       outs (pjrt/execute-executable ctx (or (:handle exec) exec) step-inputs num-outs)
                       _ (xla/destroy-buffer! ctx x-b)
                       _ (xla/destroy-buffer! ctx pos-b)
                       outs-vec (if (vector? outs) outs [outs])
                       new-logits (first outs-vec)
                       new-kv (vec (subvec outs-vec 1))
                       old-kv @kv-buffers-atom]
                   (doseq [b old-kv] (xla/destroy-buffer! ctx b))
                   (reset! kv-buffers-atom new-kv)
                   (recur (inc step) new-logits))))))
         (when is-persistent?
           (reset! kv-state {:cached-tokens @cur-tokens
                             :kv-buffers @kv-buffers-atom}))
         (let [t1 (System/nanoTime)
               total-ms (/ (- t1 t0) 1e6)
               prefill-ms (/ (- t-prefill-end t0) 1e6)
               decode-ms (/ (- t1 t-prefill-end) 1e6)
               gen-count (- (count @cur-tokens) prompt-count)
               decode-tok-s (if (pos? decode-ms) (/ (* gen-count 1000.0) decode-ms) 0.0)]
           (when-not quiet
             (println)
             (println "\n------------------------------------------------------------------")
             (println "  Telemetry Benchmark Metrics (Tensor Logic KV-Cache):")
             (println (format "    • Prefill Latency             : %8.2f ms (%d tokens, %d cached, %d new)"
                              prefill-ms prompt-count p-match (- prompt-count p-match)))
             (println (format "    • Decode Latency              : %8.2f ms (%d tokens)" decode-ms gen-count))
             (println (format "    • Decode Speed                : %8.2f tok/s (%6.2f ms/tok)" decode-tok-s (if (pos? gen-count) (/ decode-ms gen-count) 0.0)))
             (println (format "    • Total Generation Latency    : %8.2f ms" total-ms))
             (println "------------------------------------------------------------------\n"))
           @cur-tokens))
       (finally
         (when-not is-persistent?
           (doseq [b @kv-buffers-atom] (when b (xla/destroy-buffer! ctx b)))))))))

(defn run-autoregressive-generation
  "Executes autoregressive generation either via in-VRAM while loop, cached KV generation, or full sequence recomputation."
  ([session exec device-weights prompt-ids]
   (run-autoregressive-generation session exec device-weights prompt-ids nil))
  ([session exec device-weights prompt-ids max-seq-len]
   (let [{:keys [opts]} session
         method (or (:method opts) (if (:vram-loop? session) :vram-loop :kv-cache))
         vram-loop? (if (contains? opts :vram-loop?)
                      (:vram-loop? opts)
                      (= method :vram-loop))]
     (cond
       vram-loop?
       (run-vram-loop-generation session exec device-weights prompt-ids (or max-seq-len (:max-seq-len session) 128))

       (= method :tensor-logic-full)
       (run-autoregressive-generation-logic session exec device-weights prompt-ids max-seq-len)

       :else
       (run-cached-kv-generation session exec device-weights prompt-ids (or max-seq-len (:max-seq-len session) 512))))))

(def compile-executable compile-tensor-logic-executable)

(defn generate-text
  "Generates text response using Gemma 4 model via pure Tensor Logic execution."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        {:keys [max-new-tokens temperature top-k model quiet]} opts
        clean-prompt (or prompt "The capital of France is")
        model-str (or model (get-in session [:config :model-dir]) "")
        is-it-model (str/includes? (str/lower-case model-str) "-it")
        is-already-templated (or (str/includes? clean-prompt "<|turn>user") (str/includes? clean-prompt "<|turn>model"))
        prompt-ids (cond
                     (and is-it-model (not is-already-templated))
                     (let [raw-ids (encode tokenizer clean-prompt)
                           clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)
                           prefix (if (:thinking opts)
                                    ;; <bos><|turn>system\n<|think|><turn|>\n<|turn>user\n
                                    [(bos-id tokenizer) 105 9731 107 98 106 107 105 2364 107]
                                    [(bos-id tokenizer) 105 2364 107])]
                       (vec (concat prefix clean-ids [106 107 105 4368 107])))

                     :else
                     (let [raw-ids (encode tokenizer clean-prompt)]
                       (if (= (first raw-ids) (bos-id tokenizer))
                         (vec raw-ids)
                         (vec (cons (bos-id tokenizer) raw-ids)))))
        prompt-len (count prompt-ids)
        max-seq-len (long (or (:max-seq-len session) (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens 16))))]
    (when-not quiet
      (let [prompt-str (if (> (count clean-prompt) 200)
                         (str (subs clean-prompt 0 100) " ... [truncated " (count clean-prompt) " chars] ... " (subs clean-prompt (- (count clean-prompt) 100)))
                         clean-prompt)
            tok-str (if (> prompt-len 30)
                      (str "[" (str/join " " (take 10 prompt-ids)) " ... " (str/join " " (take-last 5 prompt-ids)) "]")
                      (str prompt-ids))]
        (println (format "Prompt: \"%s\"" prompt-str))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s, method=%s"
                         max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))
                         (name (or (:method opts) :kv-cache))))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len tok-str))))

    (let [metrics-atom (or (:metrics-atom session) (atom {}))
          trace-spans-atom (or (:trace-spans-atom session) (atom []))
          reuse-weights? (some? (:device-weights session))
          reuse-exec? (some? (:executable session))
          exec (binding [profile/*active-trace-spans* trace-spans-atom]
                 (if reuse-exec?
                   (:executable session)
                   (profile/with-profile metrics-atom "graph_compilation"
                     (cond
                       (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                       (compile-in-vram-loop-executable session max-seq-len)

                       (= (:method opts) :tensor-logic-full)
                       (compile-tensor-logic-executable session max-seq-len)

                       :else
                       (compile-gemma4-kv-executable session max-seq-len)))))
          safe-prefill-len (max-safe-prefill-seq-len (:config session))
          prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                         (if (:prefill-executable session)
                           (:prefill-executable session)
                           (when (and (or (= (or (:method opts) :kv-cache) :kv-cache)
                                          (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                                      (<= max-seq-len safe-prefill-len))
                             (profile/with-profile metrics-atom "graph_compilation"
                               (compile-gemma4-prefill-executable session max-seq-len)))))
          step-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                      (when (and (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                                 (> max-seq-len safe-prefill-len))
                        (profile/with-profile metrics-atom "graph_compilation"
                          (compile-gemma4-kv-executable session max-seq-len))))
          active-session (assoc session
                                :prefill-executable prefill-exec
                                :step-executable step-exec)
          device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                           (if reuse-weights?
                             (:device-weights session)
                             (profile/with-profile metrics-atom "weight_transfer"
                               (allocate-device-weights session))))]
      (when-not quiet
        (println "\nGenerating tokens autoregressively with pure Tensor Logic Gemma 4 Kernel..."))
      (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                            (profile/with-profile metrics-atom "autoregressive_generation"
                              (run-autoregressive-generation active-session exec device-weights prompt-ids max-seq-len)))
            generated-str (decode tokenizer final-context)]
        (when-not quiet
          (println)
          (println generated-str))
        (when-not reuse-weights?
          (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w)))
        (when-let [out-path (:out opts)]
          (spit out-path generated-str)
          (when-not quiet
            (println (format "\n  ↳ Written generated output to [%s]" out-path))))
        (when-let [profile-path (:profile-out opts)]
          (spit profile-path (with-out-str (pprint @metrics-atom)))
          (when-not quiet
            (println (format "  ↳ Saved telemetry profile report to [%s]" profile-path))))
        (when-let [trace-path (:chrome-trace-out opts)]
          (profile/save-chrome-trace! @trace-spans-atom trace-path)
          (when-not quiet
            (println (format "  ↳ Saved Chrome tracing JSON to [%s]" trace-path))))
        (when-not quiet
          (println "\n==================================================================")
          (println "=== Tensor Logic Gemma 4 Generation Verification Passed! ===")
          (println "=================================================================="))
        final-context))))

(defn generate-text-string
  "Generates text response using Gemma 4 model session and returns decoded text string."
  [session prompt]
  (let [{:keys [tokenizer]} session
        final-context (generate-text session prompt)]
    (decode tokenizer final-context)))

(defn init-agent-vram-session
  "Initializes a persistent VRAM session for agent loops.
   Pre-allocates weights in device memory and compiles the In-VRAM While Loop (or KV-Cache step and prefill) executable once."
  ([opts]
   (init-agent-vram-session opts (long (or (:max-seq-len opts) 1024))))
  ([opts max-seq-len]
   (let [method (or (:method opts)
                    (if (and (number? (:temperature opts)) (> (:temperature opts) 0.0))
                      :kv-cache
                      :vram-loop))
         vram-loop? (if (contains? opts :vram-loop?)
                      (:vram-loop? opts)
                      (= method :vram-loop))
         opts (assoc opts :mode :agent :max-seq-len max-seq-len :method method :vram-loop? vram-loop?)
         session (init-inference-session opts)
         _ (when-not (:quiet opts)
             (println (format "Pre-compiling Gemma 4 %s graph (max-seq-len=%d)..."
                              (if vram-loop? "In-VRAM While Loop" "KV-Cache")
                              max-seq-len)))
         exec (if vram-loop?
                (compile-in-vram-loop-executable session max-seq-len)
                (compile-gemma4-kv-executable session max-seq-len))
         safe-prefill-len (max-safe-prefill-seq-len (:config session))
         prefill-exec (when (<= max-seq-len safe-prefill-len)
                        (compile-gemma4-prefill-executable session max-seq-len))
         step-exec (when (and vram-loop? (> max-seq-len safe-prefill-len))
                     (compile-gemma4-kv-executable session max-seq-len))
         _ (when-not (:quiet opts) (println "Pinning Gemma 4 weights in PJRT VRAM..."))
         device-weights (allocate-device-weights session)]
     (assoc session
            :device-weights device-weights
            :executable exec
            :prefill-executable prefill-exec
            :step-executable step-exec
            :kv-state (atom nil)
            :max-seq-len max-seq-len
            :vram-session? true
            :vram-loop? vram-loop?))))

(defn close-agent-session!
  "Releases VRAM resources for an agent session."
  [{:keys [ctx device-weights kv-state]}]
  (when (seq device-weights)
    (doseq [w device-weights]
      (xla/destroy-buffer! ctx w)))
  (when (and kv-state @kv-state)
    (doseq [b (:kv-buffers @kv-state)]
      (xla/destroy-buffer! ctx b))
    (reset! kv-state nil)))

(defn- find-libjsig
  "Searches standard JDK paths for libjsig.so."
  []
  (let [jh (System/getProperty "java.home")
        paths [(str jh "/lib/server/libjsig.so")
               (str jh "/lib/libjsig.so")
               "/usr/lib64/openjdk-25/lib/server/libjsig.so"]]
    (first (filter #(.exists (io/file %)) paths))))

(defn needs-libjsig-reexec?
  "Returns true when running with ROCm backend and LD_PRELOAD does not
   already include libjsig.so. Signal chaining via LD_PRELOAD is required
   because the ROCm PJRT plugin bundles LLVM, which installs its own signal
   handlers that conflict with the JVM's."
  [opts]
  (and (= (:backend opts) :rocm)
       (not (some-> (System/getenv "LD_PRELOAD")
                    (.contains "libjsig")))))

(defn reexec-with-libjsig!
  "Re-launches current JVM process with LD_PRELOAD=libjsig.so for ROCm signal chaining."
  ([args] (reexec-with-libjsig! args "tools.gemma4-inference"))
  ([args main-ns]
   (let [jsig-path (find-libjsig)]
     (when-not jsig-path
       (binding [*out* *err*]
         (println "WARNING: libjsig.so not found — ROCm signal chaining unavailable."))
       (flush)
       nil)
     (when jsig-path
       (let [jh (System/getProperty "java.home")
             java-bin (str jh "/bin/java")
             rt-bean (java.lang.management.ManagementFactory/getRuntimeMXBean)
             jvm-args (.getInputArguments rt-bean)
             cp (System/getProperty "java.class.path")
             cmd (vec (concat [java-bin]
                              jvm-args
                              ["-cp" cp "clojure.main" "-m" main-ns]
                              args))
             pb (ProcessBuilder. ^java.util.List cmd)
             env (.environment pb)
             existing-preload (.get env "LD_PRELOAD")
             new-preload (if (and existing-preload (not (.isEmpty ^String existing-preload)))
                           (str jsig-path ":" existing-preload)
                           jsig-path)]
         (.put env "LD_PRELOAD" new-preload)
         (.inheritIO pb)
         (System/exit (.waitFor (.start pb))))))))

(defn -main
  "CLI entrypoint for Gemma 4 text generation."
  [& args]
  (try
    (let [opts (parse-cli-args args)]
      ;; Re-exec with LD_PRELOAD=libjsig.so for ROCm signal chaining.
      ;; Must happen before any PJRT plugin is loaded.
      (when (needs-libjsig-reexec? opts)
        (reexec-with-libjsig! args))
      (when-not (:quiet opts)
        (println "==================================================================")
        (println (str "  clj-xla Gemma 4 Single-Pass Prefill & " (case (:precision opts) :int8 "INT8" :int4 "INT4" "BF16") " Generation "))
        (println "=================================================================="))
      (let [session (init-inference-session opts)]
        (generate-text session (:prompt opts))))
    (catch Throwable e
      (println "\nExecution Exception:" (.getMessage e))
      (.printStackTrace e))
    (finally
      (.. Runtime getRuntime (halt 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
