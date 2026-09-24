(ns einsum.models.gemma4.weights
  "Gemma 4 device weight allocation, format conversion, and quantization unpacking."
  (:require [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.logic.exl3 :as exl3]
            [einsum.models.gemma :as gemma-logic]
            [einsum.quant.ternary :as ternary]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.safetensors :as st]))

(defn floats->bf16-shorts
  "Converts float array to short array of bfloat16 bit-patterns."
  ^shorts [^floats fa]
  (let [n (alength fa)
        sa (short-array n)]
    (dotimes [i n]
      (aset-short sa i (exl3/float->bf16-short (aget fa i))))
    sa))

(defn quantize-bf16-to-int8
  "Quantizes a BF16 short-array to INT8 byte-array with per-tensor symmetric quantization.
   Returns {:data byte-array :scale float}."
  [^shorts bf16-shorts]
  (let [n (alength bf16-shorts)
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

(defn load-weight-buffer
  "Loads a single weight tensor from `weights-mmap` into PJRT device memory in specified precision.
   Optionally registers the buffer in `arena` (or `arena/*active-arena*`)."
  ([ctx weights-mmap tensor-name shape weight-dtype weight-enum]
   (load-weight-buffer ctx weights-mmap tensor-name shape weight-dtype weight-enum 0.0 nil))
  ([ctx weights-mmap tensor-name shape weight-dtype weight-enum default-val]
   (load-weight-buffer ctx weights-mmap tensor-name shape weight-dtype weight-enum default-val nil))
  ([ctx weights-mmap tensor-name shape _weight-dtype weight-enum default-val arena]
   (let [target-arena (or arena arena/*active-arena*)
         header (or (:header weights-mmap) {})
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
                                  tensor-name)))
         buf (cond
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
                 (xla/buffer-from-host-buffer ctx (:client ctx) slice shape weight-enum)))]
     (when target-arena
       (xla/track! target-arena buf))
     buf)))

(defn load-linear-projection-buffers
  "Loads a linear projection matrix as either a single unquantized PJRT buffer,
   or if is-int8/is-int4/is-ternary is true, a pair [w-buf scale-buf] with in-graph quantization.
   Optionally registers allocated buffers in `arena` (or `arena/*active-arena*`)."
  ([ctx weights-mmap tensor-name shape is-int8 norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 false false norm-enum weight-enum 128 nil))
  ([ctx weights-mmap tensor-name shape is-int8 is-int4 norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 is-int4 false norm-enum weight-enum 128 nil))
  ([ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum 128 nil))
  ([ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum group-size]
   (load-linear-projection-buffers ctx weights-mmap tensor-name shape is-int8 is-int4 is-ternary norm-enum weight-enum group-size nil))
  ([ctx weights-mmap tensor-name [rows cols :as shape] is-int8 is-int4 is-ternary norm-enum _weight-enum group-size arena]
   (let [target-arena (or arena arena/*active-arena*)]
     (if-not (or is-int8 is-int4 is-ternary)
       [(load-weight-buffer ctx weights-mmap tensor-name shape (if (= norm-enum 11) :f32 :bf16) norm-enum 0.0 target-arena)]
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
             (when target-arena
               (xla/track! target-arena w-buf)
               (xla/track! target-arena scale-buf))
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
                 (when target-arena
                   (xla/track! target-arena w-buf)
                   (xla/track! target-arena scale-buf))
                 [w-buf scale-buf])

               is-int4
               (let [{:keys [data scales scale-shape]} (exl3/quantize-weights-per-row-int4 raw-arr rows cols
                                                                                           {:as scale-format
                                                                                            :group-size group-size})
                     w-buf (xla/buffer-from-host-buffer ctx (:client ctx) data [rows (quot cols 2)] 2)
                     scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scales (or scale-shape [rows]) norm-enum)]
                 (when target-arena
                   (xla/track! target-arena w-buf)
                   (xla/track! target-arena scale-buf))
                 [w-buf scale-buf])

               :else
               (let [{:keys [data scales]} (exl3/quantize-weights-per-row-int8 raw-arr rows cols {:as scale-format})
                     w-buf (xla/buffer-from-host-buffer ctx (:client ctx) data shape 2)
                     scale-buf (xla/buffer-from-host-buffer ctx (:client ctx) scales [rows] norm-enum)]
                 (when target-arena
                   (xla/track! target-arena w-buf)
                   (xla/track! target-arena scale-buf))
                 [w-buf scale-buf])))))))))

(defn allocate-device-weights
  "Loads individual weight tensors for Gemma 4 into PJRT device buffers matching build-tensor-logic-invars.
   Registers buffers in session-arena and populates weight-store device-buffers map."
  [{:keys [ctx weights-mmap config session-arena weight-store] :as _session}]
  (let [target-arena (or session-arena arena/*active-arena*)
        register-store-entry! (fn [k buf]
                                (when (and weight-store buf)
                                  (swap! (:device-buffers weight-store) assoc k buf))
                                buf)]
    (binding [arena/*active-arena* target-arena]
      (let [{:keys [prefix-base vocab-size hidden-dim total-pl-dim pl-dim weight-dtype weight-enum norm-enum layer-configs num-layers is-int8 is-int4 is-ternary group-size]} config
            has-ple? (pos? total-pl-dim)
            load-fn (fn
                      ([name shape enum] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum 0.0 target-arena))
                      ([name shape enum default-val] (load-weight-buffer ctx weights-mmap name shape weight-dtype enum default-val target-arena)))
            embed-buf (register-store-entry! :embed_tokens
                                             (load-fn (str prefix-base "embed_tokens.weight") [vocab-size hidden-dim] norm-enum))
            ple-bufs (when has-ple?
                       (let [b1 (register-store-entry! :embed_tokens_per_layer
                                                       (load-fn (str prefix-base "embed_tokens_per_layer.weight") [vocab-size total-pl-dim] norm-enum))
                             b2 (register-store-entry! :per_layer_model_projection
                                                       (load-fn (str prefix-base "per_layer_model_projection.weight") [total-pl-dim hidden-dim] norm-enum))
                             b3 (register-store-entry! :per_layer_projection_norm
                                                       (load-fn (str prefix-base "per_layer_projection_norm.weight") [pl-dim] norm-enum))]
                         [b1 b2 b3]))
            layer-bufs (mapcat (fn [i]
                                 (let [kmap (gemma-logic/gemma4-weight-key-map i (str prefix-base "layers."))
                                       cfg (nth layer-configs i)
                                       q-dim (:q-dim cfg)
                                       kv-dim (:kv-dim cfg)
                                       head-dim (:head-dim cfg)
                                       mlp-dim (:mlp-dim cfg)
                                       skipped? (contains? (set (:skip-layers config)) i)
                                       layer-is-ternary (and is-ternary (not skipped?))
                                       layer-is-int8 (and is-int8 (not skipped?))
                                       layer-is-int4 (and is-int4 (not skipped?))
                                       quantized? (or layer-is-ternary layer-is-int8 layer-is-int4)
                                       load-linear-fn (fn [name shape w-kw scale-kw]
                                                        (let [bufs (load-linear-projection-buffers ctx weights-mmap name shape layer-is-int8 layer-is-int4 layer-is-ternary norm-enum weight-enum group-size target-arena)]
                                                          (if quantized?
                                                            (do (register-store-entry! w-kw (first bufs))
                                                                (register-store-entry! scale-kw (second bufs)))
                                                            (register-store-entry! w-kw (first bufs)))
                                                          bufs))
                                       in-ln (register-store-entry! (keyword (str "input_ln_w_" i))
                                                                    (load-fn (:input-ln-w kmap) [hidden-dim] norm-enum 0.0))
                                       l-scalar (register-store-entry! (keyword (str "layer_scalar_" i))
                                                                       (load-fn (:layer-scalar-w kmap) [1] norm-enum 1.0))
                                       q-bufs (load-linear-fn (:q-w kmap) [q-dim hidden-dim] (keyword (str "q_w_" i)) (keyword (str "q_scale_" i)))
                                       k-bufs (load-linear-fn (:k-w kmap) [kv-dim hidden-dim] (keyword (str "k_w_" i)) (keyword (str "k_scale_" i)))
                                       v-bufs (load-linear-fn (:v-w kmap) [kv-dim hidden-dim] (keyword (str "v_w_" i)) (keyword (str "v_scale_" i)))
                                       o-bufs (load-linear-fn (:o-w kmap) [hidden-dim q-dim] (keyword (str "o_w_" i)) (keyword (str "o_scale_" i)))
                                       qn (register-store-entry! (keyword (str "q_norm_w_" i))
                                                                 (load-fn (:q-norm-w kmap) [head-dim] norm-enum 0.0))
                                       kn (register-store-entry! (keyword (str "k_norm_w_" i))
                                                                 (load-fn (:k-norm-w kmap) [head-dim] norm-enum 0.0))
                                       post-attn (register-store-entry! (keyword (str "post_attn_ln_w_" i))
                                                                        (load-fn (:post-attn-ln-w kmap) [hidden-dim] norm-enum 0.0))
                                       pre-mlp (register-store-entry! (keyword (str "pre_mlp_ln_w_" i))
                                                                      (load-fn (:pre-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0))
                                       post-mlp (register-store-entry! (keyword (str "post_mlp_ln_w_" i))
                                                                       (load-fn (:post-mlp-ln-w kmap) [hidden-dim] norm-enum 0.0))
                                       gate-bufs (load-linear-fn (:gate-w kmap) [mlp-dim hidden-dim] (keyword (str "gate_w_" i)) (keyword (str "gate_scale_" i)))
                                       up-bufs (load-linear-fn (:up-w kmap) [mlp-dim hidden-dim] (keyword (str "up_w_" i)) (keyword (str "up_scale_" i)))
                                       down-bufs (load-linear-fn (:down-w kmap) [hidden-dim mlp-dim] (keyword (str "down_w_" i)) (keyword (str "down_scale_" i)))
                                       ple-layer-bufs (when has-ple?
                                                        [(register-store-entry! (keyword (str "per_layer_gate_w_" i))
                                                                                (load-fn (:per-layer-gate-w kmap) [pl-dim hidden-dim] norm-enum 0.0))
                                                         (register-store-entry! (keyword (str "per_layer_proj_w_" i))
                                                                                (load-fn (:per-layer-proj-w kmap) [hidden-dim pl-dim] norm-enum 0.0))
                                                         (register-store-entry! (keyword (str "post_per_layer_norm_w_" i))
                                                                                (load-fn (:post-per-layer-norm-w kmap) [hidden-dim] norm-enum 0.0))])]
                                   (concat
                                    [in-ln l-scalar]
                                    q-bufs k-bufs v-bufs o-bufs
                                    [qn kn post-attn pre-mlp post-mlp]
                                    gate-bufs up-bufs down-bufs
                                    ple-layer-bufs)))
                               (range num-layers))
            final-norm-buf (register-store-entry! :final_norm_w
                                                  (load-fn (str prefix-base "norm.weight") [hidden-dim] norm-enum 0.0))]
        (vec (concat [embed-buf] ple-bufs layer-bufs [final-norm-buf]))))))

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
  [arena-or-ctx buffers]
  (if (arena/arena? arena-or-ctx)
    (arena/destroy! arena-or-ctx buffers)
    (doseq [b buffers]
      (when b (xla/destroy-buffer! arena-or-ctx b)))))
