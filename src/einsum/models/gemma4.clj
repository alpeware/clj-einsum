(ns einsum.models.gemma4
  "Canonical Declarative Gemma 4 Architecture definitions in pure Tensor Logic Hiccup AST."
  (:require [clojure.string :as str]
            [einsum.logic.memory.relation :as mem]))

;; ==============================================================================
;; 1. Configurations
;; ==============================================================================

(def DEFAULT_GEMMA4_E2B_CONFIG
  {:hidden-dim 1536
   :intermediate-dim 6144
   :pl-dim 256
   :total-pl-dim 8960 ;; 35 * 256
   :num-layers 35
   :num-heads 8
   :num-kv-heads 1
   :head-dim 256
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 20
   :layer-types ["sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"
                 "sliding_attention" "sliding_attention" "sliding_attention" "sliding_attention" "full_attention"]})

(def DEFAULT_GEMMA4_E4B_CONFIG
  {:hidden-dim 2560
   :intermediate-dim 10240
   :pl-dim 256
   :total-pl-dim 10752 ;; 42 * 256
   :num-layers 42
   :num-heads 8
   :num-kv-heads 2
   :head-dim 256
   :global-head-dim 512
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 18})

(def DEFAULT_GEMMA4_12B_CONFIG
  {:hidden-dim 3840
   :intermediate-dim 15360
   :pl-dim 0
   :total-pl-dim 0
   :num-layers 48
   :num-heads 16
   :num-kv-heads 8
   :num-global-kv-heads 1
   :head-dim 256
   :global-head-dim 512
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 0})

(def DEFAULT_GEMMA4_31B_CONFIG
  {:hidden-dim 5376
   :intermediate-dim 21504
   :pl-dim 0
   :total-pl-dim 0
   :num-layers 60
   :num-heads 32
   :num-kv-heads 16
   :num-global-kv-heads 4
   :head-dim 256
   :global-head-dim 512
   :vocab-size 262144
   :norm-eps 1e-6
   :final-logit-softcap 30.0
   :num-kv-shared-layers 0
   :sliding-window 1024
   :layer-types (vec (take 60 (cycle ["sliding_attention" "sliding_attention" "sliding_attention"
                                      "sliding_attention" "sliding_attention" "full_attention"])))})

(defn gemma4-config
  "Returns Gemma 4 configuration map for the specified variant (e.g. :e2b, :e4b, :12b, :31b) with optional custom overrides."
  ([] (gemma4-config :e2b {}))
  ([variant-or-overrides]
   (if (keyword? variant-or-overrides)
     (gemma4-config variant-or-overrides {})
     (gemma4-config :e2b variant-or-overrides)))
  ([variant overrides]
   (let [base (case variant
                (:31b :31B) DEFAULT_GEMMA4_31B_CONFIG
                (:12b :12B) DEFAULT_GEMMA4_12B_CONFIG
                (:e4b :e4B) DEFAULT_GEMMA4_E4B_CONFIG
                (:e2b :e2B) DEFAULT_GEMMA4_E2B_CONFIG
                DEFAULT_GEMMA4_E2B_CONFIG)]
     (merge base overrides))))

(defn gemma4-weight-key-map
  "Returns Gemma 4 safetensors weight key mapping for layer `layer-idx` including per-layer input keys."
  ([layer-idx] (gemma4-weight-key-map layer-idx "model.layers."))
  ([layer-idx prefix-base]
   (let [prefix (if (.endsWith ^String prefix-base ".")
                  (str prefix-base layer-idx ".")
                  (str prefix-base "." layer-idx "."))]
     {:input-ln-w             (str prefix "input_layernorm.weight")
      :layer-scalar-w         (str prefix "layer_scalar")
      :q-w                    (str prefix "self_attn.q_proj.weight")
      :k-w                    (str prefix "self_attn.k_proj.weight")
      :v-w                    (str prefix "self_attn.v_proj.weight")
      :o-w                    (str prefix "self_attn.o_proj.weight")
      :q-norm-w               (str prefix "self_attn.q_norm.weight")
      :k-norm-w               (str prefix "self_attn.k_norm.weight")
      :post-attn-ln-w         (str prefix "post_attention_layernorm.weight")
      :pre-mlp-ln-w           (str prefix "pre_feedforward_layernorm.weight")
      :post-mlp-ln-w          (str prefix "post_feedforward_layernorm.weight")
      :gate-w                 (str prefix "mlp.gate_proj.weight")
      :up-w                   (str prefix "mlp.up_proj.weight")
      :down-w                 (str prefix "mlp.down_proj.weight")
      :per-layer-gate-w       (str prefix "per_layer_input_gate.weight")
      :per-layer-proj-w       (str prefix "per_layer_projection.weight")
      :post-per-layer-norm-w  (str prefix "post_per_layer_input_norm.weight")})))

(defn gemma4-alias-resolver
  "Returns a resolver function mapping Gemma 4 Tensor Logic AST variable keywords
   to canonical HuggingFace Safetensors parameter paths under `prefix-base`."
  ([]
   (gemma4-alias-resolver "model."))
  ([prefix-base]
   (let [p-base (if (or (str/ends-with? prefix-base ".") (empty? prefix-base))
                  prefix-base
                  (str prefix-base "."))
         p-layers (if (str/ends-with? p-base "layers.")
                    p-base
                    (str p-base "layers."))]
     (fn [k]
       (let [k-str (if (keyword? k) (name k) (str k))]
         (cond
           (= k-str "embed_tokens") (str p-base "embed_tokens.weight")
           (= k-str "embed_tokens_per_layer") (str p-base "embed_tokens_per_layer.weight")
           (= k-str "per_layer_model_projection") (str p-base "per_layer_model_projection.weight")
           (= k-str "per_layer_projection_norm") (str p-base "per_layer_projection_norm.weight")
           (= k-str "final_norm_w") (str p-base "norm.weight")
           :else
           (if-let [[_ prefix idx-str] (re-matches #"^(input_ln_w|layer_scalar|q_w|q_scale|k_w|k_scale|v_w|v_scale|o_w|o_scale|q_norm_w|k_norm_w|post_attn_ln_w|pre_mlp_ln_w|post_mlp_ln_w|gate_w|gate_scale|up_w|up_scale|down_w|down_scale|per_layer_gate_w|per_layer_proj_w|post_per_layer_norm_w)_(\d+)$" k-str)]
             (let [i (Long/parseLong idx-str)
                   l-prefix (str p-layers i ".")]
               (case prefix
                 "input_ln_w" (str l-prefix "input_layernorm.weight")
                 "layer_scalar" (str l-prefix "layer_scalar")
                 "q_w" (str l-prefix "self_attn.q_proj.weight")
                 "q_scale" (str l-prefix "self_attn.q_proj.weight.scales")
                 "k_w" (str l-prefix "self_attn.k_proj.weight")
                 "k_scale" (str l-prefix "self_attn.k_proj.weight.scales")
                 "v_w" (str l-prefix "self_attn.v_proj.weight")
                 "v_scale" (str l-prefix "self_attn.v_proj.weight.scales")
                 "o_w" (str l-prefix "self_attn.o_proj.weight")
                 "o_scale" (str l-prefix "self_attn.o_proj.weight.scales")
                 "q_norm_w" (str l-prefix "self_attn.q_norm.weight")
                 "k_norm_w" (str l-prefix "self_attn.k_norm.weight")
                 "post_attn_ln_w" (str l-prefix "post_attention_layernorm.weight")
                 "pre_mlp_ln_w" (str l-prefix "pre_feedforward_layernorm.weight")
                 "post_mlp_ln_w" (str l-prefix "post_feedforward_layernorm.weight")
                 "gate_w" (str l-prefix "mlp.gate_proj.weight")
                 "gate_scale" (str l-prefix "mlp.gate_proj.weight.scales")
                 "up_w" (str l-prefix "mlp.up_proj.weight")
                 "up_scale" (str l-prefix "mlp.up_proj.weight.scales")
                 "down_w" (str l-prefix "mlp.down_proj.weight")
                 "down_scale" (str l-prefix "mlp.down_proj.weight.scales")
                 "per_layer_gate_w" (str l-prefix "per_layer_input_gate.weight")
                 "per_layer_proj_w" (str l-prefix "per_layer_projection.weight")
                 "post_per_layer_norm_w" (str l-prefix "post_per_layer_input_norm.weight")
                 nil))
             k-str)))))))

(defn layer-is-global?
  "Determines whether layer `layer-idx` is a full global attention layer or sliding window layer."
  [layer-types layer-idx]
  (if (seq layer-types)
    (let [t (nth layer-types layer-idx nil)]
      (or (= t "full_attention") (= t :full_attention)))
    (zero? (mod (inc layer-idx) 5))))

(defn gemma4-linear-proj
  "Emits a linear projection AST node.
   When is-ternary? is true, emits [:ternary-unpack ...] followed by contraction.
   When is-int4? is true, emits [:int4-unpack ...] followed by contraction.
   When is-int8? is true, emits convert(w_s8 -> norm-dtype) * scale followed by contraction."
  ([out-term x-term w-term is-int8? scale-var norm-dtype]
   (gemma4-linear-proj out-term x-term w-term is-int8? false false scale-var norm-dtype {}))
  ([out-term x-term w-term is-int8? scale-var norm-dtype attrs]
   (gemma4-linear-proj out-term x-term w-term is-int8? false false scale-var norm-dtype attrs))
  ([out-term x-term w-term is-int8? is-int4? scale-var norm-dtype attrs]
   (gemma4-linear-proj out-term x-term w-term is-int8? is-int4? false scale-var norm-dtype attrs))
  ([out-term x-term w-term is-int8? is-int4? is-ternary? scale-var norm-dtype attrs]
   (let [use-w4a16? (and is-int4? (:use-w4a16-gemv? attrs))
         clean-attrs (dissoc attrs :use-w4a16-gemv?)]
     (cond
       is-ternary?
       (let [head-name (first out-term)
             w-name (first w-term)
             w-idxs (vec (rest w-term))
             w-scaled (keyword (str (name w-name) "_scaled"))
             scale-dim (first w-idxs)
             quarter-in-dim (keyword (str (name (second w-idxs)) "_quarter"))
             w-packed-term [w-name scale-dim quarter-in-dim]]
         [:block {:name (keyword (str (name head-name) "_ternary_proj"))}
          [:ternary-unpack (into [w-scaled] w-idxs) w-packed-term [scale-var scale-dim]]
          (cond-> [:= out-term]
            (seq clean-attrs) (conj clean-attrs)
            :always (conj x-term (into [w-scaled] w-idxs)))])

       use-w4a16?
       (let [head-name (first out-term)
             w-name (first w-term)
             w-idxs (vec (rest w-term))
             out-dim (first w-idxs)
             in-dim (second w-idxs)]
         [:block {:name (keyword (str (name head-name) "_w4a16_proj"))}
          (cond-> [:w4a16-gemv out-term x-term [w-name in-dim out-dim] [scale-var out-dim]]
            (seq clean-attrs) (conj clean-attrs))])

       is-int4?
       (let [head-name (first out-term)
             w-name (first w-term)
             w-idxs (vec (rest w-term))
             w-scaled (keyword (str (name w-name) "_scaled"))
             scale-dim (first w-idxs)
             half-in-dim (keyword (str (name (second w-idxs)) "_half"))
             w-packed-term [w-name scale-dim half-in-dim]]
         [:block {:name (keyword (str (name head-name) "_int4_proj"))}
          [:int4-unpack (into [w-scaled] w-idxs) w-packed-term [scale-var scale-dim]]
          (cond-> [:= out-term]
            (seq clean-attrs) (conj clean-attrs)
            :always (conj x-term (into [w-scaled] w-idxs)))])

       is-int8?
       (let [head-name (first out-term)
             w-name (first w-term)
             w-idxs (vec (rest w-term))
             w-bf16 (keyword (str (name w-name) "_bf16"))
             w-scaled (keyword (str (name w-name) "_scaled"))
             scale-dim (first w-idxs)]
         [:block {:name (keyword (str (name head-name) "_int8_proj"))}
          [:convert (into [w-bf16] w-idxs) (into [w-name] w-idxs) {:target-dtype norm-dtype}]
          [:= (into [w-scaled] w-idxs) (into [w-bf16] w-idxs) [scale-var scale-dim]]
          (cond-> [:= out-term]
            (seq clean-attrs) (conj clean-attrs)
            :always (conj x-term (into [w-scaled] w-idxs)))])

       :else
       (cond-> [:= out-term]
         (seq clean-attrs) (conj clean-attrs)
         :always (conj x-term w-term))))))

(defn gemma4-layer-ast
  "Generates Tensor Logic Hiccup AST for Gemma 4 Transformer layer block `layer-idx`."
  ([layer-idx max-seq-len config]
   (gemma4-layer-ast layer-idx max-seq-len config {}))
  ([layer-idx max-seq-len config layer-opts]
   (let [i layer-idx
         {:keys [num-heads num-kv-heads pl-dim total-pl-dim layer-types layer-configs]} config
         is-shared? (:is-shared? layer-opts)
         shared-k (:shared-k layer-opts)
         shared-v (:shared-v layer-opts)
         pl-dim (long (or pl-dim 256))
         total-pl-dim (long (or total-pl-dim (* 35 pl-dim)))
         has-ple? (pos? total-pl-dim)
         cfg (when (seq layer-configs) (nth layer-configs i nil))
         is-global? (if cfg (:is-global? cfg) (layer-is-global? layer-types i))
         head-dim (long (or (:head-dim cfg)
                            (if is-global?
                              (or (:global-head-dim config) 512)
                              (or (:head-dim config) 256))))
         num-heads (long (or (:num-heads cfg) num-heads 8))
         num-kv-heads (long (or (:num-kv-heads cfg)
                                (if is-global?
                                  (or (:num-global-kv-heads config) 1)
                                  num-kv-heads)
                                1))
         q-dim (long (or (:q-dim cfg) (* num-heads head-dim)))
         kv-dim (long (or (:kv-dim cfg) (* num-kv-heads head-dim)))
         group-size (quot num-heads num-kv-heads)
         rope-prop (double (or (:rope-proportion cfg) (if is-global? 0.25 1.0)))
         theta (double (or (:theta-base cfg) (if is-global? 1000000.0 10000.0)))
         window (if is-global? nil (long (or (:sliding-window cfg) (:sliding-window config) (:sliding_window config) 512)))
         layer-seq-len (if window (min max-seq-len window) max-seq-len)
         ring-buffer? (boolean (and window (> max-seq-len window)))

         is-int8? (boolean (or (:is-int8 config) (= (:weight-dtype config) :int8)))
         is-int4? (boolean (or (:is-int4 config) (= (:weight-dtype config) :int4)))
         is-ternary? (boolean (or (:is-ternary config) (= (:weight-dtype config) :ternary) (= (:quant-type config) :ternary)))
         skipped? (contains? (set (:skip-layers config)) i)
         layer-is-int8? (and is-int8? (not skipped?))
         layer-is-int4? (and is-int4? (not skipped?))
         layer-is-ternary? (and is-ternary? (not skipped?))
         use-w4a16? (boolean (and layer-is-int4?
                                  (if (some? (:use-w4a16-gemv config))
                                    (:use-w4a16-gemv config)
                                    (or (= (:backend config) :rocm) (= (:target config) :rocm)))))
         proj-attrs {:use-w4a16-gemv? use-w4a16?}
         norm-dtype (get config :norm-dtype (if (or is-int8? is-int4? is-ternary?) :bf16 (get config :weight-dtype :bf16)))

         actual-k-ro (if is-shared? shared-k [:k_ro i])
         actual-v-heads (if is-shared? shared-v [:v_heads i])]

     [:block {:name [:gemma4_layer i]}
      ;; 1. Pre-Attention RMSNorm
      [:rms-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:input_ln_w :d] {:eps 1e-6}]

      ;; 2. Q Projection & Reshape
      (gemma4-linear-proj [:q_raw :b :p :qd] [:x_norm1 :b :p :d] [:q_w :qd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :q_scale norm-dtype proj-attrs)
      [:reshape [:q_heads_raw :b :p :h :dh] [:q_raw :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]
      [:rms-norm [:q_normed_4d :b :p :h :dh] [:q_heads_raw :b :p :h :dh] [:q_norm_w :dh] {:eps 1e-6}]
      [:reshape [:q_normed_3d :b :p :qd] [:q_normed_4d :b :p :h :dh] {:shape [1 max-seq-len q-dim]}]
      [:rope [:q_rope :b :p :qd] [:q_normed_3d :b :p :qd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
      [:reshape [:q_ro :b :p :h :dh] [:q_rope :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 3. K, V Projections (computed only if not shared)
      (when-not is-shared?
        [:block {:name :kv_proj}
         (gemma4-linear-proj [:k_raw :b :p :kvd] [:x_norm1 :b :p :d] [:k_w :kvd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :k_scale norm-dtype proj-attrs)
         (gemma4-linear-proj [:v_raw :b :p :kvd] [:x_norm1 :b :p :d] [:v_w :kvd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :v_scale norm-dtype proj-attrs)
         [:rms-norm [:v_normed :b :p :kvd] [:v_raw :b :p :kvd] {:eps 1e-6}]
         [:reshape [:k_heads_raw :b :p :kvh :dh] [:k_raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:reshape [[:v_heads i] :b :p :kvh :dh] [:v_normed :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:rms-norm [:k_normed_4d :b :p :kvh :dh] [:k_heads_raw :b :p :kvh :dh] [:k_norm_w :dh] {:eps 1e-6}]
         [:reshape [:k_normed_3d :b :p :kvd] [:k_normed_4d :b :p :kvh :dh] {:shape [1 max-seq-len kv-dim]}]
         [:rope [:k_rope :b :p :kvd] [:k_normed_3d :b :p :kvd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
         [:reshape [[:k_ro i] :b :p :kvh :dh] [:k_rope :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         (when ring-buffer?
           [:slice [[:k_ro_sl i] :b :kvs :kvh :dh] [[:k_ro i] :b :p :kvh :dh]
            {:start [0 0 0 0] :limit [1 layer-seq-len num-kv-heads head-dim]}])
         (when ring-buffer?
           [:slice [[:v_heads_sl i] :b :kvs :kvh :dh] [[:v_heads i] :b :p :kvh :dh]
            {:start [0 0 0 0] :limit [1 layer-seq-len num-kv-heads head-dim]}])])

      ;; 4. Broadcast KV Heads
      [:= [:k_rep :b :p :kvh :g :dh] [actual-k-ro :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [:k_heads :b :p :h :dh] [:k_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]
      [:= [:v_rep :b :p :kvh :g :dh] [actual-v-heads :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [:v_rep_heads :b :p :h :dh] [:v_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 5. Scaled Dot-Product Attention: QK^T -> Softmax (sliding or full) -> probs @ V
      [:= [:scores :b :h :p_q :p_k] {:scale 1.0} [:q_ro :b :p_q :h :dh] [:k_heads :b :p_k :h :dh]]
      [:causal-softmax [:probs :b :h :p_q :p_k] [:scores :b :h :p_q :p_k] (if window {:sliding-window window} {})]
      [:= [:ctx :b :p_q :h :dh] [:probs :b :h :p_q :p_k] [:v_rep_heads :b :p_k :h :dh]]
      [:reshape [:ctx_flat :b :p :qd] [:ctx :b :p_q :h :dh] {:shape [1 max-seq-len q-dim]}]

      ;; 6. Output Projection & Post-Attention RMSNorm
      (gemma4-linear-proj [:attn_raw :b :p :d] [:ctx_flat :b :p :qd] [:o_w :d :qd] layer-is-int8? layer-is-int4? layer-is-ternary? :o_scale norm-dtype proj-attrs)
      [:rms-norm [:attn_normed :b :p :d] [:attn_raw :b :p :d] [:post_attn_ln_w :d] {:eps 1e-6}]

      ;; 7. Residual Connection 1
      [:= [:res1 :b :p :d] [:h :b :p :d]]
      [:= [:res1 :b :p :d] [:attn_normed :b :p :d]]

      ;; 8. Pre-MLP RMSNorm
      [:rms-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:pre_mlp_ln_w :d] {:eps 1e-6}]

      ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
      (gemma4-linear-proj [:gate :b :p :dff] [:x_norm2 :b :p :d] [:gate_w :dff :d] layer-is-int8? layer-is-int4? layer-is-ternary? :gate_scale norm-dtype (assoc proj-attrs :act :gelu))
      (gemma4-linear-proj [:up :b :p :dff] [:x_norm2 :b :p :d] [:up_w :dff :d] layer-is-int8? layer-is-int4? layer-is-ternary? :up_scale norm-dtype proj-attrs)
      [:= [:mlp_act :b :p :dff] [:gate :b :p :dff] [:up :b :p :dff]]
      (gemma4-linear-proj [:mlp_raw :b :p :d] [:mlp_act :b :p :dff] [:down_w :d :dff] layer-is-int8? layer-is-int4? layer-is-ternary? :down_scale norm-dtype proj-attrs)
      [:rms-norm [:mlp_normed :b :p :d] [:mlp_raw :b :p :d] [:post_mlp_ln_w :d] {:eps 1e-6}]

      ;; 10. Residual Connection 2
      [:= [:res2 :b :p :d] [:res1 :b :p :d]]
      [:= [:res2 :b :p :d] [:mlp_normed :b :p :d]]

      ;; 11. Gemma 4 Per-Layer Input (PLE) Gating Sub-block (if has-ple?)
      (when has-ple?
        [:block {:name :ple_gate}
         [:= [:gate_raw :b :p :pld] {:act :gelu} [:res2 :b :p :d] [:per_layer_gate_w :pld :d]]
         [:= [:gated :b :p :pld] [:gate_raw :b :p :pld] [[:pl_in i] :b :p :pld]]
         [:= [:proj_raw :b :p :d] [:gated :b :p :pld] [:per_layer_proj_w :d :pld]]
         [:rms-norm [:ple_normed :b :p :d] [:proj_raw :b :p :d] [:post_per_layer_norm_w :d] {:eps 1e-6}]
         [:= [:res3 :b :p :d] [:res2 :b :p :d]]
         [:= [:res3 :b :p :d] [:ple_normed :b :p :d]]])

      ;; 12. Gemma 4 Layer Scalar
      [:= [:h# :b :p :d] [(if has-ple? :res3 :res2) :b :p :d] [:layer_scalar :one]]])))

(defn gemma4-model-ast
  "Generates full Gemma 4 model forward pass in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge (gemma4-config :e2b) config)
        {:keys [num-layers max-seq-len hidden-dim pl-dim total-pl-dim final-logit-softcap num-kv-shared-layers layer-types]} cfg
        num-layers (long (or num-layers 35))
        num-kv-shared (long (or num-kv-shared-layers 0))
        num-unshared (- num-layers num-kv-shared)
        has-shared-kv? (and (pos? num-unshared) (pos? num-kv-shared))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 1536))
        pl-dim (long (or pl-dim 256))
        total-pl-dim (long (or total-pl-dim (* num-layers pl-dim)))
        has-ple? (pos? total-pl-dim)

        last-unshared-sliding (when has-shared-kv? (last (filter #(not (layer-is-global? layer-types %)) (range num-unshared))))
        last-unshared-full (when has-shared-kv? (last (filter #(layer-is-global? layer-types %) (range num-unshared))))]
    [:block {:name :full_gemma4_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [[:h 0] :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Gemma 4 Per-Layer Embedding (PLE) generation
     (when has-ple?
       [:block {:name :ple_generation}
        [:gather [:raw_pl_tok :b :p :total_pl_dim] [:embed_tokens_per_layer :v :total_pl_dim] [:x :b :p]]
        [:= [:pl_tok_scaled :b :p :total_pl_dim] {:scale 16.0} [:raw_pl_tok :b :p :total_pl_dim]]
        [:= [:pl_context_raw :b :p :total_pl_dim] [[:h 0] :b :p :d] [:per_layer_model_projection :total_pl_dim :d]]
        [:reshape [:pl_tok_4d :b :p :l :pld] [:pl_tok_scaled :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:reshape [:pl_context_4d :b :p :l :pld] [:pl_context_raw :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:rms-norm [:pl_context_norm :b :p :l :pld] [:pl_context_4d :b :p :l :pld] [:per_layer_projection_norm :pld] {:eps 1e-6}]
        [:= [:pl_sum :b :p :l :pld] [:pl_context_norm :b :p :l :pld]]
        [:= [:pl_sum :b :p :l :pld] [:pl_tok_4d :b :p :l :pld]]
        [:= [:ple_all :b :p :l :pld] {:scale (/ 1.0 (Math/sqrt 2.0))} [:pl_sum :b :p :l :pld]]
        ;; Layer PLE Slices
        (mapv (fn [i]
                [:block {:name (keyword (str "ple_slice_block_" i))}
                 [:slice [[:pl_slice i] :b :p :one :pld] [:ple_all :b :p :l :pld] {:start [0 0 i 0] :limit [1 max-seq-len (inc i) pl-dim]}]
                 [:reshape [[:pl_in i] :b :p :pld] [[:pl_slice i] :b :p :one :pld] {:shape [1 max-seq-len pl-dim]}]])
              (range num-layers))])

     ;; 3. Sequential Transformer Layer Blocks
     (mapv (fn [i]
             (let [is-shared? (and has-shared-kv? (>= i num-unshared))
                   is-global? (layer-is-global? layer-types i)
                   shared-k (when is-shared?
                              (if is-global?
                                [:k_ro last-unshared-full]
                                [:k_ro last-unshared-sliding]))
                   shared-v (when is-shared?
                              (if is-global?
                                [:v_heads last-unshared-full]
                                [:v_heads last-unshared-sliding]))]
               (gemma4-layer-ast i max-seq-len cfg {:is-shared? is-shared?
                                                    :shared-k shared-k
                                                    :shared-v shared-v})))
           (range num-layers))

     ;; 4. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [[:h num-layers] :b :p :d] [:final_norm_w :d] {:eps 1e-6}]

     ;; 5. Tied LM Head with optional final logit softcapping and relational grounding
     (let [has-rel-mem? (boolean (or (:relational-memory cfg) (:relational-memory? cfg)))
           rel-cfg (or (:relational-memory cfg) {})
           mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
           entity-count (long (or (:entity-count rel-cfg) 1000))
           vocab-size (long (or (:vocab-size cfg) 262144))
           norm-dtype (if (or (:is-int8 cfg) (:is-int4 cfg) (:is-ternary cfg) (= (:weight-dtype cfg) :ternary)) :bf16 (or (:weight-dtype cfg) :f32))]
       (if (:last-token-only? cfg)
         [:block {:name :last_token_head}
          [:dynamic-slice [:normed_last :b :one :d] [:normed :b :p :d]
           {:slice-sizes [1 1 hidden-dim]
            :start-indices [0 :pos 0]}]
          (if has-rel-mem?
            [:block {:name :relational_grounded_head}
             [:= [:raw_logits :b :one :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                            {:softcap (double final-logit-softcap)}
                                            {})
              [:normed_last :b :one :d] [:embed_tokens :v :d]]
             (mem/relational-grounding-ast hidden-dim mem-dim vocab-size entity-count {:dtype norm-dtype :sequence? false})
             [:= [:logits :b :one :v] [:logits_grounded :b :one :v]]]
            [:= [:logits :b :one :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                       {:softcap (double final-logit-softcap)}
                                       {})
             [:normed_last :b :one :d] [:embed_tokens :v :d]])]
         (if has-rel-mem?
           [:block {:name :relational_grounded_seq_head}
            [:= [:raw_logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                         {:softcap (double final-logit-softcap)}
                                         {})
             [:normed :b :p :d] [:embed_tokens :v :d]]
            (mem/relational-grounding-ast hidden-dim mem-dim vocab-size entity-count {:dtype norm-dtype :sequence? true})
            [:= [:logits :b :p :v] [:logits_grounded :b :p :v]]]
           [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                    {:softcap (double final-logit-softcap)}
                                    {})
            [:normed :b :p :d] [:embed_tokens :v :d]])))]))

(defn gemma4-relational-invars
  "Returns the relational memory invars definitions matching relational-grounding-ast."
  ([config]
   (gemma4-relational-invars config :f32))
  ([config dtype]
   (let [rel-cfg (or (:relational-memory config) {})
         hidden-dim (long (or (:hidden-dim config) 1536))
         mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
         entity-count (long (or (:entity-count rel-cfg) 1000))
         vocab-size (long (or (:vocab-size config) 262144))]
     [[:w_mem_proj [:tensor [hidden-dim mem-dim] dtype]]
      [:r_active [:tensor [mem-dim mem-dim] dtype]]
      [:entity_table [:tensor [entity-count mem-dim] dtype]]
      [:threshold_const [:tensor [1 1 entity-count] dtype]]
      [:w_entity_to_vocab [:tensor [entity-count vocab-size] dtype]]])))

(defn gemma4-unbinding-ast
  "Constructs Tensor Logic AST for linear relational memory unbinding and vocabulary projection."
  ([_dim _dm _vocab-size]
   (gemma4-unbinding-ast _dim _dm _vocab-size 1.0))
  ([_dim _dm _vocab-size lambda-mem]
   (let [scale-factor (double (or lambda-mem 1.0))]
     [:block {:name :gemma_unbinding_logits}
      [:= [:u_q :b :one :dm] [:h :b :one :din] [:W_mem :din :dm]]
      [:= [:u_target :b :one :dm2] [:u_q :b :one :dm1] [:R :dm1 :dm2]]
      [:rms-norm [:u_norm :b :one :dm2] [:u_target :b :one :dm2]]
      [:= [:v_bias :b :one :din] [:u_norm :b :one :dm] [:W_mem :din :dm]]
      [:= [:delta_logits :b :one :v] [:v_bias :b :one :din] [:W_embed :v :din]]
      [:= [:scaled_delta :b :one :v] {:scale scale-factor} [:delta_logits :b :one :v]]
      [:+ [:logits_grounded :b :one :v] [:logits_base :b :one :v] [:scaled_delta :b :one :v]]])))

(defn gemma4-entity-unbinding-ast
  "Constructs Tensor Logic AST for direct subject entity unbinding."
  ([_dim _dm _vocab-size]
   (gemma4-entity-unbinding-ast _dim _dm _vocab-size 1.0))
  ([_dim _dm _vocab-size lambda-mem]
   (let [scale-factor (double (or lambda-mem 1.0))]
     [:block {:name :gemma_entity_unbinding_logits}
      [:gather [:v_h :b :one :din] [:W_embed :v :din] [:i_h :b :one]]
      [:= [:u_q :b :one :dm] [:v_h :b :one :din] [:W_mem :din :dm]]
      [:= [:u_target :b :one :dm2] [:u_q :b :one :dm1] [:R :dm1 :dm2]]
      [:rms-norm [:u_norm :b :one :dm2] [:u_target :b :one :dm2]]
      [:= [:v_bias :b :one :din] [:u_norm :b :one :dm] [:W_mem :din :dm]]
      [:= [:delta_logits :b :one :v] [:v_bias :b :one :din] [:W_embed :v :din]]
      [:= [:scaled_delta :b :one :v] {:scale scale-factor} [:delta_logits :b :one :v]]
      [:+ [:logits_grounded :b :one :v] [:logits_base :b :one :v] [:scaled_delta :b :one :v]]])))

(defn gemma4-two-stage-resolver-ast
  "Constructs Tensor Logic AST for Stage 2 Non-Linear Entity Resolver."
  [_dim dm _vocab-size _k-cands]
  (let [inv-scale (/ 1.0 (Math/sqrt (double (or dm 128))))]
    [:block {:name :gemma_two_stage_resolver}
     [:gather [:E_cand :k :din] [:W_embed :v :din] [:cand_ids :k]]
     [:= [:q_raw :b :one :dm] [:h :b :one :din] [:W_Q :din :dm]]
     [:* [:q_mod :b :one :dm] [:q_raw :b :one :dm] [:u_norm :b :one :dm]]
     [:rms-norm [:q_rel :b :one :dm] [:q_mod :b :one :dm]]
     [:= [:k_raw :k :dm] [:E_cand :k :din] [:W_K :din :dm]]
     [:rms-norm [:k_cand :k :dm] [:k_raw :k :dm]]
     [:= [:scores :b :one :k] {:scale inv-scale} [:q_rel :b :one :dm] [:k_cand :k :dm]]]))

(defn gemma4-two-stage-entity-resolver-ast
  "Constructs Tensor Logic AST for Stage 2 Direct-Entity Non-Linear Resolver."
  [_dim dm _vocab-size _k-cands]
  (let [inv-scale (/ 1.0 (Math/sqrt (double (or dm 128))))]
    [:block {:name :gemma_two_stage_entity_resolver}
     [:gather [:v_h :b :one :din] [:W_embed :v :din] [:i_h :b :one]]
     [:gather [:E_cand :k :din] [:W_embed :v :din] [:cand_ids :k]]
     [:= [:q_raw :b :one :dm] [:v_h :b :one :din] [:W_Q :din :dm]]
     [:* [:q_mod :b :one :dm] [:q_raw :b :one :dm] [:u_norm :b :one :dm]]
     [:rms-norm [:q_rel :b :one :dm] [:q_mod :b :one :dm]]
     [:= [:k_raw :k :dm] [:E_cand :k :din] [:W_K :din :dm]]
     [:rms-norm [:k_cand :k :dm] [:k_raw :k :dm]]
     [:= [:scores :b :one :k] {:scale inv-scale} [:q_rel :b :one :dm] [:k_cand :k :dm]]]))

(defn gemma4-prefill-outvars
  "Constructs output variable list for Gemma 4 prefill: [:logits (k_ro_sl_i | k_ro_i) (v_heads_sl_i | v_heads_i) ...]."
  ([config] (gemma4-prefill-outvars config (:max-seq-len config)))
  ([config max-seq-len]
   (let [max-seq-len (long (or max-seq-len (:max-seq-len config) 128))
         num-layers (long (or (:num-layers config) 35))
         num-kv-shared (long (or (:num-kv-shared-layers config) 0))
         num-unshared (- num-layers num-kv-shared)
         layer-configs (:layer-configs config)
         layer-types (:layer-types config)
         kv-outs (mapcat (fn [i]
                           (let [c (if (seq layer-configs) (nth layer-configs i nil) nil)
                                 is-global? (if c (:is-global? c) (layer-is-global? layer-types i))
                                 win (when-not is-global? (or (:sliding-window c) (:sliding-window config) (:sliding_window config) 512))
                                 ring-buffer? (boolean (and win (> max-seq-len win)))]
                             (if ring-buffer?
                               [(keyword (str "k_ro_sl_" i))
                                (keyword (str "v_heads_sl_" i))]
                               [(keyword (str "k_ro_" i))
                                (keyword (str "v_heads_" i))])))
                         (range num-unshared))]
     (vec (into [:logits] kv-outs)))))

(defn gemma4-kv-layer-ast
  "Generates Tensor Logic Hiccup AST for Gemma 4 Transformer layer `layer-idx` using KV Cache.
   Expects single-token input at index `p=1` and cached keys/values of shape [1 max-seq-len num-kv-heads head-dim]."
  ([layer-idx max-seq-len config]
   (gemma4-kv-layer-ast layer-idx max-seq-len config {}))
  ([layer-idx max-seq-len config layer-opts]
   (let [i layer-idx
         {:keys [num-heads num-kv-heads pl-dim total-pl-dim layer-types layer-configs]} config
         is-shared? (:is-shared? layer-opts)
         shared-k (:shared-k layer-opts)
         shared-v (:shared-v layer-opts)
         pl-dim (long (or pl-dim 256))
         total-pl-dim (long (or total-pl-dim (* 35 pl-dim)))
         has-ple? (pos? total-pl-dim)
         cfg (when (seq layer-configs) (nth layer-configs i nil))
         is-global? (if cfg (:is-global? cfg) (layer-is-global? layer-types i))
         head-dim (long (or (:head-dim cfg)
                            (if is-global?
                              (or (:global-head-dim config) 512)
                              (or (:head-dim config) 256))))
         num-heads (long (or (:num-heads cfg) num-heads 8))
         num-kv-heads (long (or (:num-kv-heads cfg)
                                (if is-global?
                                  (or (:num-global-kv-heads config) 1)
                                  num-kv-heads)
                                1))
         q-dim (long (or (:q-dim cfg) (* num-heads head-dim)))
         kv-dim (long (or (:kv-dim cfg) (* num-kv-heads head-dim)))
         _group-size (quot num-heads num-kv-heads)
         rope-prop (double (or (:rope-proportion cfg) (if is-global? 0.25 1.0)))
         theta (double (or (:theta-base cfg) (if is-global? 1000000.0 10000.0)))
         window (if is-global? nil (long (or (:sliding-window cfg) (:sliding-window config) (:sliding_window config) 512)))
         layer-seq-len (if window (min max-seq-len window) max-seq-len)
         ring-buffer? (boolean (and window (> max-seq-len window)))

         is-int8? (boolean (or (:is-int8 config) (= (:weight-dtype config) :int8)))
         is-int4? (boolean (or (:is-int4 config) (= (:weight-dtype config) :int4)))
         is-ternary? (boolean (or (:is-ternary config) (= (:weight-dtype config) :ternary) (= (:quant-type config) :ternary)))
         skipped? (contains? (set (:skip-layers config)) i)
         layer-is-int8? (and is-int8? (not skipped?))
         layer-is-int4? (and is-int4? (not skipped?))
         layer-is-ternary? (and is-ternary? (not skipped?))
         use-w4a16? (boolean (and layer-is-int4?
                                  (if (some? (:use-w4a16-gemv config))
                                    (:use-w4a16-gemv config)
                                    (or (= (:backend config) :rocm) (= (:target config) :rocm)))))
         proj-attrs {:use-w4a16-gemv? use-w4a16?}
         norm-dtype (get config :norm-dtype (if (or is-int8? is-int4? is-ternary?) :bf16 (get config :weight-dtype :bf16)))

         actual-k-cache (if is-shared? shared-k [:k_cache_out i])
         actual-v-cache (if is-shared? shared-v [:v_cache_out i])]

     [:block {:name [:gemma4_kv_layer i]}
      ;; 1. Pre-Attention RMSNorm
      [:rms-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:input_ln_w :d] {:eps 1e-6}]

      ;; 2. Q Projection & Reshape for single token
      (gemma4-linear-proj [:q_raw :b :p :qd] [:x_norm1 :b :p :d] [:q_w :qd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :q_scale norm-dtype proj-attrs)
      [:reshape [:q_heads_raw :b :p :h :dh] [:q_raw :b :p :qd] {:shape [1 1 num-heads head-dim]}]
      [:rms-norm [:q_normed_4d :b :p :h :dh] [:q_heads_raw :b :p :h :dh] [:q_norm_w :dh] {:eps 1e-6}]
      [:reshape [:q_normed_3d :b :p :qd] [:q_normed_4d :b :p :h :dh] {:shape [1 1 q-dim]}]
      [:rope [:q_rope :b :p :qd] [:q_normed_3d :b :p :qd] {:head-dim head-dim :theta theta :rope-proportion rope-prop :pos :pos :max-seq-len max-seq-len}]
      [:reshape [:q_ro :b :p :h :dh] [:q_rope :b :p :qd] {:shape [1 1 num-heads head-dim]}]

      ;; 3. K, V Projections & Dynamic Cache Update (if not shared)
      (when-not is-shared?
        [:block {:name :kv_proj_update}
         (gemma4-linear-proj [:k_raw :b :p :kvd] [:x_norm1 :b :p :d] [:k_w :kvd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :k_scale norm-dtype proj-attrs)
         (gemma4-linear-proj [:v_raw :b :p :kvd] [:x_norm1 :b :p :d] [:v_w :kvd :d] layer-is-int8? layer-is-int4? layer-is-ternary? :v_scale norm-dtype proj-attrs)
         [:rms-norm [:v_normed :b :p :kvd] [:v_raw :b :p :kvd] {:eps 1e-6}]
         [:reshape [:k_heads_raw :b :p :kvh :dh] [:k_raw :b :p :kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:reshape [:v_heads :b :p :kvh :dh] [:v_normed :b :p :kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:rms-norm [:k_normed_4d :b :p :kvh :dh] [:k_heads_raw :b :p :kvh :dh] [:k_norm_w :dh] {:eps 1e-6}]
         [:reshape [:k_normed_3d :b :p :kvd] [:k_normed_4d :b :p :kvh :dh] {:shape [1 1 kv-dim]}]
         [:rope [:k_rope :b :p :kvd] [:k_normed_3d :b :p :kvd] {:head-dim head-dim :theta theta :rope-proportion rope-prop :pos :pos :max-seq-len max-seq-len}]
         [:reshape [:k_ro :b :p :kvh :dh] [:k_rope :b :p :kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:dynamic-update-slice [[:k_cache_out i] :b :kvs :kvh :dh] [:k_cache_in :b :kvs :kvh :dh] [:k_ro :b :p :kvh :dh]
          (merge {:start-indices [0 :pos 0 0]}
                 (when ring-buffer? {:window window}))]
         [:dynamic-update-slice [[:v_cache_out i] :b :kvs :kvh :dh] [:v_cache_in :b :kvs :kvh :dh] [:v_heads :b :p :kvh :dh]
          (merge {:start-indices [0 :pos 0 0]}
                 (when ring-buffer? {:window window}))]])

      ;; 4 & 5. Chunked Scaled Dot-Product Attention (Online Streaming Softmax)
      [:chunked-attention [:ctx :b :p :h :dh]
       [:q_ro :b :p :h :dh]
       [actual-k-cache :b :kvs :kvh :dh]
       [actual-v-cache :b :kvs :kvh :dh]
       (merge {:pos :pos
               :chunk-size (min 64 layer-seq-len)
               :head-dim head-dim
               :num-heads num-heads
               :num-kv-heads num-kv-heads
               :max-seq-len layer-seq-len
               :shape [1 1 num-heads head-dim]}
              (when ring-buffer? {:ring-buffer true :sliding-window window})
              (when (and window (not ring-buffer?)) {:sliding-window window}))]
      [:reshape [:ctx_flat :b :p :qd] [:ctx :b :p :h :dh] {:shape [1 1 q-dim]}]

      ;; 6. Output Projection & Post-Attention RMSNorm
      (gemma4-linear-proj [:attn_raw :b :p :d] [:ctx_flat :b :p :qd] [:o_w :d :qd] layer-is-int8? layer-is-int4? layer-is-ternary? :o_scale norm-dtype proj-attrs)
      [:rms-norm [:attn_normed :b :p :d] [:attn_raw :b :p :d] [:post_attn_ln_w :d] {:eps 1e-6}]

      ;; 7. Residual Connection 1
      [:= [:res1 :b :p :d] [:h :b :p :d]]
      [:= [:res1 :b :p :d] [:attn_normed :b :p :d]]

      ;; 8. Pre-MLP RMSNorm
      [:rms-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:pre_mlp_ln_w :d] {:eps 1e-6}]

      ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
      (gemma4-linear-proj [:gate :b :p :dff] [:x_norm2 :b :p :d] [:gate_w :dff :d] layer-is-int8? layer-is-int4? layer-is-ternary? :gate_scale norm-dtype (assoc proj-attrs :act :gelu))
      (gemma4-linear-proj [:up :b :p :dff] [:x_norm2 :b :p :d] [:up_w :dff :d] layer-is-int8? layer-is-int4? layer-is-ternary? :up_scale norm-dtype proj-attrs)
      [:= [:mlp_act :b :p :dff] [:gate :b :p :dff] [:up :b :p :dff]]
      (gemma4-linear-proj [:mlp_raw :b :p :d] [:mlp_act :b :p :dff] [:down_w :d :dff] layer-is-int8? layer-is-int4? layer-is-ternary? :down_scale norm-dtype proj-attrs)
      [:rms-norm [:mlp_normed :b :p :d] [:mlp_raw :b :p :d] [:post_mlp_ln_w :d] {:eps 1e-6}]

      ;; 10. Residual Connection 2
      [:= [:res2 :b :p :d] [:res1 :b :p :d]]
      [:= [:res2 :b :p :d] [:mlp_normed :b :p :d]]

      ;; 11. Gemma 4 Per-Layer Input (PLE) Gating Sub-block (if has-ple?)
      (when has-ple?
        [:block {:name :ple_gate}
         [:= [:gate_raw :b :p :pld] {:act :gelu} [:res2 :b :p :d] [:per_layer_gate_w :pld :d]]
         [:= [:gated :b :p :pld] [:gate_raw :b :p :pld] [[:pl_in i] :b :p :pld]]
         [:= [:proj_raw :b :p :d] [:gated :b :p :pld] [:per_layer_proj_w :d :pld]]
         [:rms-norm [:ple_normed :b :p :d] [:proj_raw :b :p :d] [:post_per_layer_norm_w :d] {:eps 1e-6}]
         [:= [:res3 :b :p :d] [:res2 :b :p :d]]
         [:= [:res3 :b :p :d] [:ple_normed :b :p :d]]])

      ;; 12. Gemma 4 Layer Scalar
      [:= [:h# :b :p :d] [(if has-ple? :res3 :res2) :b :p :d] [:layer_scalar :one]]])))

(defn gemma4-kv-model-ast
  "Generates single-token step Gemma 4 model forward pass with persistent KV-Cache in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge (gemma4-config :e2b) config)
        {:keys [num-layers max-seq-len hidden-dim pl-dim total-pl-dim final-logit-softcap num-kv-shared-layers layer-types]} cfg
        num-layers (long (or num-layers 35))
        num-kv-shared (long (or num-kv-shared-layers 0))
        num-unshared (- num-layers num-kv-shared)
        has-shared-kv? (and (pos? num-unshared) (pos? num-kv-shared))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 1536))
        pl-dim (long (or pl-dim 256))
        total-pl-dim (long (or total-pl-dim (* num-layers pl-dim)))
        has-ple? (pos? total-pl-dim)

        last-unshared-sliding (when has-shared-kv? (last (filter #(not (layer-is-global? layer-types %)) (range num-unshared))))
        last-unshared-full (when has-shared-kv? (last (filter #(layer-is-global? layer-types %) (range num-unshared))))]
    [:block {:name :gemma4_kv_step_model}
     ;; 1. Token Embedding Lookup for single token [1 1]
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [[:h 0] :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Gemma 4 Per-Layer Embedding (PLE) generation at p=1
     (when has-ple?
       [:block {:name :ple_generation}
        [:gather [:raw_pl_tok :b :p :total_pl_dim] [:embed_tokens_per_layer :v :total_pl_dim] [:x :b :p]]
        [:= [:pl_tok_scaled :b :p :total_pl_dim] {:scale 16.0} [:raw_pl_tok :b :p :total_pl_dim]]
        [:= [:pl_context_raw :b :p :total_pl_dim] [[:h 0] :b :p :d] [:per_layer_model_projection :total_pl_dim :d]]
        [:reshape [:pl_tok_4d :b :p :l :pld] [:pl_tok_scaled :b :p :total_pl_dim] {:shape [1 1 num-layers pl-dim]}]
        [:reshape [:pl_context_4d :b :p :l :pld] [:pl_context_raw :b :p :total_pl_dim] {:shape [1 1 num-layers pl-dim]}]
        [:rms-norm [:pl_context_norm :b :p :l :pld] [:pl_context_4d :b :p :l :pld] [:per_layer_projection_norm :pld] {:eps 1e-6}]
        [:= [:pl_sum :b :p :l :pld] [:pl_context_norm :b :p :l :pld]]
        [:= [:pl_sum :b :p :l :pld] [:pl_tok_4d :b :p :l :pld]]
        [:= [:ple_all :b :p :l :pld] {:scale (/ 1.0 (Math/sqrt 2.0))} [:pl_sum :b :p :l :pld]]
        (mapv (fn [i]
                [:block {:name (keyword (str "ple_slice_block_" i))}
                 [:slice [[:pl_slice i] :b :p :one :pld] [:ple_all :b :p :l :pld] {:start [0 0 i 0] :limit [1 1 (inc i) pl-dim]}]
                 [:reshape [[:pl_in i] :b :p :pld] [[:pl_slice i] :b :p :one :pld] {:shape [1 1 pl-dim]}]])
              (range num-layers))])

     ;; 3. Sequential Transformer Layer Blocks with KV Cache
     (mapv (fn [i]
             (let [is-shared? (and has-shared-kv? (>= i num-unshared))
                   is-global? (layer-is-global? layer-types i)
                   shared-k (when is-shared?
                              (if is-global?
                                [:k_cache_out last-unshared-full]
                                [:k_cache_out last-unshared-sliding]))
                   shared-v (when is-shared?
                              (if is-global?
                                [:v_cache_out last-unshared-full]
                                [:v_cache_out last-unshared-sliding]))]
               (gemma4-kv-layer-ast i max-seq-len cfg {:is-shared? is-shared?
                                                       :shared-k shared-k
                                                       :shared-v shared-v})))
           (range num-layers))

     ;; 4. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [[:h num-layers] :b :p :d] [:final_norm_w :d] {:eps 1e-6}]

     ;; 5. Tied LM Head (single token logits [1 1 vocab-size])
     [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                              {:softcap (double final-logit-softcap)}
                              {})
      [:normed :b :p :d] [:embed_tokens :v :d]]]))

;; ==============================================================================
;; 2. Canonical Signature Builders (Invars and Outvars)
;; ==============================================================================

(defn build-tensor-logic-invars
  "Constructs EDN SSA signature invars for full Gemma 4 model forward pass."
  [config max-seq-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 is-int4 is-ternary layer-configs last-token-only? group-size backend target]} config
        int4? (boolean (or is-int4 (= weight-dtype :int4)))
        is-ternary (boolean (or is-ternary (= weight-dtype :ternary) (= (:quant-type config) :ternary)))
        norm-dtype (if (or is-int8 int4? is-ternary) :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        use-w4a16? (and int4?
                        (if (some? (:use-w4a16-gemv config))
                          (boolean (:use-w4a16-gemv config))
                          (or (= backend :rocm) (= target :rocm))))
        scale-groups-fn (fn [in-dim]
                          (if (and group-size (zero? (mod in-dim group-size)))
                            (quot in-dim group-size)
                            1))
        scale-shape-fn (fn [rows cols]
                         (cond
                           (and int4? group-size (zero? (mod cols group-size)))
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
                           (let [cfg (when (seq layer-configs) (nth layer-configs i nil))
                                 is-global? (if cfg (:is-global? cfg) (layer-is-global? (:layer-types config) i))
                                 head-dim (long (or (:head-dim cfg)
                                                    (if is-global?
                                                      (or (:global-head-dim config) 512)
                                                      (or (:head-dim config) 256))))
                                 num-h (long (or (:num-heads cfg) (:num-heads config) 8))
                                 num-kv (long (or (:num-kv-heads cfg)
                                                  (if is-global?
                                                    (or (:num-global-kv-heads config) 1)
                                                    (:num-kv-heads config))
                                                  1))
                                 q-dim (long (or (:q-dim cfg) (* num-h head-dim)))
                                 kv-dim (long (or (:kv-dim cfg) (* num-kv head-dim)))
                                 mlp-dim (long (or (:mlp-dim cfg) (:intermediate-dim config) 6144))
                                 skipped? (contains? (set (:skip-layers config)) i)
                                 layer-is-ternary (and is-ternary (not skipped?))
                                 layer-is-int4 (and int4? (not skipped?))
                                 layer-is-int8 (and is-int8 (not skipped?))
                                 layer-use-w4a16 (and use-w4a16? (not skipped?))]
                             (concat
                              [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "layer_scalar_" i)) [:tensor [1] norm-dtype]]]
                              (cond
                                layer-is-ternary
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 4)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                layer-use-w4a16
                                [[(keyword (str "q_w_" i)) [:tensor [(quot hidden-dim 8) q-dim] :i32]]
                                 [(keyword (str "q_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) q-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [(quot hidden-dim 8) kv-dim] :i32]]
                                 [(keyword (str "k_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) kv-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [(quot hidden-dim 8) kv-dim] :i32]]
                                 [(keyword (str "v_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) kv-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [(quot q-dim 8) hidden-dim] :i32]]
                                 [(keyword (str "o_scale_" i)) [:tensor [(scale-groups-fn q-dim) hidden-dim] norm-dtype]]]
                                layer-is-int4
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 2)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                layer-is-int8
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor [q-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] norm-dtype]]])
                              [[(keyword (str "q_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]]
                              (cond
                                layer-is-ternary
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 4)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                layer-use-w4a16
                                [[(keyword (str "gate_w_" i)) [:tensor [(quot hidden-dim 8) mlp-dim] :i32]]
                                 [(keyword (str "gate_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) mlp-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [(quot hidden-dim 8) mlp-dim] :i32]]
                                 [(keyword (str "up_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) mlp-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [(quot mlp-dim 8) hidden-dim] :i32]]
                                 [(keyword (str "down_scale_" i)) [:tensor [(scale-groups-fn mlp-dim) hidden-dim] norm-dtype]]]
                                layer-is-int4
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 2)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                layer-is-int8
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] norm-dtype]]])
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
                                  is-global? (if cfg (:is-global? cfg) (layer-is-global? layer-types i))
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
  (gemma4-prefill-outvars config (:max-seq-len config)))
