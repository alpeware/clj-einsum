(ns einsum.models.gemma4
  "Canonical Declarative Gemma 4 Architecture definitions in pure Tensor Logic Hiccup AST."
  (:require [einsum.logic.memory.relation :as mem]))

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

(defn layer-is-global?
  "Determines whether layer `layer-idx` is a full global attention layer or sliding window layer."
  [layer-types layer-idx]
  (if (seq layer-types)
    (let [t (nth layer-types layer-idx nil)]
      (or (= t "full_attention") (= t :full_attention)))
    (zero? (mod (inc layer-idx) 5))))

(defn gemma4-linear-proj
  "Emits a linear projection AST node.
   When is-int4? is true, emits [:int4-unpack ...] followed by contraction.
   When is-int8? is true, emits convert(w_s8 -> norm-dtype) * scale followed by contraction."
  ([out-term x-term w-term is-int8? scale-var norm-dtype]
   (gemma4-linear-proj out-term x-term w-term is-int8? false scale-var norm-dtype {}))
  ([out-term x-term w-term is-int8? scale-var norm-dtype attrs]
   (gemma4-linear-proj out-term x-term w-term is-int8? false scale-var norm-dtype attrs))
  ([out-term x-term w-term is-int8? is-int4? scale-var norm-dtype attrs]
   (cond
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
          (seq attrs) (conj attrs)
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
          (seq attrs) (conj attrs)
          :always (conj x-term (into [w-scaled] w-idxs)))])

     :else
     (cond-> [:= out-term]
       (seq attrs) (conj attrs)
       :always (conj x-term w-term)))))

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
         norm-dtype (get config :norm-dtype (if (or is-int8? is-int4?) :bf16 (get config :weight-dtype :bf16)))

         h-in (keyword (str "h" i))
         h-out (keyword (str "h" (inc i)))

         input-ln-w (keyword (str "input_ln_w_" i))
         layer-scalar-w (keyword (str "layer_scalar_" i))
         q-w (keyword (str "q_w_" i))
         k-w (keyword (str "k_w_" i))
         v-w (keyword (str "v_w_" i))
         o-w (keyword (str "o_w_" i))
         q-norm-w (keyword (str "q_norm_w_" i))
         k-norm-w (keyword (str "k_norm_w_" i))
         post-attn-ln-w (keyword (str "post_attn_ln_w_" i))
         pre-mlp-ln-w (keyword (str "pre_mlp_ln_w_" i))
         post-mlp-ln-w (keyword (str "post_mlp_ln_w_" i))
         gate-w (keyword (str "gate_w_" i))
         up-w (keyword (str "up_w_" i))
         down-w (keyword (str "down_w_" i))
         per-layer-gate-w (keyword (str "per_layer_gate_w_" i))
         per-layer-proj-w (keyword (str "per_layer_proj_w_" i))
         post-per-layer-norm-w (keyword (str "post_per_layer_norm_w_" i))
         pl-in-var (keyword (str "pl_in_" i))

         q-scale (keyword (str "q_scale_" i))
         k-scale (keyword (str "k_scale_" i))
         v-scale (keyword (str "v_scale_" i))
         o-scale (keyword (str "o_scale_" i))
         gate-scale (keyword (str "gate_scale_" i))
         up-scale (keyword (str "up_scale_" i))
         down-scale (keyword (str "down_scale_" i))

         x-norm1 (keyword (str "x_norm1_" i))
         q-raw (keyword (str "q_raw_" i))
         k-raw (keyword (str "k_raw_" i))
         v-raw (keyword (str "v_raw_" i))
         v-normed (keyword (str "v_normed_" i))
         q-heads-raw (keyword (str "q_heads_raw_" i))
         k-heads-raw (keyword (str "k_heads_raw_" i))
         v-heads (keyword (str "v_heads_" i))
         q-normed-4d (keyword (str "q_normed_4d_" i))
         k-normed-4d (keyword (str "k_normed_4d_" i))
         q-normed-3d (keyword (str "q_normed_3d_" i))
         k-normed-3d (keyword (str "k_normed_3d_" i))
         q-rope (keyword (str "q_rope_" i))
         k-rope (keyword (str "k_rope_" i))
         q-ro (keyword (str "q_ro_" i))
         k-ro (keyword (str "k_ro_" i))
         actual-k-ro (if is-shared? shared-k k-ro)
         actual-v-heads (if is-shared? shared-v v-heads)
         k-rep (keyword (str "k_rep_" i))
         k-heads (keyword (str "k_heads_" i))
         v-rep (keyword (str "v_rep_" i))
         v-rep-heads (keyword (str "v_rep_heads_" i))
         scores (keyword (str "scores_" i))
         probs (keyword (str "probs_" i))
         ctx (keyword (str "ctx_" i))
         ctx-flat (keyword (str "ctx_flat_" i))
         attn-raw (keyword (str "attn_raw_" i))
         attn-normed (keyword (str "attn_normed_" i))
         res1 (keyword (str "res1_" i))

         x-norm2 (keyword (str "x_norm2_" i))
         gate (keyword (str "gate_" i))
         up (keyword (str "up_" i))
         mlp-act (keyword (str "mlp_act_" i))
         mlp-raw (keyword (str "mlp_raw_" i))
         mlp-normed (keyword (str "mlp_normed_" i))
         res2 (keyword (str "res2_" i))

         gate-raw (keyword (str "gate_raw_" i))
         gated (keyword (str "gated_" i))
         proj-raw (keyword (str "proj_raw_" i))
         ple-normed (keyword (str "ple_normed_" i))
         res3 (keyword (str "res3_" i))

         dh (keyword (str "dh_" i))
         qd (keyword (str "qd_" i))
         kvd (keyword (str "kvd_" i))
         kvh (keyword (str "kvh_" i))
         g (keyword (str "g_" i))
         h (keyword (str "h_" i))
         dff (keyword (str "dff_" i))
         pld (keyword (str "pld_" i))
         p-q (keyword (str "p_q_" i))
         p-k (keyword (str "p_k_" i))
         k-ro-sl (keyword (str "k_ro_sl_" i))
         v-heads-sl (keyword (str "v_heads_sl_" i))
         kv-s (keyword (str "kvs_" i))]

     [:block {:name (keyword (str "gemma4_layer_" i))}
      ;; 1. Pre-Attention RMSNorm
      [:rms-norm [x-norm1 :b :p :d] [h-in :b :p :d] [input-ln-w :d] {:eps 1e-6}]

      ;; 2. Q Projection & Reshape
      (gemma4-linear-proj [q-raw :b :p qd] [x-norm1 :b :p :d] [q-w qd :d] is-int8? is-int4? q-scale norm-dtype {})
      [:reshape [q-heads-raw :b :p h dh] [q-raw :b :p qd] {:shape [1 max-seq-len num-heads head-dim]}]
      [:rms-norm [q-normed-4d :b :p h dh] [q-heads-raw :b :p h dh] [q-norm-w dh] {:eps 1e-6}]
      [:reshape [q-normed-3d :b :p qd] [q-normed-4d :b :p h dh] {:shape [1 max-seq-len q-dim]}]
      [:rope [q-rope :b :p qd] [q-normed-3d :b :p qd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
      [:reshape [q-ro :b :p h dh] [q-rope :b :p qd] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 3. K, V Projections (computed only if not shared)
      (when-not is-shared?
        [:block {:name (keyword (str "kv_proj_" i))}
         (gemma4-linear-proj [k-raw :b :p kvd] [x-norm1 :b :p :d] [k-w kvd :d] is-int8? is-int4? k-scale norm-dtype {})
         (gemma4-linear-proj [v-raw :b :p kvd] [x-norm1 :b :p :d] [v-w kvd :d] is-int8? is-int4? v-scale norm-dtype {})
         [:rms-norm [v-normed :b :p kvd] [v-raw :b :p kvd] {:eps 1e-6}]
         [:reshape [k-heads-raw :b :p kvh dh] [k-raw :b :p kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:reshape [v-heads :b :p kvh dh] [v-normed :b :p kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         [:rms-norm [k-normed-4d :b :p kvh dh] [k-heads-raw :b :p kvh dh] [k-norm-w dh] {:eps 1e-6}]
         [:reshape [k-normed-3d :b :p kvd] [k-normed-4d :b :p kvh dh] {:shape [1 max-seq-len kv-dim]}]
         [:rope [k-rope :b :p kvd] [k-normed-3d :b :p kvd] {:head-dim head-dim :theta theta :rope-proportion rope-prop}]
         [:reshape [k-ro :b :p kvh dh] [k-rope :b :p kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
         (when ring-buffer?
           [:slice [k-ro-sl :b kv-s kvh dh] [k-ro :b :p kvh dh]
            {:start [0 0 0 0] :limit [1 layer-seq-len num-kv-heads head-dim]}])
         (when ring-buffer?
           [:slice [v-heads-sl :b kv-s kvh dh] [v-heads :b :p kvh dh]
            {:start [0 0 0 0] :limit [1 layer-seq-len num-kv-heads head-dim]}])])

      ;; 4. Broadcast KV Heads
      [:= [k-rep :b :p kvh g dh] [actual-k-ro :b :p kvh dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [k-heads :b :p h dh] [k-rep :b :p kvh g dh] {:shape [1 max-seq-len num-heads head-dim]}]
      [:= [v-rep :b :p kvh g dh] [actual-v-heads :b :p kvh dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
      [:reshape [v-rep-heads :b :p h dh] [v-rep :b :p kvh g dh] {:shape [1 max-seq-len num-heads head-dim]}]

      ;; 5. Scaled Dot-Product Attention: QK^T -> Softmax (sliding or full) -> probs @ V
      [:= [scores :b h p-q p-k] {:scale 1.0} [q-ro :b p-q h dh] [k-heads :b p-k h dh]]
      [:causal-softmax [probs :b h p-q p-k] [scores :b h p-q p-k] (if window {:sliding-window window} {})]
      [:= [ctx :b p-q h dh] [probs :b h p-q p-k] [v-rep-heads :b p-k h dh]]
      [:reshape [ctx-flat :b :p qd] [ctx :b p-q h dh] {:shape [1 max-seq-len q-dim]}]

      ;; 6. Output Projection & Post-Attention RMSNorm
      (gemma4-linear-proj [attn-raw :b :p :d] [ctx-flat :b :p qd] [o-w :d qd] is-int8? is-int4? o-scale norm-dtype {})
      [:rms-norm [attn-normed :b :p :d] [attn-raw :b :p :d] [post-attn-ln-w :d] {:eps 1e-6}]

      ;; 7. Residual Connection 1
      [:= [res1 :b :p :d] [h-in :b :p :d]]
      [:= [res1 :b :p :d] [attn-normed :b :p :d]]

      ;; 8. Pre-MLP RMSNorm
      [:rms-norm [x-norm2 :b :p :d] [res1 :b :p :d] [pre-mlp-ln-w :d] {:eps 1e-6}]

      ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
      (gemma4-linear-proj [gate :b :p dff] [x-norm2 :b :p :d] [gate-w dff :d] is-int8? is-int4? gate-scale norm-dtype {:act :gelu})
      (gemma4-linear-proj [up :b :p dff] [x-norm2 :b :p :d] [up-w dff :d] is-int8? is-int4? up-scale norm-dtype {})
      [:= [mlp-act :b :p dff] [gate :b :p dff] [up :b :p dff]]
      (gemma4-linear-proj [mlp-raw :b :p :d] [mlp-act :b :p dff] [down-w :d dff] is-int8? is-int4? down-scale norm-dtype {})
      [:rms-norm [mlp-normed :b :p :d] [mlp-raw :b :p :d] [post-mlp-ln-w :d] {:eps 1e-6}]

      ;; 10. Residual Connection 2
      [:= [res2 :b :p :d] [res1 :b :p :d]]
      [:= [res2 :b :p :d] [mlp-normed :b :p :d]]

      ;; 11. Gemma 4 Per-Layer Input (PLE) Gating Sub-block (if has-ple?)
      (when has-ple?
        [:block {:name (keyword (str "ple_gate_" i))}
         [:= [gate-raw :b :p pld] {:act :gelu} [res2 :b :p :d] [per-layer-gate-w pld :d]]
         [:= [gated :b :p pld] [gate-raw :b :p pld] [pl-in-var :b :p pld]]
         [:= [proj-raw :b :p :d] [gated :b :p pld] [per-layer-proj-w :d pld]]
         [:rms-norm [ple-normed :b :p :d] [proj-raw :b :p :d] [post-per-layer-norm-w :d] {:eps 1e-6}]
         [:= [res3 :b :p :d] [res2 :b :p :d]]
         [:= [res3 :b :p :d] [ple-normed :b :p :d]]])

      ;; 12. Gemma 4 Layer Scalar
      [:= [h-out :b :p :d] [(if has-ple? res3 res2) :b :p :d] [layer-scalar-w :one]]])))

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
        h-final (keyword (str "h" num-layers))

        last-unshared-sliding (when has-shared-kv? (last (filter #(not (layer-is-global? layer-types %)) (range num-unshared))))
        last-unshared-full (when has-shared-kv? (last (filter #(layer-is-global? layer-types %) (range num-unshared))))]
    [:block {:name :full_gemma4_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [:h0 :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Gemma 4 Per-Layer Embedding (PLE) generation
     (when has-ple?
       [:block {:name :ple_generation}
        [:gather [:raw_pl_tok :b :p :total_pl_dim] [:embed_tokens_per_layer :v :total_pl_dim] [:x :b :p]]
        [:= [:pl_tok_scaled :b :p :total_pl_dim] {:scale 16.0} [:raw_pl_tok :b :p :total_pl_dim]]
        [:= [:pl_context_raw :b :p :total_pl_dim] [:h0 :b :p :d] [:per_layer_model_projection :total_pl_dim :d]]
        [:reshape [:pl_tok_4d :b :p :l :pld] [:pl_tok_scaled :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:reshape [:pl_context_4d :b :p :l :pld] [:pl_context_raw :b :p :total_pl_dim] {:shape [1 max-seq-len num-layers pl-dim]}]
        [:rms-norm [:pl_context_norm :b :p :l :pld] [:pl_context_4d :b :p :l :pld] [:per_layer_projection_norm :pld] {:eps 1e-6}]
        [:= [:pl_sum :b :p :l :pld] [:pl_context_norm :b :p :l :pld]]
        [:= [:pl_sum :b :p :l :pld] [:pl_tok_4d :b :p :l :pld]]
        [:= [:ple_all :b :p :l :pld] {:scale (/ 1.0 (Math/sqrt 2.0))} [:pl_sum :b :p :l :pld]]
        ;; Layer PLE Slices
        (mapv (fn [i]
                (let [sl-var (keyword (str "pl_slice_" i))
                      in-var (keyword (str "pl_in_" i))]
                  [:block {:name (keyword (str "ple_slice_block_" i))}
                   [:slice [sl-var :b :p :one :pld] [:ple_all :b :p :l :pld] {:start [0 0 i 0] :limit [1 max-seq-len (inc i) pl-dim]}]
                   [:reshape [in-var :b :p :pld] [sl-var :b :p :one :pld] {:shape [1 max-seq-len pl-dim]}]]))
              (range num-layers))])

     ;; 3. Sequential Transformer Layer Blocks
     (mapv (fn [i]
             (let [is-shared? (and has-shared-kv? (>= i num-unshared))
                   is-global? (layer-is-global? layer-types i)
                   shared-k (when is-shared?
                              (if is-global?
                                (keyword (str "k_ro_" last-unshared-full))
                                (keyword (str "k_ro_" last-unshared-sliding))))
                   shared-v (when is-shared?
                              (if is-global?
                                (keyword (str "v_heads_" last-unshared-full))
                                (keyword (str "v_heads_" last-unshared-sliding))))]
               (gemma4-layer-ast i max-seq-len cfg {:is-shared? is-shared?
                                                    :shared-k shared-k
                                                    :shared-v shared-v})))
           (range num-layers))

     ;; 4. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps 1e-6}]

     ;; 5. Tied LM Head with optional final logit softcapping and relational grounding
     (let [has-rel-mem? (boolean (or (:relational-memory cfg) (:relational-memory? cfg)))
           rel-cfg (or (:relational-memory cfg) {})
           mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
           entity-count (long (or (:entity-count rel-cfg) 1000))
           vocab-size (long (or (:vocab-size cfg) 262144))
           norm-dtype (if (or (:is-int8 cfg) (:is-int4 cfg)) :bf16 (or (:weight-dtype cfg) :f32))]
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
         norm-dtype (get config :norm-dtype (if (or is-int8? is-int4?) :bf16 (get config :weight-dtype :bf16)))

         h-in (keyword (str "h" i))
         h-out (keyword (str "h" (inc i)))

         input-ln-w (keyword (str "input_ln_w_" i))
         layer-scalar-w (keyword (str "layer_scalar_" i))
         q-w (keyword (str "q_w_" i))
         k-w (keyword (str "k_w_" i))
         v-w (keyword (str "v_w_" i))
         o-w (keyword (str "o_w_" i))
         q-norm-w (keyword (str "q_norm_w_" i))
         k-norm-w (keyword (str "k_norm_w_" i))
         post-attn-ln-w (keyword (str "post_attn_ln_w_" i))
         pre-mlp-ln-w (keyword (str "pre_mlp_ln_w_" i))
         post-mlp-ln-w (keyword (str "post_mlp_ln_w_" i))
         gate-w (keyword (str "gate_w_" i))
         up-w (keyword (str "up_w_" i))
         down-w (keyword (str "down_w_" i))
         per-layer-gate-w (keyword (str "per_layer_gate_w_" i))
         per-layer-proj-w (keyword (str "per_layer_proj_w_" i))
         post-per-layer-norm-w (keyword (str "post_per_layer_norm_w_" i))
         pl-in-var (keyword (str "pl_in_" i))

         q-scale (keyword (str "q_scale_" i))
         k-scale (keyword (str "k_scale_" i))
         v-scale (keyword (str "v_scale_" i))
         o-scale (keyword (str "o_scale_" i))
         gate-scale (keyword (str "gate_scale_" i))
         up-scale (keyword (str "up_scale_" i))
         down-scale (keyword (str "down_scale_" i))

         k-cache-in (keyword (str "k_cache_in_" i))
         v-cache-in (keyword (str "v_cache_in_" i))
         k-cache-out (keyword (str "k_cache_out_" i))
         v-cache-out (keyword (str "v_cache_out_" i))

         x-norm1 (keyword (str "x_norm1_" i))
         q-raw (keyword (str "q_raw_" i))
         k-raw (keyword (str "k_raw_" i))
         v-raw (keyword (str "v_raw_" i))
         v-normed (keyword (str "v_normed_" i))
         q-heads-raw (keyword (str "q_heads_raw_" i))
         k-heads-raw (keyword (str "k_heads_raw_" i))
         v-heads (keyword (str "v_heads_" i))
         q-normed-4d (keyword (str "q_normed_4d_" i))
         k-normed-4d (keyword (str "k_normed_4d_" i))
         q-normed-3d (keyword (str "q_normed_3d_" i))
         k-normed-3d (keyword (str "k_normed_3d_" i))
         q-rope (keyword (str "q_rope_" i))
         k-rope (keyword (str "k_rope_" i))
         q-ro (keyword (str "q_ro_" i))
         k-ro (keyword (str "k_ro_" i))

         actual-k-cache (if is-shared? shared-k k-cache-out)
         actual-v-cache (if is-shared? shared-v v-cache-out)

         ctx (keyword (str "ctx_" i))
         ctx-flat (keyword (str "ctx_flat_" i))
         attn-raw (keyword (str "attn_raw_" i))
         attn-normed (keyword (str "attn_normed_" i))
         res1 (keyword (str "res1_" i))

         x-norm2 (keyword (str "x_norm2_" i))
         gate (keyword (str "gate_" i))
         up (keyword (str "up_" i))
         mlp-act (keyword (str "mlp_act_" i))
         mlp-raw (keyword (str "mlp_raw_" i))
         mlp-normed (keyword (str "mlp_normed_" i))
         res2 (keyword (str "res2_" i))

         gate-raw (keyword (str "gate_raw_" i))
         gated (keyword (str "gated_" i))
         proj-raw (keyword (str "proj_raw_" i))
         ple-normed (keyword (str "ple_normed_" i))
         res3 (keyword (str "res3_" i))

         dh (keyword (str "dh_" i))
         qd (keyword (str "qd_" i))
         kvd (keyword (str "kvd_" i))
         kvh (keyword (str "kvh_" i))
         _g (keyword (str "g_" i))
         h (keyword (str "h_" i))
         dff (keyword (str "dff_" i))
         pld (keyword (str "pld_" i))
         p (keyword (str "p_" i))
         kv-s (keyword (str "kvs_" i))]

     [:block {:name (keyword (str "gemma4_kv_layer_" i))}
      ;; 1. Pre-Attention RMSNorm
      [:rms-norm [x-norm1 :b p :d] [h-in :b p :d] [input-ln-w :d] {:eps 1e-6}]

      ;; 2. Q Projection & Reshape for single token
      (gemma4-linear-proj [q-raw :b p qd] [x-norm1 :b p :d] [q-w qd :d] is-int8? is-int4? q-scale norm-dtype {})
      [:reshape [q-heads-raw :b p h dh] [q-raw :b p qd] {:shape [1 1 num-heads head-dim]}]
      [:rms-norm [q-normed-4d :b p h dh] [q-heads-raw :b p h dh] [q-norm-w dh] {:eps 1e-6}]
      [:reshape [q-normed-3d :b p qd] [q-normed-4d :b p h dh] {:shape [1 1 q-dim]}]
      [:rope [q-rope :b p qd] [q-normed-3d :b p qd] {:head-dim head-dim :theta theta :rope-proportion rope-prop :pos :pos :max-seq-len max-seq-len}]
      [:reshape [q-ro :b p h dh] [q-rope :b p qd] {:shape [1 1 num-heads head-dim]}]

      ;; 3. K, V Projections & Dynamic Cache Update (if not shared)
      (when-not is-shared?
        [:block {:name (keyword (str "kv_proj_update_" i))}
         (gemma4-linear-proj [k-raw :b p kvd] [x-norm1 :b p :d] [k-w kvd :d] is-int8? is-int4? k-scale norm-dtype {})
         (gemma4-linear-proj [v-raw :b p kvd] [x-norm1 :b p :d] [v-w kvd :d] is-int8? is-int4? v-scale norm-dtype {})
         [:rms-norm [v-normed :b p kvd] [v-raw :b p kvd] {:eps 1e-6}]
         [:reshape [k-heads-raw :b p kvh dh] [k-raw :b p kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:reshape [v-heads :b p kvh dh] [v-normed :b p kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:rms-norm [k-normed-4d :b p kvh dh] [k-heads-raw :b p kvh dh] [k-norm-w dh] {:eps 1e-6}]
         [:reshape [k-normed-3d :b p kvd] [k-normed-4d :b p kvh dh] {:shape [1 1 kv-dim]}]
         [:rope [k-rope :b p kvd] [k-normed-3d :b p kvd] {:head-dim head-dim :theta theta :rope-proportion rope-prop :pos :pos :max-seq-len max-seq-len}]
         [:reshape [k-ro :b p kvh dh] [k-rope :b p kvd] {:shape [1 1 num-kv-heads head-dim]}]
         [:dynamic-update-slice [k-cache-out :b kv-s kvh dh] [k-cache-in :b kv-s kvh dh] [k-ro :b p kvh dh]
          (merge {:start-indices [0 :pos 0 0]}
                 (when ring-buffer? {:window window}))]
         [:dynamic-update-slice [v-cache-out :b kv-s kvh dh] [v-cache-in :b kv-s kvh dh] [v-heads :b p kvh dh]
          (merge {:start-indices [0 :pos 0 0]}
                 (when ring-buffer? {:window window}))]])

      ;; 4 & 5. Chunked Scaled Dot-Product Attention (Online Streaming Softmax)
      [:chunked-attention [ctx :b p h dh]
       [q-ro :b p h dh]
       [actual-k-cache :b kv-s kvh dh]
       [actual-v-cache :b kv-s kvh dh]
       (merge {:pos :pos
               :chunk-size (min 64 layer-seq-len)
               :head-dim head-dim
               :num-heads num-heads
               :num-kv-heads num-kv-heads
               :max-seq-len layer-seq-len
               :shape [1 1 num-heads head-dim]}
              (when ring-buffer? {:ring-buffer true :sliding-window window})
              (when (and window (not ring-buffer?)) {:sliding-window window}))]
      [:reshape [ctx-flat :b p qd] [ctx :b p h dh] {:shape [1 1 q-dim]}]

      ;; 6. Output Projection & Post-Attention RMSNorm
      (gemma4-linear-proj [attn-raw :b p :d] [ctx-flat :b p qd] [o-w :d qd] is-int8? is-int4? o-scale norm-dtype {})
      [:rms-norm [attn-normed :b p :d] [attn-raw :b p :d] [post-attn-ln-w :d] {:eps 1e-6}]

      ;; 7. Residual Connection 1
      [:= [res1 :b p :d] [h-in :b p :d]]
      [:= [res1 :b p :d] [attn-normed :b p :d]]

      ;; 8. Pre-MLP RMSNorm
      [:rms-norm [x-norm2 :b p :d] [res1 :b p :d] [pre-mlp-ln-w :d] {:eps 1e-6}]

      ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
      (gemma4-linear-proj [gate :b p dff] [x-norm2 :b p :d] [gate-w dff :d] is-int8? is-int4? gate-scale norm-dtype {:act :gelu})
      (gemma4-linear-proj [up :b p dff] [x-norm2 :b p :d] [up-w dff :d] is-int8? is-int4? up-scale norm-dtype {})
      [:= [mlp-act :b p dff] [gate :b p dff] [up :b p dff]]
      (gemma4-linear-proj [mlp-raw :b p :d] [mlp-act :b p dff] [down-w :d dff] is-int8? is-int4? down-scale norm-dtype {})
      [:rms-norm [mlp-normed :b p :d] [mlp-raw :b p :d] [post-mlp-ln-w :d] {:eps 1e-6}]

      ;; 10. Residual Connection 2
      [:= [res2 :b p :d] [res1 :b p :d]]
      [:= [res2 :b p :d] [mlp-normed :b p :d]]

      ;; 11. Gemma 4 Per-Layer Input (PLE) Gating Sub-block (if has-ple?)
      (when has-ple?
        [:block {:name (keyword (str "ple_gate_" i))}
         [:= [gate-raw :b p pld] {:act :gelu} [res2 :b p :d] [per-layer-gate-w pld :d]]
         [:= [gated :b p pld] [gate-raw :b p pld] [pl-in-var :b p pld]]
         [:= [proj-raw :b p :d] [gated :b p pld] [per-layer-proj-w :d pld]]
         [:rms-norm [ple-normed :b p :d] [proj-raw :b p :d] [post-per-layer-norm-w :d] {:eps 1e-6}]
         [:= [res3 :b p :d] [res2 :b p :d]]
         [:= [res3 :b p :d] [ple-normed :b p :d]]])

      ;; 12. Gemma 4 Layer Scalar
      [:= [h-out :b p :d] [(if has-ple? res3 res2) :b p :d] [layer-scalar-w :one]]])))

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
        h-final (keyword (str "h" num-layers))

        last-unshared-sliding (when has-shared-kv? (last (filter #(not (layer-is-global? layer-types %)) (range num-unshared))))
        last-unshared-full (when has-shared-kv? (last (filter #(layer-is-global? layer-types %) (range num-unshared))))]
    [:block {:name :gemma4_kv_step_model}
     ;; 1. Token Embedding Lookup for single token [1 1]
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [:h0 :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Gemma 4 Per-Layer Embedding (PLE) generation at p=1
     (when has-ple?
       [:block {:name :ple_generation}
        [:gather [:raw_pl_tok :b :p :total_pl_dim] [:embed_tokens_per_layer :v :total_pl_dim] [:x :b :p]]
        [:= [:pl_tok_scaled :b :p :total_pl_dim] {:scale 16.0} [:raw_pl_tok :b :p :total_pl_dim]]
        [:= [:pl_context_raw :b :p :total_pl_dim] [:h0 :b :p :d] [:per_layer_model_projection :total_pl_dim :d]]
        [:reshape [:pl_tok_4d :b :p :l :pld] [:pl_tok_scaled :b :p :total_pl_dim] {:shape [1 1 num-layers pl-dim]}]
        [:reshape [:pl_context_4d :b :p :l :pld] [:pl_context_raw :b :p :total_pl_dim] {:shape [1 1 num-layers pl-dim]}]
        [:rms-norm [:pl_context_norm :b :p :l :pld] [:pl_context_4d :b :p :l :pld] [:per_layer_projection_norm :pld] {:eps 1e-6}]
        [:= [:pl_sum :b :p :l :pld] [:pl_context_norm :b :p :l :pld]]
        [:= [:pl_sum :b :p :l :pld] [:pl_tok_4d :b :p :l :pld]]
        [:= [:ple_all :b :p :l :pld] {:scale (/ 1.0 (Math/sqrt 2.0))} [:pl_sum :b :p :l :pld]]
        (mapv (fn [i]
                (let [sl-var (keyword (str "pl_slice_" i))
                      in-var (keyword (str "pl_in_" i))]
                  [:block {:name (keyword (str "ple_slice_block_" i))}
                   [:slice [sl-var :b :p :one :pld] [:ple_all :b :p :l :pld] {:start [0 0 i 0] :limit [1 1 (inc i) pl-dim]}]
                   [:reshape [in-var :b :p :pld] [sl-var :b :p :one :pld] {:shape [1 1 pl-dim]}]]))
              (range num-layers))])

     ;; 3. Sequential Transformer Layer Blocks with KV Cache
     (mapv (fn [i]
             (let [is-shared? (and has-shared-kv? (>= i num-unshared))
                   is-global? (layer-is-global? layer-types i)
                   shared-k (when is-shared?
                              (if is-global?
                                (keyword (str "k_cache_out_" last-unshared-full))
                                (keyword (str "k_cache_out_" last-unshared-sliding))))
                   shared-v (when is-shared?
                              (if is-global?
                                (keyword (str "v_cache_out_" last-unshared-full))
                                (keyword (str "v_cache_out_" last-unshared-sliding))))]
               (gemma4-kv-layer-ast i max-seq-len cfg {:is-shared? is-shared?
                                                       :shared-k shared-k
                                                       :shared-v shared-v})))
           (range num-layers))

     ;; 4. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps 1e-6}]

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
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 is-int4 layer-configs last-token-only? group-size]} config
        norm-dtype (if (or is-int8 is-int4) :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        scale-shape-fn (fn [rows cols]
                         (if (and is-int4 group-size (zero? (mod cols group-size)))
                           [rows (quot cols group-size)]
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

(defn build-gemma4-kv-invars
  "Constructs EDN SSA signature invars for single-token Gemma 4 forward pass with KV Cache."
  [config max-seq-len]
  (let [{:keys [num-layers num-kv-shared-layers weight-dtype is-int8 is-int4 layer-configs layer-types]} config
        norm-dtype (if (or is-int8 is-int4) :bf16 weight-dtype)
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
