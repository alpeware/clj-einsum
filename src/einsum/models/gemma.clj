(ns einsum.models.gemma
  "Declarative Gemma 1 and Gemma 2 Architecture definitions in pure Tensor Logic Hiccup AST.
   Also re-exports canonical Gemma 4 definitions from einsum.models.gemma4."
  (:require [einsum.models.gemma4 :as g4]))

;; ==============================================================================
;; 1. Gemma 1 / Gemma 2 Configurations
;; ==============================================================================

(def DEFAULT_GEMMA_CONFIG
  {:hidden-dim 2048
   :intermediate-dim 16384
   :num-layers 18
   :num-heads 8
   :num-kv-heads 1
   :head-dim 256
   :vocab-size 256000
   :norm-eps 1e-6})

(defn gemma-config
  "Returns Gemma 1/2 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA_CONFIG overrides)))

(defn weight-key-map
  "Returns Gemma 1 / Gemma 2 safetensors weight key mapping for layer `layer-idx`."
  [layer-idx]
  (let [prefix (str "model.layers." layer-idx ".")]
    {:input-ln-w     (str prefix "input_layernorm.weight")
     :q-w            (str prefix "self_attn.q_proj.weight")
     :k-w            (str prefix "self_attn.k_proj.weight")
     :v-w            (str prefix "self_attn.v_proj.weight")
     :o-w            (str prefix "self_attn.o_proj.weight")
     :post-attn-ln-w (str prefix "post_attention_layernorm.weight")
     :pre-mlp-ln-w   (str prefix "pre_feedforward_layernorm.weight")
     :post-mlp-ln-w  (str prefix "post_feedforward_layernorm.weight")
     :gate-w         (str prefix "mlp.gate_proj.weight")
     :up-w           (str prefix "mlp.up_proj.weight")
     :down-w         (str prefix "mlp.down_proj.weight")}))

;; ==============================================================================
;; 2. Gemma 2 Layer & Model ASTs
;; ==============================================================================

(defn gemma2-layer-ast
  "Generates Tensor Logic Hiccup AST for Gemma 2 Transformer layer block `layer-idx`."
  [layer-idx max-seq-len config]
  (let [i layer-idx
        {:keys [num-heads num-kv-heads head-dim layer-types norm-eps attn-softcap]} config
        num-heads (long (or num-heads 8))
        num-kv-heads (long (or num-kv-heads 1))
        head-dim (long (or head-dim 256))
        q-dim (* num-heads head-dim)
        _kv-dim (* num-kv-heads head-dim)
        group-size (quot num-heads num-kv-heads)
        norm-eps (double (or norm-eps 1e-6))
        attn-softcap (or attn-softcap 50.0)

        l-type (when layer-types (nth layer-types i nil))
        is-sliding? (or (= l-type "sliding_attention")
                        (= l-type :sliding_attention)
                        (odd? i))
        window (if is-sliding? 4096 nil)
        scale (/ 1.0 (Math/sqrt (double head-dim)))

        h-in (keyword (str "h" i))
        h-out (keyword (str "h" (inc i)))

        input-ln-w (keyword (str "input_ln_w_" i))
        q-w (keyword (str "q_w_" i))
        k-w (keyword (str "k_w_" i))
        v-w (keyword (str "v_w_" i))
        o-w (keyword (str "o_w_" i))
        post-attn-ln-w (keyword (str "post_attn_ln_w_" i))
        pre-mlp-ln-w (keyword (str "pre_mlp_ln_w_" i))
        gate-w (keyword (str "gate_w_" i))
        up-w (keyword (str "up_w_" i))
        down-w (keyword (str "down_w_" i))
        post-mlp-ln-w (keyword (str "post_mlp_ln_w_" i))

        x-norm1 (keyword (str "x_norm1_" i))
        q-raw (keyword (str "q_raw_" i))
        k-raw (keyword (str "k_raw_" i))
        v-raw (keyword (str "v_raw_" i))
        q-rope (keyword (str "q_rope_" i))
        k-rope (keyword (str "k_rope_" i))
        q-heads (keyword (str "q_heads_" i))
        k-heads (keyword (str "k_heads_" i))
        v-heads (keyword (str "v_heads_" i))
        k-rep (keyword (str "k_rep_" i))
        k-full (keyword (str "k_full_" i))
        v-rep (keyword (str "v_rep_" i))
        v-full (keyword (str "v_full_" i))
        scores-raw (keyword (str "scores_raw_" i))
        attn-weights (keyword (str "attn_weights_" i))
        attn-ctx (keyword (str "attn_ctx_" i))
        attn-proj-in (keyword (str "attn_proj_in_" i))
        attn-out (keyword (str "attn_out_" i))
        attn-normed (keyword (str "attn_normed_" i))
        res1 (keyword (str "res1_" i))
        x-norm2 (keyword (str "x_norm2_" i))
        gate-out (keyword (str "gate_out_" i))
        up-out (keyword (str "up_out_" i))
        hidden (keyword (str "hidden_" i))
        mlp-out (keyword (str "mlp_out_" i))
        mlp-normed (keyword (str "mlp_normed_" i))]
    [:block {:name (keyword (str "gemma2_layer_" i))}
     ;; 1. Input RMSNorm
     [:gemma-rms-norm [x-norm1 :b :p :d] [h-in :b :p :d] [input-ln-w :d] {:eps norm-eps}]

     ;; 2. Attention Projections
     [:= [q-raw :b :p :qd] [x-norm1 :b :p :d] [q-w :qd :d]]
     [:= [k-raw :b :p :kvd] [x-norm1 :b :p :d] [k-w :kvd :d]]
     [:= [v-raw :b :p :kvd] [x-norm1 :b :p :d] [v-w :kvd :d]]

     ;; 3. Rotary Position Embeddings
     [:rope [q-rope :b :p :qd] [q-raw :b :p :qd] {:head-dim head-dim :theta 10000.0}]
     [:rope [k-rope :b :p :kvd] [k-raw :b :p :kvd] {:head-dim head-dim :theta 10000.0}]

     ;; 4. Reshape into Multi-Head View
     [:reshape [q-heads :b :h :p :hd] [q-rope :b :p :qd] {:shape [1 num-heads max-seq-len head-dim]}]
     [:reshape [k-heads :b :kvh :p :hd] [k-rope :b :p :kvd] {:shape [1 num-kv-heads max-seq-len head-dim]}]
     [:reshape [v-heads :b :kvh :p :hd] [v-raw :b :p :kvd] {:shape [1 num-kv-heads max-seq-len head-dim]}]

     ;; 5. Grouped Query Expansion (if GQA)
     (if (> group-size 1)
       [:block {:name (keyword (str "gqa_expand_" i))}
        [:= [k-rep :b :kvh :g :p :hd] [k-heads :b :kvh :p :hd] {:shape [1 num-kv-heads group-size max-seq-len head-dim]}]
        [:reshape [k-full :b :h :p :hd] [k-rep :b :kvh :g :p :hd] {:shape [1 num-heads max-seq-len head-dim]}]
        [:= [v-rep :b :kvh :g :p :hd] [v-heads :b :kvh :p :hd] {:shape [1 num-kv-heads group-size max-seq-len head-dim]}]
        [:reshape [v-full :b :h :p :hd] [v-rep :b :kvh :g :p :hd] {:shape [1 num-heads max-seq-len head-dim]}]]
       [:block {:name (keyword (str "mha_view_" i))}
        [:= [k-full :b :h :p :hd] [k-heads :b :h :p :hd]]
        [:= [v-full :b :h :p :hd] [v-heads :b :h :p :hd]]])

     ;; 6. Scaled Dot-Product Attention with Causal Softmax
     [:= [scores-raw :b :h :p :k] (cond-> {:scale scale}
                                    (and attn-softcap (pos? (double attn-softcap)))
                                    (assoc :softcap (double attn-softcap)))
      [q-heads :b :h :p :hd] [k-full :b :h :k :hd]]
     [:causal-softmax [attn-weights :b :h :p :k] (if window {:window window} {}) [scores-raw :b :h :p :k]]
     [:= [attn-ctx :b :h :p :hd] [attn-weights :b :h :p :k] [v-full :b :h :k :hd]]

     ;; 7. Attention Out Projection
     [:reshape [attn-proj-in :b :p :qd] [attn-ctx :b :h :p :hd] {:shape [1 max-seq-len q-dim]}]
     [:= [attn-out :b :p :d] [attn-proj-in :b :p :qd] [o-w :d :qd]]

     ;; 8. Post Attention RMSNorm
     [:gemma-rms-norm [attn-normed :b :p :d] [attn-out :b :p :d] [post-attn-ln-w :d] {:eps norm-eps}]

     ;; 9. First Residual Connection
     [:= [res1 :b :p :d] [h-in :b :p :d]]
     [:= [res1 :b :p :d] [attn-normed :b :p :d]]

     ;; 10. Pre-FeedForward RMSNorm
     [:gemma-rms-norm [x-norm2 :b :p :d] [res1 :b :p :d] [pre-mlp-ln-w :d] {:eps norm-eps}]

     ;; 11. GeGLU MLP Block
     [:= [gate-out :b :p :inter] {:act :gelu} [x-norm2 :b :p :d] [gate-w :inter :d]]
     [:= [up-out :b :p :inter] [x-norm2 :b :p :d] [up-w :inter :d]]
     [:= [hidden :b :p :inter] [gate-out :b :p :inter] [up-out :b :p :inter]]
     [:= [mlp-out :b :p :d] [hidden :b :p :inter] [down-w :d :inter]]

     ;; 12. Post-FeedForward RMSNorm
     [:gemma-rms-norm [mlp-normed :b :p :d] [mlp-out :b :p :d] [post-mlp-ln-w :d] {:eps norm-eps}]

     ;; 13. Second Residual Connection
     [:= [h-out :b :p :d] [res1 :b :p :d]]
     [:= [h-out :b :p :d] [mlp-normed :b :p :d]]]))

(defn gemma2-model-ast
  "Generates full Gemma 2 model forward pass in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge DEFAULT_GEMMA_CONFIG config)
        {:keys [num-layers max-seq-len hidden-dim final-logit-softcap norm-eps]} cfg
        num-layers (long (or num-layers 18))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 2048))
        norm-eps (double (or norm-eps 1e-6))
        h-final (keyword (str "h" num-layers))]
    [:block {:name :full_gemma2_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [:h0 :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Sequential Transformer Layer Blocks
     (mapv (fn [i] (gemma2-layer-ast i max-seq-len cfg)) (range num-layers))

     ;; 3. Final RMSNorm
     [:gemma-rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps norm-eps}]

     ;; 4. Tied LM Head with optional final logit softcapping
     (if (:last-token-only? cfg)
       [:block {:name :last_token_head}
        [:dynamic-slice [:normed_last :b :one :d] [:normed :b :p :d]
         {:slice-sizes [1 1 hidden-dim]
          :start-indices [0 :pos 0]}]
        [:= [:logits :b :one :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                   {:softcap (double final-logit-softcap)}
                                   {})
         [:normed_last :b :one :d] [:embed_tokens :v :d]]]
       [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                                {:softcap (double final-logit-softcap)}
                                {})
        [:normed :b :p :d] [:embed_tokens :v :d]])]))

;; ==============================================================================
;; 3. Re-exports from einsum.models.gemma4 (Backwards Compatibility)
;; ==============================================================================

(def DEFAULT_GEMMA4_E2B_CONFIG g4/DEFAULT_GEMMA4_E2B_CONFIG)
(def DEFAULT_GEMMA4_E4B_CONFIG g4/DEFAULT_GEMMA4_E4B_CONFIG)
(def DEFAULT_GEMMA4_12B_CONFIG g4/DEFAULT_GEMMA4_12B_CONFIG)
(def DEFAULT_GEMMA4_31B_CONFIG g4/DEFAULT_GEMMA4_31B_CONFIG)

(def gemma4-config g4/gemma4-config)
(def gemma4-weight-key-map g4/gemma4-weight-key-map)
(def layer-is-global? g4/layer-is-global?)
(def gemma4-linear-proj g4/gemma4-linear-proj)
(def gemma4-layer-ast g4/gemma4-layer-ast)
(def gemma4-model-ast g4/gemma4-model-ast)
(def gemma4-relational-invars g4/gemma4-relational-invars)
(def gemma4-unbinding-ast g4/gemma4-unbinding-ast)
(def gemma4-entity-unbinding-ast g4/gemma4-entity-unbinding-ast)
(def gemma4-two-stage-resolver-ast g4/gemma4-two-stage-resolver-ast)
(def gemma4-two-stage-entity-resolver-ast g4/gemma4-two-stage-entity-resolver-ast)
(def gemma4-prefill-outvars g4/gemma4-prefill-outvars)
(def gemma4-kv-layer-ast g4/gemma4-kv-layer-ast)
(def gemma4-kv-model-ast g4/gemma4-kv-model-ast)
(def build-tensor-logic-invars g4/build-tensor-logic-invars)
(def build-gemma4-kv-invars g4/build-gemma4-kv-invars)
(def build-gemma4-kv-outvars g4/build-gemma4-kv-outvars)
(def build-gemma4-prefill-outvars g4/build-gemma4-prefill-outvars)
