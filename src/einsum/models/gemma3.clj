(ns einsum.models.gemma3
  "Declarative Gemma 3 Architecture definition in pure Tensor Logic Hiccup AST."
  (:require [clojure.string :as str]))

(def DEFAULT_GEMMA3_270M_CONFIG
  {:hidden-dim 640
   :intermediate-dim 2048
   :num-layers 18
   :num-heads 4
   :num-kv-heads 1
   :head-dim 256
   :query-pre-attn-scalar 256
   :vocab-size 262144
   :norm-eps 1e-6})

(defn gemma3-config
  "Returns Gemma 3 configuration map with optional custom overrides."
  ([] DEFAULT_GEMMA3_270M_CONFIG)
  ([overrides] (merge DEFAULT_GEMMA3_270M_CONFIG overrides)))

(defn gemma3-weight-key-map
  "Returns Gemma 3 safetensors weight key mapping for layer `layer-idx` including QK norm parameters."
  [layer-idx]
  (let [prefix (str "model.layers." layer-idx ".")]
    {:input-ln-w     (str prefix "input_layernorm.weight")
     :q-w            (str prefix "self_attn.q_proj.weight")
     :k-w            (str prefix "self_attn.k_proj.weight")
     :v-w            (str prefix "self_attn.v_proj.weight")
     :o-w            (str prefix "self_attn.o_proj.weight")
     :q-norm-w       (str prefix "self_attn.q_norm.weight")
     :k-norm-w       (str prefix "self_attn.k_norm.weight")
     :post-attn-ln-w (str prefix "post_attention_layernorm.weight")
     :pre-mlp-ln-w   (str prefix "pre_feedforward_layernorm.weight")
     :post-mlp-ln-w  (str prefix "post_feedforward_layernorm.weight")
     :gate-w         (str prefix "mlp.gate_proj.weight")
     :up-w           (str prefix "mlp.up_proj.weight")
     :down-w         (str prefix "mlp.down_proj.weight")}))

(defn gemma3-alias-resolver
  "Returns a resolver function mapping Gemma 3 Tensor Logic AST variable keywords
   to canonical HuggingFace Safetensors parameter paths under `prefix-base`."
  ([]
   (gemma3-alias-resolver "model."))
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
           (= k-str "final_norm_w") (str p-base "norm.weight")
           :else
           (if-let [[_ prefix idx-str] (re-matches #"^(input_ln_w|q_w|k_w|v_w|o_w|q_norm_w|k_norm_w|post_attn_ln_w|pre_mlp_ln_w|post_mlp_ln_w|gate_w|up_w|down_w)_(\d+)$" k-str)]
             (let [i (Long/parseLong idx-str)
                   l-prefix (str p-layers i ".")]
               (case prefix
                 "input_ln_w" (str l-prefix "input_layernorm.weight")
                 "q_w" (str l-prefix "self_attn.q_proj.weight")
                 "k_w" (str l-prefix "self_attn.k_proj.weight")
                 "v_w" (str l-prefix "self_attn.v_proj.weight")
                 "o_w" (str l-prefix "self_attn.o_proj.weight")
                 "q_norm_w" (str l-prefix "self_attn.q_norm.weight")
                 "k_norm_w" (str l-prefix "self_attn.k_norm.weight")
                 "post_attn_ln_w" (str l-prefix "post_attention_layernorm.weight")
                 "pre_mlp_ln_w" (str l-prefix "pre_feedforward_layernorm.weight")
                 "post_mlp_ln_w" (str l-prefix "post_feedforward_layernorm.weight")
                 "gate_w" (str l-prefix "mlp.gate_proj.weight")
                 "up_w" (str l-prefix "mlp.up_proj.weight")
                 "down_w" (str l-prefix "mlp.down_proj.weight")
                 nil))
             k-str)))))))

(defn gemma3-layer-ast
  "Generates Tensor Logic Hiccup AST for Gemma 3 Transformer layer block `layer-idx`."
  [layer-idx max-seq-len config]
  (let [i layer-idx
        {:keys [num-heads num-kv-heads head-dim layer-types norm-eps query-pre-attn-scalar]} config
        num-heads (long (or num-heads 4))
        num-kv-heads (long (or num-kv-heads 1))
        head-dim (long (or head-dim 256))
        q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        group-size (quot num-heads num-kv-heads)
        norm-eps (double (or norm-eps 1e-6))

        l-type (when layer-types (nth layer-types i nil))
        is-global? (or (= l-type "full_attention")
                       (= l-type :full_attention)
                       (zero? (mod (inc i) 6)))
        theta (if is-global? 1000000.0 10000.0)
        window (if is-global? nil 512)
        scale (/ 1.0 (Math/sqrt (double (or query-pre-attn-scalar head-dim))))]
    [:block {:name [:gemma3_layer i]}
     ;; 1. Pre-Attention RMSNorm (Gemma 3 uses 1 + w)
     [:rms-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:input_ln_w :d] {:eps norm-eps :gemma? true}]

     ;; 2. Q Projection & Q-Norm
     [:= [:q_raw :b :p :qd] [:x_norm1 :b :p :d] [:q_w :qd :d]]
     [:reshape [:q_heads_raw :b :p :h :dh] [:q_raw :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]
     [:rms-norm [:q_normed_4d :b :p :h :dh] [:q_heads_raw :b :p :h :dh] [:q_norm_w :dh] {:eps norm-eps :gemma? true}]
     [:reshape [:q_normed_3d :b :p :qd] [:q_normed_4d :b :p :h :dh] {:shape [1 max-seq-len q-dim]}]
     [:rope [:q_rope :b :p :qd] [:q_normed_3d :b :p :qd] {:head-dim head-dim :theta theta :rope-proportion 1.0}]
     [:reshape [:q_ro :b :p :h :dh] [:q_rope :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]

     ;; 3. K, V Projections & K-Norm
     [:= [:k_raw :b :p :kvd] [:x_norm1 :b :p :d] [:k_w :kvd :d]]
     [:= [:v_raw :b :p :kvd] [:x_norm1 :b :p :d] [:v_w :kvd :d]]
     [:reshape [:k_heads_raw :b :p :kvh :dh] [:k_raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
     [:reshape [:v_heads :b :p :kvh :dh] [:v_raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
     [:rms-norm [:k_normed_4d :b :p :kvh :dh] [:k_heads_raw :b :p :kvh :dh] [:k_norm_w :dh] {:eps norm-eps :gemma? true}]
     [:reshape [:k_normed_3d :b :p :kvd] [:k_normed_4d :b :p :kvh :dh] {:shape [1 max-seq-len kv-dim]}]
     [:rope [:k_rope :b :p :kvd] [:k_normed_3d :b :p :kvd] {:head-dim head-dim :theta theta :rope-proportion 1.0}]
     [:reshape [:k_ro :b :p :kvh :dh] [:k_rope :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]

     ;; 4. Broadcast KV Heads
     [:= [:k_rep :b :p :kvh :g :dh] [:k_ro :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
     [:reshape [:k_heads :b :p :h :dh] [:k_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]
     [:= [:v_rep :b :p :kvh :g :dh] [:v_heads :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
     [:reshape [:v_rep_heads :b :p :h :dh] [:v_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]

     ;; 5. Scaled Dot-Product Attention: QK^T -> Causal Softmax -> Context -> Out
     [:= [:scores :b :h :p-q :p-k] {:scale scale} [:q_ro :b :p-q :h :dh] [:k_heads :b :p-k :h :dh]]
     [:causal-softmax [:probs :b :h :p-q :p-k] [:scores :b :h :p-q :p-k] (if window {:sliding-window window} {})]
     [:= [:ctx :b :p-q :h :dh] [:probs :b :h :p-q :p-k] [:v_rep_heads :b :p-k :h :dh]]
     [:reshape [:ctx_flat :b :p :qd] [:ctx :b :p-q :h :dh] {:shape [1 max-seq-len q-dim]}]

     ;; 6. Output Projection & Post-Attention RMSNorm
     [:= [:attn_raw :b :p :d] [:ctx_flat :b :p :qd] [:o_w :d :qd]]
     [:rms-norm [:attn_normed :b :p :d] [:attn_raw :b :p :d] [:post_attn_ln_w :d] {:eps norm-eps :gemma? true}]

     ;; 7. Residual Connection 1
     [:= [:res1 :b :p :d] [:h :b :p :d]]
     [:= [:res1 :b :p :d] [:attn_normed :b :p :d]]

     ;; 8. Pre-MLP RMSNorm
     [:rms-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:pre_mlp_ln_w :d] {:eps norm-eps :gemma? true}]

     ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
     [:= [:gate :b :p :dff] {:act :gelu} [:x_norm2 :b :p :d] [:gate_w :dff :d]]
     [:= [:up :b :p :dff] [:x_norm2 :b :p :d] [:up_w :dff :d]]
     [:= [:mlp_act :b :p :dff] [:gate :b :p :dff] [:up :b :p :dff]]
     [:= [:mlp_raw :b :p :d] [:mlp_act :b :p :dff] [:down_w :d :dff]]
     [:rms-norm [:mlp_normed :b :p :d] [:mlp_raw :b :p :d] [:post_mlp_ln_w :d] {:eps norm-eps :gemma? true}]

     ;; 10. Residual Connection 2
     [:= [:h# :b :p :d] [:res1 :b :p :d]]
     [:= [:h# :b :p :d] [:mlp_normed :b :p :d]]]))

(defn gemma3-model-ast
  "Generates full Gemma 3 model forward pass in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge (gemma3-config) config)
        {:keys [num-layers max-seq-len hidden-dim norm-eps final-logit-softcap]} cfg
        num-layers (long (or num-layers 18))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 640))
        norm-eps (double (or norm-eps 1e-6))]
    [:block {:name :full_gemma3_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [[:h 0] :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Sequential Transformer Layer Blocks
     (mapv (fn [i] (gemma3-layer-ast i max-seq-len cfg))
           (range num-layers))

     ;; 3. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [[:h num-layers] :b :p :d] [:final_norm_w :d] {:eps norm-eps :gemma? true}]

     ;; 4. Tied LM Head with optional final logit softcapping
     [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                              {:softcap (double final-logit-softcap)}
                              {})
      [:normed :b :p :d] [:embed_tokens :v :d]]]))
