(ns einsum.models.smollm
  "Declarative SmolLM-135M Architecture definition in pure Tensor Logic Hiccup AST."
  (:require [clojure.string :as str]))

(def DEFAULT_SMOLLM_CONFIG
  {:vocab-size 49152
   :n-positions 2048
   :n-embd 576
   :intermediate-size 1536
   :n-layer 30
   :n-head 9
   :n-kv-head 3
   :rms-norm-eps 1e-5})

(defn smollm-config
  "Returns SmolLM configuration map with optional custom overrides."
  ([] DEFAULT_SMOLLM_CONFIG)
  ([overrides] (merge DEFAULT_SMOLLM_CONFIG overrides)))

(defn weight-key-map
  "Maps logical SmolLM layer index `i` to HuggingFace safetensors parameter keys."
  [i]
  {:input-ln-w (format "model.layers.%d.input_layernorm.weight" i)
   :q-w (format "model.layers.%d.self_attn.q_proj.weight" i)
   :k-w (format "model.layers.%d.self_attn.k_proj.weight" i)
   :v-w (format "model.layers.%d.self_attn.v_proj.weight" i)
   :o-w (format "model.layers.%d.self_attn.o_proj.weight" i)
   :post-attn-ln-w (format "model.layers.%d.post_attention_layernorm.weight" i)
   :gate-w (format "model.layers.%d.mlp.gate_proj.weight" i)
   :up-w (format "model.layers.%d.mlp.up_proj.weight" i)
   :down-w (format "model.layers.%d.mlp.down_proj.weight" i)})

(defn smollm-alias-resolver
  "Returns a resolver function mapping SmolLM Tensor Logic AST variable keywords
   to canonical HuggingFace Safetensors parameter paths under `prefix-base` or `weights-or-header`."
  ([]
   (smollm-alias-resolver {}))
  ([header-or-prefix]
   (let [header (cond
                  (string? header-or-prefix) {}
                  (map? header-or-prefix) (or (:header header-or-prefix) header-or-prefix)
                  :else {})
         p-base (if (string? header-or-prefix)
                  (if (or (str/ends-with? header-or-prefix ".") (empty? header-or-prefix))
                    header-or-prefix
                    (str header-or-prefix "."))
                  "model.")
         p-layers (if (str/ends-with? p-base "layers.")
                    p-base
                    (str p-base "layers."))
         has-lm-head? (or (contains? header "lm_head.weight")
                          (contains? header :lm_head.weight))]
     (fn [k]
       (let [k-str (if (keyword? k) (name k) (str k))]
         (cond
           (= k-str "embed_tokens") (str p-base "embed_tokens.weight")
           (= k-str "final_norm_w") (str p-base "norm.weight")
           (= k-str "lm_head_w") (if has-lm-head? "lm_head.weight" (str p-base "embed_tokens.weight"))
           :else
           (if-let [[_ prefix idx-str] (re-matches #"^(input_ln_w|q_w|k_w|v_w|o_w|post_attn_ln_w|gate_w|up_w|down_w)_(\d+)$" k-str)]
             (let [i (Long/parseLong idx-str)
                   l-prefix (str p-layers i ".")]
               (case prefix
                 "input_ln_w" (str l-prefix "input_layernorm.weight")
                 "q_w" (str l-prefix "self_attn.q_proj.weight")
                 "k_w" (str l-prefix "self_attn.k_proj.weight")
                 "v_w" (str l-prefix "self_attn.v_proj.weight")
                 "o_w" (str l-prefix "self_attn.o_proj.weight")
                 "post_attn_ln_w" (str l-prefix "post_attention_layernorm.weight")
                 "gate_w" (str l-prefix "mlp.gate_proj.weight")
                 "up_w" (str l-prefix "mlp.up_proj.weight")
                 "down_w" (str l-prefix "mlp.down_proj.weight")
                 nil))
             k-str)))))))

(defn smollm-layer-ast
  "Generates Tensor Logic Hiccup AST for SmolLM Transformer layer block `layer-idx`."
  [layer-idx max-seq-len]
  [:block {:name [:smollm_layer layer-idx]}
   ;; 1. Pre-RMSNorm 1
   [:rms-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:input_ln_w :d] {:eps 1e-5}]

   ;; 2. Q, K, V Linear Projections
   [:= [:q :b :p :d] [:x_norm1 :b :p :d_in] [:q_w :d :d_in]]
   [:= [:k :b :p :kv_dim] [:x_norm1 :b :p :d_in] [:k_w :kv_dim :d_in]]
   [:= [:v :b :p :kv_dim] [:x_norm1 :b :p :d_in] [:v_w :kv_dim :d_in]]

   ;; 3. Rotary Position Embeddings (RoPE) on Q and K
   [:rope [:q_rope :b :p :d] [:q :b :p :d] {:head-dim 64 :theta 10000.0}]
   [:rope [:k_rope :b :p :kv_dim] [:k :b :p :kv_dim] {:head-dim 64 :theta 10000.0}]

   ;; 4. GQA Multi-Head Reshapes & Head Group Broadcasts
   [:reshape [:q_heads :b :p :h :dh] [:q_rope :b :p :d] {:shape [1 max-seq-len 9 64]}]
   [:reshape [:k_kv :b :p :kvh :dh] [:k_rope :b :p :kv_dim] {:shape [1 max-seq-len 3 64]}]
   [:= [:k_rep :b :p :kvh :g :dh] [:k_kv :b :p :kvh :dh] {:shape [1 max-seq-len 3 3 64]}]
   [:reshape [:k_heads :b :p :h :dh] [:k_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len 9 64]}]
   [:reshape [:v_kv :b :p :kvh :dh] [:v :b :p :kv_dim] {:shape [1 max-seq-len 3 64]}]
   [:= [:v_rep :b :p :kvh :g :dh] [:v_kv :b :p :kvh :dh] {:shape [1 max-seq-len 3 3 64]}]
   [:reshape [:v_heads :b :p :h :dh] [:v_rep :b :p :kvh :g :dh] {:shape [1 max-seq-len 9 64]}]

   ;; 5. Scaled Dot-Product Attention: QK^T / sqrt(64) -> Causal Softmax -> probs @ V
   [:= [:scores :b :h :p-q :p-k] {:scale 0.125} [:q_heads :b :p-q :h :dh] [:k_heads :b :p-k :h :dh]]
   [:causal-softmax [:probs :b :h :p-q :p-k] [:scores :b :h :p-q :p-k]]
   [:= [:ctx :b :p-q :h :dh] [:probs :b :h :p-q :p-k] [:v_heads :b :p-k :h :dh]]
   [:reshape [:ctx_flat :b :p :d] [:ctx :b :p-q :h :dh] {:shape [1 max-seq-len 576]}]

   ;; 6. Output Projection
   [:= [:attn_out :b :p :d] [:ctx_flat :b :p :d_in] [:o_w :d :d_in]]

   ;; 7. Residual Connection 1
   [:= [:res1 :b :p :d] [:h :b :p :d]]
   [:= [:res1 :b :p :d] [:attn_out :b :p :d]]

   ;; 8. Pre-RMSNorm 2
   [:rms-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:post_attn_ln_w :d] {:eps 1e-5}]

   ;; 9. SwiGLU MLP Block: down_proj(silu(gate_proj(x)) * up_proj(x))
   [:= [:gate :b :p :dff] {:act :silu} [:x_norm2 :b :p :d] [:gate_w :dff :d]]
   [:= [:up :b :p :dff] [:x_norm2 :b :p :d] [:up_w :dff :d]]
   [:= [:mlp_act :b :p :dff] [:gate :b :p :dff] [:up :b :p :dff]]
   [:= [:mlp_out :b :p :d] [:mlp_act :b :p :dff] [:down_w :d :dff]]

   ;; 10. Residual Connection 2
   [:= [:h# :b :p :d] [:res1 :b :p :d]]
   [:= [:h# :b :p :d] [:mlp_out :b :p :d]]])

(defn smollm-model-ast
  "Generates full SmolLM-135M model forward pass in pure Tensor Logic Hiccup AST."
  [{:keys [num-layers max-seq-len] :or {num-layers 30 max-seq-len 128}}]
  [:block {:name :full_smollm_model}
   ;; 1. Token Embedding Lookup
   [:gather [[:h 0] :b :p :d] [:embed_tokens :v :d] [:x :b :p]]

   ;; 2. 30 Transformer Blocks
   (mapv #(smollm-layer-ast % max-seq-len) (range num-layers))

   ;; 3. Final RMSNorm
   [:rms-norm [:normed :b :p :d] [[:h num-layers] :b :p :d] [:final_norm_w :d] {:eps 1e-5}]

   ;; 4. LM Head Projection to vocabulary
   [:= [:logits :b :p :v] [:normed :b :p :d] [:lm_head_w :v :d]]])
