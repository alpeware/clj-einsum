(ns einsum.models.smollm
  "Declarative SmolLM-135M Architecture definition in pure Tensor Logic Hiccup AST.")

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

(defn smollm-layer-ast
  "Generates Tensor Logic Hiccup AST for SmolLM Transformer layer block `layer-idx`."
  [layer-idx max-seq-len]
  (let [i layer-idx
        h-in (keyword (str "h" i))
        h-out (keyword (str "h" (inc i)))
        input-ln-w (keyword (str "input_ln_w_" i))
        q-w (keyword (str "q_w_" i))
        k-w (keyword (str "k_w_" i))
        v-w (keyword (str "v_w_" i))
        o-w (keyword (str "o_w_" i))
        post-attn-ln-w (keyword (str "post_attn_ln_w_" i))
        gate-w (keyword (str "gate_w_" i))
        up-w (keyword (str "up_w_" i))
        down-w (keyword (str "down_w_" i))

        x-norm1 (keyword (str "x_norm1_" i))
        q (keyword (str "q_" i))
        k (keyword (str "k_" i))
        v (keyword (str "v_" i))
        q-rope (keyword (str "q_rope_" i))
        k-rope (keyword (str "k_rope_" i))
        q-heads (keyword (str "q_heads_" i))
        k-kv (keyword (str "k_kv_" i))
        k-rep (keyword (str "k_rep_" i))
        k-heads (keyword (str "k_heads_" i))
        v-kv (keyword (str "v_kv_" i))
        v-rep (keyword (str "v_rep_" i))
        v-heads (keyword (str "v_heads_" i))
        scores (keyword (str "scores_" i))
        probs (keyword (str "probs_" i))
        ctx (keyword (str "ctx_" i))
        ctx-flat (keyword (str "ctx_flat_" i))
        attn-out (keyword (str "attn_out_" i))
        res1 (keyword (str "res1_" i))
        x-norm2 (keyword (str "x_norm2_" i))
        gate (keyword (str "gate_" i))
        up (keyword (str "up_" i))
        mlp-act (keyword (str "mlp_act_" i))
        mlp-out (keyword (str "mlp_out_" i))]
    [:block {:name (keyword (str "smollm_layer_" i))}
     ;; 1. Pre-RMSNorm 1
     [:rms-norm [x-norm1 :b :p :d] [h-in :b :p :d] [input-ln-w :d] {:eps 1e-5}]

     ;; 2. Q, K, V Linear Projections
     [:= [q :b :p :d] [x-norm1 :b :p :d_in] [q-w :d :d_in]]
     [:= [k :b :p :kv_dim] [x-norm1 :b :p :d_in] [k-w :kv_dim :d_in]]
     [:= [v :b :p :kv_dim] [x-norm1 :b :p :d_in] [v-w :kv_dim :d_in]]

     ;; 3. Rotary Position Embeddings (RoPE) on Q and K
     [:rope [q-rope :b :p :d] [q :b :p :d] {:head-dim 64 :theta 10000.0}]
     [:rope [k-rope :b :p :kv_dim] [k :b :p :kv_dim] {:head-dim 64 :theta 10000.0}]

     ;; 4. GQA Multi-Head Reshapes & Head Group Broadcasts
     [:reshape [q-heads :b :p :h :dh] [q-rope :b :p :d] {:shape [1 max-seq-len 9 64]}]
     [:reshape [k-kv :b :p :kvh :dh] [k-rope :b :p :kv_dim] {:shape [1 max-seq-len 3 64]}]
     [:= [k-rep :b :p :kvh :g :dh] [k-kv :b :p :kvh :dh] {:shape [1 max-seq-len 3 3 64]}]
     [:reshape [k-heads :b :p :h :dh] [k-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len 9 64]}]
     [:reshape [v-kv :b :p :kvh :dh] [v :b :p :kv_dim] {:shape [1 max-seq-len 3 64]}]
     [:= [v-rep :b :p :kvh :g :dh] [v-kv :b :p :kvh :dh] {:shape [1 max-seq-len 3 3 64]}]
     [:reshape [v-heads :b :p :h :dh] [v-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len 9 64]}]

     ;; 5. Scaled Dot-Product Attention: QK^T / sqrt(64) -> Causal Softmax -> probs @ V
     [:= [scores :b :h :p-q :p-k] {:scale 0.125} [q-heads :b :p-q :h :dh] [k-heads :b :p-k :h :dh]]
     [:causal-softmax [probs :b :h :p-q :p-k] [scores :b :h :p-q :p-k]]
     [:= [ctx :b :p-q :h :dh] [probs :b :h :p-q :p-k] [v-heads :b :p-k :h :dh]]
     [:reshape [ctx-flat :b :p :d] [ctx :b :p-q :h :dh] {:shape [1 max-seq-len 576]}]

     ;; 6. Output Projection
     [:= [attn-out :b :p :d] [ctx-flat :b :p :d_in] [o-w :d :d_in]]

     ;; 7. Residual Connection 1
     [:= [res1 :b :p :d] [h-in :b :p :d]]
     [:= [res1 :b :p :d] [attn-out :b :p :d]]

     ;; 8. Pre-RMSNorm 2
     [:rms-norm [x-norm2 :b :p :d] [res1 :b :p :d] [post-attn-ln-w :d] {:eps 1e-5}]

     ;; 9. SwiGLU MLP Block: down_proj(silu(gate_proj(x)) * up_proj(x))
     [:= [gate :b :p :dff] {:act :silu} [x-norm2 :b :p :d] [gate-w :dff :d]]
     [:= [up :b :p :dff] [x-norm2 :b :p :d] [up-w :dff :d]]
     [:= [mlp-act :b :p :dff] [gate :b :p :dff] [up :b :p :dff]]
     [:= [mlp-out :b :p :d] [mlp-act :b :p :dff] [down-w :d :dff]]

     ;; 10. Residual Connection 2
     [:= [h-out :b :p :d] [res1 :b :p :d]]
     [:= [h-out :b :p :d] [mlp-out :b :p :d]]]))

(defn smollm-model-ast
  "Generates full SmolLM-135M model forward pass in pure Tensor Logic Hiccup AST."
  [{:keys [num-layers max-seq-len] :or {num-layers 30 max-seq-len 128}}]
  (let [h-final (keyword (str "h" num-layers))]
    [:block {:name :full_smollm_model}
     ;; 1. Token Embedding Lookup
     [:gather [:h0 :b :p :d] [:embed_tokens :v :d] [:x :b :p]]

     ;; 2. 30 Transformer Blocks
     (mapv #(smollm-layer-ast % max-seq-len) (range num-layers))

     ;; 3. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps 1e-5}]

     ;; 4. LM Head Projection to vocabulary
     [:= [:logits :b :p :v] [:normed :b :p :d] [:lm_head_w :v :d]]]))
