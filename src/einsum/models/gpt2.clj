(ns einsum.models.gpt2
  "Declarative GPT-2 Architecture definition in pure Tensor Logic Hiccup AST.")

(def DEFAULT_GPT2_CONFIG
  {:vocab-size 50257
   :n-positions 1024
   :n-embd 768
   :n-layer 12
   :n-head 12
   :layer-norm-epsilon 1e-5})

(defn gpt2-config
  "Returns GPT-2 configuration map with optional custom overrides."
  ([] DEFAULT_GPT2_CONFIG)
  ([overrides] (merge DEFAULT_GPT2_CONFIG overrides)))

(defn weight-key-map
  "Maps logical GPT-2 layer index `i` to HuggingFace safetensors parameter keys."
  [i]
  {:ln1-g (format "h.%d.ln_1.weight" i)
   :ln1-b (format "h.%d.ln_1.bias" i)
   :c-attn-w (format "h.%d.attn.c_attn.weight" i)
   :c-attn-b (format "h.%d.attn.c_attn.bias" i)
   :c-proj-w (format "h.%d.attn.c_proj.weight" i)
   :c-proj-b (format "h.%d.attn.c_proj.bias" i)
   :ln2-g (format "h.%d.ln_2.weight" i)
   :ln2-b (format "h.%d.ln_2.bias" i)
   :mlp-fc-w (format "h.%d.mlp.c_fc.weight" i)
   :mlp-fc-b (format "h.%d.mlp.c_fc.bias" i)
   :mlp-proj-w (format "h.%d.mlp.c_proj.weight" i)
   :mlp-proj-b (format "h.%d.mlp.c_proj.bias" i)})

(defn gpt2-layer-ast
  "Generates Tensor Logic Hiccup AST for GPT-2 Transformer layer block `layer-idx`."
  [layer-idx max-seq-len]
  (let [i layer-idx
        h-in (keyword (str "h" i))
        h-out (keyword (str "h" (inc i)))
        ln1-g (keyword (str "ln1_g_" i))
        ln1-b (keyword (str "ln1_b_" i))
        attn-w (keyword (str "attn_w_" i))
        attn-b (keyword (str "attn_b_" i))
        proj-w (keyword (str "proj_w_" i))
        proj-b (keyword (str "proj_b_" i))
        ln2-g (keyword (str "ln2_g_" i))
        ln2-b (keyword (str "ln2_b_" i))
        mlp-fc-w (keyword (str "mlp_fc_w_" i))
        mlp-fc-b (keyword (str "mlp_fc_b_" i))
        mlp-proj-w (keyword (str "mlp_proj_w_" i))
        mlp-proj-b (keyword (str "mlp_proj_b_" i))

        x-norm1 (keyword (str "x_norm1_" i))
        qkv (keyword (str "qkv_" i))
        q (keyword (str "q_" i))
        k (keyword (str "k_" i))
        v (keyword (str "v_" i))
        q-heads (keyword (str "q_heads_" i))
        k-heads (keyword (str "k_heads_" i))
        v-heads (keyword (str "v_heads_" i))
        scores (keyword (str "scores_" i))
        probs (keyword (str "probs_" i))
        ctx (keyword (str "ctx_" i))
        ctx-flat (keyword (str "ctx_flat_" i))
        attn-out (keyword (str "attn_out_" i))
        res1 (keyword (str "res1_" i))
        x-norm2 (keyword (str "x_norm2_" i))
        mlp-fc (keyword (str "mlp_fc_" i))
        mlp-act (keyword (str "mlp_act_" i))
        mlp-out (keyword (str "mlp_out_" i))]
    [:block {:name (keyword (str "gpt2_layer_" i))}
     ;; 1. Pre-LayerNorm 1
     [:layer-norm [x-norm1 :b :p :d] [h-in :b :p :d] [ln1-g :d] [ln1-b :d]]

     ;; 2. QKV Projection with bias
     [:= [qkv :b :p :qkv_dim] [x-norm1 :b :p :d] [attn-w :d :qkv_dim]]
     [:= [qkv :b :p :qkv_dim] [attn-b :qkv_dim]]

     ;; 3. Slice into Q, K, V
     [:slice [q :b :p :d] [qkv :b :p :qkv_dim] {:start_indices [0 0 0] :limit_indices [1 max-seq-len 768] :strides [1 1 1] :shape [1 max-seq-len 768]}]
     [:slice [k :b :p :d] [qkv :b :p :qkv_dim] {:start_indices [0 0 768] :limit_indices [1 max-seq-len 1536] :strides [1 1 1] :shape [1 max-seq-len 768]}]
     [:slice [v :b :p :d] [qkv :b :p :qkv_dim] {:start_indices [0 0 1536] :limit_indices [1 max-seq-len 2304] :strides [1 1 1] :shape [1 max-seq-len 768]}]

     ;; 4. Reshape to multi-head (12 heads, head-dim 64)
     [:reshape [q-heads :b :p :h :dh] [q :b :p :d] {:shape [1 max-seq-len 12 64]}]
     [:reshape [k-heads :b :p :h :dh] [k :b :p :d] {:shape [1 max-seq-len 12 64]}]
     [:reshape [v-heads :b :p :h :dh] [v :b :p :d] {:shape [1 max-seq-len 12 64]}]

     ;; 5. QK^T batched contraction scaled by 1/sqrt(64) = 0.125
     [:= [scores :b :h :p-q :p-k] {:scale 0.125} [q-heads :b :p-q :h :dh] [k-heads :b :p-k :h :dh]]

     ;; 6. Causal Mask and Softmax
     [:causal-softmax [probs :b :h :p-q :p-k] [scores :b :h :p-q :p-k]]

     ;; 7. Attention context contraction: probs @ v
     [:= [ctx :b :p-q :h :dh] [probs :b :h :p-q :p-k] [v-heads :b :p-k :h :dh]]

     ;; 8. Reshape context back to [1 max-seq-len 768]
     [:reshape [ctx-flat :b :p :d] [ctx :b :p-q :h :dh] {:shape [1 max-seq-len 768]}]

     ;; 9. Output projection with bias
     [:= [attn-out :b :p :d] [ctx-flat :b :p :d_in] [proj-w :d_in :d]]
     [:= [attn-out :b :p :d] [proj-b :d]]

     ;; 10. Residual skip connection 1
     [:= [res1 :b :p :d] [h-in :b :p :d]]
     [:= [res1 :b :p :d] [attn-out :b :p :d]]

     ;; 11. Pre-LayerNorm 2
     [:layer-norm [x-norm2 :b :p :d] [res1 :b :p :d] [ln2-g :d] [ln2-b :d]]

     ;; 12. MLP: fc projection -> GELU -> proj projection
     [:= [mlp-fc :b :p :dff] [x-norm2 :b :p :d] [mlp-fc-w :d :dff]]
     [:= [mlp-fc :b :p :dff] [mlp-fc-b :dff]]
     [:= [mlp-act :b :p :dff] {:act :gelu} [mlp-fc :b :p :dff]]
     [:= [mlp-out :b :p :d] [mlp-act :b :p :dff] [mlp-proj-w :dff :d]]
     [:= [mlp-out :b :p :d] [mlp-proj-b :d]]

     ;; 13. Residual skip connection 2
     [:= [h-out :b :p :d] [res1 :b :p :d]]
     [:= [h-out :b :p :d] [mlp-out :b :p :d]]]))

(defn gpt2-model-ast
  "Generates full GPT-2 model forward pass in pure Tensor Logic Hiccup AST."
  [{:keys [num-layers max-seq-len] :or {num-layers 12 max-seq-len 128}}]
  (let [h-final (keyword (str "h" num-layers))]
    [:block {:name :full_gpt2_model}
     ;; 1. Token & Position Embedding Lookups
     [:gather [:tok_emb :b :p :d] [:wte :v :d] [:x :b :p]]
     [:gather [:pos_emb :b :p :d] [:wpe :max_pos :d] [:pos_ids :b :p]]
     [:= [:h0 :b :p :d] [:tok_emb :b :p :d]]
     [:= [:h0 :b :p :d] [:pos_emb :b :p :d]]

     ;; 2. 12 Transformer Blocks
     (mapv #(gpt2-layer-ast % max-seq-len) (range num-layers))

     ;; 3. Final LayerNorm
     [:layer-norm [:normed :b :p :d] [h-final :b :p :d] [:ln_f_g :d] [:ln_f_b :d]]

     ;; 4. LM Head Projection to vocabulary
     [:= [:logits :b :p :v] [:normed :b :p :d] [:wte :v :d]]]))
