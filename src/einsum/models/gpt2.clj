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
  [:block {:name [:gpt2_layer layer-idx]}
   ;; 1. Pre-LayerNorm 1
   [:layer-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:ln1_g :d] [:ln1_b :d]]

   ;; 2. QKV Projection with bias
   [:= [:qkv :b :p :qkv_dim] [:x_norm1 :b :p :d] [:attn_w :d :qkv_dim]]
   [:= [:qkv :b :p :qkv_dim] [:attn_b :qkv_dim]]

   ;; 3. Slice into Q, K, V
   [:slice [:q :b :p :d] [:qkv :b :p :qkv_dim] {:start_indices [0 0 0] :limit_indices [1 max-seq-len 768] :strides [1 1 1] :shape [1 max-seq-len 768]}]
   [:slice [:k :b :p :d] [:qkv :b :p :qkv_dim] {:start_indices [0 0 768] :limit_indices [1 max-seq-len 1536] :strides [1 1 1] :shape [1 max-seq-len 768]}]
   [:slice [:v :b :p :d] [:qkv :b :p :qkv_dim] {:start_indices [0 0 1536] :limit_indices [1 max-seq-len 2304] :strides [1 1 1] :shape [1 max-seq-len 768]}]

   ;; 4. Reshape to multi-head (12 heads, head-dim 64)
   [:reshape [:q_heads :b :p :h :dh] [:q :b :p :d] {:shape [1 max-seq-len 12 64]}]
   [:reshape [:k_heads :b :p :h :dh] [:k :b :p :d] {:shape [1 max-seq-len 12 64]}]
   [:reshape [:v_heads :b :p :h :dh] [:v :b :p :d] {:shape [1 max-seq-len 12 64]}]

   ;; 5. QK^T batched contraction scaled by 1/sqrt(64) = 0.125
   [:= [:scores :b :h :p-q :p-k] {:scale 0.125} [:q_heads :b :p-q :h :dh] [:k_heads :b :p-k :h :dh]]

   ;; 6. Causal Mask and Softmax
   [:causal-softmax [:probs :b :h :p-q :p-k] [:scores :b :h :p-q :p-k]]

   ;; 7. Attention context contraction: probs @ v
   [:= [:ctx :b :p-q :h :dh] [:probs :b :h :p-q :p-k] [:v_heads :b :p-k :h :dh]]

   ;; 8. Reshape context back to [1 max-seq-len 768]
   [:reshape [:ctx_flat :b :p :d] [:ctx :b :p-q :h :dh] {:shape [1 max-seq-len 768]}]

   ;; 9. Output projection with bias
   [:= [:attn_out :b :p :d] [:ctx_flat :b :p :d_in] [:proj_w :d_in :d]]
   [:= [:attn_out :b :p :d] [:proj_b :d]]

   ;; 10. Residual skip connection 1
   [:= [:res1 :b :p :d] [:h :b :p :d]]
   [:= [:res1 :b :p :d] [:attn_out :b :p :d]]

   ;; 11. Pre-LayerNorm 2
   [:layer-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:ln2_g :d] [:ln2_b :d]]

   ;; 12. MLP: fc projection -> GELU -> proj projection
   [:= [:mlp_fc :b :p :dff] [:x_norm2 :b :p :d] [:mlp_fc_w :d :dff]]
   [:= [:mlp_fc :b :p :dff] [:mlp_fc_b :dff]]
   [:= [:mlp_act :b :p :dff] {:act :gelu} [:mlp_fc :b :p :dff]]
   [:= [:mlp_out :b :p :d] [:mlp_act :b :p :dff] [:mlp_proj_w :dff :d]]
   [:= [:mlp_out :b :p :d] [:mlp_proj_b :d]]

   ;; 13. Residual skip connection 2
   [:= [:h# :b :p :d] [:res1 :b :p :d]]
   [:= [:h# :b :p :d] [:mlp_out :b :p :d]]])

(defn gpt2-model-ast
  "Generates full GPT-2 model forward pass in pure Tensor Logic Hiccup AST."
  [{:keys [num-layers max-seq-len] :or {num-layers 12 max-seq-len 128}}]
  [:block {:name :full_gpt2_model}
   ;; 1. Token & Position Embedding Lookups
   [:gather [:tok_emb :b :p :d] [:wte :v :d] [:x :b :p]]
   [:gather [:pos_emb :b :p :d] [:wpe :max_pos :d] [:pos_ids :b :p]]
   [:= [[:h 0] :b :p :d] [:tok_emb :b :p :d]]
   [:= [[:h 0] :b :p :d] [:pos_emb :b :p :d]]

   ;; 2. 12 Transformer Blocks
   (mapv #(gpt2-layer-ast % max-seq-len) (range num-layers))

   ;; 3. Final LayerNorm
   [:layer-norm [:normed :b :p :d] [[:h num-layers] :b :p :d] [:ln_f_g :d] [:ln_f_b :d]]

   ;; 4. LM Head Projection to vocabulary
   [:= [:logits :b :p :v] [:normed :b :p :d] [:wte :v :d]]])

