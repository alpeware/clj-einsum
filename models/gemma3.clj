(ns models.gemma3
  "Declarative Gemma 3 Architecture definition in pure Tensor Logic Hiccup AST.")

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
        scale (/ 1.0 (Math/sqrt (double (or query-pre-attn-scalar head-dim))))

        h-in (keyword (str "h" i))
        h-out (keyword (str "h" (inc i)))

        input-ln-w (keyword (str "input_ln_w_" i))
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

        x-norm1 (keyword (str "x_norm1_" i))
        q-raw (keyword (str "q_raw_" i))
        k-raw (keyword (str "k_raw_" i))
        v-raw (keyword (str "v_raw_" i))
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
        mlp-normed (keyword (str "mlp_normed_" i))]

    [:block {:name (keyword (str "gemma3_layer_" i))}
     ;; 1. Pre-Attention RMSNorm (Gemma 3 uses 1 + w)
     [:rms-norm [x-norm1 :b :p :d] [h-in :b :p :d] [input-ln-w :d] {:eps norm-eps :gemma? true}]

     ;; 2. Q Projection & Q-Norm
     [:= [q-raw :b :p :qd] [x-norm1 :b :p :d] [q-w :qd :d]]
     [:reshape [q-heads-raw :b :p :h :dh] [q-raw :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]
     [:rms-norm [q-normed-4d :b :p :h :dh] [q-heads-raw :b :p :h :dh] [q-norm-w :dh] {:eps norm-eps :gemma? true}]
     [:reshape [q-normed-3d :b :p :qd] [q-normed-4d :b :p :h :dh] {:shape [1 max-seq-len q-dim]}]
     [:rope [q-rope :b :p :qd] [q-normed-3d :b :p :qd] {:head-dim head-dim :theta theta :rope-proportion 1.0}]
     [:reshape [q-ro :b :p :h :dh] [q-rope :b :p :qd] {:shape [1 max-seq-len num-heads head-dim]}]

     ;; 3. K, V Projections & K-Norm
     [:= [k-raw :b :p :kvd] [x-norm1 :b :p :d] [k-w :kvd :d]]
     [:= [v-raw :b :p :kvd] [x-norm1 :b :p :d] [v-w :kvd :d]]
     [:reshape [k-heads-raw :b :p :kvh :dh] [k-raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
     [:reshape [v-heads :b :p :kvh :dh] [v-raw :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]
     [:rms-norm [k-normed-4d :b :p :kvh :dh] [k-heads-raw :b :p :kvh :dh] [k-norm-w :dh] {:eps norm-eps :gemma? true}]
     [:reshape [k-normed-3d :b :p :kvd] [k-normed-4d :b :p :kvh :dh] {:shape [1 max-seq-len kv-dim]}]
     [:rope [k-rope :b :p :kvd] [k-normed-3d :b :p :kvd] {:head-dim head-dim :theta theta :rope-proportion 1.0}]
     [:reshape [k-ro :b :p :kvh :dh] [k-rope :b :p :kvd] {:shape [1 max-seq-len num-kv-heads head-dim]}]

     ;; 4. Broadcast KV Heads
     [:= [k-rep :b :p :kvh :g :dh] [k-ro :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
     [:reshape [k-heads :b :p :h :dh] [k-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]
     [:= [v-rep :b :p :kvh :g :dh] [v-heads :b :p :kvh :dh] {:shape [1 max-seq-len num-kv-heads group-size head-dim]}]
     [:reshape [v-rep-heads :b :p :h :dh] [v-rep :b :p :kvh :g :dh] {:shape [1 max-seq-len num-heads head-dim]}]

     ;; 5. Scaled Dot-Product Attention: QK^T -> Causal Softmax -> Context -> Out
     [:= [scores :b :h :p-q :p-k] {:scale scale} [q-ro :b :p-q :h :dh] [k-heads :b :p-k :h :dh]]
     [:causal-softmax [probs :b :h :p-q :p-k] [scores :b :h :p-q :p-k] (if window {:sliding-window window} {})]
     [:= [ctx :b :p-q :h :dh] [probs :b :h :p-q :p-k] [v-rep-heads :b :p-k :h :dh]]
     [:reshape [ctx-flat :b :p :qd] [ctx :b :p-q :h :dh] {:shape [1 max-seq-len q-dim]}]

     ;; 6. Output Projection & Post-Attention RMSNorm
     [:= [attn-raw :b :p :d] [ctx-flat :b :p :qd] [o-w :d :qd]]
     [:rms-norm [attn-normed :b :p :d] [attn-raw :b :p :d] [post-attn-ln-w :d] {:eps norm-eps :gemma? true}]

     ;; 7. Residual Connection 1
     [:= [res1 :b :p :d] [h-in :b :p :d]]
     [:= [res1 :b :p :d] [attn-normed :b :p :d]]

     ;; 8. Pre-MLP RMSNorm
     [:rms-norm [x-norm2 :b :p :d] [res1 :b :p :d] [pre-mlp-ln-w :d] {:eps norm-eps :gemma? true}]

     ;; 9. GeGLU MLP Block: down_proj(gelu(gate_proj(x)) * up_proj(x))
     [:= [gate :b :p :dff] {:act :gelu} [x-norm2 :b :p :d] [gate-w :dff :d]]
     [:= [up :b :p :dff] [x-norm2 :b :p :d] [up-w :dff :d]]
     [:= [mlp-act :b :p :dff] [gate :b :p :dff] [up :b :p :dff]]
     [:= [mlp-raw :b :p :d] [mlp-act :b :p :dff] [down-w :d :dff]]
     [:rms-norm [mlp-normed :b :p :d] [mlp-raw :b :p :d] [post-mlp-ln-w :d] {:eps norm-eps :gemma? true}]

     ;; 10. Residual Connection 2
     [:= [h-out :b :p :d] [res1 :b :p :d]]
     [:= [h-out :b :p :d] [mlp-normed :b :p :d]]]))

(defn gemma3-model-ast
  "Generates full Gemma 3 model forward pass in pure Tensor Logic Hiccup AST."
  [config]
  (let [cfg (merge (gemma3-config) config)
        {:keys [num-layers max-seq-len hidden-dim norm-eps final-logit-softcap]} cfg
        num-layers (long (or num-layers 18))
        max-seq-len (long (or max-seq-len 128))
        hidden-dim (long (or hidden-dim 640))
        norm-eps (double (or norm-eps 1e-6))
        h-final (keyword (str "h" num-layers))]
    [:block {:name :full_gemma3_model}
     ;; 1. Token Embedding Lookup (scaled by sqrt(hidden-dim))
     [:gather [:tok_embed_raw :b :p :d] [:embed_tokens :v :d] [:x :b :p]]
     [:= [:h0 :b :p :d] {:scale (Math/sqrt (double hidden-dim))} [:tok_embed_raw :b :p :d]]

     ;; 2. Sequential Transformer Layer Blocks
     (mapv (fn [i] (gemma3-layer-ast i max-seq-len cfg))
           (range num-layers))

     ;; 3. Final RMSNorm
     [:rms-norm [:normed :b :p :d] [h-final :b :p :d] [:final_norm_w :d] {:eps norm-eps :gemma? true}]

     ;; 4. Tied LM Head with optional final logit softcapping
     [:= [:logits :b :p :v] (if (and (number? final-logit-softcap) (pos? final-logit-softcap))
                              {:softcap (double final-logit-softcap)}
                              {})
      [:normed :b :p :d] [:embed_tokens :v :d]]]))
