(ns einsum.runtime.benchmark.workloads
  "Standard benchmark workload specifications (GEMM, RMSNorm, SwiGLU, GQA Attention, GPT-2 block, Gemma 4 block)."
  (:require [einsum.runtime.benchmark.core :as bcore]
            [einsum.logic.lower :as lower]))

(defn build-gemm-graph
  [m n k dtype]
  (let [invars [[:a [:tensor [m k] dtype]]
                [:b [:tensor [k n] dtype]]]
        ast [[:= [:c :m :n] [:a :m :k] [:b :k :n]]]]
    (lower/ast->graph (str "gemm_" (name dtype) "_" m "_" n "_" k)
                      invars ast [:c])))

(defn make-gemm-inputs
  [m n k dtype]
  (let [num-a (* m k)
        num-b (* k n)]
    (if (= dtype :bf16)
      {:a (short-array (repeat num-a (short 16256)))
       :b (short-array (repeat num-b (short 16256)))}
      {:a (float-array (repeat num-a 1.0))
       :b (float-array (repeat num-b 1.0))})))

(defn build-rms-norm-graph
  [batch seq-len dim dtype]
  (let [invars [[:x [:tensor [batch seq-len dim] dtype]]
                [:w [:tensor [dim] dtype]]]
        ast [[:rms-norm [:out :b :s :d] [:x :b :s :d] [:w :d] {:eps 1e-6}]]]
    (lower/ast->graph (str "rms_norm_" batch "_" seq-len "_" dim)
                      invars ast [:out])))

(defn make-rms-norm-inputs
  [batch seq-len dim _dtype]
  {:x (float-array (repeat (* batch seq-len dim) 1.0))
   :w (float-array (repeat dim 1.0))})

(defn build-swiglu-graph
  [batch seq-len dim dtype]
  (let [inter (* 4 dim)
        invars [[:x [:tensor [batch seq-len dim] dtype]]
                [:gate_w [:tensor [dim inter] dtype]]
                [:up_w [:tensor [dim inter] dtype]]]
        ast [[:= [:gate :b :s :i] {:act :swish} [:x :b :s :d] [:gate_w :d :i]]
             [:= [:up :b :s :i] [:x :b :s :d] [:up_w :d :i]]
             [:= [:out :b :s :i] [:gate :b :s :i] [:up :b :s :i]]]]
    (lower/ast->graph (str "swiglu_" batch "_" seq-len "_" dim)
                      invars ast [:out])))

(defn make-swiglu-inputs
  [batch seq-len dim _dtype]
  (let [inter (* 4 dim)]
    {:x (float-array (repeat (* batch seq-len dim) 1.0))
     :gate_w (float-array (repeat (* dim inter) 0.01))
     :up_w (float-array (repeat (* dim inter) 0.01))}))

(defn build-gqa-attn-graph
  [batch seq-len num-heads num-kv-heads head-dim dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        hidden-dim q-dim
        group-size (quot num-heads num-kv-heads)
        scale (/ 1.0 (Math/sqrt (double head-dim)))
        invars [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                [:q_w [:tensor [q-dim hidden-dim] dtype]]
                [:k_w [:tensor [kv-dim hidden-dim] dtype]]
                [:v_w [:tensor [kv-dim hidden-dim] dtype]]
                [:o_w [:tensor [hidden-dim q-dim] dtype]]
                [:pos [:tensor [seq-len] :i32]]]
        ast [;; 1. Projections
             [:= [:q_raw :b :p :qd] [:x :b :p :d] [:q_w :qd :d]]
             [:= [:k_raw :b :p :kvd] [:x :b :p :d] [:k_w :kvd :d]]
             [:= [:v_raw :b :p :kvd] [:x :b :p :d] [:v_w :kvd :d]]
             ;; 2. RoPE
             [:rope [:q_rope :b :p :qd] [:q_raw :b :p :qd] {:head-dim head-dim :theta 10000.0}]
             [:rope [:k_rope :b :p :kvd] [:k_raw :b :p :kvd] {:head-dim head-dim :theta 10000.0}]
             ;; 3. Reshape to heads
             [:reshape [:q_heads :b :h :p :hd] [:q_rope :b :p :qd] {:shape [batch num-heads seq-len head-dim]}]
             [:reshape [:k_heads :b :kvh :p :hd] [:k_rope :b :p :kvd] {:shape [batch num-kv-heads seq-len head-dim]}]
             [:reshape [:v_heads :b :kvh :p :hd] [:v_raw :b :p :kvd] {:shape [batch num-kv-heads seq-len head-dim]}]
             ;; 4. GQA Expand
             (if (> group-size 1)
               [:block {:name :gqa_expand}
                [:= [:k_rep :b :kvh :g :p :hd] [:k_heads :b :kvh :p :hd] {:shape [batch num-kv-heads group-size seq-len head-dim]}]
                [:reshape [:k_full :b :h :p :hd] [:k_rep :b :kvh :g :p :hd] {:shape [batch num-heads seq-len head-dim]}]
                [:= [:v_rep :b :kvh :g :p :hd] [:v_heads :b :kvh :p :hd] {:shape [batch num-kv-heads group-size seq-len head-dim]}]
                [:reshape [:v_full :b :h :p :hd] [:v_rep :b :kvh :g :p :hd] {:shape [batch num-heads seq-len head-dim]}]]
               [:block {:name :mha_view}
                [:= [:k_full :b :h :p :hd] [:k_heads :b :h :p :hd]]
                [:= [:v_full :b :h :p :hd] [:v_heads :b :h :p :hd]]])
             ;; 5. Attention
             [:= [:scores_raw :b :h :p :k] {:scale scale} [:q_heads :b :h :p :hd] [:k_full :b :h :k :hd]]
             [:causal-softmax [:attn_weights :b :h :p :k] [:scores_raw :b :h :p :k]]
             [:= [:attn_ctx :b :h :p :hd] [:attn_weights :b :h :p :k] [:v_full :b :h :k :hd]]
             ;; 6. Out proj
             [:reshape [:attn_proj_in :b :p :qd] [:attn_ctx :b :h :p :hd] {:shape [batch seq-len q-dim]}]
             [:= [:out :b :p :d] [:attn_proj_in :b :p :qd] [:o_w :d :qd]]]]
    (lower/ast->graph (str "gqa_attn_" batch "_" seq-len "_" num-heads "_" num-kv-heads)
                      invars ast [:out])))

(defn make-gqa-attn-inputs
  [batch seq-len num-heads num-kv-heads head-dim _dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        hidden-dim q-dim]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :q_w (float-array (repeat (* q-dim hidden-dim) 0.01))
     :k_w (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :v_w (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :o_w (float-array (repeat (* hidden-dim q-dim) 0.01))
     :pos (int-array (range seq-len))}))

(defn build-gpt2-block-graph
  [batch seq-len hidden-dim dtype]
  (let [inter (* 4 hidden-dim)
        attn-dim (* 3 hidden-dim)
        num-heads 12
        head-dim (quot hidden-dim num-heads)
        invars [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                [:ln1_g [:tensor [hidden-dim] dtype]]
                [:ln1_b [:tensor [hidden-dim] dtype]]
                [:c_attn_w [:tensor [hidden-dim attn-dim] dtype]]
                [:c_attn_b [:tensor [attn-dim] dtype]]
                [:c_proj_w [:tensor [hidden-dim hidden-dim] dtype]]
                [:c_proj_b [:tensor [hidden-dim] dtype]]
                [:ln2_g [:tensor [hidden-dim] dtype]]
                [:ln2_b [:tensor [hidden-dim] dtype]]
                [:mlp_fc_w [:tensor [hidden-dim inter] dtype]]
                [:mlp_fc_b [:tensor [inter] dtype]]
                [:mlp_proj_w [:tensor [inter hidden-dim] dtype]]
                [:mlp_proj_b [:tensor [hidden-dim] dtype]]]
        ast [;; 1. Pre-LayerNorm 1
             [:layer-norm [:x_norm1 :b :p :d] [:x :b :p :d] [:ln1_g :d] [:ln1_b :d]]
             ;; 2. QKV Projection with bias
             [:= [:qkv :b :p :qkv_dim] [:x_norm1 :b :p :d] [:c_attn_w :d :qkv_dim]]
             [:= [:qkv :b :p :qkv_dim] [:c_attn_b :qkv_dim]]
             ;; 3. Slice Q, K, V
             [:slice [:q :b :p :d] [:qkv :b :p :qkv_dim] {:start [0 0 0] :limit [batch seq-len hidden-dim]}]
             [:slice [:k :b :p :d] [:qkv :b :p :qkv_dim] {:start [0 0 hidden-dim] :limit [batch seq-len (* 2 hidden-dim)]}]
             [:slice [:v :b :p :d] [:qkv :b :p :qkv_dim] {:start [0 0 (* 2 hidden-dim)] :limit [batch seq-len attn-dim]}]
             ;; 4. Multi-head reshape
             [:reshape [:q_heads :b :h :p :hd] [:q :b :p :d] {:shape [batch num-heads seq-len head-dim]}]
             [:reshape [:k_heads :b :h :p :hd] [:k :b :p :d] {:shape [batch num-heads seq-len head-dim]}]
             [:reshape [:v_heads :b :h :p :hd] [:v :b :p :d] {:shape [batch num-heads seq-len head-dim]}]
             ;; 5. Scaled Dot-Product Causal Attention
             [:= [:scores :b :h :p :k_idx] {:scale (/ 1.0 (Math/sqrt (double head-dim)))} [:q_heads :b :h :p :hd] [:k_heads :b :h :k_idx :hd]]
             [:causal-softmax [:probs :b :h :p :k_idx] [:scores :b :h :p :k_idx]]
             [:= [:ctx :b :h :p :hd] [:probs :b :h :p :k_idx] [:v_heads :b :h :k_idx :hd]]
             [:reshape [:ctx_flat :b :p :d] [:ctx :b :h :p :hd] {:shape [batch seq-len hidden-dim]}]
             ;; 6. Attn out projection with bias
             [:= [:attn_out :b :p :d] [:ctx_flat :b :p :din] [:c_proj_w :din :d]]
             [:= [:attn_out :b :p :d] [:c_proj_b :d]]
             ;; 7. Residual 1
             [:= [:res1 :b :p :d] [:x :b :p :d]]
             [:= [:res1 :b :p :d] [:attn_out :b :p :d]]
             ;; 8. Pre-LayerNorm 2
             [:layer-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:ln2_g :d] [:ln2_b :d]]
             ;; 9. MLP with GeLU and bias
             [:= [:mlp_fc :b :p :inter] [:x_norm2 :b :p :d] [:mlp_fc_w :d :inter]]
             [:= [:mlp_fc :b :p :inter] [:mlp_fc_b :inter]]
             [:= [:mlp_act :b :p :inter] {:act :gelu} [:mlp_fc :b :p :inter]]
             [:= [:mlp_out :b :p :d] [:mlp_act :b :p :inter] [:mlp_proj_w :inter :d]]
             [:= [:mlp_out :b :p :d] [:mlp_proj_b :d]]
             ;; 10. Residual 2
             [:= [:out :b :p :d] [:res1 :b :p :d]]
             [:= [:out :b :p :d] [:mlp_out :b :p :d]]]]
    (lower/ast->graph (str "gpt2_block_" batch "_" seq-len "_" hidden-dim)
                      invars ast [:out])))

(defn make-gpt2-block-inputs
  [batch seq-len hidden-dim _dtype]
  (let [inter (* 4 hidden-dim)
        attn-dim (* 3 hidden-dim)]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :ln1_g (float-array (repeat hidden-dim 1.0))
     :ln1_b (float-array (repeat hidden-dim 0.0))
     :c_attn_w (float-array (repeat (* hidden-dim attn-dim) 0.01))
     :c_attn_b (float-array (repeat attn-dim 0.0))
     :c_proj_w (float-array (repeat (* hidden-dim hidden-dim) 0.01))
     :c_proj_b (float-array (repeat hidden-dim 0.0))
     :ln2_g (float-array (repeat hidden-dim 1.0))
     :ln2_b (float-array (repeat hidden-dim 0.0))
     :mlp_fc_w (float-array (repeat (* hidden-dim inter) 0.01))
     :mlp_fc_b (float-array (repeat inter 0.0))
     :mlp_proj_w (float-array (repeat (* inter hidden-dim) 0.01))
     :mlp_proj_b (float-array (repeat hidden-dim 0.0))}))

(defn build-gemma4-block-graph
  [batch seq-len hidden-dim num-heads num-kv-heads head-dim pl-dim dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        mlp-dim (* 4 hidden-dim)
        group-size (quot num-heads num-kv-heads)
        scale (/ 1.0 (Math/sqrt (double head-dim)))
        invars [[:x [:tensor [batch seq-len hidden-dim] dtype]]
                [:in_ln [:tensor [hidden-dim] dtype]]
                [:layer_scalar [:tensor [1] dtype]]
                [:qw [:tensor [q-dim hidden-dim] dtype]]
                [:kw [:tensor [kv-dim hidden-dim] dtype]]
                [:vw [:tensor [kv-dim hidden-dim] dtype]]
                [:ow [:tensor [hidden-dim q-dim] dtype]]
                [:qn [:tensor [head-dim] dtype]]
                [:kn [:tensor [head-dim] dtype]]
                [:post_attn_ln [:tensor [hidden-dim] dtype]]
                [:pre_mlp_ln [:tensor [hidden-dim] dtype]]
                [:post_mlp_ln [:tensor [hidden-dim] dtype]]
                [:gate_w [:tensor [mlp-dim hidden-dim] dtype]]
                [:up_w [:tensor [mlp-dim hidden-dim] dtype]]
                [:down_w [:tensor [hidden-dim mlp-dim] dtype]]
                [:per_layer_gate [:tensor [pl-dim hidden-dim] dtype]]
                [:per_layer_proj [:tensor [hidden-dim pl-dim] dtype]]
                [:post_per_layer_norm [:tensor [hidden-dim] dtype]]
                [:per_layer_in [:tensor [batch seq-len pl-dim] dtype]]
                [:pos [:tensor [seq-len] :i32]]]
        ast [;; 1. Input RMSNorm
             [:rms-norm [:x_norm1 :b :p :d] [:x :b :p :d] [:in_ln :d] {:eps 1e-6 :gemma? true}]
             ;; 2. Projections
             [:= [:q_raw :b :p :qd] [:x_norm1 :b :p :d] [:qw :qd :d]]
             [:= [:k_raw :b :p :kvd] [:x_norm1 :b :p :d] [:kw :kvd :d]]
             [:= [:v_raw :b :p :kvd] [:x_norm1 :b :p :d] [:vw :kvd :d]]
             ;; 3. Reshape and QK Norm
             [:reshape [:q_heads_raw :b :h :p :hd] [:q_raw :b :p :qd] {:shape [batch num-heads seq-len head-dim]}]
             [:reshape [:k_heads_raw :b :kvh :p :hd] [:k_raw :b :p :kvd] {:shape [batch num-kv-heads seq-len head-dim]}]
             [:reshape [:v_heads :b :kvh :p :hd] [:v_raw :b :p :kvd] {:shape [batch num-kv-heads seq-len head-dim]}]
             [:rms-norm [:q_normed_4d :b :h :p :hd] [:q_heads_raw :b :h :p :hd] [:qn :hd] {:eps 1e-6}]
             [:rms-norm [:k_normed_4d :b :kvh :p :hd] [:k_heads_raw :b :kvh :p :hd] [:kn :hd] {:eps 1e-6}]
             [:reshape [:q_normed_3d :b :p :qd] [:q_normed_4d :b :h :p :hd] {:shape [batch seq-len q-dim]}]
             [:reshape [:k_normed_3d :b :p :kvd] [:k_normed_4d :b :kvh :p :hd] {:shape [batch seq-len kv-dim]}]
             ;; 4. RoPE
             [:rope [:q_rope :b :p :qd] [:q_normed_3d :b :p :qd] {:head-dim head-dim :theta 10000.0}]
             [:rope [:k_rope :b :p :kvd] [:k_normed_3d :b :p :kvd] {:head-dim head-dim :theta 10000.0}]
             ;; 5. Reshape and GQA expand
             [:reshape [:q_heads :b :h :p :hd] [:q_rope :b :p :qd] {:shape [batch num-heads seq-len head-dim]}]
             [:reshape [:k_heads :b :kvh :p :hd] [:k_rope :b :p :kvd] {:shape [batch num-kv-heads seq-len head-dim]}]
             (if (> group-size 1)
               [:block {:name :gqa_expand}
                [:= [:k_rep :b :kvh :g :p :hd] [:k_heads :b :kvh :p :hd] {:shape [batch num-kv-heads group-size seq-len head-dim]}]
                [:reshape [:k_full :b :h :p :hd] [:k_rep :b :kvh :g :p :hd] {:shape [batch num-heads seq-len head-dim]}]
                [:= [:v_rep :b :kvh :g :p :hd] [:v_heads :b :kvh :p :hd] {:shape [batch num-kv-heads group-size seq-len head-dim]}]
                [:reshape [:v_full :b :h :p :hd] [:v_rep :b :kvh :g :p :hd] {:shape [batch num-heads seq-len head-dim]}]]
               [:block {:name :mha_view}
                [:= [:k_full :b :h :p :hd] [:k_heads :b :h :p :hd]]
                [:= [:v_full :b :h :p :hd] [:v_heads :b :h :p :hd]]])
             ;; 6. SDPA Causal Softmax
             [:= [:scores_raw :b :h :p :k] {:scale scale} [:q_heads :b :h :p :hd] [:k_full :b :h :k :hd]]
             [:causal-softmax [:attn_weights :b :h :p :k] [:scores_raw :b :h :p :k]]
             [:= [:attn_ctx :b :h :p :hd] [:attn_weights :b :h :p :k] [:v_full :b :h :k :hd]]
             ;; 7. Attn out proj
             [:reshape [:attn_proj_in :b :p :qd] [:attn_ctx :b :h :p :hd] {:shape [batch seq-len q-dim]}]
             [:= [:attn_out :b :p :d] [:attn_proj_in :b :p :qd] [:ow :d :qd]]
             ;; 8. Post-attention RMSNorm
             [:rms-norm [:attn_normed :b :p :d] [:attn_out :b :p :d] [:post_attn_ln :d] {:eps 1e-6 :gemma? true}]
             ;; 9. First residual
             [:= [:res1 :b :p :d] [:x :b :p :d]]
             [:= [:res1 :b :p :d] [:attn_normed :b :p :d]]
             ;; 10. Pre-feedforward RMSNorm
             [:rms-norm [:x_norm2 :b :p :d] [:res1 :b :p :d] [:pre_mlp_ln :d] {:eps 1e-6 :gemma? true}]
             ;; 11. GeGLU MLP
             [:= [:gate_out :b :p :inter] {:act :gelu} [:x_norm2 :b :p :d] [:gate_w :inter :d]]
             [:= [:up_out :b :p :inter] [:x_norm2 :b :p :d] [:up_w :inter :d]]
             [:= [:hidden :b :p :inter] [:gate_out :b :p :inter] [:up_out :b :p :inter]]
             [:= [:mlp_out :b :p :d] [:hidden :b :p :inter] [:down_w :d :inter]]
             ;; 12. Post-feedforward RMSNorm
             [:rms-norm [:mlp_normed :b :p :d] [:mlp_out :b :p :d] [:post_mlp_ln :d] {:eps 1e-6 :gemma? true}]
             ;; 13. Second residual
             [:= [:res2 :b :p :d] [:res1 :b :p :d]]
             [:= [:res2 :b :p :d] [:mlp_normed :b :p :d]]
             ;; 14. Gemma 4 PLE Gating
             [:= [:pl_gate :b :p :pld] {:act :sigmoid} [:res2 :b :p :d] [:per_layer_gate :pld :d]]
             [:= [:pl_gated :b :p :pld] [:pl_gate :b :p :pld] [:per_layer_in :b :p :pld]]
             [:= [:pl_proj_raw :b :p :d] [:pl_gated :b :p :pld] [:per_layer_proj :d :pld]]
             [:rms-norm [:pl_normed :b :p :d] [:pl_proj_raw :b :p :d] [:post_per_layer_norm :d] {:eps 1e-6 :gemma? true}]
             [:= [:res3 :b :p :d] [:res2 :b :p :d]]
             [:= [:res3 :b :p :d] [:pl_normed :b :p :d]]
             ;; 15. Layer Scalar
             [:= [:out :b :p :d] [:res3 :b :p :d] [:layer_scalar :one]]]]
    (lower/ast->graph (str "gemma4_block_" batch "_" seq-len "_" hidden-dim)
                      invars ast [:out])))

(defn make-gemma4-block-inputs
  [batch seq-len hidden-dim num-heads num-kv-heads head-dim pl-dim _dtype]
  (let [q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        mlp-dim (* 4 hidden-dim)]
    {:x (float-array (repeat (* batch seq-len hidden-dim) 1.0))
     :in_ln (float-array (repeat hidden-dim 1.0))
     :layer_scalar (float-array [1.0])
     :qw (float-array (repeat (* q-dim hidden-dim) 0.01))
     :kw (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :vw (float-array (repeat (* kv-dim hidden-dim) 0.01))
     :ow (float-array (repeat (* hidden-dim q-dim) 0.01))
     :qn (float-array (repeat head-dim 1.0))
     :kn (float-array (repeat head-dim 1.0))
     :post_attn (float-array (repeat hidden-dim 1.0))
     :pre_mlp (float-array (repeat hidden-dim 1.0))
     :post_mlp (float-array (repeat hidden-dim 1.0))
     :gate_w (float-array (repeat (* mlp-dim hidden-dim) 0.01))
     :up_w (float-array (repeat (* mlp-dim hidden-dim) 0.01))
     :down_w (float-array (repeat (* hidden-dim mlp-dim) 0.01))
     :per_layer_gate (float-array (repeat (* pl-dim hidden-dim) 0.01))
     :per_layer_proj (float-array (repeat (* hidden-dim pl-dim) 0.01))
     :post_per_layer_norm (float-array (repeat hidden-dim 1.0))
     :per_layer_in (float-array (repeat (* batch seq-len pl-dim) 1.0))
     :pos (int-array (range seq-len))}))

(def WORKLOADS
  {:gemm-fp32
   {:id :gemm-fp32
    :name "GEMM FP32 (1024 x 1024 x 1024)"
    :category :gemm
    :flops (bcore/gemm-flops 1024 1024 1024)
    :bytes (* 4 3 1024 1024)
    :build-graph-fn (fn [] (build-gemm-graph 1024 1024 1024 :f32))
    :make-inputs-fn (fn [] (make-gemm-inputs 1024 1024 1024 :f32))}

   :gemm-bf16
   {:id :gemm-bf16
    :name "GEMM BF16 (1024 x 1024 x 1024)"
    :category :gemm
    :flops (bcore/gemm-flops 1024 1024 1024)
    :bytes (* 2 3 1024 1024)
    :build-graph-fn (fn [] (build-gemm-graph 1024 1024 1024 :bf16))
    :make-inputs-fn (fn [] (make-gemm-inputs 1024 1024 1024 :bf16))}

   :rms-norm
   {:id :rms-norm
    :name "RMSNorm (1 x 2048 x 4096)"
    :category :norm-act
    :flops (* 1 2048 4096 3)
    :bytes (* 4 2 2048 4096)
    :build-graph-fn (fn [] (build-rms-norm-graph 1 2048 4096 :f32))
    :make-inputs-fn (fn [] (make-rms-norm-inputs 1 2048 4096 :f32))}

   :swiglu
   {:id :swiglu
    :name "SwiGLU Activation (1 x 2048 x 4096)"
    :category :norm-act
    :flops (bcore/gemm-flops 2048 (* 4 4096) 4096)
    :bytes (* 4 4 2048 4096)
    :build-graph-fn (fn [] (build-swiglu-graph 1 2048 4096 :f32))
    :make-inputs-fn (fn [] (make-swiglu-inputs 1 2048 4096 :f32))}

   :gqa-causal-attn
   {:id :gqa-causal-attn
    :name "GQA Causal Attention (1 x 128 x 8 x 256)"
    :category :attn
    :flops (bcore/gemm-flops 128 2048 2048)
    :bytes (* 4 3 128 2048)
    :build-graph-fn (fn [] (build-gqa-attn-graph 1 128 8 1 256 :f32))
    :make-inputs-fn (fn [] (make-gqa-attn-inputs 1 128 8 1 256 :f32))}

   :gpt2-block
   {:id :gpt2-block
    :name "GPT-2 Transformer Layer Block (1 x 128 x 768)"
    :category :layer
    :flops (* 12 (bcore/gemm-flops 128 768 768))
    :bytes (* 4 10 128 768)
    :build-graph-fn (fn [] (build-gpt2-block-graph 1 128 768 :f32))
    :make-inputs-fn (fn [] (make-gpt2-block-inputs 1 128 768 :f32))}

   :gemma4-block
   {:id :gemma4-block
    :name "Gemma 4 Transformer Layer Block (1 x 128 x 1536)"
    :category :layer
    :flops (* 16 (bcore/gemm-flops 128 1536 1536))
    :bytes (* 4 12 128 1536)
    :build-graph-fn (fn [] (build-gemma4-block-graph 1 128 1536 8 1 256 256 :f32))
    :make-inputs-fn (fn [] (make-gemma4-block-inputs 1 128 1536 8 1 256 256 :f32))}})
