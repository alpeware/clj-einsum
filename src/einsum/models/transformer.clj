(ns einsum.models.transformer
  "Canonical Transformer Reference Architecture (Vaswani et al., 2017).
   Implements multi-head self-attention, feed-forward network, residual connections,
   and layer normalization in pure Declarative Tensor Logic (Hiccup AST)."
  (:require [einsum.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn multi-head-attention-ast
  "Constructs pure Tensor Logic AST for Multi-Head Self-Attention:
   1. Q = X @ W_q, K = X @ W_k, V = X @ W_v
   2. scores = (Q @ K^T) / sqrt(d_k)
   3. weights = softmax(scores)  [or causal-softmax]
   4. context = weights @ V
   5. out = context @ W_o
   6. res = X + out"
  ([head-dim]
   (multi-head-attention-ast head-dim {:causal? true}))
  ([head-dim opts]
   (let [scale (/ 1.0 (Math/sqrt (double head-dim)))
         causal? (:causal? opts true)]
     [:block {:name :multi_head_attention}
      ;; Linear projections
      [:= [:Q :b :p_q :h :dh] [:X :b :p_q :d] [:W_q :d :h :dh]]
      [:= [:K :b :p_k :h :dh] [:X :b :p_k :d] [:W_k :d :h :dh]]
      [:= [:V :b :p_k :h :dh] [:X :b :p_k :d] [:W_v :d :h :dh]]

      ;; Scaled dot-product attention
      [:= [:scores :b :h :p_q :p_k] {:scale scale} [:Q :b :p_q :h :dh] [:K :b :p_k :h :dh]]
      (if causal?
        [:causal-softmax [:attn_weights :b :h :p_q :p_k] [:scores :b :h :p_q :p_k]]
        [:softmax [:attn_weights :b :h :p_q :p_k] [:scores :b :h :p_q :p_k]])
      [:= [:context :b :p_q :h :dh] [:attn_weights :b :h :p_q :p_k] [:V :b :p_k :h :dh]]
      [:= [:attn_out :b :p_q :d] [:context :b :p_q :h :dh] [:W_o :h :dh :d]]
      [:+ [:H_attn :b :p_q :d] [:X :b :p_q :d] [:attn_out :b :p_q :d]]])))

(defn ffn-block-ast
  "Constructs pure Tensor Logic AST for Feed-Forward Network:
   FFN(X) = act(X @ W_1) @ W_2
   res = X + FFN(X)"
  ([]
   (ffn-block-ast {:act :relu}))
  ([opts]
   (let [act (or (:act opts) :relu)]
     [:block {:name :ffn_block}
      [:= [:h1 :b :p :d_ff] {:act act} [:X_in :b :p :d] [:W_1 :d :d_ff]]
      [:= [:ffn_out :b :p :d] [:h1 :b :p :d_ff] [:W_2 :d_ff :d]]
      [:+ [:H_out :b :p :d] [:X_in :b :p :d] [:ffn_out :b :p :d]]])))

(defn transformer-block-ast
  "Constructs full canonical Transformer Layer Block (Vaswani et al., 2017):
   1. Multi-head self-attention with residual
   2. Layer norm / RMS norm
   3. Feed-forward network with residual
   4. Final norm"
  ([head-dim]
   (transformer-block-ast head-dim {:causal? true :act :relu}))
  ([head-dim opts]
   (let [scale (/ 1.0 (Math/sqrt (double head-dim)))
         causal? (:causal? opts true)
         act (or (:act opts) :relu)]
     [:block {:name :transformer_layer}
      ;; Attention sub-layer
      [:= [:Q :b :p_q :h :dh] [:X :b :p_q :d] [:W_q :d :h :dh]]
      [:= [:K :b :p_k :h :dh] [:X :b :p_k :d] [:W_k :d :h :dh]]
      [:= [:V :b :p_k :h :dh] [:X :b :p_k :d] [:W_v :d :h :dh]]
      [:= [:scores :b :h :p_q :p_k] {:scale scale} [:Q :b :p_q :h :dh] [:K :b :p_k :h :dh]]
      (if causal?
        [:causal-softmax [:attn_weights :b :h :p_q :p_k] [:scores :b :h :p_q :p_k]]
        [:softmax [:attn_weights :b :h :p_q :p_k] [:scores :b :h :p_q :p_k]])
      [:= [:context :b :p_q :h :dh] [:attn_weights :b :h :p_q :p_k] [:V :b :p_k :h :dh]]
      [:= [:attn_out :b :p_q :d] [:context :b :p_q :h :dh] [:W_o :h :dh :d]]
      [:+ [:H_attn :b :p_q :d] [:X :b :p_q :d] [:attn_out :b :p_q :d]]
      [:rms-norm [:H_attn_norm :b :p_q :d] [:H_attn :b :p_q :d]]

      ;; Feed-Forward sub-layer
      [:= [:h1 :b :p_q :d_ff] {:act act} [:H_attn_norm :b :p_q :d] [:W_1 :d :d_ff]]
      [:= [:ffn_out :b :p_q :d] [:h1 :b :p_q :d_ff] [:W_2 :d_ff :d]]
      [:+ [:H_final :b :p_q :d] [:H_attn_norm :b :p_q :d] [:ffn_out :b :p_q :d]]
      [:rms-norm [:out :b :p_q :d] [:H_final :b :p_q :d]]])))

;; ==============================================================================
;; 2. Compilation Interface
;; ==============================================================================

(defn compile-transformer-block
  "Compiles canonical transformer block into an OpenXLA PJRT executable."
  [ctx batch num-heads seq-len head-dim dff & [opts]]
  (let [model-dim (* num-heads head-dim)
        ast (transformer-block-ast head-dim (merge {:causal? true :act :relu} opts))
        invars [[:X [:tensor [batch seq-len model-dim] :f32]]
                [:W_q [:tensor [model-dim num-heads head-dim] :f32]]
                [:W_k [:tensor [model-dim num-heads head-dim] :f32]]
                [:W_v [:tensor [model-dim num-heads head-dim] :f32]]
                [:W_o [:tensor [num-heads head-dim model-dim] :f32]]
                [:W_1 [:tensor [model-dim dff] :f32]]
                [:W_2 [:tensor [dff model-dim] :f32]]]]
    (sym/compile-query ctx "transformer_block" invars ast [:out])))

(defn run-transformer-block!
  "Executes the canonical transformer forward pass on PJRT device runtime."
  [exec inputs]
  (sym/run-query! exec inputs))
