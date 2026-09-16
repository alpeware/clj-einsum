(ns clj-xla.logic.models.tl-block
  "The Unified TL-Transformer Layer Block for Pedro Domingos' Declarative Tensor Logic (Experiment E6).
   Synthesizes Knowledge-Graph Masked Self-Attention (E2), Cross-Attention Memory Probing (E1),
   Contrastive Subspace Projection (E3), and Ephemeral Fast-Weight Contraction (E5) into an
   end-to-end forward block lowered into StableHLO MLIR for OpenXLA PJRT execution."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn tl-transformer-block-ast
  "Constructs Tensor Logic AST for the unified TL-Transformer layer block:
   1. KG-Masked Causal Self-Attention:
      TR = T * R_adj, M_raw = TR * T^T, M_kg = gamma * M_raw
      scores = (Q . K^T) / sqrt(dh) + M_kg
      attn_probs = causal-softmax(scores)
      out = attn_probs @ V, attn_out = out @ W_o
      H_attn = H + attn_out
   2. Relational Memory Unbinding & Deductive Gating:
      u_q = H_attn @ W_mem, u_target = u_q @ R_mem, u_norm = rms_norm(u_target)
      U_cand = E_cand @ W_mem
      cand_scores = (u_norm @ U_cand^T) / tau
      valid_mask = cand_scores > threshold
      clamped_scores = cand_scores * valid_mask
      v_grounded = clamped_scores @ E_cand
      v_bias = lambda_mem * v_grounded
      H_tl = H_attn + v_bias
   3. GeGLU Feed-Forward Network:
      gate = gelu(H_tl @ W_gate)
      up = H_tl @ W_up
      mlp_act = gate * up
      mlp_out = mlp_act @ W_down
      H_out = H_tl + mlp_out"
  ([_batch _num-heads _seq-len head-dim _dff _dim-mem _entity-count _cand-count]
   (tl-transformer-block-ast _batch _num-heads _seq-len head-dim _dff _dim-mem _entity-count _cand-count nil))
  ([_batch _num-heads _seq-len head-dim _dff _dim-mem _entity-count _cand-count opts]
   (let [gamma (double (or (:gamma opts) 1.0))
         lambda-mem (double (or (:lambda-mem opts) 0.2))
         tau (double (or (:tau opts) 0.2))
         inv-tau (/ 1.0 tau)
         attn-scale (/ 1.0 (Math/sqrt (double head-dim)))]
     [:block {:name :tl_transformer_block}
      ;; 1. KG-Masked Self-Attention
      [:= [:TR :p_q :e2] [:T :p_q :e1] [:R_adj :e1 :e2]]
      [:= [:M_raw :p_q :p_k] [:TR :p_q :e2] [:T :p_k :e2]]
      [:= [:M_kg :p_q :p_k] {:scale gamma} [:M_raw :p_q :p_k]]

      [:= [:scores :b :h :p_q :p_k] {:scale attn-scale} [:Q :b :p_q :h :dh] [:K :b :p_k :h :dh]]
      [:= [:scores_biased :b :h :p_q :p_k] {:op :add} [:scores :b :h :p_q :p_k] [:M_kg :p_q :p_k]]
      [:causal-softmax [:attn_probs :b :h :p_q :p_k] [:scores_biased :b :h :p_q :p_k]]
      [:= [:out :b :p_q :h :dh] [:attn_probs :b :h :p_q :p_k] [:V :b :p_k :h :dh]]
      [:= [:attn_out :b :p_q :d] [:out :b :p_q :h :dh] [:W_o :h :dh :d]]
      [:+ [:H_attn :b :p_q :d] [:H :b :p_q :d] [:attn_out :b :p_q :d]]

      ;; 2. Relational Memory Unbinding & Deductive Gating
      [:= [:u_q :b :p_q :dm] [:H_attn :b :p_q :d] [:W_mem :d :dm]]
      [:= [:u_target :b :p_q :dm2] [:u_q :b :p_q :dm1] [:R_mem :dm1 :dm2]]
      [:rms-norm [:u_target_norm :b :p_q :dm2] [:u_target :b :p_q :dm2]]
      [:= [:U_cand :nc :dm] [:E_cand :nc :d] [:W_mem :d :dm]]
      [:= [:cand_scores :b :p_q :nc] {:scale inv-tau} [:u_target_norm :b :p_q :dm] [:U_cand :nc :dm]]

      [:compare [:valid_mask :b :p_q :nc] [:cand_scores :b :p_q :nc] [:threshold :one] {:comparison_direction "GT"}]
      [:convert [:valid_weight :b :p_q :nc] {:target-dtype :f32} [:valid_mask :b :p_q :nc]]
      [:= [:clamped_scores :b :p_q :nc] [:cand_scores :b :p_q :nc] [:valid_weight :b :p_q :nc]]
      [:= [:v_grounded :b :p_q :d] [:clamped_scores :b :p_q :nc] [:E_cand :nc :d]]
      [:= [:v_bias :b :p_q :d] {:scale lambda-mem} [:v_grounded :b :p_q :d]]
      [:+ [:H_tl :b :p_q :d] [:H_attn :b :p_q :d] [:v_bias :b :p_q :d]]

      ;; 3. GeGLU Feed-Forward Network
      [:= [:gate_out :b :p_q :dff] {:act :gelu} [:H_tl :b :p_q :d] [:W_gate :d :dff]]
      [:= [:up_out :b :p_q :dff] [:H_tl :b :p_q :d] [:W_up :d :dff]]
      [:= [:mlp_act :b :p_q :dff] [:gate_out :b :p_q :dff] [:up_out :b :p_q :dff]]
      [:= [:mlp_out :b :p_q :d] [:mlp_act :b :p_q :dff] [:W_down :dff :d]]
      [:+ [:H_out :b :p_q :d] [:H_tl :b :p_q :d] [:mlp_out :b :p_q :d]]])))

;; ==============================================================================
;; 2. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn compile-tl-transformer-block
  "Compiles OpenXLA PJRT executable for the unified TL-Transformer layer block."
  ([batch seq-len num-heads head-dim dff dim-mem entity-count cand-count]
   (compile-tl-transformer-block (xla/get-context) batch seq-len num-heads head-dim dff dim-mem entity-count cand-count nil))
  ([a b c d e f g h i]
   (if (map? a)
     (compile-tl-transformer-block a b c d e f g h i nil)
     (compile-tl-transformer-block (xla/get-context) a b c d e f g h i)))
  ([ctx batch seq-len num-heads head-dim dff dim-mem entity-count cand-count opts]
   (let [b (long batch)
         l (long seq-len)
         h (long num-heads)
         dh (long head-dim)
         d (* h dh)
         ff (long dff)
         dm (long dim-mem)
         ne (long entity-count)
         nc (long cand-count)
         invars [[:H [:tensor [b l d] :f32]]
                 [:Q [:tensor [b l h dh] :f32]]
                 [:K [:tensor [b l h dh] :f32]]
                 [:V [:tensor [b l h dh] :f32]]
                 [:W_o [:tensor [h dh d] :f32]]
                 [:T [:tensor [l ne] :f32]]
                 [:R_adj [:tensor [ne ne] :f32]]
                 [:W_mem [:tensor [d dm] :f32]]
                 [:R_mem [:tensor [dm dm] :f32]]
                 [:E_cand [:tensor [nc d] :f32]]
                 [:threshold [:tensor [1] :f32]]
                 [:W_gate [:tensor [d ff] :f32]]
                 [:W_up [:tensor [d ff] :f32]]
                 [:W_down [:tensor [ff d] :f32]]]
         ast (tl-transformer-block-ast b h l dh ff dm ne nc opts)]
     (sym/compile-query ctx "tl_transformer_block" invars ast
                        [:H_out :cand_scores :clamped_scores :attn_probs :M_kg]))))

;; ==============================================================================
;; 3. Impure Execution Wrappers (PJRT Dispatch)
;; ==============================================================================

(defn run-tl-transformer-block!
  "Executes the unified TL-Transformer forward pass on PJRT device runtime."
  [exec inputs]
  (sym/run-query! exec inputs))
