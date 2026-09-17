(ns clj-xla.logic.memory.contrastive
  "Contrastive Subspace Pre-training on Knowledge Graphs for Pedro Domingos' Declarative Tensor Logic (Experiment E3).
   Aligns LLM embedding spaces with relational memory cores via InfoNCE contrastive loss on knowledge triples.
   Lowered into StableHLO MLIR and executed on OpenXLA PJRT (Rule 4: Zero Java bypasses).
   Adjoint gradient equations derived algebraically via clj-xla.logic.autodiff."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn contrastive-forward-ast
  "Constructs Tensor Logic AST for forward contrastive projection and in-batch scoring:
   U_h = V_h * W          [B, D_in] x [D_in, D_mem] -> [B, D_mem]
   U_t = V_t * W          [B, D_in] x [D_in, D_mem] -> [B, D_mem]
   U_hr = U_h * R         [B, D_mem] x [D_mem, D_mem] -> [B, D_mem]
   Scores = (U_hr * U_t^T) / tau  [B, D_mem] x [D_mem, B] -> [B, B]"
  ([_batch-size _dim-in _dim-mem]
   (contrastive-forward-ast _batch-size _dim-in _dim-mem nil))
  ([_batch-size _dim-in _dim-mem opts]
   (let [tau (double (or (:tau opts) 0.1))]
     [:block {:name :contrastive_forward}
      [:= [:U_h :b :dm] [:V_h :b :din] [:W :din :dm]]
      [:= [:U_t :b :dm] [:V_t :b :din] [:W :din :dm]]
      [:= [:U_hr :b :dm2] [:U_h :b :dm1] [:R :dm1 :dm2]]
      [:= [:Scores :b1 :b2] {:scale (/ 1.0 tau)} [:U_hr :b1 :dm] [:U_t :b2 :dm]]])))

(defn contrastive-backward-ast
  "Constructs Tensor Logic AST for reverse-mode adjoint backward pass (derived via autodiff):
   adj_Ut  = G_S^T * U_hr
   adj_Uhr = G_S * U_t
   dR      = U_h^T * adj_Uhr
   adj_Uh  = adj_Uhr * R^T
   dW_h    = V_h^T * adj_Uh
   dW_t    = V_t^T * adj_Ut
   dW      = dW_h + dW_t"
  [_batch-size _dim-in _dim-mem]
  [:block {:name :contrastive_backward}
   [:= [:adj_Ut :b2 :dm] [:G_S :b1 :b2] [:U_hr :b1 :dm]]
   [:= [:adj_Uhr :b1 :dm] [:G_S :b1 :b2] [:U_t :b2 :dm]]
   [:= [:dR :dm1 :dm2] [:U_h :b :dm1] [:adj_Uhr :b :dm2]]
   [:= [:adj_Uh :b :dm1] [:adj_Uhr :b :dm2] [:R :dm1 :dm2]]
   [:= [:dW_h :din :dm] [:V_h :b :din] [:adj_Uh :b :dm]]
   [:= [:dW_t :din :dm] [:V_t :b :din] [:adj_Ut :b :dm]]
   [:+ [:dW :din :dm] [:dW_h :din :dm] [:dW_t :din :dm]]])

(defn contrastive-update-ast
  "Constructs Tensor Logic AST for SGD parameter update:
   W_new = W - lr * dW
   R_new = R - lr * dR"
  ([_dim-in _dim-mem]
   (contrastive-update-ast _dim-in _dim-mem nil))
  ([_dim-in _dim-mem opts]
   (let [lr (double (or (:lr opts) 0.01))]
     [:block {:name :contrastive_update}
      [:= [:scaled_dW :din :dm] {:scale lr} [:dW :din :dm]]
      [:- [:W_new :din :dm] [:W :din :dm] [:scaled_dW :din :dm]]
      [:= [:scaled_dR :dm1 :dm2] {:scale lr} [:dR :dm1 :dm2]]
      [:- [:R_new :dm1 :dm2] [:R :dm1 :dm2] [:scaled_dR :dm1 :dm2]]])))

(defn in-vram-contrastive-step-ast
  "Constructs Tensor Logic AST for an end-to-end, zero-host-transfer InfoNCE training step.
   Performs 100% in OpenXLA PJRT VRAM:
   1. Gathers entity embeddings from W_embed:
      V_h = gather(W_embed, I_h), V_t = gather(W_embed, I_t)
   2. Subspace projections:
      U_h = V_h * W, U_t = V_t * W, U_hr = U_h * R
   3. Scores & in-graph Softmax:
      Scores = (U_hr * U_t^T) / tau
      P = softmax(Scores)
   4. Analytical InfoNCE adjoint matrix:
      D_P = P - Target
      G_S = D_P * Mask_Scale
   5. Backward adjoint gradient contractions:
      adj_Ut = G_S^T * U_hr, adj_Uhr = G_S * U_t
      dR = U_h^T * adj_Uhr, adj_Uh = adj_Uhr * R^T
      dW_h = V_h^T * adj_Uh, dW_t = V_t^T * adj_Ut
      dW = dW_h + dW_t
   6. In-graph SGD parameter updates:
      W_new = W - (lr * lambda_tl) * dW
      R_new = R - (lr * lambda_tl) * dR"
  ([_k-triples _dim-in _dim-mem opts]
   (let [tau (double (or (:tau opts) 0.2))
         lr (double (or (:lr opts) 0.01))
         lambda-tl (double (or (:lambda-tl opts) 0.3))
         step-scale (* lr lambda-tl)]
     [:block {:name :in_vram_contrastive_step}
      [:gather [:V_h :k :din] [:W_embed :v :din] [:I_h :k]]
      [:gather [:V_t :k :din] [:W_embed :v :din] [:I_t :k]]
      [:= [:U_h :k :dm] [:V_h :k :din] [:W :din :dm]]
      [:= [:U_t :k :dm] [:V_t :k :din] [:W :din :dm]]
      [:= [:U_hr :k :dm2] [:U_h :k :dm1] [:R :dm1 :dm2]]
      [:= [:Scores :k1 :k2] {:scale (/ 1.0 tau)} [:U_hr :k1 :dm] [:U_t :k2 :dm]]
      [:softmax [:P :k1 :k2] [:Scores :k1 :k2]]
      [:- [:D_P :k1 :k2] [:P :k1 :k2] [:Target :k1 :k2]]
      [:* [:G_S :k1 :k2] [:D_P :k1 :k2] [:Mask_Scale :k1 :k2]]
      [:= [:adj_Ut :k2 :dm] [:G_S :k1 :k2] [:U_hr :k1 :dm]]
      [:= [:adj_Uhr :k1 :dm] [:G_S :k1 :k2] [:U_t :k2 :dm]]
      [:= [:dR :dm1 :dm2] [:U_h :k :dm1] [:adj_Uhr :k :dm2]]
      [:= [:adj_Uh :k :dm1] [:adj_Uhr :k :dm2] [:R :dm1 :dm2]]
      [:= [:dW_h :din :dm] [:V_h :k :din] [:adj_Uh :k :dm]]
      [:= [:dW_t :din :dm] [:V_t :k :din] [:adj_Ut :k :dm]]
      [:+ [:dW :din :dm] [:dW_h :din :dm] [:dW_t :din :dm]]
      [:= [:scaled_dW :din :dm] {:scale step-scale} [:dW :din :dm]]
      [:- [:W_new :din :dm] [:W :din :dm] [:scaled_dW :din :dm]]
      [:= [:scaled_dR :dm1 :dm2] {:scale step-scale} [:dR :dm1 :dm2]]
      [:- [:R_new :dm1 :dm2] [:R :dm1 :dm2] [:scaled_dR :dm1 :dm2]]])))

(defn in-vram-resolver-step-ast
  "Constructs Tensor Logic AST for an end-to-end, zero-host-transfer training step for
   the Stage 2 Non-Linear Resolver (W_Q and W_K).
   Performs 100% in OpenXLA PJRT VRAM:
   1. Gathers entity embeddings from W_embed:
      V_h = gather(W_embed, I_h), V_t = gather(W_embed, I_t)
   2. Relational type modulation from Stage 1:
      U_h = V_h * W_mem, U_hr = U_h * R, U_norm = rms_norm(U_hr)
   3. Non-linear query and key projections:
      Q_raw = V_h * W_Q
      Q_mod = Q_raw * U_norm
      Q_rel = rms_norm(Q_mod)
      K_raw = V_t * W_K
      K_cand = rms_norm(K_raw)
   4. Scores & in-graph Softmax:
      Scores = (Q_rel * K_cand^T) / tau
      P = softmax(Scores)
   5. Analytical InfoNCE adjoint matrix:
      D_P = P - Target
      G_S = D_P * Mask_Scale
   6. Backward adjoint gradient contractions:
      adj_K = G_S^T * Q_rel
      adj_Q = G_S * K_cand
      adj_Q_raw = adj_Q * U_norm
      dW_K = V_t^T * adj_K
      dW_Q = V_h^T * adj_Q_raw
   7. In-graph SGD parameter updates:
      W_Q_new = W_Q - (lr * lambda_tl) * dW_Q
      W_K_new = W_K - (lr * lambda_tl) * dW_K"
  ([_k-triples _dim-in _dim-mem opts]
   (let [tau (double (or (:tau opts) 0.1))
         lr (double (or (:lr opts) 0.05))
         lambda-tl (double (or (:lambda-tl opts) 1.0))
         step-scale (* lr lambda-tl)]
     [:block {:name :in_vram_resolver_step}
      [:gather [:V_h :k :din] [:W_embed :v :din] [:I_h :k]]
      [:gather [:V_t :k :din] [:W_embed :v :din] [:I_t :k]]
      [:= [:U_h :k :dm] [:V_h :k :din] [:W_mem :din :dm]]
      [:= [:U_hr :k :dm2] [:U_h :k :dm1] [:R :dm1 :dm2]]
      [:rms-norm [:U_norm :k :dm2] [:U_hr :k :dm2]]
      [:= [:Q_raw :k :dm] [:V_h :k :din] [:W_Q :din :dm]]
      [:* [:Q_mod :k :dm] [:Q_raw :k :dm] [:U_norm :k :dm]]
      [:rms-norm [:Q_rel :k :dm] [:Q_mod :k :dm]]
      [:= [:K_raw :k :dm] [:V_t :k :din] [:W_K :din :dm]]
      [:rms-norm [:K_cand :k :dm] [:K_raw :k :dm]]
      [:= [:Scores :k1 :k2] {:scale (/ 1.0 tau)} [:Q_rel :k1 :dm] [:K_cand :k2 :dm]]
      [:softmax [:P :k1 :k2] [:Scores :k1 :k2]]
      [:- [:D_P :k1 :k2] [:P :k1 :k2] [:Target :k1 :k2]]
      [:* [:G_S :k1 :k2] [:D_P :k1 :k2] [:Mask_Scale :k1 :k2]]
      [:= [:adj_K :k2 :dm] [:G_S :k1 :k2] [:Q_rel :k1 :dm]]
      [:= [:adj_Q :k1 :dm] [:G_S :k1 :k2] [:K_cand :k2 :dm]]
      [:* [:adj_Q_raw :k :dm] [:adj_Q :k :dm] [:U_norm :k :dm]]
      [:= [:dW_K :din :dm] [:V_t :k :din] [:adj_K :k :dm]]
      [:= [:dW_Q :din :dm] [:V_h :k :din] [:adj_Q_raw :k :dm]]
      [:= [:scaled_dW_Q :din :dm] {:scale step-scale} [:dW_Q :din :dm]]
      [:- [:W_Q_new :din :dm] [:W_Q :din :dm] [:scaled_dW_Q :din :dm]]
      [:= [:scaled_dW_K :din :dm] {:scale step-scale} [:dW_K :din :dm]]
      [:- [:W_K_new :din :dm] [:W_K :din :dm] [:scaled_dW_K :din :dm]]])))

;; ==============================================================================
;; 2. InfoNCE Loss & Gradient Utilities
;; ==============================================================================

(defn compute-infonce-loss
  "Computes categorical cross-entropy loss over in-batch negative classification.
   Scores is flat float array [B, B]."
  [^floats scores batch-size]
  (let [b (long batch-size)]
    (loop [i 0 total-loss 0.0]
      (if (>= i b)
        (/ total-loss (double b))
        (let [row-offset (* i b)
              max-s (loop [j 1 m (double (aget scores row-offset))]
                      (if (>= j b)
                        m
                        (recur (inc j) (Math/max m (double (aget scores (+ row-offset j)))))))
              exp-sum (loop [j 0 s 0.0]
                        (if (>= j b)
                          s
                          (let [val (double (aget scores (+ row-offset j)))]
                            (recur (inc j) (+ s (Math/exp (- val max-s)))))))
              pos-val (double (aget scores (+ row-offset i)))
              prob (/ (Math/exp (- pos-val max-s)) exp-sum)
              loss (- (Math/log (Math/max 1e-12 prob)))]
          (recur (inc i) (+ total-loss loss)))))))

(defn compute-infonce-grad-s
  "Computes analytical gradient dL/d(Scores) for InfoNCE loss:
   G_S = (1 / (B * tau)) * (P - I_B)."
  [^floats scores batch-size tau]
  (let [b (long batch-size)
        scale (/ 1.0 (* (double b) (double tau)))
        grad (float-array (* b b))]
    (dotimes [i b]
      (let [row-offset (* i b)
            max-s (loop [j 1 m (double (aget scores row-offset))]
                    (if (>= j b)
                      m
                      (recur (inc j) (Math/max m (double (aget scores (+ row-offset j)))))))
            exp-sum (loop [j 0 s 0.0]
                      (if (>= j b)
                        s
                        (let [val (double (aget scores (+ row-offset j)))]
                          (recur (inc j) (+ s (Math/exp (- val max-s)))))))]
        (dotimes [j b]
          (let [pos? (= i j)
                prob (/ (Math/exp (- (double (aget scores (+ row-offset j))) max-s)) exp-sum)
                target (if pos? 1.0 0.0)
                g (* scale (- prob target))]
            (aset grad (+ row-offset j) (float g))))))
    grad))

;; ==============================================================================
;; 3. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn compile-contrastive-forward
  "Compiles OpenXLA PJRT executable for forward contrastive projection."
  ([batch-size dim-in dim-mem]
   (compile-contrastive-forward (xla/get-context) batch-size dim-in dim-mem nil))
  ([a b c d]
   (if (map? a)
     (compile-contrastive-forward a b c d nil)
     (compile-contrastive-forward (xla/get-context) a b c d)))
  ([ctx batch-size dim-in dim-mem opts]
   (let [b (long batch-size)
         din (long dim-in)
         dm (long dim-mem)
         invars [[:V_h [:tensor [b din] :f32]]
                 [:V_t [:tensor [b din] :f32]]
                 [:W [:tensor [din dm] :f32]]
                 [:R [:tensor [dm dm] :f32]]]
         ast (contrastive-forward-ast b din dm opts)]
     (sym/compile-query ctx "contrastive_forward" invars ast [:Scores :U_hr :U_t :U_h]))))

(defn compile-contrastive-backward
  "Compiles OpenXLA PJRT executable for backward adjoint gradient calculation."
  ([batch-size dim-in dim-mem]
   (compile-contrastive-backward (xla/get-context) batch-size dim-in dim-mem))
  ([ctx batch-size dim-in dim-mem]
   (let [b (long batch-size)
         din (long dim-in)
         dm (long dim-mem)
         invars [[:G_S [:tensor [b b] :f32]]
                 [:U_hr [:tensor [b dm] :f32]]
                 [:U_t [:tensor [b dm] :f32]]
                 [:U_h [:tensor [b dm] :f32]]
                 [:V_h [:tensor [b din] :f32]]
                 [:V_t [:tensor [b din] :f32]]
                 [:R [:tensor [dm dm] :f32]]]
         ast (contrastive-backward-ast b din dm)]
     (sym/compile-query ctx "contrastive_backward" invars ast [:dW :dR]))))

(defn compile-contrastive-update
  "Compiles OpenXLA PJRT executable for SGD parameter updates."
  ([dim-in dim-mem]
   (compile-contrastive-update (xla/get-context) dim-in dim-mem nil))
  ([a b c]
   (if (map? a)
     (compile-contrastive-update a b c nil)
     (compile-contrastive-update (xla/get-context) a b c)))
  ([ctx dim-in dim-mem opts]
   (let [din (long dim-in)
         dm (long dim-mem)
         invars [[:W [:tensor [din dm] :f32]]
                 [:dW [:tensor [din dm] :f32]]
                 [:R [:tensor [dm dm] :f32]]
                 [:dR [:tensor [dm dm] :f32]]]
         ast (contrastive-update-ast din dm opts)]
     (sym/compile-query ctx "contrastive_update" invars ast [:W_new :R_new]))))

(defn compile-in-vram-contrastive-step
  "Compiles OpenXLA PJRT executable for unified, 100% in-VRAM InfoNCE step:
   Gathers embeddings, computes projections, in-graph softmax, analytical adjoint matrix,
   backward contractions, and SGD updates directly in GPU memory."
  ([vocab-size dim-in k-triples dim-mem]
   (compile-in-vram-contrastive-step (xla/get-context) vocab-size dim-in k-triples dim-mem nil))
  ([a b c d e]
   (if (map? a)
     (compile-in-vram-contrastive-step a b c d e nil)
     (compile-in-vram-contrastive-step (xla/get-context) a b c d e)))
  ([ctx vocab-size dim-in k-triples dim-mem opts]
   (let [v (long vocab-size)
         din (long dim-in)
         k (long k-triples)
         dm (long dim-mem)
         dt (or (:dtype opts) :f32)
         invars [[:W_embed [:tensor [v din] dt]]
                 [:I_h [:tensor [k] :i32]]
                 [:I_t [:tensor [k] :i32]]
                 [:W [:tensor [din dm] dt]]
                 [:R [:tensor [dm dm] dt]]
                 [:Target [:tensor [k k] dt]]
                 [:Mask_Scale [:tensor [k k] dt]]]
         ast (in-vram-contrastive-step-ast k din dm opts)
         targets [:W_new :R_new :P :Scores]]
     (sym/compile-query ctx "in_vram_contrastive_step" invars ast targets))))

(defn compile-in-vram-resolver-step
  "Compiles OpenXLA PJRT executable for Stage 2 Non-Linear Resolver training step."
  ([ctx vocab-size dim-in k-triples dim-mem opts]
   (let [v (long vocab-size)
         din (long dim-in)
         k (long k-triples)
         dm (long dim-mem)
         dt (or (:dtype opts) :f32)
         invars [[:W_embed [:tensor [v din] dt]]
                 [:I_h [:tensor [k] :i32]]
                 [:I_t [:tensor [k] :i32]]
                 [:W_mem [:tensor [din dm] dt]]
                 [:R [:tensor [dm dm] dt]]
                 [:W_Q [:tensor [din dm] dt]]
                 [:W_K [:tensor [din dm] dt]]
                 [:Target [:tensor [k k] dt]]
                 [:Mask_Scale [:tensor [k k] dt]]]
         ast (in-vram-resolver-step-ast k din dm opts)
         targets [:W_Q_new :W_K_new :P :Scores]]
     (sym/compile-query ctx "in_vram_resolver_step" invars ast targets))))

;; ==============================================================================
;; 4. Impure Execution Wrappers (PJRT Dispatch)
;; ==============================================================================

(defn run-contrastive-forward!
  "Executes forward contrastive pass on PJRT device runtime."
  [exec vh vt w r]
  (sym/run-query! exec {:V_h vh :V_t vt :W w :R r}))

(defn run-contrastive-backward!
  "Executes backward adjoint pass on PJRT device runtime."
  [exec grad-s u-hr u-t u-h vh vt r]
  (sym/run-query! exec {:G_S grad-s :U_hr u-hr :U_t u-t :U_h u-h :V_h vh :V_t vt :R r}))

(defn run-contrastive-update!
  "Executes SGD parameter update on PJRT device runtime."
  [exec w dw r dr]
  (sym/run-query! exec {:W w :dW dw :R r :dR dr}))

(defn run-in-vram-contrastive-step!
  "Executes unified in-VRAM InfoNCE step on PJRT runtime."
  [exec w-embed ih it w r target mask-scale]
  (sym/run-query! exec {:W_embed w-embed
                        :I_h ih
                        :I_t it
                        :W w
                        :R r
                        :Target target
                        :Mask_Scale mask-scale}))
