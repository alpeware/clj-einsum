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
