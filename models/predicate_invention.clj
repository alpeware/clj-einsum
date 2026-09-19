(ns models.predicate-invention
  "Gradient-Based Predicate Invention in Declarative Tensor Logic (Experiment E16/E17).
   Learns a latent relation core L = sigma(Z / tau) via gradient descent to recover
   hidden predicates (Parent) from observed compositions (Grandparent = L . L, Sibling = L^T . L).
   Compiled 100% in OpenXLA PJRT VRAM (Rule 4: Zero Java bypasses). Pure functions (Rule 2)."
  (:require [einsum.core :as xla]
            [einsum.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Tensor Logic AST Constructors (Sans-IO)
;; ==============================================================================

(defn predicate-invention-step-ast
  "Constructs Tensor Logic AST for an end-to-end in-VRAM training step.
   Supports ablations:
   - :a0 : Rank-0 baseline (no latent core; GP and Sibling scored via bilinear W)
   - :a1 : Frozen latent core (Z is fixed, never updated)
   - :a2 : Full invention model (learned Z with annealing and L1 sparsity)
   - :a3 : Low-rank latent logits (Z = A . B^T, r in {4, 8, 16})"
  [_num-entities _dim opts]
  (let [ablation (or (:ablation opts) :a2)
        lr (double (or (:lr opts) 0.05))
        lr-z (double (or (:lr-z opts) (* 5.0 lr)))
        tau-ce (double (or (:tau-ce opts) 0.1))
        inv-tau-ce (/ 1.0 tau-ce)]
    [:block {:name (keyword (str "predicate_invention_step_" (name ablation)))}
     ;; 1. Baseline Bilinear Scoring Path
     [:= [:U_gp :h :d2] [:E :h :d1] [:W_gp :d1 :d2]]
     [:= [:S_gp_base :h :t] [:U_gp :h :d2] [:E :t :d2]]
     [:= [:U_sib :s1 :d2] [:E :s1 :d1] [:W_sib :d1 :d2]]
     [:= [:S_sib_base :s1 :s2] [:U_sib :s1 :d2] [:E :s2 :d2]]

     ;; 2. Latent Invention Scoring Path
     (if (= ablation :a0)
       [:block {:name :a0_scoring}
        [:= [:Scores_gp :h :t] {:scale inv-tau-ce} [:S_gp_base :h :t]]
        [:= [:Scores_sib :s1 :s2] {:scale inv-tau-ce} [:S_sib_base :s1 :s2]]]

       [:block {:name :latent_scoring}
        (when (= ablation :a3)
          [:= [:Z :i :j] [:A :i :r] [:B :j :r]])
        [:* [:Z_tau :i :j] [:Z :i :j] [:Inv_tau :i :j]]
        [:= [:L :i :j] {:act :sigmoid} [:Z_tau :i :j]]
        ;; Grandparent: 2-hop composition L . L
        [:= [:S_gp_inv :h :t] [:L :h :m] [:L :m :t]]
        ;; Sibling: shared parent L^T . L
        [:= [:S_sib_inv :s1 :s2] [:L :p :s1] [:L :p :s2]]
        ;; Combined scores
        [:+ [:S_gp_sum :h :t] [:S_gp_base :h :t] [:S_gp_inv :h :t]]
        [:= [:Scores_gp :h :t] {:scale inv-tau-ce} [:S_gp_sum :h :t]]
        [:+ [:S_sib_sum :s1 :s2] [:S_sib_base :s1 :s2] [:S_sib_inv :s1 :s2]]
        [:= [:Scores_sib :s1 :s2] {:scale inv-tau-ce} [:S_sib_sum :s1 :s2]]])

     ;; 3. Row-Wise Softmax and Error Gradients
     [:softmax [:P_gp :h :t] [:Scores_gp :h :t]]
     [:softmax [:P_sib :s1 :s2] [:Scores_sib :s1 :s2]]
     [:- [:D_gp :h :t] [:P_gp :h :t] [:Target_gp :h :t]]
     [:* [:G_gp_raw :h :t] [:D_gp :h :t] [:Mask_gp :h :t]]
     [:= [:G_gp :h :t] {:scale inv-tau-ce} [:G_gp_raw :h :t]]
     [:- [:D_sib :s1 :s2] [:P_sib :s1 :s2] [:Target_sib :s1 :s2]]
     [:* [:G_sib_raw :s1 :s2] [:D_sib :s1 :s2] [:Mask_sib :s1 :s2]]
     [:= [:G_sib :s1 :s2] {:scale inv-tau-ce} [:G_sib_raw :s1 :s2]]

     ;; 4. Baseline Parameter Adjoints & Updates
     ;; Grandparent adjoints
     [:= [:adj_U_gp :h :d2] [:G_gp :h :t] [:E :t :d2]]
     [:= [:dW_gp :d1 :d2] [:E :h :d1] [:adj_U_gp :h :d2]]
     [:= [:adj_E_gp_h :h :d1] [:adj_U_gp :h :d2] [:W_gp :d1 :d2]]
     [:= [:adj_E_gp_t :t :d1] [:G_gp :h :t] [:U_gp :h :d1]]
     [:+ [:dE_gp :h :d1] [:adj_E_gp_h :h :d1] [:adj_E_gp_t :h :d1]]

     ;; Sibling adjoints
     [:= [:adj_U_sib :s1 :d2] [:G_sib :s1 :s2] [:E :s2 :d2]]
     [:= [:dW_sib :d1 :d2] [:E :s1 :d1] [:adj_U_sib :s1 :d2]]
     [:= [:adj_E_sib_h :s1 :d1] [:adj_U_sib :s1 :d2] [:W_sib :d1 :d2]]
     [:= [:adj_E_sib_t :s2 :d1] [:G_sib :s1 :s2] [:U_sib :s1 :d1]]
     [:+ [:dE_sib :h :d1] [:adj_E_sib_h :h :d1] [:adj_E_sib_t :h :d1]]

     ;; Combined E gradient & SGD updates
     [:+ [:dE :h :d1] [:dE_gp :h :d1] [:dE_sib :h :d1]]
     [:= [:scaled_dE :h :d1] {:scale lr} [:dE :h :d1]]
     [:- [:E_new :h :d1] [:E :h :d1] [:scaled_dE :h :d1]]
     [:= [:scaled_dW_gp :d1 :d2] {:scale lr} [:dW_gp :d1 :d2]]
     [:- [:W_gp_new :d1 :d2] [:W_gp :d1 :d2] [:scaled_dW_gp :d1 :d2]]
     [:= [:scaled_dW_sib :d1 :d2] {:scale lr} [:dW_sib :d1 :d2]]
     [:- [:W_sib_new :d1 :d2] [:W_sib :d1 :d2] [:scaled_dW_sib :d1 :d2]]

     ;; 5. Latent Parameter Adjoints & Updates (A2 and A3 only)
     (when (or (= ablation :a2) (= ablation :a3))
       [:block {:name :latent_backward}
        ;; Grandparent adjoint: G_gp . L^T + L^T . G_gp
        [:= [:adj_L_gp1 :h :m] [:G_gp :h :t] [:L :m :t]]
        [:= [:adj_L_gp2 :m :t] [:G_gp :h :t] [:L :h :m]]
        [:+ [:adj_L_gp :i :j] [:adj_L_gp1 :i :j] [:adj_L_gp2 :i :j]]

        ;; Sibling adjoint: L . G_sib^T + L . G_sib
        [:= [:adj_L_sib1 :p :s1] [:L :p :s2] [:G_sib :s1 :s2]]
        [:= [:adj_L_sib2 :p :s2] [:L :p :s1] [:G_sib :s1 :s2]]
        [:+ [:adj_L_sib :i :j] [:adj_L_sib1 :i :j] [:adj_L_sib2 :i :j]]

        ;; Total L adjoint + L1 sparsity penalty
        [:+ [:adj_L_loss :i :j] [:adj_L_gp :i :j] [:adj_L_sib :i :j]]
        [:+ [:adj_L_total :i :j] [:adj_L_loss :i :j] [:L1_grad :i :j]]

        ;; Sigmoid derivative: (adj_L) * L * (1 - L) * (1 / tau)
        [:- [:One_minus_L :i :j] [:Ones :i :j] [:L :i :j]]
        [:* [:L_deriv :i :j] [:L :i :j] [:One_minus_L :i :j]]
        [:* [:dZ_unscaled :i :j] [:adj_L_total :i :j] [:L_deriv :i :j]]
        [:* [:dZ :i :j] [:dZ_unscaled :i :j] [:Inv_tau :i :j]]

        (if (= ablation :a2)
          [:block {:name :a2_z_update}
           [:= [:scaled_dZ :i :j] {:scale lr-z} [:dZ :i :j]]
           [:- [:Z_new :i :j] [:Z :i :j] [:scaled_dZ :i :j]]]

          ;; A3: Low-rank updates for A and B
          [:block {:name :a3_ab_update}
           [:= [:dA :i :r] [:dZ :i :j] [:B :j :r]]
           [:= [:dB :j :r] [:dZ :i :j] [:A :i :r]]
           [:= [:scaled_dA :i :r] {:scale lr-z} [:dA :i :r]]
           [:- [:A_new :i :r] [:A :i :r] [:scaled_dA :i :r]]
           [:= [:scaled_dB :j :r] {:scale lr-z} [:dB :j :r]]
           [:- [:B_new :j :r] [:B :j :r] [:scaled_dB :j :r]]])])]))

;; ==============================================================================
;; 2. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn compile-predicate-invention-step
  "Compiles OpenXLA PJRT executable for predicate invention step."
  ([opts]
   (compile-predicate-invention-step (xla/get-context) opts))
  ([ctx opts]
   (let [n (long (or (:num-entities opts) 64))
         din (long (or (:dim opts) 64))
         r (long (or (:rank opts) 8))
         ablation (or (:ablation opts) :a2)
         dt :f32
         base-invars [[:E [:tensor [n din] dt]]
                      [:W_gp [:tensor [din din] dt]]
                      [:W_sib [:tensor [din din] dt]]
                      [:Target_gp [:tensor [n n] dt]]
                      [:Mask_gp [:tensor [n n] dt]]
                      [:Target_sib [:tensor [n n] dt]]
                      [:Mask_sib [:tensor [n n] dt]]]
         invars (cond
                  (= ablation :a0)
                  base-invars

                  (= ablation :a1)
                  (into base-invars [[:Z [:tensor [n n] dt]]
                                     [:Inv_tau [:tensor [n n] dt]]])

                  (= ablation :a2)
                  (into base-invars [[:Z [:tensor [n n] dt]]
                                     [:Inv_tau [:tensor [n n] dt]]
                                     [:L1_grad [:tensor [n n] dt]]
                                     [:Ones [:tensor [n n] dt]]])

                  (= ablation :a3)
                  (into base-invars [[:A [:tensor [n r] dt]]
                                     [:B [:tensor [n r] dt]]
                                     [:Inv_tau [:tensor [n n] dt]]
                                     [:L1_grad [:tensor [n n] dt]]
                                     [:Ones [:tensor [n n] dt]]]))
         outvars (cond
                   (= ablation :a0)
                   [:E_new :W_gp_new :W_sib_new :Scores_gp :Scores_sib :P_gp :P_sib]

                   (= ablation :a1)
                   [:E_new :W_gp_new :W_sib_new :L :Scores_gp :Scores_sib :P_gp :P_sib]

                   (= ablation :a2)
                   [:E_new :W_gp_new :W_sib_new :Z_new :L :Scores_gp :Scores_sib :P_gp :P_sib]

                   (= ablation :a3)
                   [:E_new :W_gp_new :W_sib_new :A_new :B_new :Z :L :Scores_gp :Scores_sib :P_gp :P_sib])
         ast (predicate-invention-step-ast n din opts)
         graph-name (str "pred_inv_" (name ablation) "_n" n "_d" din)]
     (sym/compile-query ctx graph-name invars ast outvars))))

(defn run-predicate-invention-step!
  "Executes the compiled predicate invention step executable with input batch map."
  [exec batch-map]
  (sym/run-query! exec batch-map))

;; ==============================================================================
;; 3. Parameter Initialization & Batch Preparation
;; ==============================================================================

(defn- gaussian-array [^long size ^double mean ^double std ^java.util.Random rnd]
  (let [arr (float-array size)]
    (dotimes [i size]
      (aset arr i (float (+ mean (* std (.nextGaussian rnd))))))
    arr))

(defn init-parameters
  "Initializes model parameters on host according to ablation.
   Options:
   - :dim (default 64)
   - :rank (default 8)
   - :seed (default 42)"
  [forest opts]
  (let [n (long (:num-entities forest))
        dim (long (or (:dim opts) 64))
        r (long (or (:rank opts) 8))
        seed (long (or (:seed opts) 42))
        rnd (java.util.Random. seed)
        std-d (/ 1.0 (Math/sqrt (double dim)))
        std-r (/ 1.0 (Math/sqrt (double r)))
        e (gaussian-array (* n dim) 0.0 std-d rnd)
        w-gp (gaussian-array (* dim dim) 0.0 std-d rnd)
        w-sib (gaussian-array (* dim dim) 0.0 std-d rnd)
        z-mean (double (or (:z-init-mean opts) 0.0))
        z (gaussian-array (* n n) z-mean 0.1 rnd)
        a (gaussian-array (* n r) 0.0 std-r rnd)
        b (gaussian-array (* n r) 0.0 std-r rnd)]
    {:n n
     :dim dim
     :rank r
     :E e
     :W_gp w-gp
     :W_sib w-sib
     :Z z
     :A a
     :B b}))

(defn- make-target-and-mask-matrices
  "Constructs Target [N x N] and Mask [N x N] float arrays for a relation's train pairs."
  [pairs ^long n]
  (let [target (float-array (* n n))
        mask (float-array (* n n))
        ;; Group targets by head
        head->tails (group-by first pairs)
        total-train (double (max 1 (count pairs)))
        weight (/ 1.0 total-train)]
    (doseq [[h group] head->tails]
      (let [k (double (count group))
            tgt-val (/ 1.0 k)]
        (doseq [[_ t] group]
          (aset target (+ (* (long h) n) (long t)) (float tgt-val)))
        ;; Mask is non-zero for active training query heads
        (dotimes [t n]
          (aset mask (+ (* (long h) n) t) (float weight)))))
    {:target target :mask mask}))

(defn make-step-batch
  "Prepares input tensors map for a single training step at current temperature tau."
  [forest state opts]
  (let [n (long (:n state))
        tau (double (or (:tau opts) 1.0))
        lambda-l1 (double (or (:lambda-l1 opts) 1e-4))
        gp-train (get-in forest [:splits :grandparent :train])
        sib-train (get-in forest [:splits :sibling :train])
        {:keys [target mask]} (make-target-and-mask-matrices gp-train n)
        sib-mats (make-target-and-mask-matrices sib-train n)
        inv-tau-arr (float-array (* n n))
        l1-arr (float-array (* n n))
        ones-arr (float-array (* n n))
        inv-tau-val (float (/ 1.0 tau))
        l1-val (float lambda-l1)]
    (dotimes [i (* n n)]
      (aset inv-tau-arr i inv-tau-val)
      (aset l1-arr i l1-val)
      (aset ones-arr i (float 1.0)))
    {:E (:E state)
     :W_gp (:W_gp state)
     :W_sib (:W_sib state)
     :Z (:Z state)
     :A (:A state)
     :B (:B state)
     :Target_gp target
     :Mask_gp mask
     :Target_sib (:target sib-mats)
     :Mask_sib (:mask sib-mats)
     :Inv_tau inv-tau-arr
     :L1_grad l1-arr
     :Ones ones-arr}))

;; ==============================================================================
;; 4. Evaluation & Metrics Protocol
;; ==============================================================================

(defn compute-f1-metrics
  "Computes precision, recall, and F1 between predicted boolean matrix L_hat
   (thresholded at `threshold`, default 0.5) and true parent matrix."
  ([pred-arr true-arr]
   (compute-f1-metrics pred-arr true-arr 0.5))
  ([pred-arr true-arr threshold]
   (let [pred-arr ^floats pred-arr
         true-arr ^floats true-arr
         threshold (double threshold)
         size (alength pred-arr)]
     (loop [i 0
            tp 0.0
            fp 0.0
            fn-count 0.0]
       (if (>= i size)
         (let [p (if (zero? (+ tp fp)) 0.0 (/ tp (+ tp fp)))
               r (if (zero? (+ tp fn-count)) 0.0 (/ tp (+ tp fn-count)))
               f1 (if (zero? (+ p r)) 0.0 (/ (* 2.0 p r) (+ p r)))]
           {:precision (double p)
            :recall (double r)
            :f1 (double f1)
            :tp (long tp)
            :fp (long fp)
            :fn (long fn-count)})
         (let [p-val (if (> (double (aget pred-arr i)) threshold) 1.0 0.0)
               t-val (if (> (double (aget true-arr i)) 0.5) 1.0 0.0)
               new-tp (if (and (= p-val 1.0) (= t-val 1.0)) (inc tp) tp)
               new-fp (if (and (= p-val 1.0) (= t-val 0.0)) (inc fp) fp)
               new-fn (if (and (= p-val 0.0) (= t-val 1.0)) (inc fn-count) fn-count)]
           (recur (inc i) new-tp new-fp new-fn)))))))

(defn compute-two-hop-closure
  "Computes % of grandparent pairs that can be explained by a 2-hop path
   through discretized L_hat (L > threshold): exists m s.t. L[h,m] > 0.5 and L[m,t] > 0.5."
  ([l-arr gp-pairs n]
   (compute-two-hop-closure l-arr gp-pairs n 0.5))
  ([l-arr gp-pairs n threshold]
   (if (empty? gp-pairs)
     1.0
     (let [l-arr ^floats l-arr
           n (long n)
           threshold (double threshold)
           explained
           (count
            (filter
             (fn [[h t]]
               (some (fn [m]
                       (and (> (double (aget l-arr (+ (* (long h) n) (long m)))) threshold)
                            (> (double (aget l-arr (+ (* (long m) n) (long t)))) threshold)))
                     (range n)))
             gp-pairs))]
       (/ (double explained) (double (count gp-pairs)))))))

(defn eval-link-prediction
  "Evaluates link prediction for held-out pairs against all N candidate entities.
   Returns {:hits-1 ... :hits-10 ... :mrr ... :mean-rank ... :n-queries ...}."
  [scores-arr held-out-pairs n]
  (if (empty? held-out-pairs)
    {:hits-1 0.0 :hits-10 0.0 :mrr 0.0 :mean-rank 0.0 :n-queries 0}
    (let [scores-arr ^floats scores-arr
          n (long n)
          n-queries (count held-out-pairs)
          ranks
          (mapv (fn [[h t]]
                  (let [target-score (double (aget scores-arr (+ (* (long h) n) (long t))))
                        ;; Rank is 1 + number of entities with strictly greater score
                        strictly-greater
                        (reduce + (map (fn [cand]
                                         (if (> (double (aget scores-arr (+ (* (long h) n) cand))) target-score)
                                           1 0))
                                       (range n)))
                        ;; Ties are broken neutrally by adding half the ties
                        ties
                        (reduce + (map (fn [cand]
                                         (if (and (not= cand (long t))
                                                  (== (double (aget scores-arr (+ (* (long h) n) cand))) target-score))
                                           1 0))
                                       (range n)))
                        rank (+ 1.0 (double strictly-greater) (* 0.5 (double ties)))]
                    rank))
                held-out-pairs)
          h1 (count (filter #(<= (double %) 1.0) ranks))
          h10 (count (filter #(<= (double %) 10.0) ranks))
          mrr (reduce + (map #(/ 1.0 (double %)) ranks))
          mean-rank (reduce + ranks)]
      {:hits-1 (/ (double h1) (double n-queries))
       :hits-10 (/ (double h10) (double n-queries))
       :mrr (/ (double mrr) (double n-queries))
       :mean-rank (/ (double mean-rank) (double n-queries))
       :n-queries n-queries})))
