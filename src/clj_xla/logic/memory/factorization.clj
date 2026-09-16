(ns clj-xla.logic.memory.factorization
  "Experiment E8: Dynamic In-VRAM Relation Induction via StableHLO Non-Negative Tensor Factorization.
   Models multi-entity tool interactions and dynamic environment co-occurrences as an incomplete
   3-way observation tensor X in R^{N x K x N}. Compiles in-graph PARAFAC / Non-Negative Matrix
   Factorization (NMF) multiplicative update rules into OpenXLA PJRT.
   Discovers latent relational predicates and extracts emergent relational cores into resident fast weights.
   Rule 2: Pure Functions (core logic sans-IO).
   Rule 4: Zero Java Escape Hatches (pure StableHLO execution via PJRT)."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Declarative Tensor Logic AST for PARAFAC Non-Negative Multiplicative Updates
;; ==============================================================================

(defn factorization-step-ast
  "Constructs Tensor Logic AST for 1 step of in-VRAM PARAFAC alternating non-negative multiplicative updates:
     1. Update A:       P_A[h, r] = sum_{k,t} (X[h, k, t] * B[k, r] * C[t, r])
                        Q_A[h, r] = sum_{k,t} (X_hat_0[h, k, t] * B[k, r] * C[t, r])
                        A_next[h, r] = A[h, r] * (P_A / (Q_A + eps))
     2. Update B:       P_B[k, r] = sum_{h,t} (X[h, k, t] * A_next[h, r] * C[t, r])
                        Q_B[k, r] = sum_{h,t} (X_hat_1[h, k, t] * A_next[h, r] * C[t, r])
                        B_next[k, r] = B[k, r] * (P_B / (Q_B + eps))
     3. Update C:       P_C[t, r] = sum_{h,k} (X[h, k, t] * A_next[h, r] * B_next[k, r])
                        Q_C[t, r] = sum_{h,k} (X_hat_2[h, k, t] * A_next[h, r] * B_next[k, r])
                        C_next[t, r] = C[t, r] * (P_C / (Q_C + eps))
     4. Final X_hat:    X_hat_final[h, k, t] = sum_r (A_next[h, r] * B_next[k, r] * C_next[t, r])"
  [n k r]
  (let [n-long (long n)
        k-long (long k)
        r-long (long r)]
    [:block {:name :nmf_factorization_step}
     ;; 1. Step 1: Update A using current B and C
     [:= [:X_hat_0 :h :k :t] [:A :h :r] [:B :k :r] [:C :t :r]]
     [:= [:P_A :h :r] [:X :h :k :t] [:B :k :r] [:C :t :r]]
     [:= [:Q_A :h :r] [:X_hat_0 :h :k :t] [:B :k :r] [:C :t :r]]
     [:constant [:eps_A] {:value 1e-7 :type [:tensor [n-long r-long] :f32] :shape [n-long r-long]}]
     [:+ [:Q_A_eps :h :r] [:Q_A :h :r] [:eps_A]]
     [:/ [:Ratio_A :h :r] [:P_A :h :r] [:Q_A_eps :h :r]]
     [:* [:A_next :h :r] [:A :h :r] [:Ratio_A :h :r]]

     ;; 2. Step 2: Update B using A_next and current C
     [:= [:X_hat_1 :h :k :t] [:A_next :h :r] [:B :k :r] [:C :t :r]]
     [:= [:P_B :k :r] [:X :h :k :t] [:A_next :h :r] [:C :t :r]]
     [:= [:Q_B :k :r] [:X_hat_1 :h :k :t] [:A_next :h :r] [:C :t :r]]
     [:constant [:eps_B] {:value 1e-7 :type [:tensor [k-long r-long] :f32] :shape [k-long r-long]}]
     [:+ [:Q_B_eps :k :r] [:Q_B :k :r] [:eps_B]]
     [:/ [:Ratio_B :k :r] [:P_B :k :r] [:Q_B_eps :k :r]]
     [:* [:B_next :k :r] [:B :k :r] [:Ratio_B :k :r]]

     ;; 3. Step 3: Update C using A_next and B_next
     [:= [:X_hat_2 :h :k :t] [:A_next :h :r] [:B_next :k :r] [:C :t :r]]
     [:= [:P_C :t :r] [:X :h :k :t] [:A_next :h :r] [:B_next :k :r]]
     [:= [:Q_C :t :r] [:X_hat_2 :h :k :t] [:A_next :h :r] [:B_next :k :r]]
     [:constant [:eps_C] {:value 1e-7 :type [:tensor [n-long r-long] :f32] :shape [n-long r-long]}]
     [:+ [:Q_C_eps :t :r] [:Q_C :t :r] [:eps_C]]
     [:/ [:Ratio_C :t :r] [:P_C :t :r] [:Q_C_eps :t :r]]
     [:* [:C_next :t :r] [:C :t :r] [:Ratio_C :t :r]]

     ;; 4. Final step reconstruction with fully updated factors
     [:= [:X_hat_final :h :k :t] [:A_next :h :r] [:B_next :k :r] [:C_next :t :r]]]))

;; ==============================================================================
;; 2. In-VRAM Session Initialization & PJRT Compilation Boundary
;; ==============================================================================

(defn init-factorization-session
  "Initializes OpenXLA PJRT factorization session:
   ctx: OpenXLA context (optional)
   opts: {:num-entities n :num-contexts k :rank r :seed seed}"
  ([opts]
   (init-factorization-session (xla/get-context) opts))
  ([ctx opts]
   (let [n (long (or (:num-entities opts) 16))
         k (long (or (:num-contexts opts) 4))
         r (long (or (:rank opts) 3))
         seed (long (or (:seed opts) 42))
         rnd (java.util.Random. seed)

         ;; Initialize strictly positive non-negative factor matrices
         a-curr (float-array (* n r))
         _ (dotimes [i (* n r)]
             (aset-float a-curr i (float (+ 0.1 (* 0.9 (.nextDouble rnd))))))

         b-curr (float-array (* k r))
         _ (dotimes [i (* k r)]
             (aset-float b-curr i (float (+ 0.1 (* 0.9 (.nextDouble rnd))))))

         c-curr (float-array (* n r))
         _ (dotimes [i (* n r)]
             (aset-float c-curr i (float (+ 0.1 (* 0.9 (.nextDouble rnd))))))

         invars [[:X [:tensor [n k n] :f32]]
                 [:A [:tensor [n r] :f32]]
                 [:B [:tensor [k r] :f32]]
                 [:C [:tensor [n r] :f32]]]
         ast (factorization-step-ast n k r)
         exec (sym/compile-query ctx "nmf_factorization_step" invars ast
                                 [:A_next :B_next :C_next :X_hat_final])

         ;; Calculate resident VRAM memory footprint
         vram-bytes (* (+ (* n k n) (* n r) (* k r) (* n r)) 4)]

     {:ctx ctx
      :n n
      :k k
      :r r
      :exec exec
      :a-curr a-curr
      :b-curr b-curr
      :c-curr c-curr
      :vram-bytes vram-bytes})))

;; ==============================================================================
;; 3. Execution & Multiplicative Update Stepper
;; ==============================================================================

(defn step-factorization!
  "Executes one turn of non-negative tensor factorization in OpenXLA PJRT:
   Returns {:A ... :B ... :C ... :X-hat ... :rel-error ... :fidelity ... :latency-ms ...}."
  [session x-data]
  (let [{:keys [n k r exec a-curr b-curr c-curr vram-bytes]} session
        n-long (long n)
        k-long (long k)
        r-long (long r)
        total-x (* n-long k-long n-long)

        t0 (System/nanoTime)
        out (sym/run-query! exec {:X x-data
                                  :A a-curr
                                  :B b-curr
                                  :C c-curr})
        elapsed-ms (/ (- (System/nanoTime) t0) 1e6)

        ^floats a-next (get out :A_next)
        ^floats b-next (get out :B_next)
        ^floats c-next (get out :C_next)
        ^floats x-hat (get out :X_hat_final)

        ;; Update in-place resident state
        _ (System/arraycopy a-next 0 a-curr 0 (* n-long r-long))
        _ (System/arraycopy b-next 0 b-curr 0 (* k-long r-long))
        _ (System/arraycopy c-next 0 c-curr 0 (* n-long r-long))

        ;; Compute Frobenius reconstruction norm host-side
        diff-sq (double-array 1)
        x-sq (double-array 1)
        _ (dotimes [i total-x]
            (let [xv (double (aget ^floats x-data i))
                  hv (double (aget x-hat i))
                  d (- xv hv)]
              (aset diff-sq 0 (+ (aget diff-sq 0) (* d d)))
              (aset x-sq 0 (+ (aget x-sq 0) (* xv xv)))))

        frob-norm (Math/sqrt (aget diff-sq 0))
        x-norm (Math/sqrt (aget x-sq 0))
        rel-err (if (pos? x-norm) (/ frob-norm x-norm) 0.0)
        fidelity (max 0.0 (- 1.0 rel-err))]

    {:A a-next
     :B b-next
     :C c-next
     :X-hat x-hat
     :frobenius-loss frob-norm
     :rel-error rel-err
     :fidelity fidelity
     :latency-ms elapsed-ms
     :vram-bytes vram-bytes}))

(defn fit-factorization!
  "Runs multiple iterations of in-VRAM tensor factorization:
   Returns final step result augmented with :history and :total-latency-ms."
  [session x-data iterations]
  (let [num-it (long iterations)
        t0 (System/nanoTime)
        history (atom [])]
    (loop [iter 0
           last-res nil]
      (if (>= iter num-it)
        (let [total-ms (/ (- (System/nanoTime) t0) 1e6)]
          (assoc last-res
                 :iterations num-it
                 :history @history
                 :total-latency-ms total-ms))
        (let [res (step-factorization! session x-data)]
          (swap! history conj {:iter iter
                               :fidelity (:fidelity res)
                               :rel-error (:rel-error res)
                               :latency-ms (:latency-ms res)})
          (recur (inc iter) res))))))

;; ==============================================================================
;; 4. Synthetic Multi-Relational Observation Tensor Generator
;; ==============================================================================

(defn generate-synthetic-observation-tensor
  "Generates an observation tensor X in R^{N x K x N} composed from R ground-truth
   non-negative latent relations with sparse tool activations."
  [n k r seed]
  (let [n-long (long n)
        k-long (long k)
        r-long (long r)
        rnd (java.util.Random. (long seed))

        ;; True latent factors
        a-true (mapv (fn [_] (mapv (fn [_] (if (< (.nextDouble rnd) 0.5) 1.0 0.0)) (range r-long))) (range n-long))
        b-true (mapv (fn [_] (mapv (fn [_] (if (< (.nextDouble rnd) 0.6) 1.0 0.0)) (range r-long))) (range k-long))
        c-true (mapv (fn [_] (mapv (fn [_] (if (< (.nextDouble rnd) 0.5) 1.0 0.0)) (range r-long))) (range n-long))

        x-data (float-array (* n-long k-long n-long))]
    (dotimes [h n-long]
      (dotimes [ctx-k k-long]
        (dotimes [t n-long]
          (let [sum-val (reduce (fn [acc comp-r]
                                  (+ acc (* (nth (nth a-true h) comp-r)
                                            (nth (nth b-true ctx-k) comp-r)
                                            (nth (nth c-true t) comp-r))))
                                0.0
                                (range r-long))
                idx (+ (* h k-long n-long) (* ctx-k n-long) t)]
            (aset-float x-data idx (float sum-val))))))
    {:x-data x-data
     :A a-true
     :B b-true
     :C c-true}))

;; ==============================================================================
;; 5. Latent Relational Core Extraction (Neuro-Symbolic Bridge)
;; ==============================================================================

(defn extract-induced-relational-cores
  "Extracts discovered latent components into crisp relational core matrices:
   For each latent factor r in {0..R-1}:
     Core_r[h, t] = A[h, r] * C[t, r]
     Energy = sqrt(sum_k B[k, r]^2)"
  [session fit-res]
  (let [{:keys [n k r]} session
        n-long (long n)
        k-long (long k)
        r-long (long r)
        ^floats a (:A fit-res)
        ^floats b (:B fit-res)
        ^floats c (:C fit-res)]
    (mapv
     (fn [comp-r]
       (let [core-matrix (float-array (* n-long n-long))
             ctx-weights (mapv (fn [ctx-k] (aget b (+ (* ctx-k r-long) comp-r))) (range k-long))
             energy (Math/sqrt (reduce + (map #(double (* % %)) ctx-weights)))]
         (dotimes [h n-long]
           (dotimes [t n-long]
             (let [a-val (double (aget a (+ (* h r-long) comp-r)))
                   c-val (double (aget c (+ (* t r-long) comp-r)))
                   idx (+ (* h n-long) t)]
               (aset-float core-matrix idx (float (* a-val c-val))))))
         {:factor-id comp-r
          :matrix core-matrix
          :energy energy
          :context-weights ctx-weights}))
     (range r-long))))
