(ns einsum.logic.memory.contrastive-test
  "Unit, generative, and invariant tests for Contrastive Subspace Pre-training on Knowledge Graphs
   (Experiment E3). Tests forward InfoNCE scoring, autodiff adjoint derivation,
   gradient descent monotonicity, and generalization to held-out test triples."
  (:require [einsum.logic.memory.contrastive :as contrast]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. AST Construction and Schema Tests
;; ==============================================================================

(deftest test-contrastive-ast-construction
  (testing "contrastive-forward-ast produces valid Hiccup block"
    (let [ast (contrast/contrastive-forward-ast 16 64 32 {:tau 0.1})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :contrastive_forward (get-in ast [1 :name])))))

  (testing "contrastive-backward-ast produces valid Hiccup block"
    (let [ast (contrast/contrastive-backward-ast 16 64 32)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :contrastive_backward (get-in ast [1 :name])))))

  (testing "contrastive-update-ast produces valid Hiccup block"
    (let [ast (contrast/contrastive-update-ast 64 32 {:lr 0.01})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :contrastive_update (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariant: Forward & Adjoint Shape Parity
;; ==============================================================================

(defspec prop-contrastive-shape-invariants 15
  (prop/for-all [b (gen/choose 4 8)
                 d-in (gen/choose 16 32)
                 d-mem (gen/choose 8 16)]
                (let [vh (float-array (* b d-in))
                      vt (float-array (* b d-in))
                      w (float-array (* d-in d-mem))
                      r (float-array (* d-mem d-mem))
          ;; Populate with deterministic test values
                      _ (dotimes [i (* b d-in)] (aset vh i (float 0.1)) (aset vt i (float 0.1)))
                      _ (dotimes [i (* d-in d-mem)] (aset w i (float 0.05)))
                      _ (dotimes [i (* d-mem d-mem)] (aset r i (if (= (mod i (inc d-mem)) 0) (float 1.0) (float 0.0))))
                      fwd-exec (contrast/compile-contrastive-forward b d-in d-mem {:tau 0.1})
                      fwd-out (contrast/run-contrastive-forward! fwd-exec vh vt w r)
                      ^floats scores (:Scores fwd-out)]
                  (and
       ;; Scores shape must be B x B
                   (= (* b b) (alength scores))
       ;; Projected shapes
                   (= (* b d-mem) (alength ^floats (:U_hr fwd-out)))
                   (= (* b d-mem) (alength ^floats (:U_t fwd-out)))))))

;; ==============================================================================
;; 3. Generative Invariant: Gradient Descent Loss Monotonicity
;; ==============================================================================

(defspec prop-gradient-step-loss-monotonicity 10
  (prop/for-all [b (gen/choose 4 8)]
                (let [d-in 16
                      d-mem 8
                      tau 0.5
                      lr 0.02
                      vh (float-array (* b d-in))
                      vt (float-array (* b d-in))
                      w (float-array (* d-in d-mem))
                      r (float-array (* d-mem d-mem))]
      ;; Populate distinct embeddings
                  (dotimes [i (* b d-in)]
                    (aset vh i (* (float 0.2) (float (Math/sin (double i)))))
                    (aset vt i (* (float 0.2) (float (Math/cos (double i))))))
                  (dotimes [i (* d-in d-mem)]
                    (aset w i (* (float 0.1) (float (Math/sin (* 1.5 (double i)))))))
                  (dotimes [i (* d-mem d-mem)]
                    (aset r i (if (= (mod i (inc d-mem)) 0) (float 1.0) (float 0.0))))

      ;; Forward pass at step 0
                  (let [fwd-exec (contrast/compile-contrastive-forward b d-in d-mem {:tau tau})
                        bwd-exec (contrast/compile-contrastive-backward b d-in d-mem)
                        upd-exec (contrast/compile-contrastive-update d-in d-mem {:lr lr})
                        fwd-0 (contrast/run-contrastive-forward! fwd-exec vh vt w r)
                        loss-0 (contrast/compute-infonce-loss (:Scores fwd-0) b)
                        grad-s (contrast/compute-infonce-grad-s (:Scores fwd-0) b tau)
                        bwd-0 (contrast/run-contrastive-backward! bwd-exec grad-s
                                                                  (:U_hr fwd-0) (:U_t fwd-0) (:U_h fwd-0)
                                                                  vh vt r)
                        upd-0 (contrast/run-contrastive-update! upd-exec w (:dW bwd-0) r (:dR bwd-0))
                        w-new (:W_new upd-0)
                        r-new (:R_new upd-0)
            ;; Forward pass at step 1
                        fwd-1 (contrast/run-contrastive-forward! fwd-exec vh vt w-new r-new)
                        loss-1 (contrast/compute-infonce-loss (:Scores fwd-1) b)]
        ;; Loss must decrease after 1 gradient step: loss-1 < loss-0
                    (< loss-1 loss-0)))))

;; ==============================================================================
;; 4. Unit Test: Pre-training Loop Convergence & Held-Out Generalization
;; ==============================================================================

(deftest test-contrastive-pretraining-generalization
  (testing "Pre-training on synthetic knowledge triples achieves > 75% Hits@1 on held-out test triples"
    (let [b 8
          d-in 16
          d-mem 8
          tau 0.2
          lr 0.08
          steps 80
          ;; Ground truth relation matrix M_rel (d-in x d-in) of rank <= d-mem
          m-rel (float-array (* d-in d-in))
          _ (dotimes [i (* d-in d-mem)]
              (aset m-rel i (float (Math/cos (* 1.9 (double i))))))
          ;; Generate 2*b distinct entity head vectors
          heads (float-array (* 2 b d-in))
          _ (dotimes [i (* 2 b d-in)]
              (aset heads i (float (Math/sin (* 2.3 (double i))))))
          ;; Project through ground-truth relation: v_t = v_h * M_rel
          tails (float-array (* 2 b d-in))]
      (dotimes [k (* 2 b)]
        (dotimes [j d-in]
          (let [sum (reduce + (map (fn [i] (* (aget heads (+ (* k d-in) i))
                                              (aget m-rel (+ (* i d-in) j))))
                                   (range d-in)))]
            (aset tails (+ (* k d-in) j) (float sum)))))

      (let [vh-train (float-array (* b d-in))
            vt-train (float-array (* b d-in))
            vh-test (float-array (* b d-in))
            vt-test (float-array (* b d-in))]
        (System/arraycopy heads 0 vh-train 0 (* b d-in))
        (System/arraycopy tails 0 vt-train 0 (* b d-in))
        (System/arraycopy heads (* b d-in) vh-test 0 (* b d-in))
        (System/arraycopy tails (* b d-in) vt-test 0 (* b d-in))

        ;; Initialize weights
        (let [w (float-array (* d-in d-mem))
              r (float-array (* d-mem d-mem))]
          (dotimes [i (* d-in d-mem)]
            (aset w i (* (float 0.1) (float (Math/cos (* 1.7 (double i)))))))
          (dotimes [i (* d-mem d-mem)]
            (aset r i (if (= (mod i (inc d-mem)) 0) (float 1.0) (float 0.0))))

          (let [fwd-exec (contrast/compile-contrastive-forward b d-in d-mem {:tau tau})
                bwd-exec (contrast/compile-contrastive-backward b d-in d-mem)
                upd-exec (contrast/compile-contrastive-update d-in d-mem {:lr lr})
                ;; Run training loop
                final-state
                (loop [step 0
                       curr-w w
                       curr-r r]
                  (if (>= step steps)
                    {:w curr-w :r curr-r}
                    (let [fwd (contrast/run-contrastive-forward! fwd-exec vh-train vt-train curr-w curr-r)
                          grad-s (contrast/compute-infonce-grad-s (:Scores fwd) b tau)
                          bwd (contrast/run-contrastive-backward! bwd-exec grad-s
                                                                  (:U_hr fwd) (:U_t fwd) (:U_h fwd)
                                                                  vh-train vt-train curr-r)
                          upd (contrast/run-contrastive-update! upd-exec curr-w (:dW bwd) curr-r (:dR bwd))]
                      (recur (inc step) (:W_new upd) (:R_new upd)))))

                ;; Evaluate on held-out test triples
                fwd-test (contrast/run-contrastive-forward! fwd-exec vh-test vt-test (:w final-state) (:r final-state))
                ^floats test-scores (:Scores fwd-test)
                hits-at-1 (reduce + (map (fn [i]
                                           (let [row-scores (mapv #(double (aget test-scores (+ (* i b) %))) (range b))
                                                 best-idx (first (apply max-key second (map-indexed vector row-scores)))]
                                             (if (= best-idx i) 1 0)))
                                         (range b)))
                acc (/ (double hits-at-1) b)]
            ;; Hits@1 on held-out test triples must exceed chance (1/8 = 12.5%) and reach >= 75%
            (is (>= acc 0.75) (format "Test Hits@1 must be >= 75%%, got %.1f%%" (* 100.0 acc)))))))))

(deftest test-in-vram-contrastive-step
  (testing "Unified in-VRAM InfoNCE step gathers, computes softmax, adjoints, and updates 100% on device"
    (let [v 64
          din 16
          k 4
          dm 8
          tau 0.2
          lr 0.05
          lambda-tl 1.0
          w-embed (float-array (* v din))
          _ (dotimes [i (* v din)]
              (aset w-embed i (* (float 0.1) (float (Math/sin (double i))))))
          ih (int-array [0 1 2 3])
          it (int-array [4 5 6 7])
          w (float-array (* din dm))
          _ (dotimes [i (* din dm)]
              (aset w i (* (float 0.1) (float (Math/cos (double i))))))
          r (float-array (* dm dm))
          _ (dotimes [i (* dm dm)]
              (aset r i (if (= (mod i (inc dm)) 0) (float 1.0) (float 0.0))))
          target (float-array (* k k))
          _ (dotimes [i k] (aset target (+ (* i k) i) (float 1.0)))
          mask-scale (float-array (* k k))
          scale-val (/ 1.0 (* (double k) tau))
          _ (dotimes [i (* k k)] (aset mask-scale i (float scale-val)))

          exec (contrast/compile-in-vram-contrastive-step v din k dm {:tau tau :lr lr :lambda-tl lambda-tl})
          out (contrast/run-in-vram-contrastive-step! exec w-embed ih it w r target mask-scale)]
      (is (contains? out :W_new))
      (is (contains? out :R_new))
      (is (contains? out :P))
      (is (contains? out :Scores))

      (let [^floats p (:P out)
            ^floats w-new (:W_new out)
            ^floats r-new (:R_new out)]
        (is (= (* din dm) (alength w-new)))
        (is (= (* dm dm) (alength r-new)))
        (is (= (* k k) (alength p)))

        ;; Softmax row sums must be ~1.0
        (dotimes [i k]
          (let [row-sum (reduce + (map #(double (aget p (+ (* i k) %))) (range k)))]
            (is (< (Math/abs (- row-sum 1.0)) 1e-4)
                (format "Row %d softmax sum must be 1.0, got %f" i row-sum))))))))
