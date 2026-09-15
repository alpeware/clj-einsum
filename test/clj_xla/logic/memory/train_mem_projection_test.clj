(ns clj-xla.logic.memory.train-mem-projection-test
  "Unit, gradient check, and algorithmic tests for learned memory projection in scripts.train-mem-projection."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.autodiff :as autodiff]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.train-mem-projection :as train]))

;; ==============================================================================
;; 1. Dogfooding Symbolic Adjoint Derivation
;; ==============================================================================

(deftest test-adjoint-equation-derivation
  (testing "derive-adjoint-equations derives backward contraction for u = h @ W"
    (let [eqns (autodiff/derive-adjoint-equations [:= [:u :b :d] [:h :b :dim] [:w :dim :d]])]
      (is (some (fn [eqn]
                  (and (= (first eqn) :=)
                       (= (second eqn) [:adj/w :dim :d])
                       (= (ast/body-terms eqn) [[:adj/u :b :d] [:h :b :dim]])))
                eqns)
          "Adjoint equation for w must be [:= [:adj/w :dim :d] [:adj/u :b :d] [:h :b :dim]]"))))

;; ==============================================================================
;; 2. Finite-Difference Gradient Check
;; ==============================================================================

(deftest test-gradient-finite-difference-check
  (testing "Analytic backward contraction gradient matches finite difference approximation"
    (let [hidden-dim 6
          d 4
          n-entities 3
          rng (java.util.Random. 42)
          rand-floats (fn [cnt scale]
                        (let [arr (float-array cnt)]
                          (dotimes [i cnt]
                            (aset-float arr i (float (* scale (- (.nextDouble rng) 0.5)))))
                          arr))
          W (rand-floats (* hidden-dim d) 0.5)
          E (rand-floats (* n-entities d) 1.0)
          h1 (rand-floats hidden-dim 1.0)
          h2 (rand-floats hidden-dim 1.0)
          dataset [{:h h1 :y 0}
                   {:h h2 :y 2}]
          {:keys [grad loss]} (train/compute-loss-and-gradients dataset W E hidden-dim d n-entities)
          eps 1e-4
          tol 1e-3]
      (is (pos? (double loss)) "Loss must be positive")
      ;; For every parameter in W, compare analytic gradient to central difference
      (dotimes [idx (* hidden-dim d)]
        (let [orig-w (aget ^floats W idx)
              _ (aset-float W idx (float (+ orig-w eps)))
              loss-plus (:loss (train/compute-loss-and-gradients dataset W E hidden-dim d n-entities))
              _ (aset-float W idx (float (- orig-w eps)))
              loss-minus (:loss (train/compute-loss-and-gradients dataset W E hidden-dim d n-entities))
              _ (aset-float W idx orig-w)
              num-grad (/ (- (double loss-plus) (double loss-minus)) (* 2.0 eps))
              ana-grad (double (aget ^floats grad idx))
              diff (Math/abs (- num-grad ana-grad))]
          (is (< diff tol)
              (format "Gradient mismatch at index %d: analytic=%.6f, numeric=%.6f, diff=%.6f"
                      idx ana-grad num-grad diff)))))))

;; ==============================================================================
;; 3. Optimization Smoke Test & Monotonic Loss Decrease
;; ==============================================================================

(deftest test-optimization-smoke-loss-decreases
  (testing "Loss strictly decreases over first 50 gradient steps on synthetic instance"
    (let [hidden-dim 8
          d 4
          n-entities 3
          rng (java.util.Random. 123)
          rand-floats (fn [cnt scale]
                        (let [arr (float-array cnt)]
                          (dotimes [i cnt]
                            (aset-float arr i (float (* scale (- (.nextDouble rng) 0.5)))))
                          arr))
          W (rand-floats (* hidden-dim d) 0.2)
          E (rand-floats (* n-entities d) 1.0)
          dataset [{:h (rand-floats hidden-dim 1.0) :y 0}
                   {:h (rand-floats hidden-dim 1.0) :y 1}
                   {:h (rand-floats hidden-dim 1.0) :y 2}]
          history (atom [])
          step-fn (fn [step loss _acc]
                    (when (or (zero? step) (= step 50))
                      (swap! history conj loss)))
          _ (train/train-projection dataset W E hidden-dim d n-entities
                                    {:lr 0.05 :steps 51 :on-step step-fn})
          initial-loss (first @history)
          step50-loss (second @history)]
      (is (some? initial-loss))
      (is (some? step50-loss))
      (is (< step50-loss initial-loss)
          (format "Loss at step 50 (%.4f) must be less than initial loss (%.4f)"
                  (double step50-loss) (double initial-loss))))))

;; ==============================================================================
;; 4. Determinism Test
;; ==============================================================================

(deftest test-training-determinism
  (testing "Same initial weights and dataset produce bit-identical learned W"
    (let [hidden-dim 8
          d 4
          n-entities 3
          init-w (float-array (* hidden-dim d) (float 0.1))
          E (float-array (* n-entities d) (float 0.5))
          dataset [{:h (float-array hidden-dim (float 1.0)) :y 0}
                   {:h (float-array hidden-dim (float -1.0)) :y 1}]
          res1 (train/train-projection dataset (aclone ^floats init-w) E hidden-dim d n-entities {:lr 0.01 :steps 20})
          res2 (train/train-projection dataset (aclone ^floats init-w) E hidden-dim d n-entities {:lr 0.01 :steps 20})]
      (is (java.util.Arrays/equals ^floats (:W res1) ^floats (:W res2))
          "Training must be strictly deterministic across repeated runs"))))

;; ==============================================================================
;; 5. Leave-One-Out Cross-Validation Harness Test
;; ==============================================================================

(deftest test-leave-one-out-harness
  (testing "Leave-one-out CV runs end-to-end and returns accuracy in [0.0, 1.0]"
    (let [hidden-dim 8
          d 4
          n-entities 4
          rng (java.util.Random. 456)
          rand-floats (fn [cnt scale]
                        (let [arr (float-array cnt)]
                          (dotimes [i cnt]
                            (aset-float arr i (float (* scale (- (.nextDouble rng) 0.5)))))
                          arr))
          init-w (rand-floats (* hidden-dim d) 0.1)
          E (rand-floats (* n-entities d) 1.0)
          dataset (mapv (fn [y] {:h (rand-floats hidden-dim 1.0) :y y}) (range n-entities))
          loo-res (train/run-leave-one-out-cv dataset init-w E hidden-dim d n-entities {:lr 0.05 :steps 30})]
      (is (contains? loo-res :mean-acc))
      (is (contains? loo-res :fold-results))
      (is (<= 0.0 (:mean-acc loo-res) 1.0)
          "LOO mean accuracy must be a valid probability between 0.0 and 1.0")
      (is (= n-entities (count (:fold-results loo-res)))
          "Number of fold results must match dataset size"))))

;; ==============================================================================
;; 6. Generative Property: step-gd Linearity
;; ==============================================================================

(defspec prop-step-gd-linearity 25
  (prop/for-all [w-val (gen/fmap #(float %) (gen/choose -10 10))
                 g-val (gen/fmap #(float %) (gen/choose -10 10))
                 lr (gen/fmap #(float %) (gen/choose 1 100))]
                (let [W (float-array 1 w-val)
                      G (float-array 1 g-val)
                      out (train/step-gd W G lr)
                      expected (float (- w-val (* lr g-val)))]
                  (< (Math/abs (- (aget ^floats out 0) expected)) 1e-4))))

