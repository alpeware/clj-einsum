(ns clj-xla.logic.memory.camp-test
  "Unit, generative, and finite-difference gradient checks for Cross-Attention Memory Probe (CAMP).
   Rule 1: Strict TDD - invariants and property tests defined before core implementation."
  (:require [clj-xla.logic.memory.camp :as camp]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Invariants: Softmax Attention Properties
;; ==============================================================================

(defspec prop-camp-attention-weights-simplex 25
  (prop/for-all [l (gen/choose 2 16)
                 dim (gen/choose 4 16)]
                (let [h-arr (float-array (* l dim))
                      k-arr (float-array dim)
                      rng (java.util.Random. 42)]
                  (dotimes [i (* l dim)]
                    (aset-float h-arr i (float (- (.nextDouble rng) 0.5))))
                  (dotimes [i dim]
                    (aset-float k-arr i (float (- (.nextDouble rng) 0.5))))
                  (let [attn (camp/compute-attention-weights h-arr l dim k-arr (Math/sqrt (double dim)))
                        sum (loop [i 0 s 0.0]
                              (if (>= i l)
                                s
                                (recur (inc i) (+ s (double (aget ^floats attn i))))))
                        all-non-neg? (loop [i 0]
                                       (if (>= i l)
                                         true
                                         (if (neg? (aget ^floats attn i))
                                           false
                                           (recur (inc i)))))]
                    (and all-non-neg?
                         (< (Math/abs (- sum 1.0)) 1e-4))))))

;; ==============================================================================
;; 2. Finite-Difference Gradient Check for CAMP Parameters (k_attn and W)
;; ==============================================================================

(deftest test-camp-gradient-finite-difference-check
  (testing "Analytic gradients dL/dk and dL/dW match central finite differences"
    (let [hidden-dim 6
          d-mem 4
          n-entities 3
          seq-len 4
          rng (java.util.Random. 101)
          rand-floats (fn [cnt scale]
                        (let [arr (float-array cnt)]
                          (dotimes [i cnt]
                            (aset-float arr i (float (* scale (- (.nextDouble rng) 0.5)))))
                          arr))
          k-attn (rand-floats hidden-dim 0.5)
          W (rand-floats (* hidden-dim d-mem) 0.5)
          R (rand-floats (* d-mem d-mem) 0.5)
          E (rand-floats (* n-entities d-mem) 1.0)
          H1 (rand-floats (* seq-len hidden-dim) 1.0)
          H2 (rand-floats (* seq-len hidden-dim) 1.0)
          dataset [{:H H1 :seq-len seq-len :y 0}
                   {:H H2 :seq-len seq-len :y 2}]
          tau (Math/sqrt (double hidden-dim))
          params {:k-attn k-attn :W W}
          cfg {:hidden-dim hidden-dim :d-mem d-mem :n-entities n-entities :tau tau :R R :E E}
          {:keys [loss grad-k grad-W]} (camp/compute-loss-and-gradients dataset params cfg)
          eps 1e-4
          tol 1e-3]
      (is (pos? (double loss)) "Loss must be positive")

      ;; Check gradient w.r.t k_attn
      (dotimes [i hidden-dim]
        (let [orig-k (aget ^floats k-attn i)
              _ (aset-float k-attn i (float (+ orig-k eps)))
              lp (:loss (camp/compute-loss-and-gradients dataset {:k-attn k-attn :W W} cfg))
              _ (aset-float k-attn i (float (- orig-k eps)))
              lm (:loss (camp/compute-loss-and-gradients dataset {:k-attn k-attn :W W} cfg))
              _ (aset-float k-attn i orig-k)
              num-grad (/ (- (double lp) (double lm)) (* 2.0 eps))
              ana-grad (double (aget ^floats grad-k i))
              diff (Math/abs (- num-grad ana-grad))]
          (is (< diff tol)
              (format "k_attn grad mismatch at %d: ana=%.6f, num=%.6f, diff=%.6f"
                      i ana-grad num-grad diff))))

      ;; Check gradient w.r.t W
      (dotimes [i (* hidden-dim d-mem)]
        (let [orig-w (aget ^floats W i)
              _ (aset-float W i (float (+ orig-w eps)))
              lp (:loss (camp/compute-loss-and-gradients dataset {:k-attn k-attn :W W} cfg))
              _ (aset-float W i (float (- orig-w eps)))
              lm (:loss (camp/compute-loss-and-gradients dataset {:k-attn k-attn :W W} cfg))
              _ (aset-float W i orig-w)
              num-grad (/ (- (double lp) (double lm)) (* 2.0 eps))
              ana-grad (double (aget ^floats grad-W i))
              diff (Math/abs (- num-grad ana-grad))]
          (is (< diff tol)
              (format "W grad mismatch at %d: ana=%.6f, num=%.6f, diff=%.6f"
                      i ana-grad num-grad diff)))))))

;; ==============================================================================
;; 3. Optimization Smoke Test: Loss Decreases Under Gradient Steps
;; ==============================================================================

(deftest test-camp-optimization-smoke-loss-decreases
  (testing "Loss strictly decreases after 20 gradient descent steps on synthetic dataset"
    (let [hidden-dim 8
          d-mem 4
          n-entities 3
          seq-len 5
          rng (java.util.Random. 202)
          rand-floats (fn [cnt scale]
                        (let [arr (float-array cnt)]
                          (dotimes [i cnt]
                            (aset-float arr i (float (* scale (- (.nextDouble rng) 0.5)))))
                          arr))
          k-attn (rand-floats hidden-dim 0.2)
          W (rand-floats (* hidden-dim d-mem) 0.2)
          R (rand-floats (* d-mem d-mem) 0.5)
          E (rand-floats (* n-entities d-mem) 1.0)
          dataset [{:H (rand-floats (* seq-len hidden-dim) 1.0) :seq-len seq-len :y 0}
                   {:H (rand-floats (* seq-len hidden-dim) 1.0) :seq-len seq-len :y 1}
                   {:H (rand-floats (* seq-len hidden-dim) 1.0) :seq-len seq-len :y 2}]
          cfg {:hidden-dim hidden-dim :d-mem d-mem :n-entities n-entities :tau 2.0 :R R :E E}
          train-res (camp/train-camp dataset {:k-attn k-attn :W W} cfg {:lr 0.05 :steps 25})
          init-loss (:initial-loss train-res)
          final-loss (:final-loss train-res)]
      (is (< (double final-loss) (double init-loss))
          (format "Final loss (%.4f) must be less than initial loss (%.4f)"
                  (double final-loss) (double init-loss))))))

;; ==============================================================================
;; 4. Declarative Tensor Logic AST Invariants
;; ==============================================================================

(deftest test-camp-ast-structure
  (testing "camp-grounding-ast produces a valid Tensor Logic block with matching bindings"
    (let [ast (camp/camp-grounding-ast 1536 256 14 {:sequence? true})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :camp_grounding_seq (get-in ast [1 :name])))
      (let [op-names (mapv first (subvec ast 2))]
        (is (some #{:=} op-names))
        (is (some #{:softmax} op-names))
        (is (some #{:rms-norm} op-names))))))

;; ==============================================================================
;; 5. Sequence Normalization & Attention Masking Invariants
;; ==============================================================================

(defspec prop-normalize-sequence-h-unit-rows 25
  (prop/for-all [l (gen/choose 2 10)
                 dim (gen/choose 4 16)]
                (let [h-arr (float-array (* l dim))
                      rng (java.util.Random. 55)]
                  (dotimes [i (* l dim)]
                    (aset-float h-arr i (float (+ 0.1 (- (.nextDouble rng) 0.5)))))
                  (let [normed (camp/normalize-sequence-h h-arr l dim)]
                    (every? (fn [t]
                              (let [sum-sq (loop [i 0 s 0.0]
                                             (if (>= i dim)
                                               s
                                               (let [val (double (aget ^floats normed (+ (* t dim) i)))]
                                                 (recur (inc i) (+ s (* val val))))))]
                                (< (Math/abs (- (Math/sqrt sum-sq) 1.0)) 1e-4)))
                            (range l))))))

(deftest test-camp-masked-attention
  (testing "Masked attention zeroes out disabled token positions"
    (let [l 4
          dim 4
          h-arr (float-array (* l dim) (float 1.0))
          k-arr (float-array dim (float 1.0))
          mask [false true true false]
          attn (camp/compute-attention-weights h-arr l dim k-arr 1.0 mask)]
      (is (zero? (aget ^floats attn 0)) "Token 0 masked out must have 0 attention")
      (is (zero? (aget ^floats attn 3)) "Token 3 masked out must have 0 attention")
      (is (< (Math/abs (- (+ (double (aget ^floats attn 1)) (double (aget ^floats attn 2))) 1.0)) 1e-4)
          "Unmasked tokens 1 and 2 must sum to 1.0"))))

