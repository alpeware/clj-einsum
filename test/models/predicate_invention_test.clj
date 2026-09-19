(ns models.predicate-invention-test
  "TDD tests for gradient-based predicate invention model (Experiment E16/E17).
   Enforces strict compilation, execution, probability, and metric invariants (Rule 1)."
  (:require [einsum.core :as xla]
            [einsum.logic.data.family-trees :as ft]
            [models.predicate-invention :as pi]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Discrete Metric Property Tests (Rule 1 & Rule 2)
;; ==============================================================================

(defspec prop-discrete-f1-invariants 20
  (prop/for-all [n (gen/elements [16 32])
                 seed (gen/choose 1 10000)]
                (let [forest (ft/generate-family-forest {:num-entities n :num-trees 2 :seed seed})
                      p-mat (:parent-matrix forest)
                      flat-true (float-array (flatten p-mat))
                      zero-mat (float-array (* n n))
          ;; Perfect recovery
                      f1-perfect (pi/compute-f1-metrics flat-true flat-true)
          ;; Zero recovery
                      f1-zero (pi/compute-f1-metrics zero-mat flat-true)]
                  (and (= 1.0 (:f1 f1-perfect))
                       (= 1.0 (:precision f1-perfect))
                       (= 1.0 (:recall f1-perfect))
                       (= 0.0 (:f1 f1-zero))
                       (= 0.0 (:precision f1-zero))
                       (= 0.0 (:recall f1-zero))))))

(defspec prop-two-hop-ground-truth-closure 20
  (prop/for-all [seed (gen/choose 1 10000)
                 n (gen/elements [32 64])]
                (let [forest (ft/generate-family-forest {:num-entities n :num-trees 4 :seed seed})
                      p-mat (:parent-matrix forest)
                      gp-pairs (:grandparent-pairs forest)
                      closure (pi/compute-two-hop-closure (float-array (flatten p-mat)) gp-pairs n)]
                  (= 1.0 closure))))

;; ==============================================================================
;; 2. OpenXLA PJRT Compilation & Execution Unit Tests
;; ==============================================================================

(deftest test-predicate-invention-compilation
  (testing "Compiles OpenXLA PJRT executables for all 4 ablations (a0, a1, a2, a3)"
    (xla/init-backend! :cpu)
    (let [ctx (xla/get-context)
          n 16
          dim 16
          opts {:num-entities n :dim dim :rank 4 :tau-ce 0.1 :lr 0.01 :lambda-l1 1e-4}]
      (doseq [ablation [:a0 :a1 :a2 :a3]]
        (let [exec (pi/compile-predicate-invention-step ctx (assoc opts :ablation ablation))]
          (is (some? exec) (str "Compilation failed for ablation " ablation))
          (is (some? (:handle exec)) (str "Missing PJRT handle for ablation " ablation)))))))

(deftest test-predicate-invention-step-execution
  (testing "Executes a single in-VRAM step for A2 and verifies probability row sums"
    (xla/init-backend! :cpu)
    (let [ctx (xla/get-context)
          n 16
          dim 16
          forest (ft/generate-family-forest {:num-entities n :num-trees 2 :seed 42})
          exec (pi/compile-predicate-invention-step ctx {:num-entities n :dim dim :ablation :a2})
          state (pi/init-parameters forest {:dim dim :seed 42})
          batch (pi/make-step-batch forest state {:tau 1.0 :lambda-l1 1e-4})
          step-res (pi/run-predicate-invention-step! exec batch)]
      (is (some? step-res))
      (is (contains? step-res :P_gp))
      (is (contains? step-res :P_sib))
      (is (contains? step-res :Z_new))
      (let [p-gp ^floats (:P_gp step-res)]
        (dotimes [i n]
          (let [row-sum (reduce + (map #(aget p-gp (+ (* i n) %)) (range n)))]
            (is (< (Math/abs (- (double row-sum) 1.0)) 1e-4)
                (str "P_gp row " i " does not sum to 1.0: " row-sum))))))))
