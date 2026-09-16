(ns clj-xla.logic.memory.factorization-test
  "Unit and generative tests for Experiment E8: Dynamic In-VRAM Relation Induction.
   Compiles non-negative tensor factorization (PARAFAC/NMF) into StableHLO MLIR via OpenXLA PJRT.
   Rule 1: Strict TDD - generative tests for invariants defined before core implementation."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.memory.factorization :as fact]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property: Non-Negativity Conservation Invariant
;; ==============================================================================

(defspec prop-factorization-non-negativity-conservation 20
  (prop/for-all [n (gen/choose 4 8)
                 k (gen/choose 2 4)
                 r (gen/choose 2 3)]
                (let [ctx (xla/get-context)
                      session (fact/init-factorization-session ctx {:num-entities n
                                                                    :num-contexts k
                                                                    :rank r})
          ;; Generate non-negative synthetic observation tensor
                      rnd (java.util.Random. 42)
                      x-data (float-array (* n k n))
                      _ (dotimes [i (* n k n)]
                          (aset-float x-data i (float (Math/abs (.nextGaussian rnd)))))
                      step-res (fact/step-factorization! session x-data)
                      ^floats a-next (:A step-res)
                      ^floats b-next (:B step-res)
                      ^floats c-next (:C step-res)]
                  (and
       ;; All elements in A, B, C must remain strictly non-negative
                   (every? #(>= % 0.0) a-next)
                   (every? #(>= % 0.0) b-next)
                   (every? #(>= % 0.0) c-next)))))

;; ==============================================================================
;; 2. Generative Property: Monotonic Reconstruction Descent
;; ==============================================================================

(defspec prop-factorization-loss-descent 15
  (prop/for-all [steps (gen/choose 5 15)]
                (let [ctx (xla/get-context)
                      n 8
                      k 3
                      r 2
                      session (fact/init-factorization-session ctx {:num-entities n
                                                                    :num-contexts k
                                                                    :rank r})
          ;; Synthesize low-rank observation tensor
                      obs (fact/generate-synthetic-observation-tensor n k r 123)
                      errors (atom [])]
                  (dotimes [_ steps]
                    (let [res (fact/step-factorization! session (:x-data obs))]
                      (swap! errors conj (:rel-error res))))
                  (let [initial-err (first @errors)
                        final-err (last @errors)]
        ;; Optimization must reduce relative reconstruction error
                    (< final-err initial-err)))))

;; ==============================================================================
;; 3. Unit Test: Ground-Truth Relation Recovery (> 85% Fidelity)
;; ==============================================================================

(deftest test-relation-induction-fidelity
  (testing "In-VRAM tensor factorization achieves > 85% reconstruction fidelity on 3-way relation observations"
    (let [ctx (xla/get-context)
          n 12
          k 4
          r 3
          session (fact/init-factorization-session ctx {:num-entities n
                                                        :num-contexts k
                                                        :rank r})
          obs (fact/generate-synthetic-observation-tensor n k r 777)
          final-res (fact/fit-factorization! session (:x-data obs) 25)
          fidelity (:fidelity final-res)]
      (is (>= fidelity 0.85)
          (format "Reconstruction fidelity (%.4f) must exceed target threshold (0.85)" fidelity)))))

;; ==============================================================================
;; 4. Unit Test: Relational Core Extraction & Fast-Weight Superposition
;; ==============================================================================

(deftest test-induced-relational-core-extraction
  (testing "Induced factors can be extracted as crisp relational core matrices"
    (let [ctx (xla/get-context)
          n 8
          k 2
          r 2
          session (fact/init-factorization-session ctx {:num-entities n
                                                        :num-contexts k
                                                        :rank r})
          obs (fact/generate-synthetic-observation-tensor n k r 999)
          fit-res (fact/fit-factorization! session (:x-data obs) 20)
          cores (fact/extract-induced-relational-cores session fit-res)]
      (is (= r (count cores)) "Must extract R relational core matrices")
      (doseq [core cores]
        (is (= (* n n) (count (:matrix core))) "Each core must be an N x N matrix")
        (is (pos? (:energy core)) "Core must have positive energy")))))
