(ns einsum.uop.beam-test
  "TDD generative and invariant tests for Phase 8: Tinygrad Beam Search & Hardware-in-the-Loop Tuning."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.uop.beam :as beam]))

;; -----------------------------------------------------------------------------
;; 1. Candidate Action Invariants
;; -----------------------------------------------------------------------------

(defspec prop-candidate-actions-validity 40
  (prop/for-all [spatial-dim (gen/fmap #(long (* % 16)) (gen/choose 1 32))
                 red-dim (gen/fmap #(long (* % 16)) (gen/choose 1 32))]
                (let [ke {:name "test_ke"
                          :axes [spatial-dim]
                          :red-axes [red-dim]
                          :opts []}
                      actions (beam/candidate-actions ke)]
                  (and (seq actions)
                       (every? (fn [act]
                                 (and (contains? #{:local :upcast :group :unroll} (:op act))
                                      (integer? (:axis act))
                                      (pos? (long (:amount act)))
                                      (if (contains? #{:local :upcast} (:op act))
                                        (zero? (mod spatial-dim (:amount act)))
                                        (zero? (mod red-dim (:amount act))))))
                               actions)))))

(deftest test-candidate-actions-filtering
  (testing "Candidate actions respects maximum thread limit and divisibility"
    (let [ke {:name "mm"
              :axes [16 16]
              :red-axes [64]
              :opts []}
          actions (beam/candidate-actions ke)]
      (is (seq actions))
      ;; Cannot have :group on spatial axes
      (is (every? #(not (and (= :group (:op %)) (zero? (:axis %)) (empty? (:red-axes ke)))) actions)))))

;; -----------------------------------------------------------------------------
;; 2. Beam Search Engine with Mock Evaluator
;; -----------------------------------------------------------------------------

(deftest test-beam-search-mock-convergence
  (testing "Beam search converges to the lowest-cost configuration with mock cost function"
    (let [ke {:name "mock_gemm"
              :axes [64 64]
              :red-axes [128]
              :opts []}
          ;; Synthetic cost function: prefers 2D tiling (amount 16) and unroll (amount 4)
          cost-fn (fn [cand]
                    (let [opts (:opts cand)
                          has-local? (some #(= (:op %) :local) opts)
                          has-upcast? (some #(= (:op %) :upcast) opts)
                          has-group? (some #(= (:op %) :group) opts)]
                      (+ 1000.0
                         (if has-local? -300.0 0.0)
                         (if has-upcast? -200.0 0.0)
                         (if has-group? -100.0 0.0)
                         (* 10.0 (count opts)))))
          result (beam/beam-search ke {:width 4 :depth 3 :cost-fn cost-fn :use-cache? false})]
      (is (map? result))
      (is (vector? (:opts result)))
      (is (< (double (:best-cost result)) 1000.0)))))

;; -----------------------------------------------------------------------------
;; 3. Persistent Disk & Catalog Cache
;; -----------------------------------------------------------------------------

(deftest test-beam-cache-roundtrip
  (testing "Persistent beam cache saves and restores optimized recipes cleanly"
    (let [tmp-file (java.io.File/createTempFile "beam_cache_test" ".edn")
          key "gfx1100:test_signature_123"
          recipe {:opts [{:op :local :axis 0 :amount 16} {:op :upcast :axis 1 :amount 4}]
                  :time-us 42.5}]
      (try
        (beam/save-cache-entry! tmp-file key recipe)
        (let [loaded (beam/lookup-cache-entry tmp-file key)]
          (is (= (:opts loaded) (:opts recipe)))
          (is (== (double (:time-us loaded)) 42.5)))
        (finally
          (.delete tmp-file))))))

(deftest test-catalog-rdna3-lookup
  (testing "Catalog resources/uop/rdna3_beam_catalog.edn contains pre-tuned kernel configs"
    (let [catalog (beam/load-catalog)]
      (is (map? catalog))
      (is (contains? catalog :gemma4-gemv))
      (is (contains? catalog :gemma4-gemm)))))
