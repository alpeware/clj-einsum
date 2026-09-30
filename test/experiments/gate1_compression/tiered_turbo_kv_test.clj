(ns experiments.gate1-compression.tiered-turbo-kv-test
  "Pre-registered generative property and invariant tests for Tiered Turbo KV
   across the 4 RSI gates on 128k context lengths (RFC tiered-turbo-kv)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.quant.eviction :as eviction]
            [experiments.gate1-compression.tiered-turbo-kv.core :as turbo-kv]))

;; =============================================================================
;; Gate 1: Resource Efficiency Invariants (Criteria 1.1, 1.2, 1.3)
;; =============================================================================

(defspec prop-gate1-kv-compression-ratio
  50
  (prop/for-all [model (gen/elements [:gemma-4-e4b :gemma-4-12b :gemma-4-31b])
                 seq-len (gen/choose 32768 131072)]
                (let [accounting (turbo-kv/compute-kv-cache-accounting model seq-len)
                      ratio (:compression-ratio accounting)]
                  (>= ratio 16.0))))

(defspec prop-gate1-31b-peak-vram-bound
  50
  (prop/for-all [seq-len (gen/choose 32768 131072)]
                (let [accounting (turbo-kv/compute-kv-cache-accounting :gemma-4-31b seq-len)
                      peak-vram-gb (:peak-vram-gb accounting)]
                  (<= peak-vram-gb 19.5))))

(deftest test-gate1-effective-bitrate-invariant
  (testing "Effective KV bitrate is strictly <= 3.0 bits/element (Criterion 1.3)"
    (let [bitrate (turbo-kv/effective-bitrate-per-element 128 32)]
      (is (<= bitrate 3.0))
      (is (pos? bitrate)))))

;; =============================================================================
;; Gate 2: Time Efficiency Invariants (Criteria 2.1, 2.2, 2.3, 2.4)
;; =============================================================================

(deftest test-gate2-eviction-latency-ceiling
  (testing "Attention sink and heavy-hitter selection on 128k sequence length runs <= 10 ms (Criterion 2.3)"
    (let [n 131072
          weights (float-array n)
          _ (dotimes [i n] (aset weights i (float (rand))))
          opts {:k-sink 4 :window 1024 :k-base 1024}
          ;; Warm up
          _ (dotimes [_ 2] (eviction/select-retained-indices 0 54 n weights opts))
          times (mapv (fn [_]
                        (let [t0 (System/nanoTime)
                              _ (eviction/select-retained-indices 0 54 n weights opts)
                              t1 (System/nanoTime)]
                          (/ (- t1 t0) 1e6)))
                      (range 5))
          median-ms (nth (sort times) 2)]
      (is (<= median-ms 10.0) (str "Median selection time: " median-ms " ms exceeds ceiling (samples: " times ")")))))

(deftest test-evict-kv-cache-buffers-slicing
  (testing "evict-kv-cache-buffers correctly extracts retained sequence tokens"
    (let [tokens (vec (map #(str "tok-" %) (range 10)))
          retained [0 1 8 9]
          res (eviction/evict-kv-cache-buffers retained tokens)]
      (is (= retained (:retained-indices res)))
      (is (= 4 (:retained-count res)))
      (is (= ["tok-0" "tok-1" "tok-8" "tok-9"] (:kv-buffers res))))))

(deftest test-gate2-prefix-hit-rate
  (testing "CliffCompaction common prefix retention across turns >= 85% (Criterion 2.4)"
    (let [hit-rate (turbo-kv/measure-cliffcompaction-prefix-hit-rate)]
      (is (>= hit-rate 0.85)))))

;; =============================================================================
;; Gate 3: Intelligence Floor Invariants (Criteria 3.1, 3.2, 3.3)
;; =============================================================================

(deftest test-gate3-qjl-inner-product-bias
  (testing "QJL residual estimator expectation bias over 500 samples <= 1.0e-2 (Criterion 3.1 unit check)"
    (let [bias (turbo-kv/evaluate-qjl-estimator-bias 500 128 32 42)]
      (is (<= (Math/abs bias) 1.0e-2)))))

(deftest test-gate3-m-niah-retrieval-floor
  (testing "Multi-needle retrieval accuracy at 64k and 128k context lengths >= 95.0% (Criterion 3.2)"
    (let [results (turbo-kv/evaluate-synthetic-m-niah 131072 [0.1 0.25 0.5 0.75 0.9])]
      (is (>= (:retrieval-accuracy results) 0.95)))))

(deftest test-gate3-multipl-e-non-regression
  (testing "MultiPL-E Clojure dev 50 verification floor >= 95% with zero regression (Criterion 3.3)"
    (let [res (turbo-kv/evaluate-multipl-e-dev50)]
      (is (>= (:pass-rate res) 0.95))
      (is (false? (get-in res [:mcnemar :significant?]))))))
