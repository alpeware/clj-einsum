(ns einsum.runtime.benchmark.core-test
  "Unit & generative property tests for pure benchmark calculations, percentile metrics, and Markdown report rendering."
  (:require [einsum.runtime.benchmark.core :as bcore]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-percentiles-invariants 50
  (prop/for-all [latencies (gen/not-empty (gen/vector (gen/double* {:min 0.001 :max 1000.0 :NaN? false :infinite? false})))]
                (let [stats (bcore/calculate-latency-stats latencies)
                      {:keys [mean p50 p90 p99 min-lat max-lat]} stats]
                  (and (number? mean) (pos? mean)
                       (number? p50) (pos? p50)
                       (number? p90) (pos? p90)
                       (number? p99) (pos? p99)
                       (<= min-lat p50 max-lat)
                       (<= p50 p90 p99)))))

(defspec prop-gemm-flops-calculation 50
  (prop/for-all [m (gen/choose 1 4096)
                 n (gen/choose 1 4096)
                 k (gen/choose 1 4096)]
                (let [flops (bcore/gemm-flops m n k)]
                  (= (* 2 m n k) flops))))

(deftest test-gflops-tflops-calc
  (testing "GFLOPS and TFLOPS calculation from FLOPs and latency in ms"
    (let [flops (* 2 1024 1024 1024) ;; ~2.147 billion FLOPs
          lat-ms 1.0 ;; 1 ms
          gflops (bcore/calculate-gflops flops lat-ms)
          tflops (bcore/calculate-tflops flops lat-ms)]
      (is (> gflops 2000.0))
      (is (> tflops 2.0))
      (is (< (Math/abs (- gflops (* tflops 1000.0))) 1e-6)))))

(deftest test-bandwidth-gbps-calc
  (testing "Effective memory bandwidth in GB/s"
    (let [bytes (* 1024 1024 4) ;; 4 MB
          lat-ms 1.0 ;; 1 ms -> ~4.194 GB/s
          gbps (bcore/calculate-gbps bytes lat-ms)]
      (is (Double/isFinite gbps))
      (is (< (Math/abs (- gbps 4.194304)) 1e-4)))))

(deftest test-markdown-table-rendering
  (testing "Rendering Markdown comparative benchmark table"
    (let [bench-data [{:kernel "gemm-fp32" :backend :sycl :tflops 12.5 :mean-ms 0.17 :p99-ms 0.21}
                      {:kernel "gemm-fp32" :backend :cpu :tflops 1.2 :mean-ms 1.78 :p99-ms 2.05}]
          table-str (bcore/render-markdown-table bench-data)]
      (is (string? table-str))
      (is (str/includes? table-str "gemm-fp32"))
      (is (str/includes? table-str "sycl"))
      (is (str/includes? table-str "cpu")))))
