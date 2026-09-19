(ns einsum.runtime.profile-test
  "Unit and generative tests for einsum.runtime.profile telemetry, profiling spans, and Chrome trace export."
  (:require [einsum.runtime.profile :as profile]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-profile-span-execution
  (testing "Measuring trace and execution spans"
    (let [metrics (atom {})
          result (profile/with-profile metrics "gemma4/prefill"
                   (Thread/sleep 10)
                   42)]
      (is (= 42 result))
      (is (contains? @metrics "gemma4/prefill"))
      (is (pos? (get-in @metrics ["gemma4/prefill" :duration-ms]))))))

(deftest test-chrome-trace-export
  (testing "Exporting profiling spans to Chrome Trace JSON format"
    (let [spans [{"name" "prefill" "cat" "gpu" "ph" "X" "ts" 1000 "dur" 5000 "pid" 1 "tid" 1}
                 {"name" "decode_step_1" "cat" "gpu" "ph" "X" "ts" 6000 "dur" 1000 "pid" 1 "tid" 1}]
          json-str (profile/export-chrome-trace spans)]
      (is (string? json-str))
      (is (str/includes? json-str "prefill"))
      (is (str/includes? json-str "decode_step_1")))))

(defspec prop-profile-summary-percentiles-invariant
  50
  (prop/for-all [latencies (gen/vector (gen/double* {:min 1.0 :max 100.0 :NaN? false}) 10 50)]
                (let [summary (profile/compute-latency-summary latencies)]
                  (and (number? (:mean summary))
                       (number? (:p50 summary))
                       (number? (:p95 summary))
                       (number? (:p99 summary))
                       (<= (:min summary) (:p50 summary))
                       (<= (:p50 summary) (:p95 summary))
                       (<= (:p95 summary) (:max summary))))))
