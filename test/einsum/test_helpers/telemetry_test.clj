(ns einsum.test-helpers.telemetry-test
  "Generative unit tests for system telemetry collection and EDN report generation."
  (:require [einsum.test-helpers.telemetry :as t]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-system-telemetry-collection
  (testing "Telemetry collection returns non-nil structured map with host and driver details"
    (let [telem (t/collect-system-telemetry)]
      (is (map? telem))
      (is (contains? telem :host))
      (is (contains? telem :timestamp))
      (is (contains? telem :backends)))))

(defspec prop-edn-report-serialization-invariants
  20
  (prop/for-all [_dummy gen/boolean]
                (let [out-file (io/file "target/test-reports/test_telemetry.edn")
                      dummy-results {:total-namespaces 1 :passed-namespaces 1 :crashed-namespaces 0 :details []}
                      report (t/generate-edn-report dummy-results out-file)]
                  (and (map? report)
                       (.exists out-file)
                       (map? (edn/read-string (slurp out-file)))))))
