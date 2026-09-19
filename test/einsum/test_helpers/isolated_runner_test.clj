(ns einsum.test-helpers.isolated-runner-test
  "Unit & property tests for process-isolated test runner harness."
  (:require [einsum.test-helpers.isolated-runner :as runner]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-isolated-runner-execution
  (testing "Running pure test namespace in isolated JVM subprocess"
    (let [res (runner/run-isolated-test 'einsum.compiler.pjrt-version-test {})]
      (is (map? res))
      (is (= 0 (:exit res)))
      (is (= :pass (:status res)))
      (is (some? (:summary res)))
      (is (pos? (get-in res [:summary :pass] 0))))))

(defspec prop-isolated-suite-aggregation-invariants
  10
  (prop/for-all [_dummy-flag gen/boolean]
                (let [suite-res (runner/run-isolated-suite ['einsum.compiler.pjrt-version-test] {})]
                  (and (map? suite-res)
                       (integer? (:total-namespaces suite-res))
                       (integer? (:passed-namespaces suite-res))
                       (vector? (:details suite-res))
                       (= 1 (:total-namespaces suite-res))
                       (= 1 (:passed-namespaces suite-res))))))
