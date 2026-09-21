(ns einsum.test-helpers.isolated-runner-test
  "Unit & property tests for process-isolated test runner harness."
  (:require [einsum.test-helpers.isolated-runner :as runner]
            [clojure.test :refer [deftest is testing]]))

(deftest test-isolated-runner-execution
  (testing "Running pure test namespace in isolated JVM subprocess"
    (let [res (runner/run-isolated-test 'einsum.compiler.pjrt-version-test {})]
      (is (map? res))
      (is (= 0 (:exit res)))
      (is (= :pass (:status res)))
      (is (some? (:summary res)))
      (is (pos? (get-in res [:summary :pass] 0))))))

(deftest test-isolated-suite-aggregation-invariants
  (testing "Aggregating isolated test suite execution results"
    (let [suite-res (runner/run-isolated-suite ['einsum.compiler.pjrt-version-test] {})]
      (is (map? suite-res))
      (is (integer? (:total-namespaces suite-res)))
      (is (integer? (:passed-namespaces suite-res)))
      (is (vector? (:details suite-res)))
      (is (= 1 (:total-namespaces suite-res)))
      (is (= 1 (:passed-namespaces suite-res))))))
