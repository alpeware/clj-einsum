(ns einsum.runtime.benchmark.workloads-test
  "Unit & generative property tests for benchmark workload definitions and graph tracing."
  (:require [einsum.runtime.benchmark.workloads :as bw]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-workload-registry-integrity
  (testing "WORKLOADS registry contains valid standard kernel definitions"
    (is (map? bw/WORKLOADS))
    (is (pos? (count bw/WORKLOADS)))
    (doseq [[id wl] bw/WORKLOADS]
      (is (keyword? id))
      (is (string? (:name wl)))
      (is (keyword? (:category wl)))
      (is (fn? (:build-graph-fn wl)))
      (is (fn? (:make-inputs-fn wl))))))

(defspec prop-workload-graphs-tracing 10
  (prop/for-all [workload-key (gen/elements (vec (keys bw/WORKLOADS)))]
                (let [wl (get bw/WORKLOADS workload-key)
                      build-fn (:build-graph-fn wl)
                      graph (build-fn)]
                  (and (map? graph)
                       (string? (:name graph))
                       (vector? (:invars graph))
                       (vector? (:outvars graph))
                       (seq (:eqns graph))))))
