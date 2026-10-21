(ns einsum.test-helpers.generators-test
  "Generative property tests for SSA graph generators."
  (:require [einsum.test-helpers.generators :as g]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.properties :as prop]))

(defspec prop-generated-graphs-satisfy-schema
  100
  (prop/for-all [graph g/gen-valid-graph]
                (and (string? (:name graph))
                     (vector? (:invars graph))
                     (vector? (:outvars graph))
                     (vector? (:eqns graph))
                     (= [:out] (:outvars graph))
                     (= 3 (count (:eqns graph))))))
