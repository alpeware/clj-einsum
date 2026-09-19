(ns einsum.test-helpers.generators-test
  "Generative property tests for SSA graph generators."
  (:require [einsum.compiler.stablehlo :as shlo]
            [einsum.test-helpers.generators :as g]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.properties :as prop]))

(defspec prop-generated-graphs-satisfy-malli-schema
  100
  (prop/for-all [graph g/gen-valid-graph]
                (let [val-res (shlo/validate-graph graph)]
                  (= graph val-res))))

(defspec prop-generated-graphs-produce-valid-mlir-text
  50
  (prop/for-all [graph g/gen-valid-graph]
                (let [mlir (shlo/graph->mlir-text graph)]
                  (and (string? mlir)
                       (re-find #"module @\w+" mlir)
                       (re-find #"func\.func @main" mlir)))))
