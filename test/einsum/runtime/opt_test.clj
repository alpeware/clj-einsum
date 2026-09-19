(ns einsum.runtime.opt-test
  "Unit and generative property tests for einsum.runtime.opt graph optimization passes."
  (:require [einsum.runtime.opt :as opt]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-dce-preserves-validity 50
  (prop/for-all [c-val gen/double]
                (let [graph {:name "dead_code_graph"
                             :invars [[:x [:tensor [1 10] :f32]]]
                             :outvars [:y]
                             :eqns [{:op :stablehlo/constant :value c-val :outvars [:dead_c]}
                                    {:op :stablehlo/multiply :invars [:x :dead_c] :outvars [:dead_t]}
                                    {:op :stablehlo/constant :value 2.0 :outvars [:c0]}
                                    {:op :stablehlo/add :invars [:x :c0] :outvars [:y]}]}
                      opt-graph (opt/dce graph)]
                  (and (some? (shlo/validate-graph opt-graph))
                       (not (some #(= (:outvars %) [:dead_t]) (:eqns opt-graph)))))))

(deftest dce-pruning-test
  (testing "Dead Code Elimination removes unused subgraphs not connected to outvars"
    (let [graph {:name "unused_nodes"
                 :invars [[:x [:tensor [1 5] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 10.0 :outvars [:unused_c]}
                        {:op :stablehlo/constant :value 2.0 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          optimized (opt/dce graph)]
      (is (= 2 (count (:eqns optimized))))
      (is (not (some #(= (:outvars %) [:unused_c]) (:eqns optimized)))))))
