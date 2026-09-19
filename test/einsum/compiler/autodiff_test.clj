(ns einsum.compiler.autodiff-test
  "Unit and generative property tests for einsum.compiler.autodiff reverse-mode VJP generation."
  (:require [einsum.compiler.autodiff :as autodiff]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-vjp-graph-validity 50
  (prop/for-all [_c-val gen/double]
                (let [f-graph {:name "fwd_add"
                               :invars [[:x [:tensor [1 10] :f32]] [:w [:tensor [1 10] :f32]]]
                               :outvars [:y]
                               :eqns [{:op :stablehlo/multiply :invars [:x :w] :outvars [:y]}]}
                      vjp-graph (autodiff/vjp f-graph)]
                  (some? (shlo/validate-graph vjp-graph)))))

(deftest cotangent-accumulation-test
  (testing "VJP cotangent accumulation for multi-use intermediate variable (y = x * x)"
    (let [f-graph {:name "square"
                   :invars [[:x [:tensor [1 10] :f32]]]
                   :outvars [:y]
                   :eqns [{:op :stablehlo/multiply :invars [:x :x] :outvars [:y]}]}
          vjp-graph (autodiff/vjp f-graph)
          eqns (:eqns vjp-graph)
          add-eqns (filter #(= (:op %) :stablehlo/add) eqns)]
      (is (seq add-eqns) "Must contain a cotangent accumulation add equation for multi-use variable x"))))
