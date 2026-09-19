(ns einsum.compiler.stablehlo-test
  "Unit and generative property tests for StableHLO EDN graph schemas and MLIR printer."
  (:require [einsum.compiler.stablehlo :as shlo]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; -----------------------------------------------------------------------------
;; Generators for Generative Testing (clojure.test.check)
;; -----------------------------------------------------------------------------

(def gen-dtype
  "Generator for valid tensor element data types."
  (gen/elements [:f16 :f32 :f64 :bf16 :i8 :i16 :i32 :i64]))

(def gen-tensor-shape
  "Generator for valid tensor shape vectors."
  (gen/vector gen/nat 1 3))

(def gen-valid-graph
  "Generator for valid EDN SSA graph data maps."
  (gen/fmap (fn [[name-suffix dims dtype c-val]]
              (let [gname (str "gen_graph_" name-suffix)
                    t-type [:tensor dims dtype]]
                {:name gname
                 :invars [[:x t-type]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value c-val :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}))
            (gen/tuple gen/nat (gen/vector (gen/choose 1 64) 2) gen-dtype gen/double)))

;; -----------------------------------------------------------------------------
;; Generative Property Tests
;; -----------------------------------------------------------------------------

(defspec prop-valid-graph-validation 50
  (prop/for-all [graph gen-valid-graph]
                (let [validated (shlo/validate-graph graph)]
                  (= validated graph))))

(defspec prop-mlir-text-serialization-invariants 50
  (prop/for-all [graph gen-valid-graph]
                (let [mlir (shlo/graph->mlir-text graph)]
                  (and (string? mlir)
                       (str/includes? mlir (str "module @" (:name graph)))
                       (str/includes? mlir "func.func @main")
                       (str/includes? mlir "return")
                       (str/includes? mlir "stablehlo.multiply")))))

;; -----------------------------------------------------------------------------
;; Unit Tests
;; -----------------------------------------------------------------------------

(deftest type->mlir-string-test
  (testing "Tensor type serialization to MLIR string"
    (is (= "tensor<1x128x768xf32>" (shlo/type->mlir-string [:tensor [1 128 768] :f32])))
    (is (= "tensor<10xf16>" (shlo/type->mlir-string [:tensor [10] :f16])))
    (is (= "tensor<f32>" (shlo/type->mlir-string [:tensor [] :f32])))))

(deftest invalid-graph-schema-test
  (testing "Validation failure on invalid EDN graph"
    (is (thrown? Exception (shlo/validate-graph {:invalid true})))
    (is (thrown? Exception (shlo/validate-graph {:name "bad" :invars [] :outvars [] :eqns "not-a-vector"})))))

(deftest test-while-equation-serialization
  (testing "Serialization of :stablehlo/while into MLIR while block"
    (let [graph {:name "while_test"
                 :invars [[:init_i [:tensor [1] :i32]]
                          [:init_sum [:tensor [1] :i32]]]
                 :outvars [:final_i :final_sum]
                 :eqns [{:op :stablehlo/while
                         :invars [:init_i :init_sum]
                         :outvars [:final_i :final_sum]
                         :attrs {:cond-mlir "      %cond = stablehlo.constant dense<true> : tensor<i1>\n      \"stablehlo.return\"(%cond) : (tensor<i1>) -> ()"
                                 :body-mlir "      \"stablehlo.return\"(%init_i, %init_sum) : (tensor<1xi32>, tensor<1xi32>) -> ()"}}]}
          mlir (shlo/graph->mlir-text graph)]
      (is (str/includes? mlir "\"stablehlo.while\""))
      (is (str/includes? mlir "final_i = stablehlo.reshape"))
      (is (str/includes? mlir "final_sum = stablehlo.reshape")))))

(deftest test-reduce-min-serialization
  (testing "Serialization and validation of :stablehlo/reduce_min"
    (let [graph {:name "red_min_test"
                 :invars [[:x [:tensor [4 8] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/reduce_min
                         :invars [:x]
                         :outvars [:y]
                         :attrs {:axes [1] :keep_dims false}}]}
          _ (is (shlo/validate-graph graph))
          mlir (shlo/graph->mlir-text graph)]
      (is (str/includes? mlir "\"stablehlo.reduce\""))
      (is (str/includes? mlir "stablehlo.minimum"))
      (is (str/includes? mlir "dimensions = array<i64: 1>")))))
