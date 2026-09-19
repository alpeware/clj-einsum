(ns einsum.test-helpers.generators
  "Generative test.check generators for valid EDN SSA graphs, tensors, and mathematical ops."
  (:require [clojure.test.check.generators :as gen]))

(def gen-dtype
  "Generator for supported tensor element dtypes."
  (gen/elements [:f32 :i32 :f16 :bf16]))

(def gen-dim
  "Generator for tensor dimension size."
  (gen/choose 1 32))

(def gen-shape
  "Generator for 1D to 4D tensor shapes."
  (gen/vector gen-dim 1 4))

(def gen-tensor-type
  "Generator for Malli tensor type specification."
  (gen/tuple (gen/return :tensor) gen-shape gen-dtype))

(def gen-elementwise-op
  "Generator for elementwise binary operations."
  (gen/elements [:stablehlo/add
                 :stablehlo/multiply
                 :stablehlo/subtract
                 :stablehlo/maximum
                 :stablehlo/minimum]))

(def gen-valid-graph
  "Generates valid Malli-schema-compliant EDN SSA compute graphs."
  (gen/fmap
   (fn [[op shape dtype c-val]]
     {:name "generated_ssa_graph"
      :invars [[:x [:tensor shape dtype]]
               [:y [:tensor shape dtype]]]
      :outvars [:out]
      :eqns [{:op :stablehlo/constant :value c-val :outvars [:c0]}
             {:op op :invars [:x :y] :outvars [:tmp]}
             {:op :stablehlo/add :invars [:tmp :c0] :outvars [:out]}]})
   (gen/tuple gen-elementwise-op gen-shape gen-dtype (gen/double* {:NaN? false :infinite? false}))))
