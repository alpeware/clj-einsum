(ns einsum.logic.ast-test
  "Generative property tests for Tensor Logic Hiccup AST schema validation."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.generators :as lg]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-primitive-equations-satisfy-schema
  100
  (prop/for-all [eqn lg/gen-primitive-eqn]
                (ast/valid-node? eqn)))

(defspec prop-binary-contractions-satisfy-schema
  100
  (prop/for-all [eqn lg/gen-binary-contraction]
                (and (ast/valid-node? eqn)
                     (ast/eqn? eqn))))

(defspec prop-while-node-satisfies-schema
  50
  (prop/for-all [out-n (gen/fmap #(keyword (str "out_" %)) (gen/choose 1 100))
                 in-n (gen/fmap #(keyword (str "in_" %)) (gen/choose 1 100))]
                (ast/valid-node? [:while [out-n] [in-n] {:max-iters 10}])))

(defspec prop-cond-node-satisfies-schema
  50
  (prop/for-all [out-n (gen/fmap #(keyword (str "out_" %)) (gen/choose 1 100))
                 arg-n (gen/fmap #(keyword (str "arg_" %)) (gen/choose 1 100))]
                (ast/valid-node?
                 [:cond [out-n] {:args [arg-n]}
                  [:compare [:step_lt] [arg-n] [arg-n] {:direction "LT"}]
                  [:not [:not_out] [:step_lt]]
                  [:and [out-n] [:step_lt] [:not_out]]])))

(defspec prop-body-node-satisfies-schema
  50
  (prop/for-all [out-n (gen/fmap #(keyword (str "out_" %)) (gen/choose 1 100))
                 arg-n (gen/fmap #(keyword (str "arg_" %)) (gen/choose 1 100))]
                (ast/valid-node?
                 [:body [out-n] {:args [arg-n]}
                  [:+ [out-n] [arg-n] [arg-n]]])))

(defspec prop-argmax-node-satisfies-schema
  50
  (prop/for-all [out-n (gen/fmap #(keyword (str "out_" %)) (gen/choose 1 100))
                 in-n (gen/fmap #(keyword (str "in_" %)) (gen/choose 1 100))]
                (ast/valid-node?
                 [:argmax [out-n] [in-n] {:axis -1}])))


