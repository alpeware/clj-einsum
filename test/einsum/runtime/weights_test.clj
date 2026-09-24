(ns einsum.runtime.weights-test
  "Generative property tests and unit tests for WeightStore and automated invar inference."
  (:require [einsum.runtime.weights :as weights]
            [einsum.logic.lower :as lower]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-weight-store-resolution
  (testing "create-weight-store normalizes dtypes and resolves weights"
    (let [header {"model.embed_tokens.weight" {"shape" [1000 64] "dtype" "BF16"}
                  "model.layers.0.mlp.weight" {"shape" [64 256] "dtype" "F32"}}
          store (weights/create-weight-store header {:aliases {:embed_tokens "model.embed_tokens.weight"
                                                               :mlp_w "model.layers.0.mlp.weight"}})]
      (is (= {:shape [1000 64] :dtype :bf16} (weights/tensor-info store :embed_tokens)))
      (is (= {:shape [64 256] :dtype :f32} (weights/tensor-info store :mlp_w))))))

(deftest test-infer-invars-basic-gemm
  (testing "infer-invars automatically resolves weight invars from AST free variables"
    (let [ast [:= [:y :b :d] [:x :b :k] [:w :k :d]]
          dynamic-inputs {:x [:tensor [1 64] :f32]}
          weights-map {:w {:shape [64 128] :dtype :f32}}
          invars (weights/infer-invars ast dynamic-inputs weights-map)]
      (is (= [[:x [:tensor [1 64] :f32]]
              [:w [:tensor [64 128] :f32]]]
             invars))
      ;; Verify the inferred invars lower into a valid StableHLO graph via 4-arity
      (let [graph (lower/ast->graph "inferred_gemm" invars ast #{:y})]
        (is (some? (shlo/validate-graph graph))))
      ;; Verify lowering directly via 3-arity ast->graph with :inputs and :weights
      (let [graph-3 (lower/ast->graph "inferred_gemm_3" ast {:inputs dynamic-inputs
                                                             :weights weights-map
                                                             :targets #{:y}})]
        (is (some? (shlo/validate-graph graph-3)))
        (is (= invars (:invars graph-3)))))))

(defspec prop-inferred-invars-match-free-vars
  50
  (prop/for-all [batch (gen/choose 1 4)
                 dim-k (gen/choose 8 32)
                 dim-d (gen/choose 8 32)]
                (let [ast [:block {}
                           [:= [:h :b :k] [:x :b :k] [:w1 :k :k]]
                           [:= [:y :b :d] [:h :b :k] [:w2 :k :d]]]
                      dynamic-inputs {:x [:tensor [batch dim-k] :f32]}
                      weights-map {:w1 {:shape [dim-k dim-k] :dtype :f32}
                                   :w2 {:shape [dim-k dim-d] :dtype :f32}}
                      invars (weights/infer-invars ast dynamic-inputs weights-map)
                      invar-names (set (map first invars))]
                  (and (= invar-names #{:x :w1 :w2})
                       (= (get (into {} invars) :x) [:tensor [batch dim-k] :f32])
                       (= (get (into {} invars) :w1) [:tensor [dim-k dim-k] :f32])
                       (= (get (into {} invars) :w2) [:tensor [dim-k dim-d] :f32])))))
