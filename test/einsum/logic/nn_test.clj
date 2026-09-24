(ns einsum.logic.nn-test
  "Tests for neural network layers and transformer blocks represented as Tensor Logic."
  (:require [einsum.logic.lower :as lower]
            [einsum.logic.nn :as nn]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]))

(deftest test-linear-layer-ast
  (let [invars [[:x [:tensor [1 4 8] :f32]]
                [:w [:tensor [8 16] :f32]]]
        ast [:= [:y :b :p :out-dim] [:x :b :p :in-dim] [:w :in-dim :out-dim]]
        graph (lower/ast->graph "linear_test" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (is (= [:y] (:outvars graph)))))

(deftest test-mlp-block-ast
  (let [invars [[:x [:tensor [1 4 8] :f32]]
                [:fc-w [:tensor [8 16] :f32]]
                [:proj-w [:tensor [16 8] :f32]]]
        ast [:block {:name :mlp}
             [:= [:h :b :p :dff] {:act :silu} [:x :b :p :d] [:fc-w :d :dff]]
             [:= [:out :b :p :d] [:h :b :p :dff] [:proj-w :dff :d]]]
        graph (lower/ast->graph "mlp_test" invars ast #{:out})]
    (is (shlo/validate-graph graph))
    (is (= [:out] (:outvars graph)))))

(deftest test-multi-head-attention-ast
  (let [invars [[:x [:tensor [1 4 16] :f32]]
                [:w-q [:tensor [16 2 8] :f32]]
                [:w-k [:tensor [16 2 8] :f32]]
                [:w-v [:tensor [16 2 8] :f32]]
                [:w-o [:tensor [2 8 16] :f32]]]
        ast [:block {:name :mha}
             [:= [:q :b :p :h :dh] [:x :b :p :d] [:w-q :d :h :dh]]
             [:= [:k :b :p :h :dh] [:x :b :p :d] [:w-k :d :h :dh]]
             [:= [:v :b :p :h :dh] [:x :b :p :d] [:w-v :d :h :dh]]
             [:= [:scores :b :h :p-q :p-k] {:scale 0.3535} [:q :b :p-q :h :dh] [:k :b :p-k :h :dh]]
             [:= [:context :b :p-q :h :dh] [:scores :b :h :p-q :p-k] [:v :b :p-k :h :dh]]
             [:= [:attn-out :b :p-q :d] [:context :b :p-q :h :dh] [:w-o :h :dh :d]]]
        graph (lower/ast->graph "mha_test" invars ast #{:attn-out})]
    (is (shlo/validate-graph graph))
    (is (= [:attn-out] (:outvars graph)))))

(deftest test-transformer-residual-block-ast
  (let [invars [[:x [:tensor [1 4 8] :f32]]
                [:w1 [:tensor [8 16] :f32]]
                [:w2 [:tensor [16 8] :f32]]]
        ;; Residual block with implicit accumulation
        ast [:block {:name :residual-block}
             [:= [:ffn :b :p :dff] {:act :silu} [:x :b :p :d] [:w1 :d :dff]]
             [:= [:out :b :p :d] [:ffn :b :p :dff] [:w2 :dff :d]]
             [:= [:out :b :p :d] [:x :b :p :d]]]
        graph (lower/ast->graph "residual_block_test" invars ast #{:out})]
    (is (shlo/validate-graph graph))
    (is (= [:out] (:outvars graph)))))

(deftest test-nn-constructors
  (let [invars [[:x [:tensor [1 4 8] :f32]]
                [:fc-w [:tensor [8 16] :f32]]
                [:proj-w [:tensor [16 8] :f32]]]
        ast (nn/mlp [:out :b :p :d] [:x :b :p :d] [:fc-w :d :dff] [:proj-w :dff :d])
        graph (lower/ast->graph "mlp_fn_test" invars ast #{:out})]
    (is (shlo/validate-graph graph))
    (is (= [:out] (:outvars graph))))
  (let [invars [[:x [:tensor [1 4 16] :f32]]
                [:w-q [:tensor [16 2 8] :f32]]
                [:w-k [:tensor [16 2 8] :f32]]
                [:w-v [:tensor [16 2 8] :f32]]
                [:w-o [:tensor [2 8 16] :f32]]]
        ast (nn/causal-attention [:attn_out :b :p-q :d] [:x :b :p :d]
                                 [:w-q :d :h :dh] [:w-k :d :h :dh]
                                 [:w-v :d :h :dh] [:w-o :h :dh :d])
        graph (lower/ast->graph "attn_fn_test" invars ast #{:attn_out})]
    (is (shlo/validate-graph graph))
    (is (= [:attn_out] (:outvars graph)))))
