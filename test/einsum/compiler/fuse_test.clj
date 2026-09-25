(ns einsum.compiler.fuse-test
  "Tests for operation fusion (graph -> graph rewrites)."
  (:require [clojure.test :refer [deftest is testing]]
            [einsum.compiler.fuse :as fuse]))

(deftest matmul-bias-fusion-test
  (testing "dot_general + broadcast + add fuses into :fused/matmul-bias"
    (let [graph {:invars [[:x [:tensor [2 3] :f32]]
                          [:w [:tensor [3 4] :f32]]
                          [:b [:tensor [4] :f32]]]
                 :eqns [{:op :stablehlo/dot_general
                         :invars [:x :w]
                         :outvars [:t1]
                         :attrs {:contracting_dims {:lhs [1] :rhs [0]}
                                 :batch_dims {}}}
                        {:op :stablehlo/broadcast_in_dim
                         :invars [:b]
                         :outvars [:t2]
                         :attrs {:shape [2 4]}}
                        {:op :stablehlo/add
                         :invars [:t1 :t2]
                         :outvars [:out]
                         :attrs {}}]
                 :outvars [:out]}
          fused (fuse/fuse-graph graph)
          fused-eqns (:eqns fused)]
      (is (= 1 (count fused-eqns)))
      (is (= :fused/matmul-bias (:op (first fused-eqns))))
      (is (= [:x :w :b] (:invars (first fused-eqns))))
      (is (= [:out] (:outvars (first fused-eqns)))))))

(deftest no-fusion-when-bias-reused-test
  (testing "Does not fuse when intermediate is used elsewhere"
    (let [graph {:invars [[:x [:tensor [2 3] :f32]]
                          [:w [:tensor [3 4] :f32]]
                          [:b [:tensor [4] :f32]]]
                 :eqns [{:op :stablehlo/dot_general
                         :invars [:x :w]
                         :outvars [:t1]
                         :attrs {:contracting_dims {:lhs [1] :rhs [0]}
                                 :batch_dims {}}}
                        {:op :stablehlo/broadcast_in_dim
                         :invars [:b]
                         :outvars [:t2]
                         :attrs {:shape [2 4]}}
                        {:op :stablehlo/add
                         :invars [:t1 :t2]
                         :outvars [:out]
                         :attrs {}}
                        {:op :stablehlo/multiply
                         :invars [:t1 :t1]
                         :outvars [:other]
                         :attrs {}}]
                 :outvars [:out :other]}
          fused (fuse/fuse-graph graph)]
      (is (= 4 (count (:eqns fused)))))))

(deftest softmax-fusion-test
  (testing "Softmax core (max/sub/exp/sum/div) fuses into :fused/softmax"
    (let [graph {:invars [[:x [:tensor [2 4] :f32]]]
                 :eqns [{:op :stablehlo/reduce_max :invars [:x] :outvars [:max]
                         :attrs {:axes [-1] :keep_dims true}}
                        {:op :stablehlo/subtract :invars [:x :max] :outvars [:diff]}
                        {:op :stablehlo/exp :invars [:diff] :outvars [:exp]}
                        {:op :stablehlo/reduce_sum :invars [:exp] :outvars [:sum]
                         :attrs {:axes [-1] :keep_dims true}}
                        {:op :stablehlo/divide :invars [:exp :sum] :outvars [:out]}]
                 :outvars [:out]}
          fused (fuse/fuse-graph graph)
          fused-eqns (:eqns fused)]
      (is (= 1 (count fused-eqns)))
      (is (= :fused/softmax (:op (first fused-eqns))))
      (is (= [:x] (:invars (first fused-eqns))))
      (is (= [:out] (:outvars (first fused-eqns)))))))

(deftest layer-norm-fusion-test
  (testing "LayerNorm pattern fuses into :fused/layer-norm"
    (let [graph {:invars [[:x [:tensor [2 4] :f32]]
                          [:gamma [:tensor [4] :f32]]
                          [:beta [:tensor [4] :f32]]]
                 :eqns [{:op :stablehlo/reduce_mean :invars [:x] :outvars [:mean]
                         :attrs {:axes [-1] :keep_dims true}}
                        {:op :stablehlo/subtract :invars [:x :mean] :outvars [:diff]}
                        {:op :stablehlo/multiply :invars [:diff :diff] :outvars [:diff_sq]}
                        {:op :stablehlo/reduce_mean :invars [:diff_sq] :outvars [:var]
                         :attrs {:axes [-1] :keep_dims true}}
                        {:op :stablehlo/constant :invars [] :outvars [:eps] :value 1e-5}
                        {:op :stablehlo/add :invars [:var :eps] :outvars [:var_eps]}
                        {:op :stablehlo/sqrt :invars [:var_eps] :outvars [:std]}
                        {:op :stablehlo/divide :invars [:diff :std] :outvars [:xhat]}
                        {:op :stablehlo/multiply :invars [:xhat :gamma] :outvars [:scaled]}
                        {:op :stablehlo/add :invars [:scaled :beta] :outvars [:out]}]
                 :outvars [:out]}
          fused (fuse/fuse-graph graph)
          fused-eqns (:eqns fused)]
      (is (= 1 (count fused-eqns)))
      (is (= :fused/layer-norm (:op (first fused-eqns))))
      (is (= [:x :gamma :beta] (:invars (first fused-eqns))))
      (is (= [:out] (:outvars (first fused-eqns))))
      (is (= 1e-5 (get-in (first fused-eqns) [:attrs :eps]))))))
