(ns einsum.logic.shape-test
  "Generative property and unit tests for Tensor Logic shape unification."
  (:require [einsum.logic.shape :as shape]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-unify-shapes-standard-matmul
  (let [in-shapes {:x [2 128 768] :w [768 3072]}
        eqn [:= [:y :b :p :dff] [:x :b :p :d] [:w :d :dff]]
        resolved (shape/unify-shapes in-shapes [eqn])]
    (is (= [2 128 3072] (get resolved :y)))))

(deftest test-unify-shapes-detects-mismatch
  (let [in-shapes {:x [2 128 768] :w [512 3072]}
        eqn [:= [:y :b :p :dff] [:x :b :p :d] [:w :d :dff]]]
    (is (thrown? Exception (shape/unify-shapes in-shapes [eqn])))))

(defspec prop-dynamic-slice-shape-resolution
  50
  (prop/for-all [b (gen/choose 1 4)
                 s (gen/choose 16 128)
                 d (gen/choose 32 256)]
                (let [in-shapes {:x [b s d] :pos [1]}
                      eqn [:dynamic-slice [:y :b :one :d] [:x :b :p :d]
                           {:slice-sizes [1 1 d] :start-indices [0 :pos 0]}]
                      resolved (shape/unify-shapes in-shapes [eqn])]
                  (= [1 1 d] (get resolved :y)))))
(defspec prop-turboquant-unpack-shape-resolution
  50
  (prop/for-all [b (gen/choose 1 4)
                 s (gen/choose 8 128)
                 kvh (gen/choose 1 8)
                 pdh (gen/choose 8 64)]
                (let [in-shapes {:k_codes [b s kvh pdh]}
                      eqn [:turboquant-unpack [:k_unpacked :b :kvs :kvh :dh] [:k_codes :b :kvs :kvh :pdh]]
                      resolved (shape/unify-shapes in-shapes [eqn])]
                  (= [b s kvh (* pdh 4)] (get resolved :k_unpacked)))))

(deftest test-turboquant-unpack-shapes
  (testing "turboquant-unpack unifies shapes across 1D to 4D tensors"
    (let [in-shapes {:c1 [32] :c2 [16 32] :c4 [1 64 2 32]}
          eqns [[:turboquant-unpack [:u1 :d] [:c1 :pd]]
                [:turboquant-unpack [:u2 :b :d] [:c2 :b :pd]]
                [:turboquant-unpack [:u4 :b :s :h :d] [:c4 :b :s :h :pd]]]
          resolved (shape/unify-shapes in-shapes eqns)]
      (is (= [128] (get resolved :u1)))
      (is (= [16 128] (get resolved :u2)))
      (is (= [1 64 2 128] (get resolved :u4))))))
