(ns einsum.logic.shape-test
  "Generative property and unit tests for Tensor Logic shape unification."
  (:require [einsum.logic.shape :as shape]
            [clojure.test :refer [deftest is]]
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

