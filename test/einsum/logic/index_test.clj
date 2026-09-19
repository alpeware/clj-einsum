(ns einsum.logic.index-test
  "Generative property tests for Tensor Logic index partitioning."
  (:require [einsum.logic.generators :as lg]
            [einsum.logic.index :as idx]
            [clojure.set :as set]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-index-partitioning-soundness
  100
  (prop/for-all [lhs-idx  (gen/vector-distinct lg/gen-index-kw {:min-elements 1 :max-elements 5})
                 rhs-idx  (gen/vector-distinct lg/gen-index-kw {:min-elements 1 :max-elements 5})
                 head-idx (gen/vector-distinct lg/gen-index-kw {:min-elements 0 :max-elements 5})]
                (let [{:keys [contracting batch lhs-free rhs-free]} (idx/partition-indices lhs-idx rhs-idx head-idx)
                      c-set (set contracting)
                      b-set (set batch)
                      lf-set (set lhs-free)
                      rf-set (set rhs-free)]
                  (and
       ;; Vectors are returned
                   (vector? contracting)
                   (vector? batch)
                   (vector? lhs-free)
                   (vector? rhs-free)
       ;; Pairwise disjoint
                   (empty? (set/intersection c-set b-set))
                   (empty? (set/intersection c-set lf-set))
                   (empty? (set/intersection c-set rf-set))
                   (empty? (set/intersection b-set lf-set))
                   (empty? (set/intersection b-set rf-set))
                   (empty? (set/intersection lf-set rf-set))
       ;; Completeness with respect to LHS
                   (= (set lhs-idx) (set/union c-set b-set lf-set))
       ;; Completeness with respect to RHS
                   (= (set rhs-idx) (set/union c-set b-set rf-set))))))

(deftest test-standard-gemm-partitioning
  (let [res (idx/partition-indices [:b :m :k] [:b :k :n] [:b :m :n])]
    (is (= [:k] (:contracting res)))
    (is (= [:b] (:batch res)))
    (is (= [:m] (:lhs-free res)))
    (is (= [:n] (:rhs-free res)))))

(deftest test-standard-vector-dot-partitioning
  (let [res (idx/partition-indices [:d] [:d] [])]
    (is (= [:d] (:contracting res)))
    (is (empty? (:batch res)))
    (is (empty? (:lhs-free res)))
    (is (empty? (:rhs-free res)))))
