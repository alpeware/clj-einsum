(ns einsum.logic.autodiff-test
  "Generative property and unit tests for Tensor Logic algebraic autodiff."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.autodiff :as ad]
            [clojure.test :refer [deftest is]]))

(deftest test-derive-adjoint-binary-contraction
  (let [eqn [:= [:out :b :m :n] [:x :b :m :k] [:w :k :n]]
        adjoints (ad/derive-adjoint-equations eqn)
        d-heads (set (map (comp first second) adjoints))]
    (is (= 2 (count adjoints)))
    (is (contains? d-heads :adj/x))
    (is (contains? d-heads :adj/w))
    (is (every? ast/eqn? adjoints))))

(deftest test-derive-adjoint-preserves-indices
  (let [eqn [:= [:out :b :m :n] [:x :b :m :k] [:w :k :n]]
        adjoints (ad/derive-adjoint-equations eqn)
        adj-x (first (filter #(= :adj/x (first (second %))) adjoints))
        adj-w (first (filter #(= :adj/w (first (second %))) adjoints))]
    (is (= [:adj/x :b :m :k] (second adj-x)))
    (is (= [:adj/w :k :n] (second adj-w)))))
