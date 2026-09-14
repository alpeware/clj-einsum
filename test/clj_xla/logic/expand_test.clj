(ns clj-xla.logic.expand-test
  "Generative property tests for Tensor Logic AST recursive expansion."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.expand :as expand]
            [clj-xla.logic.generators :as lg]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.properties :as prop]))

(defspec prop-primitive-expansion-idempotence
  100
  (prop/for-all [eqn lg/gen-binary-contraction]
                (let [ctx {}
                      exp1 (expand/expand-ast ctx [eqn])
                      exp2 (expand/expand-ast ctx exp1)]
                  (= exp1 exp2))))

(deftest test-block-container-expansion
  (let [ast [:block {:name :layer-0}
             [:= [:x1 :b :p :d] [:x0 :b :p :d] [:w1 :d :d]]
             [:= [:x2 :b :p :d] [:x1 :b :p :d] [:w2 :d :d]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    (is (= :x1 (first (second (first expanded)))))
    (is (= :x2 (first (second (second expanded)))))))

(deftest test-nested-block-expansion
  (let [ast [:block {:name :root}
             [:block {:name :child-1}
              [:= [:a :i] [:b :i]]]
             [:block {:name :child-2}
              [:= [:c :i] [:d :i]]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))))

(deftest test-multi-term-pairwise-decomposition
  (let [ast [:= [:out :i :l]
             [:a :i :j]
             [:b :j :k]
             [:c :k :l]]
        expanded (expand/expand-ast {} ast)]
    ;; 3 terms should decompose into 2 binary contraction equations
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    ;; The intermediate equation retains :i and :k (contracting :j)
    (let [eq1 (first expanded)]
      (is (= [:i :k] (vec (rest (ast/head eq1)))))
      (is (= [[:a :i :j] [:b :j :k]] (ast/body-terms eq1))))
    ;; The final equation produces :out with :i and :l (contracting :k)
    (let [eq2 (second expanded)]
      (is (= [:out :i :l] (ast/head eq2)))
      (is (= [:c :k :l] (second (ast/body-terms eq2)))))
    ;; Each equation has at most 2 body terms
    (is (every? (fn [eqn] (<= (count (ast/body-terms eqn)) 2)) expanded))))

(deftest test-relation-embedding-decomposition
  (let [ast [:= [:EmbR :i :j]
             [:R :x :y]
             [:E :x :i]
             [:E :y :j]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/valid-node? expanded))
    ;; Step 1: contracts :x, keeps :y and :i
    (let [eq1 (first expanded)]
      (is (= [:y :i] (vec (rest (ast/head eq1)))))
      (is (= [[:R :x :y] [:E :x :i]] (ast/body-terms eq1))))
    ;; Step 2: contracts :y, keeps :i and :j
    (let [eq2 (second expanded)]
      (is (= [:EmbR :i :j] (ast/head eq2)))
      (is (= [:E :y :j] (second (ast/body-terms eq2)))))))
