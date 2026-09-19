(ns einsum.logic.dce-test
  "Generative property tests for Tensor Logic backward-chaining dead-code elimination."
  (:require [einsum.logic.dce :as dce]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-dce-prunes-unused-equations
  (let [eqns [[:= [:used-1 :b :d] [:x :b :d] [:w1 :d :d]]
              [:= [:dead-branch :b :d] [:x :b :d] [:w-dead :d :d]]
              [:= [:out :b :d] [:used-1 :b :d] [:w2 :d :d]]]
        pruned (dce/prune-ast eqns #{:out})]
    (is (= 2 (count pruned)))
    (let [head-names (set (map (comp first second) pruned))]
      (is (contains? head-names :used-1))
      (is (contains? head-names :out))
      (is (not (contains? head-names :dead-branch))))))

(deftest test-dce-preserves-implicit-accumulation
  (let [eqns [[:= [:out :b :d] [:term-a :b :d]]
              [:= [:out :b :d] [:term-b :b :d]]
              [:= [:dead :b :d] [:term-c :b :d]]]
        pruned (dce/prune-ast eqns #{:out})]
    (is (= 2 (count pruned)))
    (is (every? #(= :out (first (second %))) pruned))))

(defspec prop-dce-retains-requested-targets
  50
  (prop/for-all [target (gen/elements [:out :final :res])]
                (let [eqns [[:= [:mid :b :d] [:in :b :d]]
                            [:= [target :b :d] [:mid :b :d]]
                            [:= [:unrelated :b :d] [:in :b :d]]]
                      pruned (dce/prune-ast eqns #{target})
                      heads (set (map (comp first second) pruned))]
                  (and (contains? heads target)
                       (contains? heads :mid)
                       (not (contains? heads :unrelated))))))
