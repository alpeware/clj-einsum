(ns einsum.logic.constrained-search-test
  "Unit and generative property tests for einsum.logic.constrained-search (Experiment E20).
   Repository Rule 1: Generative tests for invariants."
  (:require [einsum.logic.constrained-search :as cs]
            [einsum.logic.data.family-trees :as ft]
            [einsum.logic.predicate-crystallization :as pc]
            [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property Tests
;; ==============================================================================

(defspec prop-indegree-invariant-across-random-moves 50
  (prop/for-all [num-moves (gen/choose 10 100)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      n 64
                      final-state
                      (loop [step 0
                             state (cs/make-empty-state n)]
                        (if (>= step num-moves)
                          state
                          (recur (inc step) (cs/propose-sa-move state n rnd))))]
                  (and (cs/valid-state-indegree? final-state)
                       (cs/acyclic-state? final-state)))))

(defspec prop-boolean-composition-parity 50
  (prop/for-all [num-assignments (gen/choose 5 25)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      n 32
                      state
                      (loop [step 0
                             cur (cs/make-empty-state n)]
                        (if (>= step num-assignments)
                          cur
                          (recur (inc step) (cs/propose-sa-move cur n rnd))))
                      edges (cs/state->edges state)
                      gp-pc (pc/bool-mat-or-and-sq edges)
                      sib-pc (pc/bool-mat-trans-or-and edges)
                      gp-score (:f1 (pc/pair-set-f1 gp-pc gp-pc))
                      sib-score (:f1 (pc/pair-set-f1 sib-pc sib-pc))]
                  (and (== 1.0 gp-score)
                       (== 1.0 sib-score)
                       (cs/acyclic-state? state)))))

;; ==============================================================================
;; 2. Unit Tests
;; ==============================================================================

(deftest test-acyclicity-rejection-on-cycle
  (testing "Rejects adding an edge that would create a directed cycle"
    ;; State: 0 -> 1 -> 2
    ;; Node 1 has parents [0 3], Node 2 has parents [1 4]
    (let [state (-> (cs/make-empty-state 8)
                    (assoc 1 [0 3])
                    (assoc 2 [1 4]))]
      ;; Trying to give node 0 parent 2 creates cycle 0 -> 1 -> 2 -> 0
      (is (false? (cs/cycle-free-with? state 0 2 5)))
      ;; Trying to give node 0 parent 1 creates cycle 0 -> 1 -> 0
      (is (false? (cs/cycle-free-with? state 0 1 5)))
      ;; Giving node 0 parents [6 7] is cycle-free
      (is (true? (cs/cycle-free-with? state 0 6 7))))))

(deftest test-p0b-mating-ambiguity-live-verification
  (testing "Phase 0b gate: mating-swapped parent set produces 100% identical G and S on all 5 seeds"
    (doseq [s [42 43 44 45 46]]
      (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed s})
            trees (:trees forest)
            g-all (set (:grandparent-pairs forest))
            s-all (set (:sibling-pairs forest))
            g-train (set (get-in forest [:splits :grandparent :train]))
            s-train (set (get-in forest [:splits :sibling :train]))
            swapped-parents
            (set
             (mapcat
              (fn [tree]
                (let [true-p (set (:parent-pairs tree))
                      m (:mating-child tree)
                      o (:other-child tree)
                      g-kids (set (:gen2-children tree))]
                  (set (map (fn [[u v]] (if (and (= u m) (contains? g-kids v)) [o v] [u v])) true-p))))
              trees))
            g-swapped (pc/bool-mat-or-and-sq swapped-parents)
            s-swapped (pc/bool-mat-trans-or-and swapped-parents)
            f1-res (pc/pair-set-f1 swapped-parents (set (:parent-pairs forest)))]
        (is (= g-all g-swapped) (format "Seed %d G-all match failed" s))
        (is (= s-all s-swapped) (format "Seed %d S-all match failed" s))
        (is (set/subset? g-train g-swapped))
        (is (set/subset? s-train s-swapped))
        (is (== 0.70 (:f1 f1-res)))))))

(deftest test-p0a-oracle-ceiling
  (testing "Phase 0a gate: constrained SA restricted to true parents reaches F1* = 1.0"
    (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
          g-all (set (:grandparent-pairs forest))
          s-all (set (:sibling-pairs forest))
          res (cs/run-oracle-ceiling-sa g-all s-all forest {:steps 15000 :seed 42})]
      (is (== 1.0 (:f1-star res)))
      (is (== 80 (:recovered-count res)))
      (is (== 2.0 (:best-score res))))))
