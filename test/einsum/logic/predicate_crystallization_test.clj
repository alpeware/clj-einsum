(ns einsum.logic.predicate-crystallization-test
  "Unit and generative property tests for einsum.logic.predicate-crystallization.
   Repository Rule 1: Generative tests for invariants."
  (:require [einsum.logic.predicate-crystallization :as pc]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Naive Verification Oracles
;; ==============================================================================

(defn- naive-bool-mat-sq-oracle
  "O(N^3) naive ground-truth oracle for boolean matrix square."
  [edges n]
  (let [e-set (set edges)]
    (set
     (for [i (range n)
           j (range n)
           :when (some (fn [m] (and (contains? e-set [i m]) (contains? e-set [m j]))) (range n))]
       [i j]))))

(defn- naive-bool-sibling-oracle
  "O(N^3) naive ground-truth oracle for off-diagonal L^T o L (shared parent)."
  [edges n]
  (let [e-set (set edges)]
    (set
     (for [s1 (range n)
           s2 (range n)
           :when (and (not= s1 s2)
                      (some (fn [p] (and (contains? e-set [p s1]) (contains? e-set [p s2]))) (range n)))]
       [s1 s2]))))

;; ==============================================================================
;; 2. Generative Property Tests
;; ==============================================================================

(defspec prop-boolean-matmul-oracle-parity 50
  (prop/for-all [n (gen/choose 4 16)
                 raw-edges (gen/vector (gen/tuple (gen/choose 0 15) (gen/choose 0 15)) 1 40)]
                (let [valid-edges (set (filter (fn [[u v]] (and (< u n) (< v n))) raw-edges))
                      oracle (naive-bool-mat-sq-oracle valid-edges n)
                      actual (pc/bool-mat-or-and-sq valid-edges)]
                  (= oracle actual))))

(defspec prop-boolean-sibling-oracle-parity 50
  (prop/for-all [n (gen/choose 4 16)
                 raw-edges (gen/vector (gen/tuple (gen/choose 0 15) (gen/choose 0 15)) 1 40)]
                (let [valid-edges (set (filter (fn [[u v]] (and (< u n) (< v n))) raw-edges))
                      oracle (naive-bool-sibling-oracle valid-edges n)
                      actual (pc/bool-mat-trans-or-and valid-edges)]
                  (= oracle actual))))

(defspec prop-f1-invariants 50
  (prop/for-all [edges-a (gen/vector (gen/tuple (gen/choose 0 10) (gen/choose 0 10)) 0 20)
                 edges-b (gen/vector (gen/tuple (gen/choose 0 10) (gen/choose 0 10)) 0 20)]
                (let [set-a (set edges-a)
                      set-b (set edges-b)
                      f1-ab (:f1 (pc/pair-set-f1 set-a set-b))
                      f1-ba (:f1 (pc/pair-set-f1 set-b set-a))
                      f1-self (:f1 (pc/pair-set-f1 set-a set-a))]
                  (and (<= 0.0 f1-ab 1.0)
                       (== f1-ab f1-ba)
                       (== f1-self 1.0)))))

(defspec prop-greedy-search-monotonicity 25
  (prop/for-all [raw-cands (gen/vector (gen/tuple (gen/choose 0 7) (gen/choose 0 7)) 5 25)
                 raw-gp (gen/vector (gen/tuple (gen/choose 0 7) (gen/choose 0 7)) 1 10)
                 raw-sib (gen/vector (gen/tuple (gen/choose 0 7) (gen/choose 0 7)) 1 10)]
                (let [cands (set (filter (fn [[u v]] (not= u v)) raw-cands))
                      gp-target (set raw-gp)
                      sib-target (set (filter (fn [[u v]] (not= u v)) raw-sib))
                      lambda-val 0.01
                      fwd-res (pc/greedy-forward-search cands gp-target sib-target lambda-val)
                      bwd-res (pc/backward-elimination (:selected-edges fwd-res) gp-target sib-target lambda-val)
                      fwd-traj (:trajectory fwd-res)
                      bwd-traj (:trajectory bwd-res)]
      ;; Each step of trajectory must have monotonically non-decreasing score
                  (and (every? (fn [[s1 s2]] (>= (double (:score s2)) (double (:score s1))))
                               (partition 2 1 fwd-traj))
                       (every? (fn [[s1 s2]] (>= (double (:score s2)) (double (:score s1))))
                               (partition 2 1 bwd-traj))))))

;; ==============================================================================
;; 3. Unit Tests
;; ==============================================================================

(deftest test-crystallization-score-empty
  (testing "Empty set has 0 score"
    (is (== 0.0 (pc/crystallization-score #{} #{[0 1]} #{[1 2]} 0.01)))))

(deftest test-crystallization-exact-match
  (testing "Exact composition achieves near 2.0 score minus lambda * edges"
    (let [edges #{[0 1] [1 2]}
          gp-target #{[0 2]}
          sib-target #{}
          score (pc/crystallization-score edges gp-target sib-target 0.01)]
      ;; GP F1 = 1.0, Sib F1 = 1.0 (empty match), minus 2 * 0.01 = 0.02 -> 1.98
      (is (== 1.98 score)))))

(deftest test-oracle-ceiling-p0a
  (testing "P0a Oracle ceiling: backward elimination on candidate set containing true parents reaches F1 = 1.0"
    (let [make-tree (fn [offset]
                      (let [o offset]
                        [[(+ o 0) (+ o 1)]
                         [(+ o 0) (+ o 2)]
                         [(+ o 1) (+ o 3)]
                         [(+ o 1) (+ o 4)]
                         [(+ o 2) (+ o 5)]
                         [(+ o 2) (+ o 6)]
                         [(+ o 2) (+ o 7)]]))
          true-parents (set (mapcat make-tree (range 0 64 8)))
          gp-target (pc/bool-mat-or-and-sq true-parents)
          sib-target (pc/bool-mat-trans-or-and true-parents)
          lambda-val 0.01
          bwd-res (pc/backward-elimination true-parents gp-target sib-target lambda-val)
          recovered (:selected-edges bwd-res)
          f1 (:f1 (pc/pair-set-f1 recovered true-parents))]
      (is (= 56 (count true-parents)))
      (is (== 1.0 f1))
      (is (= true-parents recovered)))))
