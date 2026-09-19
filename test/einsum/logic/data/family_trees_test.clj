(ns einsum.logic.data.family-trees-test
  "Generative property and unit tests for synthetic family tree generator (Experiment E16/E17).
   Enforces strict invariants on parent/grandparent/sibling ground-truth relations
   and train/held-out split disjointness before implementing core logic (Rule 1: Strict TDD)."
  (:require [einsum.logic.data.family-trees :as ft]
            [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Invariant Property Tests
;; ==============================================================================

(defspec prop-family-forest-determinism 20
  (prop/for-all [seed (gen/choose 1 1000000)
                 num-entities (gen/elements [32 48 64 96])
                 num-trees (gen/elements [4 6 8])]
                (let [opts {:num-entities num-entities :num-trees num-trees :seed seed}
                      f1 (ft/generate-family-forest opts)
                      f2 (ft/generate-family-forest opts)
                      f-diff (ft/generate-family-forest (assoc opts :seed (inc seed)))]
                  (and (= (:parent-pairs f1) (:parent-pairs f2))
                       (= (:grandparent-pairs f1) (:grandparent-pairs f2))
                       (= (:sibling-pairs f1) (:sibling-pairs f2))
                       (not= (:parent-pairs f1) (:parent-pairs f-diff))))))

(defspec prop-grandparent-parent-compositional-soundness 20
  (prop/for-all [seed (gen/choose 1 1000000)
                 num-entities (gen/elements [32 48 64])
                 num-trees (gen/elements [4 8])]
                (let [forest (ft/generate-family-forest {:num-entities num-entities
                                                         :num-trees num-trees
                                                         :seed seed})
                      parent-set (set (:parent-pairs forest))
                      gp-pairs (:grandparent-pairs forest)]
                  (every? (fn [[gp gc]]
                            (some (fn [m]
                                    (and (contains? parent-set [gp m])
                                         (contains? parent-set [m gc])))
                                  (range (:num-entities forest))))
                          gp-pairs))))

(defspec prop-sibling-parent-soundness 20
  (prop/for-all [seed (gen/choose 1 1000000)
                 num-entities (gen/elements [32 48 64])
                 num-trees (gen/elements [4 8])]
                (let [forest (ft/generate-family-forest {:num-entities num-entities
                                                         :num-trees num-trees
                                                         :seed seed})
                      parent-set (set (:parent-pairs forest))
                      sib-set (set (:sibling-pairs forest))]
                  (every? (fn [[s1 s2]]
                            (and (not= s1 s2)
                                 (contains? sib-set [s2 s1])
                                 (some (fn [p]
                                         (and (contains? parent-set [p s1])
                                              (contains? parent-set [p s2])))
                                       (range (:num-entities forest)))))
                          sib-set))))

(defspec prop-split-disjointness-and-coverage 20
  (prop/for-all [seed (gen/choose 1 1000000)
                 train-ratio (gen/elements [0.6 0.7 0.8])]
                (let [forest (ft/generate-family-forest {:num-entities 64
                                                         :num-trees 8
                                                         :seed seed
                                                         :train-ratio train-ratio})
                      all-gp (set (:grandparent-pairs forest))
                      gp-train (set (get-in forest [:splits :grandparent :train]))
                      gp-heldout (set (get-in forest [:splits :grandparent :held-out]))
                      all-sib (set (:sibling-pairs forest))
                      sib-train (set (get-in forest [:splits :sibling :train]))
                      sib-heldout (set (get-in forest [:splits :sibling :held-out]))
                      gp-train-entities (set (mapcat identity gp-train))
                      gp-heldout-entities (set (mapcat identity gp-heldout))
                      sib-train-entities (set (mapcat identity sib-train))
                      sib-heldout-entities (set (mapcat identity sib-heldout))]
                  (and (empty? (set/intersection gp-train gp-heldout))
                       (= (set/union gp-train gp-heldout) all-gp)
                       (empty? (set/intersection sib-train sib-heldout))
                       (= (set/union sib-train sib-heldout) all-sib)
                       (set/subset? gp-heldout-entities gp-train-entities)
                       (set/subset? sib-heldout-entities sib-train-entities)))))

(defspec prop-matrix-representation-consistency 20
  (prop/for-all [seed (gen/choose 1 1000000)]
                (let [forest (ft/generate-family-forest {:num-entities 32 :num-trees 4 :seed seed})
                      n (:num-entities forest)
                      check-matrix (fn [pairs mat]
                                     (let [pair-set (set pairs)]
                                       (every? (fn [i]
                                                 (every? (fn [j]
                                                           (let [val (double (nth (nth mat i) j))]
                                                             (if (contains? pair-set [i j])
                                                               (= 1.0 val)
                                                               (= 0.0 val))))
                                                         (range n)))
                                               (range n))))]
                  (and (check-matrix (:parent-pairs forest) (:parent-matrix forest))
                       (check-matrix (:grandparent-pairs forest) (:grandparent-matrix forest))
                       (check-matrix (:sibling-pairs forest) (:sibling-matrix forest))))))

;; ==============================================================================
;; 2. Hand-Verified Small Scale Unit Test (N=16)
;; ==============================================================================

(deftest test-family-forest-small-inspection
  (testing "Generates valid N=16 forest with non-empty parent, grandparent, and sibling sets"
    (let [forest (ft/generate-family-forest {:num-entities 16 :num-trees 2 :seed 42})]
      (is (= 16 (:num-entities forest)))
      (is (pos? (count (:parent-pairs forest))))
      (is (pos? (count (:grandparent-pairs forest))))
      (is (pos? (count (:sibling-pairs forest))))
      ;; Check shape of matrices
      (is (= 16 (count (:parent-matrix forest))))
      (is (= 16 (count (first (:parent-matrix forest)))))
      (is (= 16 (count (:grandparent-matrix forest))))
      ;; Ensure no entity is its own parent or grandparent
      (is (not-any? (fn [[a b]] (= a b)) (:parent-pairs forest)))
      (is (not-any? (fn [[a b]] (= a b)) (:grandparent-pairs forest)))
      (is (not-any? (fn [[a b]] (= a b)) (:sibling-pairs forest))))))
