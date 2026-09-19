(ns einsum.kb.device-test
  "Unit and generative property tests for einsum.kb.device (Experiment E22).
   Repository Rule 1: Generative tests for invariants written FIRST before implementation.
   Verifies:
   - Device KB query compilation and execution via OpenXLA PJRT
   - Exact query parity between VRAM-resident relation gather and E21 KB ground truth across all 64 entities x 3 relations (Gate P0b)
   - Protocol query window recognition and token-ID parsing
   - Verbalization bridge: index-to-token translation and answer token formatting"
  (:require [einsum.core :as xla]
            [einsum.logic.data.family-trees :as ft]
            [einsum.kb.store :as kb]
            [einsum.kb.device :as kbd]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property Tests (Repository Rule 1)
;; ==============================================================================

(defspec prop-device-query-parity-across-random-graphs 25
  (prop/for-all [seed gen/nat
                 rel-idx (gen/choose 0 2)
                 arg-idx (gen/choose 0 15)]
                (let [ctx (xla/init-backend! :cpu)
                      n 16
                      rnd (java.util.Random. (long seed))
                      total-size (* 3 n n)
                      table-data (float-array total-size)
          ;; Populate random binary entries
                      _ (dotimes [i total-size]
                          (when (< (.nextDouble rnd) 0.25)
                            (aset table-data i (float 1.0))))
                      exec (kbd/compile-kb-query-executable ctx n)
                      dev-buf (kbd/allocate-device-kb! ctx table-data n)
                      dev-res (kbd/execute-device-query! ctx exec dev-buf rel-idx arg-idx n)
                      dev-indices (set (kbd/row-slice->entity-indices dev-res))
          ;; Host reference
                      offset (+ (* (long rel-idx) n n) (* (long arg-idx) n))
                      expected-indices
                      (set
                       (filter (fn [j]
                                 (> (aget table-data (+ offset j)) (float 0.5)))
                               (range n)))]
                  (xla/destroy-buffer! ctx dev-buf)
                  (= expected-indices dev-indices))))

;; ==============================================================================
;; 2. Gate P0b: Query Parity on Full E21 Family Forest (64 entities x 3 relations)
;; ==============================================================================

(deftest test-p0b-full-family-parity
  (testing "Gate P0b: For all 64 entities x 3 relations, device gather results == E21 KB answers exactly"
    (let [ctx (xla/init-backend! :cpu)
          n 64
          forest (ft/generate-family-forest {:num-entities n :num-trees 8 :seed 42})
          schema {:parent {:arity 2 :sorts [:person :person] :cardinality {:target-arg 1 :max 2} :denial #{:acyclic}}
                  :sibling {:arity 2 :sorts [:person :person] :identity {:via :parent :shared-parents true}}
                  :grandparent {:arity 2 :sorts [:person :person] :derived true
                                :rule {:head [:grandparent :?x :?z] :body [[:parent :?x :?y] [:parent :?y :?z]]}}}
          entities (into {} (map (fn [i] [i :person]) (range n)))
          kb (reduce (fn [k [p c]] (kb/assert-fact k [:parent p c]))
                     (kb/create-kb schema {:entities entities})
                     (:parent-pairs forest))
          rel-data (kbd/build-relation-matrices kb n)
          table-array (:table rel-data)
          exec (kbd/compile-kb-query-executable ctx n)
          dev-buf (kbd/allocate-device-kb! ctx table-array n)]
      (try
        (doseq [rel-kw [:parent :sibling :ancestor]]
          (let [rel-idx (kbd/relation->index rel-kw)]
            (dotimes [arg n]
              (let [dev-res (kbd/execute-device-query! ctx exec dev-buf rel-idx arg n)
                    dev-entities (set (kbd/row-slice->entity-indices dev-res))
                    expected-entities (set (kbd/query-host-kb-reference kb rel-kw arg n))]
                (is (= expected-entities dev-entities)
                    (format "Mismatch for relation %s entity %d: expected %s, got %s"
                            rel-kw arg expected-entities dev-entities))))))
        (finally
          (xla/destroy-buffer! ctx dev-buf))))))

;; ==============================================================================
;; 3. Unit Tests: Query Window Recognition & Verbalization Formatting
;; ==============================================================================

(deftest test-query-window-recognition
  (testing "Detects query window and extracts relation and argument index"
    (let [open-id 48
          close-id 49
          ancestor-id 41903
          arg-3-id 236800
          tokens [105 2364 107 open-id ancestor-id arg-3-id close-id]
          detected (kbd/find-kb-query-window tokens open-id close-id)]
      (is (some? detected))
      (is (= :ancestor (:relation detected)))
      (is (= 3 (:arg detected)))
      (is (= 2 (:rel-idx detected))))))

(deftest test-answer-token-formatting
  (testing "Formats entity indices into observation tokens"
    (let [ans-open-id 50
          ans-close-id 51
          index-token-map {0 236771, 1 236770, 2 236778, 3 236800}
          tokens (kbd/format-answer-tokens ans-open-id ans-close-id index-token-map [1 0])]
      (is (= ans-open-id (first tokens)))
      (is (= ans-close-id (last tokens)))
      (is (contains? (set tokens) 236770))
      (is (contains? (set tokens) 236771)))))

(deftest test-extract-answer-entities
  (testing "Extracts entity sets accurately from verbalized text"
    (is (= #{2 6} (kbd/extract-answer-entities "The ancestors of person 3 are persons 2 and 6." 3)))
    (is (= #{5 7} (kbd/extract-answer-entities "The siblings of person 0 are 5 and 7." 0)))
    (is (= #{} (kbd/extract-answer-entities "Person 2 has no ancestors." 2)))
    (is (= #{} (kbd/extract-answer-entities "Person 4 does not have any siblings." 4)))
    (is (= #{3} (kbd/extract-answer-entities "Person 1 only has person 3." 1)))
    (is (= #{2 3 4 6} (kbd/extract-answer-entities "The ancestors are 2, 3, 4, 6." 5)))))
