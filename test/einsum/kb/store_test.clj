(ns einsum.kb.store-test
  "Unit and generative property tests for einsum.kb.store (Experiment E21).
   Repository Rule 1: Generative tests for invariants written FIRST before implementation.
   Verifies:
   - Schema-constrained commit (domain, cardinality, denial/acyclicity, identity)
   - Constraint totality (committable <==> schema-valid)
   - Disjunction soundness (ambiguous queries never return a single completion or false certainty)
   - Supersede-chain integrity and provenance tracking
   - Derived relation materialization (Datalog rules)
   - Ground query read performance"
  (:require [einsum.logic.data.family-trees :as ft]
            [einsum.kb.store :as kb]
            [einsum.logic.predicate-crystallization :as pc]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Standard Domain Schemas for Testing
;; ==============================================================================

(def test-family-schema
  {:parent {:arity 2
            :sorts [:person :person]
            :cardinality {:target-arg 1 :max 2}
            :denial #{:acyclic}}
   :sibling {:arity 2
             :sorts [:person :person]
             :identity {:via :parent :shared-parents true}}
   :grandparent {:arity 2
                 :sorts [:person :person]
                 :derived true
                 :rule {:head [:grandparent :?x :?z]
                        :body [[:parent :?x :?y] [:parent :?y :?z]]}}})

;; ==============================================================================
;; 2. Generative Property Tests (Repository Rule 1)
;; ==============================================================================

(defspec prop-constraint-totality 50
  (prop/for-all [num-nodes (gen/choose 4 12)
                 num-edges (gen/choose 10 30)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      entities (vec (map #(keyword (str "p" %)) (range num-nodes)))
                      entity-map (into {} (map (fn [e] [e :person]) entities))
                      init-kb (kb/create-kb test-family-schema {:entities entity-map})
          ;; Generate random candidate parent edges [parent child]
                      candidates
                      (vec (repeatedly num-edges
                                       (fn []
                                         (let [u (nth entities (.nextInt rnd num-nodes))
                                               v (nth entities (.nextInt rnd num-nodes))]
                                           [:parent u v]))))]
                  (loop [remaining candidates
                         cur-kb init-kb]
                    (if (empty? remaining)
                      true
                      (let [candidate (first remaining)
                            [_ p c] candidate
                            parents-before (set (map first (filter #(= (second %) c)
                                                                   (get-in cur-kb [:facts :parent]))))
                            would-exceed-cardinality? (and (not (contains? parents-before p))
                                                           (>= (count parents-before) 2))
                            would-cycle? (or (= p c)
                                             (kb/reachable? (get-in cur-kb [:facts :parent]) c p))
                            res (kb/assert-fact cur-kb candidate {:source :test-gen})]
                        (cond
                          (or would-cycle? would-exceed-cardinality?)
                          (and (= :schema-violation (:error res))
                               (contains? #{:cardinality :acyclicity} (:violation res))
                               (recur (rest remaining) cur-kb))

              ;; Case 2: Valid commit
                          :else
                          (and (map? res)
                               (not (:error res))
                               (contains? (get-in res [:facts :parent]) [p c])
                               (recur (rest remaining) res)))))))))

(defspec prop-disjunction-soundness 50
  (prop/for-all [child-idx (gen/choose 10 30)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      c (keyword (str "c" child-idx))
                      p1 (keyword (str "pA_" (.nextInt rnd 5)))
                      p2 (keyword (str "pB_" (.nextInt rnd 5)))
                      p3 (keyword (str "pC_" (.nextInt rnd 5)))
                      p4 (keyword (str "pD_" (.nextInt rnd 5)))
                      entities {c :person p1 :person p2 :person p3 :person p4 :person}
                      kb (kb/create-kb test-family-schema {:entities entities})
                      branch-1 #{[:parent p1 c] [:parent p2 c]}
                      branch-2 #{[:parent p3 c] [:parent p4 c]}
                      disj-kb (kb/assert-disjunction kb [branch-1 branch-2] {:source :ambiguity-test})
                      q1 (kb/query-fact disj-kb [:parent p1 c])
                      q2 (kb/query-fact disj-kb [:parent p3 c])
                      unrelated (kb/query-fact disj-kb [:parent :unrelated c])
          ;; Attempting to commit an ambiguous branch directly must be rejected
                      attempt-commit (kb/assert-fact disj-kb [:parent p1 c] {:source :premature-commit})]
                  (and
       ;; Ambiguous queries return :ambiguous with both completions
                   (= :ambiguous (:status q1))
                   (= 2 (count (:completions q1)))
                   (= :ambiguous (:status q2))
                   (= 2 (count (:completions q2)))
       ;; Unrelated query returns :false
                   (= :false (:status unrelated))
       ;; Direct commit of ambiguous candidate is rejected
                   (= :schema-violation (:error attempt-commit))
                   (= :ambiguity (:violation attempt-commit))))))

(defspec prop-supersede-chain-integrity 50
  (prop/for-all [num-ops (gen/choose 5 20)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      entities (into {} (map (fn [i] [(keyword (str "n" i)) :person]) (range 20)))
                      init-kb (kb/create-kb test-family-schema {:entities entities})]
                  (loop [step 0
                         cur-kb init-kb
                         active-facts []]
                    (if (>= step num-ops)
                      (every? (fn [fact]
                                (let [hist (kb/history cur-kb fact)]
                                  (and (seq hist)
                                       (apply <= (map :tx-id hist)))))
                              active-facts)
                      (let [op-type (if (and (seq active-facts) (< (.nextDouble rnd) 0.4)) :retract :assert)]
                        (if (= op-type :assert)
                          (let [u (keyword (str "n" (.nextInt rnd 10)))
                                v (keyword (str "n" (+ 10 (.nextInt rnd 10))))
                                fact [:parent u v]
                                res (kb/assert-fact cur-kb fact {:step step})]
                            (if (:error res)
                              (recur (inc step) cur-kb active-facts)
                              (recur (inc step) res (conj active-facts fact))))
              ;; Retract an active fact
                          (let [fact (nth active-facts (.nextInt rnd (count active-facts)))
                                res (kb/retract-fact cur-kb fact {:step step :reason :superseded})]
                            (if (:error res)
                              (recur (inc step) cur-kb active-facts)
                              (recur (inc step) res (vec (remove #(= % fact) active-facts))))))))))))

;; ==============================================================================
;; 3. Unit Tests for Pre-Registered Acceptance Criteria
;; ==============================================================================

(deftest test-c1-schema-checked-commit
  (testing "Acceptance Criterion 1: all 80 true parent edges commit cleanly; 81st edge gives child 3rd parent and is rejected"
    (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
          true-parents (set (:parent-pairs forest))
          entities (into {} (map (fn [i] [i :person]) (range 64)))
          kb0 (kb/create-kb test-family-schema {:entities entities})
          ;; Commit all 80 true parent edges
          committed-kb
          (reduce (fn [k [p c]]
                    (let [res (kb/assert-fact k [:parent p c] {:source :ground-truth})]
                      (is (not (:error res)) (str "Failed to assert true parent: " [p c]))
                      res))
                  kb0
                  true-parents)]
      (is (= 80 (count (get-in committed-kb [:facts :parent]))))
      ;; Now find any child with 2 parents and attempt to add a third parent
      (let [[_ target-child] (first true-parents)
            existing-parents (set (map first (filter #(= (second %) target-child) true-parents)))
            third-parent (first (remove #(or (contains? existing-parents %) (= % target-child)) (range 64)))
            bad-res (kb/assert-fact committed-kb [:parent third-parent target-child] {:source :invalid-3rd})]
        (is (= :schema-violation (:error bad-res)))
        (is (= :cardinality (:violation bad-res)))
        (is (= target-child (:target-arg bad-res)))))))

(deftest test-c2-cycle-rejection
  (testing "Acceptance Criterion 2: asserting backward edge (child -> founder) rejected naming acyclicity violation"
    (let [entities {:founder :person :child :person :grandchild :person}
          kb0 (kb/create-kb test-family-schema {:entities entities})
          kb1 (kb/assert-fact kb0 [:parent :founder :child] {:source :setup})
          kb2 (kb/assert-fact kb1 [:parent :child :grandchild] {:source :setup})
          ;; Attempting backward edge grandchild -> founder creates directed cycle
          cycle-res (kb/assert-fact kb2 [:parent :grandchild :founder] {:source :cycle-attempt})]
      (is (= :schema-violation (:error cycle-res)))
      (is (= :acyclicity (:violation cycle-res)))
      (is (= :grandchild (:from cycle-res)))
      (is (= :founder (:to cycle-res))))))

(deftest test-c3-type-rejection
  (testing "Acceptance Criterion 3: parent(alice, 42) rejected naming domain violation"
    (let [entities {:alice :person}
          kb (kb/create-kb test-family-schema {:entities entities})
          res (kb/assert-fact kb [:parent :alice 42] {:source :type-test})]
      (is (= :schema-violation (:error res)))
      (is (= :domain (:violation res)))
      (is (= 42 (:value res)))
      (is (= :person (:expected res))))))

(deftest test-c4-ambiguity-not-guessing
  (testing "Acceptance Criterion 4: fed mating disjunction, KB stores disjunction explicitly; query returns :ambiguous with completions"
    (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
          tree0 (first (:trees forest))
          m (:mating-child tree0)
          o (:other-child tree0)
          s (:spouse tree0)
          g1 (first (:gen2-children tree0))
          entities (into {} (map (fn [i] [i :person]) (range 64)))
          kb (kb/create-kb test-family-schema {:entities entities})
          ;; Branch 1 (true): {parent(m, g1), parent(s, g1)}
          ;; Branch 2 (mating-swapped): {parent(o, g1), parent(s, g1)}
          branch-1 #{[:parent m g1] [:parent s g1]}
          branch-2 #{[:parent o g1] [:parent s g1]}
          disj-kb (kb/assert-disjunction kb [branch-1 branch-2] {:source :e20-mating-ambiguity})
          q (kb/query-fact disj-kb [:parent m g1])]
      (is (= :ambiguous (:status q)))
      (is (= 2 (count (:completions q))))
      (is (contains? (set (:completions q)) branch-1))
      (is (contains? (set (:completions q)) branch-2))
      ;; Committing either branch directly as a definite fact must be a test failure
      (let [attempt-m (kb/assert-fact disj-kb [:parent m g1] {:source :guess})
            attempt-o (kb/assert-fact disj-kb [:parent o g1] {:source :guess})]
        (is (= :schema-violation (:error attempt-m)))
        (is (= :ambiguity (:violation attempt-m)))
        (is (= :schema-violation (:error attempt-o)))
        (is (= :ambiguity (:violation attempt-o)))))))

(deftest test-c5-derived-relations
  (testing "Acceptance Criterion 5: grandparent/2 materializes exactly the generator's grandparent pairs (F1 = 1.0)"
    (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
          true-parents (set (:parent-pairs forest))
          true-gp (set (:grandparent-pairs forest))
          entities (into {} (map (fn [i] [i :person]) (range 64)))
          kb0 (kb/create-kb test-family-schema {:entities entities})
          kb-populated (reduce (fn [k [p c]] (kb/assert-fact k [:parent p c] {:source :gt}))
                               kb0
                               true-parents)
          kb-materialized (kb/materialize-derived kb-populated)
          materialized-gp (set (get-in kb-materialized [:facts :grandparent]))
          f1-res (pc/pair-set-f1 materialized-gp true-gp)]
      (is (= true-gp materialized-gp))
      (is (== 1.0 (:f1 f1-res)))
      (is (== 1.0 (:precision f1-res)))
      (is (== 1.0 (:recall f1-res))))))

(deftest test-c6-provenance-and-retraction
  (testing "Acceptance Criterion 6: retract committed edge -> supersede marker present, history returns full assertion/retraction chain"
    (let [entities {:p1 :person :c1 :person}
          kb0 (kb/create-kb test-family-schema {:entities entities})
          kb1 (kb/assert-fact kb0 [:parent :p1 :c1] {:source :agent-step-1 :confidence 0.95})
          _ (is (contains? (get-in kb1 [:facts :parent]) [:p1 :c1]))
          kb2 (kb/retract-fact kb1 [:parent :p1 :c1] {:source :agent-step-2 :reason :found-contradiction})]
      ;; Edge removed from active facts
      (is (not (contains? (get-in kb2 [:facts :parent]) [:p1 :c1])))
      ;; History contains assertion and retraction
      (let [hist (kb/history kb2 [:parent :p1 :c1])]
        (is (= 2 (count hist)))
        (let [tx-assert (first hist)
              tx-retract (second hist)]
          (is (= :assert (:op tx-assert)))
          (is (= :agent-step-1 (get-in tx-assert [:provenance :source])))
          (is (= (:tx-id tx-retract) (:superseded-by tx-assert)))
          (is (= :retract (:op tx-retract)))
          (is (= (:tx-id tx-assert) (:supersedes tx-retract)))
          (is (= :found-contradiction (get-in tx-retract [:provenance :reason]))))))))

(deftest test-c7-read-latency
  (testing "Acceptance Criterion 7: read latency p50/p99 measured host-side in microseconds"
    (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
          true-parents (set (:parent-pairs forest))
          entities (into {} (map (fn [i] [i :person]) (range 64)))
          kb (reduce (fn [k [p c]] (kb/assert-fact k [:parent p c] {:source :gt}))
                     (kb/create-kb test-family-schema {:entities entities})
                     true-parents)
          queries (vec (take 50 (map (fn [[p c]] [:parent p c]) true-parents)))
          bench (kb/measure-read-latency kb queries 2000)]
      (is (pos? (:p50-us bench)))
      (is (pos? (:p99-us bench)))
      (is (< (:p50-us bench) 50.0) (str "p50 latency too high: " (:p50-us bench) " us"))
      (is (< (:p99-us bench) 200.0) (str "p99 latency too high: " (:p99-us bench) " us")))))
