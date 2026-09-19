(ns einsum.logic.stratified-negation-test
  "Unit and generative property tests for Stratified Negation in Tensor Logic
   (Dependency cycle analysis, negative cycle rejection, and multi-stratum fixpoint execution)."
  (:require [einsum.logic.stratify :as strat]
            [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; -----------------------------------------------------------------------------
;; 1. Precedence Graph & Stratification Cycle Analysis
;; -----------------------------------------------------------------------------

(deftest test-stratify-program-valid
  (testing "stratify-program assigns valid stratum numbers to stratified Datalog program"
    (let [rules [{:head [:reach :x :z] :body [[:edge :x :z]]}
                 {:head [:reach :x :z] :body [[:reach :x :y] [:edge :y :z]]}
                 {:head [:unconnected :x :y] :body [[:not [:reach :x :y]]]}
                 {:head [:critical :x :y] :body [[:unconnected :x :y] [:alert :x :y]]}]
          strata-map (strat/stratify-program rules)]
      ;; EDB relations :edge, :alert are stratum 0
      (is (= 0 (get strata-map :edge 0)))
      (is (= 0 (get strata-map :alert 0)))
      ;; :reach depends only positively on :edge, so stratum 0
      (is (= 0 (get strata-map :reach)))
      ;; :unconnected depends negatively on :reach, so stratum >= 1
      (is (= 1 (get strata-map :unconnected)))
      ;; :critical depends positively on :unconnected, so stratum 1
      (is (= 1 (get strata-map :critical))))))

(deftest test-stratify-program-negative-cycle-rejection
  (testing "stratify-program rejects unstratifiable programs with negative cycles"
    ;; Unstratifiable mutual negation cycle: P <- not Q, Q <- P
    (let [bad-rules [{:head [:p :x :y] :body [[:not [:q :x :y]]]}
                     {:head [:q :x :y] :body [[:p :x :y]]}]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Unstratified negative cycle detected"
                            (strat/stratify-program bad-rules))))

    ;; Self-negation: P <- not P
    (let [self-neg-rules [{:head [:p :x :y] :body [[:not [:p :x :y]]]}]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Unstratified negative cycle detected"
                            (strat/stratify-program self-neg-rules))))))

;; -----------------------------------------------------------------------------
;; 2. Stratified Fixpoint Execution via OpenXLA PJRT
;; -----------------------------------------------------------------------------

(deftest test-eval-stratified-pjrt
  (testing "eval-stratified computes multi-stratum fixpoint with complement injection in OpenXLA PJRT"
    ;; Graph with two disconnected components: {0, 1} and {2, 3}
    (let [n 4
          facts {:edge #{[0 1] [1 0] [2 3] [3 2]}}
          rules [{:head [:reach :x :z] :body [[:edge :x :z]]}
                 {:head [:reach :x :z] :body [[:reach :x :y] [:edge :y :z]]}
                 {:head [:unconnected :x :y] :body [[:not [:reach :x :y]]]}]
          res (strat/eval-stratified {:rules rules
                                      :facts facts
                                      :n n
                                      :d 256
                                      :seed 42})
          all-facts (:facts res)
          reach-facts (get all-facts :reach)
          unconn-facts (get all-facts :unconnected)]
      ;; Stratum 0 reachability
      (is (contains? reach-facts [0 1]))
      (is (contains? reach-facts [2 3]))
      (is (not (contains? reach-facts [0 2])))
      (is (not (contains? reach-facts [1 3])))

      ;; Stratum 1 unconnected: complement of reachability
      ;; Nodes across components are unconnected:
      (is (contains? unconn-facts [0 2]))
      (is (contains? unconn-facts [0 3]))
      (is (contains? unconn-facts [1 2]))
      (is (contains? unconn-facts [1 3]))
      ;; Nodes within same component are NOT unconnected:
      (is (not (contains? unconn-facts [0 1])))
      (is (not (contains? unconn-facts [2 3]))))))

;; -----------------------------------------------------------------------------
;; 3. Generative Property Tests (clojure.test.check)
;; -----------------------------------------------------------------------------

(defspec prop-stratified-complement-conservation 20
  (prop/for-all [n (gen/choose 3 5)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      all-pairs (set (for [i (range n) j (range n)] [i j]))
                      sampled-edges (set (filter (fn [_] (< (.nextDouble rnd) 0.4)) all-pairs))
                      facts {:edge sampled-edges}
                      rules [{:head [:reach :x :z] :body [[:edge :x :z]]}
                             {:head [:unconnected :x :y] :body [[:not [:reach :x :y]]]}]
                      res (strat/eval-stratified {:rules rules
                                                  :facts facts
                                                  :n n
                                                  :d 1024
                                                  :seed seed})
                      reach (get-in res [:facts :reach] #{})
                      unconn (get-in res [:facts :unconnected] #{})]
                  ;; Invariant 1: Disjoint intersection (reach intersect unconn = empty)
                  ;; Invariant 2: Union covers all n x n pairs
                  (and (empty? (set/intersection reach unconn))
                       (= all-pairs (set/union reach unconn))))))

(defspec prop-stratification-order-invariants 30
  (prop/for-all [_seed gen/nat]
                (let [;; Construct a random 3-stratum DAG of relations:
                      ;; stratum 0: r0, r1
                      ;; stratum 1: r2 (depends neg on r0)
                      ;; stratum 2: r3 (depends neg on r2)
                      rules [{:head [:r0 :x :y] :body [[:e :x :y]]}
                             {:head [:r1 :x :y] :body [[:r0 :x :y]]}
                             {:head [:r2 :x :y] :body [[:not [:r0 :x :y]]]}
                             {:head [:r3 :x :y] :body [[:not [:r2 :x :y]] [:r1 :x :y]]}]
                      strata (strat/stratify-program rules)]
                  (and (= 0 (get strata :r0))
                       (= 0 (get strata :r1))
                       (= 1 (get strata :r2))
                       (= 2 (get strata :r3))
                       (< (get strata :r0) (get strata :r2))
                       (< (get strata :r2) (get strata :r3))
                       (<= (get strata :r1) (get strata :r3))))))
