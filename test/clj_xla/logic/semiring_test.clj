(ns clj-xla.logic.semiring-test
  "Unit and generative property tests for Value-Carrying Semirings in Tensor Logic
   (MIN_PLUS / Tropical APSP, MAX_PRODUCT / Viterbi, and COUNTING path multiplicities)."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.semiring :as sring]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; -----------------------------------------------------------------------------
;; 1. Semiring Descriptor Specifications
;; -----------------------------------------------------------------------------

(deftest test-semiring-descriptors
  (testing "Standard semiring definitions have required identities and algebraic operators"
    (is (= :boolean (:id sring/boolean-semiring)))
    (is (= :min-plus (:id sring/min-plus-semiring)))
    (is (= :max-product (:id sring/max-product-semiring)))
    (is (= :counting (:id sring/counting-semiring)))

    ;; Tropical MIN_PLUS identities: zero is infinity (+1e30), one is zero (0.0)
    (is (> (double (:zero sring/min-plus-semiring)) 1e20))
    (is (= 0.0 (double (:one sring/min-plus-semiring))))

    ;; Viterbi MAX_PRODUCT identities: zero is 0.0, one is 1.0
    (is (= 0.0 (double (:zero sring/max-product-semiring))))
    (is (= 1.0 (double (:one sring/max-product-semiring))))))

(deftest test-semiring-step-ast
  (testing "semiring-step-ast produces valid Tensor Logic Hiccup equation with semiring attribute"
    (let [node (sring/semiring-step-ast sring/min-plus-semiring :C :A :B [:i :j] :k)]
      (is (ast/valid-node? node))
      (is (ast/eqn? node))
      (is (= [:C :i :j] (ast/head node)))
      (is (= {:semiring :min-plus} (ast/attrs node))))))

;; -----------------------------------------------------------------------------
;; 2. Execution Tests via OpenXLA PJRT
;; -----------------------------------------------------------------------------

(deftest test-all-pairs-shortest-paths-pjrt
  (testing "all-pairs-shortest-paths computes optimal shortest paths via OpenXLA PJRT (Bellman-Ford / Floyd-Warshall)"
    ;; Directed graph with 4 nodes:
    ;; 0 -> 1: wt 1.0
    ;; 1 -> 2: wt 2.0
    ;; 0 -> 2: wt 5.0 (suboptimal direct route; path via 1 is 1+2 = 3.0)
    ;; 2 -> 3: wt 1.0
    (let [n 4
          edges {[0 1] 1.0
                 [1 2] 2.0
                 [0 2] 5.0
                 [2 3] 1.0}
          res (sring/all-pairs-shortest-paths edges n {:max-iters 5})
          d-arr ^floats (:distances res)
          get-dist (fn [i j] (aget d-arr (+ (* i n) j)))]
      (is (true? (:converged? res)))
      ;; Diagonal should be 0.0
      (dotimes [i n]
        (is (= 0.0 (double (get-dist i i)))))
      ;; Shortest path 0 -> 2 should be 3.0 (via 1), not 5.0
      (is (< (Math/abs (- (get-dist 0 2) 3.0)) 1e-5))
      ;; Shortest path 0 -> 3 should be 4.0 (0 -> 1 -> 2 -> 3)
      (is (< (Math/abs (- (get-dist 0 3) 4.0)) 1e-5))
      ;; Path 1 -> 3 should be 3.0
      (is (< (Math/abs (- (get-dist 1 3) 3.0)) 1e-5))
      ;; Unreachable 3 -> 0 should remain infinity (>= 1e20)
      (is (> (get-dist 3 0) 1e20)))))

(deftest test-path-multiplicities-pjrt
  (testing "path-multiplicities computes exact path counts without clamping via OpenXLA PJRT"
    ;; Diamond graph:
    ;; 0 -> 1, 0 -> 2
    ;; 1 -> 3, 2 -> 3
    ;; In 2 hops, there are exactly 2 paths from 0 to 3: 0->1->3 and 0->2->3.
    (let [n 4
          edges #{[0 1] [0 2] [1 3] [2 3]}
          res (sring/path-multiplicities edges n 2)
          c-arr ^floats (:counts res)
          get-count (fn [i j] (aget c-arr (+ (* i n) j)))]
      ;; 2 paths from 0 to 3
      (is (= 2.0 (double (get-count 0 3))))
      ;; 0 paths from 0 to 1 in exactly 2 hops
      (is (= 0.0 (double (get-count 0 1))))
      ;; 0 paths from 3 to 0
      (is (= 0.0 (double (get-count 3 0)))))))

(deftest test-viterbi-most-probable-paths-pjrt
  (testing "viterbi-most-probable-paths computes maximum reliability path via OpenXLA PJRT"
    ;; Probabilistic transition network:
    ;; 0 -> 1 (0.9), 1 -> 3 (0.9) => path probability 0.81
    ;; 0 -> 2 (0.5), 2 -> 3 (0.5) => path probability 0.25
    ;; 0 -> 3 direct (0.4) => path probability 0.4
    (let [n 4
          transitions {[0 1] 0.9
                       [1 3] 0.9
                       [0 2] 0.5
                       [2 3] 0.5
                       [0 3] 0.4}
          res (sring/viterbi-most-probable-paths transitions n {:max-iters 4})
          p-arr ^floats (:probabilities res)
          get-prob (fn [i j] (aget p-arr (+ (* i n) j)))]
      (is (true? (:converged? res)))
      ;; Optimal path probability 0 -> 3 is 0.81 (via 1)
      (is (< (Math/abs (- (get-prob 0 3) 0.81)) 1e-5))
      ;; Self-transition probability is 1.0
      (is (= 1.0 (double (get-prob 0 0)))))))

;; -----------------------------------------------------------------------------
;; 3. Generative Property Tests (clojure.test.check)
;; -----------------------------------------------------------------------------

(defspec prop-tropical-triangle-inequality 25
  (prop/for-all [n (gen/choose 3 5)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      all-pairs (for [i (range n) j (range n) :when (not= i j)] [i j])
                      sampled-edges (into {}
                                          (keep (fn [pair]
                                                  (when (< (.nextDouble rnd) 0.6)
                                                    [pair (+ 1.0 (* 5.0 (.nextDouble rnd)))]))
                                                all-pairs))
                      res (sring/all-pairs-shortest-paths sampled-edges n {:max-iters n})
                      d-arr ^floats (:distances res)
                      get-d (fn [i j] (aget d-arr (+ (* i n) j)))]
                  ;; Invariant: Triangle inequality D[i, j] <= D[i, k] + D[k, j] for all i, j, k
                  (every? true?
                          (for [i (range n)
                                j (range n)
                                k (range n)]
                            (let [d-ij (double (get-d i j))
                                  d-ik (double (get-d i k))
                                  d-kj (double (get-d k j))]
                              (if (or (>= d-ik 1e20) (>= d-kj 1e20))
                                true
                                (<= d-ij (+ d-ik d-kj 1e-4)))))))))

(defspec prop-counting-semiring-non-negative 25
  (prop/for-all [n (gen/choose 3 6)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      ;; DAG edges (i < j guarantees acyclicity)
                      all-possible (for [i (range n) j (range n) :when (< i j)] [i j])
                      sampled-edges (set (filter (fn [_] (< (.nextDouble rnd) 0.5)) all-possible))
                      hops 2
                      res (sring/path-multiplicities sampled-edges n hops)
                      c-arr ^floats (:counts res)]
                  ;; Invariant: All path counts are finite non-negative integers
                  (every? true?
                          (map (fn [idx]
                                 (let [v (double (aget c-arr idx))]
                                   (and (>= v 0.0)
                                        (<= (Math/abs (- v (Math/round v))) 1e-4))))
                               (range (* n n)))))))
