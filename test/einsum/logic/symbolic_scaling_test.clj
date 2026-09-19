(ns einsum.logic.symbolic-scaling-test
  "Unit, generative, and integration tests for scalability improvements to Tensor Logic
   (Shah & Zadrozny, arXiv:2601.17188v1: Direct Datalog Tensor Contraction, Compact Transformation
   Matrices, Superposition Construction, and Filtered Ranking Evaluation)."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.symbolic :as sym]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Direct Symbolic Datalog Contraction & Transitive Closure (Paper Exp 1)
;; ==============================================================================

(deftest test-datalog-transitive-step-ast
  (testing "datalog-transitive-step-ast produces valid clamped accumulation AST"
    (let [node (sym/datalog-transitive-step-ast :A :P :A_next)]
      (is (ast/valid-node? node))
      (is (= :block (first node)))
      (let [eqns (filter ast/eqn? (rest (rest node)))]
        (is (>= (count eqns) 2))
        (let [last-eqn (last eqns)]
          (is (= [:A_next :x :z] (ast/head last-eqn)))
          (is (= {:clamp [0.0 1.0]} (ast/attrs last-eqn))))))))

(deftest test-verification-checks
  (let [p-facts #{[0 1] [1 2]}
        a-valid #{[0 1] [1 2] [0 2]}
        a-missing #{[0 1] [0 2]} ;; missing [1 2]
        a-incomplete #{[0 1] [1 2]} ;; missing [0 2] (not closed)
        a-cyclic #{[0 1] [1 2] [0 2] [2 0] [0 0]} ;; has cycle
        n 3]
    (testing "containment? passes iff all direct parent edges are in closure"
      (is (true? (sym/containment? p-facts a-valid)))
      (is (false? (sym/containment? p-facts a-missing))))

    (testing "closure? passes iff A x P adds zero new edges"
      (is (true? (sym/closure? p-facts a-valid n)))
      (is (false? (sym/closure? p-facts a-incomplete n))))

    (testing "acyclic? passes iff diagonal is zero (no self-ancestor loops)"
      (is (true? (sym/acyclic? a-valid)))
      (is (false? (sym/acyclic? a-cyclic))))

    (testing "lineage-stats counts ancestors and descendants correctly"
      (let [stats-0 (sym/lineage-stats a-valid 0)
            stats-1 (sym/lineage-stats a-valid 1)
            stats-2 (sym/lineage-stats a-valid 2)]
        (is (= {:ancestors 0 :descendants 2} stats-0))
        (is (= {:ancestors 1 :descendants 1} stats-1))
        (is (= {:ancestors 2 :descendants 0} stats-2))))))

(deftest test-datalog-transitive-closure-synthetic-chain
  (testing "datalog-transitive-closure computes complete transitive closure via OpenXLA PJRT"
    (let [parent-facts #{[0 1] [1 2] [2 3] [3 4]}
          n 5
          res (sym/datalog-transitive-closure {:facts parent-facts :n n :max-iters 10})]
      (is (true? (:converged? res)))
      (is (= 4 (:iterations res)))
      (is (sym/containment? parent-facts (:facts res)))
      (is (sym/closure? parent-facts (:facts res) n))
      (is (sym/acyclic? (:facts res)))
      ;; Total edges in 5-node transitive closure is 5*4/2 = 10
      (is (= 10 (count (:facts res))))
      (is (contains? (:facts res) [0 4])))))

(deftest test-datalog-transitive-closure-scaling-large-graph
  (testing "datalog-transitive-closure scales efficiently to large graph (N=500 nodes) in OpenXLA PJRT"
    (let [n 500
          ;; 500-node forest: 5 trees of 100 nodes each
          edges (set (for [tree (range 5)
                           i (range 99)]
                       [(+ (* tree 100) i)
                        (+ (* tree 100) (inc i))]))
          res (sym/datalog-transitive-closure {:facts edges :n n :max-iters 100})]
      (is (true? (:converged? res)))
      (is (sym/containment? edges (:facts res)))
      (is (sym/closure? edges (:facts res) n))
      (is (sym/acyclic? (:facts res)))
      ;; Each 100-node line graph has 100*99/2 = 4950 reachable ancestor pairs
      ;; Total = 5 * 4950 = 24750 ancestor relationships!
      (is (= 24750 (count (:facts res)))))))

(defspec prop-datalog-transitive-closure-dag-invariants 30
  (prop/for-all [n (gen/choose 4 7)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
          ;; Generate random strict DAG edges (i < j guarantees acyclicity)
                      all-possible (for [i (range n) j (range n) :when (< i j)] [i j])
                      edge-prob 0.4
                      sampled-edges (set (filter (fn [_] (< (.nextDouble rnd) edge-prob)) all-possible))
                      res (sym/datalog-transitive-closure {:facts sampled-edges :n n :max-iters n})
                      closure-facts (:facts res)
          ;; Ground-truth reference from symbolic oracle
                      oracle-input (set (map #(into [:parent] %) sampled-edges))
                      oracle-closure (sym/symbolic-oracle-forward-chain
                                      [{:head [:ancestor :x :z] :body [[:parent :x :z]]}
                                       {:head [:ancestor :x :z] :body [[:ancestor :x :y] [:parent :y :z]]}]
                                      oracle-input)
                      expected-closure (set (keep (fn [f]
                                                    (when (= (first f) :ancestor)
                                                      [(nth f 1) (nth f 2)]))
                                                  oracle-closure))]
                  (and (sym/containment? sampled-edges closure-facts)
                       (sym/closure? sampled-edges closure-facts n)
                       (sym/acyclic? closure-facts)
                       (= expected-closure closure-facts)))))

;; ==============================================================================
;; 2. Compact Transformation Matrices & Composition (Paper Exp 2)
;; ==============================================================================

(deftest test-matrix-composition-ast-and-execution
  (testing "compose-matrices-ast produces valid contraction and chains via OpenXLA PJRT"
    (let [d 16
          ast [:block {:name :comp-test}
               (sym/compose-matrices-ast :M12 :M1 :M2 [:i :j] :k)]]
      (is (ast/valid-node? ast))
      (let [exec (sym/compile-query "comp_test"
                                    [[:M1 [:tensor [d d] :f32]]
                                     [:M2 [:tensor [d d] :f32]]]
                                    ast
                                    [:M12])
            ;; Identity matrix for M2
            m1-data (float-array (repeat (* d d) 1.0))
            m2-data (float-array (* d d))
            _ (dotimes [i d] (aset-float m2-data (int (+ (* i d) i)) (float 1.0)))
            out (sym/run-query! exec {:M1 m1-data :M2 m2-data})
            m12-data ^floats (get out :M12)]
        ;; M1 * I = M1
        (is (= 1.0 (double (aget m12-data 0))))
        (is (= 1.0 (double (aget m12-data (dec (* d d))))))))))

(deftest test-zero-shot-compositional-inference
  (testing "Composed relation matrices answer zero-shot multi-hop query in OpenXLA PJRT"
    (let [d 32
          n 4 ;; 0: Tokyo, 1: Japan, 2: Asia, 3: France
          ;; E has orthogonal unit vectors for entities
          e-data (float-array (* n d))
          _ (dotimes [i n] (aset-float e-data (int (+ (* i d) i)) (float 1.0)))
          ;; M_capital: Tokyo (0) -> Japan (1)
          m-cap (float-array (* d d))
          _ (aset-float m-cap (int (+ (* 0 d) 1)) (float 1.0))
          ;; M_located: Japan (1) -> Asia (2)
          m-loc (float-array (* d d))
          _ (aset-float m-loc (int (+ (* 1 d) 2)) (float 1.0))
          ;; Composed query: Tokyo (0) -M_cap-> -M_loc-> predict Asia (2)
          res (sym/predict-compositional-query {:subject-idx 0
                                                :relation-matrices [m-cap m-loc]
                                                :entity-embeddings {:shape [n d] :data e-data}
                                                :d d
                                                :temperature 1.0})]
      (is (= 2 (:top-entity-idx res)))
      (is (> (aget ^floats (:scores res) 2) (aget ^floats (:scores res) 1))))))

(defspec prop-matrix-composition-associativity 20
  (prop/for-all [seed gen/nat]
                (let [d 8
                      rnd (java.util.Random. (long seed))
                      rand-mat (fn []
                                 (let [arr (float-array (* d d))]
                                   (dotimes [i (* d d)] (aset-float arr i (float (- (* 2.0 (.nextDouble rnd)) 1.0))))
                                   arr))
                      m1 (rand-mat)
                      m2 (rand-mat)
                      m3 (rand-mat)
          ;; (M1 * M2) * M3 vs M1 * (M2 * M3) via OpenXLA PJRT
                      ast [:block {:name :assoc-test}
                           (sym/compose-matrices-ast :M12 :M1 :M2 [:i :j] :k)
                           (sym/compose-matrices-ast :LHS :M12 :M3 [:i :j] :k)
                           (sym/compose-matrices-ast :M23 :M2 :M3 [:i :j] :k)
                           (sym/compose-matrices-ast :RHS :M1 :M23 [:i :j] :k)]
                      exec (sym/compile-query "assoc_test"
                                              [[:M1 [:tensor [d d] :f32]]
                                               [:M2 [:tensor [d d] :f32]]
                                               [:M3 [:tensor [d d] :f32]]]
                                              ast
                                              [:LHS :RHS])
                      out (sym/run-query! exec {:M1 m1 :M2 m2 :M3 m3})
                      ^floats lhs (get out :LHS)
                      ^floats rhs (get out :RHS)]
                  (loop [idx 0 max-diff 0.0]
                    (if (>= idx (* d d))
                      (< max-diff 1e-4)
                      (let [diff (Math/abs (- (double (aget lhs idx)) (double (aget rhs idx))))]
                        (recur (inc idx) (Math/max max-diff diff))))))))

;; ==============================================================================
;; 3. Superposition Construction & Filtered Ranking Evaluation (Paper Exp 3)
;; ==============================================================================

(deftest test-superposition-relation-matrix-construction
  (testing "superposition-relation-matrix matches sum of outer products R_r = sum e_h (x) e_t"
    (let [n 4 d 8
          emb (sym/random-embeddings n d 42)
          facts #{[0 1] [2 3]}
          r-mat (sym/superposition-relation-matrix emb facts)
          ^floats r-data (:data r-mat)]
      (is (= [d d] (:shape r-mat)))
      ;; Verify against explicit manual sum of outer products
      (let [e-data ^floats (:data emb)
            expected (float-array (* d d))]
        (doseq [[h t] facts]
          (dotimes [i d]
            (let [hi (aget e-data (+ (* h d) i))]
              (dotimes [j d]
                (let [tj (aget e-data (+ (* t d) j))
                      idx (+ (* i d) j)]
                  (aset-float expected idx (+ (aget expected idx) (* hi tj))))))))
        (dotimes [k (* d d)]
          (is (< (Math/abs (- (aget r-data k) (aget expected k))) 1e-5)))))))

(defspec prop-superposition-relation-matrix-parity 20
  (prop/for-all [n (gen/choose 3 6)
                 d (gen/choose 4 8)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      emb (sym/random-embeddings n d seed)
                      all-pairs (for [i (range n) j (range n)] [i j])
                      sampled-facts (set (filter (fn [_] (< (.nextDouble rnd) 0.5)) all-pairs))
                      r-mat (sym/superposition-relation-matrix emb sampled-facts)
                      ^floats r-data (:data r-mat)
                      e-data ^floats (:data emb)
                      expected (float-array (* d d))]
                  (doseq [[h t] sampled-facts]
                    (dotimes [i d]
                      (let [hi (aget e-data (+ (* h d) i))]
                        (dotimes [j d]
                          (let [tj (aget e-data (+ (* t d) j))
                                idx (+ (* i d) j)]
                            (aset-float expected idx (+ (aget expected idx) (* hi tj))))))))
                  (loop [k 0 max-diff 0.0]
                    (if (>= k (* d d))
                      (< max-diff 1e-4)
                      (let [diff (Math/abs (- (double (aget r-data k)) (double (aget expected k))))]
                        (recur (inc k) (Math/max max-diff diff))))))))

(deftest test-bidirectional-link-prediction-pjrt
  (testing "superposition relation matrix predicts tail and head correctly via OpenXLA PJRT"
    (let [n 3 d 16
          emb (sym/random-embeddings n d 101)
          facts #{[0 1] [1 2]}
          r-mat (sym/superposition-relation-matrix emb facts)
          ;; Tail prediction for (0, r, ?): should rank 1 highest
          tail-res (sym/predict-tail-query {:head-idx 0
                                            :relation-matrix r-mat
                                            :entity-embeddings emb
                                            :temperature 0.1})
          ;; Head prediction for (?, r, 2): should rank 1 highest
          head-res (sym/predict-head-query {:tail-idx 2
                                            :relation-matrix r-mat
                                            :entity-embeddings emb
                                            :temperature 0.1})]
      (is (= 1 (:top-entity-idx tail-res)))
      (is (= 1 (:top-entity-idx head-res))))))

(deftest test-filtered-ranking-metrics
  (testing "evaluate-filtered-ranking computes exact MRR and Hits@K with known facts filtered"
    (let [queries [{:head 0 :rel :r1 :tail 2 :scores (float-array [0.1 0.9 0.8 0.2])}  ;; 0.9 is at idx 1, 0.8 at idx 2
                   {:head 1 :rel :r1 :tail 0 :scores (float-array [0.9 0.1 0.2 0.3])}] ;; 0.9 at idx 0
          all-known-triples #{[0 :r1 1]   ;; known true fact to be filtered out of query 1!
                              [0 :r1 2]   ;; target
                              [1 :r1 0]}  ;; target
          metrics (sym/evaluate-filtered-ranking queries all-known-triples)]
      ;; For query 0: idx 1 is filtered out, so target idx 2 has highest score -> rank 1
      ;; For query 1: target idx 0 has highest score -> rank 1
      (is (= 2 (:count metrics)))
      (is (= 1.0 (:mrr metrics)))
      (is (= 1.0 (:hits-1 metrics)))
      (is (= 1.0 (:hits-3 metrics)))
      (is (= 1.0 (:hits-10 metrics)))
      (is (= 1.0 (get metrics (keyword "hits@1")))))))

(defspec prop-filtered-ranking-metrics-invariants 30
  (prop/for-all [num-queries (gen/choose 5 15)
                 num-entities (gen/choose 10 20)
                 seed gen/nat]
                (let [rnd (java.util.Random. (long seed))
                      queries (mapv (fn [qid]
                                      (let [scores (float-array num-entities)]
                                        (dotimes [i num-entities]
                                          (aset-float scores i (float (.nextDouble rnd))))
                                        (let [target (mod qid num-entities)]
                                          {:head qid :rel :r :tail target :scores scores})))
                                    (range num-queries))
                      all-triples (set (map (fn [q] [(:head q) (:rel q) (:tail q)]) queries))
                      metrics (sym/evaluate-filtered-ranking queries all-triples)]
                  (and (<= 0.0 (:mrr metrics) 1.0)
                       (<= 0.0 (:hits-1 metrics) (:hits-3 metrics) (:hits-10 metrics) 1.0)
                       (= num-queries (:count metrics))))))
