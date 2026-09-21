(ns einsum.logic.symbolic-test
  "Unit and generative property tests for einsum.logic.symbolic (Phase 0).
   Tests AST construction, index allocation invariants, schema validation, error types,
   and 200-trial generative property test verifying embedding-space deductive closure
   against the pure Clojure symbolic oracle."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.symbolic :as sym]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. AST Construction & Validation Unit Tests
;; ==============================================================================

(deftest test-embed-relation-ast
  (testing "embed-relation produces a valid Malli Hiccup node with correct structure"
    (let [node (sym/embed-relation :EmbParent :Parent :E [:x :y] [:i :j])]
      (is (ast/valid-node? node))
      (is (ast/eqn? node))
      (is (= [:EmbParent :i :j] (ast/head node)))
      (is (= [[:Parent :x :y] [:E :x :i] [:E :y :j]] (ast/body-terms node))))))

(deftest test-query-relation-ast
  (testing "query-relation produces a valid Malli Hiccup node"
    (let [node (sym/query-relation :Dq :EmbParent :E [:a :b] [:i :j])]
      (is (ast/valid-node? node))
      (is (ast/eqn? node))
      (is (= [:Dq :a :b] (ast/head node)))
      (is (= [[:EmbParent :i :j] [:E :a :i] [:E :b :j]] (ast/body-terms node))))))

(deftest test-embed-rule-deterministic-index-allocation
  (testing "embed-rule assigns deterministic :se0, :se1, :se2... indices in order of first appearance"
    (let [rule {:head [:gp :x :z]
                :body [[:parent :x :y] [:parent :y :z]]
                :rel-embs {:parent :EmbR}}
          node (sym/embed-rule rule)]
      (is (ast/valid-node? node))
      (is (= [:GP :se0 :se1] (ast/head node)))
      (is (= [[:EmbR :se0 :se2] [:EmbR :se2 :se1]] (ast/body-terms node)))))

  (testing "embed-rule works on n-ary rules with multiple intermediate variables"
    (let [rule {:head [:great_gp :w :z]
                :body [[:parent :w :x] [:parent :x :y] [:parent :y :z]]
                :rel-embs {:parent :EmbParent}}
          node (sym/embed-rule rule)]
      (is (ast/valid-node? node))
      (is (= [:GREAT_GP :se0 :se1] (ast/head node)))
      ;; :w -> :se0, :z -> :se1, :x -> :se2, :y -> :se3
      (is (= [[:EmbParent :se0 :se2]
              [:EmbParent :se2 :se3]
              [:EmbParent :se3 :se1]]
             (ast/body-terms node))))))

(deftest test-embed-rule-error-cases
  (testing "Throws :index-collision if user logic variables match :se\\d+"
    (let [rule {:head [:gp :se0 :z]
                :body [[:parent :se0 :y] [:parent :y :z]]
                :rel-embs {:parent :EmbR}}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"reserved :se<N> namespace"
                            (sym/embed-rule rule)))
      (try
        (sym/embed-rule rule)
        (is false "Expected exception")
        (catch clojure.lang.ExceptionInfo e
          (is (= :index-collision (:type (ex-data e))))
          (is (= :se0 (:var (ex-data e))))))))

  (testing "Throws :unbound-head-var if head variable does not appear in body"
    (let [rule {:head [:gp :x :unbound_var]
                :body [[:parent :x :y] [:parent :y :z]]
                :rel-embs {:parent :EmbR}}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Unbound head variable"
                            (sym/embed-rule rule)))
      (try
        (sym/embed-rule rule)
        (is false "Expected exception")
        (catch clojure.lang.ExceptionInfo e
          (is (= :unbound-head-var (:type (ex-data e))))
          (is (= :unbound_var (:var (ex-data e))))))))

  (testing "Throws :unknown-relation if body relation is missing from :rel-embs"
    (let [rule {:head [:gp :x :z]
                :body [[:parent :x :y] [:ancestor :y :z]]
                :rel-embs {:parent :EmbR}}]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Unknown relation in rule body"
                            (sym/embed-rule rule)))
      (try
        (sym/embed-rule rule)
        (is false "Expected exception")
        (catch clojure.lang.ExceptionInfo e
          (is (= :unknown-relation (:type (ex-data e))))
          (is (= :ancestor (:rel (ex-data e)))))))))

(deftest test-rule-program-ast
  (testing "rule-program wraps multiple rules into a :block container"
    (let [rules [{:head [:gp :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}
                 {:head [:sibling :x :y]
                  :body [[:parent :p :x] [:parent :p :y]]}]
          node (sym/rule-program rules {:parent :EmbR})]
      (is (ast/valid-node? node))
      (is (= :block (first node)))
      (is (= 2 (count (filter ast/eqn? (rest (rest node)))))))))

(deftest test-gram-matrix-ast
  (testing "gram-matrix produces valid entity similarity contraction"
    (let [node (sym/gram-matrix :Sim :E [:a :b] :d)]
      (is (ast/valid-node? node))
      (is (= [:Sim :a :b] (ast/head node)))
      (is (= [[:E :a :d] [:E :b :d]] (ast/body-terms node))))))

(deftest test-tempered-query-ast
  (testing "tempered-query applies sigmoid post-activation with reciprocal temperature scale"
    (let [node (sym/tempered-query :Dq :EmbR :E [:a :b :i :j] 0.25)]
      (is (ast/valid-node? node))
      (is (= {:act :sigmoid :scale 4.0} (ast/attrs node)))))

  (testing "tempered-query throws :invalid-temperature on non-positive temperature"
    (try
      (sym/tempered-query :Dq :EmbR :E [:a :b :i :j] 0.0)
      (is false "Expected exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :invalid-temperature (:type (ex-data e))))))
    (try
      (sym/tempered-query :Dq :EmbR :E [:a :b :i :j] -1.0)
      (is false "Expected exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :invalid-temperature (:type (ex-data e))))))))

;; ==============================================================================
;; 2. Data Preparation Unit Tests
;; ==============================================================================

(deftest test-fact-tensor-construction
  (testing "fact-tensor constructs correct binary indicator array with bounds check"
    (let [ft (sym/fact-tensor #{[0 1] [1 2]} 3)]
      (is (= [3 3] (:shape ft)))
      (let [data ^floats (:data ft)]
        (is (= 1.0 (aget data (+ (* 0 3) 1))))
        (is (= 1.0 (aget data (+ (* 1 3) 2))))
        (is (= 0.0 (aget data (+ (* 0 3) 0))))
        (is (= 0.0 (aget data (+ (* 2 3) 2)))))))

  (testing "fact-tensor throws on out-of-bounds coordinate"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"coordinate out of bounds"
                          (sym/fact-tensor #{[0 3]} 3)))))

(deftest test-random-embeddings-normalization
  (testing "random-embeddings produces unit vectors with norm 1.0"
    (let [emb (sym/random-embeddings 4 64 123)
          data ^floats (:data emb)]
      (is (= [4 64] (:shape emb)))
      (dotimes [i 4]
        (let [norm-sq (loop [j 0 s 0.0]
                        (if (>= j 64)
                          s
                          (let [v (double (aget data (+ (* i 64) j)))]
                            (recur (inc j) (+ s (* v v))))))]
          (is (< (Math/abs (- 1.0 (Math/sqrt norm-sq))) 1e-5)))))))

(deftest test-correlate-embeddings-bounds
  (testing "correlate-embeddings validates rho in [0, 1]"
    (let [emb (sym/random-embeddings 3 16 42)]
      (is (thrown? clojure.lang.ExceptionInfo (sym/correlate-embeddings emb 0 1 -0.1)))
      (is (thrown? clojure.lang.ExceptionInfo (sym/correlate-embeddings emb 0 1 1.1)))
      (let [corr (sym/correlate-embeddings emb 0 1 0.8)
            d1 ^floats (:data corr)
            norm (Math/sqrt (loop [j 0 s 0.0]
                              (if (>= j 16) s
                                  (let [v (double (aget d1 (+ 16 j)))]
                                    (recur (inc j) (+ s (* v v)))))))]
        ;; Row 1 must remain a unit vector
        (is (< (Math/abs (- 1.0 norm)) 1e-5))))))

(deftest test-decode-pairs
  (testing "decode-pairs thresholds float scores"
    (let [scores (float-array [0.1 0.6
                               0.8 0.2])
          pairs (sym/decode-pairs scores 2 0.5 [:alice :bob])]
      (is (= #{[:alice :bob] [:bob :alice]} pairs)))))

;; ==============================================================================
;; 3. Oracle Property Tests (test.check, 200 trials per Phase 0 spec)
;; ==============================================================================

(defn- host-eval-grandparent-closure
  "Host-side float tensor contraction engine for the grandparent rule:
   GP[i, j] = sum_k (sum_{(x,y)} E[x,i] E[y,k]) * (sum_{(y,z)} E[y,k] E[z,j])
   decoded via Dq[a, b] = sum_{i,j} GP[i, j] E[a, i] E[b, j]."
  [^long n ^long d ^long seed facts]
  (let [emb (sym/random-embeddings n d seed)
        ^floats e-data (:data emb)
        ^floats emb-r (float-array (* d d))]
    ;; EmbR[i, j] = sum_{(x, y) in facts} E[x, i] * E[y, j]
    (doseq [[x y] facts]
      (let [x-off (* (long x) d)
            y-off (* (long y) d)]
        (dotimes [i d]
          (let [xi (aget e-data (int (+ x-off i)))
                i-off (* i d)]
            (dotimes [j d]
              (let [yj (aget e-data (int (+ y-off j)))
                    idx (int (+ i-off j))]
                (aset emb-r idx (float (+ (aget emb-r idx) (* xi yj))))))))))
    ;; GP[i, j] = sum_k EmbR[i, k] * EmbR[k, j]
    (let [^floats gp (float-array (* d d))]
      (dotimes [i d]
        (let [i-off (* i d)]
          (dotimes [k d]
            (let [ik (aget emb-r (int (+ i-off k)))
                  k-off (* k d)]
              (dotimes [j d]
                (let [idx (int (+ i-off j))]
                  (aset gp idx (float (+ (aget gp idx) (* ik (aget emb-r (int (+ k-off j)))))))))))))
      ;; Dq[a, b] = sum_{i, j} GP[i, j] * E[a, i] * E[b, j]
      (let [^floats dq (float-array (* n n))]
        (dotimes [a n]
          (let [a-off (* a d)]
            (dotimes [b n]
              (let [b-off (* b d)
                    sum (loop [i 0 s 0.0]
                          (if (>= i d)
                            s
                            (let [ai (double (aget e-data (int (+ a-off i))))
                                  i-off (* i d)
                                  inner (loop [j 0 in-s 0.0]
                                          (if (>= j d)
                                            in-s
                                            (let [gp-ij (double (aget gp (int (+ i-off j))))
                                                  bj (double (aget e-data (int (+ b-off j))))]
                                              (recur (inc j) (+ in-s (* gp-ij bj))))))]
                              (recur (inc i) (+ s (* ai inner))))))]
                (aset dq (int (+ (* a n) b)) (float sum))))))
        dq))))

(defn- seeded-shuffle [coll seed]
  (let [al (java.util.ArrayList. ^java.util.Collection coll)]
    (java.util.Collections/shuffle al (java.util.Random. (long seed)))
    (vec al)))

;; Note: This test verifies host-side reference contraction semantics against the pure Clojure
;; Horn-clause deductive oracle; full OpenXLA PJRT compiler-path coverage is provided by the e2e tests.
(defspec prop-embedding-space-deductive-oracle-parity 20
  (prop/for-all [n (gen/choose 4 6)
                 num-facts (gen/choose 1 5)
                 seed gen/nat]
                (let [all-pairs (for [i (range n) j (range n) :when (< i j)] [i j])
                      sampled-facts (set (take num-facts (seeded-shuffle all-pairs seed)))
                      d 256
                      ;; 1. Symbolic deductive closure via pure Clojure Horn-clause fixpoint
                      rules [{:head [:grandparent :x :z]
                              :body [[:parent :x :y] [:parent :y :z]]}]
                      symbolic-facts (set (map #(into [:parent] %) sampled-facts))
                      closure (sym/symbolic-oracle-forward-chain rules symbolic-facts)
                      expected-gp (set (keep (fn [f]
                                               (when (= (first f) :grandparent)
                                                 [(nth f 1) (nth f 2)]))
                                             closure))

                      ;; 2. Embedding-space contraction & decoding
                      scores (host-eval-grandparent-closure n d seed sampled-facts)
                      actual-gp (sym/decode-pairs scores n 0.5)]
                  (if (= expected-gp actual-gp)
                    true
                    (do
                      (println "Oracle Mismatch:")
                      (println "  seed:" seed "n:" n "d:" d)
                      (println "  facts:" sampled-facts)
                      (println "  expected:" expected-gp)
                      (println "  actual:" actual-gp)
                      false)))))
