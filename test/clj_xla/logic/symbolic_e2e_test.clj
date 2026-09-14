(ns clj-xla.logic.symbolic-e2e-test
  "End-to-end integration and generative tests for embedding-space symbolic reasoning
   (Pedro Domingos Section 5, OpenXLA PJRT execution)."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.symbolic :as sym]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn- seeded-shuffle [coll seed]
  (let [al (java.util.ArrayList. ^java.util.Collection coll)]
    (java.util.Collections/shuffle al (java.util.Random. (long seed)))
    (vec al)))

;; ==============================================================================
;; 1. Family KB Forward Chaining (Section 5 Canonical Demo)
;; ==============================================================================

(deftest test-family-kb-grandparent-forward-chain
  (testing "Grandparent rule fixpoint forward chaining on family KB with per-relation separation"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}]
          facts {:parent #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects
                                     :threshold 0.5})]
      ;; 1. Step 1 must infer exactly the direct grandparents
      (let [step1-new (get-in result [:trace 0 :new])]
        (is (= {:grandparent #{[:alice :carol] [:alice :dave] [:bob :eve]}} step1-new)))
      ;; 2. Final grandparent head-facts equals exactly #{[:alice :carol] [:alice :dave] [:bob :eve]}
      (is (= #{[:alice :carol] [:alice :dave] [:bob :eve]}
             (:grandparent (:head-facts result))))
      ;; 3. [:alice :eve] (great-grandparent) must NOT be labeled grandparent
      (is (not (contains? (:grandparent (:head-facts result)) [:alice :eve])))
      ;; 4. Parent EDB must still equal initial parent set exactly
      (is (= (:parent facts) (:parent (:facts result))))
      ;; 5. Converges in 2 iterations (derives in iter 1, confirms fixpoint in iter 2)
      (is (= 2 (:iterations result))))))

(deftest test-relation-separation-multi-rule
  (testing "Per-relation fixpoint keeps sibling and grandparent facts separated"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}
                 {:head [:sibling :x :y]
                  :body [[:parent :p :x] [:parent :p :y]]}]
          facts {:parent #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects
                                     :threshold 0.5})
          head-facts (:head-facts result)
          all-facts (:facts result)]
      ;; Sibling tuples appear only under :sibling, never under :grandparent or :parent
      (is (contains? (:sibling head-facts) [:carol :dave]))
      (is (contains? (:sibling head-facts) [:dave :carol]))
      (is (not (contains? (:grandparent head-facts) [:carol :dave])))
      (is (not (contains? (:grandparent head-facts) [:dave :carol])))
      (is (not (contains? (:parent all-facts) [:carol :dave])))
      ;; Grandparent tuples appear only under :grandparent, never under :sibling or :parent
      (is (contains? (:grandparent head-facts) [:alice :carol]))
      (is (not (contains? (:sibling head-facts) [:alice :carol])))
      (is (not (contains? (:parent all-facts) [:alice :carol])))
      ;; The :parent EDB is unchanged from the input
      (is (= (:parent facts) (:parent all-facts))))))

(deftest test-multi-rule-forward-chain
  (testing "Multi-rule forward chaining with grandparent and sibling rules under map API"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}
                 {:head [:sibling :x :y]
                  :body [[:parent :p :x] [:parent :p :y]]}]
          facts {:parent #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects
                                     :threshold 0.5})
          head-facts (:head-facts result)
          all-facts (:facts result)]
      (is (contains? head-facts :grandparent))
      (is (contains? head-facts :sibling))
      (is (contains? (:sibling head-facts) [:carol :dave]))
      (is (contains? (:sibling head-facts) [:dave :carol]))
      (is (contains? (:grandparent head-facts) [:alice :carol]))
      (is (contains? (:grandparent head-facts) [:alice :dave]))
      (is (contains? (:grandparent head-facts) [:bob :eve]))
      ;; No cross leakage into parent
      (is (= (:parent facts) (:parent all-facts))))))

(deftest test-recursive-rule-still-works
  (testing "Recursive ancestor rule correctly infers transitive closure with feedback"
    (let [rules [{:head [:ancestor :x :z]
                  :body [[:ancestor :x :y] [:parent :y :z]]}]
          parent-facts #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}
          facts {:parent parent-facts
                 :ancestor parent-facts}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects
                                     :threshold 0.5})
          ancestor-facts (get-in result [:head-facts :ancestor])]
      ;; Direct parents
      (is (contains? ancestor-facts [:alice :bob]))
      (is (contains? ancestor-facts [:bob :carol]))
      ;; Grandparents
      (is (contains? ancestor-facts [:alice :carol]))
      (is (contains? ancestor-facts [:alice :dave]))
      (is (contains? ancestor-facts [:bob :eve]))
      ;; Great-grandparent is legitimately reached through recursion
      (is (contains? ancestor-facts [:alice :eve]))
      ;; Parent relation remains intact
      (is (= parent-facts (:parent (:facts result)))))))

(deftest test-forward-chain-validation
  (testing "forward-chain validates binary-only arity and unique rule heads"
    ;; Non-binary rule head
    (try
      (sym/forward-chain {:rules [{:head [:foo :x :y :z] :body [[:bar :x :y]]}]
                          :facts {:bar #{[0 1]}}
                          :n 3 :d 128 :seed 1})
      (is false "Expected :unsupported-arity exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :unsupported-arity (:type (ex-data e))))))
    ;; Non-binary rule body
    (try
      (sym/forward-chain {:rules [{:head [:foo :x :y] :body [[:bar :x :y :z]]}]
                          :facts {:bar #{[0 1]}}
                          :n 3 :d 128 :seed 1})
      (is false "Expected :unsupported-arity exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :unsupported-arity (:type (ex-data e))))))
    ;; Non-binary fact tuple
    (try
      (sym/forward-chain {:rules [{:head [:foo :x :y] :body [[:bar :x :y]]}]
                          :facts {:bar #{[0 1 2]}}
                          :n 3 :d 128 :seed 1})
      (is false "Expected :unsupported-arity exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :unsupported-arity (:type (ex-data e))))))
    ;; Duplicate rule heads
    (try
      (sym/forward-chain {:rules [{:head [:foo :x :y] :body [[:bar :x :y]]}
                                  {:head [:foo :x :y] :body [[:baz :x :y]]}]
                          :facts {:bar #{[0 1]} :baz #{[0 1]}}
                          :n 3 :d 128 :seed 1})
      (is false "Expected :duplicate-rule-head exception")
      (catch clojure.lang.ExceptionInfo e
        (is (= :duplicate-rule-head (:type (ex-data e))))))))

;; ==============================================================================
;; 2. Backward Chaining Query Planner Helper (DCE)
;; ==============================================================================

(deftest test-query-plan-backward-chaining-dce
  (testing "query-plan prunes unrelated rules using dead-code elimination"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}
                 {:head [:sibling :x :y]
                  :body [[:parent :p :x] [:parent :p :y]]}
                 {:head [:unrelated :x :y]
                  :body [[:foo :x :y]]}]
          rel-embs {:parent :EmbParent :foo :EmbFoo}
          plan (sym/query-plan rules rel-embs :GRANDPARENT)
          pruned-ast (:pruned-ast plan)]
      (is (= #{:GRANDPARENT} (:targets plan)))
      (is (ast/valid-node? pruned-ast))
      ;; Pruned AST should contain GRANDPARENT equation but NOT SIBLING or UNRELATED
      (let [heads (set (map (fn [eqn] (first (ast/head eqn))) (filter ast/eqn? (rest (rest pruned-ast)))))]
        (is (contains? heads :GRANDPARENT))
        (is (not (contains? heads :SIBLING)))
        (is (not (contains? heads :UNRELATED)))))))

;; ==============================================================================
;; 3. Phase 2 Features: Tempered Query, Analogical Reasoning, Denoising
;; ==============================================================================

(deftest test-tempered-query-sigmoid-scaling
  (testing "tempered-query produces smooth sigmoid scaled probabilities via OpenXLA PJRT"
    (let [n 5 d 512
          emb (sym/random-embeddings n d 42)
          facts #{[0 1]}
          r-tensor (sym/fact-tensor facts n)
          ast [:block {:name :tempered-test}
               (sym/embed-relation :EmbR :R :E [:x :y] [:i :j])
               (sym/tempered-query :Dq :EmbR :E [:a :b :i :j] 0.5)]
          exec (sym/compile-query "tempered_test"
                                  [[:R [:tensor [n n] :f32]]
                                   [:E [:tensor [n d] :f32]]]
                                  ast
                                  [:Dq])
          outputs (sym/run-query! exec {:R (:data r-tensor) :E (:data emb)})
          scores (get outputs :Dq)
          ;; Index of [0, 1] is 0 * 5 + 1 = 1
          s-true (aget ^floats scores 1)
          ;; Index of [0, 2] is 0 * 5 + 2 = 2
          s-false (aget ^floats scores 2)]
      ;; sigmoid(1.0 / 0.5) = sigmoid(2.0) ~= 0.8808
      (is (> s-true 0.85))
      (is (< s-true 0.92))
      ;; sigmoid(0.0 / 0.5) = 0.5
      (is (> s-false 0.45))
      (is (< s-false 0.55)))))

(deftest test-analogical-reasoning-monotonicity
  (testing "correlate-embeddings smoothly scales inferred relation score monotonically with rho"
    (let [n 5 d 512
          base-emb (sym/random-embeddings n d 42)
          facts #{[0 1]}
          r-tensor (sym/fact-tensor facts n)
          ast [:block {:name :corr-test}
               (sym/embed-relation :EmbR :R :E [:x :y] [:i :j])
               (sym/query-relation :Dq :EmbR :E [:a :b] [:i :j])]
          exec (sym/compile-query "corr_test"
                                  [[:R [:tensor [n n] :f32]]
                                   [:E [:tensor [n d] :f32]]]
                                  ast
                                  [:Dq])
          rhos [0.0 0.25 0.5 0.75 1.0]
          scores (mapv (fn [rho]
                         (let [corr-emb (sym/correlate-embeddings base-emb 0 4 rho)
                               out (sym/run-query! exec {:R (:data r-tensor) :E (:data corr-emb)})]
                           (aget ^floats (get out :Dq) 21))) ;; [4, 1]
                       rhos)]
      ;; Check non-decreasing within epsilon tolerance (Defect 3)
      (doseq [i (range (dec (count scores)))]
        (is (< (- (nth scores i) (nth scores (inc i))) 1e-4)))
      ;; At rho=1.0, entity 4 should have score near 1.0
      (is (> (last scores) 0.99)))))

(deftest test-denoise-scores
  (testing "denoise-scores eliminates continuous noise by thresholding"
    (let [raw-scores (float-array [0.1 0.95 0.05 0.88])
          denoised (sym/denoise-scores raw-scores 2 0.5)]
      (is (= 1.0 (double (aget ^floats denoised 1))))
      (is (= 1.0 (double (aget ^floats denoised 3))))
      (is (= 0.0 (double (aget ^floats denoised 0))))
      (is (= 0.0 (double (aget ^floats denoised 2)))))))

;; ==============================================================================
;; 4. Generative Property Tests for Fixpoint Invariants and Oracle Parity
;; ==============================================================================

(defspec prop-forward-chain-fixpoint-invariants 20
  (prop/for-all [n (gen/choose 4 6)
                 num-facts (gen/choose 1 4)
                 seed gen/nat]
                (let [all-pairs (for [i (range n) j (range n) :when (< i j)] [i j])
                      sampled-facts (set (take num-facts (seeded-shuffle all-pairs seed)))
                      rules [{:head [:grandparent :x :z]
                              :body [[:parent :x :y] [:parent :y :z]]}]
                      result (sym/forward-chain {:rules rules
                                                 :facts {:parent sampled-facts}
                                                 :n n
                                                 :d 256
                                                 :seed seed
                                                 :threshold 0.5
                                                 :max-iters 5})]
                  ;; Invariant 1: Result facts for parent must equal initial parent facts
                  (and (= sampled-facts (get-in result [:facts :parent]))
                       ;; Invariant 2: Iterations must not exceed max-iters
                       (<= (:iterations result) 5)
                       ;; Invariant 3: Trace is monotonic map
                       (every? (fn [t] (map? (:new t))) (:trace result))))))

(defspec prop-forward-chain-matches-symbolic-oracle 50
  (prop/for-all [n (gen/choose 4 6)
                 num-facts (gen/choose 1 4)
                 seed gen/nat]
                (let [all-pairs (for [i (range n) j (range n) :when (< i j)] [i j])
                      sampled-parent (set (take num-facts (seeded-shuffle all-pairs seed)))
                      rules [{:head [:grandparent :x :z]
                              :body [[:parent :x :y] [:parent :y :z]]}
                             {:head [:ancestor :x :z]
                              :body [[:ancestor :x :y] [:parent :y :z]]}]
                      initial-facts {:parent sampled-parent
                                     :ancestor sampled-parent}
                      ;; 1. Run forward-chain (OpenXLA PJRT execution)
                      fc-result (sym/forward-chain {:rules rules
                                                    :facts initial-facts
                                                    :n n
                                                    :d 256
                                                    :seed seed
                                                    :threshold 0.5
                                                    :max-iters 4})
                      actual-facts (:facts fc-result)
                      ;; 2. Run reference Horn-clause oracle
                      oracle-input (set (for [[rel tuples] initial-facts
                                              tuple tuples]
                                          (into [rel] tuple)))
                      oracle-closure (sym/symbolic-oracle-forward-chain rules oracle-input)
                      expected-facts (reduce (fn [m [rel & args]]
                                               (update m rel (fnil conj #{}) (vec args)))
                                             {:parent #{} :ancestor #{} :grandparent #{}}
                                             oracle-closure)]
                  (= actual-facts expected-facts))))
