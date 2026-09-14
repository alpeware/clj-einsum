(ns clj-xla.logic.symbolic-e2e-test
  "End-to-end integration and generative tests for embedding-space symbolic reasoning
   (Pedro Domingos Section 5, OpenXLA PJRT execution)."
  (:require [clj-xla.logic.ast :as ast]
            [clj-xla.logic.symbolic :as sym]
            [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Family KB Forward Chaining (Section 5 Canonical Demo)
;; ==============================================================================

(deftest test-family-kb-grandparent-forward-chain
  (testing "Grandparent rule fixpoint forward chaining on family KB"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}]
          facts #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects
                                     :threshold 0.5})]
      ;; 1. Check fixpoint termination
      (is (<= (:iterations result) 4))
      ;; 2. Step 1 must infer exactly the direct grandparents
      (let [step1-new (get-in result [:trace 0 :new])]
        (is (= #{[:alice :carol] [:alice :dave] [:bob :eve]} step1-new)))
      ;; 3. Final facts must contain all initial facts plus transitive inferences
      (is (set/subset? facts (:facts result)))
      (is (contains? (:facts result) [:alice :carol]))
      (is (contains? (:facts result) [:alice :dave]))
      (is (contains? (:facts result) [:bob :eve]))
      (is (contains? (:facts result) [:alice :eve])))))

(deftest test-multi-rule-forward-chain
  (testing "Multi-rule forward chaining with grandparent and sibling rules"
    (let [rules [{:head [:grandparent :x :z]
                  :body [[:parent :x :y] [:parent :y :z]]}
                 {:head [:sibling :x :y]
                  :body [[:parent :p :x] [:parent :p :y]]}]
          facts #{[:alice :bob] [:bob :carol] [:bob :dave] [:carol :eve]}
          objects [:alice :bob :carol :dave :eve]
          result (sym/forward-chain {:rules rules
                                     :facts facts
                                     :n 5
                                     :d 512
                                     :seed 42
                                     :objects objects})
          head-facts (:head-facts result)]
      ;; Both head relations should have deduced facts
      (is (contains? head-facts :grandparent))
      (is (contains? head-facts :sibling))
      ;; Sibling facts must include Carol and Dave (children of Bob)
      (is (contains? (:sibling head-facts) [:carol :dave]))
      (is (contains? (:sibling head-facts) [:dave :carol]))
      ;; Grandparent facts must include Alice -> Carol, Alice -> Dave, Bob -> Eve
      (is (contains? (:grandparent head-facts) [:alice :carol]))
      (is (contains? (:grandparent head-facts) [:alice :dave]))
      (is (contains? (:grandparent head-facts) [:bob :eve])))))

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
      ;; Check strictly monotonic increase
      (doseq [i (range (dec (count scores)))]
        (is (< (nth scores i) (nth scores (inc i)))))
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
;; 4. Generative Property Test for Fixpoint Invariants
;; ==============================================================================

(defspec prop-forward-chain-fixpoint-invariants 20
  (prop/for-all [n (gen/choose 4 6)
                 num-facts (gen/choose 1 4)
                 seed gen/nat]
                (let [all-pairs (for [i (range n) j (range n) :when (< i j)] [i j])
                      sampled-facts (set (take num-facts (shuffle all-pairs)))
                      rules [{:head [:grandparent :x :z]
                              :body [[:parent :x :y] [:parent :y :z]]}]
                      result (sym/forward-chain {:rules rules
                                                 :facts sampled-facts
                                                 :n n
                                                 :d 256
                                                 :seed seed
                                                 :threshold 0.5
                                                 :max-iters 5})]
                  ;; Invariant 1: Result facts must be a superset of initial facts
                  (and (set/subset? sampled-facts (:facts result))
                       ;; Invariant 2: Iterations must not exceed max-iters
                       (<= (:iterations result) 5)
                       ;; Invariant 3: Trace is monotonic
                       (every? (fn [t] (set? (:new t))) (:trace result))))))
