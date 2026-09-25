(ns einsum.logic.expand-test
  "Generative property tests for Tensor Logic AST recursive expansion."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.expand :as expand]
            [einsum.logic.generators :as lg]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-primitive-expansion-idempotence
  100
  (prop/for-all [eqn lg/gen-binary-contraction]
                (let [ctx {}
                      exp1 (expand/expand-ast ctx [eqn])
                      exp2 (expand/expand-ast ctx exp1)]
                  (= exp1 exp2))))

(deftest test-block-container-expansion
  (let [ast [:block {:name :layer-0}
             [:= [:x1 :b :p :d] [:x0 :b :p :d] [:w1 :d :d]]
             [:= [:x2 :b :p :d] [:x1 :b :p :d] [:w2 :d :d]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    (is (= :x1 (first (second (first expanded)))))
    (is (= :x2 (first (second (second expanded)))))))

(deftest test-nested-block-expansion
  (let [ast [:block {:name :root}
             [:block {:name :child-1}
              [:= [:a :i] [:b :i]]]
             [:block {:name :child-2}
              [:= [:c :i] [:d :i]]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))))

(deftest test-multi-term-pairwise-decomposition
  (let [ast [:= [:out :i :l]
             [:a :i :j]
             [:b :j :k]
             [:c :k :l]]
        expanded (expand/expand-ast {} ast)]
    ;; 3 terms should decompose into 2 binary contraction equations
    (is (= 2 (count expanded)))
    (is (every? ast/eqn? expanded))
    ;; The intermediate equation retains :i and :k (contracting :j)
    (let [eq1 (first expanded)]
      (is (= [:i :k] (vec (rest (ast/head eq1)))))
      (is (= [[:a :i :j] [:b :j :k]] (ast/body-terms eq1))))
    ;; The final equation produces :out with :i and :l (contracting :k)
    (let [eq2 (second expanded)]
      (is (= [:out :i :l] (ast/head eq2)))
      (is (= [:c :k :l] (second (ast/body-terms eq2)))))
    ;; Each equation has at most 2 body terms
    (is (every? (fn [eqn] (<= (count (ast/body-terms eqn)) 2)) expanded))))

(deftest test-relation-embedding-decomposition
  (let [ast [:= [:EmbR :i :j]
             [:R :x :y]
             [:E :x :i]
             [:E :y :j]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    (is (every? ast/valid-node? expanded))
    ;; Step 1: contracts :x, keeps :y and :i
    (let [eq1 (first expanded)]
      (is (= [:y :i] (vec (rest (ast/head eq1)))))
      (is (= [[:R :x :y] [:E :x :i]] (ast/body-terms eq1))))
    ;; Step 2: contracts :y, keeps :i and :j
    (let [eq2 (second expanded)]
      (is (= [:EmbR :i :j] (ast/head eq2)))
      (is (= [:E :y :j] (second (ast/body-terms eq2)))))))

(deftest test-layer-block-recurrence-and-wire-scoping
  (let [ast [:block {:name [:gpt2_layer 0]}
             [:layer-norm [:x_norm1 :b :p :d] [:h :b :p :d] [:ln1_g :d] [:ln1_b :d]]
             [:= [:qkv :b :p :qkv_dim] [:x_norm1 :b :p :d] [:attn_w :d :qkv_dim]]
             [:= [:h# :b :p :d] [:qkv :b :p :d]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 3 (count expanded)))
    (is (every? ast/valid-node? expanded))
    ;; 1. Check LayerNorm: :h -> :h0, :x_norm1 -> :gpt2_layer_0_x_norm1, :ln1_g -> :ln1_g_0
    (let [eq1 (first expanded)]
      (is (= [:gpt2_layer_0_x_norm1 :b :p :d] (ast/head eq1)))
      (is (= [:h0 :b :p :d] (first (ast/body-terms eq1))))
      (is (= [:ln1_g_0 :d] (second (ast/body-terms eq1))))
      (is (= [:ln1_b_0 :d] (nth (ast/body-terms eq1) 2))))
    ;; 2. Check Contraction: :qkv -> :gpt2_layer_0_qkv, :attn_w -> :attn_w_0
    (let [eq2 (second expanded)]
      (is (= :gpt2_layer_0_qkv (first (ast/head eq2))))
      (is (= [:gpt2_layer_0_x_norm1 :b :p :d] (first (ast/body-terms eq2))))
      (is (= :attn_w_0 (first (second (ast/body-terms eq2))))))
    ;; 3. Check Recurrence Next State: :h# -> :h1
    (let [eq3 (nth expanded 2)]
      (is (= [:h1 :b :p :d] (ast/head eq3)))
      (is (= [:gpt2_layer_0_qkv :b :p :d] (first (ast/body-terms eq3)))))))

(deftest test-layer-block-chaining
  (let [ast [:block {:name :two_layers}
             [:block {:name [:test_layer 0]}
              [:= [:h# :b :p :d] [:h :b :p :d] [:w :d :d]]]
             [:block {:name [:test_layer 1]}
              [:= [:h# :b :p :d] [:h :b :p :d] [:w :d :d]]]]
        expanded (expand/expand-ast {} ast)]
    (is (= 2 (count expanded)))
    ;; Layer 0: reads :h0, writes :h1, uses parameter :w_0
    (let [eq0 (first expanded)]
      (is (= [:h1 :b :p :d] (ast/head eq0)))
      (is (= [:h0 :b :p :d] (first (ast/body-terms eq0))))
      (is (= [:w_0 :d :d] (second (ast/body-terms eq0)))))
    ;; Layer 1: reads :h1, writes :h2, uses parameter :w_1
    (let [eq1 (second expanded)]
      (is (= [:h2 :b :p :d] (ast/head eq1)))
      (is (= [:h1 :b :p :d] (first (ast/body-terms eq1))))
      (is (= [:w_1 :d :d] (second (ast/body-terms eq1)))))))

(defspec prop-normalize-id-invariants
  50
  (prop/for-all [idx (gen/choose 0 50)
                 name-kw (gen/elements [:pl_in :q_w :k_w :attn_w :scores])]
                (let [norm (expand/normalize-id [name-kw idx])]
                  (and (keyword? norm)
                       (= norm (keyword (str (str/replace (name name-kw) "-" "_") "_" idx)))))))

(deftest test-layer-block-bare-cross-layer-recurrence
  (testing "Bare cross-layer recurrence keywords like :h0, :h1 are preserved inside later layers"
    (let [ast [:block {:name [:test_layer 3]}
               [:= [:h# :b :p :d] [:h :b :p :d] [:h0 :b :p :d] [:h1 :b :p :d]]]
          expanded (expand/expand-ast {} ast)]
      (is (= 2 (count expanded)))
      (let [eq2 (second expanded)]
        (is (= [:h4 :b :p :d] (ast/head eq2)))
        ;; Body terms should contain :h0 and :h1, NOT :h0_3 or :h1_3
        (let [all-body-names (set (mapcat (fn [t] [(first t)]) (mapcat ast/body-terms expanded)))]
          (is (contains? all-body-names :h0))
          (is (contains? all-body-names :h1))
          (is (not (contains? all-body-names :h0_3)))
          (is (not (contains? all-body-names :h1_3))))))))

(deftest test-block-dims-escape-hatch
  (testing "Block-level :dims attribute protects custom dimensions from being layer-suffixed"
    (let [ast [:block {:name [:test_layer 0] :dims [:dh :custom_dim]}
               [:= [:y :b :p :dh] [:x :b :p :custom_dim] [:w :custom_dim :dh]]]
          expanded (expand/expand-ast {} ast)]
      (is (= 1 (count expanded)))
      (let [eq (first expanded)]
        (is (= [:b :p :dh] (vec (rest (ast/head eq)))))
        (is (= [:b :p :custom_dim] (vec (rest (first (ast/body-terms eq))))))
        (is (= [:custom_dim :dh] (vec (rest (second (ast/body-terms eq))))))))))

(deftest test-composite-id-attr-map-normalization
  (testing "Composite identifiers inside equation attribute maps are properly normalized"
    (let [ast [:= [:y :b :p :d] {:index [:indices 3] :weight [:w 0]} [:x :b :p :d]]
          expanded (expand/expand-ast {} ast)]
      (is (= 1 (count expanded)))
      (let [eq (first expanded)
            attrs (ast/attrs eq)]
        (is (= :indices_3 (:index attrs)))
        (is (= :w_0 (:weight attrs)))))))

(defspec prop-nested-layer-block-double-expansion-idempotence
  30
  (prop/for-all [layers (gen/choose 1 4)]
                (let [ast (into [:block {:name :model}]
                                (mapv (fn [i]
                                        [:block {:name [:layer i]}
                                         [:= [:wire :b :p :d] [:h :b :p :d] [[:w i] :d :d]]
                                         [:= [:h# :b :p :d] [:wire :b :p :d]]])
                                      (range layers)))
                      exp1 (expand/expand-ast {} ast)
                      exp2 (expand/expand-ast {} exp1)]
                  (= exp1 exp2))))

