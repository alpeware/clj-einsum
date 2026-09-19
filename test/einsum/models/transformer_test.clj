(ns einsum.models.transformer-test
  "Unit and generative invariant tests for the Canonical Transformer Block (Vaswani et al., 2017).
   Verifies pure AST construction, multi-head attention scaling, and layer composition."
  (:require [einsum.models.transformer :as tf]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Pure AST Schema & Construction Tests
;; ==============================================================================

(deftest test-multi-head-attention-ast
  (testing "multi-head-attention-ast generates a valid Hiccup block vector"
    (let [ast (tf/multi-head-attention-ast 64)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :multi_head_attention (get-in ast [1 :name]))))))

(deftest test-ffn-block-ast
  (testing "ffn-block-ast generates a valid Hiccup block vector"
    (let [ast (tf/ffn-block-ast)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ffn_block (get-in ast [1 :name]))))))

(deftest test-transformer-block-ast
  (testing "transformer-block-ast generates the full canonical layer"
    (let [ast (tf/transformer-block-ast 64)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :transformer_layer (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariants
;; ==============================================================================

(defspec prop-mha-scale-soundness 50
  (prop/for-all [head-dim (gen/choose 16 128)]
                (let [ast (tf/multi-head-attention-ast head-dim)
                      scale-entry (first (filter (fn [item] (and (vector? item)
                                                                 (= (first item) :=)
                                                                 (= (first (second item)) :scores)))
                                                 (subvec ast 2)))
                      expected-scale (/ 1.0 (Math/sqrt (double head-dim)))
                      actual-scale (get-in scale-entry [2 :scale])]
                  (<= (Math/abs (- (double actual-scale) (double expected-scale))) 1e-6))))

(defspec prop-transformer-ast-invariants 50
  (prop/for-all [head-dim (gen/choose 16 64)
                 causal? gen/boolean]
                (let [ast (tf/transformer-block-ast head-dim {:causal? causal?})]
                  (and
                   (vector? ast)
                   (= :block (first ast))
       ;; Must contain attention, FFN, and rms-norm components
                   (some (fn [term] (and (vector? term) (= :scores (first (second term))))) (subvec ast 2))
                   (some (fn [term] (and (vector? term) (= :ffn_out (first (second term))))) (subvec ast 2))
                   (some (fn [term] (and (vector? term) (= :rms-norm (first term)))) (subvec ast 2))))))
