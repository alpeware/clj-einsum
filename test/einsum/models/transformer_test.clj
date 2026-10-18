(ns einsum.models.transformer-test
  "Unit and generative invariant tests for the Canonical Transformer Block (Vaswani et al., 2017).
   Verifies pure AST construction, multi-head attention scaling, and layer composition."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.transformer :as tf]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.schedule :as schedule]))

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

(defspec prop-chunked-attention-online-softmax-parity
  25
  (prop/for-all [p (gen/elements [1 2 4])
                 dh (gen/elements [16 32])]
                (let [b 1
                      h 2
                      kvh 2
                      kvs 128
                      chunk-size 64
                      q-data (float-array (repeatedly (* b p h dh) #(float (- (rand 2.0) 1.0))))
                      k-data (float-array (repeatedly (* b kvs kvh dh) #(float (- (rand 2.0) 1.0))))
                      v-data (float-array (repeatedly (* b kvs kvh dh) #(float (- (rand 2.0) 1.0))))
                      pos-data (int-array [64])
                      scale (/ 1.0 (Math/sqrt (double dh)))
                      invars [[:Q [:tensor [b p h dh] :f32]]
                              [:K [:tensor [b kvs kvh dh] :f32]]
                              [:V [:tensor [b kvs kvh dh] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast-chunked [:block {}
                                   [:chunked-attention [:ctx :b :p :h :dh]
                                    [:Q :b :p :h :dh]
                                    [:K :b :kvs :kvh :dh]
                                    [:V :b :kvs :kvh :dh]
                                    {:pos :pos
                                     :chunk-size chunk-size
                                     :scale scale
                                     :sliding-window 64}]]
                      ast-unchunked [:block {}
                                     [:chunked-attention [:ctx_ref :b :p :h :dh]
                                      [:Q :b :p :h :dh]
                                      [:K :b :kvs :kvh :dh]
                                      [:V :b :kvs :kvh :dh]
                                      {:pos :pos
                                       :chunk-size 128
                                       :scale scale
                                       :sliding-window 64}]]
                      g-chunked (uop-lower/ast->uop-graph "attn_chunked" invars ast-chunked [:ctx])
                      s-chunked (schedule/schedule-graph g-chunked)
                      out-chunked (schedule/eval-scheduled-graph s-chunked {:Q q-data :K k-data :V v-data :pos pos-data})
                      res-chunked ^floats (get out-chunked :ctx)

                      g-unchunked (uop-lower/ast->uop-graph "attn_unchunked" invars ast-unchunked [:ctx_ref])
                      s-unchunked (schedule/schedule-graph g-unchunked)
                      out-unchunked (schedule/eval-scheduled-graph s-unchunked {:Q q-data :K k-data :V v-data :pos pos-data})
                      res-unchunked ^floats (get out-unchunked :ctx_ref)

                      n-elems (* b p h dh)]
                  (every? true?
                          (for [i (range n-elems)]
                            (< (Math/abs (- (aget res-chunked i) (aget res-unchunked i))) 1e-4))))))
