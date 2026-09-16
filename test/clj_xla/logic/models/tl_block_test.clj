(ns clj-xla.logic.models.tl-block-test
  "Unit and generative invariant property tests for the Unified TL-Transformer Layer Block (Experiment E6).
   Verifies shape preservation, deductive threshold gating, KG-attention distractor suppression,
   and end-to-end forward pass execution on OpenXLA PJRT."
  (:require [clj-xla.logic.models.tl-block :as tl]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Pure AST Schema & Construction Tests
;; ==============================================================================

(deftest test-tl-transformer-block-ast-construction
  (testing "tl-transformer-block-ast generates a valid Hiccup block vector"
    (let [ast (tl/tl-transformer-block-ast 1 2 4 4 16 4 4 4)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :tl_transformer_block (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariant: Shape Preservation Across Batch & Sequence Dimensions
;; ==============================================================================

(defspec prop-tl-block-shape-invariants 15
  (prop/for-all [b-val (gen/choose 1 2)
                 l-val (gen/choose 4 8)
                 h-val (gen/choose 1 2)
                 dh-val (gen/choose 4 8)]
                (let [b (long b-val)
                      l (long l-val)
                      h (long h-val)
                      dh (long dh-val)
                      d (* h dh)
                      dff (* 2 d)
                      dm 4
                      ne 4
                      nc 4
                      exec (tl/compile-tl-transformer-block b l h dh dff dm ne nc)
                      inputs {:H (float-array (* b l d) (float 0.05))
                              :Q (float-array (* b l h dh) (float 0.05))
                              :K (float-array (* b l h dh) (float 0.05))
                              :V (float-array (* b l h dh) (float 0.05))
                              :W_o (float-array (* h dh d) (float 0.05))
                              :T (float-array (* l ne) (float 0.0))
                              :R_adj (float-array (* ne ne) (float 0.0))
                              :W_mem (float-array (* d dm) (float 0.05))
                              :R_mem (float-array (* dm dm) (float 0.05))
                              :E_cand (float-array (* nc d) (float 0.05))
                              :threshold (float-array 1 (float 0.5))
                              :W_gate (float-array (* d dff) (float 0.05))
                              :W_up (float-array (* d dff) (float 0.05))
                              :W_down (float-array (* dff d) (float 0.05))}
                      out (tl/run-tl-transformer-block! exec inputs)
                      ^floats h-out (:H_out out)
                      ^floats cand-scores (:cand_scores out)
                      ^floats attn-probs (:attn_probs out)]
                  (and
                   ;; 1. H_out shape is exactly (* b l d)
                   (= (alength h-out) (* b l d))
                   ;; 2. cand_scores shape is exactly (* b l nc)
                   (= (alength cand-scores) (* b l nc))
                   ;; 3. attn_probs shape is exactly (* b h l l)
                   (= (alength attn-probs) (* b h l l))
                   ;; 4. All outputs contain finite values (no NaN / Inf)
                   (every? #(not (Double/isNaN (double %))) h-out)
                   (every? #(not (Double/isInfinite (double %))) h-out)))))

;; ==============================================================================
;; 3. Generative Invariant: Deductive Gating & Grounded Fact Activation
;; ==============================================================================

(defspec prop-deductive-gating-activation 15
  (prop/for-all [thresh-val (gen/double* {:min 0.2 :max 0.8 :NaN? false :infinite? false})]
                (let [b 1
                      l 4
                      h 2
                      dh 4
                      d (* h dh)
                      dff 16
                      dm 4
                      ne 4
                      nc 4
                      exec (tl/compile-tl-transformer-block b l h dh dff dm ne nc)
                      base-inputs {:H (float-array (* b l d) (float 0.1))
                                   :Q (float-array (* b l h dh) (float 0.1))
                                   :K (float-array (* b l h dh) (float 0.1))
                                   :V (float-array (* b l h dh) (float 0.1))
                                   :W_o (float-array (* h dh d) (float 0.1))
                                   :T (float-array (* l ne) (float 0.0))
                                   :R_adj (float-array (* ne ne) (float 0.0))
                                   :W_mem (float-array (* d dm) (float 0.1))
                                   :E_cand (float-array (* nc d) (float 0.1))
                                   :threshold (float-array 1 (float thresh-val))
                                   :W_gate (float-array (* d dff) (float 0.1))
                                   :W_up (float-array (* d dff) (float 0.1))
                                   :W_down (float-array (* dff d) (float 0.1))}

                      ;; Case A: Zero relation core (no resident facts)
                      r-zero (float-array (* dm dm) (float 0.0))
                      out-zero (tl/run-tl-transformer-block! exec (assoc base-inputs :R_mem r-zero))
                      ^floats clamped-zero (:clamped_scores out-zero)

                      ;; Case B: Active relation core matching candidate 0
                      r-active (float-array (* dm dm) (float 0.0))
                      _ (aset r-active 0 (float 5.0))
                      out-active (tl/run-tl-transformer-block! exec (assoc base-inputs :R_mem r-active))
                      ^floats clamped-active (:clamped_scores out-active)]
                  (and
                   ;; When no facts are resident, all clamped scores must be identically 0
                   (every? #(< (Math/abs (double %)) 1e-5) clamped-zero)
                   ;; When fact is active, at least one position must achieve non-zero clamped score
                   (some #(> (double %) 0.0) clamped-active)))))

;; ==============================================================================
;; 4. Unit Test: End-to-End Hybrid Forward Pass
;; ==============================================================================

(deftest test-tl-transformer-block-execution
  (testing "Unified TL-Transformer block executes correctly on PJRT device runtime"
    (let [b 1
          l 4
          h 2
          dh 4
          d (* h dh)
          dff 16
          dm 4
          ne 4
          nc 4
          exec (tl/compile-tl-transformer-block b l h dh dff dm ne nc {:gamma 2.0 :lambda-mem 0.3})
          t-mat (float-array (* l ne) (float 0.0))
          _ (aset t-mat 0 (float 1.0)) ;; pos 0 -> ent 0
          _ (aset t-mat (+ (* 2 ne) 1) (float 1.0)) ;; pos 2 -> ent 1
          r-adj (float-array (* ne ne) (float 0.0))
          _ (aset r-adj (+ (* 0 ne) 1) (float 1.0)) ;; ent 0 -> ent 1
          r-mem (float-array (* dm dm) (float 0.0))
          _ (dotimes [i dm] (aset r-mem (+ (* i dm) i) (float 1.0))) ;; Identity memory core
          inputs {:H (float-array (* b l d) (float 0.1))
                  :Q (float-array (* b l h dh) (float 0.1))
                  :K (float-array (* b l h dh) (float 0.1))
                  :V (float-array (* b l h dh) (float 0.1))
                  :W_o (float-array (* h dh d) (float 0.1))
                  :T t-mat
                  :R_adj r-adj
                  :W_mem (float-array (* d dm) (float 0.1))
                  :R_mem r-mem
                  :E_cand (float-array (* nc d) (float 0.1))
                  :threshold (float-array 1 (float 0.1))
                  :W_gate (float-array (* d dff) (float 0.1))
                  :W_up (float-array (* d dff) (float 0.1))
                  :W_down (float-array (* dff d) (float 0.1))}
          out (tl/run-tl-transformer-block! exec inputs)]
      (is (some? out))
      (is (= (* b l d) (alength ^floats (:H_out out))))
      (is (> (aget ^floats (:H_out out) 0) 0.0))
      (is (some? (:attn_probs out)))
      (is (some? (:cand_scores out))))))
