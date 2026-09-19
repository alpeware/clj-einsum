(ns einsum.logic.attention.kg-masked-test
  "Unit, generative, and invariant tests for Knowledge-Graph Masked Self-Attention
   in OpenXLA PJRT and Pedro Domingos' Declarative Tensor Logic (Experiment E2).
   Tests relational token adjacency generation, causal mask preservation,
   monotonic attention gating, simplex conservation, and adversarial distractor suppression."
  (:require [einsum.logic.attention.kg-masked :as kg-attn]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Pure AST Construction and Schema Tests
;; ==============================================================================

(deftest test-kg-masked-attention-ast-construction
  (testing "kg-token-adjacency-ast generates valid Hiccup block"
    (let [ast (kg-attn/kg-token-adjacency-ast 16 8 {:gamma 2.5})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :kg_token_adjacency (get-in ast [1 :name])))))

  (testing "kg-masked-self-attention-ast generates valid Hiccup block"
    (let [ast (kg-attn/kg-masked-self-attention-ast 1 4 16 64 8 {:gamma 2.5})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :kg_masked_self_attention (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariant: Relational Token Adjacency Projection
;; ==============================================================================

(defspec prop-token-adjacency-exact-projection 20
  (prop/for-all [seq-len (gen/choose 8 16)
                 n-entities (gen/choose 4 8)]
                (let [l (long seq-len)
                      n (long n-entities)
          ;; Pick two distinct positions for related entities
                      pos-h 1
                      pos-t 3
                      ent-h 0
                      ent-t 1
          ;; Construct token-entity matrix T [L, N]
                      t-mat (float-array (* l n))
                      _ (aset t-mat (+ (* pos-h n) ent-h) (float 1.0))
                      _ (aset t-mat (+ (* pos-t n) ent-t) (float 1.0))
          ;; Construct relation core R [N, N] with R[ent-h, ent-t] = 1.0
                      r-mat (float-array (* n n))
                      _ (aset r-mat (+ (* ent-h n) ent-t) (float 1.0))
          ;; Compile and execute token adjacency projection
                      exec (kg-attn/compile-kg-token-adjacency l n {:gamma 1.0})
                      out (kg-attn/compute-kg-token-adjacency! exec t-mat r-mat)
                      ^floats m-kg (:M_kg out)]
                  (and
       ;; 1. Position (pos-h, pos-t) receives exact weight 1.0
                   (< (Math/abs (- (double (aget m-kg (+ (* pos-h l) pos-t))) 1.0)) 1e-4)
       ;; 2. All other positions receive weight 0.0
                   (every? (fn [p-q]
                             (every? (fn [p-k]
                                       (if (and (= p-q pos-h) (= p-k pos-t))
                                         true
                                         (< (Math/abs (double (aget m-kg (+ (* p-q l) p-k)))) 1e-4)))
                                     (range l)))
                           (range l))))))

;; ==============================================================================
;; 3. Generative Invariant: Causality Conservation (Upper Triangle Zeroed)
;; ==============================================================================

(defspec prop-causality-conservation 15
  (prop/for-all [l (gen/choose 8 16)
                 gamma-val (gen/double* {:min 1.0 :max 10.0 :NaN? false :infinite? false})]
                (let [b 1
                      h 2
                      d-k 16
                      n 4
                      q (float-array (* b l h d-k))
                      k-mat (float-array (* b l h d-k))
                      v-mat (float-array (* b l h d-k))
                      t-mat (float-array (* l n))
                      r-mat (float-array (* n n))
          ;; Assert a bidirectional relation between pos 0 and pos 5
                      _ (aset t-mat 0 (float 1.0))       ;; pos 0 -> ent 0
                      _ (aset t-mat (+ (* 5 n) 1) (float 1.0)) ;; pos 5 -> ent 1
                      _ (aset r-mat (+ (* 0 n) 1) (float 1.0)) ;; 0 -> 1
                      _ (aset r-mat (+ (* 1 n) 0) (float 1.0)) ;; 1 -> 0
                      exec (kg-attn/compile-kg-masked-self-attention b h l d-k n {:gamma gamma-val})
                      out (kg-attn/run-kg-masked-self-attention! exec q k-mat v-mat t-mat r-mat)
                      ^floats probs (:probs out)]
      ;; For every query position p_q and key position p_k where p_k > p_q (future):
      ;; Attention probability must be strictly zero (causal masking preserved!)
                  (every? (fn [p-q]
                            (every? (fn [p-k]
                                      (if (> p-k p-q)
                                        (every? (fn [head-idx]
                                                  (let [idx (+ (* head-idx l l) (* p-q l) p-k)]
                                                    (< (Math/abs (double (aget probs idx))) 1e-6)))
                                                (range h))
                                        true))
                                    (range l)))
                          (range l)))))

;; ==============================================================================
;; 4. Generative Invariant: Attention Weights Form a Valid Simplex
;; ==============================================================================

(defspec prop-attention-weights-simplex-conservation 15
  (prop/for-all [l (gen/choose 4 12)
                 d-k (gen/choose 8 16)]
                (let [b 1
                      h 1
                      n 4
                      q (float-array (* b l h d-k))
                      k-mat (float-array (* b l h d-k))
                      v-mat (float-array (* b l h d-k))
                      t-mat (float-array (* l n))
                      r-mat (float-array (* n n))
          ;; Populate Q and K with arbitrary small values
                      _ (dotimes [i (* b l h d-k)]
                          (aset q i (* (float 0.1) (float (Math/sin (double i)))))
                          (aset k-mat i (* (float 0.1) (float (Math/cos (double i))))))
                      exec (kg-attn/compile-kg-masked-self-attention b h l d-k n {:gamma 3.0})
                      out (kg-attn/run-kg-masked-self-attention! exec q k-mat v-mat t-mat r-mat)
                      ^floats probs (:probs out)]
      ;; Every row in probs must sum to 1.0 and each weight >= 0.0
                  (every? (fn [p-q]
                            (let [row-sum (reduce + (map (fn [p-k]
                                                           (let [idx (+ (* p-q l) p-k)]
                                                             (double (aget probs idx))))
                                                         (range (inc p-q))))]
                              (< (Math/abs (- row-sum 1.0)) 1e-4)))
                          (range l)))))

;; ==============================================================================
;; 5. Unit Test: Adversarial Distractor Suppression
;; ==============================================================================

(deftest test-adversarial-distractor-suppression
  (testing "KG mask with gamma >= 3.0 focuses attention onto the true KG relation despite distractor keys"
    (let [b 1
          h 1
          l 4
          d-k 8
          n 4
          ;; Query at pos 3 (e.g. asking about entity 0)
          ;; Target entity 1 at pos 1
          ;; Distractor entity 2 at pos 2 (with large key dot product)
          q (float-array (* b l h d-k))
          k-mat (float-array (* b l h d-k))
          v-mat (float-array (* b l h d-k))
          t-mat (float-array (* l n))
          r-mat (float-array (* n n))
          ;; Token mapping: pos 3 -> ent 0, pos 1 -> ent 1, pos 2 -> ent 2
          _ (aset t-mat (+ (* 3 n) 0) (float 1.0))
          _ (aset t-mat (+ (* 1 n) 1) (float 1.0))
          _ (aset t-mat (+ (* 2 n) 2) (float 1.0))
          ;; KG relation: ent 0 -> ent 1 (target)
          _ (aset r-mat (+ (* 0 n) 1) (float 1.0))
          ;; Distractor key at pos 2 has higher raw dot product with Q(pos 3) than target at pos 1:
          ;; Q(3) = [1, 0...], K(1) = [0.5, 0...], K(2) = [2.0, 0...]
          _ (aset q (+ (* 3 d-k) 0) (float 1.0))
          _ (aset k-mat (+ (* 1 d-k) 0) (float 0.5))
          _ (aset k-mat (+ (* 2 d-k) 0) (float 2.0))
          ;; 1. Without KG mask (gamma = 0): distractor pos 2 dominates
          exec-unmasked (kg-attn/compile-kg-masked-self-attention b h l d-k n {:gamma 0.0})
          out-unmasked (kg-attn/run-kg-masked-self-attention! exec-unmasked q k-mat v-mat t-mat r-mat)
          ^floats probs-unmasked (:probs out-unmasked)
          target-p-unmasked (aget probs-unmasked (+ (* 3 l) 1))
          distractor-p-unmasked (aget probs-unmasked (+ (* 3 l) 2))
          ;; 2. With KG mask (gamma = 5.0): target pos 1 dominates
          exec-masked (kg-attn/compile-kg-masked-self-attention b h l d-k n {:gamma 5.0})
          out-masked (kg-attn/run-kg-masked-self-attention! exec-masked q k-mat v-mat t-mat r-mat)
          ^floats probs-masked (:probs out-masked)
          target-p-masked (aget probs-masked (+ (* 3 l) 1))
          distractor-p-masked (aget probs-masked (+ (* 3 l) 2))]
      ;; Unmasked: distractor > target
      (is (> distractor-p-unmasked target-p-unmasked)
          "Raw unmasked attention should be distracted by the larger key norm")
      ;; Masked: target > distractor with > 90% probability mass
      (is (> target-p-masked distractor-p-masked)
          "KG-masked attention should focus on the true relation")
      (is (> target-p-masked 0.90)
          "KG-masked attention should allocate > 90% attention mass to the true related entity"))))
