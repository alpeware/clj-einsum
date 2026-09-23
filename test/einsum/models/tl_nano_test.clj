(ns einsum.models.tl-nano-test
  "Unit and generative invariant property tests for Native TL-Nano Open-Weights Pre-training (Experiment E9).
   Verifies model configuration, pure AST construction, shape preservation, deductive gating,
   joint LM + InfoNCE loss finiteness, and autodiff gradient descent."
  (:require [einsum.models.tl-nano :as nano]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Pure AST Schema & Construction Invariant Tests
;; ==============================================================================

(deftest test-tl-nano-config-defaults
  (testing "tl-nano-config returns sound defaults and preserves overrides"
    (let [cfg (nano/tl-nano-config)]
      (is (= 1024 (:vocab-size cfg)))
      (is (= 256 (:hidden-dim cfg)))
      (is (= 512 (:intermediate-dim cfg)))
      (is (= 4 (:num-layers cfg)))
      (is (= 4 (:num-heads cfg)))
      (is (= 64 (:head-dim cfg)))
      (is (= #{1 3} (:hybrid-layers cfg))))

    (let [custom (nano/tl-nano-config {:vocab-size 2048 :hidden-dim 128 :num-layers 2})]
      (is (= 2048 (:vocab-size custom)))
      (is (= 128 (:hidden-dim custom)))
      (is (= 2 (:num-layers custom))))))

(deftest test-tl-nano-model-ast-structure
  (testing "tl-nano-model-ast produces a valid Hiccup AST vector"
    (let [cfg (nano/tl-nano-config {:num-layers 2 :hybrid-layers #{1}})
          ast (nano/tl-nano-model-ast cfg)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :tl_nano_model (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariant: Shape Preservation Across Batch & Sequence Dimensions
;; ==============================================================================

(defspec prop-tl-nano-shape-invariants 10
  (prop/for-all [b-val (gen/choose 1 2)
                 l-val (gen/choose 4 8)
                 h-val (gen/choose 1 2)
                 dh-val (gen/choose 4 8)]
                (let [b (long b-val)
                      l (long l-val)
                      h (long h-val)
                      dh (long dh-val)
                      d (* h dh)
                      v 32
                      dff (* 2 d)
                      dm 4
                      ne 4
                      nc 4
                      cfg (nano/tl-nano-config
                           {:vocab-size v
                            :hidden-dim d
                            :intermediate-dim dff
                            :num-layers 2
                            :num-heads h
                            :head-dim dh
                            :dim-mem dm
                            :entity-count ne
                            :cand-count nc
                            :hybrid-layers #{1}
                            :max-seq-len l})
                      exec (nano/compile-tl-nano-forward b l cfg)
                      w-embed (float-array (* v d) (float 0.05))
                      x-tokens (int-array (take (* b l) (cycle (range v))))
                      q-w (float-array (* d d) (float 0.05))
                      k-w (float-array (* d d) (float 0.05))
                      v-w (float-array (* d d) (float 0.05))
                      o-w (float-array (* d d) (float 0.05))
                      gate-w (float-array (* d dff) (float 0.05))
                      up-w (float-array (* d dff) (float 0.05))
                      down-w (float-array (* dff d) (float 0.05))
                      w-mem (float-array (* d dm) (float 0.05))
                      r-mem (float-array (* dm dm) (float 0.05))
                      e-cand (float-array (* nc d) (float 0.05))
                      t-ind (float-array (* l ne) (float 0.0))
                      r-adj (float-array (* ne ne) (float 0.0))
                      thresh (float-array 1 (float 0.5))

                      inputs (into {:x x-tokens
                                    :W_embed w-embed
                                    :W_mem w-mem
                                    :R_mem r-mem
                                    :E_cand e-cand
                                    :T t-ind
                                    :R_adj r-adj
                                    :threshold thresh}
                                   (mapcat
                                    (fn [i]
                                      [[(keyword (str "W_q_" i)) q-w]
                                       [(keyword (str "W_k_" i)) k-w]
                                       [(keyword (str "W_v_" i)) v-w]
                                       [(keyword (str "W_o_" i)) o-w]
                                       [(keyword (str "W_gate_" i)) gate-w]
                                       [(keyword (str "W_up_" i)) up-w]
                                       [(keyword (str "W_down_" i)) down-w]])
                                    (range 2)))
                      outputs (nano/run-tl-nano-forward! exec inputs)
                      ^floats logits (:logits outputs)
                      ^floats h-final (:H_final outputs)]
                  (and
                   ;; 1. Logits dimension is exactly [B, L, V]
                   (= (alength logits) (* b l v))
                   ;; 2. Final hidden state is exactly [B, L, D]
                   (= (alength h-final) (* b l d))
                   ;; 3. No NaN / Inf in any output
                   (every? #(not (Double/isNaN (double %))) logits)
                   (every? #(not (Double/isInfinite (double %))) logits)
                   (every? #(not (Double/isNaN (double %))) h-final)
                   (every? #(not (Double/isInfinite (double %))) h-final)))))

;; ==============================================================================
;; 3. Generative Invariant: Deductive Grounding & Semiring Gating
;; ==============================================================================

(defspec prop-tl-nano-deductive-grounding 10
  (prop/for-all [thresh-val (gen/double* {:min 0.2 :max 0.7 :NaN? false :infinite? false})]
                (let [b 1
                      l 4
                      h 2
                      dh 4
                      d (* h dh)
                      v 16
                      dff 16
                      dm 4
                      ne 4
                      nc 4
                      cfg (nano/tl-nano-config
                           {:vocab-size v
                            :hidden-dim d
                            :intermediate-dim dff
                            :num-layers 2
                            :num-heads h
                            :head-dim dh
                            :dim-mem dm
                            :entity-count ne
                            :cand-count nc
                            :hybrid-layers #{1}
                            :max-seq-len l
                            :threshold thresh-val})
                      exec (nano/compile-tl-nano-forward b l cfg)
                      base-inputs {:x (int-array [0 1 2 3])
                                   :W_embed (float-array (map #(float (* 0.05 (inc %))) (range (* v d))))
                                   :W_q_0 (float-array (* d d) (float 0.1))
                                   :W_k_0 (float-array (* d d) (float 0.1))
                                   :W_v_0 (float-array (* d d) (float 0.1))
                                   :W_o_0 (float-array (* d d) (float 0.1))
                                   :W_gate_0 (float-array (* d dff) (float 0.1))
                                   :W_up_0 (float-array (* d dff) (float 0.1))
                                   :W_down_0 (float-array (* dff d) (float 0.1))
                                   :W_q_1 (float-array (* d d) (float 0.1))
                                   :W_k_1 (float-array (* d d) (float 0.1))
                                   :W_v_1 (float-array (* d d) (float 0.1))
                                   :W_o_1 (float-array (* d d) (float 0.1))
                                   :W_gate_1 (float-array (* d dff) (float 0.1))
                                   :W_up_1 (float-array (* d dff) (float 0.1))
                                   :W_down_1 (float-array (* dff d) (float 0.1))
                                   :T (float-array (* l ne) (float 0.0))
                                   :R_adj (float-array (* ne ne) (float 0.0))
                                   :W_mem (float-array (* d dm) (float 0.1))
                                   :E_cand (float-array (map #(float (* 0.1 (inc %))) (range (* nc d))))
                                   :threshold (float-array 1 (float thresh-val))}

                      ;; Case A: R_mem = 0 (no resident facts)
                      out-zero (nano/run-tl-nano-forward! exec (assoc base-inputs :R_mem (float-array (* dm dm) (float 0.0))))
                      ^floats logits-zero (:logits out-zero)

                      ;; Case B: R_mem = strong active relational core
                      out-active (nano/run-tl-nano-forward! exec (assoc base-inputs :R_mem (float-array (* dm dm) (float 2.5))))
                      ^floats logits-active (:logits out-active)

                      diff-norm (loop [i 0 s 0.0]
                                  (if (>= i (alength logits-zero))
                                    (Math/sqrt s)
                                    (let [d (- (double (aget logits-active i)) (double (aget logits-zero i)))]
                                      (recur (inc i) (+ s (* d d))))))]
                  ;; Active relational memory strictly shifts downstream logits
                  (> diff-norm 1e-4))))

;; ==============================================================================
;; 4. Pre-training Step & Joint Loss Descent
;; ==============================================================================

(deftest test-tl-nano-joint-pretraining-step
  (testing "A single pre-training step with joint LM + InfoNCE loss descends monotonically"
    (let [b 1
          l 4
          h 1
          dh 4
          d (* h dh)
          v 8
          dff 8
          dm 4
          ne 2
          nc 2
          cfg (nano/tl-nano-config
               {:vocab-size v
                :hidden-dim d
                :intermediate-dim dff
                :num-layers 1
                :num-heads h
                :head-dim dh
                :dim-mem dm
                :entity-count ne
                :cand-count nc
                :hybrid-layers #{0}
                :max-seq-len l
                :lambda-tl 0.5
                :tau 0.2})
          batch {:tokens (int-array [0 1 2 3])
                 :targets (int-array [1 2 3 0])
                 :triples [{:head 0 :rel 0 :tail 1}]}
          params (nano/init-tl-nano-params cfg (java.util.Random. 42))
          fwd-exec (nano/compile-tl-nano-forward b l cfg)

          ;; Evaluate initial loss
          loss-0 (nano/compute-joint-loss fwd-exec params batch cfg)
          _ (is (pos? (:total-loss loss-0)))
          _ (is (pos? (:lm-loss loss-0)))

          ;; Run a single SGD step
          updated-params (nano/train-step fwd-exec params batch cfg 0.1)
          loss-1 (nano/compute-joint-loss fwd-exec updated-params batch cfg)]
      ;; Total loss decreases after gradient step
      (is (<= (:total-loss loss-1) (:total-loss loss-0))))))

;; ==============================================================================
;; 5. Real-World Knowledge Corpus & Batching Tests
;; ==============================================================================

(deftest test-real-data-corpus-structure-and-batching
  (testing "Wikifacts corpus EDN file is well-formed with required entity, relation, triple, and prompt keys"
    (let [corpus (read-string (slurp (io/resource "data/wikifacts_corpus.edn")))]
      (is (vector? (:entities corpus)))
      (is (vector? (:relations corpus)))
      (is (vector? (:triples corpus)))
      (is (vector? (:sentences corpus)))
      (is (vector? (:eval_prompts corpus)))
      (is (>= (count (:entities corpus)) 20))
      (is (>= (count (:triples corpus)) 20))
      (is (>= (count (:sentences corpus)) 50))
      (is (>= (count (:eval_prompts corpus)) 15))
      ;; Verify prompt structure
      (doseq [ep (:eval_prompts corpus)]
        (is (string? (:prompt ep)))
        (is (string? (:target ep)))
        (is (string? (:head ep)))
        (is (keyword? (:rel ep)))))))

