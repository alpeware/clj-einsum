(ns einsum.logic.gather-test
  "Generative property and unit tests for generalized Tensor Logic gather and log-softmax operations."
  (:require [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.compiler.stablehlo :as shlo]
            [einsum.logic.interpret :as interp]
            [einsum.logic.lower :as lower]))

;; =============================================================================
;; 1. StableHLO Interpreter Gather (Direct op-gather 3D Target Gathering)
;; =============================================================================

(deftest test-gather-3d-target-elements
  ;; operand [1 4 5] (batch 1, seq 4, vocab 5), coords [1 4 3] (3 coords: b, p, v)
  (let [operand (interp/tensor :f32 [1 4 5] (mapv float (range 20)))
        ;; For each position p in 0..3, target is [2, 0, 4, 1]
        ;; Coords: [b p v] for each position:
        ;; [0 0 2] -> 2.0
        ;; [0 1 0] -> 5.0
        ;; [0 2 4] -> 14.0
        ;; [0 3 1] -> 16.0
        coords (interp/tensor :i32 [1 4 3] [0 0 2, 0 1 0, 0 2 4, 0 3 1])
        r (interp/op-gather operand coords
                            {:offset_dims []
                             :collapsed_slice_dims [0 1 2]
                             :start_index_map [0 1 2]
                             :index_vector_dim 2
                             :slice_sizes [1 1 1]})]
    (is (= [1 4] (:shape r)))
    (is (= [2.0 5.0 14.0 16.0] (mapv float (vec (:data r)))))))

(defspec prop-gather-3d-target-elements-matches-clojure-indexing 50
  (prop/for-all [p-len (gen/choose 1 16)
                 vocab (gen/choose 2 32)]
                (let [n (* p-len vocab)
                      vals (mapv float (range n))
                      operand (interp/tensor :f32 [1 p-len vocab] vals)
                      targets (mapv #(mod (* % 7) vocab) (range p-len))
                      coords-flat (vec (mapcat (fn [p] [0 p (nth targets p)]) (range p-len)))
                      coords (interp/tensor :i32 [1 p-len 3] coords-flat)
                      r (interp/op-gather operand coords
                                          {:offset_dims []
                                           :collapsed_slice_dims [0 1 2]
                                           :start_index_map [0 1 2]
                                           :index_vector_dim 2
                                           :slice_sizes [1 1 1]})
                      expected (mapv (fn [p] (float (+ (* p vocab) (nth targets p)))) (range p-len))]
                  (and (= [1 p-len] (:shape r))
                       (= expected (mapv float (vec (:data r))))))))

;; =============================================================================
;; 2. Fused Log-Softmax Tests (Pure JVM)
;; =============================================================================

(deftest test-fused-log-softmax-numerical-sanity
  ;; For input [0.0 0.0], softmax is [0.5 0.5], log-softmax is [log(0.5) log(0.5)] ≈ [-0.693147 -0.693147]
  (let [t (interp/tensor :f32 [1 2] [0.0 0.0])
        r (interp/op-fused-log-softmax t {})
        expected (float (Math/log 0.5))]
    (is (= [1 2] (:shape r)))
    (is (< (Math/abs (- (aget ^floats (:data r) 0) expected)) 1e-5))
    (is (< (Math/abs (- (aget ^floats (:data r) 1) expected)) 1e-5))))

(defspec prop-fused-log-softmax-properties 50
  (prop/for-all [seq-len (gen/choose 1 8)
                 vocab-dim (gen/choose 2 16)]
                (let [n (* seq-len vocab-dim)
                      vals (mapv #(float (- (mod (* % 13) 20) 10)) (range n))
                      t (interp/tensor :f32 [1 seq-len vocab-dim] vals)
                      r (interp/op-fused-log-softmax t {})
                      data (vec (:data r))]
                  (and (= [1 seq-len vocab-dim] (:shape r))
           ;; Property 1: All log-probabilities are <= 0.0 (probability <= 1.0)
                       (every? #(<= % 0.0) data)
           ;; Property 2: No NaNs or infinities
                       (every? #(not (Double/isNaN (double %))) data)
                       (every? #(not (Double/isInfinite (double %))) data)
           ;; Property 3: exp(log-prob) sums to ~1.0 for each slice
                       (every? (fn [p]
                                 (let [slice (subvec data (* p vocab-dim) (* (inc p) vocab-dim))
                                       sum-exp (reduce + (map #(Math/exp (double %)) slice))]
                                   (< (Math/abs (- sum-exp 1.0)) 1e-4)))
                               (range seq-len))))))

;; =============================================================================
;; 3. Tensor Logic Lowering for Axis-2 Gather
;; =============================================================================

(deftest test-lower-axis-2-gather-ast
  (let [invars [[:logits [:tensor [1 4 10] :f32]]
                [:targets [:tensor [1 4] :i32]]]
        ast [:block {}
             [:log-softmax [:log_probs :b :p :v] [:logits :b :p :v] {:axis 2}]
             [:gather [:target_log_probs :b :p] [:log_probs :b :p :v] [:targets :b :p] {:axis 2}]]
        graph (lower/ast->graph "scoring_test" invars ast #{:target_log_probs})]
    (is (shlo/validate-graph graph))
    (is (= [:target_log_probs] (:outvars graph)))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/gather %) ops))
      (is (some #(= :stablehlo/log %) ops))
      (is (some #(= :stablehlo/concatenate %) ops)))))
