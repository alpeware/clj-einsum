(ns einsum.logic.lower-test
  "Generative property and unit tests for Tensor Logic StableHLO lowering."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-lower-gemm-produces-valid-ssa-graph
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:w [:tensor [32 64] :f32]]]
        ast [:= [:y :b :m :n] [:x :b :m :k] [:w :k :n]]
        graph (lower/ast->graph "gemm_graph" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (is (= [:y] (:outvars graph)))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/dot_general %) ops)))))

(deftest test-lower-implicit-accumulation
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:y [:tensor [2 16 32] :f32]]]
        ast [:block {}
             [:= [:out :b :m :d] [:x :b :m :d]]
             [:= [:out :b :m :d] [:y :b :m :d]]]
        graph (lower/ast->graph "accum_graph" invars ast #{:out})]
    (is (shlo/validate-graph graph))
    (let [ops (mapv :op (:eqns graph))]
      ;; Implicit accumulation emits an addition equation
      (is (some #(= :stablehlo/add %) ops)))))

(deftest test-lower-post-activation-scale
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:w [:tensor [32 64] :f32]]]
        ast [:= [:y :b :m :n] {:scale 0.5} [:x :b :m :k] [:w :k :n]]
        graph (lower/ast->graph "scaled_gemm" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/multiply %) ops)))))

(deftest test-lower-post-activation-clamp
  (let [invars [[:x [:tensor [2 16 32] :f32]]
                [:w [:tensor [32 64] :f32]]]
        ast [:= [:y :b :m :n] {:clamp [0.0 1.0]} [:x :b :m :k] [:w :k :n]]
        graph (lower/ast->graph "clamped_gemm" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/maximum %) ops))
      (is (some #(= :stablehlo/minimum %) ops)))))

(defspec prop-lower-clamp-activation-invariants
  30
  (prop/for-all [low-int (gen/choose -10 0)
                 range-int (gen/choose 1 10)]
                (let [low (double low-int)
                      high (+ low (double range-int))
                      ctx (xla/get-context)
                      invars [[:x [:tensor [5] :f32]]]
                      ast [:= [:y :i] {:clamp [low high]} [:x :i]]
                      graph (lower/ast->graph "clamp_test" invars ast #{:y})
                      compiled (xla/compile-graph ctx graph)
                      test-inputs (float-array [(- low 5.0) low (+ low (/ (- high low) 2.0)) high (+ high 5.0)])
                      out-buf (xla/execute compiled test-inputs)
                      res (xla/to-host-slice out-buf 0 5 4)
                      _ (xla/destroy-buffer! out-buf)]
                  (and (<= (nth res 0) low (+ (nth res 0) 1e-5))
                       (<= (Math/abs (- (nth res 1) low)) 1e-5)
                       (<= (Math/abs (- (nth res 2) (+ low (/ (- high low) 2.0)))) 1e-5)
                       (<= (Math/abs (- (nth res 3) high)) 1e-5)
                       (<= (nth res 4) high (+ (nth res 4) 1e-5))))))

(deftest test-lower-semiring-min-plus-contraction
  (testing "Lowering and PJRT execution of MIN_PLUS (tropical) semiring contraction"
    (let [ctx (xla/get-context)
          invars [[:A [:tensor [2 2] :f32]]
                  [:B [:tensor [2 2] :f32]]]
          ast [:= [:C :i :j] {:semiring :min-plus} [:A :i :k] [:B :k :j]]
          graph (lower/ast->graph "min_plus_graph" invars ast #{:C})
          _ (is (shlo/validate-graph graph))
          ops (mapv :op (:eqns graph))
          _ (is (some #(= :stablehlo/reduce_min %) ops))
          compiled (xla/compile-graph ctx graph)
          a-data (float-array [0.0 2.0
                               5.0 0.0])
          b-data (float-array [0.0 3.0
                               1.0 0.0])
          out-buf (xla/execute compiled a-data b-data)
          res (vec (seq ^floats (xla/to-host-slice out-buf 0 4 4)))
          _ (xla/destroy-buffer! out-buf)]
      ;; Expected C:
      ;; C[0,0] = min(0+0, 2+1) = 0.0
      ;; C[0,1] = min(0+3, 2+0) = 2.0
      ;; C[1,0] = min(5+0, 0+1) = 1.0
      ;; C[1,1] = min(5+3, 0+0) = 0.0
      (is (= [0.0 2.0 1.0 0.0] res)))))

(deftest test-lower-semiring-max-product-contraction
  (testing "Lowering and PJRT execution of MAX_PRODUCT (Viterbi) semiring contraction"
    (let [ctx (xla/get-context)
          invars [[:A [:tensor [2 2] :f32]]
                  [:B [:tensor [2 2] :f32]]]
          ast [:= [:C :i :j] {:semiring :max-product} [:A :i :k] [:B :k :j]]
          graph (lower/ast->graph "max_prod_graph" invars ast #{:C})
          _ (is (shlo/validate-graph graph))
          ops (mapv :op (:eqns graph))
          _ (is (some #(= :stablehlo/reduce_max %) ops))
          compiled (xla/compile-graph ctx graph)
          a-data (float-array [0.8 0.2
                               0.1 0.9])
          b-data (float-array [0.7 0.3
                               0.4 0.6])
          out-buf (xla/execute compiled a-data b-data)
          res (vec (seq ^floats (xla/to-host-slice out-buf 0 4 4)))
          _ (xla/destroy-buffer! out-buf)]
      ;; Expected C:
      ;; C[0,0] = max(0.8*0.7=0.56, 0.2*0.4=0.08) = 0.56
      ;; C[0,1] = max(0.8*0.3=0.24, 0.2*0.6=0.12) = 0.24
      ;; C[1,0] = max(0.1*0.7=0.07, 0.9*0.4=0.36) = 0.36
      ;; C[1,1] = max(0.1*0.3=0.03, 0.9*0.6=0.54) = 0.54
      (is (< (Math/abs (- (nth res 0) 0.56)) 1e-5))
      (is (< (Math/abs (- (nth res 1) 0.24)) 1e-5))
      (is (< (Math/abs (- (nth res 2) 0.36)) 1e-5))
      (is (< (Math/abs (- (nth res 3) 0.54)) 1e-5)))))

(deftest test-end-to-end-pjrt-execution-parity
  (let [ctx (xla/get-context)
        invars [[:x [:tensor [2 4] :f32]]
                [:w [:tensor [4 2] :f32]]]
        ast [:= [:y :m :n] [:x :m :k] [:w :k :n]]
        graph (lower/ast->graph "e2e_gemm" invars ast #{:y})
        compiled (xla/compile-graph ctx graph)
        x-data (float-array [1.0 2.0 3.0 4.0
                             5.0 6.0 7.0 8.0])
        w-data (float-array [1.0 0.0
                             0.0 1.0
                             1.0 0.0
                             0.0 1.0])
        out-buf (xla/execute compiled x-data w-data)
        res (xla/to-host-slice out-buf 0 4 4)]
    ;; Row 0: [1+3, 2+4] = [4.0, 6.0]
    ;; Row 1: [5+7, 6+8] = [12.0, 14.0]
    (is (= 4.0 (nth res 0)))
    (is (= 6.0 (nth res 1)))
    (is (= 12.0 (nth res 2)))
    (is (= 14.0 (nth res 3)))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-dynamic-slice-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 4)
                 s (gen/choose 16 128)
                 d (gen/choose 32 256)]
                (let [invars [[:x [:tensor [b s d] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:dynamic-slice [:y :b :one :d] [:x :b :p :d]
                           {:slice-sizes [1 1 d] :start-indices [0 :pos 0]}]
                      graph (lower/ast->graph "ds_graph" invars ast #{:y})]
                  (and (shlo/validate-graph graph)
                       (= [:y] (:outvars graph))
                       (boolean (some #(= :stablehlo/dynamic_slice (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-dynamic-slice-execution
  (let [ctx (xla/get-context)
        invars [[:x [:tensor [1 10 4] :f32]]
                [:pos [:tensor [1] :i32]]]
        ast [:dynamic-slice [:y :b :one :d] [:x :b :p :d]
             {:slice-sizes [1 1 4] :start-indices [0 :pos 0]}]
        graph (lower/ast->graph "e2e_ds" invars ast #{:y})
        compiled (xla/compile-graph ctx graph)
        x-data (float-array (range 40))
        pos-data (int-array [3])
        out-buf (xla/execute compiled x-data pos-data)
        res (xla/to-host-slice out-buf 0 4 4)]
    ;; At pos=3, elements for slice [1 1 4] starting at [0 3 0] are [12.0, 13.0, 14.0, 15.0]
    (is (= 12.0 (nth res 0)))
    (is (= 13.0 (nth res 1)))
    (is (= 14.0 (nth res 2)))
    (is (= 15.0 (nth res 3)))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [1] :i32]]]
                      ast [:while [:final_step] [:init]
                           {:cond-mlir "    ^bb0(%s: tensor<1xi32>):\n      %c = stablehlo.constant dense<10> : tensor<1xi32>\n      %cmp = \"stablehlo.compare\"(%s, %c) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>\n      %res = stablehlo.reshape %cmp : (tensor<1xi1>) -> tensor<i1>\n      \"stablehlo.return\"(%res) : (tensor<i1>) -> ()"
                            :body-mlir "    ^bb0(%s: tensor<1xi32>):\n      %one = stablehlo.constant dense<1> : tensor<1xi32>\n      %next = stablehlo.add %s, %one : tensor<1xi32>\n      \"stablehlo.return\"(%next) : (tensor<1xi32>) -> ()"}]
                      graph (lower/ast->graph "while_graph" invars ast #{:final_step})]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [1] :i32]]]
        ast [:while [:final_step] [:init]
             {:cond-mlir "    ^bb0(%s: tensor<1xi32>):\n      %c = stablehlo.constant dense<5> : tensor<1xi32>\n      %cmp = \"stablehlo.compare\"(%s, %c) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>\n      %res = stablehlo.reshape %cmp : (tensor<1xi1>) -> tensor<i1>\n      \"stablehlo.return\"(%res) : (tensor<i1>) -> ()"
              :body-mlir "    ^bb0(%s: tensor<1xi32>):\n      %one = stablehlo.constant dense<1> : tensor<1xi32>\n      %next = stablehlo.add %s, %one : tensor<1xi32>\n      \"stablehlo.return\"(%next) : (tensor<1xi32>) -> ()"}]
        graph (lower/ast->graph "e2e_while" invars ast #{:final_step})
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        out-buf (xla/execute compiled init-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 5 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-with-cond-ast-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [] :i32]]
                              [:max_step [:tensor [] :i32]]
                              [:false_c [:tensor [] :i1]]]
                      cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                                [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                                [:not [:not_stopped] [:cur_stopped]]
                                [:and [:cond_out] [:step_lt] [:not_stopped]]]
                      loop-ast [:while [:final_step :final_max :final_stopped]
                                [:init :max_step :false_c]
                                {:body-mlir "    ^bb0(%s: tensor<i32>, %m: tensor<i32>, %st: tensor<i1>):\n      \"stablehlo.return\"(%s, %m, %st) : (tensor<i32>, tensor<i32>, tensor<i1>) -> ()"}
                                cond-ast]
                      graph (lower/ast->graph "while_cond_graph" invars loop-ast [:final_step])]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-cond-ast-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [] :i32]]
                [:max_step [:tensor [] :i32]]
                [:false_c [:tensor [] :i1]]]
        cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]
        loop-ast [:while [:final_step :final_max :final_stopped]
                  [:init :max_step :false_c]
                  {:body-mlir "    ^bb0(%s: tensor<i32>, %m: tensor<i32>, %st: tensor<i1>):\n      %one = stablehlo.constant dense<1> : tensor<i32>\n      %next = stablehlo.add %s, %one : tensor<i32>\n      \"stablehlo.return\"(%next, %m, %st) : (tensor<i32>, tensor<i32>, tensor<i1>) -> ()"}
                  cond-ast]
        graph (lower/ast->graph "e2e_while_cond" invars loop-ast [:final_step])
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        max-data (int-array [7])
        false-data (byte-array [0])
        out-buf (xla/execute compiled init-data max-data false-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 7 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-argmax-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 4)
                 v (gen/choose 16 128)]
                (let [invars [[:logits [:tensor [b v] :f32]]]
                      ast [:argmax [:out] [:logits] {:axis 1}]
                      graph (lower/ast->graph "argmax_graph" invars ast #{:out})]
                  (and (shlo/validate-graph graph)
                       (= [:out] (:outvars graph))
                       (boolean (some #(= :stablehlo/argmax (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-argmax-execution
  (let [ctx (xla/get-context)
        invars [[:logits [:tensor [2 4] :f32]]]
        ast [:argmax [:out] [:logits] {:axis 1}]
        graph (lower/ast->graph "e2e_argmax" invars ast #{:out})
        compiled (xla/compile-graph ctx graph)
        logits-data (float-array [1.0 9.0 2.0 3.0
                                  0.5 1.5 8.5 2.5])
        out-buf (xla/execute compiled logits-data)
        res (xla/to-host-slice out-buf 0 2 8)
        idx0 (Float/floatToIntBits (nth res 0))
        idx1 (Float/floatToIntBits (nth res 1))]
    (is (= 1 idx0))
    (is (= 2 idx1))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-while-with-body-ast-produces-valid-graph
  50
  (prop/for-all [_lim (gen/choose 1 100)]
                (let [invars [[:init [:tensor [] :i32]]
                              [:max_step [:tensor [] :i32]]
                              [:false_c [:tensor [] :i1]]]
                      cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                                [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                                [:not [:not_stopped] [:cur_stopped]]
                                [:and [:cond_out] [:step_lt] [:not_stopped]]]
                      body-ast [:body [:next_step :target_max :cur_stopped]
                                {:args [:cur_step :target_max :cur_stopped]}
                                [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                                [:+ [:next_step] [:cur_step] [:c_one]]]
                      loop-ast [:while [:final_step :final_max :final_stopped]
                                [:init :max_step :false_c]
                                cond-ast
                                body-ast]
                      graph (lower/ast->graph "while_body_graph" invars loop-ast [:final_step])]
                  (and (shlo/validate-graph graph)
                       (= [:final_step] (:outvars graph))
                       (boolean (some #(= :stablehlo/while (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-while-declarative-body-execution
  (let [ctx (xla/get-context)
        invars [[:init [:tensor [] :i32]]
                [:max_step [:tensor [] :i32]]
                [:false_c [:tensor [] :i1]]]
        cond-ast [:cond [:cond_out] {:args [:cur_step :target_max :cur_stopped]}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]
        body-ast [:body [:next_step :target_max :cur_stopped]
                  {:args [:cur_step :target_max :cur_stopped]}
                  [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                  [:+ [:next_step] [:cur_step] [:c_one]]]
        loop-ast [:while [:final_step :final_max :final_stopped]
                  [:init :max_step :false_c]
                  cond-ast
                  body-ast]
        graph (lower/ast->graph "e2e_while_body" invars loop-ast [:final_step])
        compiled (xla/compile-graph ctx graph)
        init-data (int-array [0])
        max-data (int-array [10])
        false-data (byte-array [0])
        out-buf (xla/execute compiled init-data max-data false-data)
        res (xla/to-host-slice out-buf 0 1 4)
        val (Float/floatToIntBits (first res))]
    (is (= 10 val))
    (xla/destroy-buffer! out-buf)))

(defspec prop-lower-dynamic-update-slice-produces-valid-graph
  50
  (prop/for-all [b (gen/choose 1 2)
                 s (gen/choose 8 32)
                 d (gen/choose 16 64)]
                (let [invars [[:cache [:tensor [b s d] :f32]]
                              [:new_slice [:tensor [b 1 d] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:dynamic-update-slice [:out :b :s :d] [:cache :b :s :d] [:new_slice :b :one :d]
                           {:start-indices [0 :pos 0]}]
                      graph (lower/ast->graph "dus_graph" invars ast #{:out})]
                  (and (shlo/validate-graph graph)
                       (= [:out] (:outvars graph))
                       (boolean (some #(= :stablehlo/dynamic_update_slice (:op %)) (:eqns graph)))))))

(deftest test-end-to-end-dynamic-update-slice-execution
  (let [ctx (xla/get-context)
        invars [[:cache [:tensor [1 4 4] :f32]]
                [:new_slice [:tensor [1 1 4] :f32]]
                [:pos [:tensor [1] :i32]]]
        ast [:dynamic-update-slice [:out :b :s :d] [:cache :b :s :d] [:new_slice :b :one :d]
             {:start-indices [0 :pos 0]}]
        graph (lower/ast->graph "e2e_dus" invars ast #{:out})
        compiled (xla/compile-graph ctx graph)
        cache-data (float-array (repeat 16 0.0))
        slice-data (float-array [9.0 8.0 7.0 6.0])
        pos-data (int-array [2])
        out-buf (xla/execute compiled cache-data slice-data pos-data)
        res (xla/to-host-slice out-buf 0 16 16)]
    ;; Position 2 row in [1 4 4] should be [9.0 8.0 7.0 6.0] (indices 8, 9, 10, 11)
    (is (= 0.0 (nth res 0)))
    (is (= 9.0 (nth res 8)))
    (is (= 8.0 (nth res 9)))
    (is (= 7.0 (nth res 10)))
    (is (= 6.0 (nth res 11)))
    (is (= 0.0 (nth res 12)))
    (xla/destroy-buffer! out-buf)))

(deftest test-binary-contraction-subset-broadcast
  (testing "Binary contraction automatically broadcasts secondary operand when its indices are a subset of primary operand."
    (let [ctx (xla/get-context)
          invars [[:w [:tensor [2 3] :f32]]
                  [:scale [:tensor [2] :f32]]]
          ast [:= [:w_scaled :n :k] [:w :n :k] [:scale :n]]
          graph (lower/ast->graph "broadcast_mul" invars ast #{:w_scaled})
          _ (is (shlo/validate-graph graph))
          compiled (xla/compile-graph ctx graph)
          w-data (float-array [1.0 2.0 3.0
                               4.0 5.0 6.0])
          scale-data (float-array [2.0 0.5])
          out-buf (xla/execute compiled w-data scale-data)
          res (vec (xla/to-host-slice out-buf 0 6 4))]
      ;; Row 0 scaled by 2.0: [2.0, 4.0, 6.0]
      ;; Row 1 scaled by 0.5: [2.0, 2.5, 3.0]
      (is (= [2.0 4.0 6.0 2.0 2.5 3.0] res))
      (xla/destroy-buffer! out-buf))))

(defn- reference-rope
  [x-vec batch seq-len n-heads head-dim rope-prop theta pos-offset]
  (let [half-dim (quot head-dim 2)
        rope-angles (long (* rope-prop half-dim))
        freqs (mapv (fn [i]
                      (if (< i rope-angles)
                        (Math/pow theta (/ (* -2.0 i) (double head-dim)))
                        0.0))
                    (range half-dim))]
    (vec
     (for [b (range batch)
           s (range seq-len)
           h (range n-heads)
           d (range head-dim)]
       (let [pos (+ pos-offset s)
             rot-half (if (< d half-dim) d (- d half-dim))
             freq (nth freqs rot-half)
             angle (* (double pos) freq)
             cos-val (Math/cos angle)
             sin-val (Math/sin angle)
             idx (+ (* b seq-len n-heads head-dim)
                    (* s n-heads head-dim)
                    (* h head-dim)
                    d)
             cur-val (double (nth x-vec idx))]
         (if (< d half-dim)
           (let [pair-idx (+ idx half-dim)
                 pair-val (double (nth x-vec pair-idx))]
             (- (* cur-val cos-val) (* pair-val sin-val)))
           (let [pair-idx (- idx half-dim)
                 pair-val (double (nth x-vec pair-idx))]
             (+ (* cur-val cos-val) (* pair-val sin-val)))))))))

(defspec prop-lower-rope-produces-valid-graph
  30
  (prop/for-all [b (gen/elements [1 2])
                 seq-len (gen/elements [1 4 8])
                 head-dim (gen/elements [32 64 128])
                 n-heads (gen/elements [1 2 4])
                 rope-prop (gen/elements [1.0 0.5 0.25])]
                (let [total-dim (* n-heads head-dim)
                      dynamic? (= seq-len 1)
                      invars (if dynamic?
                               [[:x [:tensor [b seq-len total-dim] :f32]]
                                [:pos [:tensor [1] :i32]]]
                               [[:x [:tensor [b seq-len total-dim] :f32]]])
                      attrs (cond-> {:head-dim head-dim
                                     :theta 10000.0
                                     :rope-proportion rope-prop}
                              dynamic? (assoc :pos :pos :max-seq-len 64))
                      ast [:rope [:y :b :p :d] [:x :b :p :d] attrs]
                      graph (lower/ast->graph "rope_test_graph" invars ast #{:y})]
                  (and (shlo/validate-graph graph)
                       (= [:y] (:outvars graph))
                       (boolean (some #(= :stablehlo/multiply (:op %)) (:eqns graph)))))))

(deftest test-rope-parity-with-proportional-reference
  (let [ctx (xla/get-context)
        batch 1
        seq-len 4
        n-heads 2
        head-dim 8
        total-dim (* n-heads head-dim)
        rope-prop 0.5
        theta 10000.0
        invars [[:x [:tensor [batch seq-len total-dim] :f32]]]
        ast [:rope [:y :b :p :d] [:x :b :p :d]
             {:head-dim head-dim :rope-proportion rope-prop :theta theta}]
        graph (lower/ast->graph "rope_e2e_static" invars ast #{:y})
        _ (is (shlo/validate-graph graph))
        compiled (xla/compile-graph ctx graph)
        x-raw (vec (map #(float (inc %)) (range (* batch seq-len total-dim))))
        x-data (float-array x-raw)
        out-buf (xla/execute compiled x-data)
        pjrt-res (vec (xla/to-host-slice out-buf 0 (* batch seq-len total-dim) 4))
        ref-res (reference-rope x-raw batch seq-len n-heads head-dim rope-prop theta 0)]
    (doseq [i (range (count x-raw))]
      (is (< (Math/abs (- (double (nth pjrt-res i)) (double (nth ref-res i)))) 1e-4)
          (str "Mismatch at index " i " pjrt=" (nth pjrt-res i) " ref=" (nth ref-res i))))
    (xla/destroy-buffer! out-buf))

  (testing "Dynamic KV-Cache step RoPE parity"
    (let [ctx (xla/get-context)
          batch 1
          seq-len 1
          n-heads 2
          head-dim 8
          total-dim (* n-heads head-dim)
          rope-prop 0.5
          theta 10000.0
          pos-val 7
          invars [[:x [:tensor [batch seq-len total-dim] :f32]]
                  [:pos [:tensor [1] :i32]]]
          ast [:rope [:y :b :p :d] [:x :b :p :d]
               {:head-dim head-dim :rope-proportion rope-prop :theta theta :pos :pos :max-seq-len 16}]
          graph (lower/ast->graph "rope_e2e_dynamic" invars ast #{:y})
          _ (is (shlo/validate-graph graph))
          compiled (xla/compile-graph ctx graph)
          x-raw (vec (map #(float (inc %)) (range (* batch seq-len total-dim))))
          x-data (float-array x-raw)
          pos-data (int-array [pos-val])
          out-buf (xla/execute compiled x-data pos-data)
          pjrt-res (vec (xla/to-host-slice out-buf 0 (* batch seq-len total-dim) 4))
          ref-res (reference-rope x-raw batch seq-len n-heads head-dim rope-prop theta pos-val)]
      (doseq [i (range (count x-raw))]
        (is (< (Math/abs (- (double (nth pjrt-res i)) (double (nth ref-res i)))) 1e-4)
            (str "Dynamic mismatch at index " i " pjrt=" (nth pjrt-res i) " ref=" (nth ref-res i))))
      (xla/destroy-buffer! out-buf))))

(defspec prop-rope-symbolic-parity 20
  (prop/for-all [b (gen/elements [1 2])
                 seq-len (gen/elements [1 4])
                 head-dim (gen/elements [16 32 64])
                 n-heads (gen/elements [1 2 4])
                 rope-prop (gen/elements [1.0 0.5 0.25])
                 pos-val (gen/choose 0 4095)]
                (let [total-dim (* n-heads head-dim)
                      dynamic? (= seq-len 1)
                      theta 10000.0
                      invars (if dynamic?
                               [[:x [:tensor [b seq-len total-dim] :f32]]
                                [:pos [:tensor [1] :i32]]]
                               [[:x [:tensor [b seq-len total-dim] :f32]]])
                      attrs (cond-> {:head-dim head-dim
                                     :theta theta
                                     :rope-proportion rope-prop}
                              dynamic? (assoc :pos :pos :max-seq-len 4096))
                      ast [:rope [:y :b :p :d] [:x :b :p :d] attrs]
                      graph (lower/ast->graph "rope_symbolic_test" invars ast #{:y})
                      ctx (xla/get-context)
                      compiled (xla/compile-graph ctx graph)
                      n-elem (* b seq-len total-dim)
                      x-data (float-array n-elem)
                      _ (dotimes [i n-elem] (aset x-data i (float (Math/sin (double (inc i))))))
                      out-buf (if dynamic?
                                (xla/execute compiled x-data (int-array [pos-val]))
                                (xla/execute compiled x-data))
                      pjrt-res (vec (xla/to-host-slice out-buf 0 n-elem 4))
                      ref-res (reference-rope (vec x-data) b seq-len n-heads head-dim rope-prop theta (if dynamic? pos-val 0))
                      max-diff (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) pjrt-res ref-res))]
                  (xla/destroy-buffer! out-buf)
                  (< max-diff 5e-4))))


