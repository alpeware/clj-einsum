(ns einsum.uop.rangeify-test
  "Generative property and unit tests for Phase 3: UOp Kernel Scheduler (`schedule.clj`) and Backward Coordinate Propagation (`rangeify.clj`)."
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gpt2 :as gpt2]
            [einsum.models.smollm :as smollm]
            [einsum.uop.core :as uop]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.rangeify :as rangeify]
            [einsum.uop.schedule :as schedule]))

;; -----------------------------------------------------------------------------
;; 1. Rangeify Unit & Generative Property Tests
;; -----------------------------------------------------------------------------

(deftest test-rangeify-dissolves-all-movement-ops
  (testing "rangeify eliminates RESHAPE, EXPAND, PERMUTE, SHRINK, PAD, FLIP, and CAT into explicit RANGE + INDEX + LOAD/STORE UOps"
    (uop/clear-uop-cache!)
    (let [a (uop/uop-param 0 :f32 [2 4])
          b (uop/uop-param 1 :f32 [4 3])
          ;; Matmul C[2, 3] = sum_k A[i, k] * B[k, j]
          a-3d (uop/uop-expand (uop/uop-reshape a [2 1 4]) [2 3 4])
          b-3d (-> b
                   (uop/uop-permute [1 0])
                   (uop/uop-reshape [1 3 4])
                   (uop/uop-expand [2 3 4]))
          prod (uop/uop-alu :uop/mul a-3d b-3d)
          red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [2 3])
          rk (rangeify/rangeify-kernel red {:out-slot 2})
          nodes (uop/toposort (:sink rk))
          ops (set (map :op nodes))]
      ;; Zero movement ops remain in the rangeified kernel DAG!
      (is (empty? (set/intersection ops uop/MOVEMENT_OPS)))
      ;; Contains explicit loop RANGE, INDEX, LOAD, and STORE nodes
      (is (contains? ops :uop/range))
      (is (contains? ops :uop/index))
      (is (contains? ops :uop/load))
      (is (contains? ops :uop/store)))))

(defspec prop-rangeify-movement-equivalence
  50
  (prop/for-all [d0 (gen/choose 2 4)
                 d1 (gen/choose 2 4)
                 d2 (gen/choose 2 4)]
                (let [n (* d0 d1 d2)
                      in-data (float-array (map float (range n)))
                      p0 (uop/uop-param 0 :f32 [d0 d1 d2])
                      ;; Permute [2 0 1] -> Reshape [d2 (* d0 d1)] -> Shrink first axis to [0 2]
                      perm (uop/uop-permute p0 [2 0 1])
                      rsh (uop/uop-reshape perm [d2 (* d0 d1)])
                      shr (uop/uop-shrink rsh [[0 2] [0 (* d0 d1)]])
                      rk (rangeify/rangeify-kernel shr {:out-slot 1})
                      out-arr (rangeify/eval-rangeified-kernel rk {0 in-data})]
                  ;; Check that for every (k_idx in 0..1, i in 0..d0-1, j in 0..d1-1),
                  ;; out[k_idx, i*d1 + j] == in[i, j, k_idx]
                  (every? true?
                          (for [k-idx (range 2)
                                i (range d0)
                                j (range d1)]
                            (let [flat-out (+ (* k-idx d0 d1) (* i d1) j)
                                  flat-in (+ (* i d1 d2) (* j d2) k-idx)]
                              (== (aget ^floats out-arr flat-out)
                                  (aget ^floats in-data flat-in))))))))

(defspec prop-rangeify-matmul-and-bias-equivalence
  40
  (prop/for-all [m (gen/choose 1 4)
                 k (gen/choose 2 6)
                 n (gen/choose 1 5)]
                (let [a-data (float-array (repeatedly (* m k) #(- (rand 2.0) 1.0)))
                      b-data (float-array (repeatedly (* k n) #(- (rand 2.0) 1.0)))
                      bias-data (float-array (repeatedly n #(- (rand 1.0) 0.5)))
                      invars [[:A [:tensor [m k] :f32]]
                              [:B [:tensor [k n] :f32]]
                              [:bias [:tensor [n] :f32]]]
                      ast [:block {}
                           [:= [:C :i :j] [:A :i :k] [:B :k :j]]
                           [:= [:C :i :j] [:bias :j]]]
                      uop-graph (uop-lower/ast->uop-graph "mm_bias" invars ast [:C])
                      sched (schedule/schedule-graph uop-graph)
                      out-map (schedule/eval-scheduled-graph sched {:A a-data :B b-data :bias bias-data})
                      c-out ^floats (get out-map :C)]
                  (every? true?
                          (for [i (range m)
                                j (range n)]
                            (let [expected (+ (double (aget bias-data j))
                                              (reduce + 0.0
                                                      (for [kk (range k)]
                                                        (* (double (aget a-data (+ (* i k) kk)))
                                                           (double (aget b-data (+ (* kk n) j)))))))
                                  actual (double (aget c-out (+ (* i n) j)))]
                              (< (Math/abs (- expected actual)) 1e-4)))))))

;; -----------------------------------------------------------------------------
;; 2. Scheduler & Multi-Kernel Execution Tests
;; -----------------------------------------------------------------------------

(deftest test-schedule-cuts-chained-reductions-into-separate-kernels
  (testing "Chained reductions (LayerNorm, Softmax, MLP) cut cleanly at realization boundaries to prevent exponential loop recomputation"
    (let [invars [[:x [:tensor [1 4 8] :f32]]
                  [:g [:tensor [8] :f32]]
                  [:b [:tensor [8] :f32]]]
          ast [:layer-norm [:y :b :p :d] [:x :b :p :d] [:g :d] [:b :d]]
          uop-graph (uop-lower/ast->uop-graph "ln_sched" invars ast [:y])
          sched (schedule/schedule-graph uop-graph)
          kernels (:kernels sched)]
      ;; LayerNorm has 2 reductions (mean, variance) + final normalization output
      (is (>= (count kernels) 2))
      ;; Every scheduled kernel has zero remaining high-level movement ops
      (doseq [k kernels]
        (let [ops (set (map :op (uop/toposort (:sink (:rangeified k)))))]
          (is (empty? (set/intersection ops uop/MOVEMENT_OPS)))))
      ;; Verify numerical execution of scheduled LayerNorm
      (let [x-data (float-array (map float (range 32)))
            g-data (float-array (repeat 8 1.0))
            b-data (float-array (repeat 8 0.0))
            res-map (schedule/eval-scheduled-graph sched {:x x-data :g g-data :b b-data})
            y-out ^floats (:y res-map)]
        (is (= 32 (alength y-out)))
        ;; Each row of length 8 should have mean ~ 0.0 and unit variance
        (dotimes [r 4]
          (let [row (mapv #(aget y-out (+ (* r 8) %)) (range 8))
                row-mean (/ (reduce + row) 8.0)
                row-var (/ (reduce + (map #(* % %) row)) 8.0)]
            (is (< (Math/abs row-mean) 1e-4))
            (is (< (Math/abs (- row-var 1.0)) 1e-3))))))))

(deftest test-schedule-gpt2-and-smollm-1layer-end-to-end
  (testing "1-layer GPT-2 and 1-layer SmolLM schedule and rangeify cleanly"
    (let [max-seq-len 4
          gpt2-ast (gpt2/gpt2-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          gpt2-invars [[:x [:tensor [1 max-seq-len] :i32]]
                       [:pos_ids [:tensor [1 max-seq-len] :i32]]
                       [:wte [:tensor [32 768] :f32]]
                       [:wpe [:tensor [16 768] :f32]]
                       [:ln_f_g [:tensor [768] :f32]]
                       [:ln_f_b [:tensor [768] :f32]]
                       [:ln1_g_0 [:tensor [768] :f32]]
                       [:ln1_b_0 [:tensor [768] :f32]]
                       [:attn_w_0 [:tensor [768 2304] :f32]]
                       [:attn_b_0 [:tensor [2304] :f32]]
                       [:proj_w_0 [:tensor [768 768] :f32]]
                       [:proj_b_0 [:tensor [768] :f32]]
                       [:ln2_g_0 [:tensor [768] :f32]]
                       [:ln2_b_0 [:tensor [768] :f32]]
                       [:mlp_fc_w_0 [:tensor [768 3072] :f32]]
                       [:mlp_fc_b_0 [:tensor [3072] :f32]]
                       [:mlp_proj_w_0 [:tensor [3072 768] :f32]]
                       [:mlp_proj_b_0 [:tensor [768] :f32]]]
          gpt2-uop (uop-lower/ast->uop-graph "gpt2_1l" gpt2-invars gpt2-ast [:logits])
          gpt2-sched (schedule/schedule-graph gpt2-uop)
          smollm-ast (smollm/smollm-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          smollm-invars [[:x [:tensor [1 max-seq-len] :i32]]
                         [:embed_tokens [:tensor [32 576] :f32]]
                         [:final_norm_w [:tensor [576] :f32]]
                         [:lm_head_w [:tensor [32 576] :f32]]
                         [:input_ln_w_0 [:tensor [576] :f32]]
                         [:q_w_0 [:tensor [576 576] :f32]]
                         [:k_w_0 [:tensor [192 576] :f32]]
                         [:v_w_0 [:tensor [192 576] :f32]]
                         [:o_w_0 [:tensor [576 576] :f32]]
                         [:post_attn_ln_w_0 [:tensor [576] :f32]]
                         [:gate_w_0 [:tensor [1536 576] :f32]]
                         [:up_w_0 [:tensor [1536 576] :f32]]
                         [:down_w_0 [:tensor [576 1536] :f32]]]
          smollm-uop (uop-lower/ast->uop-graph "smollm_1l" smollm-invars smollm-ast [:logits])
          smollm-sched (schedule/schedule-graph smollm-uop)]
      (is (pos? (count (:kernels gpt2-sched))))
      (is (every? #(some? (:sink (:rangeified %))) (:kernels gpt2-sched)))
      (is (pos? (count (:kernels smollm-sched))))
      (is (every? #(some? (:sink (:rangeified %))) (:kernels smollm-sched))))))

(defspec prop-dynamic-update-slice-ring-buffer-wrapping
  30
  (prop/for-all [window (gen/choose 4 8)
                 d (gen/choose 2 4)
                 pos (gen/choose 0 20)]
                (let [cache-in-data (float-array (* window d) 0.0)
                      k-new-data (float-array (map float (range 1 (inc d))))
                      pos-data (int-array [pos])
                      invars [[:cache_in [:tensor [1 window d] :f32]]
                              [:k_new [:tensor [1 1 d] :f32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:block {}
                           [:dynamic-update-slice [:cache_out :b :s :d]
                            [:cache_in :b :s :d]
                            [:k_new :b :one :d]
                            {:start-indices [0 :pos 0]
                             :window window
                             :k-sink 0}]]
                      uop-graph (uop-lower/ast->uop-graph "dyn_update_wrap" invars ast [:cache_out])
                      sched (schedule/schedule-graph uop-graph)
                      out-map (schedule/eval-scheduled-graph sched {:cache_in cache-in-data
                                                                    :k_new k-new-data
                                                                    :pos pos-data})
                      out ^floats (get out-map :cache_out)
                      expected-slot (mod pos window)]
                  (every? true?
                          (for [slot (range window)
                                col (range d)]
                            (let [val (aget out (+ (* slot d) col))]
                              (if (== slot expected-slot)
                                (== val (float (inc col)))
                                (zero? val))))))))

