(ns einsum.uop.schedule-test
  "Unit and generative tests for UOp Kernel Scheduler (`schedule.clj`), verifying realization boundary cutting,
   post-reduction reshape folding, single-kernel matmul emission, and end-to-end evaluation parity."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.uop.core :as uop]
            [einsum.uop.lower :as lower]
            [einsum.uop.rewrite :as rewrite]
            [einsum.uop.schedule :as schedule]))

(defn- close-floats?
  ([^floats a ^floats b]
   (close-floats? a b 1e-4))
  ([^floats a ^floats b ^double tol]
   (and (= (alength a) (alength b))
        (every? true?
                (map (fn [x y]
                       (<= (Math/abs (- (double x) (double y))) tol))
                     (seq a)
                     (seq b))))))

(defn- reference-matmul
  "Computes reference A [m k] @ B [k n] -> C [m n] on float arrays."
  [^floats a ^floats b m k n]
  (let [c (float-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (let [acc (atom 0.0)]
          (dotimes [kk k]
            (let [val-a (aget a (+ (* i k) kk))
                  val-b (aget b (+ (* kk n) j))]
              (swap! acc + (* val-a val-b))))
          (aset c (+ (* i n) j) (float @acc)))))
    c))

;; -----------------------------------------------------------------------------
;; 1. Unit Tests: Singleton Reshape Folding & Single-Kernel Matmul
;; -----------------------------------------------------------------------------

(deftest test-fused-post-reduction-reshape-dot-general
  (testing "Matrix multiplication schedules exactly 1 kernel with folded reshape out-shape [4 16]"
    (let [ast [[:= [:y :i :j] [:x :i :k] [:w :k :j]]]
          invars [[:x [:tensor [4 8] :f32]] [:w [:tensor [8 16] :f32]]]
          outvars [:y]
          ug (lower/ast->uop-graph "mm_schedule_test" invars ast outvars)
          sched (schedule/schedule-graph ug)
          kernels (:kernels sched)
          k0 (first kernels)]
      ;; Exactly 1 kernel emitted instead of 2 (no intermediate [4 16 1] buffer copy kernel)
      (is (= 1 (count kernels)))
      (is (= [4 16] (:out-shape k0)))
      (is (= :y (:out-var k0)))
      (is (= (:out-slot k0) (get-in sched [:output-slots :y])))
      ;; Output buffer has shape [4 16] and role :output
      (let [out-buf (get-in sched [:buffers (:out-slot k0)])]
        (is (= :output (:role out-buf)))
        (is (= [4 16] (:shape out-buf)))
        (is (= 64 (:size out-buf)))))))

(deftest test-fused-matmul-eval-scheduled-graph-parity
  (testing "eval-scheduled-graph computes exact 2D matmul output on fused kernel"
    (let [ast [[:= [:y :i :j] [:x :i :k] [:w :k :j]]]
          invars [[:x [:tensor [2 3] :f32]] [:w [:tensor [3 4] :f32]]]
          outvars [:y]
          ug (lower/ast->uop-graph "mm_eval_test" invars ast outvars)
          sched (schedule/schedule-graph ug)
          x-arr (float-array [1 2 3
                              4 5 6])
          w-arr (float-array [1 0 0 1
                              0 1 0 1
                              0 0 1 1])
          res (schedule/eval-scheduled-graph sched {:x x-arr :w w-arr})
          y-arr (get res :y)
          expected (float-array [1 2 3 6
                                 4 5 6 15])]
      (is (= 1 (count (:kernels sched))))
      (is (close-floats? expected y-arr)))))

;; -----------------------------------------------------------------------------
;; 2. Generative Property Test: Evaluation Parity Across Arbitrary Shapes
;; -----------------------------------------------------------------------------

(defspec prop-fused-matmul-evaluation-parity
  30
  (prop/for-all [m (gen/choose 1 8)
                 k (gen/choose 1 16)
                 n (gen/choose 1 8)]
                (let [a-data (float-array (repeatedly (* m k) #(- (rand 4.0) 2.0)))
                      b-data (float-array (repeatedly (* k n) #(- (rand 4.0) 2.0)))
                      ast [[:= [:y :i :j] [:x :i :k] [:w :k :j]]]
                      invars [[:x [:tensor [m k] :f32]] [:w [:tensor [k n] :f32]]]
                      outvars [:y]
                      ug (lower/ast->uop-graph "mm_prop" invars ast outvars)
                      sched (schedule/schedule-graph ug)
                      ref-out (reference-matmul a-data b-data m k n)
                      res (schedule/eval-scheduled-graph sched {:x a-data :w b-data})
                      sched-out (get res :y)]
                  (and (= 1 (count (:kernels sched)))
                       (close-floats? ref-out sched-out 1e-3)))))

(deftest test-channel-splitting-reshape-not-folded-into-reduction
  (testing "Channel-splitting reshape [1 16 128] -> [1 16 4 32] is not folded into reduction kernel, and 0 extra kernels emitted when consumed"
    (let [p0 (uop/uop-param 0 :f32 [1 16 64] :x)
          p1 (uop/uop-param 1 :f32 [64 128] :w)
          e0 (uop/uop-expand (uop/uop-reshape p0 [1 16 1 64]) [1 16 128 64])
          e1 (-> p1
                 (uop/uop-permute [1 0])
                 (uop/uop-reshape [1 1 128 64])
                 (uop/uop-expand [1 16 128 64]))
          prod (uop/uop-alu :uop/mul e0 e1)
          red (uop/uop-reshape (uop/uop-reduce prod :add [3] false) [1 16 128])
          reshaped (uop/uop-reshape red [1 16 4 32])
          out (rewrite/simplify-uop (uop/uop-alu :uop/neg reshaped))
          uop-graph {:name "channel_split_test"
                     :invars [[:x [:tensor [1 16 64] :f32]]
                              [:w [:tensor [64 128] :f32]]]
                     :outvars [:out]
                     :outputs {:out out}}
          sched (schedule/schedule-graph uop-graph)
          kernels (:kernels sched)
          k0 (first kernels)
          k1 (second kernels)]
      (is (= [1 16 128] (:out-shape k0)))
      (is (= 2 (count kernels)))
      (is (= [1 16 4 32] (:out-shape k1)))
      (is (= :out (:out-var k1))))))

