(ns einsum.uop.opt-test
  "TDD generative and invariant tests for Phase 8: Real OptOps codegen, range-splitting,
   heuristic hand-coded optimizations, and shared-memory reductions."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.uop.core :as uop]
            [einsum.uop.opt :as opt]
            [einsum.uop.rangeify :as rangeify]
            [einsum.uop.render :as render]))

;; -----------------------------------------------------------------------------
;; 1. OptOps Transformations & Invariants
;; -----------------------------------------------------------------------------

(defspec prop-opt-preserves-axis-products 50
  (prop/for-all [dim (gen/choose 16 1024)
                 divisor (gen/elements [2 4 8 16 32])]
                (if (zero? (mod dim divisor))
                  (let [rk {:out-shape [dim] :out-size dim}
                        ke {:name "k" :rangeified rk :axes [dim] :opts [] :local-size [1 1 1] :global-size [dim 1 1]}
                        opt-local (opt/apply-opt ke {:op :local :axis 0 :amount divisor})
                        opt-upcast (opt/apply-opt ke {:op :upcast :axis 0 :amount divisor})]
                    (and
         ;; :local splits outer and inner
                     (= (* (first (:global-size opt-local)) (first (:local-size opt-local))) dim)
                     (= (first (:local-size opt-local)) divisor)
         ;; :upcast keeps total dimension unchanged
                     (= (:axes opt-upcast) [(quot dim divisor) divisor])))
                  true)))

(deftest test-apply-opt-operations
  (testing "apply-opt handles :local, :upcast, :group, and :unroll correctly"
    (let [rk {:out-shape [64 128] :out-size 8192}
          ke {:name "test_ke"
              :rangeified rk
              :axes [64 128]
              :red-axes [256]
              :opts []
              :local-size [1 1 1]
              :global-size [64 128 1]}
          ;; 1. Apply :local on axis 1 (amount 16)
          k1 (opt/apply-opt ke {:op :local :axis 1 :amount 16})
          _ (is (= (:local-size k1) [1 16 1]))
          _ (is (= (:global-size k1) [64 8 1]))

          ;; 2. Apply :upcast on axis 0 (amount 4)
          k2 (opt/apply-opt k1 {:op :upcast :axis 0 :amount 4})
          _ (is (some #(= (:op %) :upcast) (:opts k2)))

          ;; 3. Apply :group on reduction axis 0 (amount 64)
          k3 (opt/apply-opt k2 {:op :group :axis 0 :amount 64})
          _ (is (true? (:has-group? k3)))
          _ (is (= 64 (:group-size k3)))

          ;; 4. Apply :unroll on reduction axis 0 (amount 4)
          k4 (opt/apply-opt k3 {:op :unroll :axis 0 :amount 4})
          _ (is (= 4 (:unroll-factor k4)))]
      (is (= 4 (count (:opts k4)))))))

;; -----------------------------------------------------------------------------
;; 2. Heuristic Hand-Coded Optimizations (tinygrad heuristic.py port)
;; -----------------------------------------------------------------------------

(deftest test-hand-coded-optimizations-classification
  (testing "GEMV pattern (M=1, large N, K) triggers cooperative group reduction on K"
    (let [m 1 n 4096 k 4096
          rk {:out-shape [m n] :out-size (* m n) :red-shape [k]}
          ke {:name "gemv" :rangeified rk :axes [m n] :red-axes [k]}
          opt-ke (opt/hand-coded-optimizations ke {:target :rocm})]
      (is (true? (:has-group? opt-ke)))
      (is (pos? (long (or (:group-size opt-ke) 0))))
      (is (vector? (:local-size opt-ke)))
      (is (vector? (:global-size opt-ke)))))

  (testing "2D Matmul pattern (M >= 16, N >= 16, K >= 16) triggers 2D tiling and upcasting"
    (let [m 64 n 64 k 64
          rk {:out-shape [m n] :out-size (* m n) :red-shape [k]}
          ke {:name "gemm" :rangeified rk :axes [m n] :red-axes [k]}
          opt-ke (opt/hand-coded-optimizations ke {:target :rocm})]
      (is (some #(= (:op %) :local) (:opts opt-ke)))
      (is (some #(= (:op %) :upcast) (:opts opt-ke)))
      (is (= 256 (reduce * 1 (:local-size opt-ke))))))

  (testing "Fused RMSNorm / Softmax reduction over last axis triggers workgroup reduction"
    (let [b 4 s 128 d 256
          rk {:out-shape [b s 1] :out-size (* b s) :red-shape [d] :reduction-kind :norm}
          ke {:name "rmsnorm_red" :rangeified rk :axes [b s 1] :red-axes [d]}
          opt-ke (opt/hand-coded-optimizations ke {:target :rocm})]
      (is (or (true? (:has-group? opt-ke))
              (== 256 (first (:local-size opt-ke))))))))

;; -----------------------------------------------------------------------------
;; 3. Shared-Memory Cooperative Reductions (__shared__ LDS)
;; -----------------------------------------------------------------------------

(deftest test-render-shared-memory-group-reduction
  (testing "Kernel with :group emits valid HIP shared memory LDS reduction code"
    (let [p0 (uop/uop-param 0 :f32 [1 1024])
          p1 (uop/uop-param 1 :f32 [1024 1024])
          prod (uop/uop-alu :uop/mul
                            (uop/uop-expand (uop/uop-reshape p0 [1 1 1024]) [1 1024 1024])
                            (uop/uop-expand (uop/uop-reshape (uop/uop-permute p1 [1 0]) [1 1024 1024]) [1 1024 1024]))
          red (uop/uop-reduce prod :add [2])
          rk (rangeify/rangeify-kernel red {:out-slot 2})
          ke (-> {:name "gemv_lds" :rangeified rk :axes [1 1024] :red-axes [1024]}
                 (opt/apply-opt {:op :group :axis 0 :amount 64})
                 (opt/apply-opt {:op :local :axis 1 :amount 4}))
          hip-prog (render/render-kernel ke {:target :rocm})]
      (is (string? (:source hip-prog)))
      ;; Must contain shared memory LDS declaration and synchronization
      (is (str/includes? (:source hip-prog) "__shared__"))
      (is (str/includes? (:source hip-prog) "__syncthreads()")))))
