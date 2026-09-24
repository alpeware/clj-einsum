(ns einsum.compiler.kernel-test
  "Generative property tests and unit tests for callable compiled kernels (einsum.compiler.kernel)."
  (:require [einsum.compiler.kernel :as kernel]
            [einsum.core :as xla]
            [einsum.runtime.arena :as arena]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-compile-kernel-map-invocation
  (testing "compile-kernel produces a callable Clojure function with keyword input and output maps"
    (let [ctx (xla/init-backend! :cpu)
          ast [:= [:y :b :d] [:x :b :k] [:w :k :d]]
          k (kernel/compile-kernel ctx "gemm_kernel" ast
                                   {:in {:x [:tensor [1 2] :f32]
                                         :w [:tensor [2 2] :f32]}
                                    :out [:y]})]
      (is (ifn? k))
      (is (= [:y] (:out-spec k)))
      (let [x-data (float-array [1.0 2.0])
            w-data (float-array [1.0 0.0
                                 0.0 1.0])
            res (k {:x x-data :w w-data})]
        (is (map? res))
        (is (contains? res :y))
        (let [out-buf (:y res)
              floats (xla/to-host-slice out-buf 0 2 2 :f32)]
          (is (= [1.0 2.0] (vec floats))))))))

(deftest test-compile-kernel-with-device-arena-auto-cleanup
  (testing "compile-kernel automatically registers transient inputs and outputs in active arena"
    (let [ctx (xla/init-backend! :cpu)
          ast [:= [:y :b :d] [:x :b :k] [:w :k :d]]
          k (kernel/compile-kernel ctx "gemm_arena_test" ast
                                   {:in {:x [:tensor [1 2] :f32]
                                         :w [:tensor [2 2] :f32]}
                                    :out [:y]})
          x-data (float-array [2.0 3.0])
          w-data (float-array [2.0 0.0
                               0.0 2.0])
          captured-arena (atom nil)
          output-result (atom nil)]
      (arena/with-device-arena [a ctx]
        (reset! captured-arena a)
        (let [res (k {:x x-data :w w-data})]
          ;; Inside arena: x, w, and y are tracked
          (is (= 3 (count (arena/tracked-buffers a))))
          (let [floats (xla/to-host-slice (:y res) 0 2 2 :f32)]
            (reset! output-result (vec floats)))))
      ;; Outside arena: arena is closed and all 3 buffers were destroyed without leak
      (is (true? (arena/closed? @captured-arena)))
      (is (= [4.0 6.0] @output-result)))))

(defspec prop-compile-kernel-linear-scaling
  20
  (prop/for-all [scale (gen/choose 1 10)]
                (let [ctx (xla/init-backend! :cpu)
                      ast [:= [:y :b :d] {:scale (double scale)} [:x :b :d]]
                      k (kernel/compile-kernel ctx "scale_kernel" ast
                                               {:in {:x [:tensor [1 4] :f32]}
                                                :out [:y]})
                      input-vec [1.0 2.0 3.0 4.0]
                      expected (mapv #(float (* (float scale) %)) input-vec)]
                  (arena/with-device-arena [_a ctx]
                    (let [res (k {:x (float-array input-vec)})
                          floats (xla/to-host-slice (:y res) 0 4 4 :f32)]
                      (= expected (vec floats)))))))
