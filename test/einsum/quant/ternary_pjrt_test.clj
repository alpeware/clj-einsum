(ns einsum.quant.ternary-pjrt-test
  "OpenXLA PJRT StableHLO execution and numeric parity tests for ternary quantization."
  (:require [clojure.test :refer [deftest is testing]]
            [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [einsum.quant.ternary :as ternary]))

;; ==============================================================================
;; OpenXLA PJRT StableHLO Execution Parity
;; ==============================================================================

(deftest test-ternary-unpack-pjrt-cpu-parity
  (testing "OpenXLA PJRT StableHLO ternary-unpack matches reference unpacking exactly"
    (let [ctx (xla/init-backend! :cpu)
          rows 2
          cols 8
          ;; Sample ternary weights: rows=2, cols=8
          weights [-1.0  0.0  1.0 -1.0   0.0  1.0 -1.0  0.0
                   1.0 -1.0  0.0  1.0  -1.0  0.0  1.0 -1.0]
          packed-bytes (byte-array (ternary/pack-ternary-2bit-ref weights))
          scales (float-array [0.75 1.5])
          ast [:ternary-unpack [:w_out 2 8] [:w_packed 2 2] [:scales 2]]
          invars [[:w_packed [:tensor [2 2] :i8]]
                  [:scales [:tensor [2] :f32]]]
          compiled (logic/compile-ast ctx "ternary_unpack_test" invars ast #{:w_out})
          out-buf (xla/execute compiled packed-bytes scales)
          actual (vec (xla/to-host-slice out-buf 0 (* rows cols) 4))
          _ (xla/destroy-buffer! out-buf)
          expected-row0 (ternary/unpack-ternary-2bit-ref (subvec (vec packed-bytes) 0 2) 0.75)
          expected-row1 (ternary/unpack-ternary-2bit-ref (subvec (vec packed-bytes) 2 4) 1.5)
          expected (vec (concat expected-row0 expected-row1))]
      (dotimes [i (* rows cols)]
        (is (< (Math/abs (- (double (nth actual i)) (double (nth expected i)))) 1e-4)
            (str "Mismatch at index " i " actual: " (nth actual i) " expected: " (nth expected i)))))))

(deftest test-ternary-contraction-parity
  (testing "Ternary unpacked matrix contraction matches dense float matmul"
    (let [ctx (xla/init-backend! :cpu)
          d-in 8
          d-out 4
          weights [-1.0  0.0  1.0 -1.0   0.0  1.0 -1.0  0.0
                   1.0 -1.0  0.0  1.0  -1.0  0.0  1.0 -1.0
                   -1.0  1.0 -1.0  0.0   1.0 -1.0  0.0  1.0
                   0.0  1.0  0.0 -1.0   1.0  0.0 -1.0  1.0]
          packed-bytes (byte-array (ternary/pack-ternary-2bit-ref weights))
          scales (float-array [1.0 1.0 1.0 1.0])
          x-vec [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
          x-arr (float-array x-vec)
          ast [:block {:name :proj}
               [:ternary-unpack [:w_scaled 4 8] [:w_packed 4 2] [:scales 4]]
               [:= [:y 1 4] [:x 1 8] [:w_scaled 4 8]]]
          invars [[:x [:tensor [1 8] :f32]]
                  [:w_packed [:tensor [4 2] :i8]]
                  [:scales [:tensor [4] :f32]]]
          compiled (logic/compile-ast ctx "ternary_contract_test" invars ast #{:y})
          out-buf (xla/execute compiled x-arr packed-bytes scales)
          actual (vec (xla/to-host-slice out-buf 0 4 4))
          _ (xla/destroy-buffer! out-buf)
          ;; Dense reference matmul: y[j] = sum_k x[k] * w[j, k]
          expected (vec (mapv (fn [j]
                                (let [w-row (subvec (vec weights) (* j d-in) (* (inc j) d-in))]
                                  (reduce + (map * x-vec w-row))))
                              (range d-out)))]
      (dotimes [j d-out]
        (is (< (Math/abs (- (double (nth actual j)) (double (nth expected j)))) 1e-4)
            (str "Matmul mismatch at index " j " actual: " (nth actual j) " expected: " (nth expected j)))))))
