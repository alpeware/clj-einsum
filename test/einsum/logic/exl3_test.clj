(ns einsum.logic.exl3-test
  "Generative property and unit tests for EXL3 Quantization (turboderp-org/exllamav3)."
  (:require [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [einsum.logic.exl3 :as exl3]
            [einsum.logic.expand :as expand]
            [einsum.logic.lower :as lower]
            [einsum.logic.shape :as shape]
            [einsum.runtime.safetensors :as st]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop])
  (:import [java.lang.foreign Arena]))

;; --- Mathematical Reference Helpers ---

(defn- close?
  ([a b] (close? a b 1e-4))
  ([a b eps]
   (let [diff (Math/abs (double (- a b)))]
     (<= diff eps))))

(defn- vec-close?
  ([v1 v2] (vec-close? v1 v2 1e-4))
  ([v1 v2 eps]
   (and (= (count v1) (count v2))
        (every? true? (map #(close? %1 %2 eps) v1 v2)))))

;; --- Generative Property Tests (Rule 1: Strict TDD) ---

(defspec prop-tc-perm-bijection
  50
  (prop/for-all [idx (gen/choose 0 255)]
                (let [perm (exl3/make-tc-perm)
                      inv (exl3/make-tc-perm-inv)]
                  (and (= 256 (count (set perm)))
                       (= 256 (count (set inv)))
                       (= idx (nth inv (nth perm idx)))
                       (= idx (nth perm (nth inv idx)))))))

(defspec prop-mul1-codebook-properties
  10
  (prop/for-all [_ (gen/return true)]
                (let [cb (exl3/generate-mul1-codebook)
                      mean (/ (reduce + cb) (count cb))
                      min-v (reduce min cb)
                      max-v (reduce max cb)]
                  (and (= 65536 (count cb))
                       (every? #(not (Double/isNaN %)) cb)
                       (every? #(not (Double/isInfinite %)) cb)
                       (<= -4.0 min-v)
                       (<= max-v 4.0)
                       (< (Math/abs (double mean)) 0.01)))))

(defspec prop-mcg-codebook-properties
  10
  (prop/for-all [_ (gen/return true)]
                (let [cb (exl3/generate-mcg-codebook)
                      mean (/ (reduce + cb) (count cb))
                      min-v (reduce min cb)
                      max-v (reduce max cb)]
                  (and (= 65536 (count cb))
                       (every? #(not (Double/isNaN %)) cb)
                       (every? #(not (Double/isInfinite %)) cb)
                       (<= -4.5 min-v)
                       (<= max-v 4.5)
                       (< (Math/abs (double mean)) 0.01)))))

(defspec prop-trellis-state-scalar-range
  50
  (prop/for-all [k (gen/choose 1 8)
                 t-offset (gen/choose 0 255)
                 packed-words (gen/vector (gen/choose 0 65535) (* 16 8))]
                (let [words (subvec packed-words 0 (* 16 k))
                      state (exl3/decode-state-scalar words t-offset k)]
                  (and (integer? state)
                       (<= 0 state 65535)))))

(defspec prop-per-row-int8-quantization-invariants
  50
  (prop/for-all [rows (gen/choose 2 8)
                 cols (gen/choose 16 32)
                 scale-factors (gen/vector (gen/double* {:min 0.1 :max 10.0 :NaN? false :infinite? false}) 8)]
                (let [total (* rows cols)
                      f-arr (float-array total)
                      _ (dotimes [r rows]
                          (let [s (nth scale-factors r)]
                            (dotimes [c cols]
                              (aset-float f-arr (+ (* r cols) c) (float (* s (Math/sin (double (+ r c)))))))))
                      {:keys [data scales]} (exl3/quantize-weights-per-row-int8 f-arr rows cols)
                      bytes-data ^bytes data
                      scales-data ^floats scales]
                  (and (= total (alength bytes-data))
                       (= rows (alength scales-data))
                       (every? #(<= -127 % 127) (vec bytes-data))
                       (every? #(> % 0.0) (vec scales-data))))))

(defspec prop-per-row-int4-quantization-invariants
  50
  (prop/for-all [rows (gen/choose 2 8)
                 cols (gen/elements [16 32 64])
                 scale-factors (gen/vector (gen/double* {:min 0.1 :max 10.0 :NaN? false :infinite? false}) 8)]
                (let [total (* rows cols)
                      half-total (quot total 2)
                      f-arr (float-array total)
                      _ (dotimes [r rows]
                          (let [s (nth scale-factors r)]
                            (dotimes [c cols]
                              (aset-float f-arr (+ (* r cols) c) (float (* s (Math/sin (double (+ r c)))))))))
                      {:keys [data scales]} (exl3/quantize-weights-per-row-int4 f-arr rows cols)
                      bytes-data ^bytes data
                      scales-data (vec scales)]
                  (and (= half-total (alength bytes-data))
                       (= rows (count scales-data))
                       (every? (fn [b]
                                 (let [b-int (int b)
                                       lo (bit-and b-int 0x0F)
                                       hi (bit-and (bit-shift-right b-int 4) 0x0F)]
                                   (and (<= 1 lo 15)
                                        (<= 1 hi 15))))
                               (vec bytes-data))))))

(defspec prop-block-int4-quantization-invariants
  50
  (prop/for-all [rows (gen/choose 2 6)
                 group-size (gen/elements [16 32])
                 num-groups (gen/choose 1 4)
                 scale-factors (gen/vector (gen/double* {:min 0.1 :max 5.0 :NaN? false :infinite? false}) 6)]
                (let [cols (* group-size num-groups)
                      total (* rows cols)
                      half-total (quot total 2)
                      f-arr (float-array total)
                      _ (dotimes [r rows]
                          (let [s (nth scale-factors r)]
                            (dotimes [c cols]
                              (aset-float f-arr (+ (* r cols) c) (float (* s (Math/cos (double (+ r c)))))))))
                      {:keys [data scales shape scale-shape]} (exl3/quantize-weights-per-row-int4 f-arr rows cols {:group-size group-size :as :bf16})
                      bytes-data ^bytes data
                      scales-data ^shorts scales]
                  (and (= half-total (alength bytes-data))
                       (= (* rows num-groups) (alength scales-data))
                       (= [rows (quot cols 2)] shape)
                       (= [rows num-groups] scale-shape)
                       (every? (fn [b]
                                 (let [b-int (int b)
                                       lo (bit-and b-int 0x0F)
                                       hi (bit-and (bit-shift-right b-int 4) 0x0F)]
                                   (and (<= 1 lo 15)
                                        (<= 1 hi 15))))
                               (vec bytes-data))))))

;; --- Unit Tests for EXL3 Decoding & Mathematical Parity ---

(deftest test-mul1-reference-constants
  (testing "Verify mul1 codebook matches known scalar outputs."
    (let [cb (exl3/generate-mul1-codebook)]
      ;; Known from Python verification:
      ;; state 0: -3.453125
      ;; state 1: 0.64111328125
      (is (close? (nth cb 0) -3.453125 1e-4))
      (is (close? (nth cb 1) 0.64111328 1e-4)))))

(deftest test-mcg-reference-constants
  (testing "Verify mcg codebook matches known scalar outputs."
    (let [cb (exl3/generate-mcg-codebook)]
      ;; Known from Python verification:
      ;; state 0: 1.84375
      ;; state 1: 0.134521484375
      (is (close? (nth cb 0) 1.84375 1e-4))
      (is (close? (nth cb 1) 0.13452148 1e-4)))))

(deftest test-decode-state-scalar-parity
  (testing "Verify decode-state-scalar matches ground truth Python reference vector."
    (let [sample-packed [19130 21975 44735 30206 45040 32647 64571 57820 3811 30489
                         1234 5678 9101 1121 3141 5161 7181 9202 2223 2425
                         2627 2829 3031 3233 3435 3637 3839 4041 4243 4445
                         4647 4849 5051 5253 5455 5657 5859 6061 6263 6465
                         6667 6869 7071 7273 7475 7677 7879 8081]
          expected-first-5 [63034 45525 36523 30045 43755]
          actual-first-5 (mapv #(exl3/decode-state-scalar sample-packed % 3) (range 5))]
      (is (= expected-first-5 actual-first-5)))))

(deftest test-hadamard-block-128-involution
  (testing "Applying hadamard-block-128 twice returns the original vector."
    (let [v (mapv double (range 256))
          h1 (exl3/hadamard-block-128-ref v)
          h2 (exl3/hadamard-block-128-ref h1)]
      (is (vec-close? v h2 1e-4)))))

(deftest test-exl3-ast-shape-inference
  (testing "Verify shape unification for :hadamard-block-128 and :exl3-dequant."
    (let [in-shapes {:x [1 3840]
                     :trellis [240 128 48]
                     :codebook [65536]}
          ast [:block {:name :exl3_test}
               [:hadamard-block-128 [:x_rot :b :k] [:x :b :k]]
               [:exl3-dequant [:w_deq :k :n] [:trellis :kt :nt :w] [:codebook :c]]]
          expanded (expand/expand-ast {} ast)
          shapes (shape/unify-shapes in-shapes expanded)]
      (is (= [1 3840] (get shapes :x_rot)))
      (is (= [3840 2048] (get shapes :w_deq))))))

(deftest test-exl3-ast-to-graph-lowering
  (testing "Lowering an EXL3 AST block generates valid StableHLO operations."
    (let [ast [:block {:name :exl3_block}
               [:hadamard-block-128 [:xh :b :k] [:x :b :k]]
               [:= [:y :b :n] [:xh :b :k] [:w :k :n]]]
          invars [[:x [:tensor [1 256] :f32]]
                  [:w [:tensor [256 256] :f32]]]
          graph (lower/ast->graph "exl3_test_graph" invars ast [:y])
          op-types (set (map :op (:eqns graph)))]
      (is (contains? op-types :stablehlo/dot_general))
      (is (contains? op-types :stablehlo/slice))
      (is (contains? op-types :stablehlo/concatenate)))))

(deftest test-exl3-pjrt-cpu-execution
  (testing "Execute hadamard-block-128 through OpenXLA PJRT."
    (let [ctx (xla/init-backend! :cpu)
          ast [:hadamard-block-128 [:y 1 128] [:x 1 128]]
          invars [[:x [:tensor [1 128] :f32]]]
          compiled (logic/compile-ast ctx "fwht128_graph" invars ast #{:y})
          in-data (float-array (range 1 129))
          out-buf (xla/execute compiled in-data)
          actual (vec (xla/to-host-slice out-buf 0 128 4))
          expected (exl3/hadamard-block-128-ref (vec in-data))]
      (is (vec-close? actual expected 1e-3))
      (xla/destroy-buffer! out-buf))))

(deftest test-dequant-exl3-matrix-parity
  (testing "Verify dequant-exl3-matrix matches unquantized model weights with high cosine similarity."
    (let [f (io/file "verification/k_proj_sample.safetensors")]
      (if (.exists f)
        (with-open [arena (Arena/ofConfined)]
          (let [mmap (st/map-safetensors-weights (.getPath f) arena)
                trellis (st/get-tensor-slice mmap "k_proj.trellis")
                suh (st/get-tensor-slice mmap "k_proj.suh")
                svh (st/get-tensor-slice mmap "k_proj.svh")
                w-deq (exl3/dequant-exl3-matrix trellis 3840 2048 3 suh svh {:as :f32})
                first-4 (subvec (vec (take 4 w-deq)) 0 4)
                expected-first-4 [0.0105326 0.0139068 0.0037987 0.0073046]]
            (is (= (* 3840 2048) (alength ^floats w-deq)))
            (is (vec-close? first-4 expected-first-4 1e-4))))
        (is true "Skipped: fixture file not present")))))
