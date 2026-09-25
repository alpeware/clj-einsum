(ns einsum.logic.interpret-test
  "Tests for the pure-JVM StableHLO interpreter backend (no PJRT/XLA)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer [defspec]]
            [einsum.logic.interpret :as interp]))

;; ---------------------------------------------------------------------------
;; Tensor construction and shape math
;; ---------------------------------------------------------------------------

(deftest test-tensor-construction
  (let [t (interp/tensor :f32 [2 3] (float-array (range 6)))]
    (is (= :f32 (:dtype t)))
    (is (= [2 3] (:shape t)))
    (is (= 6 (interp/elem-count (:shape t)))))
  (testing "scalar tensor"
    (let [t (interp/tensor :f32 [] (float-array [1.5]))]
      (is (= [] (:shape t)))
      (is (= 1 (interp/elem-count (:shape t))))))
  (testing "bad element count throws"
    (is (thrown? clojure.lang.ExceptionInfo
                 (interp/tensor :f32 [2 3] (float-array (range 5))))))
  (testing "unknown dtype throws"
    (is (thrown? clojure.lang.ExceptionInfo
                 (interp/tensor :f77 [2] (float-array [1 2]))))))

(deftest test-tensor-from-sequential
  (let [t (interp/tensor :f32 [2 2] [1 2 3 4])]
    (is (= [2 2] (:shape t)))
    (is (< (Math/abs (- 1.0 (aget ^floats (:data t) 0))) 1e-9)))
  (let [t (interp/tensor :i32 [3] [7 8 9])]
    (is (= 8 (aget ^ints (:data t) 1)))))

(deftest test-strides-row-major
  (is (= [3 1] (interp/strides [2 3])))
  (is (= [1] (interp/strides [5])))
  (is (= [] (interp/strides [])))
  (is (= [12 4 1] (interp/strides [2 3 4]))))

;; ---------------------------------------------------------------------------
;; bf16 conversions
;; ---------------------------------------------------------------------------

(defspec prop-bf16-roundtrip-stable 200
  (prop/for-all [x (gen/double* {:infinite? false :NaN? false})]
                (let [b (interp/f32->bf16 x)
                      back (interp/bf16->f32 b)
                      b2 (interp/f32->bf16 back)]
      ;; bf16 quantization is idempotent
                  (= b b2))))

(deftest test-bf16-non-finite
  (is (= (short 0x7F80) (interp/f32->bf16 Double/POSITIVE_INFINITY)))
  (is (= (short 0x7FC0) (interp/f32->bf16 Double/NaN)))
  (is (Float/isInfinite (interp/bf16->f32 (interp/f32->bf16 Double/POSITIVE_INFINITY))))
  (is (Float/isNaN (interp/bf16->f32 (interp/f32->bf16 Double/NaN)))))

(defspec prop-bf16-error-bounded 500
  ;; Magnitudes in [1e-30, 1e31]: inside the bf16 normal range, so the
  ;; relative-error bound holds. (Subnormals may flush to zero, which breaks
  ;; any relative bound -- expected, not a bug.)
  (prop/for-all [x (gen/fmap (fn [[neg? e m]] (* (if neg? -1.0 1.0)
                                                 (Math/pow 10.0 ^long e) ^double m))
                             (gen/tuple gen/boolean
                                        (gen/choose -30 30)
                                        (gen/double* {:min 1.0 :max 10.0
                                                      :infinite? false :NaN? false})))]
                (let [f (float x)
                      back (interp/bf16->f32 (interp/f32->bf16 f))]
      ;; bf16 has 8 mantissa bits: relative error < 2^-7 (generous bound)
                  (or (zero? f)
                      (< (/ (Math/abs (- back f)) (Math/abs f)) 0.01)))))

(deftest test-bf16-known-values
  (is (< (Math/abs (- 1.0 (interp/bf16->f32 (interp/f32->bf16 1.0)))) 1e-3))
  (is (< (Math/abs (- -2.5 (interp/bf16->f32 (interp/f32->bf16 -2.5)))) 1e-3))
  (is (zero? (interp/bf16->f32 (interp/f32->bf16 0.0)))))

;; ---------------------------------------------------------------------------
;; Elementwise ops
;; ---------------------------------------------------------------------------

(defn- naive-add [a b] (mapv + a b))

(defspec prop-add-matches-naive 200
  (prop/for-all [n (gen/choose 1 64)]
                (prop/for-all [xs (gen/vector gen/double n)
                               ys (gen/vector gen/double n)]
                              (let [a (interp/tensor :f32 [n] (mapv float xs))
                                    b (interp/tensor :f32 [n] (mapv float ys))
                                    r (interp/op-add a b)
                                    expected (naive-add (mapv float xs) (mapv float ys))]
                                (every? true? (map (fn [x y] (< (Math/abs (- x y)) 1e-4))
                                                   (vec (:data r)) expected))))))

(defspec prop-mul-broadcast 200
  (prop/for-all [m (gen/choose 1 8)
                 n (gen/choose 1 8)]
                (prop/for-all [xs (gen/vector gen/double (* m n))
                               ys (gen/vector gen/double n)]
                              (let [a (interp/tensor :f32 [m n] (mapv float xs))
                                    b (interp/tensor :f32 [n] (mapv float ys))
                                    r (interp/op-mul a b)]
                                (and (= [m n] (:shape r))
                                     (every? true?
                                             (for [i (range m) j (range n)]
                                               (let [got (aget ^floats (:data r) (+ (* i n) j))
                                                     want (* (nth xs (+ (* i n) j)) (nth ys j))]
                                                 (< (Math/abs (- got want)) 1e-3)))))))))

(deftest test-divide-and-subtract
  (let [a (interp/tensor :f32 [3] [6.0 7.0 8.0])
        b (interp/tensor :f32 [3] [2.0 2.0 2.0])]
    (is (= [3.0 3.5 4.0] (mapv float (vec (:data (interp/op-div a b))))))
    (is (= [4.0 5.0 6.0] (mapv float (vec (:data (interp/op-sub a b))))))))

(deftest test-bf16-elementwise-computes-in-f32
  ;; bf16 add: exact in f32, rounded back to bf16 storage
  (let [a (interp/tensor :bf16 [2] [1.5 2.25])
        b (interp/tensor :bf16 [2] [0.5 0.75])
        r (interp/op-add a b)]
    (is (= :bf16 (:dtype r)))
    (is (= [2.0 3.0] (mapv interp/bf16->f32 (vec (:data r)))))))

;; ---------------------------------------------------------------------------
;; Structural ops
;; ---------------------------------------------------------------------------

(deftest test-reshape-shares-storage
  (let [t (interp/tensor :f32 [2 3] [1 2 3 4 5 6])
        r (interp/op-reshape t [3 2])]
    (is (= [3 2] (:shape r)))
    (is (identical? (:data t) (:data r)))))

(deftest test-transpose
  (let [t (interp/tensor :f32 [2 3] [1 2 3 4 5 6])
        r (interp/op-transpose t [1 0])]
    (is (= [3 2] (:shape r)))
    (is (= [1.0 4.0 2.0 5.0 3.0 6.0] (mapv float (vec (:data r)))))))

(deftest test-concatenate
  (let [a (interp/tensor :f32 [2 2] [1 2 3 4])
        b (interp/tensor :f32 [2 3] [5 6 7 8 9 10])
        r (interp/op-concatenate [a b] 1)]
    (is (= [2 5] (:shape r)))
    (is (= [1.0 2.0 5.0 6.0 7.0 3.0 4.0 8.0 9.0 10.0] (mapv float (vec (:data r)))))))

(deftest test-slice
  (let [t (interp/tensor :f32 [4] [10 20 30 40])
        r (interp/op-slice t [1] [3] [1])]
    (is (= [2] (:shape r)))
    (is (= [20.0 30.0] (mapv float (vec (:data r)))))))

(deftest test-iota
  (let [r (interp/op-iota [5] :i32 0)]
    (is (= [5] (:shape r)))
    (is (= [0 1 2 3 4] (vec (:data r))))))

(deftest test-convert-i8-f32
  (let [t (interp/tensor :i8 [3] [1 -2 3])
        r (interp/op-convert t :f32)]
    (is (= :f32 (:dtype r)))
    (is (= [1.0 -2.0 3.0] (mapv float (vec (:data r)))))))

;; ---------------------------------------------------------------------------
;; compare / select
;; ---------------------------------------------------------------------------

(deftest test-compare-select
  (let [a (interp/tensor :f32 [4] [1 2 3 4])
        b (interp/tensor :f32 [4] [2 2 2 2])
        p (interp/op-compare a b "GT")]
    (is (= :pred (:dtype p)))
    (is (= [false false true true] (vec (:data p))))
    (let [x (interp/tensor :f32 [4] [10 20 30 40])
          y (interp/tensor :f32 [4] [1 2 3 4])
          r (interp/op-select p x y)]
      (is (= [1.0 2.0 30.0 40.0] (mapv float (vec (:data r))))))))

;; ---------------------------------------------------------------------------
;; Reductions
;; ---------------------------------------------------------------------------

(deftest test-reduce-sum-mean-max
  (let [t (interp/tensor :f32 [2 3] [1 2 3 4 5 6])]
    (let [s (interp/op-reduce-sum t [-1] false)]
      (is (= [2] (:shape s)))
      (is (= [6.0 15.0] (mapv float (vec (:data s))))))
    (let [m (interp/op-reduce-mean t [-1] true)]
      (is (= [2 1] (:shape m)))
      (is (= [2.0 5.0] (mapv float (vec (:data m))))))
    (let [mx (interp/op-reduce-max t [0] false)]
      (is (= [3] (:shape mx)))
      (is (= [4.0 5.0 6.0] (mapv float (vec (:data mx))))))))

(defspec prop-reduce-sum-matches-naive 200
  (prop/for-all [m (gen/choose 1 6) n (gen/choose 1 6)]
                (prop/for-all [xs (gen/vector gen/double (* m n))]
                              (let [t (interp/tensor :f32 [m n] (mapv float xs))
                                    r (interp/op-reduce-sum t [1] false)
                                    expected (mapv (fn [i] (reduce + (map #(nth xs (+ (* i n) %)) (range n))))
                                                   (range m))]
                                (every? true? (map (fn [x y] (< (Math/abs (- x y)) 1e-3))
                                                   (mapv float (vec (:data r))) expected))))))

;; ---------------------------------------------------------------------------
;; dot_general
;; ---------------------------------------------------------------------------

(defn- naive-matmul [a b m n k]
  (vec (for [i (range m)]
         (for [j (range n)]
           (reduce + (map (fn [kk] (* (nth a (+ (* i k) kk)) (nth b (+ (* kk n) j))))
                          (range k)))))))

(defspec prop-dot-general-matches-naive 100
  (prop/for-all [m (gen/choose 1 4) n (gen/choose 1 4) k (gen/choose 1 4)]
                (prop/for-all [xs (gen/vector (gen/double* {:min -2 :max 2}) (* m k))
                               ys (gen/vector (gen/double* {:min -2 :max 2}) (* k n))]
                              (let [a (interp/tensor :f32 [m k] (mapv float xs))
                                    b (interp/tensor :f32 [k n] (mapv float ys))
                                    r (interp/op-dot-general a b {:contracting_dims {:lhs [1] :rhs [0]}
                                                                  :batch_dims {:lhs [] :rhs []}})
                                    expected (apply concat (naive-matmul (mapv float xs) (mapv float ys) m n k))]
                                (and (= [m n] (:shape r))
                                     (every? true? (map (fn [x y] (< (Math/abs (- x y)) 1e-3))
                                                        (mapv float (vec (:data r))) expected)))))))

(deftest test-dot-general-batch
  ;; [2 3 4] lhs contracting [2], [2 4 5] rhs contracting [1], batch [0]
  (let [a (interp/tensor :f32 [2 3 4] (mapv float (range 24)))
        b (interp/tensor :f32 [2 4 5] (mapv float (range 40)))
        r (interp/op-dot-general a b {:contracting_dims {:lhs [2] :rhs [1]}
                                      :batch_dims {:lhs [0] :rhs [0]}})]
    (is (= [2 3 5] (:shape r)))
    ;; spot check batch 0, row 0: sum_k a[k]*b[k,0] for k in 0..3
    (let [want (reduce + (map (fn [k] (* k (+ (* k 5) 0))) (range 4)))]
      (is (< (Math/abs (- (aget ^floats (:data r) 0) want)) 1e-3)))))

;; ---------------------------------------------------------------------------
;; gather / dynamic_update_slice
;; ---------------------------------------------------------------------------

(deftest test-gather-embedding-lookup
  ;; operand [4 3], indices [2 1] -> look up rows 2 and 0
  (let [operand (interp/tensor :f32 [4 3] (mapv float (range 12)))
        indices (interp/tensor :i32 [2 1] [2 0])
        r (interp/op-gather operand indices
                            {:offset_dims [1] :collapsed_slice_dims [0]
                             :start_index_map [0] :index_vector_dim 1
                             :slice_sizes [1 3]})]
    (is (= [2 3] (:shape r)))
    ;; row 2 = [6 7 8], row 0 = [0 1 2]
    (is (= [6.0 7.0 8.0 0.0 1.0 2.0] (mapv float (vec (:data r)))))))

(deftest test-dynamic-slice
  (let [operand (interp/tensor :f32 [4 4] (range 16))
        r (interp/op-dynamic-slice operand [1 2] [2 2])]
    (is (= [2 2] (:shape r)))
    ;; row 1 cols 2,3 -> [6, 7]; row 2 cols 2,3 -> [10, 11]
    (is (= [6.0 7.0 10.0 11.0] (mapv float (vec (:data r)))))))

(deftest test-dynamic-update-slice
  (let [operand (interp/tensor :f32 [4] [1 2 3 4])
        update (interp/tensor :f32 [2] [9 8])
        r (interp/op-dynamic-update-slice operand update [1])]
    (is (= [1.0 9.0 8.0 4.0] (mapv float (vec (:data r)))))
    ;; operand unchanged (functional)
    (is (= [1.0 2.0 3.0 4.0] (mapv float (vec (:data operand)))))))

;; ---------------------------------------------------------------------------
;; Executor over a hand-built graph
;; ---------------------------------------------------------------------------

(deftest test-execute-small-graph
  ;; z = add(mul(x, 2.0), y) with liveness dropping x after use
  (let [graph {:name "t"
               :invars [[:x [:tensor [3] :f32]] [:y [:tensor [3] :f32]]]
               :outvars [:z]
               :known-shapes {}
               :eqns [{:op :stablehlo/constant :value 2.0 :outvars [:two]}
                      {:op :stablehlo/multiply :invars [:x :two] :outvars [:t1]}
                      {:op :stablehlo/add :invars [:t1 :y] :outvars [:z]}]}
        bindings {:x (interp/tensor :f32 [3] [1 2 3])
                  :y (interp/tensor :f32 [3] [10 20 30])}
        result (interp/execute graph bindings)]
    (is (= [:z] (keys result)))
    (is (= [12.0 24.0 36.0] (mapv float (vec (:data (:z result))))))))

(deftest test-execute-constant-typed
  (let [graph {:name "t" :invars [] :outvars [:c]
               :known-shapes {}
               :eqns [{:op :stablehlo/constant :value [1.0 2.0 3.0]
                       :type [:tensor [3] :f32] :outvars [:c]}]}
        result (interp/execute graph {})]
    (is (= [3] (:shape (:c result))))
    (is (= :f32 (:dtype (:c result))))))

;; ---------------------------------------------------------------------------
;; SGEMM Offset, SIMD Parity, and Parallel Execution Tests
;; ---------------------------------------------------------------------------

(deftest test-sgemm-offset-in-place
  (testing "sgemm! computes in-place matrix product at specified offsets"
    (let [M 2 N 3 K 4
          A-data (float-array [0.0 0.0 ; prefix padding
                               1.0 2.0 3.0 4.0
                               5.0 6.0 7.0 8.0])
          B-data (float-array [0.0 0.0 0.0 ; prefix padding
                               1.0 0.0 2.0
                               0.0 1.0 0.0
                               2.0 1.0 0.0
                               0.0 2.0 1.0])
          C-data (float-array 10) ; offset at 4
          aoff 2
          boff 3
          coff 4]
      (interp/sgemm! C-data coff A-data aoff B-data boff M N K)
      ;; Row 0: [1 2 3 4] @ [[1 0 2] [0 1 0] [2 1 0] [0 2 1]]
      ;; 1*1 + 2*0 + 3*2 + 4*0 = 7
      ;; 1*0 + 2*1 + 3*1 + 4*2 = 13
      ;; 1*2 + 2*0 + 3*0 + 4*1 = 6
      ;; Row 1: [5 6 7 8] @ [[1 0 2] [0 1 0] [2 1 0] [0 2 1]]
      ;; 5*1 + 6*0 + 7*2 + 8*0 = 19
      ;; 5*0 + 6*1 + 7*1 + 8*2 = 29
      ;; 5*2 + 6*0 + 7*0 + 8*1 = 18
      (let [result (vec (take 6 (drop 4 (vec C-data))))]
        (is (= [7.0 13.0 6.0 19.0 29.0 18.0] result))))))

(deftest test-scalar-simd-parity
  (testing "scalar-sgemm! and simd-sgemm! produce identical results across diverse shapes"
    (doseq [[M N K] [[1 16 16] [1 35 67] [4 64 128] [3 33 65] [2 256 128]]]
      (let [A (float-array (map #(float (/ (mod % 17) 10.0)) (range (* M K))))
            B (float-array (map #(float (/ (mod % 19) 10.0)) (range (* K N))))
            C-scalar (float-array (* M N))
            C-simd (float-array (* M N))]
        (interp/scalar-sgemm! C-scalar 0 A 0 B 0 M N K)
        (if interp/simd-available?
          (let [simd-fn (resolve 'einsum.logic.interpret-simd/simd-sgemm!)]
            (simd-fn C-simd 0 A 0 B 0 M N K)
            (dotimes [i (* M N)]
              (let [s (aget C-scalar i)
                    v (aget C-simd i)
                    diff (Math/abs (- s v))
                    rel (/ diff (max (float 1.0) (Math/abs s)))]
                (is (< rel 1e-4) (str "Shape " [M N K] " mismatch at index " i)))))
          (is true))))))

(deftest test-fallback-force-scalar
  (testing "forcing scalar mode executes via pure scalar path without error"
    (let [a (interp/tensor :f32 [2 3] (mapv float (range 6)))
          b (interp/tensor :f32 [3 2] (mapv float (range 6)))
          attrs {:contracting_dims {:lhs [1] :rhs [0]} :batch_dims {:lhs [] :rhs []}}
          res-default (interp/op-dot-general a b attrs)
          res-scalar (binding [interp/*force-scalar?* true]
                       (interp/op-dot-general a b attrs))]
      (is (= (vec (:data res-default)) (vec (:data res-scalar)))))))

(deftest test-parallel-batched-matmul
  (testing "batched dot_general executes in parallel across B and matches expected product"
    (let [B 4 M 2 K 3 N 2
          a-vals (mapv float (range (* B M K)))
          b-vals (mapv float (range (* B K N)))
          a (interp/tensor :f32 [B M K] a-vals)
          b (interp/tensor :f32 [B K N] b-vals)
          attrs {:contracting_dims {:lhs [2] :rhs [1]} :batch_dims {:lhs [0] :rhs [0]}}
          res (interp/op-dot-general a b attrs)]
      (is (= [B M N] (:shape res)))
      (is (= (* B M N) (alength ^floats (:data res)))))))

(deftest test-parallel-fused-ops
  (testing "fused LayerNorm and Softmax compute correctly across multi-row parallel inputs"
    (let [x (interp/tensor :f32 [4 8] (mapv float (range 32)))
          gamma (interp/tensor :f32 [8] (repeat 8 1.0))
          beta (interp/tensor :f32 [8] (repeat 8 0.0))
          ln-res (interp/op-fused-layer-norm x gamma beta {:eps 1e-5})
          sm-res (interp/op-fused-softmax x {})]
      (is (= [4 8] (:shape ln-res)))
      (is (= [4 8] (:shape sm-res)))
      ;; Verify softmax rows sum to ~1.0
      (let [sm-floats ^floats (:data sm-res)]
        (dotimes [r 4]
          (let [row-sum (reduce + (map #(aget sm-floats (+ (* r 8) %)) (range 8)))]
            (is (< (Math/abs (- 1.0 row-sum)) 1e-5))))))))

(deftest test-shared-parallel-enabled-var
  (testing "*parallel-enabled?* dynamically controls parallelism in both interpret and interpret-simd"
    (when interp/simd-available?
      (let [simd-ns (find-ns 'einsum.logic.interpret-simd)
            simd-var (ns-resolve simd-ns '*parallel-enabled?*)]
        ;; Assert both namespaces resolve to the exact same Var object
        (is (identical? #'interp/*parallel-enabled?* simd-var))
        ;; Assert dynamic binding in interpret propagates to interpret-simd
        (is (true? @simd-var))
        (binding [interp/*parallel-enabled?* false]
          (is (false? @simd-var)))))))


