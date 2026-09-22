(ns einsum.quant.ternary-test
  "Generative property and unit tests for Ternary Quantization ({-1, 0, 1}, 1.58-bit / 2-bit packing)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [einsum.models.gemma4 :as gemma4]
            [einsum.quant.ternary :as ternary]))

;; ==============================================================================
;; 1. Round-Trip Bit Operations & Byte Packing
;; ==============================================================================

(deftest test-encode-decode-scalar
  (testing "Scalar 2-bit biased encoding and decoding round-trips for {-1.0, 0.0, 1.0}"
    (is (= 0.0 (ternary/decode-ternary-2bit (ternary/encode-ternary-2bit 0.0))))
    (is (= 1.0 (ternary/decode-ternary-2bit (ternary/encode-ternary-2bit 1.0))))
    (is (= -1.0 (ternary/decode-ternary-2bit (ternary/encode-ternary-2bit -1.0))))))

(defspec prop-ternary-byte-round-trip
  100
  (prop/for-all [v0 (gen/elements [-1.0 0.0 1.0])
                 v1 (gen/elements [-1.0 0.0 1.0])
                 v2 (gen/elements [-1.0 0.0 1.0])
                 v3 (gen/elements [-1.0 0.0 1.0])]
                (let [b (ternary/pack-ternary-byte v0 v1 v2 v3)
                      unpacked (ternary/unpack-ternary-byte b)]
                  (= [v0 v1 v2 v3] unpacked))))

;; ==============================================================================
;; 2. Generative Property Tests for Vector Pack / Unpack
;; ==============================================================================

(defspec prop-ternary-vector-pack-unpack-round-trip
  50
  (prop/for-all [_k (gen/choose 1 32)
                 weights (gen/vector (gen/elements [-1.0 0.0 1.0]) 64)]
                (let [packed (ternary/pack-ternary-2bit-ref weights)
                      unpacked (ternary/unpack-ternary-2bit-ref packed)]
                  (= (vec weights) unpacked))))

(deftest test-unpack-with-scale
  (testing "Unpack applies scale multiplier"
    (let [weights [1.0 -1.0 0.0 1.0]
          packed (ternary/pack-ternary-2bit-ref weights)
          unpacked (ternary/unpack-ternary-2bit-ref packed 0.5)]
      (is (= [0.5 -0.5 0.0 0.5] unpacked)))))

;; ==============================================================================
;; 3. BitNet AbsMean Quantization & Per-Row Quantization
;; ==============================================================================

(deftest test-absmean-quantization
  (testing "Quantize continuous weights to 2-bit packed ternary representation"
    (let [weights [1.8 -2.1 0.1 -0.2 3.0 -2.9 0.05 -0.1]
          res (ternary/quantize-weights-absmean-ternary weights 2 4)]
      (is (= [2 4] (:shape res)))
      (is (pos? (:scale res)))
      (is (= 2 (count (:data res))))
      (let [unpacked (ternary/unpack-ternary-2bit-ref (:data res) 1.0)]
        (is (= [1.0 -1.0 0.0 0.0 1.0 -1.0 0.0 0.0] unpacked))))))

(defspec prop-quantize-weights-per-row-ternary-invariants
  50
  (prop/for-all [rows (gen/choose 2 16)
                 quarter-cols (gen/choose 2 16)]
                (let [cols (* quarter-cols 4)
                      total (* rows cols)
                      w-arr (float-array (repeatedly total #(- (* 4.0 (rand)) 2.0)))
                      {:keys [data scales shape scale-shape]} (ternary/quantize-weights-per-row-ternary w-arr rows cols {:as :f32})]
                  (and (= [rows quarter-cols] shape)
                       (= [rows] scale-shape)
                       (= (* rows quarter-cols) (alength ^bytes data))
                       (= rows (alength ^floats scales))))))

(defspec prop-norm-preserving-lloyd-max-invariants
  50
  (prop/for-all [quarter-cols (gen/choose 8 32)]
                (let [cols (* quarter-cols 4)
                      rnd (java.util.Random. 42)
                      w-arr (float-array (repeatedly cols #(.nextGaussian rnd)))
                      norm-orig (Math/sqrt (areduce w-arr i s 0.0 (+ s (* (aget w-arr i) (aget w-arr i)))))
                      {:keys [data scales]} (ternary/quantize-weights-per-row-ternary w-arr 1 cols {:as :f32})
                      gamma (double (aget ^floats scales 0))
                      unpacked (ternary/unpack-ternary-2bit-ref data gamma)
                      norm-rec (Math/sqrt (reduce (fn ^double [^double s ^double v] (+ s (* v v))) 0.0 unpacked))
                      dot (reduce + 0.0 (map * (vec w-arr) unpacked))
                      cos-sim (/ dot (* norm-orig norm-rec))
                      norm-ratio (/ norm-rec norm-orig)]
                  (and (> gamma 0.0)
                       ;; Norm-preserving Lloyd-Max keeps norm within 10% of original
                       (> norm-ratio 0.90)
                       (< norm-ratio 1.10)
                       ;; High directional fidelity (cosine similarity >= 0.85)
                       (>= cos-sim 0.85)))))

(defspec prop-quantize-weights-grouped-ternary-invariants
  50
  (prop/for-all [rows (gen/choose 2 8)
                 num-groups (gen/choose 2 4)
                 quarter-g (gen/choose 2 4)]
                (let [group-size (* quarter-g 4)
                      cols (* num-groups group-size)
                      total (* rows cols)
                      w-arr (float-array (repeatedly total #(- (* 4.0 (rand)) 2.0)))
                      {:keys [data scales shape scale-shape]}
                      (ternary/quantize-weights-per-row-ternary w-arr rows cols {:as :f32 :group-size group-size})]
                  (and (= [rows (quot cols 4)] shape)
                       (= [rows num-groups] scale-shape)
                       (= (* rows (quot cols 4)) (alength ^bytes data))
                       (= (* rows num-groups) (alength ^floats scales))))))

;; ==============================================================================
;; 4. Declarative Tensor Logic AST Generation
;; ==============================================================================

(deftest test-ternary-linear-ast
  (testing "Constructs valid Declarative Tensor Logic block"
    (let [ast (ternary/ternary-linear-ast [:y :b :d_out] [:x :b :d_in] [:w_packed :d_out :d_in_quarter] [:gamma :d_out])]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ternary_linear (:name (second ast)))))))

(deftest test-gemma4-linear-proj-ternary
  (testing "Gemma 4 linear projection emits :ternary-unpack when is-ternary? is active"
    (let [ast (gemma4/gemma4-linear-proj [:q-raw :b :p :qd] [:x-norm :b :p :d] [:q-w :qd :d]
                                         false false true :q-scale :bf16 {})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :q-raw_ternary_proj (:name (second ast))))
      (let [unpack-op (nth ast 2)]
        (is (= :ternary-unpack (first unpack-op)))
        (is (= :q-w_scaled (first (second unpack-op))))
        (is (= [:q-w :qd :d_quarter] (nth unpack-op 2)))
        (is (= [:q-scale :qd] (nth unpack-op 3)))))))

;; ==============================================================================
;; 5. OpenXLA PJRT StableHLO Execution Parity
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

