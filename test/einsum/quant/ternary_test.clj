(ns einsum.quant.ternary-test
  "Generative property and unit tests for Ternary Quantization ({-1, 0, 1}, 1.58-bit / 2-bit packing)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
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



