(ns einsum.quant.ternary-test
  "Generative property and unit tests for Ternary Quantization ({-1, 0, 1}, 1.58-bit / 2-bit packing)."
  (:require [einsum.quant.ternary :as ternary]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Round-Trip Bit Operations & Byte Packing
;; ==============================================================================

(deftest test-encode-decode-scalar
  (testing "Scalar 2-bit encoding and decoding round-trips for {-1.0, 0.0, 1.0}"
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
;; 3. BitNet AbsMean Quantization
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

;; ==============================================================================
;; 4. Declarative Tensor Logic AST Generation
;; ==============================================================================

(deftest test-ternary-linear-ast
  (testing "Constructs valid Declarative Tensor Logic block"
    (let [ast (ternary/ternary-linear-ast [:y :b :d_out] [:x :b :d_in] [:w_packed :d_in_packed :d_out] [:gamma])]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ternary_linear (:name (second ast)))))))
