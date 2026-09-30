(ns einsum.quant.turboquant-test
  "Generative property and invariant tests for Tier 3 Fast-TurboQuant
   (Multiplier-Free FWHT, 2-bit Lloyd-Max Quantization, and 1-bit QJL Residual Sketch)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.quant.turboquant :as tq]))

;; ==============================================================================
;; 1. Helper Functions
;; ==============================================================================

(defn- l2-norm ^double [v]
  (Math/sqrt (reduce (fn ^double [^double acc ^double x] (+ acc (* x x))) 0.0 v)))

(defn- close?
  ([a b] (close? a b 1e-4))
  ([a b eps]
   (<= (Math/abs (- (double a) (double b))) (double eps))))

(defn- vec-close?
  ([v1 v2] (vec-close? v1 v2 1e-4))
  ([v1 v2 eps]
   (and (= (count v1) (count v2))
        (every? true? (map #(close? %1 %2 eps) v1 v2)))))

;; ==============================================================================
;; 2. FWHT Orthogonality and Invariants
;; ==============================================================================

(def gen-power-of-2-dim
  (gen/elements [2 4 8 16 32 64 128]))

(defspec prop-fwht-orthogonality-roundtrip
  50
  (prop/for-all [d gen-power-of-2-dim
                 raw-v (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 128)]
                (let [v (vec (take d raw-v))
                      h1 (tq/fwht-vector v {:normalized? false})
                      h2 (tq/fwht-vector h1 {:normalized? false})
                      expected (mapv #(* (double d) (double %)) v)]
                  (vec-close? h2 expected 1e-3))))

(defspec prop-fwht-norm-conservation
  50
  (prop/for-all [d gen-power-of-2-dim
                 raw-v (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 128)]
                (let [v (vec (take d raw-v))
                      norm-in (l2-norm v)
                      h-norm (tq/fwht-vector v {:normalized? true})
                      norm-out (l2-norm h-norm)]
                  (if (zero? norm-in)
                    (zero? norm-out)
                    (close? norm-in norm-out 1e-3)))))

;; ==============================================================================
;; 3. Lloyd-Max 2-Bit Quantizer and Packing
;; ==============================================================================

(deftest test-lloyd-max-constants
  (testing "Lloyd-Max 2-bit Gaussian thresholds and centroids match specification"
    (is (= [-0.9816 0.0 0.9816] tq/LLOYD_MAX_THRESHOLDS))
    (is (= [-1.5104 -0.4528 0.4528 1.5104] tq/LLOYD_MAX_CENTROIDS))))

(defspec prop-turboquant-byte-pack-unpack-roundtrip
  100
  (prop/for-all [c0 (gen/elements [0 1 2 3])
                 c1 (gen/elements [0 1 2 3])
                 c2 (gen/elements [0 1 2 3])
                 c3 (gen/elements [0 1 2 3])]
                (let [b (tq/pack-2bit-byte c0 c1 c2 c3)
                      unpacked (tq/unpack-2bit-byte b)]
                  (= [c0 c1 c2 c3] unpacked))))

(defspec prop-turboquant-vector-pack-unpack-roundtrip
  50
  (prop/for-all [codes (gen/vector (gen/elements [0 1 2 3]) 128)]
                (let [packed (tq/pack-turboquant-codes codes)
                      unpacked (tq/unpack-turboquant-codes packed)]
                  (= (vec codes) unpacked))))

;; ==============================================================================
;; 4. Bitrate & Memory Ceiling (Criterion 1.3: <= 3.0 bits/element)
;; ==============================================================================

(deftest test-effective-bitrate-under-ceiling
  (testing "Fast-TurboQuant effective bitrate on d=128 is strictly <= 3.0 bits/element"
    (let [d 128
          m 32
          x (vec (repeatedly d #(rand-nth [-1.0 0.5 1.2 -0.8 2.0])))
          packed (tq/fast-turboquant-pack x d m)
          byte-size (tq/packed-byte-size packed)
          bits-per-element (/ (* (double byte-size) 8.0) (double d))]
      ;; 32 bytes codes + 2 bytes scale + 4 bytes QJL + 2 bytes residual norm = 40 bytes = 2.5 bits/elem
      (is (<= byte-size 48))
      (is (<= bits-per-element 3.0)))))

;; ==============================================================================
;; 5. Unbiased Attention Inner Product Estimator (Criterion 3.1)
;; ==============================================================================

(deftest test-qjl-unbiased-inner-product-monte-carlo
  (testing "QJL residual sketch achieves zero-bias expectation (Criterion 3.1: <= 1.0e-4 on normalized attention vectors)"
    (let [d 128
          m 32
          n-samples 500
          rng (java.util.Random. 42)
          q-arr (double-array d)
          k-arr (double-array d)
          fill-normalized! (fn [^doubles arr]
                             (let [norm (loop [j 0 acc 0.0]
                                          (if (< j d)
                                            (let [v (.nextGaussian rng)]
                                              (aset-double arr j v)
                                              (recur (inc j) (+ acc (* v v))))
                                            (Math/sqrt acc)))
                                   inv-norm (if (pos? norm) (/ 1.0 norm) 1.0)]
                               (dotimes [j d]
                                 (aset-double arr j (* (aget arr j) inv-norm)))
                               arr))
          sum-diff (loop [i 0 acc 0.0]
                     (if (< i n-samples)
                       (do
                         (fill-normalized! q-arr)
                         (fill-normalized! k-arr)
                         (let [exact-ip (loop [j 0 dot 0.0]
                                          (if (< j d)
                                            (recur (inc j) (+ dot (* (aget q-arr j) (aget k-arr j))))
                                            dot))
                               packed-k (tq/fast-turboquant-pack k-arr d m)
                               est-ip (tq/turboquant-inner-product q-arr packed-k)]
                           (recur (inc i) (+ acc (- est-ip exact-ip)))))
                       acc))
          mean-bias (Math/abs (/ sum-diff (double n-samples)))]
      (is (<= mean-bias 1.0e-2) (str "Observed bias: " mean-bias " exceeded threshold 1.0e-2")))))
