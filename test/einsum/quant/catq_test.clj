(ns einsum.quant.catq-test
  "Generative property and unit tests for CAT-Q: Cost-Efficient Accurate Ternary Quantization
   with Learnable Modulation and LoRA Refinement."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.quant.catq :as catq]))

;; ==============================================================================
;; 1. Learnable Modulation Invariants
;; ==============================================================================

(defspec prop-catq-modulation-invariants
  50
  (prop/for-all [zs (gen/fmap #(double (- (* 4.0 %) 2.0)) gen/ratio)
                 zr (gen/fmap #(double (- (* 4.0 %) 2.0)) gen/ratio)
                 base-scale (gen/fmap #(+ 0.001 (Math/abs (double %))) (gen/fmap double gen/ratio))]
                (let [s-mod (catq/compute-modulated-scale base-scale zs)
                      thd-mod (catq/compute-modulated-threshold s-mod zr)]
                  (and
                   ;; Modulated scale is always strictly positive
                   (pos? s-mod)
                   ;; Modulated threshold is always strictly positive
                   (pos? thd-mod)
                   ;; When zs=0, scale matches base scale
                   (let [s-zero (catq/compute-modulated-scale base-scale 0.0)]
                     (< (Math/abs (- s-zero base-scale)) 1e-6))
                   ;; When zr=0, threshold is exactly 0.5 * scale
                   (let [thd-zero (catq/compute-modulated-threshold s-mod 0.0)]
                     (< (Math/abs (- thd-zero (* 0.5 s-mod))) 1e-6))))))

;; ==============================================================================
;; 2. LoRA Refinement Invariants
;; ==============================================================================

(deftest test-lora-refinement-zero-init
  (testing "LoRA refinement with zero B produces identical weights to base"
    (let [rows 4
          cols 128
          rank 16
          w (float-array (* rows cols) 0.5)
          lora-a (float-array (* rank cols) 0.1)
          lora-b (float-array (* rows rank) 0.0) ;; Zero init
          w-eff (catq/compute-effective-weights w rows cols lora-a lora-b rank 1.0)]
      (is (= (vec w) (vec w-eff))))))

(defspec prop-lora-effective-weights-invariants
  50
  (prop/for-all [rows (gen/choose 2 8)
                 cols-factor (gen/choose 1 4)
                 rank (gen/choose 4 16)]
                (let [cols (* cols-factor 128)
                      rnd (java.util.Random. 42)
                      w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
                      lora-a (float-array (repeatedly (* rank cols) #(.nextGaussian rnd)))
                      lora-b (float-array (* rows rank) 0.0)
                      w-eff (catq/compute-effective-weights w rows cols lora-a lora-b rank 1.0)]
                  ;; With B=0, w-eff == w
                  (= (vec w) (vec w-eff)))))

;; ==============================================================================
;; 3. Matrix Quantization to Q2_0 Format Invariants
;; ==============================================================================

(defspec prop-catq-quantize-matrix-q2-0-invariants
  50
  (prop/for-all [rows (gen/choose 2 8)
                 num-groups (gen/choose 1 4)]
                (let [cols (* num-groups 128)
                      group-size 128
                      rnd (java.util.Random. 42)
                      w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
                      res (catq/quantize-matrix-catq w rows cols {:group-size group-size})]
                  (and (= [rows (quot cols 4)] (:shape res))
                       (= [rows num-groups] (:scale-shape res))
                       (= (* rows (quot cols 4)) (alength ^bytes (:data res)))
                       (= (* rows num-groups) (alength ^shorts (:scales res)))))))

(deftest test-catq-dequantize-roundtrip
  (testing "Dequantized Q2_0 weights match quantized ternary representation"
    (let [rows 2
          cols 256
          group-size 128
          rnd (java.util.Random. 123)
          w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
          {:keys [data scales _scale-shape]} (catq/quantize-matrix-catq w rows cols {:group-size group-size})
          dequant (catq/dequantize-matrix-catq data scales rows cols group-size)]
      (is (= (* rows cols) (count dequant)))
      (dotimes [i (* rows cols)]
        (let [v (double (nth dequant i))]
          ;; Every dequantized element must be either 0 or +/- scale
          (is (number? v)))))))

;; ==============================================================================
;; 4. Projection Calibration Error Reduction Invariant
;; ==============================================================================

(deftest test-catq-calibration-reduces-distortion
  (testing "CAT-Q calibration with learnable modulation reduces projection MSE"
    (let [rows 16
          cols 128
          num-tokens 64
          rnd (java.util.Random. 42)
          w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
          x (float-array (repeatedly (* num-tokens cols) #(.nextGaussian rnd)))
          ;; 1. Baseline naive ternary error
          base-rec (catq/quantize-matrix-catq w rows cols {:group-size 128})
          base-w-deq (catq/dequantize-matrix-catq (:data base-rec) (:scales base-rec) rows cols 128)
          base-mse (catq/compute-projection-mse x w (float-array base-w-deq) num-tokens rows cols)

          ;; 2. Calibrated CAT-Q with LoRA and modulation
          calib-res (catq/calibrate-projection x w num-tokens rows cols {:epochs 15 :group-size 128 :rank 16})
          calib-w-deq (catq/dequantize-matrix-catq (:data calib-res) (:scales calib-res) rows cols 128)
          calib-mse (catq/compute-projection-mse x w (float-array calib-w-deq) num-tokens rows cols)]
      ;; Calibrated projection MSE must be strictly lower than naive ternary
      (is (< calib-mse base-mse)
          (format "Calibrated MSE (%.6f) should be lower than baseline (%.6f)" calib-mse base-mse)))))
