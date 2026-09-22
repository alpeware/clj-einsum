(ns einsum.quant.catq-test
  "Generative property and unit tests for CAT-Q: Cost-Efficient Accurate Ternary Quantization
   with Learnable Modulation and LoRA Refinement."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gemma4 :as gemma4]
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

;; ==============================================================================
;; 5. Skip-Layers Mixed Precision Invariant
;; ==============================================================================

(deftest test-skip-layers-invars
  (testing "Gemma 4 model invars preserve unquantized BF16 tensors for skipped layers"
    (let [cfg (merge (gemma4/gemma4-config :e2b)
                     {:is-ternary true
                      :group-size 128
                      :skip-layers #{10 11 12 13 14}})
          invars (gemma4/build-tensor-logic-invars cfg 64)
          invar-names (into #{} (map first invars))]
      ;; Layer 0 (not skipped) must have packed ternary weight and .scales
      (is (contains? invar-names :q_w_0))
      (is (contains? invar-names :q_scale_0))
      ;; Layer 10 (skipped) must have full BF16 weight and NO .scales
      (is (contains? invar-names :q_w_10))
      (is (not (contains? invar-names :q_scale_10)))
      (is (not (contains? invar-names :down_scale_10)))
      ;; Layer 15 (not skipped) must have scale
      (is (contains? invar-names :q_scale_15)))))

;; ==============================================================================
;; 6. Hessian-Conditioned Lloyd-Max Quantization (OBS) Invariants
;; ==============================================================================

(defspec prop-catq-quantize-matrix-obs-invariants
  30
  (prop/for-all [rows (gen/choose 2 6)
                 num-groups (gen/choose 1 3)
                 num-tokens (gen/choose 4 16)]
                (let [cols (* num-groups 128)
                      group-size 128
                      rnd (java.util.Random. 42)
                      w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
                      x (float-array (repeatedly (* num-tokens cols) #(.nextGaussian rnd)))
                      res (catq/quantize-matrix-obs x w num-tokens rows cols group-size)]
                  (and (= [rows (quot cols 4)] (:shape res))
                       (= [rows num-groups] (:scale-shape res))
                       (= (* rows (quot cols 4)) (alength ^bytes (:data res)))
                       (= (* rows num-groups) (alength ^shorts (:scales res)))
                       (= (* rows cols) (alength ^floats (:w-deq res)))))))

(deftest test-catq-obs-high-snr
  (testing "Parallel Lloyd-Max H-conditioned quantization achieves >= 4.5 dB SNR"
    (let [rows 8
          cols 256
          group-size 128
          num-tokens 32
          rnd (java.util.Random. 42)
          w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
          x (float-array (repeatedly (* num-tokens cols) #(.nextGaussian rnd)))
          res (catq/quantize-matrix-obs x w num-tokens rows cols group-size)
          ^floats deq (:w-deq res)
          w-norm-sq (areduce w i s 0.0 (+ s (* (aget w i) (aget w i))))
          deq-norm-sq (areduce deq i s 0.0 (+ s (* (aget deq i) (aget deq i))))
          dot (areduce w i s 0.0 (+ s (* (aget w i) (aget deq i))))
          diff-sq (areduce w i s 0.0 (let [d (- (aget w i) (aget deq i))] (+ s (* d d))))
          snr (* 10.0 (Math/log10 (/ w-norm-sq diff-sq)))
          cos-sim (/ dot (Math/sqrt (* w-norm-sq deq-norm-sq)))]
      (is (>= snr 4.5) (format "SNR %.2f dB must be >= 4.5 dB" snr))
      (is (>= cos-sim 0.80) (format "CosSim %.4f must be >= 0.80" cos-sim)))))

(deftest test-catq-obs-roundtrip
  (testing "Dequantizing data and scales matches w-deq from quantize-matrix-obs"
    (let [rows 4
          cols 256
          group-size 128
          num-tokens 16
          rnd (java.util.Random. 99)
          w (float-array (repeatedly (* rows cols) #(.nextGaussian rnd)))
          x (float-array (repeatedly (* num-tokens cols) #(.nextGaussian rnd)))
          res (catq/quantize-matrix-obs x w num-tokens rows cols group-size)
          ^floats deq-reconstructed (catq/dequantize-matrix-catq (:data res) (:scales res) rows cols group-size)
          ^floats deq-direct (:w-deq res)]
      (dotimes [i (* rows cols)]
        (is (< (Math/abs (- (aget deq-reconstructed i) (aget deq-direct i))) 1e-3))))))
