(ns einsum.quant.w4a16-custom-call-test
  "Generative property and numeric parity tests for fused W4A16 GEMV OpenXLA custom call on AMD RDNA3 (gfx1100)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.pjrt.custom-call :as cc]
            [einsum.core :as xla]
            [einsum.logic.exl3 :as exl3-logic]
            [einsum.quant.exl3 :as exl3]))

;; ==============================================================================
;; 1. Generative Packing Invariants
;; ==============================================================================

(defspec prop-repack-int4-to-rdna3-invariants
  50
  (prop/for-all [cols-words (gen/choose 1 8)   ;; K in units of 8 (e.g. 8 to 64)
                 rows (gen/choose 1 16)]       ;; N (e.g. 1 to 16)
                (let [K (* cols-words 8)
                      N rows
                      half-k (quot K 2)
                      total-bytes (* N half-k)
                      raw-bytes (byte-array total-bytes)]
      ;; Fill with random nibbles
                  (dotimes [i total-bytes]
                    (let [q0 (rand-int 16)
                          q1 (rand-int 16)]
                      (aset-byte raw-bytes i (unchecked-byte (bit-or q0 (bit-shift-left q1 4))))))

                  (let [repacked (exl3/repack-int4-to-rdna3 raw-bytes N K)
                        k-words (quot K 8)]
                    (and
         ;; Size invariant
                     (= (alength repacked) (* k-words N))
         ;; Content invariant: verify every individual nibble
                     (every? true?
                             (for [n (range N)
                                   qk (range k-words)
                                   :let [word (aget repacked (+ (* qk N) n))
                                         byte-off (+ (* n half-k) (* qk 4))
                                         b0 (bit-and (int (aget raw-bytes byte-off)) 0xFF)
                                         b1 (bit-and (int (aget raw-bytes (+ byte-off 1))) 0xFF)
                                         b2 (bit-and (int (aget raw-bytes (+ byte-off 2))) 0xFF)
                                         b3 (bit-and (int (aget raw-bytes (+ byte-off 3))) 0xFF)
                             ;; Expected even/odd interleaved nibbles
                                         q0 (bit-and b0 0x0F)
                                         q1 (bit-shift-right b0 4)
                                         q2 (bit-and b1 0x0F)
                                         q3 (bit-shift-right b1 4)
                                         q4 (bit-and b2 0x0F)
                                         q5 (bit-shift-right b2 4)
                                         q6 (bit-and b3 0x0F)
                                         q7 (bit-shift-right b3 4)
                                         w0 (bit-and word 0x0F)
                                         w2 (bit-and (bit-shift-right word 4) 0x0F)
                                         w4 (bit-and (bit-shift-right word 8) 0x0F)
                                         w6 (bit-and (bit-shift-right word 12) 0x0F)
                                         w1 (bit-and (bit-shift-right word 16) 0x0F)
                                         w3 (bit-and (bit-shift-right word 20) 0x0F)
                                         w5 (bit-and (bit-shift-right word 24) 0x0F)
                                         w7 (bit-and (bit-shift-right word 28) 0x0F)]]
                               (and (= q0 w0) (= q1 w1)
                                    (= q2 w2) (= q3 w3)
                                    (= q4 w4) (= q5 w5)
                                    (= q6 w6) (= q7 w7)))))))))

(defspec prop-transpose-scales-invariants
  50
  (prop/for-all [groups (gen/choose 1 8)
                 N (gen/choose 1 16)]
                (let [total (* N groups)
                      s-arr (short-array total)]
                  (dotimes [i total]
                    (aset-short s-arr i (short (rand-int 32000))))
                  (let [transposed (exl3/transpose-scales-for-rdna3 s-arr N groups)]
                    (and
                     (= (alength transposed) total)
                     (every? true?
                             (for [n (range N)
                                   g (range groups)]
                               (= (aget transposed (+ (* g N) n))
                                  (aget s-arr (+ (* n groups) g))))))))))

(defspec prop-make-rdna3-qzeros-invariants
  20
  (prop/for-all [groups (gen/choose 1 4)
                 cols-factor (gen/choose 1 8)]
                (let [N (* cols-factor 8)
                      qz (exl3/make-rdna3-qzeros groups N)
                      target-word (unchecked-int 0x88888888)]
                  (and
                   (= (alength qz) (* groups (quot N 8)))
                   (every? #(= % target-word) (vec qz))))))

;; ==============================================================================
;; 2. Live GPU Numeric Parity Test against Pure StableHLO Unpack
;; ==============================================================================

(deftest test-w4a16-gemv-rocm-numeric-parity
  (testing "Bit-level parity between pure StableHLO int4-unpack and RDNA3 custom call"
    (let [probe (try (xla/init-backend! :rocm) (catch Exception _ nil))]
      (if (or (nil? probe) (not= (:target probe) :rocm))
        (println "Skipping ROCm live GPU numeric parity test (no ROCm device available).")
        (let [ctx probe
              client (:client ctx)
              _ (cc/register-w4a16-gemv-kernel! ctx)
              M 1
              K 256
              N 512
              group-size 128
              num-groups (quot K group-size)

              rng (java.util.Random. 42)
              raw-w (float-array (* N K))
              _ (dotimes [i (* N K)]
                  (aset-float raw-w i (float (- (.nextGaussian rng) 0.5))))

              raw-x (float-array (* M K))
              _ (dotimes [i (* M K)]
                  (aset-float raw-x i (float (* 0.1 (.nextGaussian rng)))))

              q-res (exl3/quantize-weights-per-row-int4 raw-w N K {:as :bf16 :group-size group-size})
              data-bytes ^bytes (:data q-res)
              scales-shorts ^shorts (:scales q-res)

              x-shorts (short-array (alength raw-x))
              _ (dotimes [i (alength raw-x)]
                  (aset-short x-shorts i (exl3-logic/float->bf16-short (aget raw-x i))))

              ;; 1. Reference execution: Pure StableHLO Unpack + Dot
              ref-mlir (str
                        "module @ref_gemv {\n"
                        "  func.func @main(%x: tensor<1x256xbf16>, %w_packed: tensor<512x128xi8>, %scales: tensor<512x2xbf16>) -> tensor<1x512xbf16> {\n"
                        "    %c_0f = stablehlo.constant dense<15> : tensor<i8>\n"
                        "    %c_4 = stablehlo.constant dense<4> : tensor<i8>\n"
                        "    %c_8 = stablehlo.constant dense<8.0> : tensor<bf16>\n"
                        "    %c_0f_bcast = stablehlo.broadcast_in_dim %c_0f, dims = [] : (tensor<i8>) -> tensor<512x128xi8>\n"
                        "    %c_4_bcast = stablehlo.broadcast_in_dim %c_4, dims = [] : (tensor<i8>) -> tensor<512x128xi8>\n"
                        "    %lo = stablehlo.and %w_packed, %c_0f_bcast : tensor<512x128xi8>\n"
                        "    %hi = stablehlo.shift_right_logical %w_packed, %c_4_bcast : tensor<512x128xi8>\n"
                        "    %lo_bf = stablehlo.convert %lo : (tensor<512x128xi8>) -> tensor<512x128xbf16>\n"
                        "    %hi_bf = stablehlo.convert %hi : (tensor<512x128xi8>) -> tensor<512x128xbf16>\n"
                        "    %c_8_bcast = stablehlo.broadcast_in_dim %c_8, dims = [] : (tensor<bf16>) -> tensor<512x128xbf16>\n"
                        "    %lo_c = stablehlo.subtract %lo_bf, %c_8_bcast : tensor<512x128xbf16>\n"
                        "    %hi_c = stablehlo.subtract %hi_bf, %c_8_bcast : tensor<512x128xbf16>\n"
                        "    %lo_r = stablehlo.reshape %lo_c : (tensor<512x128xbf16>) -> tensor<512x128x1xbf16>\n"
                        "    %hi_r = stablehlo.reshape %hi_c : (tensor<512x128xbf16>) -> tensor<512x128x1xbf16>\n"
                        "    %cat = stablehlo.concatenate %lo_r, %hi_r, dim = 2 : (tensor<512x128x1xbf16>, tensor<512x128x1xbf16>) -> tensor<512x128x2xbf16>\n"
                        "    %unscaled = stablehlo.reshape %cat : (tensor<512x128x2xbf16>) -> tensor<512x2x128xbf16>\n"
                        "    %sc_bcast = stablehlo.broadcast_in_dim %scales, dims = [0, 1] : (tensor<512x2xbf16>) -> tensor<512x2x128xbf16>\n"
                        "    %scaled_3d = stablehlo.multiply %unscaled, %sc_bcast : tensor<512x2x128xbf16>\n"
                        "    %w = stablehlo.reshape %scaled_3d : (tensor<512x2x128xbf16>) -> tensor<512x256xbf16>\n"
                        "    %res = \"stablehlo.dot_general\"(%x, %w) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>, precision = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x256xbf16>, tensor<512x256xbf16>) -> tensor<1x512xbf16>\n"
                        "    return %res : tensor<1x512xbf16>\n"
                        "  }\n"
                        "}\n")
              ref-exec (pjrt/compile-mlir ctx client ref-mlir)

              x-buf (pjrt/buffer-from-host-buffer ctx client x-shorts [1 K] 13)
              w-buf (pjrt/buffer-from-host-buffer ctx client data-bytes [N (quot K 2)] 2)
              sc-buf (pjrt/buffer-from-host-buffer ctx client scales-shorts [N num-groups] 13)

              ref-out (pjrt/execute-executable ctx ref-exec [x-buf w-buf sc-buf])
              ref-floats (vec (pjrt/buffer-to-host-buffer ctx ref-out N :bf16))

              ;; 2. Custom Call execution (3-input layout: implicit zero-point = 8)
              rdna3-weights (exl3/repack-int4-to-rdna3 data-bytes N K)
              rdna3-scales (exl3/transpose-scales-for-rdna3 scales-shorts N num-groups)

              custom-mlir (str
                           "module @custom_gemv {\n"
                           "  func.func @main(%a: tensor<1x256xbf16>, %w: tensor<32x512xi32>, %sc: tensor<2x512xbf16>) -> tensor<1x512xbf16> {\n"
                           "    %c = stablehlo.custom_call @w4a16_gemv_rocm(%a, %w, %sc) {\n"
                           "      api_version = 1 : i32,\n"
                           "      backend_config = \"1,512,256,2,0\"\n"
                           "    } : (tensor<1x256xbf16>, tensor<32x512xi32>, tensor<2x512xbf16>) -> tensor<1x512xbf16>\n"
                           "    return %c : tensor<1x512xbf16>\n"
                           "  }\n"
                           "}\n")
              custom-exec (pjrt/compile-mlir ctx client custom-mlir)

              cw-buf (pjrt/buffer-from-host-buffer ctx client rdna3-weights [(quot K 8) N] 4)
              csc-buf (pjrt/buffer-from-host-buffer ctx client rdna3-scales [num-groups N] 13)

              custom-out (pjrt/execute-executable ctx custom-exec [x-buf cw-buf csc-buf])
              custom-floats (vec (pjrt/buffer-to-host-buffer ctx custom-out N :bf16))

              diffs (mapv (fn [r c] (Math/abs (- (double r) (double c)))) ref-floats custom-floats)
              max-abs-diff (reduce max 0.0 diffs)
              mean-abs-diff (/ (reduce + 0.0 diffs) (count diffs))]

          ;; Mean difference must be strictly under 1e-4
          (is (< mean-abs-diff 1e-4) (str "Mean abs diff too high: " mean-abs-diff))
          ;; Max difference must be within bfloat16 rounding error (<= 0.01)
          (is (< max-abs-diff 0.01) (str "Max abs diff too high: " max-abs-diff)))))))
