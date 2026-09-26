(ns einsum.logic.exl3
  "EXL3 Quantization (turboderp-org/exllamav3) Procedural Codebooks, Trellis Decoding, and OpenXLA PJRT Lowers.
   Delegates to einsum.quant.exl3 for backwards compatibility."
  (:require [einsum.quant.exl3 :as e]))

(def make-tc-perm e/make-tc-perm)
(def make-tc-perm-inv e/make-tc-perm-inv)
(def tc-perm e/tc-perm)
(def tc-perm-inv e/tc-perm-inv)
(def generate-mul1-codebook e/generate-mul1-codebook)
(def generate-mcg-codebook e/generate-mcg-codebook)
(def decode-state-scalar e/decode-state-scalar)
(def dequant-tile-ref e/dequant-tile-ref)
(def fwht128-in-place! e/fwht128-in-place!)
(def hadamard-block-128-ref e/hadamard-block-128-ref)
(def float->bf16-short e/float->bf16-short)
(def dequant-exl3-matrix e/dequant-exl3-matrix)
(def exl3-linear-ast e/exl3-linear-ast)
(def quantize-weights-per-row-int8 e/quantize-weights-per-row-int8)
(def quantize-weights-per-row-int4 e/quantize-weights-per-row-int4)
(def repack-int4-to-rdna3 e/repack-int4-to-rdna3)
(def transpose-scales-for-rdna3 e/transpose-scales-for-rdna3)
(def make-rdna3-qzeros e/make-rdna3-qzeros)
