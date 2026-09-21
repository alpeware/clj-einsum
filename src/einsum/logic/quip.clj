(ns einsum.logic.quip
  "QuIP# Incoherence Processing (FWHT / RHT) and E8 Lattice Codebooks for OpenXLA PJRT.
   Delegates to einsum.quant.quip for backwards compatibility."
  (:require [einsum.quant.quip :as q]))

(def fwht-ref q/fwht-ref)
(def rht-ref q/rht-ref)
(def rht-inv-ref q/rht-inv-ref)
(def generate-e8p-codebook q/generate-e8p-codebook)
(def quip-linear-ast q/quip-linear-ast)
