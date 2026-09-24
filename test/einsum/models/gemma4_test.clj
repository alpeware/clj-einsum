(ns einsum.models.gemma4-test
  "Generative specification tests for modularized Gemma 4 domain namespaces:
   config, weights, kernels, and runtime."
  (:require [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.weights :as weights]
            [einsum.models.gemma4.runtime :as rt]))

;; -----------------------------------------------------------------------------
;; 1. Config Invariants
;; -----------------------------------------------------------------------------

(defspec prop-resolve-weight-shape-direct 50
  (prop/for-all [rows (gen/choose 1 1024)
                 cols (gen/choose 1 1024)]
                (let [header {"model.layers.0.self_attn.q_proj.weight" {"shape" [rows cols]}}
                      resolved (cfg/resolve-weight-shape header "model.layers.0.self_attn.q_proj.weight" [0 0])]
                  (= resolved [rows cols]))))

(defspec prop-resolve-weight-shape-trellis-fallback 50
  (prop/for-all [rows (gen/choose 1 1024)
                 cols (gen/choose 1 1024)]
                (let [header {"model.layers.0.self_attn.q_proj.svh" {"shape" [rows 1]}
                              "model.layers.0.self_attn.q_proj.suh" {"shape" [cols 1]}}
                      resolved (cfg/resolve-weight-shape header "model.layers.0.self_attn.q_proj.weight" [0 0])]
                  (= resolved [rows cols]))))

(defspec prop-resolve-weight-shape-v-proj-fallback 50
  (prop/for-all [rows (gen/choose 1 1024)
                 cols (gen/choose 1 1024)]
                (let [header {"model.layers.0.self_attn.k_proj.svh" {"shape" [rows 1]}
                              "model.layers.0.self_attn.k_proj.suh" {"shape" [cols 1]}}
                      resolved (cfg/resolve-weight-shape header "model.layers.0.self_attn.v_proj.weight" [0 0])]
                  (= resolved [rows cols]))))

(defspec prop-max-safe-prefill-seq-len 50
  (prop/for-all [is-int4 gen/boolean
                 is-ternary gen/boolean
                 num-layers (gen/choose 1 80)]
                (let [config {:is-int4 is-int4
                              :is-ternary is-ternary
                              :num-layers num-layers}
                      limit (cfg/max-safe-prefill-seq-len config)]
                  (if (or is-int4 is-ternary (>= num-layers 60))
                    (= limit 2048)
                    (= limit 8192)))))

;; -----------------------------------------------------------------------------
;; 2. Weights Invariants
;; -----------------------------------------------------------------------------

(defspec prop-floats-bf16-conversion 50
  (prop/for-all [floats (gen/vector (gen/fmap float (gen/choose -100 100)) 1 64)]
                (let [arr (float-array floats)
                      shorts (weights/floats->bf16-shorts arr)]
                  (and (= (alength shorts) (count floats))
                       (every? (fn [s]
                                 (let [bits (unchecked-int (bit-shift-left (long (bit-and (int s) 0xffff)) 16))
                                       f (Float/intBitsToFloat bits)]
                                   (number? f)))
                               shorts)))))

(defspec prop-quantize-bf16-to-int8-bounds 50
  (prop/for-all [floats (gen/vector (gen/fmap float (gen/choose -50 50)) 1 64)]
                (let [arr (float-array floats)
                      shorts (weights/floats->bf16-shorts arr)
                      {:keys [data scale]} (weights/quantize-bf16-to-int8 shorts)]
                  (and (pos? scale)
                       (= (alength ^bytes data) (count floats))
                       (every? #(and (>= % -127) (<= % 127)) (vec data))))))

;; -----------------------------------------------------------------------------
;; 3. Runtime Invariants
;; -----------------------------------------------------------------------------

(defspec prop-common-prefix-len-soundness 50
  (prop/for-all [common (gen/vector gen/nat 0 20)
                 suffix-a (gen/vector (gen/choose 100 200) 0 10)
                 suffix-b (gen/vector (gen/choose 300 400) 0 10)]
                (let [xs (vec (concat common suffix-a))
                      ys (vec (concat common suffix-b))
                      l (rt/common-prefix-len xs ys)]
                  (and (= l (count common))
                       (= (rt/common-prefix-len xs xs) (count xs))
                       (= (rt/common-prefix-len xs ys) (rt/common-prefix-len ys xs))))))

(defspec prop-argmax-host-accuracy 50
  (prop/for-all [floats (gen/vector (gen/fmap float (gen/choose -100 100)) 1 32)
                 target-idx (gen/choose 0 31)]
                (let [valid-idx (mod target-idx (count floats))
                      mod-floats (assoc floats valid-idx (float 999.0))
                      arr (float-array mod-floats)
                      result (rt/argmax-host arr)]
                  (= result valid-idx))))
