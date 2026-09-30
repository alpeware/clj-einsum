(ns einsum.models.gemma4-test
  "Generative specification tests for modularized Gemma 4 domain namespaces:
   config, weights, kernels, and runtime."
  (:require [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.compiler.stablehlo :as shlo]
            [einsum.logic.lower :as lower]
            [einsum.models.gemma :as gemma-logic]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]
            [einsum.models.gemma4.weights :as weights]))

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

;; -----------------------------------------------------------------------------
;; 4. Scoring Invariants
;; -----------------------------------------------------------------------------

(deftest test-gemma4-scoring-graph-generation
  ;; A small 1-layer gemma4 model scoring graph lowers to a valid StableHLO SSA graph
  ;; with target_log_probs as its single output.
  (let [cfg {:num-layers 1
             :hidden-dim 64
             :head-dim 32
             :num-heads 2
             :num-kv-heads 1
             :vocab-size 256
             :intermediate-dim 128
             :max-seq-len 8
             :weight-dtype :f32
             :last-token-only? false
             :total-pl-dim 0
             :pl-dim 0}
        max-seq-len 8
        model-invars (kernels/build-tensor-logic-invars cfg max-seq-len)
        invars (vec (concat [[:x [:tensor [1 max-seq-len] :i32]]
                             [:targets [:tensor [1 max-seq-len] :i32]]]
                            (rest model-invars)))
        ast [:block {:name :gemma4_scoring}
             (gemma-logic/gemma4-model-ast cfg)
             [:log-softmax [:log_probs :b :p :v] [:logits :b :p :v] {:axis 2}]
             [:gather [:target_log_probs :b :p] [:log_probs :b :p :v] [:targets :b :p] {:axis 2}]]
        targets #{:target_log_probs}
        graph (lower/ast->graph "gemma4_scoring" invars ast targets)]
    (is (shlo/validate-graph graph))
    (is (= [:target_log_probs] (:outvars graph)))
    (let [ops (mapv :op (:eqns graph))]
      (is (some #(= :stablehlo/gather %) ops))
      (is (some #(= :stablehlo/log %) ops)))))

(defspec prop-score-sequence-log-probs-validation 50
  (prop/for-all [tokens (gen/vector (gen/choose 1 1000) 0 1)]
    ;; Should throw exception for tokens count < 2
                (try
                  (rt/score-sequence-log-probs {} nil tokens)
                  false
                  (catch clojure.lang.ExceptionInfo e
                    (= "Sequence must contain at least 2 tokens to score target log-probabilities"
                       (.getMessage e))))))

