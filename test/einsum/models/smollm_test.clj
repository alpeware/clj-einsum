(ns einsum.models.smollm-test
  "Unit and generative tests for declarative Tensor Logic SmolLM model."
  (:require [einsum.core :as xla]
            [einsum.logic.ast :as ast]
            [einsum.logic.expand :as expand]
            [einsum.logic.lower :as lower]
            [einsum.models.smollm :as smollm]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-smollm-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 29)
                 seq-len (gen/elements [8 16 32])]
                (let [layer-ast (smollm/smollm-layer-ast layer-idx seq-len)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-smollm-single-layer-lowering
  (let [num-layers 1
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [49152 576] :f32]]
                      [:final_norm_w [:tensor [576] :f32]]
                      [:lm_head_w [:tensor [49152 576] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                             (range num-layers)))
        ast (smollm/smollm-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "smollm_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-smollm-config-invariants 20
  (prop/for-all [n-embd (gen/elements [576 1024])]
                (let [cfg (smollm/smollm-config {:n-embd n-embd})]
                  (and (= (:n-embd cfg) n-embd)
                       (pos? (:vocab-size cfg))
                       (pos? (:n-layer cfg))
                       (pos? (:n-head cfg))))))

(defspec prop-smollm-weight-key-map 20
  (prop/for-all [layer-idx (gen/choose 0 29)]
                (let [kmap (smollm/weight-key-map layer-idx)]
                  (and (string? (:input-ln-w kmap))
                       (string? (:q-w kmap))
                       (string? (:k-w kmap))
                       (string? (:v-w kmap))
                       (string? (:o-w kmap))
                       (string? (:post-attn-ln-w kmap))
                       (string? (:gate-w kmap))
                       (string? (:up-w kmap))
                       (string? (:down-w kmap))))))
