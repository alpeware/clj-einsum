(ns models.gpt2-test
  "Unit and compilation parity tests for declarative Tensor Logic GPT-2 model."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [models.gpt2 :as gpt2]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-gpt2-single-layer-lowering
  (let [num-layers 1
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos_ids [:tensor [1 max-seq-len] :i32]]
                      [:ln_f_g [:tensor [768] :f32]]
                      [:ln_f_b [:tensor [768] :f32]]
                      [:wte [:tensor [50257 768] :f32]]
                      [:wpe [:tensor [1024 768] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                             (range num-layers)))
        ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "gpt2_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gpt2-12-layer-lowering
  (let [num-layers 12
        max-seq-len 16
        invars (into [[:x [:tensor [1 max-seq-len] :i32]]
                      [:pos_ids [:tensor [1 max-seq-len] :i32]]
                      [:ln_f_g [:tensor [768] :f32]]
                      [:ln_f_b [:tensor [768] :f32]]
                      [:wte [:tensor [50257 768] :f32]]
                      [:wpe [:tensor [1024 768] :f32]]]
                     (mapcat (fn [i]
                               [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                             (range num-layers)))
        ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
        graph (lower/ast->graph "gpt2_12_layers" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-gpt2-config-invariants 20
  (prop/for-all [n-embd (gen/elements [768 1024 1280 1600])]
                (let [cfg (gpt2/gpt2-config {:n-embd n-embd})]
                  (and (= (:n-embd cfg) n-embd)
                       (pos? (:vocab-size cfg))
                       (pos? (:n-layer cfg))
                       (pos? (:n-head cfg))))))

(defspec prop-gpt2-weight-key-map 20
  (prop/for-all [layer-idx (gen/choose 0 35)]
                (let [kmap (gpt2/weight-key-map layer-idx)]
                  (and (string? (:ln1-g kmap))
                       (string? (:ln1-b kmap))
                       (string? (:c-attn-w kmap))
                       (string? (:c-attn-b kmap))
                       (string? (:c-proj-w kmap))
                       (string? (:c-proj-b kmap))
                       (string? (:ln2-g kmap))
                       (string? (:ln2-b kmap))
                       (string? (:mlp-fc-w kmap))
                       (string? (:mlp-fc-b kmap))
                       (string? (:mlp-proj-w kmap))
                       (string? (:mlp-proj-b kmap))))))
