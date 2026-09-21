(ns einsum.models.gemma3-test
  "Unit and generative tests for declarative Tensor Logic Gemma 3 model architecture."
  (:require [einsum.core :as xla]
            [einsum.logic.ast :as ast]
            [einsum.logic.expand :as expand]
            [einsum.logic.lower :as lower]
            [einsum.models.gemma3 :as gemma3]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gemma3-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 17)
                 seq-len (gen/elements [4 8 16])]
                (let [config (assoc (gemma3/gemma3-config) :head-dim 256)
                      layer-ast (gemma3/gemma3-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma3-model-ast-schema 10
  (prop/for-all [num-layers (gen/choose 1 4)
                 seq-len (gen/elements [4 8])]
                (let [config {:vocab-size 1000
                              :hidden-dim 640
                              :intermediate-dim 2048
                              :num-layers num-layers
                              :num-heads 4
                              :num-kv-heads 1
                              :head-dim 256
                              :max-seq-len seq-len}
                      model-ast (gemma3/gemma3-model-ast config)
                      expanded (expand/expand-ast {} model-ast)]
                  (and (vector? model-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma3-single-layer-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 640
        intermediate-dim 2048
        head-dim 256
        q-dim 1024
        kv-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :num-layers num-layers
                :num-heads 4
                :num-kv-heads 1
                :head-dim head-dim
                :max-seq-len max-seq-len}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :bf16]]
                [:final_norm_w [:tensor [hidden-dim] :bf16]]
                [:input_ln_w_0 [:tensor [hidden-dim] :bf16]]
                [:q_w_0 [:tensor [q-dim hidden-dim] :bf16]]
                [:k_w_0 [:tensor [kv-dim hidden-dim] :bf16]]
                [:v_w_0 [:tensor [kv-dim hidden-dim] :bf16]]
                [:o_w_0 [:tensor [hidden-dim q-dim] :bf16]]
                [:q_norm_w_0 [:tensor [head-dim] :bf16]]
                [:k_norm_w_0 [:tensor [head-dim] :bf16]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :bf16]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :bf16]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :bf16]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :bf16]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :bf16]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :bf16]]]
        ast (gemma3/gemma3-model-ast config)
        graph (lower/ast->graph "gemma3_single_layer" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gemma3-full-model-lowering
  (let [num-layers 18
        max-seq-len 4
        vocab-size 256
        hidden-dim 640
        intermediate-dim 2048
        head-dim 256
        q-dim 1024
        kv-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :num-layers num-layers
                :num-heads 4
                :num-kv-heads 1
                :head-dim head-dim
                :max-seq-len max-seq-len}
        invars (vec (concat
                     [[:x [:tensor [1 max-seq-len] :i32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] :bf16]]
                      [:final_norm_w [:tensor [hidden-dim] :bf16]]]
                     (mapcat (fn [i]
                               [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] :bf16]]
                                [(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] :bf16]]
                                [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] :bf16]]
                                [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] :bf16]]
                                [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] :bf16]]
                                [(keyword (str "q_norm_w_" i)) [:tensor [head-dim] :bf16]]
                                [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] :bf16]]
                                [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] :bf16]]
                                [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] :bf16]]
                                [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] :bf16]]
                                [(keyword (str "gate_w_" i)) [:tensor [intermediate-dim hidden-dim] :bf16]]
                                [(keyword (str "up_w_" i)) [:tensor [intermediate-dim hidden-dim] :bf16]]
                                [(keyword (str "down_w_" i)) [:tensor [hidden-dim intermediate-dim] :bf16]]])
                             (range num-layers))))
        ast (gemma3/gemma3-model-ast config)
        graph (lower/ast->graph "gemma3_full_model" invars ast #{:logits})]
    (is (shlo/validate-graph graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (xla/get-context)
          compiled (xla/compile-graph ctx graph)]
      (is (some? compiled)))))
