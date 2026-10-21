(ns einsum.models.gemma-test
  "Unit and generative tests for declarative Tensor Logic Gemma 4 model architecture."
  (:require [einsum.core :as ein]
            [einsum.logic.ast :as ast]
            [einsum.logic.expand :as expand]
            [einsum.models.gemma :as gemma]
            [einsum.logic.memory.contrastive :as contrastive]
            [einsum.uop.lower :as uop-lower]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-gemma4-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 seq-len (gen/elements [8 16 32])]
                (let [config (gemma/gemma4-config :e2b)
                      layer-ast (gemma/gemma4-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-int8-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 seq-len (gen/elements [8 16 32])]
                (let [config (assoc (gemma/gemma4-config :e2b) :is-int8 true)
                      layer-ast (gemma/gemma4-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma2-alias-resolver
  (testing "gemma2-alias-resolver correctly resolves AST keywords to safetensors paths"
    (let [resolver (gemma/gemma2-alias-resolver "model.")]
      (is (= "model.embed_tokens.weight" (resolver :embed_tokens)))
      (is (= "model.norm.weight" (resolver :final_norm_w)))
      (is (= "model.layers.0.input_layernorm.weight" (resolver :input_ln_w_0)))
      (is (= "model.layers.0.self_attn.q_proj.weight" (resolver :q_w_0)))
      (is (= "model.layers.0.mlp.down_proj.weight" (resolver :down_w_0)))
      (is (= "model.layers.17.mlp.up_proj.weight" (resolver :up_w_17))))))

(defspec prop-gemma2-alias-resolver-matches-weight-key-map 35
  (prop/for-all [layer-idx (gen/choose 0 17)]
                (let [resolver (gemma/gemma2-alias-resolver "model.layers.")
                      kmap (gemma/weight-key-map layer-idx)]
                  (and (= (resolver (keyword (str "input_ln_w_" layer-idx))) (:input-ln-w kmap))
                       (= (resolver (keyword (str "q_w_" layer-idx))) (:q-w kmap))
                       (= (resolver (keyword (str "k_w_" layer-idx))) (:k-w kmap))
                       (= (resolver (keyword (str "v_w_" layer-idx))) (:v-w kmap))
                       (= (resolver (keyword (str "o_w_" layer-idx))) (:o-w kmap))
                       (= (resolver (keyword (str "post_attn_ln_w_" layer-idx))) (:post-attn-ln-w kmap))
                       (= (resolver (keyword (str "pre_mlp_ln_w_" layer-idx))) (:pre-mlp-ln-w kmap))
                       (= (resolver (keyword (str "post_mlp_ln_w_" layer-idx))) (:post-mlp-ln-w kmap))
                       (= (resolver (keyword (str "gate_w_" layer-idx))) (:gate-w kmap))
                       (= (resolver (keyword (str "up_w_" layer-idx))) (:up-w kmap))
                       (= (resolver (keyword (str "down_w_" layer-idx))) (:down-w kmap))))))

(deftest test-gemma4-alias-resolver
  (testing "gemma4-alias-resolver correctly resolves AST keywords to safetensors paths"
    (let [resolver (gemma/gemma4-alias-resolver "model.language_model.")]
      (is (= "model.language_model.embed_tokens.weight" (resolver :embed_tokens)))
      (is (= "model.language_model.norm.weight" (resolver :final_norm_w)))
      (is (= "model.language_model.layers.0.input_layernorm.weight" (resolver :input_ln_w_0)))
      (is (= "model.language_model.layers.0.self_attn.q_proj.weight" (resolver :q_w_0)))
      (is (= "model.language_model.layers.0.mlp.down_proj.weight" (resolver :down_w_0)))
      (is (= "model.language_model.layers.34.mlp.up_proj.weight" (resolver :up_w_34))))))

(defspec prop-gemma4-alias-resolver-matches-weight-key-map 35
  (prop/for-all [layer-idx (gen/choose 0 34)]
                (let [resolver (gemma/gemma4-alias-resolver "model.layers.")
                      kmap (gemma/gemma4-weight-key-map layer-idx "model.layers.")]
                  (and (= (resolver (keyword (str "input_ln_w_" layer-idx))) (:input-ln-w kmap))
                       (= (resolver (keyword (str "layer_scalar_" layer-idx))) (:layer-scalar-w kmap))
                       (= (resolver (keyword (str "q_w_" layer-idx))) (:q-w kmap))
                       (= (resolver (keyword (str "k_w_" layer-idx))) (:k-w kmap))
                       (= (resolver (keyword (str "v_w_" layer-idx))) (:v-w kmap))
                       (= (resolver (keyword (str "o_w_" layer-idx))) (:o-w kmap))
                       (= (resolver (keyword (str "q_norm_w_" layer-idx))) (:q-norm-w kmap))
                       (= (resolver (keyword (str "k_norm_w_" layer-idx))) (:k-norm-w kmap))
                       (= (resolver (keyword (str "post_attn_ln_w_" layer-idx))) (:post-attn-ln-w kmap))
                       (= (resolver (keyword (str "pre_mlp_ln_w_" layer-idx))) (:pre-mlp-ln-w kmap))
                       (= (resolver (keyword (str "post_mlp_ln_w_" layer-idx))) (:post-mlp-ln-w kmap))
                       (= (resolver (keyword (str "gate_w_" layer-idx))) (:gate-w kmap))
                       (= (resolver (keyword (str "up_w_" layer-idx))) (:up-w kmap))
                       (= (resolver (keyword (str "down_w_" layer-idx))) (:down-w kmap))
                       (= (resolver (keyword (str "per_layer_gate_w_" layer-idx))) (:per-layer-gate-w kmap))
                       (= (resolver (keyword (str "per_layer_proj_w_" layer-idx))) (:per-layer-proj-w kmap))
                       (= (resolver (keyword (str "post_per_layer_norm_w_" layer-idx))) (:post-per-layer-norm-w kmap))))))

(deftest test-gemma4-single-layer-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_single_layer" invars ast #{:logits})]
    (is (some? graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gemma4-single-layer-int8-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :is-int8 true
                :norm-dtype :f32}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :i8]]
                [:q_scale_0 [:tensor [2048] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :i8]]
                [:k_scale_0 [:tensor [256] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :i8]]
                [:v_scale_0 [:tensor [256] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :i8]]
                [:o_scale_0 [:tensor [hidden-dim] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:gate_scale_0 [:tensor [intermediate-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:up_scale_0 [:tensor [intermediate-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :i8]]
                [:down_scale_0 [:tensor [hidden-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_single_layer_int8" invars ast #{:logits})]
    (is (some? graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gemma4-last-token-only-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :last-token-only? true}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:pos [:tensor [1] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_last_token" invars ast #{:logits})]
    (is (some? graph))
    (is (= [:logits] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-gemma4-config-invariants 20
  (prop/for-all [model-kw (gen/elements [:e2b :e4b :12b :26b :31b])]
                (let [cfg (gemma/gemma4-config model-kw)]
                  (and (pos? (:hidden-dim cfg))
                       (pos? (:num-layers cfg))
                       (pos? (:num-heads cfg))
                       (pos? (:head-dim cfg))
                       (pos? (:vocab-size cfg))))))

(defspec prop-gemma4-weight-key-map 20
  (prop/for-all [layer-idx (gen/choose 0 34)]
                (let [kmap (gemma/gemma4-weight-key-map layer-idx "layers.")]
                  (and (string? (:q-w kmap))
                       (string? (:k-w kmap))
                       (string? (:v-w kmap))
                       (string? (:o-w kmap))
                       (string? (:gate-w kmap))
                       (string? (:up-w kmap))
                       (string? (:down-w kmap))))))

(defspec prop-gemma4-kv-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 max-seq-len (gen/elements [16 32 64])]
                (let [config (assoc (gemma/gemma4-config :e2b) :max-seq-len max-seq-len)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx max-seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-kv-int8-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 max-seq-len (gen/elements [16 32 64])]
                (let [config (assoc (gemma/gemma4-config :e2b) :max-seq-len max-seq-len :is-int8 true)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx max-seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-kv-int4-unpack-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 34)
                 max-seq-len (gen/elements [16 32 64])]
                (let [config (assoc (gemma/gemma4-config :e2b) :max-seq-len max-seq-len :is-int4 true)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx max-seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-kv-cache-step-int4-unpack-ast
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 256
        intermediate-dim 512
        pl-dim 64
        total-pl-dim 64
        group-size 128
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 4
                :num-kv-heads 1
                :head-dim 64
                :max-seq-len max-seq-len
                :is-int4 true
                :group-size group-size}
        ast (gemma/gemma4-kv-model-ast config)
        ast-ops (map first (filter #(and (vector? %) (keyword? (first %))) (tree-seq coll? seq ast)))
        unpack-count (count (filter #(= % :int4-unpack) ast-ops))
        w4a16-count (count (filter #(= % :w4a16-gemv) ast-ops))]
    (is (> unpack-count 0))
    (is (zero? w4a16-count))))

(deftest test-gemma4-kv-cache-step-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len}
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]
                [:k_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
                [:v_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-kv-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_kv_step" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (some? graph))
    (is (= [:logits :k_cache_out_0 :v_cache_out_0] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)
          _ (is (some? compiled))
          x-data (int-array [42])
          pos-data (int-array [3])
          k-cache (float-array (repeat (* max-seq-len 256) 0.0))
          v-cache (float-array (repeat (* max-seq-len 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape _]]]
                               (let [size (reduce * shape)]
                                 (float-array (repeat size 0.01))))
                             (subvec invars 4))
          all-inputs (into [x-data pos-data k-cache v-cache] weights-data)
          outs (apply ein/execute compiled all-inputs)
          [logits-buf k-out-buf v-out-buf] (if (vector? outs) outs [outs])
          logits (ein/to-host-slice logits-buf 0 vocab-size vocab-size)
          k-out (ein/to-host-slice k-out-buf 0 (* max-seq-len 256) (* max-seq-len 256))]
      (is (= vocab-size (count logits)))
      (is (some #(not= 0.0 %) (subvec (vec k-out) (* 3 256) (* 4 256))))
      (is (every? #(= 0.0 %) (subvec (vec k-out) 0 256)))
      (ein/destroy-buffer! logits-buf)
      (ein/destroy-buffer! k-out-buf)
      (ein/destroy-buffer! v-out-buf))))

(deftest test-gemma4-kv-cache-step-ring-buffer-lowering
  (let [num-layers 1
        max-seq-len 16
        window 4
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :sliding-window window}
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]]
        invars (into invars
                     [[:k_cache_in_0 [:tensor [1 window 1 256] :f32]]
                      [:v_cache_in_0 [:tensor [1 window 1 256] :f32]]
                      [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                      [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                      [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                      [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                      [:final_norm_w [:tensor [hidden-dim] :f32]]
                      [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                      [:layer_scalar_0 [:tensor [1] :f32]]
                      [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                      [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                      [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                      [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                      [:q_norm_w_0 [:tensor [256] :f32]]
                      [:k_norm_w_0 [:tensor [256] :f32]]
                      [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                      [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                      [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                      [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                      [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                      [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                      [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                      [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                      [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]])
        ast (gemma/gemma4-kv-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_kv_ring_step" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (some? graph))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)
          x-data (int-array [42])
          pos-data (int-array [6]) ;; 6 mod 4 = 2
          k-cache (float-array (repeat (* window 256) 0.0))
          v-cache (float-array (repeat (* window 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape _]]]
                               (let [size (reduce * shape)]
                                 (float-array (repeat size 0.01))))
                             (subvec invars 4))
          all-inputs (into [x-data pos-data k-cache v-cache] weights-data)
          outs (apply ein/execute compiled all-inputs)
          [logits-buf k-out-buf v-out-buf] (if (vector? outs) outs [outs])
          logits (ein/to-host-slice logits-buf 0 vocab-size vocab-size)
          k-out (ein/to-host-slice k-out-buf 0 (* window 256) (* window 256))]
      (is (= vocab-size (count logits)))
      ;; Slot 2 (indices 512 to 767) should be populated because 6 mod 4 = 2!
      (is (some #(not= 0.0 %) (subvec (vec k-out) (* 2 256) (* 3 256))))
      ;; Slot 0 (indices 0 to 255) and slot 1 (indices 256 to 511) should remain 0.0
      (is (every? #(= 0.0 %) (subvec (vec k-out) 0 256)))
      (is (every? #(= 0.0 %) (subvec (vec k-out) 256 512)))
      (ein/destroy-buffer! logits-buf)
      (ein/destroy-buffer! k-out-buf)
      (ein/destroy-buffer! v-out-buf))))

(deftest test-gemma4-kv-cache-step-int8-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :is-int8 true
                :norm-dtype :f32}
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]
                [:k_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
                [:v_cache_in_0 [:tensor [1 max-seq-len 1 256] :f32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :i8]]
                [:q_scale_0 [:tensor [2048] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :i8]]
                [:k_scale_0 [:tensor [256] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :i8]]
                [:v_scale_0 [:tensor [256] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :i8]]
                [:o_scale_0 [:tensor [hidden-dim] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:gate_scale_0 [:tensor [intermediate-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :i8]]
                [:up_scale_0 [:tensor [intermediate-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :i8]]
                [:down_scale_0 [:tensor [hidden-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-kv-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_kv_step_int8" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (some? graph))
    (is (= [:logits :k_cache_out_0 :v_cache_out_0] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)
          _ (is (some? compiled))
          x-data (int-array [42])
          pos-data (int-array [3])
          k-cache (float-array (repeat (* max-seq-len 256) 0.0))
          v-cache (float-array (repeat (* max-seq-len 256) 0.0))
          weights-data (mapv (fn [[_ [_ shape dtype]]]
                               (let [size (reduce * shape)]
                                 (if (= dtype :i8)
                                   (byte-array (repeat size (byte 1)))
                                   (float-array (repeat size 0.01)))))
                             (subvec invars 4))
          all-inputs (into [x-data pos-data k-cache v-cache] weights-data)
          outs (apply ein/execute compiled all-inputs)
          [logits-buf k-out-buf v-out-buf] (if (vector? outs) outs [outs])
          logits (ein/to-host-slice logits-buf 0 vocab-size vocab-size)
          k-out (ein/to-host-slice k-out-buf 0 (* max-seq-len 256) (* max-seq-len 256))]
      (is (= vocab-size (count logits)))
      (is (some #(not= 0.0 %) (subvec (vec k-out) (* 3 256) (* 4 256))))
      (is (every? #(= 0.0 %) (subvec (vec k-out) 0 256)))
      (ein/destroy-buffer! logits-buf)
      (ein/destroy-buffer! k-out-buf)
      (ein/destroy-buffer! v-out-buf))))

(defspec prop-layer-is-global-schedules 50
  (prop/for-all [layer-idx (gen/choose 0 41)]
                (let [e2b-types nil
                      e4b-types (vec (mapcat (fn [_] (concat (repeat 5 "sliding_attention") ["full_attention"])) (range 7)))
                      e2b-res (gemma/layer-is-global? e2b-types layer-idx)
                      e4b-res (gemma/layer-is-global? e4b-types layer-idx)]
                  (and (= e2b-res (zero? (mod (inc layer-idx) 5)))
                       (= e4b-res (= 5 (mod layer-idx 6)))))))

(deftest test-gemma4-prefill-lowering
  (let [num-layers 1
        max-seq-len 8
        vocab-size 1000
        hidden-dim 1536
        intermediate-dim 6144
        pl-dim 256
        total-pl-dim 256
        config {:vocab-size vocab-size
                :hidden-dim hidden-dim
                :intermediate-dim intermediate-dim
                :pl-dim pl-dim
                :total-pl-dim total-pl-dim
                :num-layers num-layers
                :num-heads 8
                :num-kv-heads 1
                :head-dim 256
                :max-seq-len max-seq-len
                :last-token-only? true}
        invars [[:x [:tensor [1 max-seq-len] :i32]]
                [:pos [:tensor [1] :i32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] :f32]]
                [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] :f32]]
                [:per_layer_projection_norm [:tensor [pl-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [2048 hidden-dim] :f32]]
                [:k_w_0 [:tensor [256 hidden-dim] :f32]]
                [:v_w_0 [:tensor [256 hidden-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim 2048] :f32]]
                [:q_norm_w_0 [:tensor [256] :f32]]
                [:k_norm_w_0 [:tensor [256] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                [:per_layer_gate_w_0 [:tensor [pl-dim hidden-dim] :f32]]
                [:per_layer_proj_w_0 [:tensor [hidden-dim pl-dim] :f32]]
                [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_prefill" invars ast [:logits :k_ro_0 :v_heads_0])]
    (is (some? graph))
    (is (= [:logits :k_ro_0 :v_heads_0] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-chunked-attention-ast-validity 20
  (prop/for-all [max-seq-len (gen/elements [64 128 256])
                 chunk-size (gen/elements [32 64])
                 num-heads (gen/elements [4 8 16])
                 num-kv-heads (gen/elements [1 2 4])
                 head-dim (gen/elements [64 128 256])]
                (let [group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      node [:chunked-attention [:ctx :b :p :h :dh]
                            [:q_ro :b :p :h :dh]
                            [:k_cache :b :kv-s :kvh :dh]
                            [:v_cache :b :kv-s :kvh :dh]
                            {:pos :pos
                             :chunk-size chunk-size
                             :head-dim head-dim
                             :num-heads num-heads
                             :num-kv-heads num-kv-heads
                             :max-seq-len max-seq-len}]]
                  (ast/valid-node? node))))

(defspec prop-chunked-attention-parity 10
  (prop/for-all [max-seq-len (gen/elements [64 128])
                 chunk-size (gen/elements [32 64])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [32 64])
                 pos-idx (gen/choose 0 63)]
                (let [_pos (min pos-idx (dec max-seq-len))
                      group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      invars [[:pos [:tensor [1] :i32]]
                              [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                              [:k_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]
                              [:v_cache [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]]
                      ast [:block {:name :chunked_block}
                           [:chunked-attention [:ctx :b :p :h :dh]
                            [:q_ro :b :p :h :dh]
                            [:k_cache :b :kv-s :kvh :dh]
                            [:v_cache :b :kv-s :kvh :dh]
                            {:pos :pos
                             :chunk-size chunk-size
                             :head-dim head-dim
                             :num-heads num-heads
                             :num-kv-heads num-kv-heads
                             :max-seq-len max-seq-len
                             :shape [1 1 num-heads head-dim]}]]
                      g-chunk (uop-lower/ast->uop-graph "test_chunk_parity" invars ast [:ctx])]
                  (and (some? g-chunk)
                       (= [:ctx] (:outvars g-chunk))
                       (contains? (:outputs g-chunk) :ctx)))))

(defspec prop-fused-turboquant-attention-parity 10
  (prop/for-all [max-seq-len (gen/elements [64 128])
                 chunk-size (gen/elements [32 64])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [32 64])
                 k-scale (gen/elements [0.0625 0.125 1.0])
                 pos-idx (gen/choose 0 63)]
                (let [_pos (min pos-idx (dec max-seq-len))
                      group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      packed-dim (quot head-dim 4)
                      invars [[:pos [:tensor [1] :i32]]
                              [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                              [:k_cache [:tensor [1 max-seq-len num-kv-heads packed-dim] :i8]]
                              [:v_cache [:tensor [1 max-seq-len num-kv-heads packed-dim] :i8]]]
                      ast-fused [:block {:name :fused_tq_block}
                                 [:chunked-attention [:ctx :b :p :h :dh]
                                  [:q_ro :b :p :h :dh]
                                  [:k_cache :b :kvs :kvh :pdh]
                                  [:v_cache :b :kvs :kvh :pdh]
                                  {:pos :pos
                                   :chunk-size chunk-size
                                   :head-dim head-dim
                                   :num-heads num-heads
                                   :num-kv-heads num-kv-heads
                                   :max-seq-len max-seq-len
                                   :shape [1 1 num-heads head-dim]
                                   :turboquant? true
                                   :k-scale k-scale
                                   :v-scale 1.0}]]
                      g-fused (uop-lower/ast->uop-graph "test_tq_fused" invars ast-fused [:ctx])]
                  (and (some? g-fused)
                       (= [:ctx] (:outvars g-fused))
                       (contains? (:outputs g-fused) :ctx)))))

(defspec prop-turboquant-pack-unpack-cosine-similarity 10
  (prop/for-all [d (gen/elements [64 128 256])]
                (let [invars [[:x [:tensor [1 d] :f32]]]
                      ast [:block {:name :tq_pack_unpack_block}
                           [:turboquant-pack [:packed :b :pd] [:x :b :d] {:scale 1.0}]
                           [:turboquant-unpack [:recon :b :d] [:packed :b :pd] {:scale 1.0}]]
                      g (uop-lower/ast->uop-graph "test_tq_pack_unpack" invars ast [:recon])]
                  (and (some? g)
                       (= [:recon] (:outvars g))
                       (contains? (:outputs g) :recon)))))

(defspec prop-sliding-window-ring-buffer-parity 10
  (prop/for-all [window (gen/elements [16 32])
                 mult (gen/elements [2 3])
                 chunk-size (gen/elements [8 16])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [16 32])
                 pos-idx (gen/choose 0 95)]
                (let [max-seq-len (* window mult)
                      _pos (min pos-idx (dec max-seq-len))
                      group-size (quot num-heads num-kv-heads)
                      num-heads (* num-kv-heads group-size)
                      invars-ring [[:pos [:tensor [1] :i32]]
                                   [:q_ro [:tensor [1 1 num-heads head-dim] :f32]]
                                   [:k_cache [:tensor [1 window num-kv-heads head-dim] :f32]]
                                   [:v_cache [:tensor [1 window num-kv-heads head-dim] :f32]]]
                      ast-ring [:block {:name :ring_block}
                                [:chunked-attention [:ctx :b :p :h :dh]
                                 [:q_ro :b :p :h :dh]
                                 [:k_cache :b :kv-s :kvh :dh]
                                 [:v_cache :b :kv-s :kvh :dh]
                                 {:pos :pos
                                  :chunk-size chunk-size
                                  :head-dim head-dim
                                  :num-heads num-heads
                                  :num-kv-heads num-kv-heads
                                  :max-seq-len window
                                  :sliding-window window
                                  :ring-buffer true
                                  :shape [1 1 num-heads head-dim]}]]
                      g-ring (uop-lower/ast->uop-graph "test_ring_parity" invars-ring ast-ring [:ctx])]
                  (and (some? g-ring)
                       (= [:ctx] (:outvars g-ring))
                       (contains? (:outputs g-ring) :ctx)))))

(defspec prop-dynamic-update-slice-sink-pinning 10
  (prop/for-all [window (gen/elements [16 32])
                 k-sink (gen/elements [2 4])
                 _pos (gen/choose 0 95)]
                (let [invars [[:cache [:tensor [1 window] :i32]]
                              [:up [:tensor [1 1] :i32]]
                              [:pos [:tensor [1] :i32]]]
                      ast [:block {:name :update_block}
                           [:dynamic-update-slice [:out :b :s] [:cache :b :s] [:up :b :one]
                            {:start-indices [0 :pos] :window window :k-sink k-sink}]]
                      g (uop-lower/ast->uop-graph "test_sink_update" invars ast [:out])]
                  (and (some? g)
                       (= [:out] (:outvars g))
                       (contains? (:outputs g) :out)))))

(defspec prop-gemma4-prefill-ring-buffer-shapes 10
  (prop/for-all [window (gen/elements [8 16])
                 mult (gen/elements [2 3])
                 num-heads (gen/elements [4 8])
                 num-kv-heads (gen/elements [1 2])
                 head-dim (gen/elements [16 32])]
                (let [max-seq-len (* window mult)
                      hidden-dim (* num-heads head-dim)
                      intermediate-dim (* 2 hidden-dim)
                      config {:vocab-size 100
                              :hidden-dim hidden-dim
                              :intermediate-dim intermediate-dim
                              :total-pl-dim 0
                              :num-layers 2
                              :num-heads num-heads
                              :num-kv-heads num-kv-heads
                              :head-dim head-dim
                              :sliding-window window
                              :layer-types [:sliding :sliding]
                              :max-seq-len max-seq-len
                              :last-token-only? true}
                      targets (gemma/gemma4-prefill-outvars config max-seq-len)
                      ast (gemma/gemma4-model-ast config)]
                  (and (= targets [:logits :k_ro_sl_0 :v_heads_sl_0 :k_ro_sl_1 :v_heads_sl_1])
                       (let [invars [[:x [:tensor [1 max-seq-len] :i32]]
                                     [:pos [:tensor [1] :i32]]
                                     [:embed_tokens [:tensor [100 hidden-dim] :f32]]
                                     [:final_norm_w [:tensor [hidden-dim] :f32]]
                                     [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                                     [:layer_scalar_0 [:tensor [1] :f32]]
                                     [:q_w_0 [:tensor [(* num-heads head-dim) hidden-dim] :f32]]
                                     [:k_w_0 [:tensor [(* num-kv-heads head-dim) hidden-dim] :f32]]
                                     [:v_w_0 [:tensor [(* num-kv-heads head-dim) hidden-dim] :f32]]
                                     [:o_w_0 [:tensor [hidden-dim (* num-heads head-dim)] :f32]]
                                     [:q_norm_w_0 [:tensor [head-dim] :f32]]
                                     [:k_norm_w_0 [:tensor [head-dim] :f32]]
                                     [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                                     [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                                     [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                                     [:gate_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                                     [:up_w_0 [:tensor [intermediate-dim hidden-dim] :f32]]
                                     [:down_w_0 [:tensor [hidden-dim intermediate-dim] :f32]]
                                     [:input_ln_w_1 [:tensor [hidden-dim] :f32]]
                                     [:layer_scalar_1 [:tensor [1] :f32]]
                                     [:q_w_1 [:tensor [(* num-heads head-dim) hidden-dim] :f32]]
                                     [:k_w_1 [:tensor [(* num-kv-heads head-dim) hidden-dim] :f32]]
                                     [:v_w_1 [:tensor [(* num-kv-heads head-dim) hidden-dim] :f32]]
                                     [:o_w_1 [:tensor [hidden-dim (* num-heads head-dim)] :f32]]
                                     [:q_norm_w_1 [:tensor [head-dim] :f32]]
                                     [:k_norm_w_1 [:tensor [head-dim] :f32]]
                                     [:post_attn_ln_w_1 [:tensor [hidden-dim] :f32]]
                                     [:pre_mlp_ln_w_1 [:tensor [hidden-dim] :f32]]
                                     [:post_mlp_ln_w_1 [:tensor [hidden-dim] :f32]]
                                     [:gate_w_1 [:tensor [intermediate-dim hidden-dim] :f32]]
                                     [:up_w_1 [:tensor [intermediate-dim hidden-dim] :f32]]
                                     [:down_w_1 [:tensor [hidden-dim intermediate-dim] :f32]]]
                             graph (uop-lower/ast->uop-graph "test_prefill_ring" invars ast targets)
                             expected-shape [1 window num-kv-heads head-dim]]
                         (and (some? graph)
                              (= (get-in graph [:outputs :k_ro_sl_0 :shape]) expected-shape)
                              (= (get-in graph [:outputs :v_heads_sl_0 :shape]) expected-shape)
                              (let [ctx (ein/init-cpu!)
                                    exec (ein/compile-graph ctx graph)]
                                (some? exec))))))))

(defspec prop-gemma4-31b-config-invariants 10
  (prop/for-all [layer-idx (gen/choose 0 59)]
                (let [cfg (gemma/gemma4-config :31b)]
                  (and (= (:num-layers cfg) 60)
                       (= (:hidden-dim cfg) 5376)
                       (= (:intermediate-dim cfg) 21504)
                       (= (:num-heads cfg) 32)
                       (= (:num-kv-heads cfg) 16)
                       (= (:num-global-kv-heads cfg) 4)
                       (= (:head-dim cfg) 256)
                       (= (:global-head-dim cfg) 512)
                       (= (:sliding-window cfg) 1024)
                       (= (:pl-dim cfg) 0)
                       (= (:total-pl-dim cfg) 0)
                       (= (gemma/layer-is-global? (:layer-types cfg) layer-idx)
                          (= (mod layer-idx 6) 5))))))

(defspec prop-gemma4-31b-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 59)
                 seq-len (gen/elements [8 16 32])]
                (let [config (gemma/gemma4-config :31b)
                      layer-ast (gemma/gemma4-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-31b-kv-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 59)
                 seq-len (gen/elements [1024 2048])]
                (let [config (gemma/gemma4-config :31b)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-31b-int4-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 59)
                 seq-len (gen/elements [8 16 32])]
                (let [config (assoc (gemma/gemma4-config :31b) :is-int4 true)
                      layer-ast (gemma/gemma4-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-31b-int4-kv-layer-ast-validity 20
  (prop/for-all [layer-idx (gen/choose 0 59)
                 seq-len (gen/elements [1024 2048])]
                (let [config (assoc (gemma/gemma4-config :31b) :is-int4 true)
                      layer-ast (gemma/gemma4-kv-layer-ast layer-idx seq-len config)
                      expanded (expand/expand-ast {} layer-ast)]
                  (and (vector? layer-ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-31b-int4-step-lowering-and-compilation
  (let [num-layers 1
        max-seq-len 8
        vocab-size 256
        hidden-dim 5376
        intermediate-dim 21504
        head-dim 256
        num-heads 32
        num-kv-heads 16
        q-dim (* num-heads head-dim)
        kv-dim (* num-kv-heads head-dim)
        config (assoc (gemma/gemma4-config :31b)
                      :num-layers num-layers
                      :vocab-size vocab-size
                      :max-seq-len max-seq-len
                      :is-int4 true)
        invars [[:x [:tensor [1 1] :i32]]
                [:pos [:tensor [1] :i32]]
                [:k_cache_in_0 [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]
                [:v_cache_in_0 [:tensor [1 max-seq-len num-kv-heads head-dim] :f32]]
                [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                [:final_norm_w [:tensor [hidden-dim] :f32]]
                [:input_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:layer_scalar_0 [:tensor [1] :f32]]
                [:q_w_0 [:tensor [q-dim (quot hidden-dim 2)] :i8]]
                [:q_scale_0 [:tensor [q-dim] :f32]]
                [:k_w_0 [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                [:k_scale_0 [:tensor [kv-dim] :f32]]
                [:v_w_0 [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                [:v_scale_0 [:tensor [kv-dim] :f32]]
                [:o_w_0 [:tensor [hidden-dim (quot q-dim 2)] :i8]]
                [:o_scale_0 [:tensor [hidden-dim] :f32]]
                [:q_norm_w_0 [:tensor [head-dim] :f32]]
                [:k_norm_w_0 [:tensor [head-dim] :f32]]
                [:post_attn_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:pre_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:post_mlp_ln_w_0 [:tensor [hidden-dim] :f32]]
                [:gate_w_0 [:tensor [intermediate-dim (quot hidden-dim 2)] :i8]]
                [:gate_scale_0 [:tensor [intermediate-dim] :f32]]
                [:up_w_0 [:tensor [intermediate-dim (quot hidden-dim 2)] :i8]]
                [:up_scale_0 [:tensor [intermediate-dim] :f32]]
                [:down_w_0 [:tensor [hidden-dim (quot intermediate-dim 2)] :i8]]
                [:down_scale_0 [:tensor [hidden-dim] :f32]]]
        ast (gemma/gemma4-kv-model-ast config)
        graph (uop-lower/ast->uop-graph "gemma4_31b_int4_kv_step" invars ast [:logits :k_cache_out_0 :v_cache_out_0])]
    (is (some? graph))
    (is (= [:logits :k_cache_out_0 :v_cache_out_0] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-gemma4-unbinding-ast-validity 20
  (prop/for-all [dim (gen/elements [256 512 1536])
                 dm (gen/elements [64 128 256])
                 vocab-size (gen/elements [1000 5000 10000])]
                (let [ast (gemma/gemma4-unbinding-ast dim dm vocab-size 1.0)
                      expanded (expand/expand-ast {} ast)]
                  (and (vector? ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-entity-unbinding-ast-validity 20
  (prop/for-all [dim (gen/elements [256 512 1536])
                 dm (gen/elements [64 128 256])
                 vocab-size (gen/elements [1000 5000 10000])]
                (let [ast (gemma/gemma4-entity-unbinding-ast dim dm vocab-size 1.0)
                      expanded (expand/expand-ast {} ast)]
                  (and (vector? ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-unbinding-ast-compilation
  (let [dim 1536
        dm 128
        vocab-size 2000
        invars [[:h [:tensor [1 1 dim] :f32]]
                [:logits_base [:tensor [1 1 vocab-size] :f32]]
                [:W_mem [:tensor [dim dm] :f32]]
                [:R [:tensor [dm dm] :f32]]
                [:W_embed [:tensor [vocab-size dim] :f32]]]
        ast (gemma/gemma4-unbinding-ast dim dm vocab-size 1.0)
        graph (uop-lower/ast->uop-graph "gemma_unbinding_test" invars ast [:logits_grounded :delta_logits])]
    (is (some? graph))
    (is (= [:logits_grounded :delta_logits] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(defspec prop-gemma4-two-stage-resolver-ast-validity 20
  (prop/for-all [dim (gen/elements [256 512 1536])
                 dm (gen/elements [64 128 256])
                 vocab-size (gen/elements [1000 5000 10000])
                 k (gen/elements [8 16 32])]
                (let [ast (gemma/gemma4-two-stage-resolver-ast dim dm vocab-size k)
                      expanded (expand/expand-ast {} ast)]
                  (and (vector? ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(defspec prop-gemma4-two-stage-entity-resolver-ast-validity 20
  (prop/for-all [dim (gen/elements [256 512 1536])
                 dm (gen/elements [64 128 256])
                 vocab-size (gen/elements [1000 5000 10000])
                 k (gen/elements [8 16 32])]
                (let [ast (gemma/gemma4-two-stage-entity-resolver-ast dim dm vocab-size k)
                      expanded (expand/expand-ast {} ast)]
                  (and (vector? ast)
                       (seq expanded)
                       (every? ast/valid-node? expanded)))))

(deftest test-gemma4-two-stage-resolver-ast-compilation
  (let [dim 1536
        dm 128
        vocab-size 2000
        k 16
        invars [[:h [:tensor [1 1 dim] :f32]]
                [:u_norm [:tensor [1 1 dm] :f32]]
                [:cand_ids [:tensor [k] :i32]]
                [:W_embed [:tensor [vocab-size dim] :f32]]
                [:W_Q [:tensor [dim dm] :f32]]
                [:W_K [:tensor [dim dm] :f32]]]
        ast (gemma/gemma4-two-stage-resolver-ast dim dm vocab-size k)
        graph (uop-lower/ast->uop-graph "gemma_resolver_test" invars ast [:scores])]
    (is (some? graph))
    (is (= [:scores] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-gemma4-two-stage-entity-resolver-ast-compilation
  (let [dim 1536
        dm 128
        vocab-size 2000
        k 16
        invars [[:i_h [:tensor [1 1] :i32]]
                [:u_norm [:tensor [1 1 dm] :f32]]
                [:cand_ids [:tensor [k] :i32]]
                [:W_embed [:tensor [vocab-size dim] :f32]]
                [:W_Q [:tensor [dim dm] :f32]]
                [:W_K [:tensor [dim dm] :f32]]]
        ast (gemma/gemma4-two-stage-entity-resolver-ast dim dm vocab-size k)
        graph (uop-lower/ast->uop-graph "gemma_entity_resolver_test" invars ast [:scores])]
    (is (some? graph))
    (is (= [:scores] (:outvars graph)))
    (let [ctx (ein/get-context)
          compiled (ein/compile-graph ctx graph)]
      (is (some? compiled)))))

(deftest test-in-vram-resolver-step-compilation
  (let [ctx (ein/get-context)
        vocab-size 2000
        dim-in 1536
        k-triples 16
        dim-mem 128
        compiled (contrastive/compile-in-vram-resolver-step ctx vocab-size dim-in k-triples dim-mem {:dtype :f32})]
    (is (some? compiled))))


