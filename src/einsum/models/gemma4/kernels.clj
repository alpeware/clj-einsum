(ns einsum.models.gemma4.kernels
  "Gemma 4 StableHLO MLIR executable compilation for prefill, KV-cache step, and in-VRAM while loop."
  (:require [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.models.gemma :as gemma-logic]
            [einsum.runtime.arena :as arena]))

(def GEMMA4-STOP-TOKEN-IDS
  "Special token IDs marking end-of-turn or end-of-generation in Gemma 4."
  #{1 106 49 50})

(defn build-tensor-logic-invars
  "Constructs EDN SSA signature invars for full Gemma 4 model forward pass."
  [config max-seq-len]
  (let [{:keys [vocab-size hidden-dim total-pl-dim pl-dim num-layers weight-dtype is-int8 is-int4 is-ternary layer-configs last-token-only? group-size backend target]} config
        int4? (boolean (or is-int4 (= weight-dtype :int4)))
        is-ternary (boolean (or is-ternary (= weight-dtype :ternary) (= (:quant-type config) :ternary)))
        norm-dtype (if (or is-int8 int4? is-ternary) :bf16 weight-dtype)
        has-ple? (pos? total-pl-dim)
        use-w4a16? (and int4?
                        (if (some? (:use-w4a16-gemv config))
                          (boolean (:use-w4a16-gemv config))
                          (or (= backend :rocm) (= target :rocm))))
        scale-groups-fn (fn [in-dim]
                          (if (and group-size (zero? (mod in-dim group-size)))
                            (quot in-dim group-size)
                            1))
        scale-shape-fn (fn [rows cols]
                         (cond
                           (and int4? group-size (zero? (mod cols group-size)))
                           [rows (quot cols group-size)]
                           (and is-ternary group-size (zero? (mod cols group-size)))
                           [rows (quot cols group-size)]
                           :else
                           [rows]))]
    (vec (concat [[:x [:tensor [1 max-seq-len] :i32]]]
                 (when last-token-only?
                   [[:pos [:tensor [1] :i32]]])
                 [[:embed_tokens [:tensor [vocab-size hidden-dim] norm-dtype]]]
                 (when has-ple?
                   [[:embed_tokens_per_layer [:tensor [vocab-size total-pl-dim] norm-dtype]]
                    [:per_layer_model_projection [:tensor [total-pl-dim hidden-dim] norm-dtype]]
                    [:per_layer_projection_norm [:tensor [pl-dim] norm-dtype]]])
                 (mapcat (fn [i]
                           (let [cfg (when (seq layer-configs) (nth layer-configs i nil))
                                 is-global? (if cfg (:is-global? cfg) (gemma-logic/layer-is-global? (:layer-types config) i))
                                 head-dim (long (or (:head-dim cfg)
                                                    (if is-global?
                                                      (or (:global-head-dim config) 512)
                                                      (or (:head-dim config) 256))))
                                 num-h (long (or (:num-heads cfg) (:num-heads config) 8))
                                 num-kv (long (or (:num-kv-heads cfg)
                                                  (if is-global?
                                                    (or (:num-global-kv-heads config) 1)
                                                    (:num-kv-heads config))
                                                  1))
                                 q-dim (long (or (:q-dim cfg) (* num-h head-dim)))
                                 kv-dim (long (or (:kv-dim cfg) (* num-kv head-dim)))
                                 mlp-dim (long (or (:mlp-dim cfg) (:intermediate-dim config) 6144))
                                 skipped? (contains? (set (:skip-layers config)) i)
                                 layer-is-ternary (and is-ternary (not skipped?))
                                 layer-is-int4 (and int4? (not skipped?))
                                 layer-is-int8 (and is-int8 (not skipped?))
                                 layer-use-w4a16 (and use-w4a16? (not skipped?))]
                             (concat
                              [[(keyword (str "input_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "layer_scalar_" i)) [:tensor [1] norm-dtype]]]
                              (cond
                                layer-is-ternary
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 4)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                layer-use-w4a16
                                [[(keyword (str "q_w_" i)) [:tensor [(quot hidden-dim 8) q-dim] :i32]]
                                 [(keyword (str "q_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) q-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [(quot hidden-dim 8) kv-dim] :i32]]
                                 [(keyword (str "k_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) kv-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [(quot hidden-dim 8) kv-dim] :i32]]
                                 [(keyword (str "v_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) kv-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [(quot q-dim 8) hidden-dim] :i32]]
                                 [(keyword (str "o_scale_" i)) [:tensor [(scale-groups-fn q-dim) hidden-dim] norm-dtype]]]
                                layer-is-int4
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor (scale-shape-fn q-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor (scale-shape-fn kv-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim (quot q-dim 2)] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor (scale-shape-fn hidden-dim q-dim) norm-dtype]]]
                                layer-is-int8
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] :i8]]
                                 [(keyword (str "q_scale_" i)) [:tensor [q-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "k_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] :i8]]
                                 [(keyword (str "v_scale_" i)) [:tensor [kv-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] :i8]]
                                 [(keyword (str "o_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "q_w_" i)) [:tensor [q-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "k_w_" i)) [:tensor [kv-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "v_w_" i)) [:tensor [kv-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "o_w_" i)) [:tensor [hidden-dim q-dim] norm-dtype]]])
                              [[(keyword (str "q_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "k_norm_w_" i)) [:tensor [head-dim] norm-dtype]]
                               [(keyword (str "post_attn_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "pre_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]
                               [(keyword (str "post_mlp_ln_w_" i)) [:tensor [hidden-dim] norm-dtype]]]
                              (cond
                                layer-is-ternary
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 4)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 4)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                layer-use-w4a16
                                [[(keyword (str "gate_w_" i)) [:tensor [(quot hidden-dim 8) mlp-dim] :i32]]
                                 [(keyword (str "gate_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) mlp-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [(quot hidden-dim 8) mlp-dim] :i32]]
                                 [(keyword (str "up_scale_" i)) [:tensor [(scale-groups-fn hidden-dim) mlp-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [(quot mlp-dim 8) hidden-dim] :i32]]
                                 [(keyword (str "down_scale_" i)) [:tensor [(scale-groups-fn mlp-dim) hidden-dim] norm-dtype]]]
                                layer-is-int4
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim (quot hidden-dim 2)] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor (scale-shape-fn mlp-dim hidden-dim) norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim (quot mlp-dim 2)] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor (scale-shape-fn hidden-dim mlp-dim) norm-dtype]]]
                                layer-is-int8
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "gate_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] :i8]]
                                 [(keyword (str "up_scale_" i)) [:tensor [mlp-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] :i8]]
                                 [(keyword (str "down_scale_" i)) [:tensor [hidden-dim] norm-dtype]]]
                                :else
                                [[(keyword (str "gate_w_" i)) [:tensor [mlp-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "up_w_" i)) [:tensor [mlp-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "down_w_" i)) [:tensor [hidden-dim mlp-dim] norm-dtype]]])
                              (when has-ple?
                                [[(keyword (str "per_layer_gate_w_" i)) [:tensor [pl-dim hidden-dim] norm-dtype]]
                                 [(keyword (str "per_layer_proj_w_" i)) [:tensor [hidden-dim pl-dim] norm-dtype]]
                                 [(keyword (str "post_per_layer_norm_w_" i)) [:tensor [hidden-dim] norm-dtype]]]))))
                         (range num-layers))
                 [[:final_norm_w [:tensor [hidden-dim] norm-dtype]]]
                 (when (or (:relational-memory config) (:relational-memory? config))
                   (let [rel-cfg (or (:relational-memory config) {})
                         mem-dim (long (or (:dim rel-cfg) (:memory-dim rel-cfg) 256))
                         entity-count (long (or (:entity-count rel-cfg) 1000))]
                     [[:w_mem_proj [:tensor [hidden-dim mem-dim] norm-dtype]]
                      [:r_active [:tensor [mem-dim mem-dim] norm-dtype]]
                      [:entity_table [:tensor [entity-count mem-dim] norm-dtype]]
                      [:threshold_const [:tensor [1 1 entity-count] norm-dtype]]
                      [:w_entity_to_vocab [:tensor [entity-count vocab-size] norm-dtype]]]))))))

(defn build-gemma4-kv-invars
  "Constructs EDN SSA signature invars for single-step Gemma 4 KV-Cache model forward pass."
  [config max-seq-len]
  (let [num-layers (long (or (:num-layers config) 35))
        num-kv-shared (long (or (:num-kv-shared-layers config) 0))
        num-unshared (- num-layers num-kv-shared)
        norm-dtype (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config)) :bf16 (get config :weight-dtype :bf16))
        layer-configs (:layer-configs config)
        layer-types (:layer-types config)
        kv-invars (mapcat (fn [i]
                            (let [cfg (when (seq layer-configs) (nth layer-configs i nil))
                                  is-global? (if cfg (:is-global? cfg) (gemma-logic/layer-is-global? layer-types i))
                                  win (when-not is-global? (or (:sliding-window cfg) (:sliding-window config) (:sliding_window config) 512))
                                  seq-l (if win (min max-seq-len win) max-seq-len)
                                  h-dim (long (or (:head-dim cfg)
                                                  (if is-global? 512 256)))
                                  n-kv (long (or (:num-kv-heads cfg) 1))]
                              [[(keyword (str "k_cache_in_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]
                               [(keyword (str "v_cache_in_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]]))
                          (range num-unshared))
        weight-invars (subvec (build-tensor-logic-invars (assoc config :last-token-only? true) max-seq-len) 2)]
    (vec (concat [[:x [:tensor [1 1] :i32]]
                  [:pos [:tensor [1] :i32]]]
                 kv-invars
                 weight-invars))))

(defn build-gemma4-kv-outvars
  "Constructs output variable list for Gemma 4 KV-Cache step: [:logits k_cache_out_0 v_cache_out_0 ...]."
  [config]
  (let [num-layers (long (or (:num-layers config) 35))
        num-kv-shared (long (or (:num-kv-shared-layers config) 0))
        num-unshared (- num-layers num-kv-shared)
        out-kv-heads (mapcat (fn [i]
                               [(keyword (str "k_cache_out_" i))
                                (keyword (str "v_cache_out_" i))])
                             (range num-unshared))]
    (vec (into [:logits] out-kv-heads))))

(defn build-gemma4-prefill-outvars
  "Constructs output variable list for Gemma 4 prefill: [:logits (k_ro_sl_i | k_ro_i) (v_heads_sl_i | v_heads_i) ...]."
  [config]
  (gemma-logic/gemma4-prefill-outvars config (:max-seq-len config)))

(defn allocate-kv-cache-buffers
  "Allocates initial zero-filled device VRAM buffers for KV cache of unshared layers.
   Optionally registers the buffers in `arena` (or `session-arena`)."
  ([session max-seq-len]
   (allocate-kv-cache-buffers session max-seq-len (or (:session-arena session) arena/*active-arena*)))
  ([{:keys [ctx config]} max-seq-len target-arena]
   (let [num-layers (long (or (:num-layers config) 35))
         num-kv-shared (long (or (:num-kv-shared-layers config) 0))
         num-unshared (- num-layers num-kv-shared)
         norm-enum (or (:norm-enum config) 13)
         layer-configs (:layer-configs config)
         layer-types (:layer-types config)]
     (vec (mapcat (fn [i]
                    (let [c (if (seq layer-configs) (nth layer-configs i nil) nil)
                          is-global? (if c (:is-global? c) (gemma-logic/layer-is-global? layer-types i))
                          win (when-not is-global? (or (:sliding-window c) (:sliding-window config) (:sliding_window config) 512))
                          seq-l (if win (min max-seq-len win) max-seq-len)
                          n-kv (long (or (:num-kv-heads c) 1))
                          h-dim (long (or (:head-dim c) (if is-global? 512 256)))
                          zeros (float-array (* seq-l n-kv h-dim))
                          k-buf (pjrt/buffer-from-host-buffer ctx (:client ctx) zeros [1 seq-l n-kv h-dim] norm-enum)
                          v-buf (pjrt/buffer-from-host-buffer ctx (:client ctx) zeros [1 seq-l n-kv h-dim] norm-enum)]
                      (when target-arena
                        (xla/track! target-arena k-buf)
                        (xla/track! target-arena v-buf))
                      [k-buf v-buf]))
                  (range num-unshared))))))

(defn compile-tensor-logic-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [last-token? (get opts :last-token-only? true)
        config-with-len (assoc config :max-seq-len max-seq-len :last-token-only? last-token?)
        invars (build-tensor-logic-invars config-with-len max-seq-len)
        ast (gemma-logic/gemma4-model-ast config-with-len)
        target-heads (or (:targets opts) #{:logits})
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 AST (%d layers, max-seq-len=%d, last-token-only=%s) to StableHLO..."
                             (:num-layers config) max-seq-len (str last-token?))))
        graph (lower/ast->graph "gemma4_model" invars ast target-heads)]
    (when-not (:quiet opts)
      (println "Compiling native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn compile-gemma4-prefill-executable
  "Compiles Gemma 4 model AST into a native StableHLO MLIR prefill executable that produces
   the next-token logits and initial populated KV cache tensors in a single parallel step."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        invars (build-tensor-logic-invars cfg max-seq-len)
        targets (build-gemma4-prefill-outvars cfg)
        ast (gemma-logic/gemma4-model-ast cfg)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 Prefill (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        graph (lower/ast->graph "gemma4_prefill" invars ast targets)]
    (when-not (:quiet opts)
      (println "Compiling Gemma 4 Prefill graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn compile-gemma4-kv-executable
  "Compiles single-step Gemma 4 KV-Cache AST into a native StableHLO MLIR executable."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        invars (build-gemma4-kv-invars cfg max-seq-len)
        targets (build-gemma4-kv-outvars cfg)
        ast (gemma-logic/gemma4-kv-model-ast cfg)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic Gemma 4 KV Cache Step (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        graph (lower/ast->graph "gemma4_kv_step" invars ast targets)]
    (when-not (:quiet opts)
      (println "Compiling Gemma 4 KV Cache step graph to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx graph)))

(defn compile-in-vram-loop-executable
  "Compiles an end-to-end in-VRAM autoregressive generation loop using StableHLO while-loop lowering with loop-carried KV-Cache."
  [{:keys [ctx config opts]} max-seq-len]
  (let [cfg (assoc config :max-seq-len max-seq-len :last-token-only? true)
        vocab-size (long (or (:vocab-size cfg) 262144))
        num-layers (long (or (:num-layers cfg) 35))
        num-kv-shared (long (or (:num-kv-shared-layers cfg) 0))
        num-unshared (- num-layers num-kv-shared)
        norm-dtype (if (or (:is-int8 cfg) (:is-int4 cfg) (:is-ternary cfg)) :bf16 (get cfg :weight-dtype :bf16))
        layer-configs (:layer-configs cfg)
        layer-types (:layer-types cfg)

        kv-invars (mapcat (fn [i]
                            (let [c (if (seq layer-configs) (nth layer-configs i nil) nil)
                                  is-global? (if c (:is-global? c) (gemma-logic/layer-is-global? layer-types i))
                                  win (when-not is-global? (or (:sliding-window c) (:sliding-window cfg) (:sliding_window cfg) 512))
                                  seq-l (if win (min max-seq-len win) max-seq-len)
                                  h-dim (or (:head-dim c) (if is-global? 512 256))
                                  n-kv (or (:num-kv-heads c) 1)]
                              [[(keyword (str "init_k_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]
                               [(keyword (str "init_v_" i)) [:tensor [1 seq-l n-kv h-dim] norm-dtype]]]))
                          (range num-unshared))
        weight-invars (subvec (build-tensor-logic-invars cfg max-seq-len) 2)
        kv-init-names (mapcat (fn [i] [(keyword (str "init_k_" i)) (keyword (str "init_v_" i))]) (range num-unshared))
        kv-state-names (mapcat (fn [i] [(keyword (str "k_cache_in_" i)) (keyword (str "v_cache_in_" i))]) (range num-unshared))
        kv-out-names (mapcat (fn [i] [(keyword (str "k_cache_out_" i)) (keyword (str "v_cache_out_" i))]) (range num-unshared))
        kv-final-names (mapcat (fn [i] [(keyword (str "final_k_" i)) (keyword (str "final_v_" i))]) (range num-unshared))

        cond-args (into [:cur_step :target_max :cur_toks :cur_stopped] kv-state-names)
        cond-ast [:cond [:cond_out] {:args cond-args}
                  [:compare [:step_lt] [:cur_step] [:target_max] {:direction "LT"}]
                  [:not [:not_stopped] [:cur_stopped]]
                  [:and [:cond_out] [:step_lt] [:not_stopped]]]

        body-in-args cond-args
        body-out-args (into [:next_step :target_max :next_toks :is_stop] kv-out-names)
        model-ast (gemma-logic/gemma4-kv-model-ast cfg)

        body-ast [:body body-out-args
                  {:args body-in-args}

                  ;; 1. Current position & single token input bindings
                  [:constant [:c_one] {:value 1 :type [:tensor [] :i32] :shape []}]
                  [:- [:pos_i32] [:cur_step] [:c_one]]
                  [:reshape [:pos] [:pos_i32] {:shape [1]}]
                  [:dynamic-slice [:cur_tok_2d] [:cur_toks] {:start-indices [0 :pos] :slice-sizes [1 1]}]
                  [:reshape [:x :b :p] [:cur_tok_2d] {:shape [1 1]}]

                  ;; 2. Single-step Gemma 4 forward pass with KV cache
                  model-ast

                  ;; 3. Argmax & loop state update
                  [:reshape [:logits_2d] [:logits] {:shape [1 vocab-size]}]
                  [:convert [:logits_f32] [:logits_2d] {:target-dtype :f32}]
                  [:argmax [:next_tok] [:logits_f32] {:axis 1}]
                  [:reshape [:next_tok_1d] [:next_tok] {:shape [1 1]}]
                  [:dynamic-update-slice [:next_toks] [:cur_toks] [:next_tok_1d] {:start-indices [0 :cur_step]}]
                  [:+ [:next_step] [:cur_step] [:c_one]]
                  [:constant [:eos_c] {:value 1 :type [:tensor [1] :i32] :shape [1]}]
                  [:constant [:eot_c] {:value 106 :type [:tensor [1] :i32] :shape [1]}]
                  [:constant [:etc_c] {:value 49 :type [:tensor [1] :i32] :shape [1]}]
                  [:constant [:str_c] {:value 50 :type [:tensor [1] :i32] :shape [1]}]
                  [:compare [:c_eos] [:next_tok] [:eos_c] {:direction "EQ"}]
                  [:compare [:c_eot] [:next_tok] [:eot_c] {:direction "EQ"}]
                  [:compare [:c_etc] [:next_tok] [:etc_c] {:direction "EQ"}]
                  [:compare [:c_str] [:next_tok] [:str_c] {:direction "EQ"}]
                  [:or [:or_stop_0] [:c_eos] [:c_eot]]
                  [:or [:or_stop_1] [:c_etc] [:c_str]]
                  [:or [:or_stop] [:or_stop_0] [:or_stop_1]]
                  [:reshape [:is_stop] [:or_stop] {:shape []}]]

        init-loop-args (into [:init_step :max_step :init_tokens :false_c] kv-init-names)
        while-out-vars (into [:final_step :final_max :final_tokens :final_stopped] kv-final-names)

        loop-ast [:block {}
                  [:constant [:false_c] {:value false :type [:tensor [] :i1] :shape []}]
                  [:while while-out-vars
                   init-loop-args
                   cond-ast
                   body-ast]]

        loop-invars (vec (concat [[:init_step [:tensor [] :i32]]
                                  [:max_step [:tensor [] :i32]]
                                  [:init_tokens [:tensor [1 max-seq-len] :i32]]]
                                 kv-invars
                                 weight-invars))
        loop-targets (into [:final_step :final_tokens] kv-final-names)
        _ (when-not (:quiet opts)
            (println (format "Lowering declarative Tensor Logic In-VRAM Gemma 4 KV Loop (%d layers, max-seq-len=%d) to StableHLO..."
                             (:num-layers config) max-seq-len)))
        loop-graph (lower/ast->graph "gemma4_in_vram_kv_loop" loop-invars loop-ast loop-targets)]
    (when-not (:quiet opts)
      (println "Compiling In-VRAM KV Loop to native XLA PjRtLoadedExecutable..."))
    (xla/compile-graph ctx loop-graph)))
