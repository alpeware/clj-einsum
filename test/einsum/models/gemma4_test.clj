(ns einsum.models.gemma4-test
  "Generative specification tests for modularized Gemma 4 domain namespaces:
   config, weights, kernels, and runtime."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gemma :as gemma-logic]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]
            [einsum.models.gemma4.weights :as weights]
            [einsum.runtime.arena :as arena]
            [einsum.uop.core :as uop]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.schedule :as schedule]))

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
  ;; A small 1-layer gemma4 model scoring graph lowers to a valid UOp graph
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
        graph (uop-lower/ast->uop-graph "gemma4_scoring" invars ast targets)]
    (is (some? graph))
    (is (= [:target_log_probs] (:outvars graph)))))

(defspec prop-score-sequence-log-probs-validation 50
  (prop/for-all [tokens (gen/vector (gen/choose 1 1000) 0 1)]
    ;; Should throw exception for tokens count < 2
                (try
                  (rt/score-sequence-log-probs {} nil tokens)
                  false
                  (catch clojure.lang.ExceptionInfo e
                    (= "Sequence must contain at least 2 tokens to score target log-probabilities"
                       (.getMessage e))))))

(deftest test-lower-chunked-attention-prefill-bypass
  (testing "lower-chunked-attention with p in [8 16 32], kvs = 128, chunk-size = 64 lowers directly via 4D attention path (4 kernels instead of 6)"
    (doseq [p [8 16 32]]
      (let [b 1 h 2 dh 16 kvs 128 kvh 2
            chunk-size 64
            scale (/ 1.0 (Math/sqrt (double dh)))
            invars [[:Q [:tensor [b p h dh] :f32]]
                    [:K [:tensor [b kvs kvh dh] :f32]]
                    [:V [:tensor [b kvs kvh dh] :f32]]
                    [:pos [:tensor [1] :i32]]]
            ast [:block {}
                 [:chunked-attention [:ctx :b :p :h :dh]
                  [:Q :b :p :h :dh]
                  [:K :b :kvs :kvh :dh]
                  [:V :b :kvs :kvh :dh]
                  {:pos :pos
                   :chunk-size chunk-size
                   :scale scale
                   :sliding-window 64}]]
            g (uop-lower/ast->uop-graph (str "attn_prefill_" p) invars ast [:ctx])
            s (schedule/schedule-graph g)
            ;; Directly lowers via 4D attention path: 4 reduction kernels (QK, max, sum, SV)
            ;; without 5D intermediate chunk tensors (6 total scheduled kernels vs 9 in chunked mode).
            red-kernels (filter #(some (fn [n] (= (:op n) :uop/reduce-axis))
                                       (uop/toposort (:sink (:rangeified %))))
                                (:kernels s))]
        (is (= 4 (count red-kernels)))
        (is (= 6 (count (:kernels s))))
        (is (every? #(== 4 (count (:out-shape (:rangeified %)))) (:kernels s)))))))

(deftest test-precompiled-chunk-executable-short-prompt-prefill
  (testing "Precompiled :chunked-prefill-executable (chunk-size=64) preserves target-chunk-sz=64 on sub-64 prompts and routes through :delta-chunk-executables without dispatching 64-token exec on a short buffer, while bespoke sessions clamp to prompt-count"
    (let [ctx {:backend :uop :uop-target :cpu}
          cfg {:num-layers 1
               :num-kv-shared-layers 0
               :vocab-size 16
               :weight-dtype :f32
               :max-seq-len 128}
          exec-1 {:id :exec-1}
          exec-16 {:id :exec-16}
          exec-32 {:id :exec-32}
          exec-64 {:id :exec-64}
          delta-execs (into (sorted-map-by >) {64 exec-64 32 exec-32 16 exec-16})
          short-prompt (vec (range 1 23)) ;; 22 tokens (< 64)
          compile-calls (atom [])
          dispatched-calls (atom [])]
      (with-redefs [arena/device-buffer
                    (fn
                      ([host-data shape dtype]
                       {:dtype dtype :shape (vec shape) :data host-data})
                      ([_arena host-data shape dtype]
                       {:dtype dtype :shape (vec shape) :data host-data}))
                    kernels/compile-gemma4-chunked-prefill-executable
                    (fn [_sess max-seq csz]
                      (swap! compile-calls conj {:max-seq-len max-seq :chunk-size csz})
                      {:id (keyword (str "bespoke-" csz))})
                    kernels/allocate-kv-cache-buffers
                    (fn [_sess _seq-len _arena]
                      [{:id :k0} {:id :v0}])
                    weights/allocate-device-weights
                    (fn [_sess] [{:id :w0}])
                    rt/execute-step-executable
                    (fn [_ctx ex step-inputs _num-outs]
                      (let [x-buf (first step-inputs)
                            pos-buf (second step-inputs)
                            pos-val (aget ^ints (:data pos-buf) 0)]
                        (swap! dispatched-calls conj {:exec-id (:id ex)
                                                      :x-shape (:shape x-buf)
                                                      :pos pos-val})
                        [{:id :logits :pos pos-val} {:id :k-out} {:id :v-out}]))]
        ;; 1. Precompiled session (chunk-size=64, prompt=22 tokens):
        ;;    Must NOT clamp target-chunk-sz to 22 or dispatch :exec-64 on [1 22];
        ;;    must route all 22 tokens via :delta-chunk-executables (:exec-16 at pos 0 and suffix pos 6).
        (let [precompiled-session {:ctx ctx
                                   :config cfg
                                   :opts {:chunk-size 64 :quiet true}
                                   :session-arena (arena/create-arena ctx)
                                   :device-weights [{:id :w0}]
                                   :executable exec-1
                                   :chunked-prefill-executable exec-64
                                   :delta-chunk-executables delta-execs}
              resolved (rt/resolve-active-session precompiled-session (:opts precompiled-session) short-prompt 128)
              active-sess (:active-session resolved)
              prefill-res (rt/prefill-cached-kv-session! active-sess exec-1 [{:id :w0}] short-prompt 128)]
          (is (= exec-64 (:chunked-prefill-executable active-sess)))
          (is (nil? (:tail-prefill-executable active-sess)))
          (is (empty? @compile-calls))
          (is (= [{:exec-id :exec-16 :x-shape [1 16] :pos 0}
                  {:exec-id :exec-16 :x-shape [1 16] :pos 6}]
                 @dispatched-calls))
          (is (= {:id :logits :pos 6} (:last-logits prefill-res))))

        ;; 2. Bespoke session (no precompiled :chunked-prefill-executable, chunk-size=64, prompt=22 tokens):
        ;;    Must clamp target-chunk-sz to (min 22 64) = 22 and compile/dispatch a 22-token chunk executable.
        (reset! compile-calls [])
        (reset! dispatched-calls [])
        (let [bespoke-session {:ctx ctx
                               :config cfg
                               :opts {:chunk-size 64 :quiet true}
                               :session-arena (arena/create-arena ctx)
                               :device-weights [{:id :w0}]
                               :executable exec-1}
              resolved (rt/resolve-active-session bespoke-session (:opts bespoke-session) short-prompt 128)
              active-sess (:active-session resolved)
              prefill-res (rt/prefill-cached-kv-session! active-sess exec-1 [{:id :w0}] short-prompt 128)]
          (is (= [{:max-seq-len 128 :chunk-size 22}] @compile-calls))
          (is (= {:id :bespoke-22} (:chunked-prefill-executable active-sess)))
          (is (nil? (:tail-prefill-executable active-sess)))
          (is (= [{:exec-id :bespoke-22 :x-shape [1 22] :pos 0}]
                 @dispatched-calls))
          (is (= {:id :logits :pos 0} (:last-logits prefill-res))))))))
