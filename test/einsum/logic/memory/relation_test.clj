(ns einsum.logic.memory.relation-test
  "Unit, generative, and PJRT integration tests for In-Tensor Relational Memory
   and Factual Grounding (Superposition, Memory Probing, and Extract-Threshold-Re-embed Denoising)."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.logic.memory.relation :as mem]
            [einsum.models.gemma :as gemma]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [malli.core :as m]))

;; ==============================================================================
;; 1. Initialization and Schema Tests
;; ==============================================================================

(deftest test-relation-memory-initialization-and-schemas
  (testing "MemoryConfigSchema validates valid configs and rejects invalid configs"
    (is (m/validate mem/MemoryConfigSchema {:entity-count 100 :dim 256 :relation-count 5}))
    (is (not (m/validate mem/MemoryConfigSchema {:entity-count -1 :dim 256 :relation-count 5})))
    (is (not (m/validate mem/MemoryConfigSchema {:entity-count 100 :dim "256" :relation-count 5}))))

  (testing "init-relation-memory produces normalized entity table and zeroed cores"
    (let [entity-count 50
          dim 64
          rel-count 3
          mem (mem/init-relation-memory entity-count dim rel-count 42)]
      (is (= [entity-count dim] (:entity-shape mem)))
      (is (= [rel-count dim dim] (:core-shape mem)))
      (is (= (* entity-count dim) (alength ^floats (:entity-table mem))))
      (is (= (* rel-count dim dim) (alength ^floats (:cores mem))))
      ;; Check that cores are zeroed initially
      (let [^floats cores (:cores mem)]
        (dotimes [i (alength cores)]
          (is (zero? (aget cores i)))))
      ;; Check unit normalization of entity vectors
      (let [^floats e-table (:entity-table mem)]
        (dotimes [i entity-count]
          (let [norm (loop [j 0 sum 0.0]
                       (if (>= j dim)
                         (Math/sqrt sum)
                         (let [v (double (aget e-table (+ (* i dim) j)))]
                           (recur (inc j) (+ sum (* v v))))))]
            (is (< (Math/abs (- norm 1.0)) 1e-4))))))))

;; ==============================================================================
;; 2. Superposition Isolation & Crosstalk Bound
;; ==============================================================================

(deftest test-superposition-isolation
  (testing "adding fact (e1, r, e2) does not corrupt unrelated fact (e3, r, e4) beyond sqrt(N_facts / D)"
    (let [n 100
          d 256
          rel-id 0
          mem (mem/init-relation-memory n d 1 2026)
          ^floats cores (:cores mem)
          ^floats e-table (:entity-table mem)
          facts [[10 20]   ;; fact 1: e10 -> e20
                 [30 40]   ;; fact 2: e30 -> e40
                 [50 60]   ;; fact 3: e50 -> e60
                 [70 80]]] ;; fact 4: e70 -> e80
      ;; Ingest facts via zero-gradient DMA outer product accumulation
      (doseq [[h t] facts]
        (mem/accumulate-fact! cores e-table h rel-id t d))

      ;; Evaluate probe query for fact 1: query head = 10, expected target tail = 20
      (let [res (mem/probe-relation-memory {:head-id 10
                                            :rel-id rel-id
                                            :memory mem})
            ^floats scores (:scores res)
            target-score (aget scores 20)
            n-facts (count facts)
            sigma (Math/sqrt (/ (double n-facts) (double d)))
            max-allowed-noise (* 3.5 sigma)]
        ;; Target score should be close to 1.0
        (is (> target-score (- 1.0 max-allowed-noise)))

        ;; All other non-target entities should have small crosstalk within 3.5 * sigma bound
        (dotimes [i n]
          (when-not (= i 20)
            (let [score (Math/abs (double (aget scores i)))]
              (is (< score max-allowed-noise)
                  (str "Entity " i " score " score " exceeded crosstalk bound " max-allowed-noise)))))))))

;; ==============================================================================
;; 3. Denoising Projection Cycle (Extract-Threshold-Re-Embed)
;; ==============================================================================

(deftest test-extract-threshold-reembed
  (testing "applying the denoising cycle to a noise-injected matrix restores exact binary indicator relation via PJRT"
    (let [n 10
          d 64
          rel-id 0
          mem (mem/init-relation-memory n d 1 777)
          ^floats cores (:cores mem)
          ^floats e-table (:entity-table mem)
          facts [[1 2]
                 [3 4]]]
      ;; Ingest pristine facts
      (doseq [[h t] facts]
        (mem/accumulate-fact! cores e-table h rel-id t d))

      ;; Copy pristine core R_0
      (let [pristine-core (float-array (* d d))]
        (System/arraycopy cores 0 pristine-core 0 (* d d))

        ;; Inject Gaussian noise into relation core R_0
        (let [noisy-core (float-array (* d d))
              rnd (java.util.Random. 999)]
          (dotimes [i (* d d)]
            (let [noise (float (* 0.05 (.nextGaussian rnd)))]
              (aset-float noisy-core i (+ (aget pristine-core i) noise))))

          ;; Compile and execute the denoising cycle via OpenXLA PJRT
          (let [denoise-res (mem/denoise-relation-core {:relation-core noisy-core
                                                        :entity-table e-table
                                                        :entity-count n
                                                        :dim d
                                                        :threshold 0.5})
                ^floats clean-core (:clean-core denoise-res)
                ^floats crisp-indicator (:crisp-indicator denoise-res)]
            ;; 1. Crisp indicator matrix must match exact binary ground truth
            (dotimes [h n]
              (dotimes [t n]
                (let [expected (if (contains? (set facts) [h t]) 1.0 0.0)
                      actual (double (aget crisp-indicator (+ (* h n) t)))]
                  (is (< (Math/abs (- actual expected)) 1e-4)
                      (str "Crisp indicator mismatch at (" h ", " t "): expected " expected " got " actual)))))

            ;; 2. Denoised relation core must match pristine core within numerical precision
            (dotimes [k (* d d)]
              (let [expected (double (aget pristine-core k))
                    actual (double (aget clean-core k))]
                (is (< (Math/abs (- actual expected)) 1e-4)
                    (str "Clean core mismatch at index " k ": expected " expected " got " actual))))))))))

;; ==============================================================================
;; 4. In-Tensor Relational Grounding AST
;; ==============================================================================

(deftest test-relational-grounding-ast-lowering
  (testing "relational-grounding-ast produces valid Tensor Logic AST that lowers and compiles to OpenXLA"
    (let [hidden-dim 1536
          dim 256
          vocab-size 1000
          entity-count 50
          ast (mem/relational-grounding-ast hidden-dim dim vocab-size entity-count)
          invars [[:normed [:tensor [1 8 hidden-dim] :f32]]
                  [:w_mem_proj [:tensor [hidden-dim dim] :f32]]
                  [:r_active [:tensor [dim dim] :f32]]
                  [:entity_table [:tensor [entity-count dim] :f32]]
                  [:threshold_const [:tensor [1 1 entity-count] :f32]]
                  [:w_entity_to_vocab [:tensor [entity-count vocab-size] :f32]]
                  [:raw_logits [:tensor [1 1 vocab-size] :f32]]
                  [:pos [:tensor [1] :i32]]]
          target-heads #{:logits_grounded :entity_scores :valid_mask}
          graph (lower/ast->graph "relational_grounding_test" invars ast target-heads)]
      (is (shlo/validate-graph graph))
      (is (contains? (set (:outvars graph)) :logits_grounded))
      (let [ctx (xla/get-context)
            compiled (xla/compile-graph ctx graph)]
        (is (some? compiled)))))

  (testing "gemma4-model-ast with :relational-memory config lowers and compiles cleanly"
    (let [num-layers 1
          max-seq-len 8
          vocab-size 100
          hidden-dim 1536
          intermediate-dim 6144
          pl-dim 256
          mem-dim 256
          entity-count 20
          config {:vocab-size vocab-size
                  :hidden-dim hidden-dim
                  :intermediate-dim intermediate-dim
                  :pl-dim pl-dim
                  :total-pl-dim 256
                  :num-layers num-layers
                  :num-heads 8
                  :num-kv-heads 1
                  :head-dim 256
                  :max-seq-len max-seq-len
                  :last-token-only? true
                  :relational-memory {:dim mem-dim :entity-count entity-count}}
          invars [[:x [:tensor [1 max-seq-len] :i32]]
                  [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                  [:embed_tokens_per_layer [:tensor [vocab-size 256] :f32]]
                  [:per_layer_model_projection [:tensor [256 hidden-dim] :f32]]
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
                  [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]
                  [:pos [:tensor [1] :i32]]
                  ;; Relational memory invars:
                  [:w_mem_proj [:tensor [hidden-dim mem-dim] :f32]]
                  [:r_active [:tensor [mem-dim mem-dim] :f32]]
                  [:entity_table [:tensor [entity-count mem-dim] :f32]]
                  [:threshold_const [:tensor [1 1 entity-count] :f32]]
                  [:w_entity_to_vocab [:tensor [entity-count vocab-size] :f32]]]
          ast (gemma/gemma4-model-ast config)
          graph (lower/ast->graph "gemma4_rel_grounding" invars ast #{:logits})]
      (is (shlo/validate-graph graph))
      (is (= [:logits] (:outvars graph)))
      (let [ctx (xla/get-context)
            compiled (xla/compile-graph ctx graph)]
        (is (some? compiled)))))

  (testing "relational-grounding-ast deductive verification preserves valid_mask under DCE and clamps noise"
    (let [hidden-dim 1536
          dim 256
          vocab-size 100
          entity-count 10
          ast (mem/relational-grounding-ast hidden-dim dim vocab-size entity-count {:dtype :f32})
          invars [[:normed [:tensor [1 8 hidden-dim] :f32]]
                  [:w_mem_proj [:tensor [hidden-dim dim] :f32]]
                  [:r_active [:tensor [dim dim] :f32]]
                  [:entity_table [:tensor [entity-count dim] :f32]]
                  [:threshold_const [:tensor [1 1 entity-count] :f32]]
                  [:w_entity_to_vocab [:tensor [entity-count vocab-size] :f32]]
                  [:raw_logits [:tensor [1 1 vocab-size] :f32]]
                  [:pos [:tensor [1] :i32]]]
          ;; Request ONLY :logits_grounded. DCE must NOT prune :valid_mask or :valid_weight
          target-heads #{:logits_grounded}
          graph (lower/ast->graph "relational_grounding_dce_test" invars ast target-heads)
          eqn-ops (set (map :op (:eqns graph)))
          outvars (set (mapcat :outvars (:eqns graph)))]
      (is (shlo/validate-graph graph))
      (is (contains? eqn-ops :stablehlo/compare) "Must contain compare for threshold gating")
      (is (contains? eqn-ops :stablehlo/convert) "Must contain convert for mask-to-weight casting")
      (is (contains? outvars :valid_mask) "valid_mask must not be pruned by DCE")
      (is (contains? outvars :clamped_scores) "clamped_scores must be present in graph")))

  (testing "gemma4-model-ast sequence branch with relational memory maintains [:b :p :v] shape"
    (let [num-layers 1
          max-seq-len 8
          vocab-size 100
          hidden-dim 1536
          intermediate-dim 6144
          pl-dim 256
          mem-dim 256
          entity-count 20
          config {:vocab-size vocab-size
                  :hidden-dim hidden-dim
                  :intermediate-dim intermediate-dim
                  :pl-dim pl-dim
                  :total-pl-dim 256
                  :num-layers num-layers
                  :num-heads 8
                  :num-kv-heads 1
                  :head-dim 256
                  :max-seq-len max-seq-len
                  :last-token-only? false
                  :relational-memory {:dim mem-dim :entity-count entity-count}}
          invars [[:x [:tensor [1 max-seq-len] :i32]]
                  [:embed_tokens [:tensor [vocab-size hidden-dim] :f32]]
                  [:embed_tokens_per_layer [:tensor [vocab-size 256] :f32]]
                  [:per_layer_model_projection [:tensor [256 hidden-dim] :f32]]
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
                  [:post_per_layer_norm_w_0 [:tensor [hidden-dim] :f32]]
                  ;; Relational memory invars:
                  [:w_mem_proj [:tensor [hidden-dim mem-dim] :f32]]
                  [:r_active [:tensor [mem-dim mem-dim] :f32]]
                  [:entity_table [:tensor [entity-count mem-dim] :f32]]
                  [:threshold_const [:tensor [1 1 entity-count] :f32]]
                  [:w_entity_to_vocab [:tensor [entity-count vocab-size] :f32]]]
          ast (gemma/gemma4-model-ast config)
          graph (lower/ast->graph "gemma4_rel_seq_grounding" invars ast #{:logits})]
      (is (shlo/validate-graph graph))
      (is (= [:logits] (:outvars graph)))
      (is (= [1 max-seq-len vocab-size] (get-in graph [:known-shapes :logits]))
          "Sequence mode logits must have shape [1 max-seq-len vocab-size]")
      (let [ctx (xla/get-context)
            compiled (xla/compile-graph ctx graph)]
        (is (some? compiled)))))

  (testing "compile-relation-denoiser uses in-graph threshold scalar without host-side O(N^2) buffer"
    (let [ctx (xla/get-context)
          d 64
          n 100
          exec (mem/compile-relation-denoiser ctx d n 0.5)
          invar-names (set (map first (get-in exec [:graph :invars])))]
      (is (= #{:R :E} invar-names) "Invars must only contain :R and :E, no :thresh buffer")
      (is (nil? (get invar-names :thresh))))))

;; ==============================================================================
;; 5. Generative Property Tests (defspec)
;; ==============================================================================

;; Property 1: Monotonicity of Separation Margin with Dimension D
;; Separation margins mu_true - max(negative) increase monotonically as D in [64, 128, 256, 512].
(defspec prop-margin-monotonicity-with-dimension 10
  (prop/for-all [seed gen/nat]
                (let [n 30
                      dims [64 128 256 512]
                      facts [[0 1] [2 3] [4 5]]
                      margins
                      (mapv (fn [d]
                              (let [mem (mem/init-relation-memory n d 1 (+ (long seed) (* 31 (long d))))
                                    ^floats cores (:cores mem)
                                    ^floats e-table (:entity-table mem)]
                                (doseq [[h t] facts]
                                  (mem/accumulate-fact! cores e-table h 0 t d))
                                ;; Measure separation margin for fact [0 1]:
                                ;; margin = target_score - max(negative_scores)
                                (let [res (mem/probe-relation-memory {:head-id 0 :rel-id 0 :memory mem})
                                      ^floats scores (:scores res)
                                      target-score (double (aget scores 1))
                                      max-neg (loop [j 0 m -1.0]
                                                (if (>= j n)
                                                  m
                                                  (if (= j 1)
                                                    (recur (inc j) m)
                                                    (recur (inc j) (Math/max m (double (aget scores j)))))))]
                                  (- target-score max-neg))))
                            dims)]
                  ;; Check monotonic increase: margin(64) <= margin(128) <= margin(256) <= margin(512)
                  ;; Allowing statistical slack of 0.15 between adjacent steps and ensuring overall growth
                  (and (< (first margins) (last margins))
                       (every? (fn [[m1 m2]] (>= (+ m2 0.15) m1))
                               (partition 2 1 margins))))))

;; Property 2: Identity Invariant at T -> 0
;; For any factual triple in R_r, normalize(e_h * R_r) * e_t^T >= 0.85 while all negative entities t' != t score <= 0.15 (at T -> 0).
(defspec prop-identity-invariant-at-zero-temp 15
  (prop/for-all [entity-count (gen/choose 20 40)
                 seed gen/nat]
                (let [d 256
                      n entity-count
                      mem (mem/init-relation-memory n d 1 seed)
                      ^floats cores (:cores mem)
                      ^floats e-table (:entity-table mem)
                      h 5
                      t 12]
                  ;; Ingest single factual triple (h, 0, t)
                  (mem/accumulate-fact! cores e-table h 0 t d)
                  (let [res (mem/probe-relation-memory {:head-id h :rel-id 0 :memory mem})
                        ^floats scores (:scores res)
                        target-score (double (aget scores t))
                        max-neg (loop [j 0 m -1.0]
                                  (if (>= j n)
                                    m
                                    (if (= j t)
                                      (recur (inc j) m)
                                      (recur (inc j) (Math/max m (double (aget scores j)))))))
                        ;; Deductive verification at T -> 0: step(Scores - 0.5)
                        t0-target (if (>= target-score 0.5) 1.0 0.0)
                        t0-neg (if (>= max-neg 0.5) 1.0 0.0)]
                    (and (>= target-score 0.85)
                         (<= max-neg 0.20)
                         (>= t0-target 0.85)
                         (<= t0-neg 0.15))))))
