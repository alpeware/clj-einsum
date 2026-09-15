(ns clj-xla.logic.memory.relation
  "In-tensor relational memory core and embedding-space operations for OpenXLA PJRT execution.
   Implements zero-gradient outer-product superposition, compiled in-tensor memory probing,
   extract-threshold-re-embed denoising cycles, and Gemma 4 relational grounding AST."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

(def MemoryConfigSchema
  "Schema for relational memory configuration."
  [:map
   [:entity-count [:and :int [:> 0]]]
   [:dim [:and :int [:> 0]]]
   [:relation-count [:and :int [:> 0]]]])

(defn init-relation-memory
  "Generates initial normalized entity table E [N D] and zeroed relation cores [K D D]."
  [entity-count dim relation-count seed]
  (let [n (long entity-count)
        d (long dim)
        k (long relation-count)
        emb (sym/random-embeddings n d seed)]
    {:entity-table (:data emb)
     :entity-shape [n d]
     :cores (float-array (* k d d))
     :core-shape [k d d]
     :entity-embeddings emb
     :entity-count n
     :dim d
     :relation-count k}))

(defn accumulate-fact!
  "In-place zero-gradient addition of fact outer product: R_r += e_h^T * e_t."
  [^floats cores ^floats entity-table head-id rel-id tail-id dim]
  (let [d (long dim)
        rel-offset (* (long rel-id) d d)
        h-offset (* (long head-id) d)
        t-offset (* (long tail-id) d)]
    (dotimes [i d]
      (let [h-val (aget entity-table (+ h-offset i))
            row-offset (+ rel-offset (* i d))]
        (dotimes [j d]
          (let [t-val (aget entity-table (+ t-offset j))
                idx (+ row-offset j)]
            (aset cores idx (+ (aget cores idx) (* h-val t-val)))))))))

;; ==============================================================================
;; OpenXLA PJRT Compiled In-Tensor Memory Probing
;; ==============================================================================

(defn- probe-memory-ast
  [dim _entity-count]
  (let [d (long dim)]
    [:block {:name :probe_relation_memory}
     [:= [:V_target :d] [:V_q :k] [:R :k :d]]
     [:rms-norm [:V_norm :d] [:V_target :d]]
     [:= [:V_unit :d] {:scale (/ 1.0 (Math/sqrt (double d)))} [:V_norm :d]]
     [:= [:Scores :n] [:V_unit :d] [:E :n :d]]]))

(defn compile-probe-query
  "Compiles an OpenXLA PJRT executable for querying relation memory cores."
  ([dim entity-count]
   (compile-probe-query (xla/get-context) dim entity-count))
  ([ctx dim entity-count]
   (let [d (long dim)
         n (long entity-count)
         invars [[:V_q [:tensor [d] :f32]]
                 [:R [:tensor [d d] :f32]]
                 [:E [:tensor [n d] :f32]]]
         ast (probe-memory-ast d n)]
     (sym/compile-query ctx "probe_relation_memory" invars ast [:Scores]))))

(defn probe-relation-memory
  "Evaluates in-tensor probe query via OpenXLA PJRT (Rule 4: Zero Java bypasses).
   Contraction: v_target = v_q * R_r, v_unit = normalize(v_target), scores = v_unit * E^T.
   Accepts {:keys [head-id rel-id memory query-vec relation-core entity-table dim entity-count]}."
  ([opts]
   (probe-relation-memory (xla/get-context) opts))
  ([ctx {:keys [head-id rel-id memory query-vec relation-core entity-table dim entity-count]}]
   (let [mem memory
         e-arr ^floats (or entity-table (:entity-table mem))
         d (long (or dim (:dim mem)))
         n (long (or entity-count (:entity-count mem)))
         r-id (long (or rel-id 0))
         h-id (when head-id (long head-id))
         vq-arr (if query-vec
                  ^floats query-vec
                  (let [buf (float-array d)]
                    (System/arraycopy e-arr (int (* h-id d)) buf 0 (int d))
                    buf))
         r-arr (if relation-core
                 ^floats relation-core
                 (let [all-cores ^floats (:cores mem)
                       buf (float-array (* d d))
                       offset (* r-id d d)]
                   (System/arraycopy all-cores (int offset) buf 0 (int (* d d)))
                   buf))
         exec (compile-probe-query ctx d n)
         out (sym/run-query! exec {:V_q vq-arr :R r-arr :E e-arr})
         scores (get out :Scores)]
     {:scores scores})))

;; ==============================================================================
;; OpenXLA PJRT Compiled Extract-Threshold-Re-Embed Denoising Cycle
;; ==============================================================================

(defn- denoise-core-ast
  []
  [:block {:name :denoise_relation_core}
   ;; 1. ER = E * R  [n, d] x [d, d] -> [n, d]
   [:= [:ER :n :d] [:E :n :k] [:R :k :d]]
   ;; 2. S = ER * E^T  [n, d] x [n, d] -> [n, n]
   [:= [:S :h :t] [:ER :h :d] [:E :t :d]]
   ;; 3. Deductive verification at T -> 0: crisp indicator A = step(S - threshold)
   [:compare [:mask :h :t] [:S :h :t] [:thresh :h :t] {:comparison_direction "GE"}]
   [:convert [:A_crisp :h :t] {:target-dtype :f32} [:mask :h :t]]
   ;; 4. Re-embed: R_clean = E^T * A_crisp * E  [d, n] x [n, n] x [n, d] -> [d, d]
   [:= [:ETA :i :t] [:E :h :i] [:A_crisp :h :t]]
   [:= [:R_clean :i :j] [:ETA :i :t] [:E :t :j]]])

(defn compile-relation-denoiser
  "Compiles an OpenXLA PJRT executable for algebraic denoising (Section 2.4):
   A_crisp = step(E * R * E^T - 0.5), R_clean = E^T * A_crisp * E."
  ([dim entity-count]
   (compile-relation-denoiser (xla/get-context) dim entity-count))
  ([ctx dim entity-count]
   (let [d (long dim)
         n (long entity-count)
         invars [[:R [:tensor [d d] :f32]]
                 [:E [:tensor [n d] :f32]]
                 [:thresh [:tensor [n n] :f32]]]
         ast (denoise-core-ast)]
     (sym/compile-query ctx "denoise_relation_core" invars ast [:R_clean :A_crisp]))))

(defn denoise-relation-core
  "Applies extract-threshold-re-embed denoising cycle via OpenXLA PJRT (Rule 4).
   Restores pristine superposition matrix and extracts crisp binary fact indicators."
  ([opts]
   (denoise-relation-core (xla/get-context) opts))
  ([ctx {:keys [relation-core entity-table entity-count dim threshold]}]
   (let [d (long dim)
         n (long entity-count)
         thresh-val (float (or threshold 0.5))
         thresh-arr (float-array (* n n))
         _ (java.util.Arrays/fill thresh-arr thresh-val)
         exec (compile-relation-denoiser ctx d n)
         out (sym/run-query! exec {:R relation-core
                                   :E entity-table
                                   :thresh thresh-arr})]
     {:clean-core (get out :R_clean)
      :crisp-indicator (get out :A_crisp)})))

(defn denoise-relation-core!
  "In-place algebraic denoising of relation core `rel-id` in resident `cores` array."
  ([cores entity-table rel-id entity-count dim]
   (denoise-relation-core! (xla/get-context) cores entity-table rel-id entity-count dim 0.5))
  ([ctx ^floats cores ^floats entity-table rel-id entity-count dim threshold]
   (let [d (long dim)
         rel-offset (* (long rel-id) d d)
         noisy-core (float-array (* d d))]
     (System/arraycopy cores (int rel-offset) noisy-core 0 (int (* d d)))
     (let [res (denoise-relation-core ctx {:relation-core noisy-core
                                           :entity-table entity-table
                                           :entity-count entity-count
                                           :dim dim
                                           :threshold threshold})
           ^floats clean (:clean-core res)]
       (System/arraycopy clean 0 cores (int rel-offset) (int (* d d)))
       res))))

;; ==============================================================================
;; In-Tensor Relational Grounding AST Constructor
;; ==============================================================================

(defn relational-grounding-ast
  "Constructs Tensor Logic AST for in-tensor memory probing and logit clamping (Section 4.2).
   Integrates into Gemma 4 forward graph right before or alongside token logit projection."
  ([hidden-dim dim vocab-size entity-count]
   (relational-grounding-ast hidden-dim dim vocab-size entity-count {}))
  ([hidden-dim _dim _vocab-size _entity-count _opts]
   [:block {:name :relational_grounding}
    ;; 1. Extract probe vector from last-token hidden state
    [:dynamic-slice [:h_probe :b :one :dim] [:normed :b :p :dim]
     {:slice-sizes [1 1 (long hidden-dim)] :start-indices [0 :pos 0]}]

    ;; 2. Project from transformer hidden-dim to memory dim D
    [:= [:v_q :b :one :d]
     [:h_probe :b :one :dim] [:w_mem_proj :dim :d]]

    ;; 3. Contraction across the active relation core: v_target = v_q * R_r
    [:= [:v_target :b :one :d]
     [:v_q :b :one :k] [:r_active :k :d]]
    [:rms-norm [:v_target_norm :b :one :d] [:v_target :b :one :d]]

    ;; 4. Entity scoring: Scores = v_target_norm * E^T
    [:= [:entity_scores :b :one :n_entities]
     [:v_target_norm :b :one :d] [:entity_table :n_entities :d]]

    ;; 5. Deductive verification at T -> 0: step(Scores - 0.5)
    [:compare [:valid_mask :b :one :n_entities]
     [:entity_scores :b :one :n_entities] [:threshold_const :one]
     {:comparison_direction "GT"}]

    ;; 6. Back-projection into vocab space for logit clamping
    [:= [:vocab_bias :b :one :v]
     [:entity_scores :b :one :n_entities] [:w_entity_to_vocab :n_entities :v]]

    ;; 7. Fused logit addition
    [:+ [:logits_grounded :b :one :v]
     [:raw_logits :b :one :v] [:vocab_bias :b :one :v]]]))
