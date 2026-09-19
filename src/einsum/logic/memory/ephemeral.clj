(ns einsum.logic.memory.ephemeral
  "Zero-Gradient Ephemeral Online Learning for Pedro Domingos' Declarative Tensor Logic (Experiment E5).
   Compiles Hebbian outer-product fast-weight writes, batch superposition, fact retractions,
   in-graph relational composition, and extract-threshold-re-embed denoising cycles into OpenXLA PJRT.
   Strictly adheres to Rule 2 (Pure Functions) and Rule 4 (Zero Java bypasses / pure XLA execution)."
  (:require [einsum.core :as xla]
            [einsum.logic.symbolic :as sym]))

;; ==============================================================================
;; Schemas
;; ==============================================================================

(def EphemeralWriteOptionsSchema
  "Schema for online memory write options."
  [:map
   [:alpha {:optional true} [:and :double [:>= 0.0] [:<= 1.0]]]
   [:beta {:optional true} [:and :double [:> 0.0]]]])

(def EphemeralQueryOptionsSchema
  "Schema for memory query options."
  [:map
   [:temperature {:optional true} [:and :double [:> 0.0]]]])

(def EphemeralDenoiseOptionsSchema
  "Schema for extract-threshold-re-embed denoising options."
  [:map
   [:threshold {:optional true} [:and :double [:>= 0.0] [:<= 1.0]]]])

;; ==============================================================================
;; 1. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn write-fact-ast
  "Constructs Tensor Logic AST for single fact outer-product injection:
   Outer = beta * (e_h (x) e_t)
   Scaled_Old = alpha * R
   R_updated = Scaled_Old + Outer"
  ([_dim]
   (write-fact-ast _dim nil))
  ([_dim opts]
   (let [a (double (or (:alpha opts) 1.0))
         b (double (or (:beta opts) 1.0))]
     [:block {:name :ephemeral_write_fact}
      [:= [:outer :d1 :d2] {:scale b} [:e_h 1 :d1] [:e_t 1 :d2]]
      [:= [:scaled_old :d1 :d2] {:scale a} [:R :d1 :d2]]
      [:+ [:R_updated :d1 :d2] [:scaled_old :d1 :d2] [:outer :d1 :d2]]])))

(defn write-facts-batch-ast
  "Constructs Tensor Logic AST for batch fact outer-product superposition:
   Outer_sum = beta * sum_b (H_b (x) T_b)
   Scaled_Old = alpha * R
   R_updated = Scaled_Old + Outer_sum"
  ([_batch-size _dim]
   (write-facts-batch-ast _batch-size _dim nil))
  ([_batch-size _dim opts]
   (let [a (double (or (:alpha opts) 1.0))
         b (double (or (:beta opts) 1.0))]
     [:block {:name :ephemeral_write_facts_batch}
      [:= [:outer_sum :d1 :d2] {:scale b} [:H :b :d1] [:T :b :d2]]
      [:= [:scaled_old :d1 :d2] {:scale a} [:R :d1 :d2]]
      [:+ [:R_updated :d1 :d2] [:scaled_old :d1 :d2] [:outer_sum :d1 :d2]]])))

(defn retract-fact-ast
  "Constructs Tensor Logic AST for single fact retraction:
   Outer = e_h (x) e_t
   R_updated = R - Outer"
  [_dim]
  [:block {:name :ephemeral_retract_fact}
   [:= [:outer :d1 :d2] [:e_h 1 :d1] [:e_t 1 :d2]]
   [:- [:R_updated :d1 :d2] [:R :d1 :d2] [:outer :d1 :d2]]])

(defn retract-facts-batch-ast
  "Constructs Tensor Logic AST for batch fact retraction:
   Outer_sum = sum_b (H_b (x) T_b)
   R_updated = R - Outer_sum"
  [_batch-size _dim]
  [:block {:name :ephemeral_retract_facts_batch}
   [:= [:outer_sum :d1 :d2] [:H :b :d1] [:T :b :d2]]
   [:- [:R_updated :d1 :d2] [:R :d1 :d2] [:outer_sum :d1 :d2]]])

(defn query-memory-ast
  "Constructs Tensor Logic AST for unbinding and scoring against entity table E:
   unbound = e_q * R
   scores = unbound * E^T"
  ([_dim _entity-count]
   (query-memory-ast _dim _entity-count nil))
  ([_dim _entity-count _opts]
   [:block {:name :ephemeral_query_memory}
    [:= [:unbound 1 :d2] [:e_q 1 :d1] [:R :d1 :d2]]
    [:= [:scores 1 :n] [:unbound 1 :d2] [:E :n :d2]]]))

(defn compose-relations-ast
  "Constructs Tensor Logic AST for transitive relational composition:
   R_composed = R1 * R2"
  [_dim]
  [:block {:name :ephemeral_compose_relations}
   [:= [:R_composed :d1 :d3] [:R1 :d1 :d2] [:R2 :d2 :d3]]])

(defn denoise-relation-ast
  "Constructs Tensor Logic AST for extract-threshold-re-embed denoising cycle:
   S = E * R * E^T
   A_crisp = step(S - threshold)
   R_clean = E^T * A_crisp * E"
  ([_dim _entity-count]
   (denoise-relation-ast _dim _entity-count nil))
  ([_dim _entity-count opts]
   (let [thresh (double (or (:threshold opts) 0.5))]
     [:block {:name :ephemeral_denoise_relation}
      [:= [:ER :n :d] [:E :n :k] [:R :k :d]]
      [:= [:S :h :t] [:ER :h :d] [:E :t :d]]
      [:constant [:thresh] {:value thresh :type [:tensor [] :f32] :shape []}]
      [:compare [:mask :h :t] [:S :h :t] [:thresh] {:comparison_direction "GE"}]
      [:convert [:A_crisp :h :t] {:target-dtype :f32} [:mask :h :t]]
      [:= [:ETA :i :t] [:E :h :i] [:A_crisp :h :t]]
      [:= [:R_clean :i :j] [:ETA :i :t] [:E :t :j]]])))

;; ==============================================================================
;; 2. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn compile-write-fact
  "Compiles OpenXLA PJRT executable for single fact write."
  ([dim]
   (compile-write-fact (xla/get-context) dim nil))
  ([a b]
   (if (map? a)
     (compile-write-fact a b nil)
     (compile-write-fact (xla/get-context) a b)))
  ([ctx dim opts]
   (let [d (long dim)
         invars [[:R [:tensor [d d] :f32]]
                 [:e_h [:tensor [1 d] :f32]]
                 [:e_t [:tensor [1 d] :f32]]]
         ast (write-fact-ast d opts)]
     (sym/compile-query ctx "ephemeral_write_fact" invars ast [:R_updated]))))

(defn compile-write-facts-batch
  "Compiles OpenXLA PJRT executable for batch fact superposition."
  ([batch-size dim]
   (compile-write-facts-batch (xla/get-context) batch-size dim nil))
  ([a b c]
   (if (map? a)
     (compile-write-facts-batch a b c nil)
     (compile-write-facts-batch (xla/get-context) a b c)))
  ([ctx batch-size dim opts]
   (let [b (long batch-size)
         d (long dim)
         invars [[:R [:tensor [d d] :f32]]
                 [:H [:tensor [b d] :f32]]
                 [:T [:tensor [b d] :f32]]]
         ast (write-facts-batch-ast b d opts)]
     (sym/compile-query ctx "ephemeral_write_facts_batch" invars ast [:R_updated]))))

(defn compile-retract-fact
  "Compiles OpenXLA PJRT executable for single fact retraction."
  ([dim]
   (compile-retract-fact (xla/get-context) dim))
  ([ctx dim]
   (let [d (long dim)
         invars [[:R [:tensor [d d] :f32]]
                 [:e_h [:tensor [1 d] :f32]]
                 [:e_t [:tensor [1 d] :f32]]]
         ast (retract-fact-ast d)]
     (sym/compile-query ctx "ephemeral_retract_fact" invars ast [:R_updated]))))

(defn compile-retract-facts-batch
  "Compiles OpenXLA PJRT executable for batch fact retraction."
  ([batch-size dim]
   (compile-retract-facts-batch (xla/get-context) batch-size dim))
  ([ctx batch-size dim]
   (let [b (long batch-size)
         d (long dim)
         invars [[:R [:tensor [d d] :f32]]
                 [:H [:tensor [b d] :f32]]
                 [:T [:tensor [b d] :f32]]]
         ast (retract-facts-batch-ast b d)]
     (sym/compile-query ctx "ephemeral_retract_facts_batch" invars ast [:R_updated]))))

(defn compile-query-memory
  "Compiles OpenXLA PJRT executable for unbinding and scoring against entity dictionary."
  ([dim entity-count]
   (compile-query-memory (xla/get-context) dim entity-count nil))
  ([a b c]
   (if (map? a)
     (compile-query-memory a b c nil)
     (compile-query-memory (xla/get-context) a b c)))
  ([ctx dim entity-count opts]
   (let [d (long dim)
         n (long entity-count)
         invars [[:e_q [:tensor [1 d] :f32]]
                 [:R [:tensor [d d] :f32]]
                 [:E [:tensor [n d] :f32]]]
         ast (query-memory-ast d n opts)]
     (sym/compile-query ctx "ephemeral_query_memory" invars ast [:scores]))))

(defn compile-compose-relations
  "Compiles OpenXLA PJRT executable for transitive relational composition."
  ([dim]
   (compile-compose-relations (xla/get-context) dim))
  ([ctx dim]
   (let [d (long dim)
         invars [[:R1 [:tensor [d d] :f32]]
                 [:R2 [:tensor [d d] :f32]]]
         ast (compose-relations-ast d)]
     (sym/compile-query ctx "ephemeral_compose_relations" invars ast [:R_composed]))))

(defn compile-denoise-relation
  "Compiles OpenXLA PJRT executable for extract-threshold-re-embed denoising cycle."
  ([dim entity-count]
   (compile-denoise-relation (xla/get-context) dim entity-count nil))
  ([a b c]
   (if (map? a)
     (compile-denoise-relation a b c nil)
     (compile-denoise-relation (xla/get-context) a b c)))
  ([ctx dim entity-count opts]
   (let [d (long dim)
         n (long entity-count)
         invars [[:R [:tensor [d d] :f32]]
                 [:E [:tensor [n d] :f32]]]
         ast (denoise-relation-ast d n opts)]
     (sym/compile-query ctx "ephemeral_denoise_relation" invars ast [:R_clean :A_crisp]))))

;; ==============================================================================
;; 3. Impure Execution Wrappers (PJRT Dispatch)
;; ==============================================================================

(defn write-fact!
  "Executes compiled single fact write executable."
  [exec r eh et]
  (sym/run-query! exec {:R r :e_h eh :e_t et}))

(defn write-facts-batch!
  "Executes compiled batch fact write executable."
  [exec r h t]
  (sym/run-query! exec {:R r :H h :T t}))

(defn retract-fact!
  "Executes compiled single fact retraction executable."
  [exec r eh et]
  (sym/run-query! exec {:R r :e_h eh :e_t et}))

(defn retract-facts-batch!
  "Executes compiled batch fact retraction executable."
  [exec r h t]
  (sym/run-query! exec {:R r :H h :T t}))

(defn query-memory!
  "Executes compiled memory query executable."
  [exec eq r e]
  (sym/run-query! exec {:e_q eq :R r :E e}))

(defn compose-relations!
  "Executes compiled relational composition executable."
  [exec r1 r2]
  (sym/run-query! exec {:R1 r1 :R2 r2}))

(defn denoise-relation!
  "Executes compiled denoising cycle executable."
  [exec r e]
  (sym/run-query! exec {:R r :E e}))

;; ==============================================================================
;; 4. Ephemeral Relational Memory Session (Agent State Management)
;; ==============================================================================

(defn init-ephemeral-session
  "Initializes an in-VRAM ephemeral memory session for an autonomous agent.
   `dim`: embedding dimension D
   `entity-count`: entity dictionary size N
   `entity-table`: flat float-array of length N*D
   `relation-keys`: collection of relation keywords (e.g. [:parent :located_at :authored])"
  [dim entity-count entity-table relation-keys]
  (let [d (long dim)
        n (long entity-count)
        cores (into {}
                    (map (fn [k] [k (float-array (* d d))])
                         relation-keys))]
    (atom {:dim d
           :entity-count n
           :entity-table entity-table
           :cores cores
           :exec-cache {}})))

(defn session-write-fact!
  "Writes fact (head -> tail) into relation `rel-kw` in session."
  [session-atom rel-kw head-vec tail-vec & [opts]]
  (let [{:keys [dim cores]} @session-atom
        d (long dim)
        current-r (get cores rel-kw)
        _ (when-not current-r
            (throw (ex-info "Unknown relation in ephemeral session" {:relation rel-kw})))
        exec (compile-write-fact d opts)
        out (write-fact! exec current-r head-vec tail-vec)
        r-new (:R_updated out)]
    (swap! session-atom assoc-in [:cores rel-kw] r-new)
    r-new))

(defn session-retract-fact!
  "Retracts fact (head -> tail) from relation `rel-kw` in session."
  [session-atom rel-kw head-vec tail-vec]
  (let [{:keys [dim cores]} @session-atom
        d (long dim)
        current-r (get cores rel-kw)
        _ (when-not current-r
            (throw (ex-info "Unknown relation in ephemeral session" {:relation rel-kw})))
        exec (compile-retract-fact d)
        out (retract-fact! exec current-r head-vec tail-vec)
        r-new (:R_updated out)]
    (swap! session-atom assoc-in [:cores rel-kw] r-new)
    r-new))

(defn session-query
  "Queries relation `rel-kw` with `query-vec`."
  [session-atom rel-kw query-vec & [opts]]
  (let [{:keys [dim entity-count entity-table cores]} @session-atom
        d (long dim)
        n (long entity-count)
        current-r (get cores rel-kw)
        _ (when-not current-r
            (throw (ex-info "Unknown relation in ephemeral session" {:relation rel-kw})))
        exec (compile-query-memory d n opts)]
    (query-memory! exec query-vec current-r entity-table)))

(defn session-compose!
  "Composes two relations in-graph: target-rel = rel1 * rel2."
  [session-atom target-rel-kw rel1-kw rel2-kw]
  (let [{:keys [dim cores]} @session-atom
        d (long dim)
        r1 (get cores rel1-kw)
        r2 (get cores rel2-kw)
        _ (when-not (and r1 r2)
            (throw (ex-info "Unknown relations for composition" {:rel1 rel1-kw :rel2 rel2-kw})))
        exec (compile-compose-relations d)
        out (compose-relations! exec r1 r2)
        r-comp (:R_composed out)]
    (swap! session-atom assoc-in [:cores target-rel-kw] r-comp)
    r-comp))

(defn session-denoise!
  "Applies extract-threshold-re-embed denoising cycle to `rel-kw`."
  [session-atom rel-kw & [opts]]
  (let [{:keys [dim entity-count entity-table cores]} @session-atom
        d (long dim)
        n (long entity-count)
        current-r (get cores rel-kw)
        _ (when-not current-r
            (throw (ex-info "Unknown relation in ephemeral session" {:relation rel-kw})))
        exec (compile-denoise-relation d n opts)
        out (denoise-relation! exec current-r entity-table)
        r-clean (:R_clean out)]
    (swap! session-atom assoc-in [:cores rel-kw] r-clean)
    r-clean))
