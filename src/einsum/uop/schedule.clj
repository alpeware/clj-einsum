(ns einsum.uop.schedule
  "UOp Kernel Scheduler (`STAGE` / realization boundary cutter).
   Modeled on tinygrad's scheduler (`tinygrad/schedule/`) and geohot's ML compiler syllabus (Week 4).
   Cuts a global multi-layer UOp DAG at realization boundaries (`:uop/reduce` outputs, `:uop/index` gathers,
   and multi-consumer compute subgraphs) so that each kernel contains at most one reduction loop and zero
   exponential recomputation across layers."
  (:require [einsum.uop.core :as uop]
            [einsum.uop.rangeify :as rangeify]))

;; -----------------------------------------------------------------------------
;; 1. Consumer Analysis & Realization Boundary Selection
;; -----------------------------------------------------------------------------

(defn- build-consumer-counts
  "Returns a map `{uop-id -> consumer-count}` across all nodes reachable from `roots`."
  [roots]
  (let [nodes (uop/toposort roots)]
    (reduce
     (fn [counts node]
       (reduce
        (fn [acc child]
          (update acc (:id child) (fnil inc 0)))
        counts
        (:src node)))
     {}
     nodes)))

(defn- should-realize?
  "Determines whether `node` must be materialized into an intermediate `STAGE` buffer:
   1. Any `:uop/reduce` node (so reductions never nest or duplicate across consumers).
   2. Any `:uop/index` gather node (so embedding lookups materialize once).
   3. Any non-trivial ALU node with > 1 consumer that sits above a reduction or large fan-out,
      preventing exponential recomputation across residual branches (e.g., `res1` or `x_norm1` in Transformers).
   4. Any explicit output target root."
  [node consumer-counts output-ids]
  (let [id (:id node)
        op (:op node)
        n-consumers (get consumer-counts id 0)]
    (or (contains? output-ids id)
        (= op :uop/reduce)
        (and (= op :uop/index) (= :gather (:op (:arg node))))
        ;; Realize multi-consumer ALU nodes that are not trivial scalar broadcasts
        (and (contains? uop/ALU_OPS op)
             (> n-consumers 1)
             (seq (:shape node))
             (> (long (reduce * 1 (:shape node))) 1)))))

;; -----------------------------------------------------------------------------
;; 2. Graph Cutter & Kernel Schedule Builder
;; -----------------------------------------------------------------------------

(defn schedule-graph
  "Splits a high-level UOp graph (from `einsum.uop.lower/ast->uop-graph`) into a topological sequence
   of `ScheduledKernel` maps, and runs `rangeify/rangeify-kernel` on each kernel.
   Returns:
     `{:name ...
       :invars ...
       :invar-slots {var-kw -> slot-int}
       :output-slots {out-kw -> slot-int}
       :buffers {slot-int -> {:slot int :dtype kw :shape vec :size long :role (:input|:intermediate|:output)}}
       :kernels [{:kernel-id int :out-slot int :out-shape vec :out-dtype kw :root-uop uop :rangeified rk} ...]}`"
  [uop-graph]
  (let [{:keys [name invars outvars outputs]} uop-graph
        out-uops (mapv #(get outputs %) outvars)
        output-id->kw (into {} (map (fn [[k u]] [(:id u) k]) outputs))
        output-ids (set (keys output-id->kw))
        all-nodes (uop/toposort out-uops)
        consumer-counts (build-consumer-counts out-uops)

        ;; Input parameter slots: 0 .. (count invars - 1)
        invar-slots (into {} (map-indexed (fn [idx [k _]] [k idx]) invars))
        initial-buffers (into {}
                              (map-indexed
                               (fn [slot [k [_ sh dt]]]
                                 [slot {:slot slot
                                        :name k
                                        :dtype (or dt :f32)
                                        :shape (vec sh)
                                        :size (long (reduce * 1 sh))
                                        :role :input}])
                               invars))
        next-slot (atom (count invars))
        axis-counter (atom 0)
        vconst-base (atom 10000)

        ;; Walk nodes in topological order; whenever a node must be realized, emit a kernel for it
        ;; and replace the node in `realized-map` with a `:uop/param` pointing to its newly allocated buffer slot!
        {:keys [kernels buffers node->slot]}
        (reduce
         (fn [{:keys [realized-map kernels buffers node->slot] :as state} node]
           ;; First, rewrite `node`'s immediate children to substitute any already-realized children with their buffer params
           (let [rewritten-srcs (mapv #(get realized-map (:id %) %) (:src node))
                 substituted-node (if (= rewritten-srcs (:src node))
                                    node
                                    (uop/uop (:op node) (:dtype node) rewritten-srcs (:arg node) (:shape node)))
                 id (:id node)]
             ;; Special case: if an output target is already a `:uop/param` (or pure reshape of a realized buffer)
             ;; and not yet materialized in its own output slot, or if `should-realize?` holds:
             (if (and (should-realize? node consumer-counts output-ids)
                      (not (contains? uop/SOURCE_OPS (:op node))))
               (let [slot (swap! next-slot inc)
                     sh (vec (or (:shape substituted-node) []))
                     dt (:dtype substituted-node)
                     sz (long (reduce * 1 sh))
                     is-out? (contains? output-ids id)
                     out-kw (get output-id->kw id)
                     rk (rangeify/rangeify-kernel substituted-node
                                                  {:out-slot slot
                                                   :axis-counter axis-counter
                                                   :vconst-base-slot (swap! vconst-base + 100)})
                     kernel-idx (count kernels)
                     kernel-entry {:kernel-id kernel-idx
                                   :name (str name "_k" kernel-idx (when out-kw (str "_" (clojure.core/name out-kw))))
                                   :out-slot slot
                                   :out-var out-kw
                                   :out-shape sh
                                   :out-dtype dt
                                   :out-size sz
                                   :root-uop substituted-node
                                   :rangeified rk}
                     buf-param (uop/uop-param slot dt sh (or out-kw (keyword (str "stage_" slot))))
                     buf-info {:slot slot
                               :name (or out-kw (keyword (str "stage_" slot)))
                               :dtype dt
                               :shape sh
                               :size sz
                               :role (if is-out? :output :intermediate)}]
                 {:realized-map (assoc realized-map id buf-param)
                  :kernels (conj kernels kernel-entry)
                  :buffers (assoc buffers slot buf-info)
                  :node->slot (assoc node->slot id slot)})
               ;; Not realized here: keep substituted-node in `realized-map` so downstream ops inline it
               ;; unless it is an output target that is just a pure movement view (e.g. final `:reshape` on an already realized node)
               (if (contains? output-ids id)
                 (let [slot (swap! next-slot inc)
                       sh (vec (or (:shape substituted-node) []))
                       dt (:dtype substituted-node)
                       sz (long (reduce * 1 sh))
                       out-kw (get output-id->kw id)
                       rk (rangeify/rangeify-kernel substituted-node
                                                    {:out-slot slot
                                                     :axis-counter axis-counter
                                                     :vconst-base-slot (swap! vconst-base + 100)})
                       kernel-idx (count kernels)
                       kernel-entry {:kernel-id kernel-idx
                                     :name (str name "_k" kernel-idx "_" (clojure.core/name out-kw))
                                     :out-slot slot
                                     :out-var out-kw
                                     :out-shape sh
                                     :out-dtype dt
                                     :out-size sz
                                     :root-uop substituted-node
                                     :rangeified rk}
                       buf-param (uop/uop-param slot dt sh out-kw)
                       buf-info {:slot slot
                                 :name out-kw
                                 :dtype dt
                                 :shape sh
                                 :size sz
                                 :role :output}]
                   {:realized-map (assoc realized-map id buf-param)
                    :kernels (conj kernels kernel-entry)
                    :buffers (assoc buffers slot buf-info)
                    :node->slot (assoc node->slot id slot)})
                 (assoc state :realized-map (assoc realized-map id substituted-node))))))
         {:realized-map {}
          :kernels []
          :buffers initial-buffers
          :node->slot {}}
         all-nodes)

        output-slots (into {}
                           (map (fn [out-kw]
                                  (let [u (get outputs out-kw)]
                                    [out-kw (get node->slot (:id u))]))
                                outvars))]
    {:name name
     :invars invars
     :outvars outvars
     :invar-slots invar-slots
     :output-slots output-slots
     :buffers buffers
     :kernels kernels}))

;; -----------------------------------------------------------------------------
;; 3. Reference Scheduled Graph Evaluator
;; -----------------------------------------------------------------------------

(defn eval-scheduled-graph
  "Executes all `ScheduledKernel` steps in `scheduled-graph` sequentially using `rangeify/eval-rangeified-kernel`.
   `inputs-map`: map `{var-kw -> primitive-array}`."
  [scheduled-graph inputs-map]
  (let [{:keys [invar-slots output-slots kernels]} scheduled-graph
        initial-slot-bufs (into {}
                                (keep (fn [[k slot]]
                                        (when-let [arr (get inputs-map k)]
                                          [slot arr]))
                                      invar-slots))
        final-slot-bufs (reduce
                         (fn [slot-bufs {:keys [out-slot rangeified]}]
                           (let [out-arr (rangeify/eval-rangeified-kernel rangeified slot-bufs)]
                             (assoc slot-bufs out-slot out-arr)))
                         initial-slot-bufs
                         kernels)]
    (into {}
          (map (fn [[out-kw slot]]
                 [out-kw (get final-slot-bufs slot)])
               output-slots))))
