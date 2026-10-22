(ns einsum.graph.binder
  "Single-Device Capability Binder: compiles Hiccup device specifications,
   manages capability lifecycles, and binds agent capability requirements to device capabilities."
  (:require [einsum.graph.capability :as cap]
            [einsum.graph.cas :as cas]
            [einsum.graph.runtime :as runtime]
            [einsum.graph.schema :as schema]))

;; =============================================================================
;; 1. Device Compilation & Lifecycle
;; =============================================================================

(defn compile-device
  "Validates and compiles a [:device attrs & caps] Hiccup AST into a running DeviceInstance.
   Instantiates and starts all child capabilities, sharing a CAS store."
  ([device-hiccup]
   (compile-device device-hiccup {}))
  ([device-hiccup opts]
   (schema/validate-device-graph! device-hiccup)
   (let [{:keys [attrs children]} (schema/parse-hiccup-node device-hiccup)
         device-id (:id attrs)
         cas-store (or (:cas-store opts) (cas/create-cas-store))
         device-ctx {:cas-store cas-store :device-id device-id}
         caps-map (reduce
                   (fn [acc child]
                     (let [[_tag cap-attrs] child
                           cap-id (:id cap-attrs)
                           cap-inst (cap/start-cap! (cap/build-capability cap-attrs device-ctx))]
                       (assoc acc cap-id {:id cap-id
                                          :attrs cap-attrs
                                          :instance cap-inst})))
                   {}
                   children)]
     {:id device-id
      :attrs attrs
      :cas-store cas-store
      :caps caps-map})))

(defn stop-device!
  "Tears down all running capability instances on `device-inst` and returns nil."
  [device-inst]
  (doseq [[_ cap-entry] (:caps device-inst)]
    (when-let [inst (:instance cap-entry)]
      (cap/stop-cap! inst)))
  nil)

;; =============================================================================
;; 2. Capability Matching
;; =============================================================================

(defn match-capability?
  "Returns true if `cap-attrs` satisfies `req-spec`.
   - If `req-spec` is :any, returns true.
   - If `req-spec` is a keyword, returns true if (:type cap-attrs) matches (or if req-spec is :any).
   - If `req-spec` is a map, checks type match and ensures any matching attributes align."
  [req-spec cap-attrs]
  (cond
    (= req-spec :any)
    true

    (keyword? req-spec)
    (or (= req-spec :any)
        (= req-spec (:type cap-attrs)))

    (map? req-spec)
    (let [type-req (:type req-spec)
          type-matches? (or (nil? type-req)
                            (= type-req :any)
                            (= type-req (:type cap-attrs)))]
      (if-not type-matches?
        false
        (let [filtered-spec (dissoc req-spec :type :optional? :ast :in :out :generate-fn :eval-fn :session)]
          (every? (fn [[k v]]
                    (if (contains? cap-attrs k)
                      (= (get cap-attrs k) v)
                      true))
                  filtered-spec))))

    :else
    false))

;; =============================================================================
;; 3. Agent-Device Binding
;; =============================================================================

(defn- collect-all-requires
  "Recursively collects all :requires specifications from root agent and any nested subagents."
  [agent-hiccup]
  (let [{:keys [attrs children]} (schema/parse-hiccup-node agent-hiccup)
        root-reqs (or (:requires attrs) {})
        sub-reqs (reduce (fn [acc child]
                           (if (and (vector? child) (= (first child) :agent))
                             (merge acc (collect-all-requires child))
                             acc))
                         {}
                         children)]
    (merge sub-reqs root-reqs)))

(defn bind-agent
  "Validates an agent graph, satisfies capability requirements against a device instance
   (or compiles a device Hiccup AST), and returns an executable GraphInstance with
   bound capabilities and shared CAS store."
  ([agent-hiccup device-or-hiccup]
   (bind-agent agent-hiccup device-or-hiccup {}))
  ([agent-hiccup device-or-hiccup opts]
   (schema/validate-agent-graph! agent-hiccup)
   (let [device-inst (if (vector? device-or-hiccup)
                       (compile-device device-or-hiccup opts)
                       device-or-hiccup)
         agent-attrs (:attrs (schema/parse-hiccup-node agent-hiccup))
         all-reqs (collect-all-requires agent-hiccup)
         bound-caps (reduce
                     (fn [acc [req-key req-spec]]
                       (let [norm-key (keyword (name req-key))
                             override-cap (or (get-in opts [:capabilities req-key])
                                              (get-in opts [:capabilities norm-key]))]
                         (if override-cap
                           (-> acc
                               (assoc norm-key override-cap)
                               (assoc (keyword "cap" (name req-key)) override-cap))
                           (let [cap-entries (vals (:caps device-inst))
                                 matching-cap (some (fn [entry]
                                                      (when (match-capability? req-spec (:attrs entry))
                                                        entry))
                                                    cap-entries)]
                             (cond
                               matching-cap
                               (let [has-overrides? (and (map? req-spec)
                                                         (some #(contains? req-spec %)
                                                               [:ast :in :out :generate-fn :eval-fn :session :sandbox :timeout-ms :backend]))
                                     cap-inst (if has-overrides?
                                                (let [specialized-attrs (merge (:attrs matching-cap) (dissoc req-spec :optional?))
                                                      device-ctx {:cas-store (:cas-store device-inst) :device-id (:id device-inst)}]
                                                  (cap/start-cap! (cap/build-capability specialized-attrs device-ctx)))
                                                (:instance matching-cap))]
                                 (-> acc
                                     (assoc norm-key cap-inst)
                                     (assoc (keyword "cap" (name req-key)) cap-inst)))

                               (and (map? req-spec) (:optional? req-spec))
                               acc

                               :else
                               (throw (ex-info "Unsatisfied capability requirement"
                                               {:agent (:id agent-attrs)
                                                :requirement req-key
                                                :spec req-spec
                                                :available-caps (vec (keys (:caps device-inst)))})))))))
                     {}
                     all-reqs)
         graph-opts (assoc (or opts {})
                           :capabilities bound-caps
                           :cas-store (:cas-store device-inst))
         graph-inst (runtime/compile-graph agent-hiccup graph-opts)]
     (assoc graph-inst :device device-inst))))
