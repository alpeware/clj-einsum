(ns einsum.graph.schema
  "Malli schemas and validation helpers for the Agent, Device, and Capability Graph Architecture."
  (:require [malli.core :as m]))

;; -----------------------------------------------------------------------------
;; 1. Core Pointer & Envelope Schemas
;; -----------------------------------------------------------------------------

(def CasPtrSchema
  "Schema for a Content-Addressed Pointer referencing off-heap or persistent memory."
  [:map
   [:cas/cid :string]
   [:cas/kind :keyword]
   [:cas/uri :string]
   [:cas/shape {:optional true} [:vector :int]]
   [:cas/bytes {:optional true} :int]])

(def EventEnvelopeSchema
  "Schema for a deterministic, hash-chained event envelope in the graph."
  [:map
   [:event/id :uuid]
   [:event/tx :int]
   [:event/prev-hash :string]
   [:event/hash :string]
   [:event/source [:tuple :keyword :keyword]]
   [:event/target {:optional true} [:tuple :keyword :keyword]]
   [:event/payload :any]
   [:event/meta {:optional true} :map]])

;; -----------------------------------------------------------------------------
;; 2. Validation Predicates
;; -----------------------------------------------------------------------------

(defn cas-ptr?
  "Returns true if x conforms to a CAS pointer structure."
  [x]
  (and (map? x)
       (string? (:cas/cid x))
       (keyword? (:cas/kind x))))

(defn valid-envelope?
  "Validates whether x conforms to EventEnvelopeSchema."
  [x]
  (m/validate EventEnvelopeSchema x))

;; -----------------------------------------------------------------------------
;; 3. Hiccup Grammar Parsing & Validation
;; -----------------------------------------------------------------------------

(defn parse-hiccup-node
  "Parses a Hiccup vector [tag attrs? & children] into a normalized map:
   {:tag tag :attrs map :children sequence}."
  [node]
  (when (vector? node)
    (let [tag (first node)
          second-elem (second node)
          has-attrs? (map? second-elem)
          attrs (if has-attrs? second-elem {})
          children (if has-attrs? (subvec node 2) (subvec node 1))]
      {:tag tag
       :attrs attrs
       :children children})))

(def MealyAttrsSchema
  [:map
   [:id :keyword]
   [:ins [:vector :keyword]]
   [:outs [:vector :keyword]]
   [:init-state :any]
   [:step ifn?]])

(def VerifyAttrsSchema
  [:map
   [:id :keyword]
   [:ins [:vector :keyword]]
   [:outs [:vector :keyword]]
   [:pred ifn?]
   [:init-state {:optional true} :any]])

(def CapAttrsSchema
  [:map
   [:id :keyword]
   [:type :keyword]])

(def DeviceAttrsSchema
  [:map
   [:id :keyword]])

(def AgentAttrsSchema
  [:map
   [:id :keyword]
   [:ins [:vector :keyword]]
   [:outs [:vector :keyword]]
   [:requires {:optional true} :map]])

(defn validate-agent-graph!
  "Recursively validates an [:agent attrs & children] Hiccup graph AST.
   Throws ex-info with descriptive context on schema violation, returns agent-hiccup on success."
  [agent-hiccup]
  (when-not (vector? agent-hiccup)
    (throw (ex-info "Agent graph must be a vector" {:hiccup agent-hiccup})))
  (let [{:keys [tag attrs children]} (parse-hiccup-node agent-hiccup)]
    (when-not (= tag :agent)
      (throw (ex-info (str "Expected :agent root tag, got " (pr-str tag)) {:tag tag :hiccup agent-hiccup})))
    (when-not (m/validate AgentAttrsSchema attrs)
      (throw (ex-info "Agent attributes do not match AgentAttrsSchema"
                      {:attrs attrs :errors (m/explain AgentAttrsSchema attrs)})))
    (let [agent-id (:id attrs)]
      (doseq [child children]
        (when-not (vector? child)
          (throw (ex-info "Child node in agent graph must be a vector" {:agent agent-id :child child})))
        (let [child-tag (first child)]
          (case child-tag
            :mealy
            (let [child-attrs (second child)]
              (when-not (m/validate MealyAttrsSchema child-attrs)
                (throw (ex-info "Mealy node attributes do not match MealyAttrsSchema"
                                {:child child :errors (m/explain MealyAttrsSchema child-attrs)}))))

            :verify
            (let [child-attrs (second child)]
              (when-not (m/validate VerifyAttrsSchema child-attrs)
                (throw (ex-info "Verify node attributes do not match VerifyAttrsSchema"
                                {:child child :errors (m/explain VerifyAttrsSchema child-attrs)}))))

            :conn
            (let [src (nth child 1 nil)
                  dst (nth child 2 nil)]
              (when-not (and (vector? src) (== (count src) 2) (keyword? (first src)) (keyword? (second src)))
                (throw (ex-info "Connection source must be a 2-element keyword tuple [node port]"
                                {:conn child :src src})))
              (when-not (and (vector? dst) (== (count dst) 2) (keyword? (first dst)) (keyword? (second dst)))
                (throw (ex-info "Connection destination must be a 2-element keyword tuple [node port]"
                                {:conn child :dst dst}))))

            :agent
            (validate-agent-graph! child)

            (throw (ex-info (str "Unsupported child tag in agent graph: " (pr-str child-tag))
                            {:agent agent-id :child child}))))))
    agent-hiccup))

(defn validate-device-graph!
  "Validates a [:device attrs & cap-children] Hiccup graph AST.
   Throws ex-info on schema violation, returns device-hiccup on success."
  [device-hiccup]
  (when-not (vector? device-hiccup)
    (throw (ex-info "Device graph must be a vector" {:hiccup device-hiccup})))
  (let [{:keys [tag attrs children]} (parse-hiccup-node device-hiccup)]
    (when-not (= tag :device)
      (throw (ex-info (str "Expected :device root tag, got " (pr-str tag)) {:tag tag :hiccup device-hiccup})))
    (when-not (m/validate DeviceAttrsSchema attrs)
      (throw (ex-info "Device attributes do not match DeviceAttrsSchema"
                      {:attrs attrs :errors (m/explain DeviceAttrsSchema attrs)})))
    (doseq [child children]
      (when-not (vector? child)
        (throw (ex-info "Child node in device graph must be a vector" {:device (:id attrs) :child child})))
      (let [child-tag (first child)
            child-attrs (second child)]
        (when-not (= child-tag :cap)
          (throw (ex-info (str "Child in device graph must be :cap, got " (pr-str child-tag))
                          {:device (:id attrs) :child child})))
        (when-not (m/validate CapAttrsSchema child-attrs)
          (throw (ex-info "Capability attributes do not match CapAttrsSchema"
                          {:child child :errors (m/explain CapAttrsSchema child-attrs)})))))
    device-hiccup))
