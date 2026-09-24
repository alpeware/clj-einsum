(ns einsum.logic.ast
  "Hiccup AST Malli schemas, node constructors, and syntax helpers for Tensor Logic."
  (:require [malli.core :as m]))

(def TermIdSchema
  "Schema for tensor identifier: keyword or composite tuple [:name & args]."
  [:or :keyword [:and [:vector :any] [:fn seq]]])

(def HeadSchema
  "Schema for equation head [id & indices]."
  [:and [:vector [:or :keyword TermIdSchema]] [:fn seq]])

(def BodyTermSchema
  "Schema for equation body term [id & indices]."
  [:and [:vector [:or :keyword TermIdSchema]] [:fn seq]])

(def EquationSchema
  "Schema for primitive contraction equation [:= head ?attrs & body-terms]."
  [:cat
   [:fn (fn [op] (or (= op ':=) (= op :=)))]
   HeadSchema
   [:* [:or BodyTermSchema [:map-of :keyword :any]]]])

(def HookNodeSchema
  "Schema for lowering hook nodes [:hook-op head & body-terms-and-attrs]."
  [:cat
   :keyword
   HeadSchema
   [:* [:or BodyTermSchema [:map-of :keyword :any]]]])

(def WhileNodeSchema
  "Schema for while loop nodes [:while out-specs in-specs ?attrs & children]."
  [:cat
   [:enum :while]
   [:vector :any]
   [:vector :any]
   [:? [:map-of :keyword :any]]
   [:* [:or [:vector :any] :keyword number? string?]]])

(def CondNodeSchema
  "Schema for condition nodes [:cond head ?attrs & children]."
  [:cat
   [:enum :cond]
   HeadSchema
   [:? [:map-of :keyword :any]]
   [:* [:or [:vector :any] :keyword number? string?]]])

(def BodyNodeSchema
  "Schema for while-loop body nodes [:body head ?attrs & children]."
  [:cat
   [:enum :body]
   [:vector :any]
   [:? [:map-of :keyword :any]]
   [:* [:or [:vector :any] :keyword number? string?]]])

(def ContainerNodeSchema
  "Schema for container nodes [:tag ?attrs & children]."
  [:cat [:or :keyword :symbol]
   [:? [:map-of :keyword :any]]
   [:* [:or [:vector :any] :keyword number? string?]]])

(def NodeSchema
  "General Malli schema for Tensor Logic Hiccup nodes (equations, lowering hooks, while loops, cond nodes, body nodes, and containers)."
  [:or EquationSchema HookNodeSchema WhileNodeSchema CondNodeSchema BodyNodeSchema ContainerNodeSchema])

(defn valid-node?
  "Validates whether node matches the general Hiccup NodeSchema."
  [node]
  (boolean (and (vector? node) (m/validate NodeSchema node))))

(defn eqn?
  "Checks if node is a Tensor Logic primitive equation [:= ...]."
  [node]
  (and (vector? node)
       (let [op (first node)]
         (or (= op :=) (= op ':=)))
       (>= (count node) 2)
       (vector? (second node))))

(defn head
  "Returns head term [:name & indices] or output terms from an equation, hook, or while node."
  [node]
  (when (and (vector? node) (>= (count node) 2) (vector? (second node)))
    (second node)))

(defn attrs
  "Returns attributes map from equation, hook, or container node, or nil."
  [node]
  (when (vector? node)
    (some #(when (map? %) %) (rest node))))

(defn body-terms
  "Returns vector of body terms from an equation, hook, or while node."
  [node]
  (when (and (vector? node) (>= (count node) 2) (vector? (second node)))
    (if (= (first node) :while)
      (let [in-spec (nth node 2)]
        (if (vector? in-spec)
          (vec (mapv #(if (vector? %) % [%]) in-spec))
          [[in-spec]]))
      (let [tail (drop 2 node)]
        (vec (filter vector? (if (map? (first tail))
                               (rest tail)
                               tail)))))))

(defn eqn
  "Constructs a primitive equation [:= head ?attrs & body-terms]."
  ([head & args]
   (let [[attrs body] (if (map? (first args))
                        [(first args) (rest args)]
                        [nil args])]
     (cond-> [:= head]
       attrs (conj attrs)
       (seq body) (into body)))))

(defn block
  "Constructs a sequential container block [:block ?attrs & children]."
  ([attrs & children]
   (into [:block attrs] children))
  ([children]
   (into [:block {}] children)))
