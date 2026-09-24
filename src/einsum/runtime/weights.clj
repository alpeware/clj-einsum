(ns einsum.runtime.weights
  "Unified WeightStore, safetensors metadata integration, and automated invar inference."
  (:require [einsum.logic.dce :as dce]
            [einsum.logic.expand :as expand]
            [clojure.set :as set]
            [clojure.string :as str]))

(defn normalize-dtype
  "Normalizes string or keyword `dt` to canonical dtype keyword (:bf16, :f32, :i8, :i32, :f16)."
  [dt]
  (cond
    (nil? dt) :f32
    (keyword? dt) (keyword (str/lower-case (name dt)))
    (string? dt) (keyword (str/lower-case dt))
    :else :f32))

(defn- normalize-header-entry
  [entry]
  (cond
    (map? entry)
    {:shape (vec (or (:shape entry) (get entry "shape") []))
     :dtype (normalize-dtype (or (:dtype entry) (get entry "dtype")))}

    (vector? entry)
    {:shape (vec (first entry))
     :dtype (normalize-dtype (second entry))}

    :else
    {:shape [] :dtype :f32}))

(defrecord WeightStore [header device-buffers aliases arena ctx])

(defn weight-store?
  "Returns true if `x` is an instance of WeightStore."
  [x]
  (instance? WeightStore x))

(defn create-weight-store
  "Creates a WeightStore from header metadata map, safetensors map, or file path."
  ([header-or-map]
   (create-weight-store nil header-or-map {}))
  ([header-or-map opts]
   (create-weight-store nil header-or-map opts))
  ([ctx header-or-map opts]
   (let [raw-header (or (:header header-or-map) header-or-map)
         normalized-header (into {} (map (fn [[k v]]
                                           [(if (keyword? k) k (str k))
                                            (normalize-header-entry v)])
                                         raw-header))
         aliases (or (:aliases opts) {})
         arena (:arena opts)]
     (->WeightStore normalized-header (atom {}) aliases arena ctx))))

(defn tensor-info
  "Resolves shape and dtype map `{:shape [..] :dtype :bf16}` for tensor key `k` in `store-or-map`."
  [store-or-map k]
  (let [header (if (weight-store? store-or-map)
                 (:header store-or-map)
                 (or (:header store-or-map) store-or-map))
        aliases (when (weight-store? store-or-map) (:aliases store-or-map))
        k-str (if (keyword? k) (name k) (str k))
        k-dot-weight (str k-str ".weight")
        alias-target (when aliases (get aliases k))]
    (or (when alias-target
          (normalize-header-entry (get header alias-target)))
        (when-let [v (get header k)]
          (normalize-header-entry v))
        (when-let [v (get header k-str)]
          (normalize-header-entry v))
        (when-let [v (get header k-dot-weight)]
          (normalize-header-entry v))
        (when (map? store-or-map)
          (when-let [v (get store-or-map k)]
            (normalize-header-entry v))))))

(defn find-ast-free-vars
  "Extracts all free variable names from `ast` that are referenced in body terms
   but never bound as an equation head."
  [ast]
  (let [eqns (expand/expand-ast {} ast)
        head-names (set (mapcat dce/eqn-head-names eqns))
        body-names (set (mapcat dce/eqn-body-names eqns))]
    (set/difference body-names head-names)))

(defn infer-invars
  "Derives the full input variable signature `[[:var-name [:tensor shape dtype]] ...]`
   for `ast` by combining explicit `dynamic-inputs` with weight shapes resolved from `weights-store-or-map`."
  [ast dynamic-inputs weights-store-or-map]
  (let [;; 1. Parse dynamic inputs into map of {var-kw -> [:tensor shape dtype]}
        dyn-map (if (map? dynamic-inputs)
                  dynamic-inputs
                  (into {} (map (fn [[v spec]] [v spec]) dynamic-inputs)))
        dyn-order (if (vector? dynamic-inputs)
                    (mapv first dynamic-inputs)
                    (vec (keys dyn-map)))

        ;; 2. Find free variables in AST
        free-vars (find-ast-free-vars ast)

        ;; 3. Dynamic invars preserve user order
        dyn-invars (mapv (fn [v] [v (get dyn-map v)]) dyn-order)
        dyn-names (set dyn-order)

        ;; 4. Weight invars for remaining free variables
        weight-vars (vec (sort-by name (set/difference free-vars dyn-names)))
        weight-invars (mapv (fn [v]
                              (let [info (tensor-info weights-store-or-map v)]
                                (when-not info
                                  (throw (ex-info (str "Unable to infer shape and dtype for free variable: " v)
                                                  {:var v :free-vars free-vars})))
                                [v [:tensor (:shape info) (:dtype info)]]))
                            weight-vars)]
    (vec (concat dyn-invars weight-invars))))
