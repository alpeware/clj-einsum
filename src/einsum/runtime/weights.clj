(ns einsum.runtime.weights
  "Unified WeightStore, safetensors metadata integration, and automated invar inference."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.logic.dce :as dce]
            [einsum.logic.expand :as expand]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.safetensors :as st])
  (:import [java.lang.foreign Arena MemorySegment]))

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

(defrecord WeightStore [header device-buffers aliases arena ctx source])

(defn weight-store?
  "Returns true if `x` is an instance of WeightStore."
  [x]
  (instance? WeightStore x))

(defn create-weight-store
  "Creates a WeightStore from header metadata map, safetensors map, or file path."
  ([header-or-map]
   (create-weight-store nil header-or-map {}))
  ([header-or-map opts]
   (if (and (map? header-or-map) (contains? header-or-map :client))
     (create-weight-store header-or-map opts {})
     (create-weight-store nil header-or-map opts)))
  ([ctx header-or-source opts]
   (let [source (cond
                  (or (string? header-or-source) (instance? java.io.File header-or-source))
                  (st/map-safetensors-weights (str header-or-source) (Arena/ofAuto))

                  :else
                  header-or-source)
         raw-header (or (:header source) source)
         normalized-header (into {} (map (fn [[k v]]
                                           [(if (keyword? k) k (str k))
                                            (normalize-header-entry v)])
                                         raw-header))
         aliases (or (:aliases opts) {})
         arena (:arena opts)]
     (->WeightStore normalized-header (atom {}) aliases arena ctx source))))

(defn resolve-tensor-name
  "Resolves key `k` to its canonical safetensors tensor name using `aliases`."
  [store-or-aliases k]
  (let [aliases (if (weight-store? store-or-aliases)
                  (:aliases store-or-aliases)
                  store-or-aliases)
        k-str (if (keyword? k) (name k) (str k))]
    (or (cond
          (fn? aliases) (aliases k)
          (map? aliases) (or (get aliases k) (get aliases k-str))
          :else nil)
        k-str)))

(defn tensor-info
  "Resolves shape and dtype map `{:shape [..] :dtype :bf16}` for tensor key `k` in `store-or-map`."
  [store-or-map k]
  (let [header (if (weight-store? store-or-map)
                 (:header store-or-map)
                 (or (:header store-or-map) store-or-map))
        aliases (when (weight-store? store-or-map) (:aliases store-or-map))
        k-str (if (keyword? k) (name k) (str k))
        k-dot-weight (str k-str ".weight")
        alias-target (cond
                       (fn? aliases) (aliases k)
                       (map? aliases) (or (get aliases k) (get aliases k-str))
                       :else nil)]
    (or (when alias-target
          (or (when-let [v (get header alias-target)] (normalize-header-entry v))
              (when (keyword? alias-target)
                (when-let [v (get header (name alias-target))] (normalize-header-entry v)))
              (when (string? alias-target)
                (when-let [v (get header (keyword alias-target))] (normalize-header-entry v)))))
        (when-let [v (get header k)]
          (normalize-header-entry v))
        (when-let [v (get header k-str)]
          (normalize-header-entry v))
        (when-let [v (get header k-dot-weight)]
          (normalize-header-entry v))
        (when (map? store-or-map)
          (when-let [v (get store-or-map k)]
            (normalize-header-entry v))))))

(defn get-device-buffer
  "Retrieves existing buffer or lazily transfers tensor key `k` into `store`'s arena.
   Caller-supplied MemorySegments in source maps are returned as-is without arena tracking.
   Caches the device buffer in `@(:device-buffers store)`."
  [store k]
  (when (weight-store? store)
    (if-let [existing (get @(:device-buffers store) k)]
      existing
      (let [canonical-name (resolve-tensor-name store k)
            info (tensor-info store k)]
        (when-not info
          (throw (ex-info (str "Tensor info not found for weight key: " k)
                          {:key k :canonical-name canonical-name})))
        (let [raw-src (:source store)
              source (if (and (map? raw-src) (contains? raw-src :source))
                       (or (:source raw-src) raw-src)
                       raw-src)
              arena (or (:arena store) arena/*active-arena*)
              ctx (or (:ctx store)
                      (when-let [f (resolve 'einsum.core/get-context)] (f))
                      {})
              cli (or (:client ctx) (:client arena) ctx)
              shape (:shape info)
              dtype (:dtype info)
              buf (cond
                    ;; 1. MemorySegment already in source map (caller-owned buffer or host segment, never auto-tracked)
                    (and (map? source) (contains? source k) (instance? MemorySegment (get source k)))
                    (get source k)

                    ;; 2. Safetensors mapped weights
                    (and (map? source) (or (:segment source) (:tensors source)))
                    (let [c-str (if (keyword? canonical-name) (name canonical-name) (str canonical-name))
                          slice (st/get-tensor-slice source c-str)]
                      (if arena
                        (arena/device-buffer arena slice shape dtype)
                        (let [dt-enum (arena/dtype->enum dtype)]
                          (pjrt/buffer-from-host-buffer ctx cli slice shape dt-enum))))

                    ;; 3. Host array in source map
                    (and (map? source) (or (contains? source k) (contains? source canonical-name)))
                    (let [arr (or (get source k) (get source canonical-name))]
                      (if arena
                        (arena/device-buffer arena arr shape dtype)
                        (let [dt-enum (arena/dtype->enum dtype)]
                          (pjrt/buffer-from-host-buffer ctx cli arr shape dt-enum))))

                    :else
                    (throw (ex-info (str "Unable to retrieve device buffer for tensor: " k)
                                    {:key k :canonical-name canonical-name})))]
          (swap! (:device-buffers store) assoc k buf)
          buf)))))

(defn find-ast-free-vars
  "Extracts all free variable names from `ast` that are referenced in body terms
   but never bound as an equation head."
  [ast]
  (let [eqns (expand/expand-ast {} ast)
        head-names (set (mapcat dce/eqn-head-names eqns))
        body-names (set (mapcat dce/eqn-body-names eqns))]
    (set/difference body-names head-names)))

(defn load-device-buffers!
  "Pre-transfers all weights in `store` (or for variables in `ast-or-keys`) into device memory.
   Throws ExceptionInfo if any weight fails to load. Returns `store`."
  ([store]
   (load-device-buffers! store (keys (:header store))))
  ([store ast-or-keys]
   (let [ks (cond
              (nil? ast-or-keys) (keys (:header store))
              (and (vector? ast-or-keys) (seq ast-or-keys) (not (keyword? (first ast-or-keys))))
              (find-ast-free-vars ast-or-keys)
              (coll? ast-or-keys) ast-or-keys
              :else (keys (:header store)))
         failures (reduce (fn [acc k]
                            (try
                              (get-device-buffer store k)
                              acc
                              (catch Exception e
                                (conj acc {:key k :error (.getMessage e)}))))
                          []
                          ks)]
     (when (seq failures)
       (throw (ex-info (str "Failed to load device buffers for " (count failures) " weights: "
                            (vec (map :key failures)))
                       {:failed-count (count failures)
                        :failed-keys (vec (map :key failures))
                        :failures failures})))
     store)))

(defn infer-invars
  "Derives the full input variable signature `[[:var-name [:tensor shape dtype]] ...]`
   for `ast` by combining explicit `dynamic-inputs` with weight shapes resolved from `weights-store-or-map`.
   When strict dynamic input ordering is required, pass `dynamic-inputs` as a vector of pairs
   `[[:var spec] ...]` (standard Clojure maps with > 8 entries do not preserve key insertion order)."
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
