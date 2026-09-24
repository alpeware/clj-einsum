(ns einsum.logic.expand
  "Recursive subgraph expansion and N-ary contraction decomposition for Tensor Logic AST."
  (:require [einsum.logic.ast :as ast]
            [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]))

(def ^:private GLOBAL-DIMS
  #{:b :p :d :v :max_pos :one :l :total_pl_dim})

(def ^:private GLOBAL-INPUTS
  #{:x :pos_ids :pos :embed_tokens :wte :wpe :normed :logits :total_pl_dim :one
    :W_mem :R_mem :E_cand :T :R_adj :threshold})

(defn normalize-id
  "Normalizes a composite tuple identifier [:name & args] or keyword to a canonical keyword."
  [x]
  (if (and (vector? x) (seq x) (keyword? (first x)))
    (let [[prefix & args] x
          p-str (name prefix)]
      (if (and (= p-str "h") (number? (first args)))
        (keyword (str "h" (first args)))
        (keyword (str (str/replace p-str "-" "_") "_" (str/join "_" args)))))
    x))

(defn- extract-layer-block-info [attrs]
  (when (map? attrs)
    (let [name-val (:name attrs)
          layer-val (:layer attrs)]
      (cond
        (and (vector? name-val) (>= (count name-val) 2) (number? (second name-val)))
        {:prefix (str/replace (name (first name-val)) "-" "_")
         :layer (second name-val)}

        (number? layer-val)
        {:prefix (if (keyword? name-val) (str/replace (name name-val) "-" "_") "layer")
         :layer layer-val}

        :else nil))))

(defn- collect-defined-heads [stmts]
  (set (mapcat (fn [stmt]
                 (cond
                   (and (vector? stmt) (= (first stmt) :block))
                   (collect-defined-heads (if (map? (second stmt)) (drop 2 stmt) (rest stmt)))

                   (and (vector? stmt) (>= (count stmt) 2) (vector? (second stmt)))
                   [(first (second stmt))]

                   :else []))
               stmts)))

(defn- expand-indexed-layer-block [block-info attrs children]
  (let [{:keys [prefix layer]} block-info
        b-prefix (str prefix "_" layer)
        h-in-kw (keyword (str "h" layer))
        h-out-kw (keyword (str "h" (inc layer)))
        defined-heads (collect-defined-heads children)
        internal-wires (set (filter keyword? (disj defined-heads :h# :h-out :h h-in-kw h-out-kw)))
        global-inputs (into GLOBAL-INPUTS (or (:globals attrs) (:inputs attrs) (:shared attrs)))
        remap-id (fn [id]
                   (cond
                     (or (= id :h) (= id :h-in) (= id h-in-kw))
                     h-in-kw

                     (or (= id :h#) (= id :h-out) (= id h-out-kw))
                     h-out-kw

                     (contains? internal-wires id)
                     (keyword (str b-prefix "_" (str/replace (name id) "-" "_")))

                     (contains? global-inputs id)
                     id

                     (and (keyword? id) (not (contains? defined-heads id)))
                     (if (or (re-find #"_\d+$" (name id))
                             (str/starts-with? (name id) prefix))
                       id
                       (keyword (str (str/replace (name id) "-" "_") "_" layer)))

                     :else
                     (normalize-id id)))
        remap-idx (fn [idx]
                    (if (or (not (keyword? idx))
                            (contains? GLOBAL-DIMS idx)
                            (str/ends-with? (name idx) (str "_" layer))
                            (re-find #"_\d+$" (name idx)))
                      idx
                      (keyword (str (str/replace (name idx) "-" "_") "_" layer))))
        remap-term (fn [term]
                     (if (and (vector? term) (seq term))
                       (let [[id & idxs] term]
                         (into [(remap-id id)] (map remap-idx idxs)))
                       term))
        remap-stmt (fn remap-fn [stmt]
                     (cond
                       (and (vector? stmt) (= (first stmt) :block))
                       (let [b-attrs (when (map? (second stmt)) (second stmt))
                             b-children (if b-attrs (drop 2 stmt) (rest stmt))]
                         (into [:block b-attrs] (mapv remap-fn b-children)))

                       (and (vector? stmt) (>= (count stmt) 2))
                       (let [[op head & tail] stmt
                             new-head (remap-term head)
                             new-tail (mapv (fn [elem]
                                              (cond
                                                (vector? elem) (remap-term elem)
                                                (map? elem) (into {} (map (fn [[k v]]
                                                                            [k (if (vector? v)
                                                                                 (mapv normalize-id v)
                                                                                 v)])) elem)
                                                :else elem))
                                            tail)]
                         (into [op new-head] new-tail))
                       :else stmt))]
    (mapv remap-stmt children)))

(defmulti expand-node
  "Expands a high-level container or composite layer node into primitive equations."
  (fn [node _ctx]
    (when (vector? node) (first node))))

(defmethod expand-node :default [_ _] nil)

(defn- decompose-n-ary-eqn
  "Decomposes an equation with N > 2 body terms into a sequence of binary contractions."
  [eqn-node ctx]
  (let [head (ast/head eqn-node)
        head-name (first head)
        head-idxs (vec (rest head))
        attrs (ast/attrs eqn-node)
        body (ast/body-terms eqn-node)]
    (if (<= (count body) 2)
      nil ;; Already binary or unary, no decomposition needed
      (let [counter (or (:synth-counter ctx) (atom 0))]
        (loop [acc []
               prev-term (first body)
               remaining (vec (rest body))]
          (if (empty? remaining)
            acc
            (let [next-term (first remaining)
                  rest-remaining (subvec remaining 1)
                  is-last? (empty? rest-remaining)
                  rem-indices (set (mapcat rest rest-remaining))
                  needed-later (set/union rem-indices (set head-idxs))
                  synth-idxs (vec (filter #(contains? needed-later %)
                                          (distinct (concat (rest prev-term) (rest next-term)))))
                  synth-head (if is-last?
                               (into [head-name] head-idxs)
                               (into [(keyword (str "synth_" (swap! counter inc)))] synth-idxs))
                  eqn (if (and is-last? attrs)
                        [:= synth-head attrs prev-term next-term]
                        [:= synth-head prev-term next-term])]
              (recur (conj acc eqn) synth-head rest-remaining))))))))

(defmethod expand-node := [node ctx]
  (decompose-n-ary-eqn node ctx))

(defmethod expand-node :block [node _ctx]
  (let [attrs (when (map? (second node)) (second node))
        children (if attrs (drop 2 node) (rest node))]
    (if-let [block-info (extract-layer-block-info attrs)]
      (expand-indexed-layer-block block-info attrs children)
      (vec children))))

(defmethod expand-node :relational-grounding [node _ctx]
  (let [children (if (map? (second node))
                   (drop 2 node)
                   (rest node))]
    (vec children)))

(defmethod expand-node :residual [node _ctx]
  ;; [:residual [:out ...] [:in ...] & children]
  ;; Automatically handles residual addition by emitting implicit accumulation or sum
  (let [elems (rest node)
        [head in-head & children] (if (map? (first elems)) (rest elems) elems)
        head-idxs (vec (rest head))
        in-head-idxs (vec (rest in-head))
        skip-eqn [:= (into [(first head)] head-idxs) (into [(first in-head)] in-head-idxs)]]
    (into [skip-eqn] children)))

(defn- statement? [node]
  (and (vector? node)
       (>= (count node) 2)
       (let [op (first node)]
         (or (keyword? op) (symbol? op)))
       (vector? (second node))))

(defn- flatten-statements [tree]
  (cond
    (nil? tree) []
    (statement? tree) [tree]
    (sequential? tree) (vec (mapcat flatten-statements tree))
    :else []))

(defn- normalize-remaining-terms [stmts]
  (mapv (fn [stmt]
          (if (and (vector? stmt) (>= (count stmt) 2))
            (mapv (fn [elem]
                    (if (and (vector? elem) (seq elem) (vector? (first elem)))
                      (into [(normalize-id (first elem))] (rest elem))
                      elem))
                  stmt)
            stmt))
        stmts))

(defn expand-ast
  "Recursively expands Hiccup AST using fixed-point prewalk until only primitive
   equations ([:= ...], direct StableHLO hooks) remain."
  ([ast] (expand-ast {} ast))
  ([ctx ast]
   (let [synth-counter (atom 0)
         full-ctx (assoc ctx :synth-counter synth-counter)
         step-fn (fn [tree]
                   (walk/prewalk
                    (fn [node]
                      (if-let [expanded (expand-node node full-ctx)]
                        expanded
                        node))
                    tree))]
     (loop [current ast
            depth 0]
       (if (> depth 100)
         (throw (ex-info "Infinite loop detected during AST expansion" {:ast ast}))
         (let [stepped (step-fn current)]
           (if (= current stepped)
             (normalize-remaining-terms (flatten-statements stepped))
             (recur stepped (inc depth)))))))))
