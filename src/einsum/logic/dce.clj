(ns einsum.logic.dce
  "Backward-chaining dead-code elimination and implicit accumulation grouping for Tensor Logic."
  (:require [clojure.string :as str]
            [einsum.logic.ast :as ast]))

(defn eqn-head-names [eqn]
  (let [h (ast/head eqn)]
    (cond
      (= (first eqn) :while)
      (let [out-spec (second eqn)]
        (mapv #(if (vector? %) (first %) %) (if (vector? out-spec) out-spec [out-spec])))

      (= (first eqn) :constant)
      (let [h (second eqn)]
        [(if (vector? h) (first h) h)])

      (and (vector? h) (seq h) (every? vector? h))
      (mapv (fn [elem] (if (vector? (first elem)) (first (first elem)) (first elem))) h)

      (and (vector? h) (vector? (first h)))
      [(let [f (first h)]
         (if (and (= (name (first f)) "h") (number? (second f)))
           (keyword (str "h" (second f)))
           (keyword (str (str/replace (name (first f)) #"-" "_") "_" (str/join "_" (map (fn [a] (if (keyword? a) (name a) (str a))) (rest f)))))))]

      (vector? h)
      [(first h)]

      (keyword? h)
      [h]

      :else
      [])))

(defn eqn-body-names [eqn]
  (cond
    (= (first eqn) :while)
    (let [in-spec (nth eqn 2)
          in-names (mapv #(if (vector? %) (first %) %) (if (vector? in-spec) in-spec [in-spec]))
          attrs (ast/attrs eqn)
          cond-names (when-let [cg (:cond-graph attrs)] (map first (:invars cg)))
          body-names (when-let [bg (:body-graph attrs)] (map first (:invars bg)))]
      (set (concat in-names cond-names body-names)))

    (= (first eqn) :constant)
    #{}

    :else
    (let [body-names (set (map first (ast/body-terms eqn)))
          attrs (ast/attrs eqn)
          start-idx-names (when-let [starts (or (:start_indices attrs) (:start-indices attrs))]
                            (cond
                              (coll? starts) (set (filter keyword? starts))
                              (keyword? starts) #{starts}
                              :else nil))
          pos-name (when-let [p (:pos attrs)] (when (keyword? p) #{p}))]
      (cond-> body-names
        (seq start-idx-names) (into start-idx-names)
        pos-name (into pos-name)))))

(defn prune-ast
  "Performs backward-chaining dead-code elimination starting from `target-heads`.
   Prunes equations that do not contribute transitively to any target head."
  [equations target-heads]
  (let [target-set (set target-heads)
        ;; Map head names to all defining equations
        head->eqns (reduce (fn [acc eqn]
                             (reduce (fn [a h]
                                       (update a h (fnil conj []) eqn))
                                     acc
                                     (eqn-head-names eqn)))
                           {}
                           equations)]
    (loop [worklist (into [] target-set)
           needed-heads #{}
           visited-heads #{}]
      (if (empty? worklist)
        ;; Filter equations in original order that produce needed heads
        (filterv (fn [eqn]
                   (some #(contains? needed-heads %) (eqn-head-names eqn)))
                 equations)
        (let [curr-head (first worklist)
              rest-worklist (subvec worklist 1)]
          (if (contains? visited-heads curr-head)
            (recur rest-worklist needed-heads visited-heads)
            (let [def-eqns (get head->eqns curr-head [])
                  input-names (mapcat eqn-body-names def-eqns)
                  new-work (filter #(and (contains? head->eqns %)
                                         (not (contains? visited-heads %)))
                                   input-names)]
              (recur (into rest-worklist new-work)
                     (conj needed-heads curr-head)
                     (conj visited-heads curr-head)))))))))

(defn group-accumulations
  "Groups equations sharing an identical head signature for implicit addition lowering."
  [equations]
  (let [by-head (group-by ast/head equations)]
    (mapv (fn [eqn]
            (let [h (ast/head eqn)
                  eqns (get by-head h)]
              (if (> (count eqns) 1)
                (assoc eqn :accum-group eqns)
                eqn)))
          equations)))
