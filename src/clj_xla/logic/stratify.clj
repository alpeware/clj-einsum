(ns clj-xla.logic.stratify
  "Stratified Negation Engine for Tensor Logic (Pedro Domingos, arXiv:2510.12269).
   Implements dependency graph cycle analysis, rejection of unstratifiable negative cycles,
   topological stratum decomposition, and multi-stage fixpoint execution via OpenXLA PJRT (Rule 4)."
  (:require [clj-xla.logic.symbolic :as sym]
            [clojure.set :as set]))

;; ==============================================================================
;; 1. Rule & Atom Parsing
;; ==============================================================================

(defn parse-atom
  "Parses a positive or negative rule body atom.
   Supports:
   - Positive: [:rel-name :x :y]
   - Negative nested: [:not [:rel-name :x :y]]
   - Negative flat: [:not :rel-name :x :y]
   Returns {:neg? boolean :rel keyword :args [keyword ...]}."
  [atom]
  (cond
    (= (first atom) :not)
    (let [inner (second atom)]
      (if (vector? inner)
        {:neg? true :rel (first inner) :args (vec (rest inner))}
        {:neg? true :rel inner :args (vec (drop 2 atom))}))
    :else
    {:neg? false :rel (first atom) :args (vec (rest atom))}))

;; ==============================================================================
;; 2. Dependency Graph & Stratification Cycle Analysis
;; ==============================================================================

(defn stratify-program
  "Analyzes dependency graph of Datalog rules with negation and assigns stratum numbers.
   Constraints:
   - For positive dependency B -> H: stratum(H) >= stratum(B)
   - For negative dependency B -> H: stratum(H) >= stratum(B) + 1
   Throws ex-info with :type :unstratified-negative-cycle if a negative cycle is detected.
   Returns map from relation keyword to integer stratum {rel-kw stratum-int}."
  [rules]
  (let [;; Collect all distinct relations
        head-rels (set (map (comp first :head) rules))
        body-atoms (mapcat (fn [r] (map parse-atom (:body r))) rules)
        body-rels (set (map :rel body-atoms))
        all-rels (vec (distinct (concat head-rels body-rels)))
        num-rels (count all-rels)

        ;; Build dependency constraints: [body-rel head-rel min-diff]
        ;; min-diff is 0 for positive dependencies, 1 for negative dependencies
        constraints (vec
                     (mapcat
                      (fn [r]
                        (let [h-rel (first (:head r))]
                          (map (fn [atom]
                                 (let [{:keys [neg? rel]} (parse-atom atom)]
                                   [rel h-rel (if neg? 1 0)]))
                               (:body r))))
                      rules))

        ;; Bellman-Ford style stratum relaxation
        init-strata (into {} (map (fn [r] [r 0]) all-rels))
        relax-step (fn [curr-strata]
                     (reduce
                      (fn [s [b h min-diff]]
                        (let [b-val (get s b 0)
                              h-val (get s h 0)
                              needed (+ b-val min-diff)]
                          (if (> needed h-val)
                            (assoc s h needed)
                            s)))
                      curr-strata
                      constraints))
        final-strata (loop [s init-strata
                            iter 0]
                       (if (>= iter num-rels)
                         s
                         (let [next-s (relax-step s)]
                           (if (= next-s s)
                             next-s
                             (recur next-s (inc iter))))))
        after-extra (relax-step final-strata)]
    (when (not= final-strata after-extra)
      (throw (ex-info "Unstratified negative cycle detected in rule dependencies"
                      {:type :unstratified-negative-cycle
                       :rules rules
                       :strata final-strata})))
    final-strata))

;; ==============================================================================
;; 3. Stratified Fixpoint Evaluation via OpenXLA PJRT
;; ==============================================================================

(defn eval-stratified
  "Evaluates a stratified Datalog program with negation stratum by stratum via OpenXLA PJRT.
   Each stratum is evaluated to fixpoint. When moving to stratum k, negated relations
   from lower strata (< k) are frozen and their active-domain complement is injected as fixed EDB.
   Supports multiple rules per head relation within any stratum.
   `opts`: {:rules [...]
            :facts {rel-kw #{[x y] ...}}
            :n integer domain size
            :d integer embedding dimension (default 256)
            :seed int (optional)
            :max-iters int (optional)
            :threshold double (optional, default 0.5)}
   Returns {:facts {rel-kw #{...}} :strata {rel-kw stratum-int}}."
  [{:keys [rules facts n d seed max-iters threshold] :as _opts}]
  (let [n-long (long n)
        d-long (long (or d 256))
        thresh (double (or threshold 0.5))
        max-it (long (or max-iters 10))
        strata (stratify-program rules)
        distinct-strata (vec (sort (distinct (vals strata))))
        all-pairs (set (for [i (range n-long) j (range n-long)] [i j]))
        init-facts (into {} (map (fn [[k v]] [k (set v)]) facts))]

    (loop [curr-facts init-facts
           strata-seq distinct-strata]
      (if (empty? strata-seq)
        {:facts curr-facts
         :strata strata}
        (let [stratum-num (first strata-seq)
              stratum-rules (filter (fn [r]
                                      (= (get strata (first (:head r))) stratum-num))
                                    rules)]
          (if (empty? stratum-rules)
            (recur curr-facts (rest strata-seq))
            ;; For each negated atom in stratum-rules, replace with complement EDB relation
            (let [neg-atoms (filter :neg?
                                    (mapcat (fn [r] (map parse-atom (:body r))) stratum-rules))
                  comp-facts-map (into {}
                                       (map (fn [{:keys [rel]}]
                                              (let [comp-rel (keyword (str "not_" (name rel)))
                                                    positive-facts (get curr-facts rel #{})
                                                    comp-facts (set/difference all-pairs positive-facts)]
                                                [comp-rel comp-facts]))
                                            neg-atoms))
                  transformed-rules
                  (mapv (fn [r]
                          (let [new-body
                                (mapv (fn [atom]
                                        (let [{:keys [neg? rel args]} (parse-atom atom)]
                                          (if neg?
                                            (into [(keyword (str "not_" (name rel)))] args)
                                            atom)))
                                      (:body r))]
                            (assoc r :body new-body)))
                        stratum-rules)

                  merged-facts (merge curr-facts comp-facts-map)

                  heads (mapv (comp first :head) transformed-rules)
                  distinct-heads? (= (count heads) (count (set heads)))

                  stratum-final-facts
                  (if distinct-heads?
                    (let [res (sym/forward-chain
                               {:rules transformed-rules
                                :facts merged-facts
                                :n n-long
                                :d d-long
                                :seed (or seed 42)
                                :max-iters max-it
                                :threshold thresh})]
                      (merge-with set/union curr-facts (:facts res)))
                    ;; Iterate rules with duplicate heads in an outer fixpoint loop
                    (loop [s-facts merged-facts
                           iter 0]
                      (let [next-s-facts
                            (reduce
                             (fn [f-acc r]
                               (let [r-head (first (:head r))
                                     res (sym/forward-chain
                                          {:rules [r]
                                           :facts f-acc
                                           :n n-long
                                           :d d-long
                                           :seed (or seed 42)
                                           :max-iters 1
                                           :threshold thresh})
                                     new-tuples (get-in res [:head-facts r-head] #{})]
                                 (update f-acc r-head (fnil into #{}) new-tuples)))
                             s-facts
                             transformed-rules)]
                        (if (or (= next-s-facts s-facts) (>= iter max-it))
                          (merge-with set/union curr-facts next-s-facts)
                          (recur next-s-facts (inc iter))))))]
              (recur stratum-final-facts (rest strata-seq)))))))))
