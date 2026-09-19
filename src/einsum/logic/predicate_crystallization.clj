(ns einsum.logic.predicate-crystallization
  "Hybrid Crystallization: Gradient-Based Proposer + Discrete Combinatorial Search (Experiment E19).
   Pure Clojure (Sans-IO) conforming to Repository Rules 1 and 2.
   Evaluates boolean composition matrices (Grandparent = L o L, Sibling = L^T o L)
   over candidate edge sets using exact boolean semiring matrix multiplication and
   deterministic greedy search with backward elimination."
  (:require [clojure.set :as set]))

;; ==============================================================================
;; 1. Exact Boolean Semiring Operations (Pure Functions)
;; ==============================================================================

(defn bool-mat-or-and-sq
  "Computes Bool(L o L) on the boolean OR-AND semiring.
   Returns the set of directed 2-hop pairs [h t] reachable via intermediate node m."
  [edges]
  (let [head->tails (group-by first edges)
        res (java.util.HashSet.)]
    (doseq [[h m] edges]
      (when-let [tails (get head->tails m)]
        (doseq [[_ t] tails]
          (.add res [h t]))))
    (set res)))

(defn bool-mat-trans-or-and
  "Computes off-diagonal Bool(L^T o L) on the boolean OR-AND semiring.
   Returns the set of pairs [s1 s2] with s1 != s2 that share at least one parent p."
  [edges]
  (let [parent->children (group-by first edges)
        res (java.util.HashSet.)]
    (doseq [[_p children] parent->children]
      (let [kids (mapv second children)
            k (count kids)]
        (dotimes [i k]
          (let [s1 (nth kids i)]
            (dotimes [j k]
              (let [s2 (nth kids j)]
                (when (not= s1 s2)
                  (.add res [s1 s2]))))))))
    (set res)))

;; ==============================================================================
;; 2. F1 and Objective Function
;; ==============================================================================

(defn pair-set-f1
  "Computes discrete precision, recall, and F1 between a predicted pair set and a true pair set."
  [pred-set true-set]
  (if (and (empty? pred-set) (empty? true-set))
    {:precision 1.0 :recall 1.0 :f1 1.0 :tp 0 :fp 0 :fn 0}
    (let [tp (count (set/intersection pred-set true-set))
          fp (- (count pred-set) tp)
          fn-cnt (- (count true-set) tp)
          p (if (zero? (+ tp fp)) 0.0 (/ (double tp) (double (+ tp fp))))
          r (if (zero? (+ tp fn-cnt)) 0.0 (/ (double tp) (double (+ tp fn-cnt))))
          f1 (if (zero? (+ p r)) 0.0 (/ (* 2.0 p r) (+ p r)))]
      {:precision (double p)
       :recall (double r)
       :f1 (double f1)
       :tp (long tp)
       :fp (long fp)
       :fn (long fn-cnt)})))

(defn crystallization-score
  "Computes the E19 objective:
   score(L_hat) = F1(Bool(L_hat o L_hat), G_target) + F1(Bool(L_hat^T o L_hat), S_target) - lambda * ||L_hat||_0."
  ([edges g-target s-target]
   (crystallization-score edges g-target s-target 0.01))
  ([edges g-target s-target lambda-val]
   (let [gp-pred (bool-mat-or-and-sq edges)
         sib-pred (bool-mat-trans-or-and edges)
         f1-gp (:f1 (pair-set-f1 gp-pred g-target))
         f1-sib (:f1 (pair-set-f1 sib-pred s-target))
         sparsity-cost (* (double lambda-val) (double (count edges)))]
     (- (+ f1-gp f1-sib) sparsity-cost))))

;; ==============================================================================
;; 3. Deterministic Greedy Search with Backward Elimination
;; ==============================================================================

(defn greedy-forward-search
  "Forward greedy selection starting from initial-edges (default empty set).
   Repeatedly adds the edge in candidates \\ selected that strictly maximizes score.
   When no 1-edge addition improves (e.g. at cold-start where a single edge produces 0 compositions),
   evaluates connected 2-edge additions.
   Stops when no addition improves score.
   Returns {:selected-edges ... :eval-count ... :trajectory ...}."
  ([candidates g-target s-target lambda-val]
   (greedy-forward-search candidates g-target s-target lambda-val #{}))
  ([candidates g-target s-target lambda-val initial-edges]
   (let [cand-vec (vec (sort candidates))]
     (loop [selected (set initial-edges)
            cur-score (crystallization-score selected g-target s-target lambda-val)
            evals 1
            traj [{:step 0 :edges (count selected) :score cur-score}]]
       (let [avail (vec (filter #(not (contains? selected %)) cand-vec))
             ;; 1. Single-edge greedy additions
             scored-1 (mapv (fn [e]
                              [e (crystallization-score (conj selected e) g-target s-target lambda-val)])
                            avail)
             new-evals (+ evals (count scored-1))
             best-1 (when (seq scored-1) (apply max-key second scored-1))]
         (if (and best-1 (> (double (second best-1)) (double cur-score)))
           (let [next-selected (conj selected (first best-1))
                 next-score (second best-1)]
             (recur next-selected
                    next-score
                    new-evals
                    (conj traj {:step (count traj) :added [(first best-1)] :edges (count next-selected) :score next-score})))

           ;; 2. Connected 2-edge lookahead fallback (cold-start / disjoint-tree barrier resolution)
           (let [avail-set (set avail)
                 gp-pairs (for [[u w] g-target
                                m (range 64)
                                :let [e1 [u m] e2 [m w]]
                                :when (and (contains? avail-set e1)
                                           (contains? avail-set e2))]
                            [e1 e2])
                 sib-pairs (for [[s1 s2] s-target
                                 p (range 64)
                                 :let [e1 [p s1] e2 [p s2]]
                                 :when (and (contains? avail-set e1)
                                            (contains? avail-set e2))]
                             [e1 e2])
                 pair-cands (vec (distinct (concat gp-pairs sib-pairs)))
                 scored-2 (mapv (fn [[e1 e2]]
                                  [[e1 e2] (crystallization-score (conj selected e1 e2) g-target s-target lambda-val)])
                                pair-cands)
                 evals-with-2 (+ new-evals (count scored-2))
                 best-2 (when (seq scored-2) (apply max-key second scored-2))]
             (if (and best-2 (> (double (second best-2)) (double cur-score)))
               (let [next-selected (into selected (first best-2))
                     next-score (second best-2)]
                 (recur next-selected
                        next-score
                        evals-with-2
                        (conj traj {:step (count traj) :added (first best-2) :edges (count next-selected) :score next-score})))
               ;; No 1-edge or 2-edge addition improves score; terminate forward pass
               {:selected-edges selected
                :final-score cur-score
                :eval-count evals-with-2
                :trajectory traj}))))))))

(defn backward-elimination
  "Backward elimination pass: repeatedly removes the edge from selected whose removal
   most improves score. Stops when no single-edge removal strictly improves score.
   Returns {:selected-edges ... :eval-count ... :trajectory ...}."
  [selected-edges g-target s-target lambda-val]
  (loop [selected (set selected-edges)
         cur-score (crystallization-score selected g-target s-target lambda-val)
         evals 0
         traj [{:step 0 :edges (count selected) :score cur-score}]]
    (if (empty? selected)
      {:selected-edges #{} :final-score cur-score :eval-count evals :trajectory traj}
      (let [scored-removals
            (mapv (fn [e]
                    [e (crystallization-score (disj selected e) g-target s-target lambda-val)])
                  (vec (sort selected)))
            new-evals (+ evals (count scored-removals))
            best-removal (when (seq scored-removals) (apply max-key second scored-removals))]
        (if (and best-removal (> (double (second best-removal)) (double cur-score)))
          (let [next-selected (disj selected (first best-removal))
                next-score (second best-removal)]
            (recur next-selected
                   next-score
                   new-evals
                   (conj traj {:step (count traj) :removed (first best-removal) :edges (count next-selected) :score next-score})))
          {:selected-edges selected
           :final-score cur-score
           :eval-count new-evals
           :trajectory traj})))))

(defn discrete-crystallization-search
  "Full hybrid crystallization search (forward selection + backward elimination).
   Options:
   - :lambda (default 0.01)
   - :initial-edges (default #{})"
  ([candidates g-target s-target]
   (discrete-crystallization-search candidates g-target s-target nil))
  ([candidates g-target s-target opts]
   (let [lambda-val (double (or (:lambda opts) 0.01))
         init-edges (or (:initial-edges opts) #{})
         t0 (System/nanoTime)
         fwd-res (greedy-forward-search candidates g-target s-target lambda-val init-edges)
         fwd-edges (:selected-edges fwd-res)
         bwd-res (backward-elimination fwd-edges g-target s-target lambda-val)
         final-edges (:selected-edges bwd-res)
         final-score (:final-score bwd-res)
         total-evals (+ (long (:eval-count fwd-res)) (long (:eval-count bwd-res)))
         elapsed-ms (/ (- (System/nanoTime) t0) 1e6)]
     {:edges final-edges
      :edge-count (count final-edges)
      :score final-score
      :eval-count total-evals
      :time-ms elapsed-ms
      :fwd-count (count fwd-edges)
      :bwd-count (count final-edges)})))
