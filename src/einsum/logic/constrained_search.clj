(ns einsum.logic.constrained-search
  "Experiment E20: Constrained Crystallization via Pair-Assignment Simulated Annealing.
   Pure Clojure (Sans-IO) conforming to Repository Rules 1 and 2.
   Enforces structural domain priors:
   - Indegree in {0, 2} (each entity is either a root or has exactly 2 parents)
   - Acyclicity (strictly directed acyclic graph)
   Evaluates boolean composition matrices on the OR-AND semiring:
   - Grandparent: L o L
   - Sibling: L^T o L (off-diagonal)
   Computes ambiguity-aware F1* (modulo mating child swap) and raw Parent F1."
  (:require [einsum.logic.predicate-crystallization :as pc]))

;; ==============================================================================
;; 1. State Representation & Invariant Validation
;; ==============================================================================

(defn make-empty-state
  "Constructs an empty state vector of n entities (all roots, indegree 0)."
  ([] (make-empty-state 64))
  ([^long n]
   (vec (repeat n nil))))

(defn state->edges
  "Converts a state vector (where state[c] is nil or [p1 p2]) to a set of directed pairs [p c]."
  [state]
  (let [n (count state)
        res (java.util.HashSet.)]
    (dotimes [c n]
      (when-let [parents (nth state c)]
        (let [[p1 p2] parents]
          (.add res [p1 c])
          (.add res [p2 c]))))
    (set res)))

(defn valid-state-indegree?
  "Checks that every node in state has indegree in {0, 2} and parents are valid distinct nodes."
  [state]
  (let [n (count state)]
    (every? (fn [c]
              (let [p (nth state c)]
                (or (nil? p)
                    (and (vector? p)
                         (= 2 (count p))
                         (let [[p1 p2] p]
                           (and (integer? p1) (integer? p2)
                                (< -1 p1 n) (< -1 p2 n)
                                (not= p1 p2)
                                (not= p1 c) (not= p2 c)))))))
            (range n))))

;; ==============================================================================
;; 2. Incremental Reachability & Cycle Detection
;; ==============================================================================

(defn reachable?
  "Checks whether target is reachable from start in the directed graph defined by state."
  [state ^long start ^long target]
  (if (== start target)
    true
    (let [n (count state)
          visited (boolean-array n)
          q (java.util.ArrayDeque.)]
      (.add q (int start))
      (aset visited (int start) true)
      (loop []
        (if (.isEmpty q)
          false
          (let [u (.poll q)]
            (if (== u (int target))
              true
              (do
                ;; Follow outgoing edges u -> c
                (dotimes [c n]
                  (when-not (aget visited c)
                    (when-let [p (nth state c)]
                      (when (or (== (int (first p)) u) (== (int (second p)) u))
                        (aset visited c true)
                        (.add q (int c))))))
                (recur)))))))))

(defn cycle-free-with?
  "Checks if assigning parents [p1 p2] to child c in state would create a directed cycle.
   A cycle would be created iff c can already reach p1 or p2 in state (with c's parents removed)."
  [state ^long c ^long p1 ^long p2]
  (let [state-without-c (assoc state c nil)]
    (not (or (reachable? state-without-c c p1)
             (reachable? state-without-c c p2)))))

(defn acyclic-state?
  "Full validation checking whether state contains any directed cycles."
  [state]
  (let [n (count state)]
    (every? (fn [c]
              (if-let [[p1 p2] (nth state c)]
                (cycle-free-with? state c p1 p2)
                true))
            (range n))))

;; ==============================================================================
;; 3. Objective Function & Ambiguity-Aware F1* Metric
;; ==============================================================================

(defn state-score
  "Computes composition score on the boolean OR-AND semiring:
   score(state) = F1(Bool(L o L), G_target) + F1(Bool(L^T o L), S_target)."
  [state g-target s-target]
  (let [edges (state->edges state)
        gp (pc/bool-mat-or-and-sq edges)
        sib (pc/bool-mat-trans-or-and edges)
        f1-gp (:f1 (pc/pair-set-f1 gp g-target))
        f1-sib (:f1 (pc/pair-set-f1 sib s-target))]
    (+ f1-gp f1-sib)))

(defn compute-f1-star
  "Computes ambiguity-aware F1* (mean over family trees of max(F1 vs truth, F1 vs mating-swapped truth))."
  [recovered-edges forest]
  (let [trees (:trees forest)]
    (if (empty? trees)
      0.0
      (let [f1-stars
            (mapv (fn [tree]
                    (let [true-p (set (:parent-pairs tree))
                          m (:mating-child tree)
                          o (:other-child tree)
                          g-kids (set (:gen2-children tree))
                          swapped-p (set (map (fn [[u v]]
                                                (if (and (= u m) (contains? g-kids v))
                                                  [o v]
                                                  [u v]))
                                              true-p))
                          tree-nodes (set (concat (:founders tree) [(:spouse tree)]
                                                  (:gen1-children tree) (:gen2-children tree)))
                          rec-tree (set (filter (fn [[u v]]
                                                  (and (contains? tree-nodes u)
                                                       (contains? tree-nodes v)))
                                                recovered-edges))
                          f1-true (:f1 (pc/pair-set-f1 rec-tree true-p))
                          f1-swapped (:f1 (pc/pair-set-f1 rec-tree swapped-p))]
                      (max f1-true f1-swapped)))
                  trees)]
        (/ (reduce + (map double f1-stars)) (double (count f1-stars)))))))

;; ==============================================================================
;; 4. Move Generation & Simulated Annealing Kernel
;; ==============================================================================

(defn random-distinct-pair
  "Generates a random sorted pair [p1 p2] with p1 != p2, p1 != c, p2 != c."
  [^long c ^long n ^java.util.Random rnd]
  (let [p1 (loop []
             (let [x (.nextInt rnd (int n))]
               (if (== x c) (recur) x)))
        p2 (loop []
             (let [x (.nextInt rnd (int n))]
               (if (or (== x c) (== x p1)) (recur) x)))]
    [(min p1 p2) (max p1 p2)]))

(defn propose-sa-move
  "Proposes a new candidate state according to move types:
   0: Reassign parent pair to node c
   1: Make node c a root (nil)
   2: Swap parent assignments between node c and node c2."
  [state ^long n ^java.util.Random rnd]
  (let [move-type (.nextInt rnd 3)
        c (.nextInt rnd (int n))]
    (case move-type
      ;; Move 0: Reassign to random candidate pair
      0 (let [[p1 p2] (random-distinct-pair c n rnd)]
          (if (cycle-free-with? state c p1 p2)
            (assoc state c [p1 p2])
            state))

      ;; Move 1: Make root
      1 (if (nil? (nth state c))
          state
          (assoc state c nil))

      ;; Move 2: Swap pair assignments of two nodes
      2 (let [c2 (.nextInt rnd (int n))]
          (if (== c c2)
            state
            (let [p-c (nth state c)
                  p-c2 (nth state c2)]
              (if (= p-c p-c2)
                state
                (let [s1 (assoc state c nil c2 nil)
                      ok-c (or (nil? p-c2)
                               (and (not (or (== (first p-c2) c) (== (second p-c2) c)))
                                    (cycle-free-with? s1 c (first p-c2) (second p-c2))))
                      s2 (if ok-c (assoc s1 c p-c2) s1)
                      ok-c2 (and ok-c
                                 (or (nil? p-c)
                                     (and (not (or (== (first p-c) c2) (== (second p-c) c2)))
                                          (cycle-free-with? s2 c2 (first p-c) (second p-c)))))]
                  (if ok-c2
                    (assoc s2 c2 p-c)
                    state)))))))))

(defn run-constrained-sa-restart
  "Runs a single simulated annealing trajectory of n-steps from init-state."
  [g-target s-target opts]
  (let [n (long (or (:num-entities opts) 64))
        n-steps (long (or (:steps opts) 200000))
        t-init (double (or (:t-init opts) 0.1))
        t-min (double (or (:t-min opts) 1e-4))
        rnd ^java.util.Random (or (:rnd opts) (java.util.Random. 42))
        init-state (or (:init-state opts) (make-empty-state n))
        decay (Math/pow (/ t-min t-init) (/ 1.0 (double (max 1 (dec n-steps)))))
        tiebreaker-z ^floats (:tiebreaker-z opts)]
    (loop [step 0
           state init-state
           score (double (state-score init-state g-target s-target))
           temp t-init
           best-state init-state
           best-score score]
      (if (>= step n-steps)
        {:best-state best-state
         :best-score best-score
         :final-state state
         :final-score score
         :steps n-steps}
        (let [cand-state (propose-sa-move state n rnd)
              cand-score (if (= cand-state state)
                           score
                           (double (state-score cand-state g-target s-target)))
              delta (- cand-score score)
              accept?
              (cond
                (> delta 0.0) true
                (< delta 0.0) (> (Math/exp (/ delta temp)) (.nextDouble rnd))
                ;; Tiebreaker mode (delta == 0.0)
                :else
                (if (and tiebreaker-z (not= cand-state state))
                  ;; Break tie using mean Z logit of assigned pairs
                  (let [z-mean (fn [s]
                                 (let [pairs (filter some? s)]
                                   (if (empty? pairs)
                                     0.0
                                     (/ (reduce + (map (fn [[p1 p2]]
                                                         (+ (double (aget tiebreaker-z (+ (* p1 n) p2)))
                                                            (double (aget tiebreaker-z (+ (* p2 n) p1)))))
                                                       pairs))
                                        (double (* 2 (count pairs)))))))]
                    (>= (z-mean cand-state) (z-mean state)))
                  ;; Neutral acceptance on exact tie
                  true))
              next-state (if accept? cand-state state)
              next-score (if accept? cand-score score)
              next-temp (* temp decay)
              new-best? (> next-score best-score)]
          (recur (inc step)
                 next-state
                 next-score
                 next-temp
                 (if new-best? next-state best-state)
                 (if new-best? next-score best-score)))))))

(defn run-constrained-sa
  "Runs multi-restart constrained simulated annealing.
   Returns the best result across restarts."
  [g-target s-target opts]
  (let [restarts (long (or (:restarts opts) 3))
        base-seed (long (or (:seed opts) 42))
        t0 (System/nanoTime)
        results
        (mapv (fn [r-idx]
                (let [rnd (java.util.Random. (+ (* base-seed 1000) r-idx))
                      restart-opts (assoc opts :rnd rnd)]
                  (run-constrained-sa-restart g-target s-target restart-opts)))
              (range restarts))
        best-res (apply max-key :best-score results)
        elapsed-ms (/ (- (System/nanoTime) t0) 1e6)]
    (assoc best-res
           :time-ms elapsed-ms
           :restarts restarts
           :all-restart-scores (mapv :best-score results))))

;; ==============================================================================
;; 5. Phase 0a Oracle Ceiling Search (Restricted to True Pairs)
;; ==============================================================================

(defn run-oracle-ceiling-sa
  "Runs simulated annealing restricted exclusively to the true parent-pairs (P0a gate)."
  [g-target s-target forest opts]
  (let [n (long (or (:num-entities opts) 64))
        n-steps (long (or (:steps opts) 30000))
        true-parents (set (:parent-pairs forest))
        child->true-parents
        (into {} (map (fn [[c pairs]] [c (vec (sort (map first pairs)))])
                      (group-by second true-parents)))
        init-state (make-empty-state n)
        rnd (java.util.Random. (long (or (:seed opts) 42)))
        t-init (double (or (:t-init opts) 0.1))
        t-min (double (or (:t-min opts) 1e-4))
        decay (Math/pow (/ t-min t-init) (/ 1.0 (double n-steps)))
        t0 (System/nanoTime)
        sa-res
        (loop [step 0
               state init-state
               score (double (state-score init-state g-target s-target))
               temp t-init
               best-state init-state
               best-score score]
          (if (>= step n-steps)
            {:best-state best-state :best-score best-score}
            (let [c (.nextInt rnd (int n))
                  true-p (get child->true-parents c)
                  cand-state
                  (if true-p
                    (if (nil? (nth state c))
                      (assoc state c true-p)
                      (assoc state c nil))
                    state)
                  cand-score (if (= cand-state state)
                               score
                               (double (state-score cand-state g-target s-target)))
                  delta (- cand-score score)
                  accept? (or (>= delta 0.0)
                              (> (Math/exp (/ delta temp)) (.nextDouble rnd)))
                  next-state (if accept? cand-state state)
                  next-score (if accept? cand-score score)
                  next-temp (* temp decay)
                  new-best? (> next-score best-score)]
              (recur (inc step)
                     next-state
                     next-score
                     next-temp
                     (if new-best? next-state best-state)
                     (if new-best? next-score best-score)))))
        elapsed-ms (/ (- (System/nanoTime) t0) 1e6)
        recovered-edges (state->edges (:best-state sa-res))
        f1-star (compute-f1-star recovered-edges forest)
        raw-f1 (:f1 (pc/pair-set-f1 recovered-edges true-parents))]
    {:best-state (:best-state sa-res)
     :best-score (:best-score sa-res)
     :recovered-edges recovered-edges
     :recovered-count (count recovered-edges)
     :f1-star f1-star
     :raw-f1 raw-f1
     :time-ms elapsed-ms
     :steps n-steps}))
