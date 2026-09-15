(ns clj-xla.logic.semiring
  "Value-Carrying Semirings for Tensor Logic (Pedro Domingos, arXiv:2510.12269).
   Generalizes Datalog contractions from crisp reachability to arbitrary algebraic semirings:
   - MIN_PLUS / Tropical (min, +): All-Pairs Shortest Paths (APSP / Bellman-Ford)
   - MAX_PRODUCT / Viterbi (max, *): Maximum probability paths & reliability networks
   - COUNTING (+, *): Path multiplicities and graph cycle counts
   Compiled and executed natively through OpenXLA PJRT via StableHLO (Rule 4)."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.ast :as ast]
            [clj-xla.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Canonical Semiring Definitions
;; ==============================================================================

(def boolean-semiring
  "Standard boolean reachability semiring: (OR, AND) over {0, 1}."
  {:id :boolean
   :add-op :or
   :mul-op :and
   :zero 0.0
   :one 1.0})

(def min-plus-semiring
  "Tropical (min-plus) semiring: (min, +) over R union {+infinity}.
   Used for shortest-path optimization (Bellman-Ford, Floyd-Warshall).
   Additive identity (zero) is +infinity (represented as 1.0e30).
   Multiplicative identity (one) is 0.0."
  {:id :min-plus
   :add-op :min
   :mul-op :add
   :zero 1.0e30
   :one 0.0})

(def tropical-semiring
  "Alias for min-plus-semiring."
  min-plus-semiring)

(def max-product-semiring
  "Viterbi (max-product) semiring: (max, *) over [0, 1].
   Used for most-probable path finding and maximum-reliability networks.
   Additive identity (zero) is 0.0.
   Multiplicative identity (one) is 1.0."
  {:id :max-product
   :add-op :max
   :mul-op :mul
   :zero 0.0
   :one 1.0})

(def viterbi-semiring
  "Alias for max-product-semiring."
  max-product-semiring)

(def counting-semiring
  "Counting semiring: (+, *) over non-negative reals/integers.
   Used for counting path multiplicities, cycle counts, and grammar parse trees.
   Additive identity (zero) is 0.0.
   Multiplicative identity (one) is 1.0."
  {:id :counting
   :add-op :add
   :mul-op :mul
   :zero 0.0
   :one 1.0})

;; ==============================================================================
;; 2. Declarative AST Constructors
;; ==============================================================================

(defn semiring-step-ast
  "Constructs a Tensor Logic contraction equation under a specified semiring:
   Head[i, j] = sum_{k}^{semiring} A[i, k] (x)^{semiring} B[k, j].
   `semiring`: a semiring map containing `:id` (e.g. :min-plus, :max-product, :counting)."
  ([semiring c-name a-name b-name]
   (semiring-step-ast semiring c-name a-name b-name [:i :j] :k))
  ([semiring c-name a-name b-name [i j] k]
   (let [s-id (if (keyword? semiring) semiring (:id semiring))
         node [:= [c-name i j] {:semiring s-id} [a-name i k] [b-name k j]]]
     (assert (ast/valid-node? node) (str "Invalid AST node in semiring-step-ast: " node))
     node)))

;; ==============================================================================
;; 3. OpenXLA PJRT Execution Drivers (Rule 4: Zero Java/Host Math Escape Hatches)
;; ==============================================================================

(defn all-pairs-shortest-paths
  "Computes all-pairs shortest paths on a directed weighted graph via OpenXLA PJRT
   using the Tropical MIN_PLUS semiring (min, +).
   `edges`: map of {[u v] weight} or sequence of [[u v] weight].
   `n`: integer number of vertices.
   `opts`: {:max-iters int (default n), :ctx OpenXLA-context}.
   Returns {:distances float-array :converged? bool :iterations int}."
  ([edges n]
   (all-pairs-shortest-paths edges n {}))
  ([edges n {:keys [max-iters ctx]}]
   (let [n-long (long n)
         max-it (long (or max-iters n-long))
         total (* n-long n-long)
         ctx-obj (or ctx (xla/get-context))
         ;; Initialize distance matrix D0
         inf 1.0e30
         d-arr (float-array total (float inf))]
     ;; Diagonal distance is 0.0
     (dotimes [i n-long]
       (aset-float d-arr (int (+ (* i n-long) i)) (float 0.0)))
     ;; Populate given edge weights
     (doseq [[[u v] w] edges]
       (let [u-idx (long u)
             v-idx (long v)]
         (when (and (>= u-idx 0) (< u-idx n-long)
                    (>= v-idx 0) (< v-idx n-long))
           (aset-float d-arr (int (+ (* u-idx n-long) v-idx)) (float w)))))

     ;; Compile single OpenXLA PJRT execution graph for iterative tropical Bellman-Ford step:
     ;; D_step[i, j] = min_k (D[i, k] + W[k, j])
     ;; D_next[i, j] = min(D[i, j], D_step[i, j])
     (let [ast [:block {:name :tropical-apsp-step}
                (semiring-step-ast min-plus-semiring :D_step :D :W [:i :j] :k)
                [:min [:D_next :i :j] [:D :i :j] [:D_step :i :j]]]
           invars [[:D [:tensor [n-long n-long] :f32]]
                   [:W [:tensor [n-long n-long] :f32]]]
           exec (sym/compile-query ctx-obj "tropical_apsp_step" invars ast [:D_next])
           w-arr (float-array total)]
       (System/arraycopy d-arr 0 w-arr 0 total)

       ;; Iterate until convergence
       (loop [curr-d d-arr
              iter 1]
         (let [out (sym/run-query! exec {:D curr-d :W w-arr})
               ^floats next-d (get out :D_next)
               max-diff (loop [idx 0 m 0.0]
                          (if (>= idx total)
                            m
                            (let [diff (Math/abs (- (double (aget curr-d idx))
                                                    (double (aget next-d idx))))]
                              (recur (inc idx) (Math/max m diff)))))
               converged? (< max-diff 1e-5)]
           (if (or converged? (>= iter max-it))
             {:distances next-d
              :converged? converged?
              :iterations iter}
             (recur next-d (inc iter)))))))))

(defn path-multiplicities
  "Computes exact path counts between all pairs of nodes of length `hops`
   via OpenXLA PJRT using the COUNTING semiring (+, *).
   `edges`: set or sequence of [u v] edges.
   `n`: integer number of vertices.
   `hops`: integer number of edges in path (>= 1).
   `opts`: {:ctx OpenXLA-context}.
   Returns {:counts float-array}."
  ([edges n hops]
   (path-multiplicities edges n hops {}))
  ([edges n hops {:keys [ctx]}]
   (let [n-long (long n)
         hops-long (long (max 1 hops))
         total (* n-long n-long)
         ctx-obj (or ctx (xla/get-context))
         a-arr (float-array total (float 0.0))]
     ;; Encode adjacency matrix
     (doseq [tuple edges]
       (let [u (long (first tuple))
             v (long (second tuple))]
         (when (and (>= u 0) (< u n-long)
                    (>= v 0) (< v n-long))
           (aset-float a-arr (int (+ (* u n-long) v)) (float 1.0)))))

     (if (= hops-long 1)
       {:counts a-arr}
       ;; Compile chained matrix multiplication under COUNTING semiring
       (let [ast [:block {:name :counting-step}
                  (semiring-step-ast counting-semiring :C_next :C :A [:i :j] :k)]
             invars [[:C [:tensor [n-long n-long] :f32]]
                     [:A [:tensor [n-long n-long] :f32]]]
             exec (sym/compile-query ctx-obj "counting_step" invars ast [:C_next])]
         (loop [curr-c a-arr
                hop 2]
           (let [out (sym/run-query! exec {:C curr-c :A a-arr})
                 ^floats next-c (get out :C_next)]
             (if (>= hop hops-long)
               {:counts next-c}
               (recur next-c (inc hop))))))))))

(defn viterbi-most-probable-paths
  "Computes maximum-probability paths on a transition network via OpenXLA PJRT
   using the MAX_PRODUCT semiring (max, *).
   `transitions`: map of {[u v] probability} with 0.0 <= prob <= 1.0.
   `n`: integer number of vertices.
   `opts`: {:max-iters int (default n), :ctx OpenXLA-context}.
   Returns {:probabilities float-array :converged? bool :iterations int}."
  ([transitions n]
   (viterbi-most-probable-paths transitions n {}))
  ([transitions n {:keys [max-iters ctx]}]
   (let [n-long (long n)
         max-it (long (or max-iters n-long))
         total (* n-long n-long)
         ctx-obj (or ctx (xla/get-context))
         p-arr (float-array total (float 0.0))]
     ;; Diagonal self-transition is 1.0
     (dotimes [i n-long]
       (aset-float p-arr (int (+ (* i n-long) i)) (float 1.0)))
     ;; Populate transition probabilities
     (doseq [[[u v] prob] transitions]
       (let [u-idx (long u)
             v-idx (long v)]
         (when (and (>= u-idx 0) (< u-idx n-long)
                    (>= v-idx 0) (< v-idx n-long))
           (aset-float p-arr (int (+ (* u-idx n-long) v-idx)) (float prob)))))

     (let [ast [:block {:name :viterbi-step}
                (semiring-step-ast max-product-semiring :P_step :P :T [:i :j] :k)
                [:max [:P_next :i :j] [:P :i :j] [:P_step :i :j]]]
           invars [[:P [:tensor [n-long n-long] :f32]]
                   [:T [:tensor [n-long n-long] :f32]]]
           exec (sym/compile-query ctx-obj "viterbi_step" invars ast [:P_next])
           t-arr (float-array total)]
       (System/arraycopy p-arr 0 t-arr 0 total)

       (loop [curr-p p-arr
              iter 1]
         (let [out (sym/run-query! exec {:P curr-p :T t-arr})
               ^floats next-p (get out :P_next)
               max-diff (loop [idx 0 m 0.0]
                          (if (>= idx total)
                            m
                            (let [diff (Math/abs (- (double (aget curr-p idx))
                                                    (double (aget next-p idx))))]
                              (recur (inc idx) (Math/max m diff)))))
               converged? (< max-diff 1e-5)]
           (if (or converged? (>= iter max-it))
             {:probabilities next-p
              :converged? converged?
              :iterations iter}
             (recur next-p (inc iter)))))))))
