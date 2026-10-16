(ns einsum.uop.rangeify
  "Backward coordinate propagation (`rangeify`) pass for UOp kernels.
   Modeled on tinygrad's `rangeify` (`tinygrad/schedule/rangeify.py`) and geohot's ML compiler syllabus (Weeks 3-5).
   Walks a kernel's UOp DAG in reverse topological order, dissolving all high-level Movement Ops
   (`RESHAPE`, `EXPAND`, `PERMUTE`, `SHRINK`, `PAD`, `FLIP`, `CAT`) into explicit `RANGE` loops
   and simplified integer `INDEX` arithmetic over flat 1D buffers."
  (:require [einsum.uop.core :as uop]
            [einsum.uop.rewrite :as rewrite]))

;; -----------------------------------------------------------------------------
;; 1. Stride & Linear Index Helpers
;; -----------------------------------------------------------------------------

(defn row-major-strides
  "Computes row-major stride vector for `shape` (e.g. `[2 3 4]` -> `[12 4 1]`)."
  [shape]
  (let [n (count shape)]
    (if (zero? n)
      []
      (vec (reverse (reductions * 1 (reverse (rest shape))))))))

(defn coords->linear-index
  "Combines a vector of coordinate UOps `coords` and `shape` into a simplified 1D linear index UOp
   $\\sum_i c_i \\cdot \\text{stride}_i$."
  [coords shape]
  (if (empty? shape)
    (uop/uop-const :i32 0)
    (let [strides (row-major-strides shape)
          terms (keep (fn [[c s]]
                        (let [s-long (long s)]
                          (cond
                            (zero? s-long) nil
                            (and (= :uop/const (:op c)) (zero? (long (:arg c)))) nil
                            (== 1 s-long) c
                            :else (uop/uop-alu :uop/mul c (uop/uop-const :i32 s-long)))))
                      (map vector coords strides))]
      (if (empty? terms)
        (uop/uop-const :i32 0)
        (rewrite/simplify-uop
         (reduce (fn [acc t] (uop/uop-alu :uop/add acc t)) terms))))))

(defn linear-index->coords
  "Decomposes a 1D linear index UOp `lin-idx` into a vector of coordinate UOps for `target-shape`
   using `//` (`:uop/idiv`) and `%` (`:uop/mod`), simplified via `rewrite/simplify-uop`."
  [lin-idx target-shape]
  (if (empty? target-shape)
    []
    (let [strides (row-major-strides target-shape)]
      (mapv (fn [dim stride]
              (let [d (long dim)
                    s (long stride)]
                (if (== 1 d)
                  (uop/uop-const :i32 0)
                  (let [div-node (if (== 1 s)
                                   lin-idx
                                   (uop/uop-alu :uop/idiv lin-idx (uop/uop-const :i32 s)))
                        mod-node (uop/uop-alu :uop/mod div-node (uop/uop-const :i32 d))]
                    (rewrite/simplify-uop mod-node)))))
            target-shape
            strides))))

;; -----------------------------------------------------------------------------
;; 2. Backward Coordinate Propagation (`rangeify`)
;; -----------------------------------------------------------------------------

(defn rangeify-kernel
  "Transforms a high-level kernel root UOp (containing at most one `:uop/reduce` and arbitrary
   Movement + ALU UOps) into a shape-free, loop-explicit UOp DAG rooted at `:uop/sink` -> `:uop/store`.
   Options:
     `:out-slot` - output buffer slot integer (default 0)
     `:axis-counter` - optional atom for generating unique range axis IDs."
  ([root-uop]
   (rangeify-kernel root-uop {}))
  ([root-uop opts]
   (let [out-shape (vec (or (:shape root-uop) []))
         out-dtype (:dtype root-uop)
         out-size (long (reduce * 1 out-shape))
         out-slot (long (or (:out-slot opts) 0))
         axis-counter (or (:axis-counter opts) (atom 0))
         vconst-Atom (atom {})
         next-vconst-slot (atom (long (or (:vconst-base-slot opts) 10000)))
         ;; Create output loop RANGE UOps for each dimension of out-shape
         out-ranges (mapv (fn [dim]
                            (let [ax-id (swap! axis-counter inc)]
                              (if (== 1 (long dim))
                                (uop/uop-const :i32 0)
                                (uop/uop-range (long dim) ax-id :loop))))
                          out-shape)
         active-out-ranges (vec (filter #(= :uop/range (:op %)) out-ranges))
         memo (java.util.HashMap.)]
     (letfn [(lower-at [node coords]
               (let [coord-ids (mapv :id coords)
                     memo-key [(:id node) coord-ids]]
                 (if-let [cached (.get memo memo-key)]
                   cached
                   (let [{:keys [op dtype src arg shape]} node
                         res
                         (case op
                           ;; 1. Leaf Constants
                           :uop/const
                           (uop/uop-const dtype arg [])

                           ;; 2. Vector Constants (e.g. RoPE cos/sin tables, causal mask)
                           :uop/vconst
                           (let [slot (or (get @vconst-Atom (:id node))
                                          (let [s (swap! next-vconst-slot inc)]
                                            (swap! vconst-Atom assoc (:id node) {:slot s :values arg :dtype dtype :size (count arg)})
                                            s))
                                 actual-slot (if (map? slot) (:slot slot) slot)
                                 buf (uop/uop-buffer actual-slot dtype (count arg) :vconst)
                                 lin-idx (coords->linear-index coords shape)]
                             (uop/uop-load (uop/uop-index buf lin-idx)))

                           ;; 3. Parameter / Buffer Leaves -> explicit INDEX + LOAD
                           (:uop/param :uop/buffer)
                           (let [slot (:slot arg)
                                 n-elems (long (reduce * 1 (or shape [1])))
                                 buf (uop/uop-buffer slot dtype n-elems (:name arg))
                                 lin-idx (coords->linear-index coords shape)]
                             (uop/uop-load (uop/uop-index buf lin-idx)))

                           ;; 4. Movement Ops (Dissolved into pure coordinate transformations!)
                           :uop/expand
                           (let [in-node (first src)
                                 in-shape (:shape in-node)
                                 in-coords (mapv (fn [c in-d out-d]
                                                   (if (and (== 1 (long in-d)) (not= 1 (long out-d)))
                                                     (uop/uop-const :i32 0)
                                                     c))
                                                 coords in-shape shape)]
                             (lower-at in-node in-coords))

                           :uop/permute
                           (let [in-node (first src)
                                 perm arg
                                 ;; Output axis `i` came from input axis `perm[i]`, so input axis `k` gets `coords[inv_perm[k]]`
                                 inv-perm (rewrite/invert-permutation perm)
                                 in-coords (mapv #(nth coords %) inv-perm)]
                             (lower-at in-node in-coords))

                           :uop/shrink
                           (let [in-node (first src)
                                 bounds arg
                                 in-coords (mapv (fn [c [s _e]]
                                                   (if (zero? (long s))
                                                     c
                                                     (rewrite/simplify-uop
                                                      (uop/uop-alu :uop/add c (uop/uop-const :i32 (long s))))))
                                                 coords bounds)]
                             (lower-at in-node in-coords))

                           :uop/flip
                           (let [in-node (first src)
                                 flip-mask arg
                                 in-coords (mapv (fn [c flp? d]
                                                   (if flp?
                                                     (rewrite/simplify-uop
                                                      (uop/uop-alu :uop/sub (uop/uop-const :i32 (dec (long d))) c))
                                                     c))
                                                 coords flip-mask shape)]
                             (lower-at in-node in-coords))

                           :uop/pad
                           (let [in-node (first src)
                                 in-shape (:shape in-node)
                                 padding arg
                                 shifted-coords (mapv (fn [c [b _a]]
                                                        (if (zero? (long b))
                                                          c
                                                          (rewrite/simplify-uop
                                                           (uop/uop-alu :uop/sub c (uop/uop-const :i32 (long b))))))
                                                      coords padding)
                                 ;; Check bounds: for each padded axis, 0 <= shifted_c < in_dim
                                 valid-checks (keep (fn [[sc [b a] in-d]]
                                                      (when (or (pos? (long b)) (pos? (long a)))
                                                        (let [ge0 (uop/uop-alu :uop/cmpge sc (uop/uop-const :i32 0))
                                                              lt-d (uop/uop-alu :uop/cmplt sc (uop/uop-const :i32 (long in-d)))]
                                                          (uop/uop-alu :uop/and ge0 lt-d))))
                                                    (map vector shifted-coords padding in-shape))
                                 clamped-coords (mapv (fn [sc in-d]
                                                        (rewrite/simplify-uop
                                                         (uop/uop-alu :uop/min
                                                                      (uop/uop-alu :uop/max sc (uop/uop-const :i32 0))
                                                                      (uop/uop-const :i32 (dec (long in-d))))))
                                                      shifted-coords in-shape)
                                 inner-val (lower-at in-node clamped-coords)]
                             (if (empty? valid-checks)
                               inner-val
                               (let [gate (reduce #(uop/uop-alu :uop/and %1 %2) valid-checks)
                                     zero-val (uop/uop-const dtype 0)]
                                 (uop/uop-alu :uop/where gate inner-val zero-val))))

                           :uop/cat
                           (let [cat-axis (long arg)
                                 c-ax (nth coords cat-axis)
                                 ;; Walk concatenated children along cat-axis with cumulative offsets
                                 [_ final-expr]
                                 (reduce
                                  (fn [[offset acc-expr] child-uop]
                                    (let [child-dim (long (nth (:shape child-uop) cat-axis))
                                          next-offset (+ offset child-dim)
                                          shifted-c (if (zero? offset)
                                                      c-ax
                                                      (rewrite/simplify-uop
                                                       (uop/uop-alu :uop/sub c-ax (uop/uop-const :i32 offset))))
                                          clamped-c (rewrite/simplify-uop
                                                     (uop/uop-alu :uop/min
                                                                  (uop/uop-alu :uop/max shifted-c (uop/uop-const :i32 0))
                                                                  (uop/uop-const :i32 (dec child-dim))))
                                          child-coords (assoc coords cat-axis clamped-c)
                                          child-val (lower-at child-uop child-coords)]
                                      (if (nil? acc-expr)
                                        [next-offset child-val]
                                        (let [in-prev? (uop/uop-alu :uop/cmplt c-ax (uop/uop-const :i32 offset))]
                                          [next-offset (uop/uop-alu :uop/where in-prev? acc-expr child-val)]))))
                                  [0 nil]
                                  src)]
                             final-expr)

                           :uop/reshape
                           (let [in-node (first src)
                                 in-shape (:shape in-node)]
                             (if (= in-shape shape)
                               (lower-at in-node coords)
                               (let [lin-idx (coords->linear-index coords shape)
                                     in-coords (linear-index->coords lin-idx in-shape)]
                                 (lower-at in-node in-coords))))

                           ;; 5. Gather (:uop/index with {:op :gather})
                           :uop/index
                           (if (= :gather (:op arg))
                             (let [[w-uop idx-uop] src
                                   ;; coords has shape [b..., p..., d] where last coord is embedding col `d`
                                   idx-coords (subvec coords 0 (dec (count coords)))
                                   d-coord (last coords)
                                   tok-id (lower-at idx-uop idx-coords)
                                   hidden-dim (long (last (:shape w-uop)))
                                   ;; Compute w[tok_id * hidden_dim + d_coord]
                                   row-offset (uop/uop-alu :uop/mul tok-id (uop/uop-const :i32 hidden-dim))
                                   w-lin-idx (rewrite/simplify-uop (uop/uop-alu :uop/add row-offset d-coord))
                                   w-slot (:slot (:arg w-uop))
                                   w-size (long (reduce * 1 (:shape w-uop)))
                                   w-buf (uop/uop-buffer w-slot (:dtype w-uop) w-size (:name (:arg w-uop)))]
                               (uop/uop-load (uop/uop-index w-buf w-lin-idx)))
                             node)

                           ;; 6. Reduction (:uop/reduce) -> introduces `:reduce` RANGE loops!
                           :uop/reduce
                           (let [in-node (first src)
                                 in-shape (:shape in-node)
                                 {:keys [axes] red-op :op} arg
                                 axes-set (set axes)
                                 red-ranges (atom [])
                                 in-coords (mapv (fn [ax-idx in-d c]
                                                   (if (contains? axes-set ax-idx)
                                                     (if (== 1 (long in-d))
                                                       (uop/uop-const :i32 0)
                                                       (let [r-id (swap! axis-counter inc)
                                                             r-uop (uop/uop-range (long in-d) r-id :reduce)]
                                                         (swap! red-ranges conj r-uop)
                                                         r-uop))
                                                     c))
                                                 (range (count in-shape))
                                                 in-shape
                                                 coords)
                                 body-val (lower-at in-node in-coords)]
                             (if (empty? @red-ranges)
                               body-val
                               (uop/uop :uop/reduce-axis dtype (into [body-val] @red-ranges) {:op red-op} [])))

                           ;; 7. Elementwise ALU & Cast Ops
                           :uop/cast
                           (let [in-val (lower-at (first src) coords)]
                             (uop/uop-cast in-val arg))

                           ;; Default: pure elementwise ALU op -> propagate coords to all srcs
                           (let [lowered-srcs (mapv #(lower-at % coords) src)]
                             (rewrite/simplify-uop
                              (uop/uop op dtype lowered-srcs arg []))))]
                     (.put memo memo-key res)
                     res))))]
       (let [val-expr (lower-at root-uop out-ranges)
             out-buf (uop/uop-buffer out-slot out-dtype out-size :out)
             out-lin-idx (coords->linear-index out-ranges out-shape)
             out-idx-uop (uop/uop-index out-buf out-lin-idx)
             store-uop (uop/uop-store out-idx-uop val-expr active-out-ranges)
             sink-uop (uop/uop :uop/sink :void [store-uop] {:out-shape out-shape :out-dtype out-dtype :out-slot out-slot} [])]
         {:sink sink-uop
          :store store-uop
          :val-expr val-expr
          :out-ranges out-ranges
          :active-out-ranges active-out-ranges
          :out-shape out-shape
          :out-dtype out-dtype
          :out-size out-size
          :out-slot out-slot
          :vconsts (vals @vconst-Atom)})))))

;; -----------------------------------------------------------------------------
;; 3. Reference Evaluator for Rangeified Kernels (for TDD & Verification)
;; -----------------------------------------------------------------------------

(defn- read-buf-elem [buf idx]
  (let [i (int idx)]
    (cond
      (instance? (Class/forName "[F") buf) (double (aget ^floats buf i))
      (instance? (Class/forName "[I") buf) (long (aget ^ints buf i))
      (instance? (Class/forName "[D") buf) (double (aget ^doubles buf i))
      (vector? buf) (nth buf i)
      :else (double (nth buf i)))))

(defn- write-buf-elem! [buf idx val dtype]
  (let [i (int idx)]
    (if (contains? #{:i32 :i64 :i8} dtype)
      (aset ^ints buf i (int val))
      (aset ^floats buf i (float val)))))

(defn eval-rangeified-kernel
  "Evaluates a rangeified kernel `rk` on a slot->array map `slot-buffers`.
   Executes the explicit `RANGE` loops and `INDEX`/`LOAD`/`STORE` UOps directly."
  [rk slot-buffers]
  (let [{:keys [val-expr active-out-ranges out-ranges out-shape out-dtype out-size out-slot vconsts]} rk
        out-buf (or (get slot-buffers out-slot)
                    (if (contains? #{:i32 :i64 :i8} out-dtype)
                      (int-array out-size)
                      (float-array out-size)))
        vconst-map (into {} (map (fn [{:keys [slot values]}] [slot (vec values)]) vconsts))
        all-buffers (merge vconst-map (assoc slot-buffers out-slot out-buf))
        out-lin-idx-uop (coords->linear-index out-ranges out-shape)]
    (letfn [(eval-node [node range-env]
              (if (contains? range-env node)
                (get range-env node)
                (let [{:keys [op src arg]} node]
                  (case op
                    :uop/const arg
                    :uop/load
                    (let [idx-uop (first src)
                          buf-uop (first (:src idx-uop))
                          lin-uop (second (:src idx-uop))
                          slot (:slot (:arg buf-uop))
                          idx-val (eval-node lin-uop range-env)
                          buf (get all-buffers slot)]
                      (read-buf-elem buf idx-val))

                    :uop/reduce-axis
                    (let [body-uop (first src)
                          r-uops (rest src)
                          red-op (:op arg)
                          init-val (case red-op
                                     :max Double/NEGATIVE_INFINITY
                                     :min Double/POSITIVE_INFINITY
                                     0.0)]
                      (letfn [(loop-red [rem-ranges cur-env]
                                (if (empty? rem-ranges)
                                  (double (eval-node body-uop cur-env))
                                  (let [r (first rem-ranges)
                                        bound (long (eval-node (first (:src r)) cur-env))]
                                    (loop [i 0
                                           acc init-val]
                                      (if (< i bound)
                                        (let [v (loop-red (rest rem-ranges) (assoc cur-env r i))
                                              next-acc (case red-op
                                                         :max (Math/max acc v)
                                                         :min (Math/min acc v)
                                                         (+ acc v))]
                                          (recur (inc i) next-acc))
                                        acc)))))]
                        (loop-red r-uops range-env)))

                    ;; Scalar ALU ops
                    (let [child-vals (mapv #(eval-node % range-env) src)
                          dummy-env (zipmap src child-vals)]
                      (rewrite/eval-scalar-uop node dummy-env))))))

            (run-out-loops [rem-ranges cur-env]
              (if (empty? rem-ranges)
                (let [v (eval-node val-expr cur-env)
                      out-idx (eval-node out-lin-idx-uop cur-env)]
                  (write-buf-elem! out-buf out-idx v out-dtype))
                (let [r (first rem-ranges)
                      bound (long (eval-node (first (:src r)) cur-env))]
                  (dotimes [i bound]
                    (run-out-loops (rest rem-ranges) (assoc cur-env r i))))))]
      (run-out-loops active-out-ranges {})
      out-buf)))
