(ns einsum.uop.rewrite
  "Bottom-up greedy UOp graph rewrite engine, constant folding, symbolic div/mod algebra,
   and Movement Op canonicalization.
   Modeled on tinygrad's symbolic simplifier and Compilers for Machine Learning (Week 1)."
  (:require [einsum.uop.core :as uop]))

;; -----------------------------------------------------------------------------
;; 1. Scalar Evaluator (for Constant Folding & Generative Property Testing)
;; -----------------------------------------------------------------------------

(defn eval-scalar-uop
  "Evaluates a scalar/elementwise UOp DAG given an `env` map of `{uop-node -> scalar-val}`."
  ([node]
   (eval-scalar-uop node {}))
  ([node env]
   (if (contains? env node)
     (get env node)
     (let [{:keys [op src arg]} node
           vals (mapv #(eval-scalar-uop % env) src)]
       (case op
         :uop/const arg
         :uop/neg (- (first vals))
         :uop/recip (/ 1.0 (double (first vals)))
         :uop/exp2 (Math/pow 2.0 (double (first vals)))
         :uop/log2 (/ (Math/log (double (first vals))) (Math/log 2.0))
         :uop/sin (Math/sin (double (first vals)))
         :uop/cos (Math/cos (double (first vals)))
         :uop/sqrt (Math/sqrt (double (first vals)))
         :uop/trunc (long (first vals))
         :uop/cast (let [v (first vals)]
                     (if (contains? #{:f16 :bf16 :f32 :f64} arg)
                       (double v)
                       (long v)))
         :uop/add (+ (first vals) (second vals))
         :uop/sub (- (first vals) (second vals))
         :uop/mul (* (first vals) (second vals))
         :uop/div (/ (double (first vals)) (double (second vals)))
         :uop/idiv (quot (long (first vals)) (long (second vals)))
         :uop/mod (rem (long (first vals)) (long (second vals)))
         :uop/max (max (first vals) (second vals))
         :uop/min (min (first vals) (second vals))
         :uop/shl (bit-shift-left (long (first vals)) (long (second vals)))
         :uop/shr (unsigned-bit-shift-right (long (first vals)) (long (second vals)))
         :uop/and (bit-and (long (first vals)) (long (second vals)))
         :uop/or (bit-or (long (first vals)) (long (second vals)))
         :uop/xor (bit-xor (long (first vals)) (long (second vals)))
         :uop/cmplt (< (first vals) (second vals))
         :uop/cmple (<= (first vals) (second vals))
         :uop/cmpeq (= (first vals) (second vals))
         :uop/cmpne (not= (first vals) (second vals))
         :uop/cmpgt (> (first vals) (second vals))
         :uop/cmpge (>= (first vals) (second vals))
         :uop/where (if (first vals) (second vals) (nth vals 2))
         (throw (ex-info "Cannot evaluate non-scalar or unbound UOp in eval-scalar-uop"
                         {:op op :node node})))))))

;; -----------------------------------------------------------------------------
;; 2. Bounded Range Analysis (vmin / vmax) for Div/Mod Index Simplification
;; -----------------------------------------------------------------------------

(defn uop-bounds
  "Computes conservative integer `[vmin vmax]` bounds for an integer UOp node.
   Returns `[min-long max-long]` when unbounded."
  [node]
  (case (:op node)
    :uop/const
    (let [v (long (:arg node))]
      [v v])

    :uop/range
    (let [[b-min b-max] (uop-bounds (first (:src node)))]
      (if (= b-min b-max)
        [0 (max 0 (dec b-max))]
        [0 Long/MAX_VALUE]))

    :uop/add
    (let [[a0 a1] (uop-bounds (first (:src node)))
          [b0 b1] (uop-bounds (second (:src node)))]
      (if (and (< a1 1000000000) (< b1 1000000000) (> a0 -1000000000) (> b0 -1000000000))
        [(+ a0 b0) (+ a1 b1)]
        [Long/MIN_VALUE Long/MAX_VALUE]))

    :uop/mul
    (let [[a0 a1] (uop-bounds (first (:src node)))
          [b0 b1] (uop-bounds (second (:src node)))]
      (if (and (>= a0 0) (>= b0 0) (< a1 1000000) (< b1 1000000))
        [(* a0 b0) (* a1 b1)]
        [Long/MIN_VALUE Long/MAX_VALUE]))

    :uop/mod
    (let [[_ b1] (uop-bounds (second (:src node)))]
      (if (and (> b1 0) (< b1 1000000000))
        [0 (dec b1)]
        [0 Long/MAX_VALUE]))

    :uop/idiv
    (let [[a0 a1] (uop-bounds (first (:src node)))
          [b0 b1] (uop-bounds (second (:src node)))]
      (if (and (>= a0 0) (= b0 b1) (pos? b0) (< a1 1000000000))
        [(quot a0 b0) (quot a1 b0)]
        [Long/MIN_VALUE Long/MAX_VALUE]))

    [Long/MIN_VALUE Long/MAX_VALUE]))

;; -----------------------------------------------------------------------------
;; 3. Single-Step Algebraic, Div/Mod, and Movement Rewrite Rules
;; -----------------------------------------------------------------------------

(defn- const? [n] (= :uop/const (:op n)))
(defn- zero-const? [n] (and (const? n) (number? (:arg n)) (zero? (:arg n))))
(defn- one-const? [n] (and (const? n) (number? (:arg n)) (== 1 (:arg n))))

(defn- invert-perm [perm]
  (let [inv (int-array (count perm))]
    (doseq [[i p] (map-indexed vector perm)]
      (aset inv (int p) (int i)))
    (vec inv)))

(defn- compose-perms [p1 p2]
  (mapv #(nth p1 %) p2))

(defn- rewrite-step
  "Applies one local rewrite rule to `node` (assuming its children are already simplified).
   Returns a simplified UOp or `node` if no rule matches."
  [node]
  (let [{:keys [op dtype src arg shape]} node
        [s0 s1 s2] src]
    (cond
      ;; 1. Constant Folding on pure ALU ops with all-constant inputs
      (and (contains? uop/ALU_OPS op)
           (seq src)
           (every? const? src)
           (not (and (contains? #{:uop/idiv :uop/mod :uop/div} op)
                     (zero-const? s1))))
      (let [val (eval-scalar-uop node)]
        (uop/uop-const dtype val shape))

      ;; 2. Addition identities: x + 0 -> x, 0 + x -> x
      (= op :uop/add)
      (cond
        (zero-const? s1) s0
        (zero-const? s0) s1
        ;; (a + c1) + c2 -> a + (c1 + c2)
        (and (= :uop/add (:op s0)) (const? (second (:src s0))) (const? s1))
        (uop/uop :uop/add dtype [(first (:src s0))
                                 (uop/uop-const dtype (+ (:arg (second (:src s0))) (:arg s1)) shape)]
                 nil shape)
        :else node)

      ;; 3. Subtraction identities: x - 0 -> x, x - x -> 0
      (= op :uop/sub)
      (cond
        (zero-const? s1) s0
        (= s0 s1) (uop/uop-const dtype 0 shape)
        :else node)

      ;; 4. Multiplication identities: x * 1 -> x, 1 * x -> x, x * 0 -> 0, 0 * x -> 0
      (= op :uop/mul)
      (cond
        (one-const? s1) s0
        (one-const? s0) s1
        (zero-const? s1) (uop/uop-const dtype 0 shape)
        (zero-const? s0) (uop/uop-const dtype 0 shape)
        ;; (a * c1) * c2 -> a * (c1 * c2)
        (and (= :uop/mul (:op s0)) (const? (second (:src s0))) (const? s1))
        (uop/uop :uop/mul dtype [(first (:src s0))
                                 (uop/uop-const dtype (* (:arg (second (:src s0))) (:arg s1)) shape)]
                 nil shape)
        :else node)

      ;; 5. Integer Division (idiv) identities & rangeify index algebra
      (= op :uop/idiv)
      (cond
        (one-const? s1) s0
        (zero-const? s0) (uop/uop-const dtype 0 shape)
        (= s0 s1) (uop/uop-const dtype 1 shape)

        ;; If 0 <= s0 < k, then s0 // k -> 0
        (and (const? s1) (pos? (long (:arg s1)))
             (let [[vmin vmax] (uop-bounds s0)]
               (and (>= vmin 0) (< vmax (long (:arg s1))))))
        (uop/uop-const dtype 0 shape)

        ;; (a * k) // k -> a
        (and (= :uop/mul (:op s0))
             (= (second (:src s0)) s1)
             (const? s1) (pos? (long (:arg s1))))
        (first (:src s0))

        ;; ((a * k) + b) // k -> a + (b // k) when a >= 0 and b >= 0
        (and (= :uop/add (:op s0))
             (const? s1) (pos? (long (:arg s1))))
        (let [[lhs rhs] (:src s0)
              k (long (:arg s1))]
          (cond
            (and (= :uop/mul (:op lhs))
                 (const? (second (:src lhs)))
                 (zero? (rem (long (:arg (second (:src lhs)))) k))
                 (>= (first (uop-bounds rhs)) 0))
            (let [mult (quot (long (:arg (second (:src lhs)))) k)
                  term-a (if (= mult 1)
                           (first (:src lhs))
                           (uop/uop-alu :uop/mul (first (:src lhs)) (uop/uop-const dtype mult shape)))
                  term-b (uop/uop-alu :uop/idiv rhs s1)]
              (uop/uop-alu :uop/add term-a term-b))

            (and (= :uop/mul (:op rhs))
                 (const? (second (:src rhs)))
                 (zero? (rem (long (:arg (second (:src rhs)))) k))
                 (>= (first (uop-bounds lhs)) 0))
            (let [mult (quot (long (:arg (second (:src rhs)))) k)
                  term-b (if (= mult 1)
                           (first (:src rhs))
                           (uop/uop-alu :uop/mul (first (:src rhs)) (uop/uop-const dtype mult shape)))
                  term-a (uop/uop-alu :uop/idiv lhs s1)]
              (uop/uop-alu :uop/add term-a term-b))

            :else node))

        ;; (a // c1) // c2 -> a // (c1 * c2)
        (and (= :uop/idiv (:op s0)) (const? (second (:src s0))) (const? s1))
        (uop/uop :uop/idiv dtype [(first (:src s0))
                                  (uop/uop-const dtype (* (long (:arg (second (:src s0)))) (long (:arg s1))) shape)]
                 nil shape)

        :else node)

      ;; 6. Integer Modulo (mod) identities & rangeify index algebra
      (= op :uop/mod)
      (cond
        (one-const? s1) (uop/uop-const dtype 0 shape)
        (zero-const? s0) (uop/uop-const dtype 0 shape)
        (= s0 s1) (uop/uop-const dtype 0 shape)

        ;; If 0 <= s0 < k, then s0 % k -> s0
        (and (const? s1) (pos? (long (:arg s1)))
             (let [[vmin vmax] (uop-bounds s0)]
               (and (>= vmin 0) (< vmax (long (:arg s1))))))
        s0

        ;; (a * c) % k -> 0 when c is a multiple of k
        (and (= :uop/mul (:op s0))
             (const? (second (:src s0)))
             (const? s1) (pos? (long (:arg s1)))
             (zero? (rem (long (:arg (second (:src s0)))) (long (:arg s1)))))
        (uop/uop-const dtype 0 shape)

        ;; ((a * c) + b) % k -> b % k when c is a multiple of k and a, b >= 0
        (and (= :uop/add (:op s0))
             (const? s1) (pos? (long (:arg s1))))
        (let [[lhs rhs] (:src s0)
              k (long (:arg s1))]
          (cond
            (and (= :uop/mul (:op lhs))
                 (const? (second (:src lhs)))
                 (zero? (rem (long (:arg (second (:src lhs)))) k))
                 (>= (first (uop-bounds rhs)) 0))
            (uop/uop-alu :uop/mod rhs s1)

            (and (= :uop/mul (:op rhs))
                 (const? (second (:src rhs)))
                 (zero? (rem (long (:arg (second (:src rhs)))) k))
                 (>= (first (uop-bounds lhs)) 0))
            (uop/uop-alu :uop/mod lhs s1)

            :else node))

        :else node)

      ;; 7. Where / Select identities
      (= op :uop/where)
      (cond
        (and (const? s0) (true? (:arg s0))) s1
        (and (const? s0) (false? (:arg s0))) s2
        (= s1 s2) s1
        :else node)

      ;; 8. Movement Op Canonicalizations
      ;; reshape(reshape(x, s1), s2) -> reshape(x, s2)
      (= op :uop/reshape)
      (cond
        (= (:shape s0) arg) s0
        (= :uop/reshape (:op s0)) (uop/uop-reshape (first (:src s0)) arg)
        :else node)

      ;; permute(permute(x, p1), p2) -> permute(x, compose(p1, p2))
      (= op :uop/permute)
      (cond
        (= arg (vec (range (count arg)))) s0
        (= :uop/permute (:op s0))
        (uop/uop-permute (first (:src s0)) (compose-perms (:arg s0) arg))
        :else node)

      ;; expand(expand(x, s1), s2) -> expand(x, s2)
      (= op :uop/expand)
      (cond
        (= (:shape s0) arg) s0
        (= :uop/expand (:op s0)) (uop/uop-expand (first (:src s0)) arg)
        :else node)

      ;; shrink(shrink(x, b1), b2) -> shrink(x, merged-bounds)
      (= op :uop/shrink)
      (if (= :uop/shrink (:op s0))
        (let [b1 (:arg s0)
              b2 arg
              merged (mapv (fn [[s1 _e1] [s2 e2]] [(+ s1 s2) (+ s1 e2)]) b1 b2)]
          (uop/uop-shrink (first (:src s0)) merged))
        node)

      ;; cast(cast(x, t1), t2) -> cast(x, t2) when lossless or identical
      (= op :uop/cast)
      (if (= (:dtype s0) arg)
        s0
        node)

      :else node)))

;; -----------------------------------------------------------------------------
;; 4. Bottom-Up Fixed-Point Graph Rewriter
;; -----------------------------------------------------------------------------

(defn simplify-uop
  "Bottom-up greedy UOp graph rewriter with memoization.
   Repeatedly simplifies children and applies local rewrite rules until fixed point."
  [root]
  (let [memo (java.util.HashMap.)]
    (letfn [(simplify-node [node]
              (if (.containsKey memo (:id node))
                (.get memo (:id node))
                (let [new-src (mapv simplify-node (:src node))
                      rebuilt (if (= new-src (:src node))
                                node
                                (uop/uop (:op node) (:dtype node) new-src (:arg node) (:shape node)))
                      stepped (rewrite-step rebuilt)
                      final-node (if (identical? stepped rebuilt)
                                   stepped
                                   (simplify-node stepped))]
                  (.put memo (:id node) final-node)
                  final-node)))]
      (simplify-node root))))

(defn invert-permutation
  "Returns the inverse permutation vector `inv-perm` such that `(nth (permute v perm) i) == (nth v (nth inv-perm i))`."
  [perm]
  (invert-perm perm))
