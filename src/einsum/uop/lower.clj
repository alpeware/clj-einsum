(ns einsum.uop.lower
  "Lowering engine from normalized Tensor Logic AST to High-Level UOp DAG.
   Translates Einsum contractions, activations, norms, RoPE, causal-softmax, gather, and slice
   into pure compositions of tinyspec Movement Ops + ALU Ops + Reduce Ops."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [einsum.logic.ast :as ast]
            [einsum.logic.dce :as dce]
            [einsum.logic.expand :as expand]
            [einsum.logic.index :as idx]
            [einsum.logic.shape :as shape]
            [einsum.runtime.weights :as weights]
            [einsum.uop.core :as uop]
            [einsum.uop.rewrite :as rewrite]))

(def ^:private LOG2_E (/ 1.0 (Math/log 2.0)))
(def ^:private LN_2 (Math/log 2.0))

;; -----------------------------------------------------------------------------
;; 1. Broadcast & Movement Alignment Helpers
;; -----------------------------------------------------------------------------

(defn- indices->dim-numbers [ordered-idxs target-idxs]
  (let [idx->pos (into {} (map-indexed (fn [pos k] [k pos]) ordered-idxs))]
    (mapv #(get idx->pos %) target-idxs)))

(defn- broadcast-scalar
  "Broadcasts scalar constant `val` of `dtype` to `target-shape` via reshape + expand."
  [dtype val target-shape]
  (let [c (uop/uop-const dtype val [])
        rank (count target-shape)]
    (if (zero? rank)
      c
      (-> c
          (uop/uop-reshape (vec (repeat rank 1)))
          (uop/uop-expand target-shape)))))

(declare align-tensor-indices)

(defn- align-and-broadcast
  "Given a tensor UOp `t-uop` with symbolic indices `t-idxs`, aligns and broadcasts it
   to `target-idxs` with `target-shape` using only `:uop/permute`, `:uop/reshape`, and `:uop/expand`."
  [t-uop t-idxs target-idxs target-shape]
  (cond
    (= t-idxs target-idxs)
    t-uop

    ;; Empty indices (scalar)
    (empty? t-idxs)
    (let [rank (count target-shape)]
      (-> t-uop
          (uop/uop-reshape (vec (repeat rank 1)))
          (uop/uop-expand target-shape)))

    ;; Same index set, just a permutation
    (= (set t-idxs) (set target-idxs))
    (let [perm (indices->dim-numbers t-idxs target-idxs)]
      (uop/uop-permute t-uop perm))

    ;; Scalar / 1-element tensor can broadcast to any target shape
    (== 1 (reduce * 1 (or (:shape t-uop) [])))
    (-> t-uop
        (uop/uop-reshape (vec (repeat (count target-shape) 1)))
        (uop/uop-expand target-shape))

    ;; Subset of target-idxs: permute to target relative order, reshape with 1s, expand to target-shape
    (set/subset? (set t-idxs) (set target-idxs))
    (let [t-set (set t-idxs)
          ordered-present (vec (filter #(contains? t-set %) target-idxs))
          permuted (if (= t-idxs ordered-present)
                     t-uop
                     (uop/uop-permute t-uop (indices->dim-numbers t-idxs ordered-present)))
          perm-shape (:shape permuted)
          idx->dim (zipmap ordered-present perm-shape)
          reshaped-1s (mapv #(get idx->dim % 1) target-idxs)]
      (-> permuted
          (uop/uop-reshape reshaped-1s)
          (uop/uop-expand target-shape)))

    ;; General case with singleton dummy axes (e.g. [:one] or size-1 axes not in target-idxs)
    (let [dim-map (zipmap t-idxs (:shape t-uop))
          non-singleton-idxs (vec (remove #(= 1 (get dim-map % 0)) t-idxs))]
      (set/subset? (set non-singleton-idxs) (set target-idxs)))
    (let [dim-map (zipmap t-idxs (:shape t-uop))
          non-singleton-idxs (vec (remove #(= 1 (get dim-map % 0)) t-idxs))
          non-singleton-shape (mapv #(get dim-map %) non-singleton-idxs)
          squeezed (uop/uop-reshape t-uop non-singleton-shape)]
      (align-tensor-indices squeezed non-singleton-idxs target-shape target-idxs))

    :else
    (throw (ex-info "Cannot align tensor indices to target indices"
                    {:tensor-idxs t-idxs :target-idxs target-idxs :target-shape target-shape}))))

(defn- broadcast-shape-suffix
  "Broadcasts a lower-rank UOp `x` (e.g. `[D]` or `[B, S, 1]`) to `target-shape`."
  [x target-shape]
  (let [in-shape (:shape x)]
    (cond
      (= in-shape target-shape)
      x

      (empty? in-shape)
      (-> x
          (uop/uop-reshape (vec (repeat (count target-shape) 1)))
          (uop/uop-expand target-shape))

      (= (count in-shape) (count target-shape))
      (uop/uop-expand x target-shape)

      (< (count in-shape) (count target-shape))
      (let [pad-rank (- (count target-shape) (count in-shape))
            padded-shape (into (vec (repeat pad-rank 1)) in-shape)]
        (-> x
            (uop/uop-reshape padded-shape)
            (uop/uop-expand target-shape)))

      :else
      (uop/uop-reshape x target-shape))))

;; -----------------------------------------------------------------------------
;; 2. Mathematical & Activation Composites in Pure UOps (exp, tanh, sigmoid, gelu, silu)
;; -----------------------------------------------------------------------------

(defn uop-exp
  "Natural exponential $e^x = 2^{x \\cdot \\log_2(e)}$ in pure UOps."
  [x]
  (let [dt (:dtype x)
        sh (:shape x)
        log2e (broadcast-scalar dt LOG2_E sh)]
    (uop/uop-alu :uop/exp2 (uop/uop-alu :uop/mul x log2e))))

(defn uop-log
  "Natural logarithm $\\ln(x) = \\log_2(x) \\cdot \\ln(2)$ in pure UOps."
  [x]
  (let [dt (:dtype x)
        sh (:shape x)
        ln2 (broadcast-scalar dt LN_2 sh)]
    (uop/uop-alu :uop/mul (uop/uop-alu :uop/log2 x) ln2)))

(defn uop-sigmoid
  "Logistic sigmoid $\\sigma(x) = \\frac{1}{1 + e^{-x}}$ in pure UOps."
  [x]
  (let [dt (:dtype x)
        sh (:shape x)
        c-min (broadcast-scalar dt -80.0 sh)
        c-max (broadcast-scalar dt 80.0 sh)
        x-clamped (uop/uop-alu :uop/max (uop/uop-alu :uop/min x c-max) c-min)
        one (broadcast-scalar dt 1.0 sh)
        neg-x (uop/uop-alu :uop/neg x-clamped)
        exp-neg (uop-exp neg-x)
        denom (uop/uop-alu :uop/add one exp-neg)]
    (uop/uop-alu :uop/recip denom)))

(defn uop-tanh
  "Hyperbolic tangent $\\tanh(x) = 2\\sigma(2x) - 1$ in pure UOps."
  [x]
  (let [dt (:dtype x)
        sh (:shape x)
        two (broadcast-scalar dt 2.0 sh)
        one (broadcast-scalar dt 1.0 sh)
        sig2x (uop-sigmoid (uop/uop-alu :uop/mul x two))]
    (uop/uop-alu :uop/sub (uop/uop-alu :uop/mul two sig2x) one)))

(defn- apply-post-activation
  "Applies optional `:scale`, `:act` (`:gelu`, `:silu`, `:sigmoid`, `:tanh`, `:relu`, `:clamp`),
   or `:softcap` to `in-uop`."
  [in-uop attrs]
  (let [{:keys [scale act softcap clamp]} attrs
        dt (:dtype in-uop)
        sh (:shape in-uop)
        scaled (if scale
                 (uop/uop-alu :uop/mul in-uop (broadcast-scalar dt (double scale) sh))
                 in-uop)
        activated (cond
                    (or clamp (= act :clamp))
                    (let [[low high] (if (sequential? clamp) (vec clamp) [0.0 1.0])
                          c-low (broadcast-scalar dt (double low) sh)
                          c-high (broadcast-scalar dt (double high) sh)
                          clamped-low (uop/uop-alu :uop/max scaled c-low)]
                      (uop/uop-alu :uop/min clamped-low c-high))

                    (= act :relu)
                    (uop/uop-alu :uop/max scaled (broadcast-scalar dt 0.0 sh))

                    (or (= act :sigmoid) (= act :logistic))
                    (uop-sigmoid scaled)

                    (or (= act :silu) (= act :swish))
                    (uop/uop-alu :uop/mul scaled (uop-sigmoid scaled))

                    (= act :tanh)
                    (uop-tanh scaled)

                    (= act :gelu)
                    (let [c-half (broadcast-scalar dt 0.5 sh)
                          c-sqrt (broadcast-scalar dt 0.7978845608 sh)
                          c-poly (broadcast-scalar dt 0.044715 sh)
                          c-one  (broadcast-scalar dt 1.0 sh)
                          x2 (uop/uop-alu :uop/mul scaled scaled)
                          x3 (uop/uop-alu :uop/mul x2 scaled)
                          poly (uop/uop-alu :uop/mul x3 c-poly)
                          inner (uop/uop-alu :uop/add scaled poly)
                          s-inner (uop/uop-alu :uop/mul inner c-sqrt)
                          th (uop-tanh s-inner)
                          plus-one (uop/uop-alu :uop/add th c-one)
                          half-x (uop/uop-alu :uop/mul scaled c-half)]
                      (uop/uop-alu :uop/mul half-x plus-one))

                    :else scaled)]
    (if softcap
      (let [cap (double softcap)
            c-cap (broadcast-scalar dt cap sh)
            c-inv (broadcast-scalar dt (/ 1.0 cap) sh)
            scaled-down (uop/uop-alu :uop/mul activated c-inv)
            tanhed (uop-tanh scaled-down)]
        (uop/uop-alu :uop/mul tanhed c-cap))
      activated)))

;; -----------------------------------------------------------------------------
;; 3. Contraction & Hook Lowering Rules
;; -----------------------------------------------------------------------------

(defn- lower-binary-contraction
  "Lowers a binary Tensor Logic equation `[:= head attrs lhs rhs]` into Movement + ALU + Reduce UOps."
  [head attrs lhs rhs env known-shapes]
  (let [head-name (first head)
        head-idxs (vec (rest head))
        lhs-name (first lhs)
        lhs-idxs (vec (rest lhs))
        rhs-name (first rhs)
        rhs-idxs (vec (rest rhs))
        lhs-uop (get env lhs-name)
        rhs-uop (get env rhs-name)
        {:keys [contracting batch lhs-free rhs-free]}
        (idx/partition-indices lhs-idxs rhs-idxs head-idxs)
        semiring (:semiring attrs)
        [mul-op red-op] (cond
                          (or (= semiring :min-plus) (= semiring :tropical)) [:uop/add :min]
                          (or (= semiring :max-product) (= semiring :viterbi)) [:uop/mul :max]
                          (or (= (:op attrs) :add) (= (:op attrs) :+) (:add? attrs)) [:uop/add :add]
                          :else [:uop/mul :add])]
    (if (empty? contracting)
      ;; Elementwise / broadcast product or addition
      (let [target-shape (get known-shapes head-name)
            lhs-aligned (align-and-broadcast lhs-uop lhs-idxs head-idxs target-shape)
            rhs-aligned (align-and-broadcast rhs-uop rhs-idxs head-idxs target-shape)
            prod (uop/uop-alu mul-op lhs-aligned rhs-aligned)]
        (apply-post-activation prod attrs))
      ;; General Einsum Contraction:
      ;; 1. Construct full product space: [batch..., lhs-free..., rhs-free..., contracting...]
      (let [free-idxs (vec (concat batch lhs-free rhs-free))
            all-idxs (vec (concat free-idxs contracting))
            lhs-shape (get known-shapes lhs-name (:shape lhs-uop))
            rhs-shape (get known-shapes rhs-name (:shape rhs-uop))
            idx->dim (into (zipmap lhs-idxs lhs-shape) (zipmap rhs-idxs rhs-shape))
            all-shape (mapv idx->dim all-idxs)
            free-shape (mapv idx->dim free-idxs)
            lhs-bcast (align-and-broadcast lhs-uop lhs-idxs all-idxs all-shape)
            rhs-bcast (align-and-broadcast rhs-uop rhs-idxs all-idxs all-shape)
            prod (uop/uop-alu mul-op lhs-bcast rhs-bcast)
            reduce-axes (vec (range (count free-idxs) (count all-idxs)))
            ;; Reduce along contracting axes and squeeze them out to free-shape
            reduced (uop/uop-reduce prod red-op reduce-axes false)
            reshaped-free (uop/uop-reshape reduced free-shape)
            ;; Permute free-idxs to match head-idxs if needed
            permuted (if (= free-idxs head-idxs)
                       reshaped-free
                       (uop/uop-permute reshaped-free (indices->dim-numbers free-idxs head-idxs)))]
        (apply-post-activation permuted attrs)))))

(defn- lower-unary-equation
  "Lowers a unary Tensor Logic equation `[:= head attrs body]` (transpose, broadcast, or reduction)."
  [head attrs body env known-shapes]
  (let [head-name (first head)
        head-idxs (vec (rest head))
        body-name (first body)
        body-idxs (vec (rest body))
        body-uop (get env body-name)
        target-shape (or (:shape attrs) (get known-shapes head-name) (:shape body-uop))
        contracting (vec (remove (set head-idxs) body-idxs))]
    (if (seq contracting)
      ;; Unary reduction over contracting indices
      (let [red-op (case (:op attrs)
                     (:max :maximum) :max
                     (:min :minimum) :min
                     :add)
            reduce-axes (indices->dim-numbers body-idxs contracting)
            rem-idxs (vec (remove (set contracting) body-idxs))
            reduced (uop/uop-reduce body-uop red-op reduce-axes false)
            permuted (if (= rem-idxs head-idxs)
                       reduced
                       (uop/uop-permute reduced (indices->dim-numbers rem-idxs head-idxs)))]
        (apply-post-activation permuted attrs))
      ;; Unary transpose, broadcast, or identity + post-activation
      (let [aligned (if (and (seq body-idxs) (seq head-idxs))
                      (align-and-broadcast body-uop body-idxs head-idxs target-shape)
                      (broadcast-shape-suffix body-uop target-shape))]
        (apply-post-activation aligned attrs)))))

(defn- lower-layer-norm
  "Lowers `[:layer-norm head x gamma beta {:eps eps}]` into pure UOps."
  [x-uop gamma-uop beta-uop attrs]
  (let [eps (double (or (:eps attrs) 1e-5))
        dt (:dtype x-uop)
        sh (:shape x-uop)
        last-axis (dec (count sh))
        d (double (last sh))
        inv-d (broadcast-scalar dt (/ 1.0 d) sh)
        ;; mean = sum(x, axis=-1, keep_dims=true) * (1/D)
        sum-x (uop/uop-reduce x-uop :add [last-axis] true)
        mean-x (uop/uop-alu :uop/mul (uop/uop-expand sum-x sh) inv-d)
        diff (uop/uop-alu :uop/sub x-uop mean-x)
        diff-sq (uop/uop-alu :uop/mul diff diff)
        sum-sq (uop/uop-reduce diff-sq :add [last-axis] true)
        var-x (uop/uop-alu :uop/mul (uop/uop-expand sum-sq sh) inv-d)
        var-eps (uop/uop-alu :uop/add var-x (broadcast-scalar dt eps sh))
        rstd (uop/uop-alu :uop/recip (uop/uop-alu :uop/sqrt var-eps))
        normed (uop/uop-alu :uop/mul diff rstd)
        scaled (if gamma-uop
                 (uop/uop-alu :uop/mul normed (broadcast-shape-suffix gamma-uop sh))
                 normed)]
    (if beta-uop
      (uop/uop-alu :uop/add scaled (broadcast-shape-suffix beta-uop sh))
      scaled)))

(defn- lower-rms-norm
  "Lowers `[:rms-norm head x w {:eps eps :gemma? bool}]` into pure UOps."
  [x-uop w-uop attrs]
  (let [eps (double (or (:eps attrs) 1e-5))
        gemma? (boolean (:gemma? attrs))
        dt (:dtype x-uop)
        sh (:shape x-uop)
        last-axis (dec (count sh))
        d (double (last sh))
        inv-d (broadcast-scalar dt (/ 1.0 d) sh)
        x-sq (uop/uop-alu :uop/mul x-uop x-uop)
        sum-sq (uop/uop-reduce x-sq :add [last-axis] true)
        ms (uop/uop-alu :uop/mul (uop/uop-expand sum-sq sh) inv-d)
        ms-eps (uop/uop-alu :uop/add ms (broadcast-scalar dt eps sh))
        rstd (uop/uop-alu :uop/recip (uop/uop-alu :uop/sqrt ms-eps))
        normed (uop/uop-alu :uop/mul x-uop rstd)
        res (if w-uop
              (let [w-bcast (broadcast-shape-suffix w-uop sh)
                    w-eff (if gemma?
                            (uop/uop-alu :uop/add w-bcast (broadcast-scalar dt 1.0 sh))
                            w-bcast)]
                (uop/uop-alu :uop/mul normed w-eff))
              normed)]
    (if (not= dt :f32)
      (uop/uop-cast res dt)
      res)))

(defn- lower-softmax-core
  "Lowers numerically stable softmax along `axis` (default -1) in pure UOps."
  [x-uop axis]
  (let [sh (:shape x-uop)
        ax (if (neg? axis) (+ (count sh) axis) (long axis))
        max-x (uop/uop-expand (uop/uop-reduce x-uop :max [ax] true) sh)
        diff (uop/uop-alu :uop/sub x-uop max-x)
        ex (uop-exp diff)
        sum-ex (uop/uop-expand (uop/uop-reduce ex :add [ax] true) sh)]
    (uop/uop-alu :uop/mul ex (uop/uop-alu :uop/recip sum-ex))))

(defn- lower-causal-softmax
  "Lowers `[:causal-softmax head scores attrs]` into a causal upper-triangular mask + numerically stable softmax."
  [scores-uop attrs]
  (let [sh (:shape scores-uop)
        dt (:dtype scores-uop)
        rank (count sh)
        q-len (nth sh (- rank 2) 1)
        k-len (nth sh (dec rank) 1)
        window (or (:sliding-window attrs) (:sliding_window attrs) (:window attrs))
        ;; Build causal mask tensor of shape [q-len, k-len] where col > row -> -1e9 else 0.0
        ;; If sliding-window is specified, also mask out entries where (row - col) >= window
        mask-vals (vec (for [r (range q-len)
                             c (range k-len)]
                         (if (or (> c r) (and window (>= (- r c) window)))
                           -1.0e9
                           0.0)))
        mask-uop (-> (uop/uop :uop/vconst dt [] mask-vals [(* q-len k-len)])
                     (uop/uop-reshape [q-len k-len])
                     (broadcast-shape-suffix sh))
        masked-scores (uop/uop-alu :uop/add scores-uop mask-uop)]
    (lower-softmax-core masked-scores -1)))

(defn- lower-rope
  "Lowers Rotary Position Embeddings `[:rope head x {:head-dim dh :theta theta :rope-proportion prop :pos pos}]` into pure UOps."
  ([x-uop attrs]
   (lower-rope x-uop attrs {}))
  ([x-uop attrs env]
   (let [sh (:shape x-uop)
         dt (:dtype x-uop)
         b (first sh)
         seq-len (second sh)
         total-d (last sh)
         head-dim (long (or (:head-dim attrs) 64))
         half-dim (quot head-dim 2)
         num-heads (quot total-d head-dim)
         rope-prop (double (or (:rope-proportion attrs) (:partial-rotary-factor attrs) 1.0))
         rope-angles (long (* rope-prop half-dim))
         theta (double (or (:theta attrs) 10000.0))
         freqs (vec (for [i (range half-dim)]
                      (if (< i rope-angles)
                        (/ 1.0 (Math/pow theta (/ (* 2.0 i) (double head-dim))))
                        0.0)))
         half-shape [b seq-len num-heads half-dim]
         pos-attr (:pos attrs)
         dynamic-pos-uop (when (and pos-attr (or (keyword? pos-attr) (symbol? pos-attr)))
                           (get env (keyword pos-attr)))
         [cos-uop sin-uop]
         (cond
           dynamic-pos-uop
           (let [pos-f32 (uop/uop-cast dynamic-pos-uop :f32)
                 pos-4d (uop/uop-reshape pos-f32 [1 1 1 1])
                 eff-pos (if (== 1 seq-len)
                           pos-4d
                           (let [p-vals (vec (map float (range seq-len)))
                                 p-uop (-> (uop/uop :uop/vconst :f32 [] p-vals [seq-len])
                                           (uop/uop-reshape [1 seq-len 1 1]))]
                             (uop/uop-alu :uop/add pos-4d p-uop)))
                 eff-pos-bc (broadcast-shape-suffix eff-pos [1 seq-len 1 half-dim])
                 freqs-uop (-> (uop/uop :uop/vconst :f32 [] freqs [half-dim])
                               (uop/uop-reshape [1 1 1 half-dim]))
                 freqs-bc (broadcast-shape-suffix freqs-uop [1 seq-len 1 half-dim])
                 angle (uop/uop-alu :uop/mul eff-pos-bc freqs-bc)
                 c (-> (uop/uop-alu :uop/cos angle)
                       (uop/uop-expand half-shape))
                 s (-> (uop/uop-alu :uop/sin angle)
                       (uop/uop-expand half-shape))]
             [c s])

           :else
           (let [p-start (if (number? pos-attr) (long pos-attr) 0)
                 cos-vals (vec (for [pos (range p-start (+ p-start seq-len))
                                     i (range half-dim)]
                                 (Math/cos (* (double pos) (nth freqs i)))))
                 sin-vals (vec (for [pos (range p-start (+ p-start seq-len))
                                     i (range half-dim)]
                                 (Math/sin (* (double pos) (nth freqs i)))))
                 c (-> (uop/uop :uop/vconst dt [] cos-vals [(* seq-len half-dim)])
                       (uop/uop-reshape [1 seq-len 1 half-dim])
                       (uop/uop-expand half-shape))
                 s (-> (uop/uop :uop/vconst dt [] sin-vals [(* seq-len half-dim)])
                       (uop/uop-reshape [1 seq-len 1 half-dim])
                       (uop/uop-expand half-shape))]
             [c s]))
         ;; Reshape x to [b, seq-len, num-heads, head-dim] and split into x1, x2 halves via shrink
         x-4d (uop/uop-reshape x-uop [b seq-len num-heads head-dim])
         x1 (uop/uop-shrink x-4d [[0 b] [0 seq-len] [0 num-heads] [0 half-dim]])
         x2 (uop/uop-shrink x-4d [[0 b] [0 seq-len] [0 num-heads] [half-dim head-dim]])
         ;; y1 = x1 * cos - x2 * sin, y2 = x2 * cos + x1 * sin
         y1 (uop/uop-alu :uop/sub
                         (uop/uop-alu :uop/mul x1 cos-uop)
                         (uop/uop-alu :uop/mul x2 sin-uop))
         y2 (uop/uop-alu :uop/add
                         (uop/uop-alu :uop/mul x2 cos-uop)
                         (uop/uop-alu :uop/mul x1 sin-uop))
         cat-4d (uop/uop-cat [y1 y2] 3)
         out (uop/uop-reshape cat-4d sh)]
     (if (not= dt :f32)
       (uop/uop-cast out dt)
       out))))

(defn- lower-gather
  "Lowers `[:gather head weight indices]` into an indexed gather UOp."
  [w-uop idx-uop out-shape]
  (uop/uop :uop/index (:dtype w-uop) [w-uop idx-uop] {:op :gather} (vec out-shape)))

(defn- lower-dynamic-slice
  "Lowers `[:dynamic-slice head in-tensor {:slice-sizes [...] :start-indices [...]}]`."
  [in-uop attrs env out-shape]
  (let [start-indices (or (:start-indices attrs) (:start_indices attrs) (:starts attrs))
        slice-sizes (or (:slice-sizes attrs) (:slice_sizes attrs) out-shape (:shape in-uop))
        starts-spec (mapv (fn [s]
                            (cond
                              (uop/uop? s) s
                              (or (keyword? s) (symbol? s)) (get env (keyword s) s)
                              (number? s) (long s)
                              :else s))
                          start-indices)
        dyn-srcs (filterv uop/uop? starts-spec)
        src (into [in-uop] dyn-srcs)
        arg {:starts-spec starts-spec :slice-sizes (vec slice-sizes)}]
    (uop/uop :uop/dynamic-slice (:dtype in-uop) src arg (vec slice-sizes))))

(defn- lower-dynamic-update-slice
  "Lowers `[:dynamic-update-slice head cache-in k-new {:start-indices [...] :window ... :k-sink ...}]`."
  [cache-in k-new attrs env]
  (let [start-indices (or (:start-indices attrs) (:start_indices attrs) [0 :pos 0 0])
        pos-spec (or (:pos attrs)
                     (some #(when (or (keyword? %) (symbol? %) (uop/uop? %)) %) start-indices)
                     (nth start-indices 1 0))
        pos-node (cond
                   (uop/uop? pos-spec) pos-spec
                   (or (keyword? pos-spec) (symbol? pos-spec)) (get env (keyword pos-spec))
                   (number? pos-spec) (uop/uop-const :i32 (long pos-spec))
                   :else (uop/uop-const :i32 0))
        window (or (:window attrs) (:sliding-window attrs) (:sliding_window attrs))
        k-sink (or (:k-sink attrs) (:k_sink attrs) 0)
        target-dt (:dtype cache-in)
        k-new-cast (if (= (:dtype k-new) target-dt)
                     k-new
                     (uop/uop-cast k-new target-dt))
        src [cache-in k-new-cast pos-node]
        arg {:window window :k-sink k-sink}]
    (uop/uop :uop/dynamic-update-slice target-dt src arg (:shape cache-in))))

(defn- lower-chunked-attention
  "Lowers chunked/decode attention `[:chunked-attention head q k v attrs]` into pure UOps."
  [q-uop k-uop v-uop attrs env out-shape]
  (let [q-sh (:shape q-uop)
        k-sh (:shape k-uop)
        b (nth q-sh 0 1)
        p (nth q-sh 1 1)
        h (nth q-sh 2 8)
        dh (nth q-sh 3 256)
        kvs (nth k-sh 1 128)
        kvh (nth k-sh 2 1)
        ;; Repeat KV heads if MQA/GQA
        k-rep (if (< kvh h)
                (let [g (quot h kvh)]
                  (-> (uop/uop-reshape k-uop [b kvs kvh 1 dh])
                      (uop/uop-expand [b kvs kvh g dh])
                      (uop/uop-reshape [b kvs h dh])))
                k-uop)
        v-rep (if (< kvh h)
                (let [g (quot h kvh)]
                  (-> (uop/uop-reshape v-uop [b kvs kvh 1 dh])
                      (uop/uop-expand [b kvs kvh g dh])
                      (uop/uop-reshape [b kvs h dh])))
                v-uop)
        ;; Q: [b, h, p, dh]
        q-trans (uop/uop-permute q-uop [0 2 1 3])
        ;; K: [b, h, dh, kvs]
        k-trans (uop/uop-permute k-rep [0 2 3 1])
        ;; Scores: [b, h, p, kvs] <- Q [b, h, p, dh] * K [b, h, dh, kvs]
        q-exp (-> (uop/uop-reshape q-trans [b h p 1 dh])
                  (uop/uop-expand [b h p kvs dh]))
        k-exp (-> (uop/uop-reshape (uop/uop-permute k-trans [0 1 3 2]) [b h 1 kvs dh])
                  (uop/uop-expand [b h p kvs dh]))
        qk-prod (uop/uop-alu :uop/mul q-exp k-exp)
        raw-scores (uop/uop-reduce qk-prod :add [4] false)
        scale-val (double (or (:scale attrs) 1.0))
        scale (uop/uop-const :f32 scale-val [1 1 1 1])
        scaled-scores (uop/uop-alu :uop/mul raw-scores (broadcast-shape-suffix scale [b h p kvs]))
        pos-attr (or (:pos attrs) (:start-indices attrs))
        pos-uop (cond
                  (uop/uop? pos-attr) pos-attr
                  (or (keyword? pos-attr) (symbol? pos-attr)) (get env (keyword pos-attr))
                  (number? pos-attr) (uop/uop-const :i32 (long pos-attr) [1 1 1 1])
                  :else nil)
        window (or (:sliding-window attrs) (:sliding_window attrs) (:window attrs))
        masked-scores
        (if (or pos-uop (not= p kvs))
          (let [pos-node (if pos-uop
                           (-> (if (= (:dtype pos-uop) :i32) pos-uop (uop/uop-cast pos-uop :i32))
                               (uop/uop-reshape [1 1 1 1]))
                           (uop/uop-const :i32 0 [1 1 1 1]))
                r-uop (-> (uop/uop :uop/vconst :i32 [] (vec (range p)) [p])
                          (uop/uop-reshape [1 1 p 1]))
                q-pos (uop/uop-alu :uop/add pos-node (broadcast-shape-suffix r-uop [1 1 p 1]))
                c-uop (-> (uop/uop :uop/vconst :i32 [] (vec (range kvs)) [kvs])
                          (uop/uop-reshape [1 1 1 kvs]))
                c-bc (broadcast-shape-suffix c-uop [b h p kvs])
                q-pos-bc (broadcast-shape-suffix q-pos [b h p kvs])
                causal-cond (uop/uop-alu :uop/cmple c-bc q-pos-bc)
                valid-cond (if window
                             (let [win-const (broadcast-shape-suffix (uop/uop-const :i32 (long window) [1 1 1 1]) [b h p kvs])
                                   diff (uop/uop-alu :uop/sub q-pos-bc c-bc)
                                   win-cond (uop/uop-alu :uop/cmplt diff win-const)]
                               (uop/uop-alu :uop/and causal-cond win-cond))
                             causal-cond)
                neg-inf (broadcast-shape-suffix (uop/uop-const :f32 -1.0e9 [1 1 1 1]) [b h p kvs])
                masked (uop/uop-alu :uop/where valid-cond scaled-scores neg-inf)]
            (lower-softmax-core masked -1))
          (lower-causal-softmax scaled-scores attrs))
        ;; Attn output: [b, h, p, dh] <- masked-scores [b, h, p, kvs] * V [b, h, kvs, dh]
        s-exp (-> (uop/uop-reshape masked-scores [b h p kvs 1])
                  (uop/uop-expand [b h p kvs dh]))
        v-trans (uop/uop-permute v-rep [0 2 1 3])
        v-exp (-> (uop/uop-reshape v-trans [b h 1 kvs dh])
                  (uop/uop-expand [b h p kvs dh]))
        sv-prod (uop/uop-alu :uop/mul s-exp v-exp)
        out-trans (uop/uop-reduce sv-prod :add [3] false)
        ;; Permute back to [b, p, h, dh]
        out (uop/uop-permute out-trans [0 2 1 3])]
    (if out-shape (uop/uop-reshape out (vec out-shape)) out)))

;; -----------------------------------------------------------------------------
;; 4. Public AST -> UOp Graph Compiler
;; -----------------------------------------------------------------------------

(defn ast->uop-graph
  "Compiles a Tensor Logic Hiccup AST into a validated High-Level UOp DAG.
   Pipeline:
     1. Expand containers & multi-term contractions (`einsum.logic.expand`)
     2. Prune dead equations via backward-chaining DCE (`einsum.logic.dce`)
     3. Unify shapes across all equations (`einsum.logic.shape`)
     4. Lower each equation to hash-consed UOps & simplify via `einsum.uop.rewrite`."
  ([graph-name ast opts]
   (let [targets (or (:targets opts) (:target-heads opts) [:logits])
         invars (or (:invars opts)
                    (weights/infer-invars ast (or (:inputs opts) []) (or (:weights opts) {})))]
     (ast->uop-graph graph-name invars ast targets)))
  ([graph-name invars ast target-heads]
   (let [targets (if (set? target-heads) target-heads (set target-heads))
         expanded (expand/expand-ast {} ast)
         pruned (dce/prune-ast expanded targets)
         in-shapes (into {} (map (fn [[v [_ sh _]]] [v (vec sh)]) invars))
         in-dtypes (into {} (map (fn [[v [_ _ dt]]] [v (or dt :f32)]) invars))
         known-shapes (shape/unify-shapes in-shapes pruned)
         ;; Initialize environment with `:uop/param` nodes for each invar
         initial-env (into {}
                           (map-indexed
                            (fn [slot [var-kw [_ sh dt]]]
                              (let [storage-dt (or dt :f32)
                                    p-uop (uop/uop-param slot storage-dt sh var-kw)]
                                [var-kw (if (contains? #{:bf16 :f16} storage-dt)
                                          (uop/uop-cast p-uop :f32)
                                          p-uop)]))
                            invars))
         ;; Step through pruned equations in topological order, accumulating repeated heads
         final-env
         (reduce
          (fn [env eqn]
            (let [op (first eqn)
                  head (ast/head eqn)
                  h-name (if (vector? head) (first head) head)
                  attrs (ast/attrs eqn)
                  body (ast/body-terms eqn)
                  out-shape (or (:shape attrs) (get known-shapes h-name))
                  term-uop
                  (cond
                    (ast/eqn? eqn)
                    (cond
                      (= (count body) 2)
                      (lower-binary-contraction head attrs (first body) (second body) env known-shapes)

                      (= (count body) 1)
                      (lower-unary-equation head attrs (first body) env known-shapes)

                      :else
                      (throw (ex-info "Unsupported contraction body arity in UOp lowering" {:eqn eqn})))

                    (= op :gather)
                    (let [w-uop (get env (first (first body)))
                          idx-uop (get env (first (second body)))]
                      (lower-gather w-uop idx-uop out-shape))

                    (= op :slice)
                    (let [in-uop (get env (first (first body)))
                          starts (or (:start_indices attrs) (:start attrs) (:starts attrs))
                          limits (or (:limit_indices attrs) (:limit attrs) (:limits attrs))
                          bounds (mapv vector starts limits)]
                      (uop/uop-shrink in-uop bounds))

                    (= op :reshape)
                    (let [in-uop (get env (first (first body)))]
                      (uop/uop-reshape in-uop out-shape))

                    (= op :layer-norm)
                    (let [x-uop (get env (first (first body)))
                          g-uop (when (second body) (get env (first (second body))))
                          b-uop (when (nth body 2 nil) (get env (first (nth body 2))))]
                      (lower-layer-norm x-uop g-uop b-uop attrs))

                    (or (= op :rms-norm) (= op :gemma-rms-norm))
                    (let [x-uop (get env (first (first body)))
                          w-uop (when (second body) (get env (first (second body))))
                          eff-attrs (if (= op :gemma-rms-norm) (assoc attrs :gemma? true) attrs)]
                      (lower-rms-norm x-uop w-uop eff-attrs))

                    (= op :softmax)
                    (let [x-uop (get env (first (first body)))]
                      (lower-softmax-core x-uop (or (:axis attrs) -1)))

                    (= op :causal-softmax)
                    (let [scores-uop (get env (first (first body)))]
                      (lower-causal-softmax scores-uop attrs))

                    (= op :dynamic-slice)
                    (let [in-uop (get env (first (first body)))]
                      (lower-dynamic-slice in-uop attrs env out-shape))

                    (= op :dynamic-update-slice)
                    (let [cache-in (get env (first (first body)))
                          k-new (get env (first (second body)))]
                      (lower-dynamic-update-slice cache-in k-new attrs env))

                    (= op :chunked-attention)
                    (let [q-uop (get env (first (first body)))
                          k-cache (get env (first (second body)))
                          v-cache (get env (first (nth body 2)))]
                      (lower-chunked-attention q-uop k-cache v-cache attrs env out-shape))

                    (= op :rope)
                    (let [x-uop (get env (first (first body)))]
                      (lower-rope x-uop attrs env))

                    (or (= op :+) (= op :add))
                    (let [a (get env (first (first body)))
                          b (get env (first (second body)))]
                      (uop/uop-alu :uop/add a (broadcast-shape-suffix b (:shape a))))

                    (or (= op :-) (= op :subtract))
                    (let [a (get env (first (first body)))
                          b (get env (first (second body)))]
                      (uop/uop-alu :uop/sub a (broadcast-shape-suffix b (:shape a))))

                    (or (= op :*) (= op :multiply))
                    (let [a (get env (first (first body)))
                          b (get env (first (second body)))]
                      (uop/uop-alu :uop/mul a (broadcast-shape-suffix b (:shape a))))

                    (or (= op :/) (= op :divide))
                    (let [a (get env (first (first body)))
                          b (get env (first (second body)))]
                      (uop/uop-alu :uop/div a (broadcast-shape-suffix b (:shape a))))

                    (= op :constant)
                    (let [val (get attrs :value 0.0)
                          val-t (or (:type attrs) [:tensor [] :f32])
                          dt (last val-t)]
                      (uop/uop-const dt val (or out-shape [])))

                    :else
                    (throw (ex-info "Unsupported AST op in UOp lowering" {:op op :eqn eqn})))

                  ;; Implicit multi-equation accumulation (e.g. adding bias or residual connections)
                  accumulated-uop
                  (if-let [prev-uop (get env h-name)]
                    (let [target-sh (:shape prev-uop)
                          aligned-term (broadcast-shape-suffix term-uop target-sh)]
                      (uop/uop-alu :uop/add prev-uop aligned-term))
                    term-uop)]
              (assoc env h-name (rewrite/simplify-uop accumulated-uop))))
          initial-env
          pruned)
         out-vec (if (sequential? target-heads) (vec target-heads) (vec targets))
         target-entries (mapv (fn [t]
                                (if (vector? t) [(first t) (second t)] [t nil]))
                              out-vec)
         outvars (mapv first target-entries)
         outputs
         (into {}
               (map (fn [[k explicit-dt]]
                      (let [node (get final-env k)
                            target-dt (or explicit-dt
                                          ;; If target corresponds to a cache_out, match cache_in
                                          (when-let [in-k (some #(when (and (str/includes? (name %) "cache_in")
                                                                            (= (str/replace (name %) "cache_in" "cache_out") (name k))) %)
                                                                (keys in-dtypes))]
                                            (get in-dtypes in-k))
                                          ;; If target is prefill KV output (k_ro / v_heads), match embed_tokens
                                          (when (or (str/starts-with? (name k) "k_ro")
                                                    (str/starts-with? (name k) "v_heads"))
                                            (get in-dtypes :embed_tokens))
                                          ;; If target is logits, match embed_tokens
                                          (when (= k :logits)
                                            (get in-dtypes :embed_tokens)))]
                        [k (if (and target-dt node (not= (:dtype node) target-dt))
                             (uop/uop-cast node target-dt)
                             node)]))
                    target-entries))]
     {:name graph-name
      :invars invars
      :in-dtypes in-dtypes
      :known-shapes known-shapes
      :outvars outvars
      :outputs outputs})))
