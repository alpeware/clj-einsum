(ns einsum.uop.core
  "Core UOp (Micro-Operation) IR, hash-consing, shape/dtype inference, and .uir serialization.
   Modeled on tinygrad's tinyspec and Compilers for Machine Learning."
  (:require [clojure.string :as str]
            [malli.core :as m]))

;; -----------------------------------------------------------------------------
;; 1. UOp Taxonomy & Schemas (per tinyspec)
;; -----------------------------------------------------------------------------

(def DTYPES
  "Supported scalar data types in the UOp IR."
  #{:void :bool :i1 :i8 :u8 :i16 :u16 :i32 :u32 :i64 :u64 :f16 :bf16 :f32 :f64})

(def SOURCE_OPS
  "Leaf / source UOps."
  #{:uop/buffer :uop/buffer-view :uop/param :uop/const :uop/vconst :uop/define-local :uop/special})

(def MOVEMENT_OPS
  "Pure view / layout movement UOps (zero-compute coordinate transformations)."
  #{:uop/reshape :uop/expand :uop/permute :uop/shrink :uop/pad :uop/flip :uop/cat})

(def UNARY_ALU_OPS
  "Elementwise unary ALU UOps."
  #{:uop/neg :uop/recip :uop/exp2 :uop/log2 :uop/sin :uop/sqrt :uop/trunc :uop/cast :uop/bitcast})

(def BINARY_ALU_OPS
  "Elementwise binary ALU UOps."
  #{:uop/add :uop/sub :uop/mul :uop/div :uop/idiv :uop/mod
    :uop/max :uop/min
    :uop/shl :uop/shr :uop/and :uop/or :uop/xor
    :uop/cmplt :uop/cmple :uop/cmpeq :uop/cmpne :uop/cmpgt :uop/cmpge})

(def TERNARY_ALU_OPS
  "Elementwise ternary ALU UOps."
  #{:uop/where :uop/mulacc})

(def ALU_OPS
  "All elementwise ALU UOps."
  (into (into UNARY_ALU_OPS BINARY_ALU_OPS) TERNARY_ALU_OPS))

(def REDUCE_OPS
  "Reduction UOps."
  #{:uop/reduce :uop/reduce-axis})

(def CONTROL_AND_MEM_OPS
  "Low-level loop, indexing, memory, and program structure UOps (used during & after rangeify)."
  #{:uop/index :uop/load :uop/store :uop/alloc
    :uop/range :uop/end :uop/if :uop/endif :uop/barrier
    :uop/wmma :uop/after :uop/assign :uop/sink :uop/linear :uop/call :uop/program})

(def ALL_UOP_OPS
  "Complete set of valid UOp operation keywords."
  (reduce into #{} [SOURCE_OPS MOVEMENT_OPS ALU_OPS REDUCE_OPS CONTROL_AND_MEM_OPS]))

(def UOpSchema
  "Malli schema for a single UOp DAG node."
  [:map
   [:id :int]
   [:op [:and :keyword [:fn #(contains? ALL_UOP_OPS %)]]]
   [:dtype [:and :keyword [:fn #(contains? DTYPES %)]]]
   [:src [:vector :any]]
   [:arg :any]
   [:shape [:maybe [:vector :int]]]])

;; -----------------------------------------------------------------------------
;; 2. Hash-Consing Intern Table
;; -----------------------------------------------------------------------------

(defonce ^:private uop-intern-table (atom {}))
(defonce ^:private uop-id-counter (atom 0))

(defn clear-uop-cache!
  "Clears the global hash-consing table and resets node IDs (useful for deterministic tests)."
  []
  (reset! uop-intern-table {})
  (reset! uop-id-counter 0))

(defn uop
  "Hash-consed smart constructor for a UOp node.
   Identical [op dtype src-ids arg shape] tuples return the exact same canonical map instance."
  ([op dtype src arg]
   (uop op dtype src arg nil))
  ([op dtype src arg shape]
   (when (nil? op)
     (throw (ex-info "Cannot construct UOp with nil :op" {:dtype dtype :src src :arg arg :shape shape})))
   (let [src-vec (vec src)
         src-ids (mapv :id src-vec)
         shape-vec (when shape (mapv long shape))
         intern-key [op dtype src-ids arg shape-vec]
         existing (get @uop-intern-table intern-key)]
     (if existing
       existing
       (let [new-id (swap! uop-id-counter inc)
             node {:id new-id
                   :op op
                   :dtype dtype
                   :src src-vec
                   :arg arg
                   :shape shape-vec}
             tbl (swap! uop-intern-table
                        (fn [m]
                          (if (contains? m intern-key)
                            m
                            (assoc m intern-key node))))]
         (get tbl intern-key))))))

(defn valid-uop?
  "Returns true if `node` and all of its transitive `:src` dependencies satisfy UOpSchema."
  [node]
  (and (map? node)
       (m/validate UOpSchema node)
       (every? #(and (map? %) (m/validate UOpSchema %)) (:src node))))

;; -----------------------------------------------------------------------------
;; 3. Topological Sort & Graph Traversal
;; -----------------------------------------------------------------------------

(defn toposort
  "Returns a vector of all UOp nodes reachable from `root` (or collection of roots)
   in topological order (parents/sources before dependents)."
  [root-or-roots]
  (let [roots (if (map? root-or-roots) [root-or-roots] (vec root-or-roots))
        visited (java.util.HashSet.)
        result (java.util.ArrayList.)]
    (letfn [(visit [node]
              (when (and node (not (.contains visited (:id node))))
                (.add visited (:id node))
                (doseq [s (:src node)]
                  (visit s))
                (.add result node)))]
      (doseq [r roots]
        (visit r))
      (vec result))))

;; -----------------------------------------------------------------------------
;; 4. Smart Constructors for Source, Movement, ALU, Reduce, and Loop UOps
;; -----------------------------------------------------------------------------

(defn uop-const
  "Creates a scalar constant UOp with shape []."
  ([dtype val]
   (uop-const dtype val []))
  ([dtype val shape]
   (let [norm-val (cond
                    (contains? #{:f16 :bf16 :f32 :f64} dtype) (double val)
                    (contains? #{:bool :i1} dtype) (boolean val)
                    :else (long val))]
     (uop :uop/const dtype [] norm-val (vec shape)))))

(defn uop-buffer
  "Creates a global persistent Buffer UOp of 1D size `n-elements`."
  [slot dtype n-elements & [device]]
  (uop :uop/buffer dtype [] {:slot (long slot) :size (long n-elements) :device (or device :default)} [(long n-elements)]))

(defn uop-param
  "Creates a Parameter UOp for slot `slot` with given `dtype` and `shape`."
  [slot dtype shape & [var-name]]
  (uop :uop/param dtype [] {:slot (long slot) :name var-name} (vec shape)))

(defn uop-reshape
  "Movement Op: reinterprets tensor `x` into `new-shape` (preserving element count)."
  [x new-shape]
  (let [target-shape (mapv long new-shape)]
    (if (= (:shape x) target-shape)
      x
      (uop :uop/reshape (:dtype x) [x] target-shape target-shape))))

(defn uop-expand
  "Movement Op: broadcasts size-1 dimensions of `x` to `target-shape`."
  [x target-shape]
  (let [t-shape (mapv long target-shape)
        in-shape (:shape x)]
    (when (and in-shape (not= (count in-shape) (count t-shape)))
      (throw (ex-info "uop-expand requires matching rank (reshape with 1s first)"
                      {:in-shape in-shape :target-shape t-shape})))
    (if (= in-shape t-shape)
      x
      (uop :uop/expand (:dtype x) [x] t-shape t-shape))))

(defn uop-permute
  "Movement Op: reorders dimensions of `x` according to permutation vector `perm`."
  [x perm]
  (let [p-vec (mapv long perm)
        in-shape (:shape x)]
    (if (= p-vec (vec (range (count p-vec))))
      x
      (let [out-shape (when in-shape (mapv #(nth in-shape %) p-vec))]
        (uop :uop/permute (:dtype x) [x] p-vec out-shape)))))

(defn uop-shrink
  "Movement Op: slices `x` along each axis according to `bounds` vector `[[start0 end0] [start1 end1] ...]`."
  [x bounds]
  (let [b-vec (mapv (fn [[s e]] [(long s) (long e)]) bounds)
        in-shape (:shape x)
        out-shape (mapv (fn [[s e]] (- e s)) b-vec)]
    (if (and in-shape (= b-vec (mapv (fn [d] [0 (long d)]) in-shape)))
      x
      (uop :uop/shrink (:dtype x) [x] b-vec out-shape))))

(defn uop-pad
  "Movement Op: zero-pads `x` along each axis according to `padding` vector `[[before0 after0] ...]`."
  [x padding]
  (let [p-vec (mapv (fn [[b a]] [(long b) (long a)]) padding)
        in-shape (:shape x)]
    (if (every? (fn [[b a]] (and (zero? b) (zero? a))) p-vec)
      x
      (let [out-shape (when in-shape
                        (mapv (fn [d [b a]] (+ (long d) b a)) in-shape p-vec))]
        (uop :uop/pad (:dtype x) [x] p-vec out-shape)))))

(defn uop-flip
  "Movement Op: reverses `x` along axes where `flip-mask` boolean vector is true."
  [x flip-mask]
  (let [f-vec (mapv boolean flip-mask)]
    (if (every? false? f-vec)
      x
      (uop :uop/flip (:dtype x) [x] f-vec (:shape x)))))

(defn uop-cat
  "Movement Op: concatenates `srcs` along `axis`."
  [srcs axis]
  (let [src-vec (vec srcs)
        first-uop (first src-vec)
        in-shape (:shape first-uop)
        ax (long axis)
        ax-norm (if (neg? ax) (+ (count in-shape) ax) ax)
        total-dim (reduce + (map #(nth (:shape %) ax-norm) src-vec))
        out-shape (assoc (vec in-shape) ax-norm total-dim)]
    (if (= 1 (count src-vec))
      first-uop
      (uop :uop/cat (:dtype first-uop) src-vec ax-norm out-shape))))

(defn uop-cast
  "Unary cast of `x` to `target-dtype`."
  [x target-dtype]
  (if (= (:dtype x) target-dtype)
    x
    (uop :uop/cast target-dtype [x] target-dtype (:shape x))))

(defn uop-alu
  "Constructs an elementwise ALU UOp (:uop/add, :uop/mul, :uop/where, etc.)."
  [op & srcs]
  (let [src-vec (vec srcs)]
    (when (some nil? src-vec)
      (throw (ex-info "Nil source passed to uop-alu" {:op op :srcs src-vec})))
    (let [first-src (first src-vec)
          out-dtype (if (contains? #{:uop/cmplt :uop/cmple :uop/cmpeq :uop/cmpne :uop/cmpgt :uop/cmpge} op)
                      :bool
                      (if (= op :uop/where)
                        (:dtype (second src-vec))
                        (:dtype first-src)))
          out-shape (some :shape src-vec)]
      (uop op out-dtype src-vec nil out-shape))))

(defn uop-reduce
  "Reduces `x` using `reduce-op` (:add, :max, :min) along `axes` (vector of axis indices).
   Per tinyspec, reduced axes are collapsed to size 1 in `:shape` unless `:keep-dims? false` is passed,
   in which case a reshape is applied or axes are popped."
  ([x reduce-op axes]
   (uop-reduce x reduce-op axes true))
  ([x reduce-op axes keep-dims?]
   (let [in-shape (:shape x)
         axes-vec (vec (sort (map (fn [a] (if (neg? a) (+ (count in-shape) a) (long a))) axes)))
         axes-set (set axes-vec)
         reduced-shape (when in-shape
                         (mapv (fn [idx d] (if (contains? axes-set idx) 1 d))
                               (range (count in-shape))
                               in-shape))
         red-node (uop :uop/reduce (:dtype x) [x] {:op reduce-op :axes axes-vec} reduced-shape)]
     (if (and (not keep-dims?) in-shape)
       (let [squeezed (vec (keep-indexed (fn [idx d] (when-not (contains? axes-set idx) d)) in-shape))]
         (uop-reshape red-node squeezed))
       red-node))))

(defn uop-range
  "Low-level loop iterator UOp from 0 to `bound` (exclusive)."
  ([bound axis-id]
   (uop-range bound axis-id :loop))
  ([bound axis-id axis-type]
   (let [b-uop (if (map? bound) bound (uop-const :i32 bound))]
     (uop :uop/range :i32 [b-uop] {:axis (long axis-id) :type axis-type} []))))

(defn uop-index
  "Low-level buffer indexing UOp: indexes `buf` at linear or multi-dim offset `idx`."
  [buf idx & [gate]]
  (let [srcs (if gate [buf idx gate] [buf idx])]
    (uop :uop/index (:dtype buf) srcs nil [])))

(defn uop-load
  "Low-level memory load UOp from `idx-uop`."
  [idx-uop]
  (uop :uop/load (:dtype idx-uop) [idx-uop] nil []))

(defn uop-store
  "Low-level memory store UOp writing `val-uop` to `idx-uop`."
  [idx-uop val-uop & [ranges]]
  (let [srcs (into [idx-uop val-uop] (or ranges []))]
    (uop :uop/store :void srcs nil [])))

;; -----------------------------------------------------------------------------
;; 5. Universal Interchange Format (.uir) Serializer
;; -----------------------------------------------------------------------------

(defn uop->uir-string
  "Serializes a UOp DAG rooted at `root` into geohot's human-readable `.uir` text format."
  [root]
  (let [nodes (toposort root)
        id->idx (into {} (map-indexed (fn [i n] [(:id n) i]) nodes))]
    (str/join
     "\n"
     (map-indexed
      (fn [idx {:keys [op dtype src arg shape]}]
        (let [op-name (name op)
              src-refs (str/join " " (map #(str "%" (get id->idx (:id %))) src))]
          (case op
            :uop/buffer
            (format "%%%d = buffer %d : dtype=%s slot=%d"
                    idx (:size arg) (name dtype) (:slot arg))

            :uop/param
            (format "%%%d = param %s : dtype=%s slot=%d%s"
                    idx (pr-str shape) (name dtype) (:slot arg)
                    (if (:name arg) (str " name=" (name (:name arg))) ""))

            :uop/const
            (format "%%%d = const %s : dtype=%s"
                    idx (str arg) (name dtype))

            (:uop/reshape :uop/expand :uop/permute :uop/shrink :uop/pad)
            (format "%%%d = %s %s %s"
                    idx op-name src-refs (pr-str arg))

            :uop/reduce
            (format "%%%d = reduce %s : op=%s axes=%s"
                    idx src-refs (name (:op arg)) (pr-str (:axes arg)))

            (format "%%%d = %s%s%s"
                    idx op-name
                    (if (seq src-refs) (str " " src-refs) "")
                    (if (some? arg) (str " : " (pr-str arg)) "")))))
      nodes))))
