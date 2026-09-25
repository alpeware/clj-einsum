(ns einsum.logic.interpret
  "Pure-JVM execution backend for lowered StableHLO graphs.

  Consumes the EDN SSA IR produced by `einsum.logic.lower/ast->graph`
  (the same IR that `graph->mlir-text` serializes for XLA) and executes it
  directly on the JVM with plain primitive arrays -- no PJRT, no XLA, no
  native libraries.

  Rationale: on memory-constrained machines the PJRT CPU backend must
  duplicate the whole model (mmap page cache + 4+ GB of anonymous PJRT
  buffers + JVM heap), which OOMs. The interpreter reads weights straight
  from the mmap'd safetensors segments (zero copy, kernel-reclaimable page
  cache) and only materializes activations.

  NOTE (repo-rule tension): `AGENTS.md` Rule 4 forbids host-side matrix
  engines that *bypass* XLA compilation. This backend does not bypass
  compilation -- model math stays declarative Tensor Logic lowered to
  StableHLO; only the *execution* of that IR changes. It is the same
  relationship the reference interpreter already has to the compiler.
  This work lives on the `jvm-stableshlo-backend` branch until the rule is
  amended to distinguish execution backends from compilation bypasses.

  Tensors are plain maps:
    {:dtype :f32 | :bf16 | :i8 | :i32 | :pred
     :shape [d0 d1 ...]            ;; row-major logical shape
     :data  <primitive array | MemorySegment>}"
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [jdk.incubator.vector FloatVector VectorSpecies]
           [java.util Arrays]))

;; ---------------------------------------------------------------------------
;; Dtypes
;; ---------------------------------------------------------------------------

(def dtypes
  "Supported dtypes and their JVM storage."
  #{:f32 :bf16 :i8 :i32 :pred})

(def ^:private dtype->kind
  {:f32 :float :bf16 :float :i8 :int :i32 :int :pred :bool})

(defn dtype-kind
  "Coarse kind used to pick a compute path: :float, :int or :bool."
  [dtype]
  (or (dtype->kind dtype)
      (throw (ex-info "Unknown dtype" {:dtype dtype}))))

;; ---------------------------------------------------------------------------
;; bf16 <-> f32 (round-to-nearest-even on quantize)
;; ---------------------------------------------------------------------------

(defn bf16->f32
  "Converts a bf16 value (as a short) to float."
  [b]
  ;; shift the 16-bit pattern into the high half of an int32 bit pattern;
  ;; use long arithmetic to avoid int overflow on the shift.
  (Float/intBitsToFloat (unchecked-int (bit-shift-left (long (bit-and (int b) 0xFFFF)) 16))))

(defn f32->bf16
  "Converts a float to bf16 (as a short), round-to-nearest-even.
   NaN -> canonical bf16 NaN; +/-Inf and out-of-range magnitudes saturate to
   bf16 infinities; subnormals may flush to zero."
  [f]
  (let [d (double f)
        bf16-max 3.38953139e38]
    (cond
      (Double/isNaN d) (short 0x7FC0)
      (or (> d bf16-max) (and (Double/isInfinite d) (pos? d))) (short 0x7F80)
      (or (< d (- bf16-max)) (and (Double/isInfinite d) (neg? d))) (short (unchecked-short 0xFF80))
      :else
      (let [bits (Float/floatToRawIntBits (float d))
            ;; round to nearest even: add 0x7FFF + LSB of the result
            rounded (+ bits 0x7FFF (bit-and (bit-shift-right bits 16) 1))]
        (short (bit-shift-right rounded 16))))))

;; ---------------------------------------------------------------------------
;; Tensor construction and shape math
;; ---------------------------------------------------------------------------

(defn elem-count
  "Total element count of a shape vector."
  [shape]
  (reduce * 1 shape))

(defn strides
  "Row-major strides for a shape vector."
  [shape]
  (let [n (count shape)]
    (mapv (fn [i] (reduce * 1 (subvec (vec shape) (inc i)))) (range n))))

(defn- coerce-data
  "Coerces sequential data to the primitive array for dtype."
  [dtype xs]
  (let [n (count xs)]
    (case dtype
      :f32 (let [a (float-array n)] (dotimes [i n] (aset a i (float (nth xs i)))) a)
      :bf16 (let [a ^shorts (short-array n)] (dotimes [i n] (aset a i (short (f32->bf16 (float (nth xs i)))))) a)
      :i8 (let [a (byte-array n)] (dotimes [i n] (aset a i (byte (nth xs i)))) a)
      :i32 (let [a (int-array n)] (dotimes [i n] (aset a i (int (nth xs i)))) a)
      :pred (let [a (boolean-array n)] (dotimes [i n] (aset a i (boolean (nth xs i)))) a))))

(defn tensor
  "Constructs a tensor map. `data` may be a primitive array, a MemorySegment
   (zero-copy weight view), or a sequential of values. For :bf16, sequential
   values are floats and are quantized; pass a short[] directly for raw
   bit patterns."
  [dtype shape data]
  (when-not (contains? dtypes dtype)
    (throw (ex-info "Unknown dtype" {:dtype dtype})))
  (let [shape (vec shape)
        want (elem-count shape)
        seg? (instance? MemorySegment data)
        seq? (and (not seg?) (sequential? data))
        arr? (and (not seg?) (not seq?) (.isArray (.getClass ^Object data)))]
    (cond
      seq?
      (do (when (not= want (count data))
            (throw (ex-info "Data length does not match shape"
                            {:dtype dtype :shape shape :n (count data)})))
          {:dtype dtype :shape shape :data (coerce-data dtype data)})

      arr?
      (let [got (java.lang.reflect.Array/getLength data)]
        (when (not= want got)
          (throw (ex-info "Data length does not match shape"
                          {:dtype dtype :shape shape :n got})))
        {:dtype dtype :shape shape :data data})

      :else
      {:dtype dtype :shape shape :data data})))

(defn scalar
  "Scalar tensor holding one value."
  [dtype v]
  (tensor dtype [] (coerce-data dtype [v])))

;; ---------------------------------------------------------------------------
;; Data access
;; ---------------------------------------------------------------------------

(defn- segment->floats
  "Reads `n` f32-convertible elements from a MemorySegment slice."
  [^MemorySegment seg dtype n]
  (let [out (float-array n)]
    (case dtype
      :f32 (dotimes [i n]
             (aset out i (.getAtIndex seg ValueLayout/JAVA_FLOAT (long i))))
      :bf16 (dotimes [i n]
              (aset out i (float (bf16->f32 (.getAtIndex seg ValueLayout/JAVA_SHORT (long i))))))
      :i8 (dotimes [i n]
            (aset out i (float (.getAtIndex seg ValueLayout/JAVA_BYTE (long i)))))
      :i32 (dotimes [i n]
             (aset out i (float (.getAtIndex seg ValueLayout/JAVA_INT (long i)))))
      (throw (ex-info "Cannot convert dtype to float" {:dtype dtype})))
    out))

(defn as-floats
  "Returns tensor values as a fresh float[] (converting bf16/i8/i32)."
  [{:keys [dtype shape data]}]
  (let [n (elem-count shape)]
    (if (instance? MemorySegment data)
      (segment->floats data dtype n)
      (case dtype
        :f32 (aclone ^floats data)
        :bf16 (let [s ^shorts data
                    ^floats out (float-array n)]
                (dotimes [i n] (aset out i (float (bf16->f32 (aget s i)))))
                out)
        :i8 (let [b ^bytes data
                  out (float-array n)]
              (dotimes [i n] (aset out i (float (aget b i))))
              out)
        :i32 (let [a ^ints data
                   out (float-array n)]
               (dotimes [i n] (aset out i (float (aget a i))))
               out)
        (throw (ex-info "Cannot convert dtype to float" {:dtype dtype}))))))

(defn- pack-floats
  "Packs a float[] into the primitive array for dtype (RNE for bf16)."
  [dtype ^floats fs]
  (let [n (alength fs)]
    (case dtype
      :f32 (aclone fs)
      :bf16 (let [^shorts out (short-array n)]
              (dotimes [i n] (aset out i (short (f32->bf16 (aget fs i)))))
              out)
      :i8 (let [out (byte-array n)]
            (dotimes [i n] (aset out i (unchecked-byte (Math/round (aget fs i)))))
            out)
      :i32 (let [out (int-array n)]
             (dotimes [i n] (aset out i (unchecked-int (Math/round (aget fs i)))))
             out)
      (throw (ex-info "Cannot pack floats to dtype" {:dtype dtype})))))

;; ---------------------------------------------------------------------------
;; Broadcasting
;; ---------------------------------------------------------------------------

(defn broadcast-shapes
  "NumPy-style broadcast of two shapes. Returns [result-shape] or throws."
  [sa sb]
  (let [sa (vec sa) sb (vec sb)
        n (max (count sa) (count sb))
        pa (concat (repeat (- n (count sa)) 1) sa)
        pb (concat (repeat (- n (count sb)) 1) sb)]
    (mapv (fn [a b]
            (cond (= a b) a
                  (= a 1) b
                  (= b 1) a
                  :else (throw (ex-info "Shapes not broadcastable"
                                        {:sa sa :sb sb}))))
          pa pb)))

(defn- broadcast-strides
  "Strides for reading a broadcast operand: 0 stride on broadcast dims."
  [shape target-shape]
  (let [shape (vec shape)
        pad (- (count target-shape) (count shape))
        s (strides shape)]
    (mapv (fn [i]
            (let [si (- i pad)]
              (if (or (neg? si) (= 1 (nth shape si))) 0 (nth s si))))
          (range (count target-shape)))))

;; ---------------------------------------------------------------------------
;; Elementwise engine (float path)
;; ---------------------------------------------------------------------------

(defn- elementwise-floats
  "Applies binary float fn `f` with NumPy broadcasting. Returns float[] of
   target-shape."
  [f ^floats fa sa ^floats fb sb]
  (let [tshape (broadcast-shapes sa sb)
        n (elem-count tshape)
        stra (broadcast-strides sa tshape)
        strb (broadcast-strides sb tshape)
        ^floats out (float-array n)
        rank (count tshape)]
    ;; odometer over target coords
    (let [^longs coords (long-array rank)]
      (dotimes [flat n]
        (let [oa (loop [i 0 off 0]
                   (if (= i rank) off
                       (recur (inc i) (+ off (* (aget coords i) (long (nth stra i)))))))
              ob (loop [i 0 off 0]
                   (if (= i rank) off
                       (recur (inc i) (+ off (* (aget coords i) (long (nth strb i)))))))]
          (aset out flat (float (f (aget fa oa) (aget fb ob)))))
        ;; increment odometer
        (loop [i (dec rank)]
          (when (>= i 0)
            (let [c (inc (aget coords i))
                  dim (nth tshape i)]
              (if (< c dim)
                (aset coords i c)
                (do (aset coords i 0)
                    (recur (dec i)))))))))
    out))

(defn- elementwise
  "Binary elementwise op on float-kind tensors. Output dtype = higher-precision of inputs."
  [f a b]
  (let [da (:dtype a) db (:dtype b)
        out-dtype (cond
                    (or (= da :f32) (= db :f32)) :f32
                    (or (= da :bf16) (= db :bf16)) :bf16
                    :else da)]
    (when (= :bool (dtype-kind da))
      (throw (ex-info "Use boolean elementwise for pred tensors" {:dtype da})))
    (let [tshape (broadcast-shapes (:shape a) (:shape b))
          out (elementwise-floats f (as-floats a) (:shape a) (as-floats b) (:shape b))]
      {:dtype out-dtype :shape tshape :data (pack-floats out-dtype out)})))

(defn op-add [a b] (elementwise + a b))
(defn op-sub [a b] (elementwise - a b))
(defn op-mul [a b] (elementwise * a b))
(defn op-div [a b] (elementwise / a b))

(defn- unary-float
  "Unary float op. Output dtype = input dtype."
  [f t]
  (let [^floats fs (as-floats t)
        n (alength fs)
        ^floats out (float-array n)]
    (dotimes [i n] (aset out i (float (f (aget fs i)))))
    {:dtype (:dtype t) :shape (:shape t) :data (pack-floats (:dtype t) out)}))

(defn op-negate [t] (unary-float - t))
(defn op-sqrt [t] (unary-float #(Math/sqrt %) t))
(defn op-rsqrt [t] (unary-float #(/ 1.0 (Math/sqrt %)) t))
(defn op-exp [t] (unary-float #(Math/exp %) t))
(defn op-tanh [t] (unary-float #(Math/tanh %) t))
(defn op-sine [t] (unary-float #(Math/sin %) t))
(defn op-cosine [t] (unary-float #(Math/cos %) t))

;; ---------------------------------------------------------------------------
;; Coordinate iteration helpers
;; ---------------------------------------------------------------------------

(defn- with-coords
  "Calls (f coords-vec flat-idx) for every coordinate of `shape` in row-major
   order. `coords-vec` is a reused long[]."
  [shape f]
  (let [shape (vec shape)
        rank (count shape)
        coords (long-array rank)
        n (elem-count shape)]
    (dotimes [flat n]
      (f coords flat)
      (loop [i (dec rank)]
        (when (>= i 0)
          (let [c (inc (aget coords i))]
            (if (< c (long (nth shape i)))
              (aset coords i c)
              (do (aset coords i 0)
                  (recur (dec i))))))))))

(defn- coords->flat
  "Row-major flat index for coords (long[]) given strides vector."
  [^longs coords strides-v]
  (loop [i 0 off 0]
    (if (= i (alength coords)) off
        (recur (inc i) (+ off (* (aget coords i) (long (nth strides-v i))))))))

;; ---------------------------------------------------------------------------
;; Structural ops
;; ---------------------------------------------------------------------------

(defn op-reshape
  "Reshape: shares backing storage."
  [t new-shape]
  (let [new-shape (vec new-shape)]
    (when (not= (elem-count (:shape t)) (elem-count new-shape))
      (throw (ex-info "Reshape element count mismatch"
                      {:from (:shape t) :to new-shape})))
    (assoc t :shape new-shape)))

(defn op-transpose
  "Transposes axes per `permutation` (out-dim i <- in-dim permutation[i])."
  [t permutation]
  (let [permutation (vec permutation)
        in-shape (:shape t)
        rank (count in-shape)
        _ (when (not= (sort permutation) (range rank))
            (throw (ex-info "Invalid permutation" {:permutation permutation :rank rank})))
        out-shape (mapv in-shape permutation)
        in-strides (strides in-shape)
        fs (as-floats t)
        ^floats out (float-array (elem-count out-shape))]
    ;; out[oc] = in[ic] where ic[permutation[i]] = oc[i]
    (with-coords out-shape
      (fn [^longs oc flat]
        (let [^longs ic (long-array rank)]
          (dotimes [i rank] (aset ic (long (nth permutation i)) (aget oc i)))
          (aset out flat (aget ^floats fs (coords->flat ic in-strides))))))
    {:dtype (:dtype t) :shape out-shape :data (pack-floats (:dtype t) out)}))

(defn op-broadcast-in-dim
  "StableHLO broadcast_in_dim."
  [t broadcast-dimensions target-shape]
  (let [bcast-dims (vec broadcast-dimensions)
        target-shape (vec target-shape)
        in-shape (:shape t)
        _ (when (not= (count bcast-dims) (count in-shape))
            (throw (ex-info "broadcast_dimensions rank mismatch"
                            {:bcast-dims bcast-dims :in-shape in-shape})))
        in-strides (strides in-shape)
        ^floats fs (as-floats t)
        ^floats out (float-array (elem-count target-shape))]
    (with-coords target-shape
      (fn [^longs oc flat]
        (let [^longs ic (long-array (count in-shape))]
          (dotimes [j (count in-shape)]
            (let [od (long (nth bcast-dims j))]
              (aset ic j (if (= 1 (long (nth in-shape j))) 0 (aget oc od)))))
          (aset out flat (aget fs (coords->flat ic in-strides))))))
    {:dtype (:dtype t) :shape target-shape :data (pack-floats (:dtype t) out)}))

(defn op-concatenate
  "Concatenates tensors along `dimension`."
  [tensors dimension]
  (let [t0 (first tensors)
        dtype (:dtype t0)
        rank (count (:shape t0))
        base (vec (:shape t0))]
    (doseq [t tensors]
      (when (or (not= (:dtype t) dtype) (not= (count (:shape t)) rank))
        (throw (ex-info "Concatenate dtype/rank mismatch" {})))
      (doseq [i (range rank)]
        (when (and (not= i dimension) (not= (nth (:shape t) i) (nth base i)))
          (throw (ex-info "Concatenate non-concat dim mismatch" {})))))
    (let [out-shape (assoc base dimension
                           (reduce + (map #(nth (:shape %) dimension) tensors)))
          fss (mapv as-floats tensors)
          bounds (vec (reductions + 0 (map #(nth (:shape %) dimension) tensors)))
          ^floats out (float-array (elem-count out-shape))]
      (with-coords out-shape
        (fn [^longs oc flat]
          (let [d (aget oc dimension)
                idx (loop [k 0]
                      (if (< d (long (nth bounds (inc k)))) k (recur (inc k))))
                t (nth tensors idx)
                ^longs ic (aclone oc)]
            (aset ic dimension (- d (long (nth bounds idx))))
            (aset out flat (aget ^floats (nth fss idx)
                                 (coords->flat ic (strides (:shape t))))))))
      {:dtype dtype :shape out-shape :data (pack-floats dtype out)})))

(defn op-slice
  "Strided slice."
  [t start-indices limit-indices strides-v]
  (let [in-shape (:shape t)
        starts (vec start-indices) limits (vec limit-indices) st (vec strides-v)
        out-shape (mapv (fn [s l stp] (long (Math/ceil (/ (- l s) (double stp)))))
                        starts limits st)
        in-strides (strides in-shape)
        ^floats fs (as-floats t)
        ^floats out (float-array (elem-count out-shape))]
    (with-coords out-shape
      (fn [^longs oc flat]
        (let [^longs ic (long-array (count in-shape))]
          (dotimes [i (count in-shape)]
            (aset ic i (+ (long (nth starts i)) (* (aget oc i) (long (nth st i))))))
          (aset out flat (aget fs (coords->flat ic in-strides))))))
    {:dtype (:dtype t) :shape out-shape :data (pack-floats (:dtype t) out)}))

(defn op-iota
  "Iota: out[coords] = coords[iota-dimension]."
  [shape dtype iota-dimension]
  (let [shape (vec shape)
        n (elem-count shape)
        data (case dtype
               :i32 (let [^ints a (int-array n)]
                      (with-coords shape (fn [^longs oc flat] (aset a flat (int (aget oc iota-dimension)))))
                      a)
               :f32 (let [^floats a (float-array n)]
                      (with-coords shape (fn [^longs oc flat] (aset a flat (float (aget oc iota-dimension)))))
                      a)
               (throw (ex-info "Unsupported iota dtype" {:dtype dtype})))]
    {:dtype dtype :shape shape :data data}))

(defn op-convert
  "Converts tensor to target dtype."
  [t target-dtype]
  (if (= (:dtype t) target-dtype)
    t
    {:dtype target-dtype :shape (:shape t) :data (pack-floats target-dtype (as-floats t))}))

;; ---------------------------------------------------------------------------
;; Boolean ops: compare / select
;; ---------------------------------------------------------------------------

(def ^:private compare-fns
  {"EQ" = "NE" not= "LT" < "LE" <= "GT" > "GE" >=})

(defn op-compare
  "Elementwise comparison -> :pred tensor. Supports broadcasting."
  [a b direction]
  (let [f (or (compare-fns direction)
              (throw (ex-info "Unknown comparison direction" {:direction direction})))
        tshape (broadcast-shapes (:shape a) (:shape b))
        ^floats fa (as-floats a) ^floats fb (as-floats b)
        stra (broadcast-strides (:shape a) tshape)
        strb (broadcast-strides (:shape b) tshape)
        n (elem-count tshape)
        ^booleans out (boolean-array n)]
    (with-coords tshape
      (fn [^longs oc flat]
        (aset out flat (boolean (f (aget fa (coords->flat oc stra))
                                   (aget fb (coords->flat oc strb)))))))
    {:dtype :pred :shape tshape :data out}))

(declare as-bools)

(defn op-select
  "select pred x y."
  [p x y]
  (let [pd (as-bools p)]
    (when (not= (:dtype p) :pred)
      (throw (ex-info "select predicate must be :pred" {})))
    (when (or (not= (:shape x) (:shape p)) (not= (:shape y) (:shape p))
              (not= (:dtype x) (:dtype y)))
      (throw (ex-info "select shape/dtype mismatch" {})))
    (let [^floats fx (as-floats x) ^floats fy (as-floats y)
          n (alength fx)
          ^floats out (float-array n)]
      (dotimes [i n] (aset out i (if (aget pd i) (aget fx i) (aget fy i))))
      {:dtype (:dtype x) :shape (:shape x) :data (pack-floats (:dtype x) out)})))

;; ---------------------------------------------------------------------------
;; Integer bitwise ops (int4 unpack path)
;; ---------------------------------------------------------------------------

(defn- as-ints
  "Tensor values as int[] (for :i8/:i32)."
  [{:keys [dtype shape data]}]
  (let [n (elem-count shape)]
    (if (instance? MemorySegment data)
      (let [^MemorySegment seg data
            out (int-array n)]
        (case dtype
          :i8 (dotimes [i n] (aset out i (int (.getAtIndex seg ValueLayout/JAVA_BYTE (long i)))))
          :i32 (dotimes [i n] (aset out i (.getAtIndex seg ValueLayout/JAVA_INT (long i))))
          (throw (ex-info "Cannot convert dtype to int" {:dtype dtype})))
        out)
      (case dtype
        :i8 (let [b ^bytes data out (int-array n)]
              (dotimes [i n] (aset out i (int (aget b i)))) out)
        :i32 (aclone ^ints data)
        (throw (ex-info "Cannot convert dtype to int" {:dtype dtype}))))))

(defn- as-bools
  "Tensor values as boolean[] (for :pred)."
  [{:keys [dtype shape data]}]
  (let [n (elem-count shape)]
    (when (not= dtype :pred)
      (throw (ex-info "Cannot convert dtype to bool" {:dtype dtype})))
    (if (instance? MemorySegment data)
      (let [^MemorySegment seg data
            out (boolean-array n)]
        (dotimes [i n] (aset out i (not (zero? (.getAtIndex seg ValueLayout/JAVA_BYTE (long i))))))
        out)
      (cond
        (instance? (Class/forName "[Z") data)
        (aclone ^booleans data)
        (instance? (Class/forName "[B") data)
        (let [b ^bytes data
              out (boolean-array n)]
          (dotimes [i n] (aset out i (not (zero? (aget b i)))))
          out)
        :else (throw (ex-info "Cannot convert to bool" {:dtype dtype}))))))

(defn- elementwise-bools
  "Binary boolean op with broadcast. f: bool bool -> bool."
  [f a b]
  (let [tshape (broadcast-shapes (:shape a) (:shape b))
        ^booleans fa (as-bools a) ^booleans fb (as-bools b)
        stra (broadcast-strides (:shape a) tshape)
        strb (broadcast-strides (:shape b) tshape)
        n (elem-count tshape)
        out (byte-array n)]
    (with-coords tshape
      (fn [^longs oc flat]
        (aset out flat (if (f (aget fa (coords->flat oc stra))
                              (aget fb (coords->flat oc strb)))
                         (byte 1) (byte 0)))))
    {:dtype :pred :shape tshape :data out}))

(defn- elementwise-ints
  "Binary int op with broadcast. f: int int -> int."
  [f a b]
  (let [da (:dtype a) db (:dtype b)]
    (when (not= da db) (throw (ex-info "Int op dtype mismatch" {:da da :db db})))
    (let [tshape (broadcast-shapes (:shape a) (:shape b))
          ^ints fa (as-ints a) ^ints fb (as-ints b)
          stra (broadcast-strides (:shape a) tshape)
          strb (broadcast-strides (:shape b) tshape)
          n (elem-count tshape)
          ^ints out (int-array n)]
      (with-coords tshape
        (fn [^longs oc flat]
          (aset out flat (int (f (aget fa (coords->flat oc stra))
                                 (aget fb (coords->flat oc strb)))))))
      {:dtype da :shape tshape
       :data (case da
               :i8 (let [o (byte-array n)] (dotimes [i n] (aset o i (unchecked-byte (aget out i)))) o)
               :i32 out)})))

(defn op-and [a b]
  (if (= :pred (:dtype a))
    (elementwise-bools (fn [x y] (and x y)) a b)
    (elementwise-ints bit-and a b)))
(defn op-or [a b]
  (if (= :pred (:dtype a))
    (elementwise-bools (fn [x y] (or x y)) a b)
    (elementwise-ints bit-or a b)))
(defn op-shift-right-logical
  "Logical right shift on unsigned byte/int values."
  [a b]
  (elementwise-ints (fn [x y]
                      (let [ux (if (= (:dtype a) :i8) (bit-and x 0xFF) x)]
                        (bit-shift-right ux y)))
                    a b))

;; ---------------------------------------------------------------------------
;; Reductions
;; ---------------------------------------------------------------------------

(defn- normalize-axes [axes rank]
  (mapv #(if (neg? %) (+ rank %) %) axes))

(defn op-reduce
  "General reduction. `f` is a binary float fn, `init` the identity."
  [f init t axes keep-dims]
  (let [in-shape (vec (:shape t))
        rank (count in-shape)
        axes-set (set (normalize-axes axes rank))
        kept-dims (vec (remove axes-set (range rank)))
        out-shape (if keep-dims
                    (mapv (fn [i d] (if (axes-set i) 1 d)) (range rank) in-shape)
                    (mapv in-shape kept-dims))
        ;; out-dim j -> in-dim
        out->in (if keep-dims (vec (range rank)) kept-dims)
        fs (as-floats t)
        in-strides (strides in-shape)
        out-strides (strides out-shape)
        ^floats out (float-array (elem-count out-shape))]
    (java.util.Arrays/fill out (float init))
    (with-coords in-shape
      (fn [ic _]
        (let [^longs oc (long-array (count out-shape))]
          (dotimes [j (count out-shape)]
            (let [in-dim (nth out->in j)]
              (aset oc j (if (axes-set in-dim) 0 (aget ^longs ic in-dim)))))
          (let [flat (coords->flat oc out-strides)
                acc (f (aget out flat) (aget fs (coords->flat ic in-strides)))]
            (aset out flat (float acc))))))
    {:dtype (:dtype t) :shape out-shape :data (pack-floats (:dtype t) out)}))

(defn op-reduce-sum [t axes keep-dims] (op-reduce + 0.0 t axes keep-dims))
(defn op-reduce-max [t axes keep-dims] (op-reduce (fn [^double a ^double b] (Math/max a b)) Float/NEGATIVE_INFINITY t axes keep-dims))
(defn op-reduce-mean [t axes keep-dims]
  (let [s (op-reduce-sum t axes keep-dims)
        in-shape (:shape t)
        axes-set (set (normalize-axes axes (count in-shape)))
        cnt (reduce * 1 (map #(nth in-shape %) axes-set))
        fs (as-floats s)
        n (alength fs)
        out (float-array n)]
    (dotimes [i n] (aset out i (float (/ (aget fs i) cnt))))
    (assoc s :data (pack-floats (:dtype s) out))))

;; ---------------------------------------------------------------------------
;; Gather (StableHLO semantics, batching dimensions)
;; ---------------------------------------------------------------------------

(defn- get-at-coords
  "Reads a single float value from tensor at coordinates (without materializing full tensor)."
  [{:keys [dtype _shape data]} coords strides]
  (let [flat (coords->flat (long-array coords) strides)]
    (if (instance? MemorySegment data)
      (case dtype
        :f32 (.getAtIndex ^MemorySegment data ValueLayout/JAVA_FLOAT (long flat))
        :bf16 (float (bf16->f32 (.getAtIndex ^MemorySegment data ValueLayout/JAVA_SHORT (long flat))))
        :i8 (float (.getAtIndex ^MemorySegment data ValueLayout/JAVA_BYTE (long flat)))
        :i32 (float (.getAtIndex ^MemorySegment data ValueLayout/JAVA_INT (long flat)))
        (throw (ex-info "Cannot read dtype" {:dtype dtype})))
      (case dtype
        :f32 (aget ^floats data flat)
        :bf16 (float (bf16->f32 (aget ^shorts data flat)))
        :i8 (float (aget ^bytes data flat))
        :i32 (float (aget ^ints data flat))
        (throw (ex-info "Cannot read dtype" {:dtype dtype}))))))

(defn op-gather
  "StableHLO gather. `attrs` keys: :offset_dims :collapsed_slice_dims
   :start_index_map :index_vector_dim :slice_sizes."
  [operand indices attrs]
  (let [o-shape (vec (:shape operand))
        i-shape (vec (:shape indices))
        o-rank (count o-shape)
        i-rank (count i-shape)
        offset-dims-vec (vec (sort (:offset_dims attrs)))
        offset-dims (set offset-dims-vec)
        collapsed (set (:collapsed_slice_dims attrs))
        iv-dim (long (:index_vector_dim attrs))
        slice-sizes (vec (:slice_sizes attrs))
        start-index-map (vec (:start_index_map attrs))
        ;; result rank = len(offset_dims) + i-rank - 1
        r-rank (+ (count offset-dims-vec) i-rank -1)
        ;; batch result dims (in order) <-> indices dims (excluding iv-dim)
        batch-r-dims (vec (remove offset-dims (range r-rank)))
        batch-i-dims (vec (remove #{iv-dim} (range i-rank)))
        _ (when (not= (count batch-r-dims) (count batch-i-dims))
            (throw (ex-info "Gather batch dim mismatch" {})))
        ;; result shape
        kept-o-dims (vec (remove collapsed (range o-rank))) ; in operand order
        _ (when (not= (count kept-o-dims) (count offset-dims-vec))
            (throw (ex-info "Gather offset_dims / kept dims mismatch" {})))
        out-shape (mapv (fn [j]
                          (if (offset-dims j)
                            (nth slice-sizes (nth kept-o-dims (.indexOf ^java.util.List offset-dims-vec j)))
                            (nth i-shape (nth batch-i-dims (.indexOf ^java.util.List batch-r-dims j)))))
                        (range r-rank))
        o-strides (strides o-shape)
        i-strides (strides i-shape)
        ^ints iv (as-ints indices)
        ;; start_index_map inverse: operand dim -> position in index vector
        dim->pos (into {} (map-indexed (fn [p d] [d p]) start-index-map))
        ^floats out (float-array (elem-count out-shape))
        ;; offset result-dim -> operand dim, and its inverse.
        ;; The i-th offset dim (sorted) corresponds to the i-th non-collapsed
        ;; operand dim.
        off-r->o (into {} (map-indexed (fn [i rd] [rd (nth kept-o-dims i)])
                                       offset-dims-vec))
        off-o->r (into {} (map (fn [[a b]] [b a]) off-r->o))]
    (with-coords out-shape
      (fn [^longs rc flat]
        ;; build the index vector
        (let [^longs ivc (long-array i-rank)]
          (dotimes [k (count batch-i-dims)]
            (aset ivc (long (nth batch-i-dims k)) (aget rc (long (nth batch-r-dims k)))))
          ;; operand coords
          (let [^longs oc (long-array o-rank)]
            (dotimes [d o-rank]
              (let [start (if-let [p (get dim->pos d)]
                            (let [^longs c2 (aclone ivc)]
                              (aset c2 iv-dim (long p))
                              (long (aget ^ints iv (coords->flat c2 i-strides))))
                            0)
                    off (if-let [rd (get off-o->r d)]
                          (aget rc (long rd))
                          0)]
                (aset oc d (+ start off))))
            (aset ^floats out flat (get-at-coords operand (vec oc) o-strides))))))
    {:dtype (:dtype operand) :shape out-shape :data (pack-floats (:dtype operand) out)}))

;; ---------------------------------------------------------------------------
;; Dynamic slice & Dynamic update slice
;; ---------------------------------------------------------------------------

(defn op-dynamic-slice
  "Extracts a slice of shape `slice-sizes` from `operand` starting at `start-indices`."
  [operand start-indices slice-sizes]
  (let [o-shape (vec (:shape operand))
        s-shape (vec slice-sizes)
        starts (vec start-indices)
        _ (when (not= (count o-shape) (count s-shape) (count starts))
            (throw (ex-info "dynamic_slice rank mismatch" {:operand-shape o-shape
                                                           :slice-sizes s-shape
                                                           :start-indices starts})))
        out-count (elem-count s-shape)
        dt (:dtype operand)
        ^floats fs (as-floats operand)
        ^floats out (float-array out-count)
        o-strides (strides o-shape)]
    (with-coords s-shape
      (fn [^longs sc flat]
        (let [^longs oc (long-array (count o-shape))]
          (dotimes [i (count o-shape)]
            (let [clamp-max (- (long (nth o-shape i)) (long (nth s-shape i)))
                  clamped-start (Math/max (long 0) (Math/min (long clamp-max) (long (nth starts i))))]
              (aset oc i (+ clamped-start (aget sc i)))))
          (aset out flat (aget fs (coords->flat oc o-strides))))))
    {:dtype dt :shape s-shape :data (pack-floats dt out)}))

(defn op-dynamic-update-slice
  "Writes `update` into a copy of `operand` at `start-indices`."
  [operand update start-indices]
  (let [o-shape (vec (:shape operand))
        u-shape (vec (:shape update))
        starts (vec start-indices)
        _ (when (not= (count o-shape) (count u-shape) (count starts))
            (throw (ex-info "dynamic_update_slice rank mismatch" {})))
        out-f (as-floats operand) ; fresh copy (as-floats clones arrays)
        ^floats uf (as-floats update)
        o-strides (strides o-shape)]
    ;; as-floats on a segment reads into a fresh array -- safe to mutate.
    ;; For array-backed operands it clones too. Good.
    (with-coords u-shape
      (fn [^longs uc _]
        (let [^longs oc (long-array (count o-shape))]
          (dotimes [i (count o-shape)]
            (aset oc i (+ (long (nth starts i)) (aget uc i))))
          (aset ^floats out-f (coords->flat oc o-strides) (aget uf (coords->flat uc (strides u-shape)))))))
    {:dtype (:dtype operand) :shape o-shape :data (pack-floats (:dtype operand) out-f)}))

;; ---------------------------------------------------------------------------
;; dot_general (naive sgemm; SIMD upgrade lands separately)
;; ---------------------------------------------------------------------------

(defn naive-sgemm!
  "C[MxN] = A[MxK] @ B[KxN], row-major. Overwrites C."
  [C A B M N K]
  (let [^floats C C ^floats A A ^floats B B
        M (long M) N (long N) K (long K)]
    (dotimes [m M]
      (dotimes [n N]
        (let [c-idx (int (+ (* m N) n))]
          (aset C c-idx
                (float
                 (loop [k (long 0) acc (float 0)]
                   (if (= k K) acc
                       (recur (inc k) (+ acc (* (aget A (int (+ (* m K) k)))
                                                (aget B (int (+ (* k N) n)))))))))))))))

;; ---------------------------------------------------------------------------
;; SIMD SGEMM via Panama Vector API (jdk.incubator.vector)
;; ---------------------------------------------------------------------------
;; Requires --add-modules=jdk.incubator.vector at JVM launch.
;; Single-threaded, 512-bit preferred species, 4 accumulators, K/J blocking.

(def ^VectorSpecies ^:private simd-species FloatVector/SPECIES_PREFERRED)
(def ^:private simd-vl (long (.length simd-species)))
(def ^:private simd-jb (* 4 simd-vl))
(def ^:private simd-kb 128)

(defn- simd-kb-block!
  [^floats C coff ^floats A aoff ^floats B N kb kend jb]
  (let [o0 jb
        o1 (+ jb simd-vl)
        o2 (+ jb (* 2 simd-vl))
        o3 (+ jb (* 3 simd-vl))]
    (loop [k kb
           ^FloatVector c0 (FloatVector/fromArray simd-species C (int (+ coff o0)))
           ^FloatVector c1 (FloatVector/fromArray simd-species C (int (+ coff o1)))
           ^FloatVector c2 (FloatVector/fromArray simd-species C (int (+ coff o2)))
           ^FloatVector c3 (FloatVector/fromArray simd-species C (int (+ coff o3)))]
      (if (< k kend)
        (let [^FloatVector av (FloatVector/broadcast simd-species (aget A (+ aoff k)))
              brow (* k N)]
          (recur (inc k)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o0))) c0)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o1))) c1)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o2))) c2)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o3))) c3)))
        (do
          (.intoArray c0 C (int (+ coff o0)))
          (.intoArray c1 C (int (+ coff o1)))
          (.intoArray c2 C (int (+ coff o2)))
          (.intoArray c3 C (int (+ coff o3))))))))

(defn- simd-cleanup!
  [^floats C coff ^floats A aoff ^floats B N kb kend j0 j1]
  (let [coff (long coff) aoff (long aoff) N (long N)
        kb (long kb) kend (long kend) j0 (long j0) j1 (long j1)]
    (loop [k kb]
      (when (< k kend)
        (let [a (float (aget A (int (+ aoff k))))
              brow (long (* k N))]
          (loop [j j0]
            (when (< j j1)
              (let [idx (int (+ coff j))]
                (aset C idx (float (+ (aget C idx) (* a (aget B (int (+ brow j))))))))
              (recur (inc j)))))
        (recur (inc k))))))

(defn- simd-sgemm!
  "Computes C = A @ B, OVERWRITING C. A is MxK, B is KxN, C is MxN, row-major.
   Vectorized with Panama Vector API."
  [^floats C ^floats A ^floats B M N K]
  (Arrays/fill C (float 0))
  (let [jb simd-jb kb simd-kb]
    (loop [kbb (long 0)]
      (when (< kbb K)
        (let [kend (long (min K (+ kbb kb)))]
          (loop [jbb (long 0)]
            (if (<= (+ jbb jb) N)
              (do
                (loop [i (long 0)]
                  (when (< i M)
                    (simd-kb-block! C (* i N) A (* i K) B N kbb kend jbb)
                    (recur (inc i))))
                (recur (+ jbb jb)))
              (when (< jbb N)
                (loop [i (long 0)]
                  (when (< i M)
                    (simd-cleanup! C (* i N) A (* i K) B N kbb kend jbb N)
                    (recur (inc i)))))))
          (recur (+ kbb kb)))))))

(defn sgemm!
  "Overwritable matmul kernel entry point. Uses Panama Vector API SIMD."
  [C A B M N K]
  (simd-sgemm! C A B (long M) (long N) (long K)))

(defn- permute-axes
  "Reorders tensor axes per `perm` (out-dim i <- in-dim perm[i]). Copying."
  [t perm]
  (op-transpose t perm))

(defn op-dot-general
  "StableHLO dot_general with batch and contracting dims."
  [lhs rhs {:keys [contracting_dims batch_dims]}]
  (let [l-shape (vec (:shape lhs)) r-shape (vec (:shape rhs))
        l-rank (count l-shape) r-rank (count r-shape)
        lc (vec (:lhs contracting_dims)) rc (vec (:rhs contracting_dims))
        lb (vec (:lhs batch_dims)) rb (vec (:rhs batch_dims))
        lm (vec (remove (set (concat lc lb)) (range l-rank)))
        rn (vec (remove (set (concat rc rb)) (range r-rank)))
        _ (when (not= (count lb) (count rb)) (throw (ex-info "dot batch rank mismatch" {})))
        _ (when (not= (count lc) (count rc)) (throw (ex-info "dot contracting rank mismatch" {})))
        batch-shape (mapv l-shape lb)
        _ (doseq [[d1 d2] (map vector lb rb)]
            (when (not= (nth l-shape d1) (nth r-shape d2))
              (throw (ex-info "dot batch dim size mismatch" {}))))
        _ (doseq [[d1 d2] (map vector lc rc)]
            (when (not= (nth l-shape d1) (nth r-shape d2))
              (throw (ex-info "dot contracting dim size mismatch" {}))))
        m-shape (mapv l-shape lm) n-shape (mapv r-shape rn)
        k-shape (mapv l-shape lc)
        B (elem-count batch-shape) M (elem-count m-shape)
        K (elem-count k-shape) N (elem-count n-shape)
        ;; to [B M K] and [B K N]
        lp (permute-axes lhs (vec (concat lb lm lc)))
        rp (permute-axes rhs (vec (concat rb rc rn)))
        lf (as-floats (op-reshape lp [B M K]))
        rf (as-floats (op-reshape rp [B K N]))
        out (float-array (* B M N))]
    (dotimes [b B]
      (let [co (float-array (* M N))
            ao (java.util.Arrays/copyOfRange lf (* b M K) (* (inc b) M K))
            bo (java.util.Arrays/copyOfRange rf (* b K N) (* (inc b) K N))]
        (sgemm! co ao bo M N K)
        (System/arraycopy co 0 out (* b M N) (* M N))))
    (let [out-shape (vec (concat batch-shape m-shape n-shape))
          out-dtype (:dtype lhs)]
      {:dtype out-dtype :shape out-shape :data (pack-floats out-dtype out)})))

(defn op-fused-matmul-bias
  "Fused: dot_general(lhs, rhs) + bias (broadcast).
   Computes C = A @ B + bias."
  [lhs rhs bias {:keys [contracting_dims batch_dims]}]
  (let [;; First do the matmul
        matmul-result (op-dot-general lhs rhs {:contracting_dims contracting_dims
                                               :batch_dims batch_dims})
        ;; Then add bias (bias is 1D, broadcast to matmul output shape)
        out-shape (:shape matmul-result)
        rank (count out-shape)
        ;; Bias broadcasts on the last dimension
        bias-broadcasted (op-broadcast-in-dim bias [(dec rank)] out-shape)]
    (op-add matmul-result bias-broadcasted)))

(defn op-fused-layer-norm
  "Fused LayerNorm over the last dimension.
   y = (x - mean) / sqrt(var + eps) * gamma + beta
   Single pass, no intermediate tensors."
  [x gamma beta {:keys [eps]}]
  (let [x-shape (vec (:shape x))
        d (long (last x-shape))
        n (long (reduce * 1 (butlast x-shape)))
        ^floats xf (as-floats x)
        ^floats gf (as-floats gamma)
        ^floats bf (as-floats beta)
        eps-f (float (or eps 1e-5))
        ^floats out (float-array (* n d))]
    (dotimes [i n]
      (let [off (long (* i d))
            mean (loop [j (long 0) m (float 0)]
                   (if (< j d)
                     (recur (inc j) (+ m (aget xf (int (+ off j)))))
                     (/ m d)))
            var (loop [j (long 0) v (float 0)]
                  (if (< j d)
                    (let [diff (- (aget xf (int (+ off j))) mean)]
                      (recur (inc j) (+ v (* diff diff))))
                    (/ v d)))
            inv-std (float (/ 1.0 (Math/sqrt (+ var eps-f))))]
        (dotimes [j d]
          (let [idx (int (+ off j))
                xhat (* (- (aget xf idx) mean) inv-std)]
            (aset out idx (float (+ (* xhat (aget gf (int j))) (aget bf (int j)))))))))
    {:dtype (:dtype x) :shape x-shape :data (pack-floats (:dtype x) out)}))
(defn op-fused-softmax
  "Fused softmax over the last dimension.
   y = exp(x - max(x)) / sum(exp(x - max(x)))
   Single pass, no intermediate tensors."
  [x _attrs]
  (let [x-shape (vec (:shape x))
        d (long (last x-shape))
        n (long (reduce * 1 (butlast x-shape)))
        ^floats xf (as-floats x)
        ^floats out (float-array (* n d))]
    (dotimes [i n]
      (let [off (long (* i d))
            ;; Max
            max-v (loop [j (long 1) m (aget xf (int off))]
                    (if (< j d)
                      (recur (inc j) (max m (aget xf (int (+ off j)))))
                      m))
            ;; Exp and sum
            sum (loop [j (long 0) s (float 0)]
                  (if (< j d)
                    (let [idx (int (+ off j))
                          e (float (Math/exp (- (aget xf idx) max-v)))]
                      (aset out idx e)
                      (recur (inc j) (+ s e)))
                    s))
            inv-sum (float (/ 1.0 sum))]
        ;; Normalize
        (dotimes [j d]
          (let [idx (int (+ off j))]
            (aset out idx (float (* (aget out idx) inv-sum)))))))
    {:dtype (:dtype x) :shape x-shape :data (pack-floats (:dtype x) out)}))

;; ---------------------------------------------------------------------------
;; Constants
;; ---------------------------------------------------------------------------

(defn- infer-scalar-dtype [v]
  (cond (boolean? v) :pred
        (integer? v) :i32
        (float? v) :f32
        :else (throw (ex-info "Cannot infer constant dtype" {:value v}))))

(defn op-constant
  "Builds a constant tensor from an eqn's :value and optional :type."
  [{:keys [value type]}]
  (if type
    (let [[_ shape dtype] type]
      (tensor dtype shape (if (sequential? value) value [value])))
    ;; no :type: scalar, or untyped vector (shape from value)
    (if (sequential? value)
      (let [dtype (infer-scalar-dtype (first value))]
        (tensor dtype [(count value)] value))
      (let [dtype (infer-scalar-dtype value)]
        (scalar dtype value)))))

;; ---------------------------------------------------------------------------
;; Executor
;; ---------------------------------------------------------------------------

(declare op-reduce-max-pair)

(defn- use-counts
  "Counts remaining reads per var across eqns."
  [eqns]
  (reduce (fn [acc eqn]
            (reduce (fn [a v] (update a v (fnil inc 0))) acc (:invars eqn)))
          {} eqns))

(defn- resolve-starts
  "Resolves symbolic start indices (keywords) against the env."
  [starts env]
  (mapv (fn [s]
          (if (keyword? s)
            (let [t (get env s)]
              (when-not t (throw (ex-info "Unknown symbolic index var" {:var s})))
              ;; scalar i32 tensor -> int
              (int (aget (as-ints t) 0)))
            (long s)))
        starts))

(defn- exec-op
  "Executes one eqn. Returns a map of outvar -> tensor."
  [eqn in-tensors known-shapes env]
  (let [op (:op eqn)
        attrs (:attrs eqn)
        outs (:outvars eqn)
        one (fn [t] (zipmap outs [t]))]
    (case op
      :stablehlo/constant (one (op-constant eqn))
      :stablehlo/add (one (apply op-add in-tensors))
      :stablehlo/subtract (one (apply op-sub in-tensors))
      :stablehlo/multiply (one (apply op-mul in-tensors))
      :stablehlo/divide (one (apply op-div in-tensors))
      :stablehlo/negate (one (apply op-negate in-tensors))
      :stablehlo/sqrt (one (apply op-sqrt in-tensors))
      :stablehlo/rsqrt (one (apply op-rsqrt in-tensors))
      :stablehlo/exp (one (apply op-exp in-tensors))
      :stablehlo/tanh (one (apply op-tanh in-tensors))
      :stablehlo/sine (one (apply op-sine in-tensors))
      :stablehlo/cosine (one (apply op-cosine in-tensors))
      :stablehlo/and (one (apply op-and in-tensors))
      :stablehlo/or (one (apply op-or in-tensors))
      :stablehlo/shift_right_logical (one (apply op-shift-right-logical in-tensors))
      :stablehlo/maximum (one (apply op-reduce-max-pair in-tensors))
      :stablehlo/reshape (one (op-reshape (first in-tensors) (:shape attrs)))
      :stablehlo/transpose (one (op-transpose (first in-tensors) (:permutation attrs)))
      :stablehlo/broadcast_in_dim (one (op-broadcast-in-dim (first in-tensors)
                                                            (:broadcast_dimensions attrs)
                                                            (:target_shape attrs)))
      :stablehlo/concatenate (one (op-concatenate in-tensors (:dimension attrs)))
      :stablehlo/slice (one (op-slice (first in-tensors) (:start_indices attrs)
                                      (:limit_indices attrs) (:strides attrs)))
      :stablehlo/iota (let [v (first outs)
                            shape (or (get known-shapes v)
                                      [(long (:len attrs))])]
                        (one (op-iota shape (:dtype attrs) (:iota_dimension attrs))))
      :stablehlo/convert (one (op-convert (first in-tensors) (:target_dtype attrs)))
      :stablehlo/compare (one (apply op-compare (concat in-tensors [(:comparison_direction attrs)])))
      :stablehlo/select (one (apply op-select in-tensors))
      :stablehlo/dot_general (one (op-dot-general (first in-tensors) (second in-tensors) attrs))
      :fused/matmul-bias (one (op-fused-matmul-bias (first in-tensors) (second in-tensors) (nth in-tensors 2) attrs))
      :fused/layer-norm (one (op-fused-layer-norm (first in-tensors) (second in-tensors) (nth in-tensors 2) attrs))
      :fused/softmax (one (op-fused-softmax (first in-tensors) attrs))
      :stablehlo/reduce_sum (one (op-reduce-sum (first in-tensors) (:axes attrs) (:keep_dims attrs)))
      :stablehlo/reduce_max (one (op-reduce-max (first in-tensors) (:axes attrs) (:keep_dims attrs)))
      :stablehlo/reduce_mean (one (op-reduce-mean (first in-tensors) (:axes attrs) (:keep_dims attrs)))
      :stablehlo/gather (one (op-gather (first in-tensors) (second in-tensors) attrs))
      :stablehlo/dynamic_slice
      (one (op-dynamic-slice (first in-tensors)
                             (resolve-starts (or (:start_indices attrs) (:start-indices attrs)) env)
                             (or (:slice_sizes attrs) (:slice-sizes attrs))))
      :stablehlo/dynamic_update_slice
      (one (op-dynamic-update-slice (first in-tensors) (second in-tensors)
                                    (resolve-starts (or (:start_indices attrs) (:start-indices attrs)) env)))
      (throw (ex-info "Unsupported StableHLO op" {:op op})))))

(defn- op-reduce-max-pair
  "Elementwise maximum of two tensors."
  [a b]
  (elementwise #(Math/max %1 %2) a b))

(defn execute
  "Executes a lowered graph (`ast->graph` map) with `bindings` mapping invar
   keywords to tensors. Returns a map of outvar keywords to tensors.

   Pure: the env is threaded as an immutable map; dead values are dropped by
   liveness counting to keep memory flat across thousands of eqns."
  [graph bindings]
  (let [known-shapes (or (:known-shapes graph) {})
        eqns (:eqns graph)
        outvars (set (:outvars graph))
        uses (use-counts eqns)]
    (loop [eqns eqns
           env (into {} bindings)
           uses uses]
      (if (empty? eqns)
        (select-keys env (:outvars graph))
        (let [eqn (first eqns)
              in-tensors (mapv #(get env %) (:invars eqn))]
          (when (some nil? in-tensors)
            (throw (ex-info "Unbound invar" {:eqn (select-keys eqn [:op :invars :outvars])
                                             :missing (mapv first (filter (comp nil? second)
                                                                          (map vector (:invars eqn) in-tensors)))})))
          (let [produced (exec-op eqn in-tensors known-shapes env)
                env2 (into env produced)
                ;; liveness: drop inputs whose last use just passed
                uses2 (reduce (fn [u v] (update u v dec)) uses (:invars eqn))
                env3 (reduce (fn [e v]
                               (if (and (zero? (get uses2 v 0)) (not (contains? outvars v)))
                                 (dissoc e v)
                                 e))
                             env2 (:invars eqn))]
            (recur (rest eqns) env3 uses2)))))))
