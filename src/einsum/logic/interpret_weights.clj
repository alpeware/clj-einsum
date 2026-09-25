(ns einsum.logic.interpret-weights
  "Zero-copy weight bindings for the JVM StableHLO interpreter.

  Memory-maps a .safetensors file and exposes each tensor as an interpreter
  tensor map {:dtype ... :shape [...] :data <MemorySegment slice>} pointing
  directly into the mmap'd region. No weight bytes are copied into the JVM
  heap; the OS page cache is the backing store and pages are reclaimable
  under memory pressure."
  (:require [einsum.runtime.safetensors :as st])
  (:import [java.lang.foreign Arena MemorySegment]))

(def ^:private dtype-map
  {"F32"      :f32
   "BF16"     :bf16
   "BFLOAT16" :bf16
   "I8"       :i8
   "U8"       :i8
   "I32"      :i32
   "I64"      :i32
   "BOOL"     :pred})

(defn slice-aligned?
  "Checks if a MemorySegment slice is aligned for the given dtype's element size."
  [^MemorySegment slice elem-size]
  (let [addr (.address slice)]
    (zero? (mod addr elem-size))))

(defn copy-slice-to-array
  "Copies a MemorySegment slice to a heap array (for unaligned data).
   Uses byte-level access to avoid alignment faults."
  [^MemorySegment slice dtype n-elems]
  (let [^java.nio.ByteBuffer bb (-> (.asByteBuffer slice)
                                    (.order java.nio.ByteOrder/LITTLE_ENDIAN))]
    (case dtype
      :f32 (let [arr (float-array n-elems)]
             (dotimes [i n-elems] (aset arr i (.getFloat bb)))
             arr)
      :bf16 (let [arr (short-array n-elems)]
              (dotimes [i n-elems] (aset arr i (.getShort bb)))
              arr)
      :i32 (let [arr (int-array n-elems)]
             (dotimes [i n-elems] (aset arr i (.getInt bb)))
             arr)
      :i8 (let [arr (byte-array n-elems)]
            (.get bb arr)
            arr)
      ;; fallback: raw bytes
      (let [arr (byte-array (.byteSize slice))]
        (.get bb arr)
        arr))))

(defn align-slice
  "Ensures a MemorySegment slice is aligned for its dtype, copying to a heap array only if unaligned."
  [^MemorySegment slice dtype shape]
  (let [elem-size (case dtype :f32 4 :bf16 2 :i32 4 :i8 1 :pred 1 4)
        n-elems (reduce * 1 shape)]
    (if (and (> elem-size 1) (not (slice-aligned? slice elem-size)))
      (copy-slice-to-array slice dtype n-elems)
      slice)))

(defn- header->tensor
  "Builds an interpreter tensor from safetensors header info and a segment slice.
   If the slice is not aligned for the dtype, copies to a heap array."
  [t-name t-info ^MemorySegment slice]
  (let [dtype (get dtype-map (get t-info "dtype") :f32)
        shape (mapv long (get t-info "shape"))
        data (align-slice slice dtype shape)]
    {:dtype dtype
     :shape shape
     :data data
     :name t-name}))

(defn load-weights
  "Memory-maps safetensors-path using arena and returns a map of
   tensor-name -> interpreter tensor with zero-copy MemorySegment data.
   The arena must stay open for as long as the tensors are used."
  [safetensors-path ^Arena arena]
  (let [{:keys [header segment]} (st/map-safetensors-weights safetensors-path arena)]
    (into {}
          (comp (remove (fn [[t-name _]] (= t-name "__metadata__")))
                (map (fn [[t-name t-info]]
                       (let [[start end] (get t-info "data_offsets")
                             len (- (long end) (long start))
                             slice (.asSlice ^MemorySegment segment (long start) len)]
                         [t-name (header->tensor t-name t-info slice)]))))
          header)))

(defn load-weights-shared
  "Like load-weights but creates its own shared Arena and returns
   [weights close-fn]. Call close-fn when done."
  [safetensors-path]
  (let [arena (Arena/ofShared)
        weights (load-weights safetensors-path arena)]
    [weights #(.close arena)]))
