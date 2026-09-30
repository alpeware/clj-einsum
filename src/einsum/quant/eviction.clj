(ns einsum.quant.eviction
  "Tier 2 Token Eviction: Attention Sinks & Pyramidal Heavy-Hitter Eviction
   (StreamingLLM / SnapKV / PyramidKV, arXiv:2309.17453, arXiv:2404.14469).
   Provides pure functions for selecting retained KV positions and slicing KV cache buffers.")

;; ==============================================================================
;; 1. Pyramidal Budget Schedule
;; ==============================================================================

(defn pyramidal-heavy-budget
  "Calculates heavy-hitter budget for layer l in [0, L-1] under pyramidal schedule:
   K_heavy(l) = round(K_base * (2 - l/L)).
   Lower layers retain up to 2 * K_base for diffuse attention;
   Deeper layers retain K_base for focused retrieval."
  ^long [^long layer-idx ^long num-layers ^long k-base]
  (let [l (double layer-idx)
        L (double (max 1 num-layers))
        factor (- 2.0 (/ l L))]
    (long (Math/round (* (double k-base) factor)))))

;; ==============================================================================
;; 2. Retained Index Selection (Sinks + Pyramidal Heavy Hitters + Local Window)
;; ==============================================================================

(def ^:private ^Class float-arr-cls (Class/forName "[F"))
(def ^:private ^Class double-arr-cls (Class/forName "[D"))

(defn- float-array? [x]
  (and (some? x) (identical? (.getClass ^Object x) float-arr-cls)))

(defn- double-array? [x]
  (and (some? x) (identical? (.getClass ^Object x) double-arr-cls)))

(defn- to-double-array ^doubles [v ^long n]
  (let [arr (double-array n)]
    (dotimes [i n]
      (aset-double arr i (double (nth v i 0.0))))
    arr))

(defn- heap-sift-up-float!
  [^ints heap ^floats mass ^long start-idx]
  (loop [i (int start-idx)]
    (when (pos? i)
      (let [parent (unchecked-int (bit-shift-right (unchecked-dec-int i) 1))
            val-i (aget mass (aget heap i))
            val-p (aget mass (aget heap parent))]
        (when (< val-i val-p)
          (let [tmp (aget heap i)]
            (aset-int heap i (aget heap parent))
            (aset-int heap parent tmp)
            (recur parent)))))))

(defn- heap-sift-down-float!
  [^ints heap ^floats mass ^long start-idx ^long n]
  (let [half (int (bit-shift-right (int n) 1))]
    (loop [i (int start-idx)]
      (when (< i half)
        (let [left (unchecked-int (unchecked-inc-int (bit-shift-left i 1)))
              right (unchecked-int (unchecked-inc-int left))
              child (if (and (< right (int n))
                             (< (aget mass (aget heap right))
                                (aget mass (aget heap left))))
                      right
                      left)]
          (when (< (aget mass (aget heap child))
                   (aget mass (aget heap i)))
            (let [tmp (aget heap i)]
              (aset-int heap i (aget heap child))
              (aset-int heap child tmp)
              (recur child))))))))

(defn- heap-sift-up-double!
  [^ints heap ^doubles mass ^long start-idx]
  (loop [i (int start-idx)]
    (when (pos? i)
      (let [parent (unchecked-int (bit-shift-right (unchecked-dec-int i) 1))
            val-i (aget mass (aget heap i))
            val-p (aget mass (aget heap parent))]
        (when (< val-i val-p)
          (let [tmp (aget heap i)]
            (aset-int heap i (aget heap parent))
            (aset-int heap parent tmp)
            (recur parent)))))))

(defn- heap-sift-down-double!
  [^ints heap ^doubles mass ^long start-idx ^long n]
  (let [half (int (bit-shift-right (int n) 1))]
    (loop [i (int start-idx)]
      (when (< i half)
        (let [left (unchecked-int (unchecked-inc-int (bit-shift-left i 1)))
              right (unchecked-int (unchecked-inc-int left))
              child (if (and (< right (int n))
                             (< (aget mass (aget heap right))
                                (aget mass (aget heap left))))
                      right
                      left)]
          (when (< (aget mass (aget heap child))
                   (aget mass (aget heap i)))
            (let [tmp (aget heap i)]
              (aset-int heap i (aget heap child))
              (aset-int heap child tmp)
              (recur child))))))))

(defn- select-retained-float
  [mass-arr total-len k-sink window k-heavy]
  (let [^floats mass-arr mass-arr
        total-len (long total-len)
        k-sink (long k-sink)
        window (long window)
        k-heavy (long k-heavy)
        window-start (- total-len window)
        heap (int-array k-heavy)]
    (loop [i (int k-sink)
           sz (int 0)]
      (when (< i (int window-start))
        (if (< sz (int k-heavy))
          (do
            (aset-int heap sz i)
            (heap-sift-up-float! heap mass-arr sz)
            (recur (unchecked-inc-int i) (unchecked-inc-int sz)))
          (do
            (let [min-idx (aget heap 0)
                  val-min (aget mass-arr min-idx)
                  val-i (aget mass-arr i)]
              (when (> val-i val-min)
                (aset-int heap 0 i)
                (heap-sift-down-float! heap mass-arr 0 k-heavy)))
            (recur (unchecked-inc-int i) sz)))))
    (java.util.Arrays/sort heap)
    (let [k-sink-int (int k-sink)
          k-heavy-int (int k-heavy)
          window-int (int window)
          total-retained (+ k-sink-int k-heavy-int window-int)
          out (int-array total-retained)]
      (dotimes [i k-sink-int] (aset-int out i (int i)))
      (System/arraycopy heap 0 out k-sink-int k-heavy-int)
      (let [w-off (+ k-sink-int k-heavy-int)
            w-start-int (int window-start)]
        (dotimes [i window-int] (aset-int out (+ w-off i) (int (+ w-start-int i)))))
      (vec out))))

(defn- select-retained-double
  [mass-arr total-len k-sink window k-heavy]
  (let [^doubles mass-arr mass-arr
        total-len (long total-len)
        k-sink (long k-sink)
        window (long window)
        k-heavy (long k-heavy)
        window-start (- total-len window)
        heap (int-array k-heavy)]
    (loop [i (int k-sink)
           sz (int 0)]
      (when (< i (int window-start))
        (if (< sz (int k-heavy))
          (do
            (aset-int heap sz i)
            (heap-sift-up-double! heap mass-arr sz)
            (recur (unchecked-inc-int i) (unchecked-inc-int sz)))
          (do
            (let [min-idx (aget heap 0)
                  val-min (aget mass-arr min-idx)
                  val-i (aget mass-arr i)]
              (when (> val-i val-min)
                (aset-int heap 0 i)
                (heap-sift-down-double! heap mass-arr 0 k-heavy)))
            (recur (unchecked-inc-int i) sz)))))
    (java.util.Arrays/sort heap)
    (let [k-sink-int (int k-sink)
          k-heavy-int (int k-heavy)
          window-int (int window)
          total-retained (+ k-sink-int k-heavy-int window-int)
          out (int-array total-retained)]
      (dotimes [i k-sink-int] (aset-int out i (int i)))
      (System/arraycopy heap 0 out k-sink-int k-heavy-int)
      (let [w-off (+ k-sink-int k-heavy-int)
            w-start-int (int window-start)]
        (dotimes [i window-int] (aset-int out (+ w-off i) (int (+ w-start-int i)))))
      (vec out))))

(defn select-retained-indices
  "Selects the monotonically sorted set of token indices to retain in the KV cache:
   - First K_sink = 4 tokens (<bos> + system prompt framing) permanently pinned
   - Most recent W_local tokens retained verbatim
   - Top K_heavy(l) heavy hitters from intermediate tokens based on cumulative attention mass.
   Returns a vector of sorted, unique 0-based token indices."
  ([layer-idx num-layers total-len attn-mass]
   (select-retained-indices layer-idx num-layers total-len attn-mass {}))
  ([layer-idx num-layers total-len attn-mass opts]
   (let [total-len (long total-len)
         k-sink (long (or (:k-sink opts) 4))
         window (long (or (:window opts) 1024))
         k-base (long (or (:k-base opts) 512))
         num-layers (long (or num-layers 54))
         layer-idx (long (or layer-idx 0))]
     (if (<= total-len (+ k-sink window))
       (vec (range total-len))
       (let [intermediate-count (- total-len window k-sink)
             k-heavy (min intermediate-count
                          (pyramidal-heavy-budget layer-idx num-layers k-base))]
         (if (<= intermediate-count k-heavy)
           (vec (range total-len))
           (cond
             (float-array? attn-mass)
             (select-retained-float attn-mass total-len k-sink window k-heavy)

             (double-array? attn-mass)
             (select-retained-double attn-mass total-len k-sink window k-heavy)

             :else
             (let [^doubles mass-arr (to-double-array attn-mass total-len)]
               (select-retained-double mass-arr total-len k-sink window k-heavy)))))))))

;; ==============================================================================
;; 3. KV Buffer Slicing & Eviction
;; ==============================================================================

(defn- slice-buffer
  [retained buf]
  (cond
    ;; 4D tensor: [batch, seq, heads, dim]
    (and (vector? buf) (vector? (first buf)) (vector? (first (first buf))))
    (mapv (fn [batch-elem]
            (mapv (fn [idx] (nth batch-elem idx)) retained))
          buf)

    ;; 2D: [batch, seq]
    (and (vector? buf) (vector? (first buf)))
    (mapv (fn [batch-elem]
            (mapv (fn [idx] (nth batch-elem idx)) retained))
          buf)

    ;; 1D: [tok0, tok1, ...]
    (sequential? buf)
    (mapv (fn [idx] (nth buf idx)) retained)

    :else buf))

(defn evict-kv-cache-buffers
  "Slices a sequence, vector, or map of KV buffer tensors along dimension 1 (sequence length),
   retaining only the specified indices.
   Host-side buffers (nested persistent vectors / sequences) are sliced into new contiguous structures.
   Note: In-place accelerator device buffer compaction via PJRT DynamicUpdateSlice is staged as future work."
  [retained-indices kv-buffers]
  (let [retained (vec (sort retained-indices))
        n-retained (count retained)
        sliced (cond
                 (map? kv-buffers)
                 (into {} (map (fn [[k buf]] [k (slice-buffer retained buf)]) kv-buffers))

                 ;; Vector of buffers (where each element is a tensor or layer buffer):
                 (and (vector? kv-buffers) (vector? (first kv-buffers)))
                 (mapv #(slice-buffer retained %) kv-buffers)

                 (sequential? kv-buffers)
                 (slice-buffer retained kv-buffers)

                 :else kv-buffers)]
    {:retained-indices retained
     :retained-count n-retained
     :kv-buffers sliced}))
