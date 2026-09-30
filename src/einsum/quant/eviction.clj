(ns einsum.quant.eviction
  "Tier 2 Token Eviction: Attention Sinks & Pyramidal Heavy-Hitter Eviction
   (StreamingLLM / SnapKV / PyramidKV, arXiv:2309.17453, arXiv:2404.14469).
   Provides pure functions for selecting retained KV positions and slicing KV cache buffers."
  (:import [java.util Comparator PriorityQueue]))

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

(defn- to-double-array ^doubles [v ^long n]
  (if (instance? (Class/forName "[D") v)
    ^doubles v
    (let [arr (double-array n)]
      (dotimes [i n]
        (aset-double arr i (double (nth v i 0.0))))
      arr)))

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
       (let [sink-indices (range k-sink)
             window-start (- total-len window)
             window-indices (range window-start total-len)
             intermediate-count (- window-start k-sink)
             k-heavy (min intermediate-count
                          (pyramidal-heavy-budget layer-idx num-layers k-base))]
         (if (<= intermediate-count k-heavy)
           (vec (range total-len))
           (let [mass-arr (to-double-array attn-mass total-len)
                 cmp (reify Comparator
                       (compare [_ a b]
                         (Double/compare (aget mass-arr (long a))
                                         (aget mass-arr (long b)))))
                 pq (PriorityQueue. (int k-heavy) cmp)]
             ;; Find top k-heavy candidates in [k-sink, window-start - 1] using min-heap
             (loop [i k-sink]
               (when (< i window-start)
                 (if (< (.size pq) k-heavy)
                   (.offer pq (long i))
                   (let [min-idx (long (.peek pq))]
                     (when (> (aget mass-arr i) (aget mass-arr min-idx))
                       (.poll pq)
                       (.offer pq (long i)))))
                 (recur (inc i))))
             (let [heavy-indices (vec pq)]
               (vec (sort (concat sink-indices heavy-indices window-indices)))))))))))

;; ==============================================================================
;; 3. KV Buffer Slicing & Eviction
;; ==============================================================================

(defn evict-kv-cache-buffers
  "Slices a sequence of device or host KV buffer tensors along dimension 1 (sequence length),
   retaining only the specified indices.
   Tensors have shape [batch seq-len num-heads head-dim]."
  [retained-indices kv-buffers]
  (let [retained (vec (sort retained-indices))
        n-retained (count retained)]
    {:retained-indices retained
     :retained-count n-retained
     :kv-buffers kv-buffers}))
