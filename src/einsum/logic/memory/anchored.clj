(ns einsum.logic.memory.anchored
  "LLM-anchored entity representations, Modified Gram-Schmidt QR orthonormalization,
   and cosine geometry statistics for relational memory."
  (:require [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.protocol :refer [bos-id encode]]
            [clojure.string :as str])
  (:import [java.lang.foreign MemorySegment]))

(defn build-identity-matrix
  "Returns a flat float array of shape [dim, dim] representing the identity matrix I."
  [dim]
  (let [d (long dim)
        arr (float-array (* d d))]
    (dotimes [i d]
      (aset-float arr (+ (* i d) i) (float 1.0)))
    arr))

(defn mean-pool-embeddings
  "Computes element-wise arithmetic mean across a list of equal-length float arrays."
  [vectors hidden-dim]
  (let [cnt (count vectors)
        h-dim (long hidden-dim)
        out (float-array h-dim)]
    (when (pos? cnt)
      (dotimes [i h-dim]
        (let [sum (loop [idx 0 s 0.0]
                    (if (>= idx cnt)
                      s
                      (recur (inc idx) (+ s (double (aget ^floats (nth vectors idx) i))))))]
          (aset-float out i (float (/ sum (double cnt)))))))
    out))

(defn l2-normalize
  "Normalizes float vector to Euclidean length 1.0. Zero vector remains zero."
  [^floats v]
  (let [len (alength v)
        sum-sq (loop [i 0 s 0.0]
                 (if (>= i len)
                   s
                   (let [x (double (aget v i))]
                     (recur (inc i) (+ s (* x x))))))
        norm (Math/sqrt sum-sq)
        scale (if (> norm 1e-12) (/ 1.0 norm) 0.0)
        out (float-array len)]
    (dotimes [i len]
      (aset-float out i (float (* (double (aget v i)) scale))))
    out))

(defn read-embed-row-floats
  "Reads a single token embedding row from an off-heap embed-tokens MemorySegment or reader function."
  [weights-source token-id hidden-dim]
  (cond
    (fn? weights-source)
    (weights-source token-id)

    (instance? MemorySegment weights-source)
    (let [h-dim (long hidden-dim)
          offset (* (long token-id) h-dim 2)
          ^MemorySegment row-seg (.asSlice ^MemorySegment weights-source (long offset) (long (* h-dim 2)))
          sa (short-array h-dim)
          ^MemorySegment target (MemorySegment/ofArray sa)
          _ (.copyFrom target row-seg)
          fa (float-array h-dim)]
      (dotimes [i h-dim]
        (let [s (int (aget sa i))
              bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
          (aset-float fa i (Float/intBitsToFloat bits))))
      fa)

    :else
    (throw (ex-info "Unsupported weights-source type" {:type (type weights-source)}))))

(defn build-llm-anchored-table
  "Builds relation memory map initialized with LLM-anchored entity embeddings derived from
   the model's tied language_model.embed_tokens.weight."
  ([weights-source prefix-base tokenizer entities hidden-dim]
   (build-llm-anchored-table weights-source prefix-base tokenizer entities hidden-dim 3))
  ([weights-source prefix-base tokenizer entities hidden-dim relation-count]
   (let [n (count entities)
         d (long hidden-dim)
         k (long relation-count)
         entity-table (float-array (* n d))
         embed-slice (cond
                       (instance? MemorySegment weights-source)
                       weights-source

                       (fn? weights-source)
                       weights-source

                       (map? weights-source)
                       (let [p-base (or prefix-base
                                        (if (contains? (:header weights-source) "model.language_model.embed_tokens.weight")
                                          "model.language_model."
                                          "model."))
                             tname (str p-base "embed_tokens.weight")]
                         (st/get-tensor-slice weights-source tname))

                       :else
                       (throw (ex-info "Invalid weights-source for anchored table" {:source weights-source})))]

     (dotimes [idx n]
       (let [entity-name (nth entities idx)
             b-id (bos-id tokenizer)
             raw-ids (encode tokenizer (str " " (str/trim entity-name)))
             clean-ids (vec (remove #(= % b-id) raw-ids))
             token-vectors (mapv #(read-embed-row-floats embed-slice % d) clean-ids)
             pooled (mean-pool-embeddings token-vectors d)
             normed (l2-normalize pooled)
             row-offset (* idx d)]
         (dotimes [col d]
           (aset-float entity-table (+ row-offset col) (aget ^floats normed col)))))

     {:entity-table entity-table
      :entity-shape [n d]
      :cores (float-array (* k d d))
      :core-shape [k d d]
      :entity-count n
      :dim d
      :relation-count k})))

(defn orthonormalize-table
  "Applies Modified Gram-Schmidt (MGS) in f64 to orthonormalize the rows of an [N, D] table.
   Handles degenerate inputs (duplicate/parallel/zero rows) via a deterministic canonical basis
   search policy to guarantee no NaNs and strictly orthonormal rows."
  [^floats table n d]
  (let [n-long (long n)
        d-long (long d)
        q-rows (object-array n-long)
        eps 1e-12]
    (dotimes [i n-long]
      (let [v (double-array d-long)
            offset (* i d-long)]
        ;; 1. Copy row i in double precision
        (dotimes [k d-long]
          (aset v k (double (aget table (+ offset k)))))

        ;; 2. Modified Gram-Schmidt: sequentially project out prior q_j
        (dotimes [j i]
          (let [^doubles q-j (aget q-rows j)
                dot (loop [k 0 s 0.0]
                      (if (>= k d-long)
                        s
                        (recur (inc k) (+ s (* (aget v k) (aget q-j k))))))]
            (dotimes [k d-long]
              (aset v k (- (aget v k) (* dot (aget q-j k)))))))

        ;; 3. Compute Euclidean norm
        (let [norm (Math/sqrt (loop [k 0 s 0.0]
                                (if (>= k d-long)
                                  s
                                  (let [val (aget v k)]
                                    (recur (inc k) (+ s (* val val)))))))]
          ;; 4. Check for degeneracy (norm < eps)
          (if (< norm eps)
            (let [resolved (double-array d-long)]
              (loop [m 0]
                (if (>= m d-long)
                  (throw (ex-info "Failed to resolve degenerate row in MGS" {:row i :d d-long}))
                  (do
                    (dotimes [k d-long]
                      (aset resolved k (if (= k m) 1.0 0.0)))
                    (dotimes [j i]
                      (let [^doubles q-j (aget q-rows j)
                            dot (aget q-j m)]
                        (dotimes [k d-long]
                          (aset resolved k (- (aget resolved k) (* dot (aget q-j k)))))))
                    (let [c-norm (Math/sqrt (loop [k 0 s 0.0]
                                              (if (>= k d-long)
                                                s
                                                (let [val (aget resolved k)]
                                                  (recur (inc k) (+ s (* val val)))))))]
                      (if (> c-norm 1e-6)
                        (dotimes [k d-long]
                          (aset v k (/ (aget resolved k) c-norm)))
                        (recur (inc m))))))))
            (dotimes [k d-long]
              (aset v k (/ (aget v k) norm)))))

        (aset q-rows i v)))

    ;; 5. Convert to output flat float-array
    (let [out (float-array (* n-long d-long))]
      (dotimes [i n-long]
        (let [^doubles q-i (aget q-rows i)
              row-offset (* i d-long)]
          (dotimes [k d-long]
            (aset-float out (+ row-offset k) (float (aget q-i k))))))
      out)))

(defn init-qr-anchored-memory
  "Builds relation memory map with the same shape as mem/init-relation-memory,
   where :entity-table is the orthonormalized table Q derived from raw-anchored-mem."
  [raw-anchored-mem]
  (let [n (long (:entity-count raw-anchored-mem))
        d (long (:dim raw-anchored-mem))
        k (long (:relation-count raw-anchored-mem))
        q-table (orthonormalize-table (:entity-table raw-anchored-mem) n d)]
    {:entity-table q-table
     :entity-shape [n d]
     :cores (float-array (* k d d))
     :core-shape [k d d]
     :entity-count n
     :dim d
     :relation-count k}))

(defn compute-pairwise-cosine-stats
  "Computes mean, max, and min pairwise cosine similarities across all pairs in table [N, D]."
  [^floats table n dim]
  (let [n-long (long n)
        d-long (long dim)
        norms (float-array n-long)
        _ (dotimes [i n-long]
            (let [offset (* i d-long)
                  sum-sq (loop [k 0 s 0.0]
                           (if (>= k d-long)
                             s
                             (let [v (double (aget table (+ offset k)))]
                               (recur (inc k) (+ s (* v v))))))]
              (aset-float norms i (float (Math/sqrt sum-sq)))))
        cosines (atom [])]
    (dotimes [i n-long]
      (let [i-offset (* i d-long)
            norm-i (double (aget norms i))]
        (loop [j (inc i)]
          (when (< j n-long)
            (let [j-offset (* j d-long)
                  norm-j (double (aget norms j))
                  dot (loop [k 0 s 0.0]
                        (if (>= k d-long)
                          s
                          (let [vi (double (aget table (+ i-offset k)))
                                vj (double (aget table (+ j-offset k)))]
                            (recur (inc k) (+ s (* vi vj))))))
                  denom (* norm-i norm-j)
                  cos-sim (if (> denom 1e-12) (/ dot denom) 0.0)]
              (swap! cosines conj cos-sim)
              (recur (inc j)))))))
    (let [vals @cosines
          cnt (count vals)]
      (if (zero? cnt)
        {:mean 0.0 :max 0.0 :min 0.0}
        (let [sorted-vals (sort vals)
              mean-val (/ (reduce + vals) (double cnt))
              max-val (last sorted-vals)
              min-val (first sorted-vals)]
          {:mean (double mean-val)
           :max (double max-val)
           :min (double min-val)})))))

(defn compute-margin-stats
  "Computes factual margin statistics for target vs top distractor scores."
  [targets top-distractors]
  (let [margins (mapv (fn [t d] (- (double t) (double d))) targets top-distractors)
        cnt (count margins)]
    (if (zero? cnt)
      {:mean 0.0 :min 0.0 :max 0.0 :positive-rate 0.0}
      (let [pos-cnt (count (filter pos? margins))]
        {:mean (/ (reduce + margins) (double cnt))
         :min (apply min margins)
         :max (apply max margins)
         :positive-rate (/ (double pos-cnt) (double cnt))}))))

(defn compute-score-scale-stats
  "Computes score-scale statistics across a sequence of query evaluation results:
     - mean top-1 score
     - std dev of top-1 score
     - fixed threshold (e.g. 0.5) pass count and rate
     - calibrated threshold (0.5 * mean top-1 score) pass count and rate"
  [results fixed-threshold]
  (let [scores (mapv (fn [r] (double (or (:top-1-score r) 0.0))) results)
        cnt (count scores)]
    (if (zero? cnt)
      {:mean 0.0 :std 0.0
       :fixed-threshold (double fixed-threshold) :fixed-passes 0 :fixed-pass-rate 0.0
       :calibrated-threshold 0.0 :calibrated-passes 0 :calibrated-pass-rate 0.0}
      (let [mean (/ (reduce + scores) (double cnt))
            var-sum (reduce + (map (fn [s] (let [diff (- s mean)] (* diff diff))) scores))
            std (Math/sqrt (/ var-sum (double cnt)))
            cal-thresh (* 0.5 mean)
            f-passes (count (filter #(> % (double fixed-threshold)) scores))
            c-passes (count (filter #(> % cal-thresh) scores))]
        {:mean (double mean)
         :std (double std)
         :fixed-threshold (double fixed-threshold)
         :fixed-passes f-passes
         :fixed-pass-rate (/ (double f-passes) (double cnt))
         :calibrated-threshold (double cal-thresh)
         :calibrated-passes c-passes
         :calibrated-pass-rate (/ (double c-passes) (double cnt))}))))
