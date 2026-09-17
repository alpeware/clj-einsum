(ns clj-xla.logic.memory.sampler
  "Pure host-side relational batch sampling, multi-positive Target matrix construction,
   and false-negative masking for InfoNCE contrastive training (Repo Rule 2: Pure Functions).")

(defn sample-relation-batch
  "Samples K triples from a single relation.
   If n >= K: samples K distinct triples without replacement.
   If n < K: includes all n distinct triples and pads remaining K - n slots deterministically."
  [triples-vec k ^java.util.Random rnd]
  (let [n (count triples-vec)]
    (if (>= n k)
      (let [al (java.util.ArrayList. ^java.util.Collection triples-vec)]
        (java.util.Collections/shuffle al rnd)
        (vec (.subList al 0 k)))
      (let [al (java.util.ArrayList. ^java.util.Collection triples-vec)]
        (java.util.Collections/shuffle al rnd)
        (let [shuffled (vec al)]
          (mapv (fn [i] (nth shuffled (mod i n))) (range k)))))))

(defn build-multi-positive-target
  "Constructs a K x K Target matrix where row i puts uniform probability mass
   1.0 / m_i on all column positions j sharing the identical target token (:t-tok).
   Ensures every row sums strictly to 1.0 (eliminates self-as-negative penalty)."
  [batch k]
  (let [^floats target (float-array (* (long k) (long k)) 0.0)]
    (dotimes [i k]
      (let [ti (:t-tok (nth batch i))
            matching-indices (filterv #(= (:t-tok (nth batch %)) ti) (range k))
            m (count matching-indices)
            p (float (/ 1.0 (double m)))]
        (doseq [j matching-indices]
          (aset target (int (+ (* (long i) (long k)) (long j))) p))))
    target))

(defn build-hygienic-mask-scale
  "Constructs a K x K Mask_Scale matrix for InfoNCE adjoint scaling:
   - Scale factor: 1.0 / (K * tau)
   - Zeroes Mask_Scale[i, j] wherever (h_i, r, t_j) is in known-triples-set (false negative masking).
   - Preserves standard scale on all true positive positions and unmasked negative positions."
  [batch known-triples-set k tau]
  (let [scale (float (/ 1.0 (* (double k) (double tau))))
        ^floats mask (float-array (* (long k) (long k)) scale)]
    (dotimes [i k]
      (let [tr-i (nth batch i)
            hi (:h-tok tr-i)
            ri (:rel tr-i)
            ti (:t-tok tr-i)]
        (dotimes [j k]
          (let [tr-j (nth batch j)
                tj (:t-tok tr-j)]
            (when (and (not= tj ti)
                       (contains? known-triples-set [hi ri tj]))
              (aset mask (int (+ (* (long i) (long k)) (long j))) (float 0.0)))))))
    mask))
