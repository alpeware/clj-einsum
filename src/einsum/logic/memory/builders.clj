(ns einsum.logic.memory.builders
  "Buffer builders and static entity tables for relational memory."
  (:require [einsum.runtime.tokenizer.protocol :as tok-proto]))

(defn build-random-projection
  "Generates a fixed [in-dim x d] random projection matrix from Gaussian samples
   scaled by 1/sqrt(in-dim) using a seeded RNG. Pure function of (in-dim, d, seed)."
  [in-dim d seed]
  (let [in-dim (long in-dim)
        d (long d)
        scale (float (/ 1.0 (Math/sqrt (double in-dim))))
        rnd (java.util.Random. (long seed))
        arr (float-array (* in-dim d))]
    (dotimes [i (* in-dim d)]
      (aset-float arr i (float (* (.nextGaussian rnd) scale))))
    arr))

(defn build-entity-token-table
  "Constructs a static [n_entities x vocab_size] float array mapping each entity
   to its constituent tokens uniformly with 1.0 weight."
  [tokenizer entities vocab-size]
  (let [n (count entities)
        v (long vocab-size)
        table (float-array (* n v))]
    (dotimes [i n]
      (let [entity-name (nth entities i)
            token-ids (tok-proto/encode tokenizer (str " " entity-name))
            row-offset (* i v)]
        (doseq [tid token-ids]
          (let [t-idx (long tid)]
            (when (and (>= t-idx 0) (< t-idx v))
              (aset-float table (+ row-offset t-idx) (float 1.0)))))))
    table))
