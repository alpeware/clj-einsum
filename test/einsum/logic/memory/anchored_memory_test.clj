(ns einsum.logic.memory.anchored-memory-test
  "Unit, generative, and invariant tests for LLM-anchored entity embeddings in tools.poc-anchored-memory."
  (:require [einsum.logic.memory.relation :as mem]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.logic.memory.anchored :as anchored]))

;; ==============================================================================
;; 1. Mean-Pooling Over Token Spans (Synthetic & Hand-Computed)
;; ==============================================================================

(deftest test-mean-pool-embeddings-synthetic
  (testing "mean-pool-embeddings computes exact arithmetic mean across vector list"
    (let [v1 (float-array [1.0 2.0 3.0 4.0])
          v2 (float-array [3.0 4.0 5.0 6.0])
          v3 (float-array [5.0 6.0 7.0 8.0])
          dim 4
          res (anchored/mean-pool-embeddings [v1 v2 v3] dim)]
      (is (= [3.0 4.0 5.0 6.0]
             (mapv #(double (aget ^floats res %)) (range dim))))))

  (testing "mean-pool-embeddings handles single vector"
    (let [v (float-array [10.0 -5.0 0.5])
          res (anchored/mean-pool-embeddings [v] 3)]
      (is (= [10.0 -5.0 0.5]
             (mapv #(double (aget ^floats res %)) (range 3)))))))

;; ==============================================================================
;; 2. Identity Matrix Builder & Invariants
;; ==============================================================================

(deftest test-build-identity-matrix-synthetic
  (testing "build-identity-matrix returns exact identity matrix on small dim"
    (let [dim 3
          id-mat (anchored/build-identity-matrix dim)]
      (is (= 9 (alength ^floats id-mat)))
      ;; Diagonal elements must be 1.0
      (is (== 1.0 (aget ^floats id-mat 0)))
      (is (== 1.0 (aget ^floats id-mat 4)))
      (is (== 1.0 (aget ^floats id-mat 8)))
      ;; Off-diagonal elements must be 0.0
      (is (== 0.0 (aget ^floats id-mat 1)))
      (is (== 0.0 (aget ^floats id-mat 2)))
      (is (== 0.0 (aget ^floats id-mat 3)))
      (is (== 0.0 (aget ^floats id-mat 5)))
      (is (== 0.0 (aget ^floats id-mat 6)))
      (is (== 0.0 (aget ^floats id-mat 7))))))

(defspec prop-identity-matrix-invariants 25
  (prop/for-all [dim (gen/choose 1 16)]
                (let [m (anchored/build-identity-matrix dim)]
                  (and (= (* dim dim) (alength ^floats m))
           ;; Diagonal sum (trace) == dim
                       (== (double dim)
                           (double (reduce + (map (fn [i] (double (aget ^floats m (+ (* i dim) i))))
                                                  (range dim)))))
           ;; Off-diagonal sum == 0.0
                       (zero? (reduce + (for [i (range dim)
                                              j (range dim)
                                              :when (not= i j)]
                                          (Math/abs (double (aget ^floats m (+ (* i dim) j)))))))))))

;; ==============================================================================
;; 3. Pairwise Cosine Diagnostics (Synthetic Verification)
;; ==============================================================================

(deftest test-pairwise-cosine-stats-synthetic
  (testing "Orthonormal basis vectors produce pairwise cosine mean 0.0 and max 0.0"
    (let [dim 3
          n 3
          table (float-array [1.0 0.0 0.0
                              0.0 1.0 0.0
                              0.0 0.0 1.0])
          stats (anchored/compute-pairwise-cosine-stats table n dim)]
      (is (< (Math/abs (double (:mean stats))) 1e-6))
      (is (< (Math/abs (double (:max stats))) 1e-6))
      (is (< (Math/abs (double (:min stats))) 1e-6))))

  (testing "Identical collinear vectors produce pairwise cosine 1.0"
    (let [dim 2
          n 2
          table (float-array [1.0 2.0
                              1.0 2.0])
          stats (anchored/compute-pairwise-cosine-stats table n dim)]
      (is (< (Math/abs (- (double (:mean stats)) 1.0)) 1e-6))
      (is (< (Math/abs (- (double (:max stats)) 1.0)) 1e-6))
      (is (< (Math/abs (- (double (:min stats)) 1.0)) 1e-6)))))

;; ==============================================================================
;; 4. Span Round-Trip for All 14 Entities (Heads and Tails)
;; ==============================================================================

(deftest test-span-round-trip-all-14-entities
  (testing "All 14 entity names in wiki_recent_triples.edn decode back from their token spans"
    (let [model-dir ".models/gemma-4-E2B-it"
          triples-file (io/resource "data/wiki_recent_triples.edn")]
      (when (and (.exists (io/file model-dir)) triples-file)
        (let [tokenizer (tok/from-file model-dir)
              kb-data (edn/read-string (slurp triples-file))
              entities (:entities kb-data)]
          (is (= 14 (count entities)))
          (doseq [name entities]
            (let [b-id (bos-id tokenizer)
                  raw-ids (encode tokenizer (str " " name))
                  clean-ids (vec (remove #(= % b-id) raw-ids))
                  decoded (str/trim (decode tokenizer clean-ids))]
              (is (pos? (count clean-ids)) (str "Token ids for '" name "' must not be empty"))
              (is (= name decoded)
                  (str "Decoded token span '" decoded "' must match original entity name '" name "'")))))))))

;; ==============================================================================
;; 5. Purity, De-Oracle Independence, and Determinism
;; ==============================================================================

(deftest test-anchored-table-purity-and-de-oracle
  (testing "Row embedding for an entity is invariant to entity position and ordering"
    (let [dim 4
          vocab-mock {100 (float-array [1.0 0.0 0.0 0.0])
                      101 (float-array [0.0 2.0 0.0 0.0])
                      102 (float-array [0.0 0.0 3.0 0.0])
                      103 (float-array [0.0 0.0 0.0 4.0])}
          reader-fn (fn [tid] (get vocab-mock tid (float-array dim)))
          tok-mock (reify einsum.runtime.tokenizer.protocol/Tokenizer
                     (encode [_ text]
                       (case (str/trim text)
                         "Alpha" [100 101]
                         "Beta"  [102]
                         "Gamma" [103]
                         [100]))
                     (encode [this text _] (encode this text))
                     (decode [_ _] "")
                     (decode [_ _ _] "")
                     (bos-id [_] -1)
                     (eos-id [_] -2))
          entities1 ["Alpha" "Beta" "Gamma"]
          entities2 ["Gamma" "Alpha" "Beta"]
          res1 (anchored/build-llm-anchored-table reader-fn nil tok-mock entities1 dim 1)
          res2 (anchored/build-llm-anchored-table reader-fn nil tok-mock entities2 dim 1)
          row-alpha-1 (subvec (vec (:entity-table res1)) 0 dim)
          row-alpha-2 (subvec (vec (:entity-table res2)) dim (* 2 dim))]
      ;; Alpha is at index 0 in res1, index 1 in res2
      (is (= row-alpha-1 row-alpha-2)
          "Alpha vector must be bit-identical regardless of its position in the entity table"))))

(deftest test-anchored-table-determinism
  (testing "Repeated calls with identical inputs produce bit-identical tables"
    (let [dim 4
          vocab-mock {10 (float-array [0.5 0.5 0.5 0.5])}
          reader-fn (fn [tid] (get vocab-mock tid (float-array dim)))
          tok-mock (reify einsum.runtime.tokenizer.protocol/Tokenizer
                     (encode [_ _] [10])
                     (encode [this text _] (encode this text))
                     (decode [_ _] "")
                     (decode [_ _ _] "")
                     (bos-id [_] -1)
                     (eos-id [_] -2))
          entities ["A" "B"]
          t1 (anchored/build-llm-anchored-table reader-fn nil tok-mock entities dim 1)
          t2 (anchored/build-llm-anchored-table reader-fn nil tok-mock entities dim 1)]
      (is (java.util.Arrays/equals ^floats (:entity-table t1) ^floats (:entity-table t2))
          "Anchored tables must be bit-identical across runs"))))

;; ==============================================================================
;; 6. Orthonormalization (Modified Gram-Schmidt in f64 -> f32)
;; ==============================================================================

(def gen-matrix-dimensions
  (gen/bind (gen/choose 2 5)
            (fn [n]
              (gen/bind (gen/choose n 12)
                        (fn [d]
                          (gen/tuple (gen/return n)
                                     (gen/return d)
                                     (gen/vector (gen/double* {:min -5.0 :max 5.0 :NaN? false :infinite? false})
                                                 (* n d))))))))

(defspec prop-mgs-orthonormality 25
  (prop/for-all [[n d flat-doubles] gen-matrix-dimensions]
                (let [arr (float-array (map float flat-doubles))
                      q (anchored/orthonormalize-table arr n d)]
      ;; Every row must have unit Euclidean length
                  (and (every? (fn [i]
                                 (let [row-offset (* i d)
                                       sum-sq (reduce + (map (fn [k]
                                                               (let [val (double (aget ^floats q (+ row-offset k)))]
                                                                 (* val val)))
                                                             (range d)))]
                                   (< (Math/abs (- sum-sq 1.0)) 1e-4)))
                               (range n))
           ;; Every distinct pair must have dot product close to 0
                       (every? (fn [[i j]]
                                 (let [oi (* i d)
                                       oj (* j d)
                                       dot (reduce + (map (fn [k]
                                                            (* (double (aget ^floats q (+ oi k)))
                                                               (double (aget ^floats q (+ oj k)))))
                                                          (range d)))]
                                   (< (Math/abs dot) 1e-4)))
                               (for [i (range n) j (range n) :when (not= i j)] [i j]))))))

(deftest test-mgs-reconstruction-in-span
  (testing "Every row vector in the original matrix can be reconstructed from the orthonormal basis Q"
    (let [n 3
          d 5
          raw (float-array [1.0 2.0 0.0 -1.0 0.5
                            0.0 1.0 3.0 2.0 -0.5
                            2.0 -1.0 1.0 0.0 1.0])
          q (anchored/orthonormalize-table raw n d)]
      (dotimes [i n]
        (let [row-offset (* i d)
              x-rec (float-array d)]
          (dotimes [j n]
            (let [qj-offset (* j d)
                  dot (loop [k 0 s 0.0]
                        (if (>= k d)
                          s
                          (recur (inc k) (+ s (* (double (aget ^floats raw (+ row-offset k)))
                                                 (double (aget ^floats q (+ qj-offset k))))))))]
              (dotimes [k d]
                (aset-float x-rec k (+ (aget x-rec k) (float (* dot (aget ^floats q (+ qj-offset k)))))))))
          (dotimes [k d]
            (let [orig (double (aget ^floats raw (+ row-offset k)))
                  rec (double (aget x-rec k))]
              (is (< (Math/abs (- orig rec)) 1e-4)
                  (format "Row %d, col %d reconstruction error: orig=%.4f, rec=%.4f" i k orig rec)))))))))

(deftest test-orthonormalize-table-determinism
  (testing "Repeated calls to orthonormalize-table on identical inputs produce bit-identical results"
    (let [n 3
          d 4
          raw (float-array [1.0 2.0 3.0 4.0
                            0.0 1.0 0.0 1.0
                            -1.0 0.5 2.0 0.0])
          q1 (anchored/orthonormalize-table raw n d)
          q2 (anchored/orthonormalize-table raw n d)]
      (is (java.util.Arrays/equals ^floats q1 ^floats q2)
          "Orthonormalized tables must be bit-identical across runs"))))

(deftest test-orthonormalize-table-degenerate-input
  (testing "Orthonormalization handles duplicate, parallel, and zero rows without producing NaNs"
    (let [n 5
          d 6
          raw (float-array [1.0 2.0 0.0 1.0 0.0 0.0
                            1.0 2.0 0.0 1.0 0.0 0.0
                            2.0 4.0 0.0 2.0 0.0 0.0
                            0.0 0.0 0.0 0.0 0.0 0.0
                            0.0 0.0 1.0 0.0 0.0 1.0])
          q (anchored/orthonormalize-table raw n d)]
      ;; 1. No NaNs or Infinities
      (dotimes [idx (* n d)]
        (let [v (aget ^floats q idx)]
          (is (not (Float/isNaN v)) (str "Value at index " idx " must not be NaN"))
          (is (not (Float/isInfinite v)) (str "Value at index " idx " must not be Infinite"))))

      ;; 2. Strict orthonormality: Q Q^T = I
      (dotimes [i n]
        (let [oi (* i d)
              norm-sq (loop [k 0 s 0.0]
                        (if (>= k d) s (let [v (double (aget ^floats q (+ oi k)))]
                                         (recur (inc k) (+ s (* v v))))))]
          (is (< (Math/abs (- norm-sq 1.0)) 1e-4) (str "Row " i " must have unit length"))
          (dotimes [j i]
            (let [oj (* j d)
                  dot (loop [k 0 s 0.0]
                        (if (>= k d) s (recur (inc k) (+ s (* (double (aget ^floats q (+ oi k)))
                                                              (double (aget ^floats q (+ oj k))))))))]
              (is (< (Math/abs dot) 1e-4) (str "Rows " i " and " j " must be orthogonal")))))))))

(deftest test-qr-memory-pipeline-synthetic
  (testing "Pipeline with init-qr-anchored-memory and accumulate-fact! on synthetic case"
    (let [n 3
          d 4
          k 1
          raw-table (float-array [1.0 2.0 0.0 1.0
                                  0.0 1.0 3.0 0.0
                                  2.0 0.0 1.0 4.0])
          raw-mem {:entity-table raw-table
                   :entity-shape [n d]
                   :cores (float-array (* k d d))
                   :core-shape [k d d]
                   :entity-count n
                   :dim d
                   :relation-count k}
          qr-mem (anchored/init-qr-anchored-memory raw-mem)
          ^floats q-table (:entity-table qr-mem)]
      (is (= [n d] (:entity-shape qr-mem)))
      (is (= [k d d] (:core-shape qr-mem)))
      (is (= (* n d) (alength q-table)))
      ;; Accumulate fact: head=0, rel=0, tail=1
      (mem/accumulate-fact! (:cores qr-mem) q-table 0 0 1 d)
      ;; Hand-computed outer product: cores[i * d + j] = q[0, i] * q[1, j]
      (dotimes [i d]
        (dotimes [j d]
          (let [expected (* (double (aget q-table i))
                            (double (aget q-table (+ d j))))
                actual (double (aget ^floats (:cores qr-mem) (+ (* i d) j)))]
            (is (< (Math/abs (- actual expected)) 1e-6)
                (format "Core[%d, %d] mismatch: actual=%.6f, expected=%.6f" i j actual expected))))))))

(deftest test-calibrated-threshold-metrics-synthetic
  (testing "compute-score-scale-stats calculates accurate mean, std, calibrated threshold, and pass rates"
    (let [results [{:top-1-score 2.0}
                   {:top-1-score 4.0}
                   {:top-1-score 6.0}]
          stats (anchored/compute-score-scale-stats results 0.5)]
      (is (== 4.0 (double (:mean stats))))
      (is (< (Math/abs (- (double (:std stats)) 1.632993)) 1e-4))
      (is (== 0.5 (double (:fixed-threshold stats))))
      (is (== 2.0 (double (:calibrated-threshold stats))))
      (is (= 3 (:fixed-passes stats)))
      (is (== 1.0 (double (:fixed-pass-rate stats))))
      (is (= 2 (:calibrated-passes stats)))
      (is (< (Math/abs (- (double (:calibrated-pass-rate stats)) (/ 2.0 3.0))) 1e-6)))

    (testing "compute-score-scale-stats handles compressed scores below fixed threshold"
      (let [results [{:top-1-score 0.1}
                     {:top-1-score 0.3}
                     {:top-1-score 0.2}]
            stats (anchored/compute-score-scale-stats results 0.5)]
        (is (< (Math/abs (- (double (:mean stats)) 0.2)) 1e-6))
        (is (< (Math/abs (- (double (:calibrated-threshold stats)) 0.1)) 1e-6))
        (is (= 0 (:fixed-passes stats)))
        (is (== 0.0 (double (:fixed-pass-rate stats))))
        (is (= 2 (:calibrated-passes stats)))
        (is (< (Math/abs (- (double (:calibrated-pass-rate stats)) (/ 2.0 3.0))) 1e-6))))))

