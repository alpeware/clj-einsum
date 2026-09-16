(ns clj-xla.logic.memory.anchored-memory-test
  "Unit, generative, and invariant tests for LLM-anchored entity embeddings in scripts.poc-anchored-memory."
  (:require [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.poc-anchored-memory :as anchored]))

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
          triples-file "data/wiki_recent_triples.edn"]
      (when (.exists (io/file model-dir))
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
          tok-mock (reify clj-xla.tokenizer.protocol/Tokenizer
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
          tok-mock (reify clj-xla.tokenizer.protocol/Tokenizer
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
