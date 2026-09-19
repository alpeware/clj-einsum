(ns einsum.logic.memory.ephemeral-test
  "Unit, generative, and invariant tests for Zero-Gradient Ephemeral Online Learning
   in Pedro Domingos' Declarative Tensor Logic (Experiment E5).
   Tests outer-product fast-weight memory writes, batch superposition, fact retraction,
   in-graph relational composition, and extract-threshold-re-embed denoising cycles."
  (:require [einsum.logic.memory.ephemeral :as eph]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; Helper: Pure Orthonormal Basis Table Generator
;; ==============================================================================

(defn- canonical-orthonormal-table
  "Creates an exact orthonormal basis table [N, D] using canonical unit vectors."
  [n d]
  (let [n-long (long n)
        d-long (long d)
        arr (float-array (* n-long d-long))]
    (dotimes [i n-long]
      (aset arr (+ (* i d-long) i) (float 1.0)))
    arr))

(defn- extract-vector
  "Extracts a 1-D vector of length d at row index `row` from flat table [N, D]."
  [^floats table row d]
  (let [d-long (long d)
        row-long (long row)
        v (float-array d-long)]
    (System/arraycopy table (int (* row-long d-long)) v 0 (int d-long))
    v))

;; ==============================================================================
;; 1. AST Construction and Schema Tests
;; ==============================================================================

(deftest test-ephemeral-ast-construction
  (testing "write-fact-ast generates valid Hiccup block"
    (let [ast (eph/write-fact-ast 64 {:alpha 0.95 :beta 1.0})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_write_fact (get-in ast [1 :name])))))

  (testing "write-facts-batch-ast generates valid Hiccup block"
    (let [ast (eph/write-facts-batch-ast 4 64 {:alpha 1.0 :beta 1.0})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_write_facts_batch (get-in ast [1 :name])))))

  (testing "retract-fact-ast generates valid Hiccup block"
    (let [ast (eph/retract-fact-ast 64)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_retract_fact (get-in ast [1 :name])))))

  (testing "query-memory-ast generates valid Hiccup block"
    (let [ast (eph/query-memory-ast 64 16 {:temperature 0.5})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_query_memory (get-in ast [1 :name])))))

  (testing "compose-relations-ast generates valid Hiccup block"
    (let [ast (eph/compose-relations-ast 64)]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_compose_relations (get-in ast [1 :name])))))

  (testing "denoise-relation-ast generates valid Hiccup block"
    (let [ast (eph/denoise-relation-ast 64 16 {:threshold 0.5})]
      (is (vector? ast))
      (is (= :block (first ast)))
      (is (= :ephemeral_denoise_relation (get-in ast [1 :name]))))))

;; ==============================================================================
;; 2. Generative Invariant: Single-Fact Exact Recall & Zero Cross-Talk
;; ==============================================================================

(defspec prop-single-fact-zero-crosstalk-recall 20
  (prop/for-all [d (gen/choose 16 64)
                 n (gen/choose 4 12)
                 h-idx (gen/choose 0 3)
                 t-idx (gen/choose 0 3)]
                (let [table (canonical-orthonormal-table n d)
                      eh (extract-vector table h-idx d)
                      et (extract-vector table t-idx d)
                      r-zero (float-array (* d d))
          ;; Compile write and query executables
                      write-exec (eph/compile-write-fact d {:alpha 0.0 :beta 1.0})
                      query-exec (eph/compile-query-memory d n {:temperature 1.0})
          ;; Write fact into memory: R = e_h (x) e_t
                      write-out (eph/write-fact! write-exec r-zero eh et)
                      r-updated (:R_updated write-out)
          ;; Query with head entity e_h
                      query-out (eph/query-memory! query-exec eh r-updated table)
                      ^floats scores (:scores query-out)
                      retrieved-tail-score (double (aget scores (int t-idx)))]
                  (and
       ;; 1. Retrieved score for expected tail entity is 1.0
                   (< (Math/abs (- retrieved-tail-score 1.0)) 1e-4)
       ;; 2. All other entities have score 0.0 (zero cross-talk)
                   (every? (fn [j]
                             (if (= j t-idx)
                               true
                               (< (Math/abs (double (aget scores (int j)))) 1e-4)))
                           (range n))))))

;; ==============================================================================
;; 3. Generative Invariant: Fact Retraction Exactness
;; ==============================================================================

(defspec prop-fact-retraction-exactness 15
  (prop/for-all [d (gen/choose 16 32)
                 n (gen/choose 4 8)
                 h-idx (gen/choose 0 3)
                 t-idx (gen/choose 0 3)]
                (let [table (canonical-orthonormal-table n d)
                      eh (extract-vector table h-idx d)
                      et (extract-vector table t-idx d)
                      r-zero (float-array (* d d))
                      write-exec (eph/compile-write-fact d {:alpha 0.0 :beta 1.0})
                      retract-exec (eph/compile-retract-fact d)
                      query-exec (eph/compile-query-memory d n {:temperature 1.0})
          ;; 1. Write fact
                      r-written (:R_updated (eph/write-fact! write-exec r-zero eh et))
          ;; 2. Retract fact: R_cleared = R_written - e_h (x) e_t
                      r-cleared (:R_updated (eph/retract-fact! retract-exec r-written eh et))
          ;; 3. Query cleared memory
                      query-out (eph/query-memory! query-exec eh r-cleared table)
                      ^floats scores (:scores query-out)]
                  (and
       ;; Relation matrix elements must all be 0.0
                   (every? (fn [k] (< (Math/abs (double (aget ^floats r-cleared (int k)))) 1e-4))
                           (range (* d d)))
       ;; Query scores must all be 0.0
                   (every? (fn [j] (< (Math/abs (double (aget scores (int j)))) 1e-4))
                           (range n))))))

;; ==============================================================================
;; 4. Generative Invariant: Batch Superposition & Capacity SNR
;; ==============================================================================

(defspec prop-batch-superposition-capacity 15
  (prop/for-all [k (gen/choose 2 4)]
                (let [d 64
                      n 8
                      table (canonical-orthonormal-table n d)
          ;; Generate K distinct facts: [h_i -> t_i] with distinct heads
                      heads (vec (range k))
                      tails (vec (reverse heads))
          ;; Build batch arrays [K, D]
                      h-batch (float-array (* k d))
                      t-batch (float-array (* k d))]
                  (dotimes [i k]
                    (let [eh (extract-vector table (nth heads i) d)
                          et (extract-vector table (nth tails i) d)]
                      (System/arraycopy eh 0 h-batch (* i d) d)
                      (System/arraycopy et 0 t-batch (* i d) d)))
                  (let [r-zero (float-array (* d d))
                        write-batch-exec (eph/compile-write-facts-batch k d {:alpha 0.0 :beta 1.0})
                        query-exec (eph/compile-query-memory d n {:temperature 1.0})
            ;; Superpose all K facts in parallel
                        r-superposed (:R_updated (eph/write-facts-batch! write-batch-exec r-zero h-batch t-batch))]
        ;; Query each head entity individually and verify exact recall
                    (every? (fn [i]
                              (let [h-idx (nth heads i)
                                    t-idx (nth tails i)
                                    eh (extract-vector table h-idx d)
                                    query-out (eph/query-memory! query-exec eh r-superposed table)
                                    ^floats scores (:scores query-out)
                                    tail-score (double (aget scores (int t-idx)))]
                                (and (< (Math/abs (- tail-score 1.0)) 1e-4)
                                     (every? (fn [j]
                                               (if (= j t-idx)
                                                 true
                                                 (< (Math/abs (double (aget scores (int j)))) 1e-4)))
                                             (range n)))))
                            (range k))))))

;; ==============================================================================
;; 5. Generative Invariant: In-Graph Transitive Relational Composition
;; ==============================================================================

(defspec prop-compositional-transitive-reasoning 15
  (prop/for-all [d (gen/choose 16 32)
                 n (gen/choose 4 8)]
                (let [table (canonical-orthonormal-table n d)
          ;; Select 3 chained entities: A -> B -> C
                      a-idx 0
                      b-idx 1
                      c-idx 2
                      ea (extract-vector table a-idx d)
                      eb (extract-vector table b-idx d)
                      ec (extract-vector table c-idx d)
                      r-zero (float-array (* d d))
                      write-exec (eph/compile-write-fact d {:alpha 0.0 :beta 1.0})
                      compose-exec (eph/compile-compose-relations d)
                      query-exec (eph/compile-query-memory d n {:temperature 1.0})
          ;; Write R1: A -> B
                      r1 (:R_updated (eph/write-fact! write-exec r-zero ea eb))
          ;; Write R2: B -> C
                      r2 (:R_updated (eph/write-fact! write-exec r-zero eb ec))
          ;; Compose in-graph: R_12 = R1 * R2
                      r12 (:R_composed (eph/compose-relations! compose-exec r1 r2))
          ;; Query 2-hop unbinding: ea * R_12
                      query-out (eph/query-memory! query-exec ea r12 table)
                      ^floats scores (:scores query-out)
                      target-score (double (aget scores (int c-idx)))]
                  (and
       ;; 1. Transitive target entity C receives exact score 1.0
                   (< (Math/abs (- target-score 1.0)) 1e-4)
       ;; 2. Intermediate entity B receives score 0.0
                   (< (Math/abs (double (aget scores (int b-idx)))) 1e-4)
       ;; 3. Source entity A receives score 0.0
                   (< (Math/abs (double (aget scores (int a-idx)))) 1e-4)
       ;; 4. All other entities receive score 0.0
                   (every? (fn [j]
                             (if (= j c-idx)
                               true
                               (< (Math/abs (double (aget scores (int j)))) 1e-4)))
                           (range n))))))

;; ==============================================================================
;; 6. Generative Invariant: Extract-Threshold-Re-Embed Denoising
;; ==============================================================================

(defspec prop-denoising-restores-superposition 15
  (prop/for-all [d (gen/choose 16 32)
                 n (gen/choose 4 8)]
                (let [table (canonical-orthonormal-table n d)
          ;; Fact: 0 -> 1
                      e0 (extract-vector table 0 d)
                      e1 (extract-vector table 1 d)
                      r-zero (float-array (* d d))
                      write-exec (eph/compile-write-fact d {:alpha 0.0 :beta 1.0})
                      denoise-exec (eph/compile-denoise-relation d n {:threshold 0.5})
                      query-exec (eph/compile-query-memory d n {:temperature 1.0})
          ;; Clean fact write
                      r-clean (:R_updated (eph/write-fact! write-exec r-zero e0 e1))
          ;; Add noise (magnitude 0.25 < threshold 0.5)
                      r-noisy (float-array (* d d))]
                  (dotimes [i (* d d)]
                    (let [clean-val (aget ^floats r-clean i)
                          noise (* (float 0.2) (float (Math/sin (double i))))]
                      (aset r-noisy i (+ clean-val noise))))
      ;; Run PJRT compiled extract-threshold-re-embed denoising cycle
                  (let [denoise-out (eph/denoise-relation! denoise-exec r-noisy table)
                        ^floats r-restored (:R_clean denoise-out)
                        query-out (eph/query-memory! query-exec e0 r-restored table)
                        ^floats scores (:scores query-out)]
                    (and
         ;; 1. Restored matrix matches clean matrix closely
                     (every? (fn [k]
                               (< (Math/abs (- (double (aget r-restored (int k)))
                                               (double (aget ^floats r-clean (int k)))))
                                  1e-3))
                             (range (* d d)))
         ;; 2. Target fact has score 1.0
                     (< (Math/abs (- (double (aget scores 1)) 1.0)) 1e-3)
         ;; 3. Distractor entity 0 has score 0.0
                     (< (Math/abs (double (aget scores 0))) 1e-3))))))
