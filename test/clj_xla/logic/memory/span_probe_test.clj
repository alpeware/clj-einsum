(ns clj-xla.logic.memory.span-probe-test
  "Unit and generative tests for span-pooled probe ablation helpers in scripts.poc-span-probe:
   subsequence search, head span location, and pooling operations (mean/max)."
  (:require [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :as tok-proto :refer [bos-id decode encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.poc-span-probe :as span]))

;; ==============================================================================
;; 1. Subsequence Search Tests
;; ==============================================================================

(deftest test-find-subsequence-basic
  (testing "find-subsequence finds needle at start, middle, and end of haystack"
    (is (= 0 (span/find-subsequence [1 2] [1 2 3 4 5])))
    (is (= 2 (span/find-subsequence [3 4] [1 2 3 4 5])))
    (is (= 3 (span/find-subsequence [4 5] [1 2 3 4 5]))))

  (testing "find-subsequence returns nil when absent or invalid"
    (is (nil? (span/find-subsequence [9 9] [1 2 3 4 5])))
    (is (nil? (span/find-subsequence [1 2 3 4 5 6] [1 2 3 4 5])))
    (is (nil? (span/find-subsequence [2 1] [1 2 3 4 5]))))

  (testing "find-subsequence handles empty needle"
    (is (= 0 (span/find-subsequence [] [1 2 3])))))

(defspec prop-find-subsequence-arbitrary 50
  (prop/for-all [prefix (gen/vector gen/nat 0 10)
                 needle (gen/vector gen/nat 1 5)
                 suffix (gen/vector gen/nat 0 10)]
                (let [haystack (vec (concat prefix needle suffix))
                      found-idx (span/find-subsequence needle haystack)]
                  (and (some? found-idx)
                       (<= found-idx (count prefix))
                       (= needle (subvec haystack found-idx (+ found-idx (count needle))))))))

;; ==============================================================================
;; 2. Pooling Helper Tests (Pure Functions on Synthetic Matrices)
;; ==============================================================================

(deftest test-pooling-helpers-synthetic
  (testing "Synthetic [P=4, N=3] matrix pooling"
    ;; P=4 positions, N=3 entities (12 floats total)
    ;; Pos 0: [1.0, 2.0, 3.0]
    ;; Pos 1: [4.0, 0.0, 1.0]
    ;; Pos 2: [2.0, 6.0, 5.0]
    ;; Pos 3: [0.0, 1.0, 0.0]
    (let [scores (float-array [1.0 2.0 3.0
                               4.0 0.0 1.0
                               2.0 6.0 5.0
                               0.0 1.0 0.0])
          n 3
          ;; Last position = Pos 3 (prompt-len = 4)
          last-scores (span/pool-scores-last-pos scores 4 n)
          ;; Span = Pos 1 to Pos 3 (indices 1, 2; span-len = 2)
          mean-scores (span/pool-scores-span-mean scores 1 2 n)
          max-scores (span/pool-scores-span-max scores 1 2 n)]

      ;; Last pos: [0.0, 1.0, 0.0] -> top entity is entity 1
      (is (= [0.0 1.0 0.0] (mapv #(double (aget ^floats last-scores %)) (range n))))
      (is (= 1 (span/top-entity-id last-scores n)))

      ;; Mean pool over pos 1, 2:
      ;; Entity 0: (4.0 + 2.0) / 2 = 3.0
      ;; Entity 1: (0.0 + 6.0) / 2 = 3.0
      ;; Entity 2: (1.0 + 5.0) / 2 = 3.0
      (is (= [3.0 3.0 3.0] (mapv #(double (aget ^floats mean-scores %)) (range n))))

      ;; Max pool over pos 1, 2:
      ;; Entity 0: max(4.0, 2.0) = 4.0
      ;; Entity 1: max(0.0, 6.0) = 6.0
      ;; Entity 2: max(1.0, 5.0) = 5.0
      (is (= [4.0 6.0 5.0] (mapv #(double (aget ^floats max-scores %)) (range n))))
      (is (= 1 (span/top-entity-id max-scores n))))))

;; ==============================================================================
;; 3. Head Span Location & Tokenizer Round-Trip Tests (All 7 Triples)
;; ==============================================================================

(deftest test-locate-head-span-all-triples-roundtrip
  (testing "For each of the 7 heads in wiki_recent_triples.edn, the located span decodes back to the head name"
    (let [model-dir ".models/gemma-4-E2B-it"
          triples-file "data/wiki_recent_triples.edn"]
      (when (.exists (io/file model-dir))
        (let [tokenizer (tok/from-file model-dir)
              kb-data (edn/read-string (slurp triples-file))
              triples (:triples kb-data)]
          (is (= 7 (count triples)))
          (doseq [[head _rel _tail] triples]
            (let [head-clean (str/trim head)
                  query-prompt (str "Who is the CEO of " head-clean "?")
                  raw-ids (encode tokenizer query-prompt)
                  clean-ids (if (= (first raw-ids) (bos-id tokenizer))
                              (vec (rest raw-ids))
                              (vec raw-ids))
                  prefix [(bos-id tokenizer) 105 2364 107]
                  suffix [106 107 105 4368 107]
                  prompt-ids (vec (concat prefix clean-ids suffix))
                  [start span-len] (span/locate-head-span tokenizer prompt-ids head-clean)
                  span-tokens (subvec prompt-ids start (+ start span-len))
                  decoded-span (str/trim (decode tokenizer span-tokens))]
              (is (some? start) (str "Head span start must be found for: " head))
              (is (pos? span-len) (str "Head span len must be > 0 for: " head))
              (is (= head-clean decoded-span)
                  (str "Decoded head span '" decoded-span "' must match original head '" head-clean "'")))))))))
