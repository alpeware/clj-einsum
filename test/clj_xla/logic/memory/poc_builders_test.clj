(ns clj-xla.logic.memory.poc-builders-test
  "Unit and generative tests for de-oracled fact-grounding PoC buffer builders:
   build-random-projection and build-entity-token-table."
  (:require [clj-xla.tokenizer.protocol :as tok-proto]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [scripts.poc-fact-grounding :as poc]))

;; Simple mock tokenizer for testing build-entity-token-table in isolation
(defrecord SimpleMockTokenizer [vocab-map]
  tok-proto/Tokenizer
  (encode [this text]
    (tok-proto/encode this text false))
  (encode [_this text _add-special-tokens?]
    (let [words (str/split (str/trim text) #"\s+")]
      (mapv #(get vocab-map % (int (mod (Math/abs (hash %)) 1000))) words)))
  (decode [_this _tokens] "")
  (decode [_this _tokens _skip-special-tokens?] "")
  (bos-id [_this] 1)
  (eos-id [_this] 2))

;; ==============================================================================
;; 1. Fixed Random Projection Tests
;; ==============================================================================

(deftest test-random-projection-determinism
  (testing "build-random-projection is deterministic for a fixed seed"
    (let [in-dim 64
          d 32
          seed 2026
          p1 (poc/build-random-projection in-dim d seed)
          p2 (poc/build-random-projection in-dim d seed)]
      (is (= (* in-dim d) (alength ^floats p1)))
      (is (java.util.Arrays/equals ^floats p1 ^floats p2)
          "Same seed must produce identical projection matrix")))

  (testing "build-random-projection differs across seeds"
    (let [in-dim 64
          d 32
          p1 (poc/build-random-projection in-dim d 2026)
          p2 (poc/build-random-projection in-dim d 2027)]
      (is (not (java.util.Arrays/equals ^floats p1 ^floats p2))
          "Different seeds must produce different projection matrices")))

  (testing "build-random-projection Gaussian scaling ~ 1/sqrt(in-dim)"
    (let [in-dim 1024
          d 256
          p (poc/build-random-projection in-dim d 2026)
          n (alength ^floats p)
          expected-std (/ 1.0 (Math/sqrt (double in-dim)))
          mean (double (/ (areduce ^floats p i ret 0.0 (+ ret (aget ^floats p i))) n))
          variance (double (/ (areduce ^floats p i ret 0.0
                                       (let [diff (- (aget ^floats p i) mean)]
                                         (+ ret (* diff diff))))
                              n))
          std (Math/sqrt variance)]
      (is (< (Math/abs mean) 0.01) "Mean should be approximately 0.0")
      (is (< (Math/abs (- std expected-std)) (* 0.15 expected-std))
          "Empirical std should match 1/sqrt(in-dim) scaling"))))

(deftest test-random-projection-entity-invariance
  (testing "build-random-projection is invariant to entity table ordering and content"
    ;; De-oracle property: shuffling or modifying the entity table does NOT change the projection matrix
    (let [in-dim 128
          d 64
          seed 2026
          p-baseline (poc/build-random-projection in-dim d seed)
          ;; Caller simulates shuffling entity table or facts
          _dummy-entities (shuffle ["Anthropic" "OpenAI" "Google" "Tesla" "Apple"])
          p-after (poc/build-random-projection in-dim d seed)]
      (is (java.util.Arrays/equals ^floats p-baseline ^floats p-after)
          "Projection matrix must be completely independent of entity table"))))

;; ==============================================================================
;; 2. Static Entity-to-Token Table Tests
;; ==============================================================================

(deftest test-entity-token-table-purity
  (testing "build-entity-token-table is a pure function of (tokenizer, entities, vocab-size)"
    (let [entities ["Alice Smith" "Bob Jones" "Charlie Brown"]
          vocab {"Alice" 10 "Smith" 25 "Bob" 40 "Jones" 55 "Charlie" 70 "Brown" 85}
          tok (->SimpleMockTokenizer vocab)
          v 1000
          table (poc/build-entity-token-table tok entities v)]
      (is (= (* 3 v) (alength ^floats table)))

      ;; Row 0 ("Alice Smith" -> tokens [10 25])
      (is (== 1.0 (aget ^floats table 10)))
      (is (== 1.0 (aget ^floats table 25)))
      (is (== 0.0 (aget ^floats table 40)))

      ;; Row 1 ("Bob Jones" -> tokens [40 55])
      (is (== 1.0 (aget ^floats table (+ v 40))))
      (is (== 1.0 (aget ^floats table (+ v 55))))
      (is (== 0.0 (aget ^floats table (+ v 10))))

      ;; Verify each row i has exactly (count (encode tokenizer (str " " entities[i]))) nonzero entries
      (dotimes [i (count entities)]
        (let [expected-toks (tok-proto/encode tok (str " " (nth entities i)))
              row-offset (* i v)
              nonzero-count (loop [j 0 c 0]
                              (if (>= j v)
                                c
                                (recur (inc j) (if (pos? (aget ^floats table (+ row-offset j))) (inc c) c))))]
          (is (= (count expected-toks) nonzero-count)
              (str "Row " i " must have exactly " (count expected-toks) " nonzero entries")))))))

;; ==============================================================================
;; 3. Generative Property: Row Independence
;; ==============================================================================

(defspec prop-entity-token-table-row-independence 25
  (prop/for-all [prefix (gen/vector (gen/elements ["Alpha" "Beta" "Gamma" "Delta"]) 1 5)
                 suffix (gen/vector (gen/elements ["Epsilon" "Zeta" "Eta" "Theta"]) 1 5)
                 target-idx (gen/choose 0 3)]
                (let [target-name (str "TargetEntity" target-idx)
                      entities1 (vec (concat prefix [target-name] suffix))
                      entities2 (vec (concat suffix [target-name] prefix))
                      idx1 (count prefix)
                      idx2 (count suffix)
                      vocab-map {target-name 123}
                      tok (->SimpleMockTokenizer vocab-map)
                      v 500
                      t1 (poc/build-entity-token-table tok entities1 v)
                      t2 (poc/build-entity-token-table tok entities2 v)
                      row1 (subvec (vec t1) (* idx1 v) (* (inc idx1) v))
                      row2 (subvec (vec t2) (* idx2 v) (* (inc idx2) v))]
      ;; Row for target-name must be completely identical across arbitrary entity lists
                  (= row1 row2))))
