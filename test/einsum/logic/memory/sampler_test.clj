(ns einsum.logic.memory.sampler-test
  "Generative property tests for Experiment E18 hygienic sampler, multi-positive Target,
   and false-negative masking invariants (Repo Rule 1: Strict TDD)."
  (:require [einsum.logic.memory.sampler :as sampler]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; Generators
;; ==============================================================================

(def gen-triple
  (gen/fmap (fn [[h-id r-id t-id]]
              {:h-str (str "E_" h-id)
               :h-tok (int h-id)
               :rel (keyword (str "rel_" r-id))
               :t-str (str "E_" t-id)
               :t-tok (int t-id)})
            (gen/tuple (gen/choose 1 50)
                       (gen/choose 1 10)
                       (gen/choose 1 50))))

(def gen-triples-list
  (gen/such-that seq (gen/vector gen-triple 2 40)))

;; ==============================================================================
;; Invariant 1: Sampling Distinct When n >= K
;; ==============================================================================

(defspec prop-hygienic-sampling-distinct-when-n-ge-k 50
  (prop/for-all [triples (gen/such-that #(>= (count %) 16)
                                        (gen/fmap (fn [ts] (vec (distinct ts)))
                                                  (gen/vector gen-triple 16 50)))
                 seed gen/nat]
                (let [k 16
                      rnd (java.util.Random. (long seed))
                      sampled (sampler/sample-relation-batch triples k rnd)]
                  (and (= k (count sampled))
                       (= k (count (distinct sampled)))))))

;; ==============================================================================
;; Invariant 2: Target Rows Sum to 1.0
;; ==============================================================================

(defspec prop-multi-positive-target-rows-sum-to-one 50
  (prop/for-all [batch (gen/vector gen-triple 16 16)]
                (let [k 16
                      ^floats target (sampler/build-multi-positive-target batch k)]
                  (every? (fn [i]
                            (let [row-sum (loop [j 0 s 0.0]
                                            (if (>= j k)
                                              s
                                              (recur (inc j) (+ s (double (aget target (+ (* i k) j)))))))]
                              (< (Math/abs (- row-sum 1.0)) 1e-5)))
                          (range k)))))

;; ==============================================================================
;; Invariant 3: Target Marks All Matching Tails Uniformly
;; ==============================================================================

(defspec prop-multi-positive-target-marks-all-matching-tails 50
  (prop/for-all [batch (gen/vector gen-triple 16 16)]
                (let [k 16
                      ^floats target (sampler/build-multi-positive-target batch k)]
                  (every? (fn [i]
                            (let [target-tok (:t-tok (nth batch i))
                                  matching-indices (filterv #(= (:t-tok (nth batch %)) target-tok) (range k))
                                  m (count matching-indices)
                                  expected-val (/ 1.0 (double m))]
                              (every? (fn [j]
                                        (let [actual-val (double (aget target (+ (* i k) j)))]
                                          (if (some #(= % j) matching-indices)
                                            (< (Math/abs (- actual-val expected-val)) 1e-5)
                                            (< actual-val 1e-6))))
                                      (range k))))
                          (range k)))))

;; ==============================================================================
;; Invariant 4: False-Negative Masking Zeroes Known True Facts
;; ==============================================================================

(defspec prop-false-negative-masking-zeroes-known-facts 50
  (prop/for-all [batch (gen/vector gen-triple 16 16)
                 extra-facts (gen/vector (gen/tuple (gen/choose 1 50) (gen/choose 1 10) (gen/choose 1 50)) 0 20)
                 tau (gen/fmap #(+ 0.05 (* 0.01 (double %))) (gen/choose 1 100))]
                (let [k 16
                      batch-facts (map (fn [tr] [(:h-tok tr) (:rel tr) (:t-tok tr)]) batch)
                      extra-fact-maps (map (fn [[h r t]] [h (keyword (str "rel_" r)) t]) extra-facts)
                      known-set (set (concat batch-facts extra-fact-maps))
                      ^floats mask (sampler/build-hygienic-mask-scale batch known-set k (double tau))
                      expected-scale (float (/ 1.0 (* (double k) (double tau))))]
                  (every? (fn [i]
                            (let [h-i (:h-tok (nth batch i))
                                  r-i (:rel (nth batch i))
                                  t-i (:t-tok (nth batch i))]
                              (every? (fn [j]
                                        (let [t-j (:t-tok (nth batch j))
                                              val (aget mask (+ (* i k) j))]
                                          (cond
                                ;; Positive positions (matching tail) keep scale
                                            (= t-j t-i)
                                            (< (Math/abs (- val expected-scale)) 1e-5)

                                ;; Off-diagonal with known true fact must be zeroed
                                            (contains? known-set [h-i r-i t-j])
                                            (zero? val)

                                ;; Otherwise unmasked negative keeps scale
                                            :else
                                            (< (Math/abs (- val expected-scale)) 1e-5))))
                                      (range k))))
                          (range k)))))

;; ==============================================================================
;; Unit Tests for Edge Cases
;; ==============================================================================

(deftest test-sampler-edge-cases
  (testing "Relations with n < K pad by cycling without crashing"
    (let [triples [{:h-tok 1 :rel :rel_a :t-tok 10}
                   {:h-tok 2 :rel :rel_a :t-tok 20}
                   {:h-tok 3 :rel :rel_a :t-tok 30}]
          k 16
          rnd (java.util.Random. 42)
          sampled (sampler/sample-relation-batch triples k rnd)]
      (is (= 16 (count sampled)))
      ;; First 3 are distinct
      (is (= 3 (count (distinct (take 3 sampled)))))))

  (testing "Multi-positive target with duplicate entries in batch distributes weight equally"
    (let [batch (vec (repeat 16 {:h-tok 1 :rel :rel_a :t-tok 10}))
          k 16
          ^floats target (sampler/build-multi-positive-target batch k)]
      (dotimes [i k]
        (dotimes [j k]
          (is (< (Math/abs (- (double (aget target (+ (* i k) j))) (/ 1.0 16.0))) 1e-5)))))))
