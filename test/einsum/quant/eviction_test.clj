(ns einsum.quant.eviction-test
  "Generative property and invariant tests for Tier 2 Attention Sinks and Pyramidal Heavy-Hitter Eviction
   (StreamingLLM / SnapKV / PyramidKV, Criterion 2.3 and 3.2)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.quant.eviction :as evict]))

;; ==============================================================================
;; 1. Attention Sink Invariant (Tokens 0..3 permanently pinned)
;; ==============================================================================

(defspec prop-sink-indices-invariant
  50
  (prop/for-all [total-len (gen/choose 2000 8000)
                 layer-idx (gen/choose 0 53)]
                (let [sink-size 4
                      window-size 1024
                      k-base 512
                      num-layers 54
          ;; Random attention mass for intermediate tokens
                      attn-mass (vec (repeatedly total-len #(rand)))
                      retained (evict/select-retained-indices layer-idx num-layers total-len attn-mass
                                                              {:k-sink sink-size
                                                               :window window-size
                                                               :k-base k-base})
                      retained-set (set retained)]
                  (and (contains? retained-set 0)
                       (contains? retained-set 1)
                       (contains? retained-set 2)
                       (contains? retained-set 3)))))

;; ==============================================================================
;; 2. Local Sliding Window Invariant (Last W tokens retained verbatim)
;; ==============================================================================

(defspec prop-local-window-invariant
  50
  (prop/for-all [total-len (gen/choose 2000 8000)
                 layer-idx (gen/choose 0 53)]
                (let [window-size 1024
                      attn-mass (vec (repeatedly total-len #(rand)))
                      retained (evict/select-retained-indices layer-idx 54 total-len attn-mass
                                                              {:k-sink 4
                                                               :window window-size
                                                               :k-base 512})
                      retained-set (set retained)
                      expected-window (range (- total-len window-size) total-len)]
                  (every? #(contains? retained-set %) expected-window))))

;; ==============================================================================
;; 3. Pyramidal Budget Monotonicity Invariant
;; ==============================================================================

(deftest test-pyramidal-schedule-formula
  (testing "Pyramidal schedule K_heavy(l) = K_base * (2 - l/L) decreases monotonically with depth"
    (let [k-base 512
          num-layers 54
          budgets (mapv #(evict/pyramidal-heavy-budget % num-layers k-base) (range num-layers))]
      ;; Lower layer budget = 2 * K_base = 1024
      (is (= 1024 (first budgets)))
      ;; Deep layer budget approaches K_base = 512
      (is (<= (last budgets) 530))
      (is (>= (last budgets) 512))
      ;; Monotonically non-increasing
      (is (every? (fn [[b1 b2]] (>= b1 b2)) (partition 2 1 budgets))))))

;; ==============================================================================
;; 4. Causal Monotonic Ordering Invariant
;; ==============================================================================

(defspec prop-monotonic-causal-ordering
  50
  (prop/for-all [total-len (gen/choose 2000 6000)
                 layer-idx (gen/choose 0 41)]
                (let [attn-mass (vec (repeatedly total-len #(rand)))
                      retained (evict/select-retained-indices layer-idx 42 total-len attn-mass
                                                              {:k-sink 4
                                                               :window 1024
                                                               :k-base 512})]
                  (and (apply < retained)
                       (= (count retained) (count (distinct retained)))))))

;; ==============================================================================
;; 5. Capacity Bound Invariant (LDS Workgroup Safety, Criterion 2.3)
;; ==============================================================================

(defspec prop-capacity-bound-invariant
  50
  (prop/for-all [total-len (gen/choose 2000 16000)
                 layer-idx (gen/choose 0 53)]
                (let [k-sink 4
                      window 1024
                      k-base 512
                      num-layers 54
                      attn-mass (vec (repeatedly total-len #(rand)))
                      retained (evict/select-retained-indices layer-idx num-layers total-len attn-mass
                                                              {:k-sink k-sink
                                                               :window window
                                                               :k-base k-base})
                      max-allowed (+ k-sink (* 2 k-base) window)]
                  (<= (count retained) max-allowed))))

;; ==============================================================================
;; 6. Synthetic Multi-Needle-in-a-Haystack (M-NIAH) Retrieval Retention (Criterion 3.2)
;; ==============================================================================

(deftest test-mniah-synthetic-needle-retention
  (testing "High-attention needle positions across 10 depth bins are retained with >= 95% accuracy"
    (let [context-lengths [16384 32768 65536 131072]
          k-sink 4
          window 1024
          k-base 512
          num-layers 54
          depth-fractions [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]]
      (doseq [t context-lengths]
        (let [needle-positions (mapv #(long (min (dec t) (max k-sink (Math/round (* (double %) (double t))))))
                                     depth-fractions)
              ;; Synthesize background noise + spike at needle positions
              attn-mass (double-array t)
              _ (dotimes [i t] (aset-double attn-mass i (* 0.01 (Math/random))))
              _ (doseq [pos needle-positions]
                  (aset-double attn-mass pos (+ 10.0 (Math/random))))
              ;; Middle layer
              retained (set (evict/select-retained-indices 27 num-layers t (vec attn-mass)
                                                           {:k-sink k-sink
                                                            :window window
                                                            :k-base k-base}))
              retained-needles (count (filter #(contains? retained %) needle-positions))
              retention-rate (/ (double retained-needles) (double (count needle-positions)))]
          ;; Retain at least 95% (e.g. 10/10 or 9.5/10)
          (is (>= retention-rate 0.95)
              (str "Needle retention rate at length " t " was " retention-rate " (< 0.95)")))))))
