(ns experiments.gate3-evals.perplexity-probe-v1-test
  "Unit and generative property tests for Perplexity Probe v1 (Gate 3 Evals)."
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.runtime.tokenizer.protocol :as tok-proto]
            [experiments.gate3-evals.perplexity-probe-v1.core :as probe-core]))

;; -----------------------------------------------------------------------------
;; 1. Structured Turn Delineation & Evaluation Masking Invariants
;; -----------------------------------------------------------------------------

(deftest test-build-evaluation-segments-structure
  (let [mock-task {:id "sample-task" :fn-name "sample-fn"
                   :prompt "Write sample-fn." :public-tests []}
        mock-transcript [{:turn 1 :role :model :thought "Thinking about sample-fn."
                          :response "```clojure\n(defn sample-fn [x] x)\n```"}
                         {:turn 1 :role :tool :content "```clojure_result\npass\n```"}
                         {:turn 2 :role :model :thought nil
                          :response "Done!"}]
        segments (probe-core/build-evaluation-segments mock-task mock-transcript)]
    ;; Segment framing checks
    (is (= "<bos>" (first (first segments))))
    (is (= 0 (nth (first segments) 3)))
    ;; Model turn 1 content
    (let [think-seg (first (filter #(= (second %) :think) segments))
          code-segs (filter #(= (second %) :code) segments)
          tool-segs (filter #(= (second %) :tool) segments)]
      (is (some? think-seg))
      (is (= 1 (nth think-seg 3)))
      (is (= "Thinking about sample-fn." (first think-seg)))
      (is (= 2 (count code-segs)))
      (is (every? #(= 1 (nth % 3)) code-segs))
      (is (every? #(= 0 (nth % 3)) tool-segs)))))

(deftest test-mock-tokenizer-mask-invariants
  ;; Verify with a simple character/word mock tokenizer
  (let [mock-tokenizer (reify
                         tok-proto/Tokenizer
                         (encode [_ text]
                           (mapv int (str text)))
                         (decode [_ _] "")
                         (bos-id [_] 2)
                         (eos-id [_] 1))
        mock-task {:id "sample-task" :fn-name "sample-fn"
                   :prompt "Write sample-fn." :public-tests []}
        mock-transcript [{:turn 1 :role :model :thought "Think" :response "Code"}
                         {:turn 1 :role :tool :content "Result"}]
        mask-info (probe-core/build-evaluation-mask mock-task mock-transcript mock-tokenizer {})]
    (is (= 0 (first (:mask mask-info))))
    (is (= (:num-total mask-info) (count (:token-ids mask-info))))
    (is (= (:num-total mask-info) (count (:mask mask-info))))
    (is (= (:num-scored mask-info) (+ (:num-think mask-info) (:num-code mask-info))))
    (is (= 5 (:num-think mask-info))) ;; "Think" = 5 chars
    (is (= 4 (:num-code mask-info))))) ;; "Code" = 4 chars

;; -----------------------------------------------------------------------------
;; 2. Generative Properties for Perplexity Calculation & Micro-Aggregation
;; -----------------------------------------------------------------------------

(defspec prop-sequence-perplexity-soundness 50
  (prop/for-all [lps (gen/vector (gen/double* {:min -50.0 :max -0.01 :NaN? false :infinite? false}) 2 50)]
                (let [token-pairs (vec (map-indexed (fn [i _]
                                                      {:id i :pos i
                                                       :mask (if (zero? i) 0 (if (even? i) 1 0))
                                                       :segment (if (zero? (mod i 4)) :think :code)
                                                       :role :model})
                                                    lps))
                      metrics (probe-core/compute-sequence-metrics lps token-pairs)]
                  (and (>= (:sum-nll metrics) 0.0)
                       (>= (:cross-entropy metrics) 0.0)
                       (>= (:perplexity metrics) 1.0)
                       (or (zero? (:num-think metrics)) (>= (:think-perplexity metrics) 1.0))
                       (or (zero? (:num-code metrics)) (>= (:code-perplexity metrics) 1.0))))))

(defspec prop-corpus-micro-average-invariants 50
  (prop/for-all [gap-nlls (gen/vector (gen/double* {:min 10.0 :max 1000.0 :NaN? false :infinite? false}) 4)
                 gap-toks (gen/vector (gen/choose 100 2000) 4)
                 ref-nlls (gen/vector (gen/double* {:min 10.0 :max 1000.0 :NaN? false :infinite? false}) 2)
                 ref-toks (gen/vector (gen/choose 100 2000) 2)]
                (let [gap-rows (mapv (fn [i tid]
                                       {:task-id tid :group :gap
                                        :metrics {:num-scored (nth gap-toks i)
                                                  :sum-nll (nth gap-nlls i)
                                                  :perplexity (Math/exp (/ (nth gap-nlls i) (nth gap-toks i)))}})
                                     (range 4) probe-core/GAP-TASKS)
                      ref-rows (mapv (fn [i tid]
                                       {:task-id tid :group :ref
                                        :metrics {:num-scored (nth ref-toks i)
                                                  :sum-nll (nth ref-nlls i)
                                                  :perplexity (Math/exp (/ (nth ref-nlls i) (nth ref-toks i)))}})
                                     (range 2) probe-core/REF-TASKS)
                      all-rows (into gap-rows ref-rows)
                      agg (probe-core/compute-corpus-aggregates all-rows)

                      expected-gap-tokens (reduce + 0 gap-toks)
                      expected-gap-nll (reduce + 0.0 gap-nlls)
                      expected-gap-ppl (Math/exp (/ expected-gap-nll expected-gap-tokens))

                      expected-ref-tokens (reduce + 0 ref-toks)
                      expected-ref-nll (reduce + 0.0 ref-nlls)
                      expected-ref-ppl (Math/exp (/ expected-ref-nll expected-ref-tokens))

                      expected-ratio (/ expected-gap-ppl expected-ref-ppl)
                      expected-verdict (if (>= expected-ratio probe-core/GO-RATIO-THRESHOLD) :GO :NO-GO)]
                  (and (< (Math/abs (- (:gap-micro-ppl agg) expected-gap-ppl)) 1e-4)
                       (< (Math/abs (- (:ref-micro-ppl agg) expected-ref-ppl)) 1e-4)
                       (< (Math/abs (- (:ratio agg) expected-ratio)) 1e-4)
                       (= (:verdict agg) expected-verdict)
                       (contains? #{:GO :NO-GO} (:verdict agg))))))

(defspec prop-sft-task-partitioning-soundness 50
  (prop/for-all [ratios (gen/vector (gen/double* {:min 0.5 :max 3.0 :NaN? false :infinite? false}) 4)]
                (let [task-rows (mapv (fn [i tid]
                                        {:task-id tid :group :gap
                                         :metrics {:num-scored 100 :sum-nll (* 100.0 (Math/log (nth ratios i)))
                                                   :perplexity (nth ratios i)}})
                                      (range 4) probe-core/GAP-TASKS)
                      ref-row {:task-id "ref-1" :group :ref
                               :metrics {:num-scored 100 :sum-nll 0.0 :perplexity 1.0}}
                      agg (probe-core/compute-corpus-aggregates (conj task-rows ref-row))
                      sft-tasks (set (:sft-candidates agg))
                      flagged (set (:flagged-tasks agg))]
                  (and (= (into sft-tasks flagged) probe-core/GAP-TASKS-SET)
                       (empty? (set/intersection sft-tasks flagged))))))
