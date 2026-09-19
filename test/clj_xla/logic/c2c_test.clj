(ns clj-xla.logic.c2c-test
  "Generative property and unit tests for Experiment E24: In-VRAM Multi-Instance Handover (C2C).
   Repository Rule 1: Strict TDD — generative tests for invariants written BEFORE implementation.
   Verifies:
   - In-VRAM handover offset soundness and delta position sequence integrity
   - Token splicing totality and zero index overlap
   - Handover state validity predicates
   - Answer extraction robustness and multi-variant evaluation
   - Multi-turn monotonicity across chained debate rounds (n instances, m rounds)
   - Chat template formatting for Cell H (C2C), Cell B1 (Host-mediated), and Cell B0 (Single-instance)"
  (:require [clj-xla.logic.c2c :as c2c]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generators for Handover Geometry
;; ==============================================================================

(def gen-token-id
  (gen/choose 1 262000))

(def gen-token-seq
  (gen/not-empty (gen/vector gen-token-id 1 50)))

(def gen-c2c-geometry
  (gen/let [p-len (gen/choose 10 200)
            draft-len (gen/choose 10 150)
            delta-len (gen/choose 4 30)
            extra-slack (gen/choose 50 200)]
    (let [p1 (+ p-len draft-len)
          p2 (+ p1 delta-len)
          max-seq-len (+ p2 extra-slack)
          max-new-tokens (gen/choose 20 extra-slack)]
      (gen/tuple (gen/return p1)
                 (gen/vector gen-token-id p1)
                 (gen/vector gen-token-id delta-len)
                 (gen/return max-seq-len)
                 max-new-tokens))))

;; ==============================================================================
;; 2. Generative Property Tests (Repository Rule 1)
;; ==============================================================================

(defspec prop-c2c-transition-offset-soundness 50
  (prop/for-all [[p1 prior-tokens trans-tokens max-seq-len max-new] gen-c2c-geometry]
                (let [offsets (c2c/compute-handover-offsets p1 prior-tokens trans-tokens max-seq-len max-new)
                      {:keys [p2 delta-len delta-positions target-max spliced-tokens]} offsets]
                  (and
       ;; 1. Delta length matches transition tokens count
                   (= delta-len (count trans-tokens))
       ;; 2. Start of Instance 2 is exactly p1 + delta-len
                   (= p2 (+ p1 delta-len))
       ;; 3. Delta positions are contiguous and strictly monotonically increasing
                   (= delta-positions (vec (range p1 p2)))
                   (= (count delta-positions) delta-len)
       ;; 4. Target max is strictly greater than p2 and <= max-seq-len
                   (> target-max p2)
                   (<= target-max max-seq-len)
       ;; 5. Spliced tokens array has length max-seq-len
                   (= (count spliced-tokens) max-seq-len)
       ;; 6. Prefix of spliced tokens matches prior tokens
                   (= (subvec spliced-tokens 0 p1) prior-tokens)
       ;; 7. Delta slice of spliced tokens matches transition tokens
                   (= (subvec spliced-tokens p1 p2) trans-tokens)
       ;; 8. Remaining tail is padded with zeros
                   (every? zero? (subvec spliced-tokens p2 max-seq-len))))))

(defspec prop-c2c-token-splicing-totality 50
  (prop/for-all [prefix (gen/vector gen-token-id 5 30)
                 trans (gen/vector gen-token-id 3 15)
                 gen-step (gen/choose 5 25)]
                (let [p1 (count prefix)
                      p2 (+ p1 (count trans))
                      p-final (+ p2 gen-step)
                      max-len (+ p-final 20)
                      offsets (c2c/compute-handover-offsets p1 prefix trans max-len 50)
                      spliced (:spliced-tokens offsets)
          ;; Simulate instance 2 generating tokens
                      instance2-tokens (vec (range 1000 (+ 1000 gen-step)))
                      full-tokens (vec (concat (subvec spliced 0 p2) instance2-tokens (repeat (- max-len p-final) 0)))
                      ext-draft (c2c/extract-instance-tokens full-tokens 0 p1)
                      ext-i2 (c2c/extract-instance-tokens full-tokens p2 p-final)]
                  (and
       ;; Draft extraction recovers exact prefix
                   (= ext-draft prefix)
       ;; Instance 2 extraction recovers exact generated tokens
                   (= ext-i2 instance2-tokens)
       ;; Disjointness of token segments
                   (= (count ext-draft) p1)
                   (= (count ext-i2) gen-step)
                   (= (+ (count ext-draft) (count trans) (count ext-i2)) p-final)))))

(defspec prop-c2c-state-soundness 50
  (prop/for-all [[p1 prior-tokens trans-tokens max-seq-len max-new] gen-c2c-geometry
                 corrupt? gen/boolean]
                (let [offsets (c2c/compute-handover-offsets p1 prior-tokens trans-tokens max-seq-len max-new)
                      test-state (if corrupt?
                                   (assoc offsets :p2 (dec (:p1 offsets))) ;; Corrupted: p2 < p1
                                   offsets)]
                  (if corrupt?
                    (not (c2c/valid-c2c-handover? test-state))
                    (c2c/valid-c2c-handover? test-state)))))

(defspec prop-multi-turn-monotonicity 50
  (prop/for-all [turns (gen/vector (gen/choose 5 25) 2 5)]
                (let [max-seq-len 1024
                      trans-tokens [106 107 105 2364 107 99 106 107 105 4368 107]
                      m (count turns)
          ;; Simulate m turns of in-VRAM handover
                      state-history
                      (loop [idx 0
                             cur-p (first turns)
                             cur-tokens (vec (concat (repeat cur-p 42) (repeat (- max-seq-len cur-p) 0)))
                             acc []]
                        (if (>= idx (dec m))
                          acc
                          (let [step-gen (nth turns (inc idx))
                                offsets (c2c/compute-handover-offsets cur-p (subvec cur-tokens 0 cur-p) trans-tokens max-seq-len step-gen)
                                next-p (+ (:p2 offsets) step-gen)
                                next-tokens (vec (concat (subvec (:spliced-tokens offsets) 0 (:p2 offsets))
                                                         (repeat step-gen 77)
                                                         (repeat (- max-seq-len next-p) 0)))]
                            (recur (inc idx) next-p next-tokens (conj acc offsets)))))]
                  (and
                   (= (count state-history) (dec m))
       ;; All turn boundaries are strictly monotonic
                   (every? (fn [[s1 s2]] (< (:p2 s1) (:p1 s2)))
                           (partition 2 1 state-history))
       ;; Every handover state is valid
                   (every? c2c/valid-c2c-handover? state-history)))))

(def gen-numeric-answer
  (gen/choose -100000 100000))

(defspec prop-c2c-answer-extraction-robustness 50
  (prop/for-all [ans gen-numeric-answer
                 wrapper-idx (gen/choose 0 5)]
                (let [ans-str (str ans)
                      wrapped-text
                      (case wrapper-idx
                        0 (str "Therefore, the result is computed.\nFinal Answer: " ans-str)
                        1 (str "Calculating step by step...\nFinal Answer: " ans-str "\nHope this helps!")
                        2 (str "We find that the value is \\boxed{" ans-str "}.")
                        3 (str "After careful checking, the answer is " ans-str ".")
                        4 (str "Step 1: calculate. Step 2: verify.\nFinal Answer:  " ans-str "  \n<turn|>")
                        5 (str "The final answer is: " ans-str))
                      extracted (c2c/extract-answer wrapped-text)]
                  (c2c/evaluate-answer extracted ans-str))))

;; ==============================================================================
;; 3. Unit Tests: Boundary Conditions & Formats
;; ==============================================================================

(deftest test-transition-tokens-deduplication
  (testing "Ensures transition tokens do not duplicate turn delimiters"
    (let [t-with-eot (c2c/make-transition-tokens {:prior-ended-with-eot? true})
          t-without-eot (c2c/make-transition-tokens {:prior-ended-with-eot? false})]
      ;; When prior ended with EOT (106), transition starts directly with newline or turn open (107 / 105)
      (is (not= (first t-with-eot) 106))
      ;; When prior did NOT end with EOT, transition begins with 106 to close prior turn
      (is (= (first t-without-eot) 106))
      ;; Both end with model turn prefix: <|turn>model\n -> [105 4368 107]
      (is (= (take-last 3 t-with-eot) [105 4368 107]))
      (is (= (take-last 3 t-without-eot) [105 4368 107])))))

(deftest test-prompt-formatting-parity
  (testing "Validates formatting for Cell H, Cell B1, and Cell B0"
    (let [problem "What is 15 * 14?"
          h-p (c2c/format-proposer-prompt problem)
          b0-p (c2c/format-baseline-b0-prompt problem)
          draft "15 * 14 = 210"
          b1-p (c2c/format-baseline-b1-prompt problem draft)]
      (is (str/includes? h-p "What is 15 * 14?"))
      (is (str/includes? h-p "<|turn>user"))
      (is (str/includes? h-p "<|turn>model"))
      (is (str/includes? b0-p "What is 15 * 14?"))
      (is (str/includes? b0-p "Final Answer:"))
      (is (str/includes? b1-p "15 * 14 = 210"))
      (is (str/includes? b1-p "Final Answer:")))))

(deftest test-answer-evaluation-variants
  (testing "Evaluates semantic answer matches across different surface forms"
    (is (c2c/evaluate-answer "210" "210"))
    (is (c2c/evaluate-answer "210.0" "210"))
    (is (c2c/evaluate-answer "$210" "210"))
    (is (c2c/evaluate-answer "30 loaves" "30"))
    (is (c2c/evaluate-answer "76.36 km/h" "76.36"))
    (is (c2c/evaluate-answer "$57.50" "57.50"))
    (is (c2c/evaluate-answer "Alice, Bob" "Bob, Alice"))
    (is (c2c/evaluate-answer "yes" "Yes"))
    (is (not (c2c/evaluate-answer "210" "215")))))

(deftest test-conversational-answer-extraction
  (testing "Extracts answers from multi-line reasoning traces"
    (is (= (c2c/extract-answer "Number of rye loaves left = 0.25 * 120 = 1/4 * 120 = 30 loaves.") "30 loaves"))
    (is (= (c2c/extract-answer "Round to two decimal places: 76.36 km/h.") "76.36"))
    (is (= (c2c/extract-answer "Cost: C = (3 * $18) + (2 * $1.75) = $54 + $3.50 = $57.50.") "57.50"))
    (is (c2c/evaluate-answer
         (c2c/extract-answer "Number of rye loaves left = 0.25 * 120 = 1/4 * 120 = 30 loaves.")
         "30"))
    (is (c2c/evaluate-answer
         (c2c/extract-answer "Round to two decimal places: 76.36 km/h.")
         "76.36"))
    (is (c2c/evaluate-answer
         (c2c/extract-answer "Cost: C = (3 * $18) + (2 * $1.75) = $54 + $3.50 = $57.50.")
         "57.50"))))
