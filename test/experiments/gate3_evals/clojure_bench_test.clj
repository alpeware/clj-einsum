(ns experiments.gate3-evals.clojure-bench-test
  "Generative invariant tests and verification for clojure_bench (Gate 3 Evals, Stage 2).
   Tests pure prompt synthesis, submission detection, hermetic SCI grading sandbox,
   public test feedback formatting, best-public submission selection, and metric aggregations."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]))

;; =============================================================================
;; 1. Verified Reference Solutions for all 10 Tasks
;; =============================================================================

(def REFERENCE-SOLUTIONS
  {"first-n"
   "(defn first-n [coll] (vec (take 10 coll)))"

   "my-range"
   "(defn my-range
      ([end] (my-range 0 end))
      ([start end]
       (loop [i start acc []]
         (if (>= i end)
           acc
           (recur (inc i) (conj acc i))))))"

   "deep-flatten"
   "(defn deep-flatten [coll]
      (reduce (fn [acc x]
                (if (sequential? x)
                  (into acc (deep-flatten x))
                  (conj acc x)))
              []
              coll))"

   "freqs"
   "(defn freqs [coll]
      (reduce (fn [m x] (update m x (fnil inc 0))) {} coll))"

   "partition-by-parity"
   "(defn partition-by-parity [coll]
      (reduce (fn [m x]
                (if (even? x)
                  (update m :even conj x)
                  (update m :odd conj x)))
              {:even [] :odd []}
              coll))"

   "my-comp"
   "(defn my-comp [& fns]
      (if (empty? fns)
        identity
        (let [rfns (reverse fns)
              r (first rfns)
              rest-fns (rest rfns)]
          (fn [& args]
            (reduce (fn [acc f] (f acc))
                    (apply r args)
                    rest-fns)))))"

   "balanced-delims?"
   "(defn balanced-delims? [s]
      (let [matches {\\) \\( \\] \\[ \\} \\{}
            opens #{\\( \\[ \\{}]
        (loop [chars (seq s)
               stack ()]
          (if-let [c (first chars)]
            (cond
              (opens c) (recur (rest chars) (conj stack c))
              (matches c) (if (= (first stack) (matches c))
                            (recur (rest chars) (pop stack))
                            false)
              :else (recur (rest chars) stack))
            (empty? stack)))))"

   "deep-update-vals"
   "(defn deep-update-vals [f x]
      (cond
        (map? x) (reduce-kv (fn [m k v] (assoc m k (deep-update-vals f v))) {} x)
        (vector? x) (mapv #(deep-update-vals f %) x)
        :else (f x)))"

   "lazy-interleave"
   "(defn lazy-interleave [c1 c2]
      (lazy-seq
        (let [s1 (seq c1)
              s2 (seq c2)]
          (cond
            (and s1 s2) (cons (first s1) (cons (first s2) (lazy-interleave (rest s1) (rest s2))))
            s1 s1
            s2 s2
            :else nil))))"

   "my-or"
   "(defmacro my-or
      ([] nil)
      ([x] x)
      ([x & next]
       (list 'let ['or# x]
             (list 'if 'or# 'or# (cons 'my-or next)))))"})

;; =============================================================================
;; 2. Sealed Tests Integrity (Pre-registered SHA-256)
;; =============================================================================

(def EXPECTED-SEALED-SHA256
  "656f97b09127561b857f10eba46262750eb358279705d252cfea247d0300cda0")

(deftest test-sealed-tests-sha256-integrity
  (testing "Sealed tests file matches pre-registered SHA-256 checksum exactly"
    (let [sealed-file (io/file "resources/proposals/gate3_evals/clojure_bench/tasks_sealed.edn")
          computed-sha (bench-core/compute-file-sha256 sealed-file)]
      (is (= EXPECTED-SEALED-SHA256 computed-sha)
          "SHA-256 of tasks_sealed.edn must remain immutable to protect against benchmark contamination."))))

;; =============================================================================
;; 3. Reference Solutions Verification (10/10 Tasks, 68/68 Tests)
;; =============================================================================

(deftest test-reference-solutions-all-tasks
  (testing "All 10 tasks reference solutions pass 100% of public and sealed tests in SCI"
    (let [public-tasks (edn/read-string (slurp (io/file "resources/proposals/gate3_evals/clojure_bench/tasks_public.edn")))
          sealed-tasks (into {} (map (juxt :id :hidden-tests)
                                     (edn/read-string (slurp (io/file "resources/proposals/gate3_evals/clojure_bench/tasks_sealed.edn")))))]
      (is (= 10 (count public-tasks)))
      (is (= 10 (count sealed-tasks)))
      (doseq [t public-tasks]
        (let [task-id (:id t)
              code (get REFERENCE-SOLUTIONS task-id)
              _ (is (some? code) (str "Missing reference solution for " task-id))
              pub-res (bench-core/grade-submission code (:public-tests t))
              hidden-tests (get sealed-tasks task-id)
              hidden-res (bench-core/grade-submission code hidden-tests)]
          (is (:all-passed? pub-res)
              (str "Public tests failed for " task-id ": " (pr-str pub-res)))
          (is (:all-passed? hidden-res)
              (str "Hidden tests failed for " task-id ": " (pr-str hidden-res))))))))

;; =============================================================================
;; 4. Generative Properties
;; =============================================================================

(def gen-ident-name
  (gen/fmap (fn [[c chars]] (str c (apply str chars)))
            (gen/tuple (gen/elements [\a \b \c \d \e \f])
                       (gen/vector (gen/elements [\a \b \c \1 \2 \- \_]) 1 8))))

(defspec prop-submission-form-detection 50
  (prop/for-all [fn-name gen-ident-name
                 other-name gen-ident-name]
                (let [defn-code (format "(defn %s [x] (* x 2))" fn-name)
                      defmacro-code (format "(defmacro %s [x] `(inc ~x))" fn-name)
                      def-fn-code (format "(def %s (fn [x] (+ x 1)))" fn-name)
                      unrelated-code (format "(defn %s_alt [x] x)" other-name)
                      math-expr "(+ 1 2 3)"]
                  (and (true? (bench-core/submission-form? defn-code fn-name))
                       (true? (bench-core/submission-form? defmacro-code fn-name))
                       (true? (bench-core/submission-form? def-fn-code fn-name))
                       (false? (bench-core/submission-form? unrelated-code fn-name))
                       (false? (bench-core/submission-form? math-expr fn-name))))))

(defspec prop-render-benchmark-prompt-invariants 50
  (prop/for-all [id gen-ident-name
                 title (gen/not-empty gen/string-alphanumeric)
                 fn-name gen-ident-name
                 instructions (gen/not-empty gen/string-alphanumeric)]
                (let [test-case {:code (format "(%s 42)" fn-name) :expected "84"}
                      task {:id id
                            :title title
                            :fn-name (symbol fn-name)
                            :prompt instructions
                            :public-tests [test-case]}
                      rendered (bench-core/render-benchmark-prompt task)]
                  (and (str/includes? rendered title)
                       (str/includes? rendered fn-name)
                       (str/includes? rendered instructions)
                       (str/includes? rendered (:code test-case))
                       (str/includes? rendered (:expected test-case))
                       (not (str/includes? rendered "tasks_sealed"))))))

(deftest test-grading-fault-injection
  (testing "Syntax errors result in failure with error diagnostic"
    (let [tests [{:code "(broken-fn 1)" :expected "2"}]
          res (bench-core/grade-submission "(defn broken-fn [x" tests)]
      (is (false? (:all-passed? res)))
      (is (seq (:error res)))))

  (testing "Semantic error results in failure with expected vs actual diagnostic"
    (let [tests [{:code "(bad-add 2 3)" :expected "5"}]
          res (bench-core/grade-submission "(defn bad-add [a b] (* a b))" tests)]
      (is (false? (:all-passed? res)))
      (is (= 0 (:passed-count res)))
      (is (= 1 (:total-count res)))
      (let [t-res (first (:results res))]
        (is (false? (:passed? t-res)))
        (is (= "6" (str (:actual t-res)))))))

  (testing "Grading sandbox isolates file I/O and System calls"
    (let [res (bench-core/grade-submission "(slurp \"deps.edn\")" [{:code "1" :expected "1"}])]
      (is (false? (:all-passed? res)))
      (is (str/includes? (or (:error res) "") "Could not resolve symbol: slurp")))))

(deftest test-public-feedback-formatting
  (testing "Feedback indicates 100% pass when all public tests pass"
    (let [grade-res {:all-passed? true :passed-count 1 :total-count 1 :results [{:code "(first-n [1 2 3])" :passed? true}]}
          feedback (bench-core/format-public-feedback grade-res)]
      (is (str/includes? feedback "Public tests: 1/1 passed."))))

  (testing "Feedback lists specific failing tests and values when tests fail"
    (let [grade-res {:all-passed? false
                     :passed-count 0
                     :total-count 1
                     :results [{:code "(my-range 2 5)" :expected "[2 3 4]" :actual "[2 3]" :passed? false}]}
          feedback (bench-core/format-public-feedback grade-res)]
      (is (str/includes? feedback "Public tests: 0/1 passed"))
      (is (str/includes? feedback "Expected: [2 3 4]"))
      (is (str/includes? feedback "Actual: [2 3]")))))

(deftest test-pick-best-public-submission
  (testing "Picks submission with highest public test pass rate"
    (let [sub1 {:code "(defn my-range [x] [])" :public-res {:passed-count 0 :total-count 2}}
          sub2 {:code "(defn my-range [x] (vec (take x (iterate inc 0))))" :public-res {:passed-count 2 :total-count 2}}
          sub3 {:code "(defn my-range [x] [0])" :public-res {:passed-count 1 :total-count 2}}
          best (bench-core/pick-best-public-submission [sub1 sub2 sub3])]
      (is (= (:code sub2) (:code best)))))

  (testing "Falls back to last submission if pass counts are equal"
    (let [sub1 {:code "(defn f [x] 1)" :public-res {:passed-count 1 :total-count 2}}
          sub2 {:code "(defn f [x] 2)" :public-res {:passed-count 1 :total-count 2}}
          best (bench-core/pick-best-public-submission [sub1 sub2])]
      (is (= (:code sub2) (:code best))))))

(defspec prop-metric-aggregation-invariants 50
  (prop/for-all [passed-flags (gen/vector gen/boolean 10)]
                (let [rows (mapv (fn [idx pass?]
                                   {:model "gemma-4-E2B-it-int4"
                                    :task (str "task-" idx)
                                    :mode :agentic
                                    :passed? pass?
                                    :n-submissions (if pass? 1 3)
                                    :tokens-in 100
                                    :tokens-out 50
                                    :wall-ms 1500.0})
                                 (range (count passed-flags))
                                 passed-flags)
                      metrics (bench-core/calculate-metrics rows)
                      pass-count (count (filter true? passed-flags))
                      expected-mean (double (/ pass-count (count passed-flags)))]
                  (and (<= (Math/abs (- ^double (:mean-hidden-pass-rate metrics) expected-mean)) 1e-6)
                       (= (count passed-flags) (:total-evals metrics))
                       (= pass-count (:passed-evals metrics))
                       (string? (bench-core/format-summary-csv metrics))))))

(deftest test-extract-candidate-code
  (testing "Unwraps native Gemma 4 tool call code into bare s-expression"
    (let [text "<|tool_call>call:eval_clojure{code:<|\"|>(defn first-n [coll] (vec (take 10 coll)))<|\"|>}<tool_call|>"
          code (bench-core/extract-candidate-code text "first-n")]
      (is (= "(defn first-n [coll] (vec (take 10 coll)))" code))))

  (testing "Unwraps markdown clojure code block"
    (let [text "Here is my solution:\n```clojure\n(defn first-n [coll] (take 10 coll))\n```\nEnjoy!"
          code (bench-core/extract-candidate-code text "first-n")]
      (is (= "(defn first-n [coll] (take 10 coll))" code))))

  (testing "Extracts raw s-expression matching target fn"
    (let [text "Sure! (defn first-n [coll] (vec (take 10 coll))) is the function."
          code (bench-core/extract-candidate-code text "first-n")]
      (is (= "(defn first-n [coll] (vec (take 10 coll)))" code))))

  (testing "Returns nil when no valid submission or code block exists"
    (let [text "I don't know how to write Clojure."
          code (bench-core/extract-candidate-code text "first-n")]
      (is (nil? code)))))

(deftest test-format-results-row-diagnostics
  (testing "Formats diagnostic failure row with candidate-code, test-summary, and failure details"
    (let [grade-res {:all-passed? false
                     :passed-count 4
                     :total-count 5
                     :results [{:code "(vector? (first-n (range 100)))"
                                :expected "true"
                                :actual "false"
                                :passed? false}]
                     :error nil}
          row (bench-core/format-results-row
               {:model "gemma-4-E2B-it-int4"
                :task "first-n"
                :mode :single-shot
                :candidate-code "(defn first-n [coll] (take 10 coll))"
                :grade-res grade-res
                :tokens-in 266
                :tokens-out 512
                :wall-ms 8229.7
                :sealed-sha "dummy-sha"
                :checkpoint-sha "dummy-cp"
                :dry-run? false})]
      (is (= :test-failure (:stop-reason row)))
      (is (false? (:passed? row)))
      (is (= "(defn first-n [coll] (take 10 coll))" (:candidate-code row)))
      (is (= {:passed 4 :total 5} (:test-summary row)))
      (is (= 1 (count (:failures row))))
      (is (= "(vector? (first-n (range 100)))" (:code (first (:failures row)))))
      (is (= "false" (:actual (first (:failures row)))))))

  (testing "Formats no-extraction row with clear stop-reason and error"
    (let [row (bench-core/format-results-row
               {:model "gemma-4-E2B-it-int4"
                :task "first-n"
                :mode :single-shot
                :candidate-code nil
                :error "No Clojure code extracted from model generation"
                :tokens-in 266
                :tokens-out 512
                :wall-ms 8000.0
                :sealed-sha "dummy-sha"
                :checkpoint-sha "dummy-cp"
                :dry-run? false})]
      (is (= :no-extraction (:stop-reason row)))
      (is (false? (:passed? row)))
      (is (nil? (:candidate-code row)))
      (is (str/includes? (:error row) "No Clojure code extracted"))))

  (testing "Formats compilation error row"
    (let [grade-res {:all-passed? false
                     :passed-count 0
                     :total-count 5
                     :results []
                     :error "Compilation/Execution Exception: Unmatched delimiter )"}
          row (bench-core/format-results-row
               {:model "gemma-4-E2B-it-int4"
                :task "first-n"
                :mode :single-shot
                :candidate-code "(defn first-n [coll] (take 10 coll)))"
                :grade-res grade-res
                :tokens-in 266
                :tokens-out 512
                :wall-ms 8000.0
                :sealed-sha "dummy-sha"
                :checkpoint-sha "dummy-cp"
                :dry-run? false})]
      (is (= :compilation-error (:stop-reason row)))
      (is (str/includes? (:error row) "Unmatched delimiter"))))

  (testing "Formats passing row with passed-all"
    (let [grade-res {:all-passed? true
                     :passed-count 5
                     :total-count 5
                     :results [{:code "(first-n [1])" :expected "[1]" :actual "[1]" :passed? true}]
                     :error nil}
          row (bench-core/format-results-row
               {:model "gemma-4-E2B-it-int4"
                :task "first-n"
                :mode :single-shot
                :candidate-code "(defn first-n [coll] (vec (take 10 coll)))"
                :grade-res grade-res
                :tokens-in 266
                :tokens-out 512
                :wall-ms 8000.0
                :sealed-sha "dummy-sha"
                :checkpoint-sha "dummy-cp"
                :dry-run? false})]
      (is (= :passed-all (:stop-reason row)))
      (is (true? (:passed? row)))
      (is (nil? (:failures row)))
      (is (= {:passed 5 :total 5} (:test-summary row))))))

