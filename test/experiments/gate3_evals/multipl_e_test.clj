(ns experiments.gate3-evals.multipl-e-test
  "Unit and generative tests for MultiPL-E Clojure suite port (Gate 3 Evals).
   Validates schema invariants, exact 111 held-out partition, test-split rules,
   SCI sandbox verification floor (>= 95%), quarantine cataloging, and dry-run execution."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [experiments.gate3-evals.clojure-bench.run :as bench-run]
            [experiments.gate3-evals.multipl-e.ingest :as ingest]))

;; =============================================================================
;; 1. Catalog Partition & Schema Invariants (AC1, AC3, AC6)
;; =============================================================================

(deftest test-catalog-partition-and-schemas
  (testing "MultiPL-E catalog files exist and maintain invariant counts"
    (let [dev-file (io/file "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn")
          sealed-file (io/file "resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn")
          sols-file (io/file "resources/catalog/gate3_evals/multipl_e/solutions.edn")
          quarantine-file (io/file "resources/catalog/gate3_evals/multipl_e/quarantine.edn")
          dev50-pub (io/file "resources/catalog/gate3_evals/multipl_e/dev_50_public.edn")
          dev50-sealed (io/file "resources/catalog/gate3_evals/multipl_e/dev_50_sealed.edn")]
      (is (.exists dev-file) "tasks_dev.edn must exist")
      (is (.exists sealed-file) "tasks_sealed.edn must exist")
      (is (.exists sols-file) "solutions.edn must exist")
      (is (.exists quarantine-file) "quarantine.edn must exist")
      (is (.exists dev50-pub) "dev_50_public.edn backwards compatibility must be preserved")
      (is (.exists dev50-sealed) "dev_50_sealed.edn backwards compatibility must be preserved")

      (let [dev-tasks (edn/read-string (slurp dev-file))
            sealed-tasks (edn/read-string (slurp sealed-file))
            sols (edn/read-string (slurp sols-file))
            quarantine (edn/read-string (slurp quarantine-file))]

        ;; Task Counts
        (is (= 447 (count dev-tasks)) "Exactly 447 open dev tasks")
        (is (= 111 (count sealed-tasks)) "Exactly 111 held-out eval tasks")
        (is (= 558 (+ (count dev-tasks) (count sealed-tasks))) "Total 558 tasks in MultiPL-E suite")
        (is (map? sols) "Solutions file must contain solution map")

        ;; Disjointness
        (let [dev-ids (set (map :id dev-tasks))
              sealed-ids (set (map :id sealed-tasks))
              overlap (set/intersection dev-ids sealed-ids)]
          (is (empty? overlap) "Dev and sealed sets must be strictly disjoint"))

        ;; Verbatim 111 Held-Out IDs Alignment (AC3)
        (let [sealed-ids (set (map :id sealed-tasks))]
          (is (= ingest/CANONICAL-HELDOUT-IDS sealed-ids)
              "Held-out IDs must match verbatim the canonical 111 IDs from clojure-llm"))

        ;; Stratification across sources
        (let [he-sealed (filter #(= (:source %) :multipl-e/humaneval-clj) sealed-tasks)
              mb-sealed (filter #(= (:source %) :multipl-e/mbpp-clj) sealed-tasks)]
          (is (= 32 (count he-sealed)) "Exactly 32 HumanEval tasks in held-out eval set")
          (is (= 79 (count mb-sealed)) "Exactly 79 MBPP tasks in held-out eval set"))

        ;; Schema Compliance for all tasks
        (doseq [t (concat dev-tasks sealed-tasks)]
          (is (string? (:id t)) (str "Task ID must be string: " (:id t)))
          (is (string? (:title t)) (str "Task title must be string: " (:id t)))
          (is (contains? #{:multipl-e/humaneval-clj :multipl-e/mbpp-clj} (:source t))
              (str "Task source must be known MultiPL-E keyword: " (:id t)))
          (is (symbol? (:fn-name t)) (str "Task fn-name must be symbol: " (:id t)))
          (is (string? (:prompt t)) (str "Task prompt must be string: " (:id t)))
          (is (vector? (:public-tests t)) (str "Task public-tests must be vector: " (:id t)))
          (is (vector? (:hidden-tests t)) (str "Task hidden-tests must be vector: " (:id t))))

        ;; Quarantine Schema Compliance
        (is (pos? (count quarantine)) "Quarantine file must document quarantined tasks")
        (is (<= (count quarantine) 28) "Quarantine must not exceed 5% of dataset (max 28 tasks)")
        (doseq [q quarantine]
          (is (string? (:id q)))
          (is (keyword? (:source q)))
          (is (string? (:reason q)))
          (is (pos? (count (:reason q)))))))))

;; =============================================================================
;; 2. Generative Test Split Rule Invariants (AC4)
;; =============================================================================

(def test-case-gen
  (gen/fmap (fn [[sym arg expected]]
              {:code (format "(%s %d)" sym arg)
               :expected (str expected)})
            (gen/tuple (gen/elements ["foo" "bar" "baz" "calc"])
                       gen/nat
                       gen/nat)))

(defspec prop-test-split-rule-invariants 50
  (prop/for-all [test-cases (gen/vector test-case-gen 1 10)]
                (let [k (count test-cases)
                      {:keys [public-tests hidden-tests]} (ingest/split-tests test-cases)]
                  (cond
                    (>= k 3)
                    (and (= 2 (count public-tests))
                         (= (- k 2) (count hidden-tests))
                         (= test-cases (into (vec public-tests) hidden-tests)))

                    (= k 2)
                    (and (= 1 (count public-tests))
                         (= 1 (count hidden-tests))
                         (= (first test-cases) (first public-tests))
                         (= (second test-cases) (first hidden-tests)))

                    (= k 1)
                    (and (= 1 (count public-tests))
                         (= 1 (count hidden-tests))
                         (= (first test-cases) (first public-tests))
                         (= (first test-cases) (first hidden-tests)))

                    :else false))))

;; =============================================================================
;; 3. SCI Verification Floor Invariant (AC2)
;; =============================================================================

(deftest test-sci-verification-floor
  (testing "Reference solutions meet or exceed >= 95% verification floor in pure SCI sandbox"
    (let [dev-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"))
          sealed-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn"))
          all-tasks (concat dev-tasks sealed-tasks)
          sols (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/solutions.edn"))
          passed-count (atom 0)]
      (doseq [t all-tasks
              :let [code (get sols (:id t))
                    all-tests (concat (:public-tests t) (:hidden-tests t))]]
        (when (and (seq code) (seq all-tests))
          (let [res (bench-core/grade-submission code all-tests)]
            (when (:all-passed? res)
              (swap! passed-count inc)))))
      (let [n 558
            pass-rate (double (/ @passed-count n))]
        (is (>= pass-rate 0.95)
            (format "SCI verification floor violated: %d/558 passed (%.2f%% < 95.0%%)"
                    @passed-count (* 100.0 pass-rate)))
        (is (>= @passed-count 530)
            "At least 530 tasks must pass pure SCI verification")))))

;; =============================================================================
;; 4. Prompt Template Rendering (AC5)
;; =============================================================================

(deftest test-prompt-template-rendering
  (testing ":multipl-e-v0 prompt template renders correctly in single-shot and agentic modes"
    (let [task {:id "humaneval-clj-000"
                :title "Has close elements"
                :fn-name 'has_close_elements
                :prompt "(defn has_close_elements [numbers threshold]\n  \"Check close elements\")"
                :public-tests [{:code "(has_close_elements [1.0 2.0] 0.5)" :expected "false"}]}
          ss-prompt (bench-core/render-benchmark-prompt task :multipl-e-v0 :single-shot)
          ag-prompt (bench-core/render-benchmark-prompt task :multipl-e-v0 :agentic)]
      (is (= (:prompt task) ss-prompt)
          "Single-shot :multipl-e-v0 prompt should match verbatim task prompt")
      (is (str/includes? ag-prompt (:prompt task))
          "Agentic :multipl-e-v0 prompt must include docstring/signature")
      (is (str/includes? ag-prompt "Public Examples & Unit Tests:")
          "Agentic :multipl-e-v0 prompt must include formatted public test examples")
      (is (str/includes? ag-prompt "(has_close_elements [1.0 2.0] 0.5)")
          "Agentic prompt must include public test code"))))

;; =============================================================================
;; 5. Test Harness Dry-Run Execution (AC7)
;; =============================================================================

(deftest test-dry-run-harness-execution
  (testing "clojure_bench runner executes --dry-run on multipl-e-sealed with valid ledger output"
    (let [tmp-results (str "scratch/test_multipl_e_sealed_results_" (System/currentTimeMillis) ".edn")
          tmp-summary (str "scratch/test_multipl_e_sealed_summary_" (System/currentTimeMillis) ".csv")
          _ (io/make-parents (io/file tmp-results))
          _ (io/make-parents (io/file tmp-summary))
          opts (bench-run/parse-bench-cli-args
                ["--tasks" "multipl-e-sealed"
                 "--dry-run" "true"
                 "--mode" "single-shot"
                 "--quiet" "true"
                 "--overwrite" "true"
                 "--results-file" tmp-results
                 "--summary-file" tmp-summary])
          _ (bench-run/run-benchmark opts)
          rows (with-open [r (io/reader tmp-results)]
                 (mapv edn/read-string (line-seq r)))]
      (is (= 111 (count rows)) "Dry-run on multipl-e-sealed must yield exactly 111 rows")
      (is (every? #(and (string? (:harness-sha %)) (pos? (count (:harness-sha %)))) rows))
      (is (every? #(and (string? (:sealed-sha %)) (pos? (count (:sealed-sha %)))) rows))
      (is (every? #(and (string? (:prompt-sha %)) (pos? (count (:prompt-sha %)))) rows))
      (is (every? #(number? (:temperature %)) rows))
      (is (every? #(boolean? (:dry-run? %)) rows))
      ;; Cleanup temporary files
      (io/delete-file tmp-results true)
      (io/delete-file tmp-summary true))))
