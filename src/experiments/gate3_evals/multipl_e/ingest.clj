(ns experiments.gate3-evals.multipl-e.ingest
  "Ingestion and normalization pipeline for MultiPL-E Clojure suite (Gate 3 Evals).
   Ingests 558 tasks (161 humaneval-clj + 397 mbpp-clj) from pinned MultiPL-E / clojure-llm.
   Applies deterministic public/hidden test split (AC4), exact 111 held-out partition (AC3),
   and in-process SCI sandbox verification with quarantine escape hatch (AC2)."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.pprint :as pprint]
            [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [experiments.gate3-evals.clojure-bench.core :as bench-core])
  (:import [java.io PushbackReader StringReader]))

;; =============================================================================
;; 1. Upstream Provenance Metadata
;; =============================================================================

(def UPSTREAM-PROVENANCE
  {:huggingface {:url "https://huggingface.co/datasets/nuprl/MultiPL-E"
                 :commit "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"}
   :github {:url "https://github.com/nuprl/MultiPL-E"
            :commit "3025a531af7450e7df8b96fe0440e9804480bbad"}
   :cross-check-reference {:repo "https://github.com/nibzard/clojure-llm"
                           :commit "8ed80ac59ad26413ed8f9e7f167860b384478e41"
                           :manifest "benchmark/tasks-v0.edn"
                           :held-out-split "benchmark/runs/2026-04-20-rlvr-qwen3-30b-heldout.edn"}})

;; =============================================================================
;; 2. Canonical 111 Held-Out Evaluation Task IDs (AC3)
;; =============================================================================

(def CANONICAL-HELDOUT-IDS
  #{"humaneval-clj-001" "humaneval-clj-003" "humaneval-clj-012" "humaneval-clj-018"
    "humaneval-clj-019" "humaneval-clj-020" "humaneval-clj-031" "humaneval-clj-039"
    "humaneval-clj-043" "humaneval-clj-053" "humaneval-clj-062" "humaneval-clj-067"
    "humaneval-clj-068" "humaneval-clj-073" "humaneval-clj-077" "humaneval-clj-086"
    "humaneval-clj-090" "humaneval-clj-095" "humaneval-clj-097" "humaneval-clj-105"
    "humaneval-clj-121" "humaneval-clj-123" "humaneval-clj-124" "humaneval-clj-132"
    "humaneval-clj-134" "humaneval-clj-137" "humaneval-clj-141" "humaneval-clj-143"
    "humaneval-clj-144" "humaneval-clj-153" "humaneval-clj-157" "humaneval-clj-158"
    "mbpp-clj-006" "mbpp-clj-007" "mbpp-clj-009" "mbpp-clj-022" "mbpp-clj-047"
    "mbpp-clj-057" "mbpp-clj-060" "mbpp-clj-069" "mbpp-clj-074" "mbpp-clj-075"
    "mbpp-clj-076" "mbpp-clj-082" "mbpp-clj-084" "mbpp-clj-087" "mbpp-clj-091"
    "mbpp-clj-093" "mbpp-clj-097" "mbpp-clj-117" "mbpp-clj-119" "mbpp-clj-120"
    "mbpp-clj-132" "mbpp-clj-133" "mbpp-clj-137" "mbpp-clj-144" "mbpp-clj-148"
    "mbpp-clj-154" "mbpp-clj-156" "mbpp-clj-162" "mbpp-clj-163" "mbpp-clj-176"
    "mbpp-clj-179" "mbpp-clj-180" "mbpp-clj-181" "mbpp-clj-185" "mbpp-clj-191"
    "mbpp-clj-196" "mbpp-clj-197" "mbpp-clj-201" "mbpp-clj-204" "mbpp-clj-214"
    "mbpp-clj-216" "mbpp-clj-220" "mbpp-clj-226" "mbpp-clj-245" "mbpp-clj-248"
    "mbpp-clj-252" "mbpp-clj-258" "mbpp-clj-259" "mbpp-clj-266" "mbpp-clj-267"
    "mbpp-clj-268" "mbpp-clj-269" "mbpp-clj-279" "mbpp-clj-283" "mbpp-clj-286"
    "mbpp-clj-287" "mbpp-clj-295" "mbpp-clj-297" "mbpp-clj-299" "mbpp-clj-303"
    "mbpp-clj-306" "mbpp-clj-308" "mbpp-clj-313" "mbpp-clj-316" "mbpp-clj-326"
    "mbpp-clj-330" "mbpp-clj-336" "mbpp-clj-342" "mbpp-clj-346" "mbpp-clj-350"
    "mbpp-clj-361" "mbpp-clj-363" "mbpp-clj-366" "mbpp-clj-368" "mbpp-clj-371"
    "mbpp-clj-375" "mbpp-clj-376" "mbpp-clj-384" "mbpp-clj-395"})

;; =============================================================================
;; 3. Test Parsing & Dynamic Target Symbol Extraction
;; =============================================================================

(defn extract-candidate-target
  "Extracts the target symbol from `(def candidate <target-sym>)` in test file text."
  [tests-str]
  (when (string? tests-str)
    (let [reader (PushbackReader. (StringReader. tests-str))]
      (loop []
        (let [form (try (read reader false :eof) (catch Exception _ :eof))]
          (cond
            (or (nil? form) (= form :eof)) nil
            (and (seq? form)
                 (= "def" (name (first form)))
                 (= "candidate" (name (second form))))
            (nth form 2)
            :else (recur)))))))

(defn is-form->test-case
  "Converts a single `(is ...)` form into normalized `{:code ... :expected ...}`."
  [is-form target-sym]
  (let [inner (second is-form)
        replaced (walk/postwalk (fn [x] (if (= x 'candidate) target-sym x)) inner)]
    (if (and (seq? replaced) (= '= (first replaced)) (= 3 (count replaced)))
      {:code (pr-str (nth replaced 1))
       :expected (pr-str (nth replaced 2))}
      {:code (pr-str replaced)
       :expected "true"})))

(defn parse-test-cases
  "Parses all assertion forms from a MultiPL-E test file into a vector of test cases."
  [tests-str target-sym]
  (when (and (string? tests-str) target-sym)
    (let [reader (PushbackReader. (StringReader. tests-str))
          deftest-body (loop []
                         (let [form (try (read reader false :eof) (catch Exception _ :eof))]
                           (cond
                             (or (nil? form) (= form :eof)) nil
                             (and (seq? form) (= "deftest" (name (first form))))
                             (drop 2 form)
                             :else (recur))))]
      (when deftest-body
        (filterv some?
                 (mapv (fn [form]
                         (when (and (seq? form) (= "is" (name (first form))))
                           (is-form->test-case form target-sym)))
                       deftest-body))))))

;; =============================================================================
;; 4. Deterministic Test Splitting (AC4)
;; =============================================================================

(defn split-tests
  "Splits parsed test cases [a_1, ..., a_K] into {:public-tests [...] :hidden-tests [...]}
   according to AC4:
     - K >= 3: first 2 public, remaining K-2 hidden
     - K == 2: 1 public, 1 hidden
     - K == 1: 1 present in both public and hidden"
  [test-cases]
  (let [k (count test-cases)]
    (cond
      (>= k 3)
      {:public-tests (subvec test-cases 0 2)
       :hidden-tests (subvec test-cases 2)}

      (= k 2)
      {:public-tests (subvec test-cases 0 1)
       :hidden-tests (subvec test-cases 1 2)}

      (= k 1)
      {:public-tests test-cases
       :hidden-tests test-cases}

      :else
      {:public-tests []
       :hidden-tests []})))

;; =============================================================================
;; 5. Task Normalization & Ingestion (AC1)
;; =============================================================================

(defn extract-title-from-docstring
  "Derives a concise human-readable title from docstring or task ID."
  [prompt id]
  (let [first-line (-> (or prompt "")
                       str/split-lines
                       (->> (drop 1)
                            (map str/trim)
                            (remove str/blank?)
                            first))
        clean (when first-line
                (-> first-line
                    (str/replace #"^\"+\s*" "")
                    (str/replace #"\s*\"+$" "")
                    str/trim))]
    (if (and clean (pos? (count clean)) (< (count clean) 80))
      clean
      (str/replace id #"[-_]" " "))))

(defn normalize-task
  "Normalizes a raw MultiPL-E task map into canonical harness schema."
  [raw-task base-dir]
  (let [id (:id raw-task)
        source (:source raw-task)
        prompt-path (str base-dir "/" (get-in raw-task [:prompt-ref :path]))
        tests-path (str base-dir "/" (get-in raw-task [:tests-ref :path]))
        prompt-file (io/file prompt-path)
        tests-file (io/file tests-path)
        prompt (when (.exists prompt-file) (slurp prompt-file))
        tests-str (when (.exists tests-file) (slurp tests-file))
        target-sym (or (extract-candidate-target tests-str) (:entrypoint raw-task))
        test-cases (parse-test-cases tests-str target-sym)
        {:keys [public-tests hidden-tests]} (split-tests (or test-cases []))]
    {:id id
     :title (extract-title-from-docstring prompt id)
     :source source
     :fn-name target-sym
     :prompt prompt
     :public-tests public-tests
     :hidden-tests hidden-tests
     :test-cases test-cases
     :has-tests? (boolean (seq test-cases))}))

;; =============================================================================
;; 6. SCI Verification & Quarantine Classification (AC2)
;; =============================================================================

(defn verify-solution-in-sci
  "Evaluates candidate solution code against test cases in create-benchmark-sci-ctx.
   Returns {:passed? boolean :error string}."
  [code test-cases]
  (if (or (str/blank? code) (empty? test-cases))
    {:passed? false :error "Missing code or test cases"}
    (try
      (let [res (bench-core/grade-submission code test-cases)]
        {:passed? (boolean (:all-passed? res))
         :error (:error res)
         :passed-count (:passed-count res)
         :total-count (:total-count res)})
      (catch Throwable t
        {:passed? false :error (.getMessage t)}))))

;; =============================================================================
;; 7. Full Ingestion & Catalog Assembly
;; =============================================================================

(defn run-ingest
  "Executes the full ingestion pipeline:
   1. Ingests all 558 tasks from upstream reference.
   2. Normalizes schemas and applies AC4 public/hidden test split.
   3. Partitions into 447 dev tasks and 111 sealed tasks (AC3).
   4. Runs SCI verification on all solutions (AC2).
   5. Emits catalog files (tasks_dev.edn, tasks_sealed.edn, solutions.edn, quarantine.edn).
   6. Validates invariants: >= 95% floor, partition disjointness, backwards compatibility."
  [{:keys [base-dir catalog-dir solutions-map] :or {base-dir "scratch/clojure-llm"
                                                    catalog-dir "resources/catalog/gate3_evals/multipl_e"}}]
  (println "==================================================")
  (println "=== Ingesting MultiPL-E Clojure Benchmark Suite ===")
  (println "==================================================")
  (println "Source Directory   :" base-dir)
  (println "Target Catalog Dir :" catalog-dir)

  (let [tasks-file (io/file base-dir "benchmark/tasks-v0.edn")]
    (when-not (.exists tasks-file)
      (throw (IllegalStateException. (str "Upstream tasks file not found: " tasks-file))))

    (let [raw-tasks (edn/read-string (slurp tasks-file))
          _ (println (format "Loaded %d raw MultiPL-E tasks from %s" (count raw-tasks) (.getPath tasks-file)))
          normalized (mapv #(normalize-task % base-dir) raw-tasks)

          ;; Check total inventory
          _ (when-not (= 558 (count normalized))
              (throw (IllegalStateException. (format "Expected 558 tasks, found %d" (count normalized)))))

          ;; Partition into Dev (447) and Sealed (111)
          dev-tasks (filterv #(not (contains? CANONICAL-HELDOUT-IDS (:id %))) normalized)
          sealed-tasks (filterv #(contains? CANONICAL-HELDOUT-IDS (:id %)) normalized)

          _ (when-not (= 447 (count dev-tasks))
              (throw (IllegalStateException. (format "Expected 447 dev tasks, found %d" (count dev-tasks)))))
          _ (when-not (= 111 (count sealed-tasks))
              (throw (IllegalStateException. (format "Expected 111 sealed tasks, found %d" (count sealed-tasks)))))

          ;; Verify disjointness
          dev-ids (set (map :id dev-tasks))
          sealed-ids (set (map :id sealed-tasks))
          overlap (set/intersection dev-ids sealed-ids)
          _ (when (seq overlap)
              (throw (IllegalStateException. (format "Dev and Sealed sets overlap: %s" overlap))))

          ;; Load reference solutions
          sols (or solutions-map
                   (try (edn/read-string (slurp (io/file catalog-dir "solutions.edn")))
                        (catch Exception _ {})))

          ;; Evaluate all solutions in SCI
          passed (atom [])
          quarantine (atom [])]

      (println "\nEvaluating ground-truth reference solutions in hermetic SCI sandbox...")
      (doseq [t normalized
              :let [id (:id t)
                    code (get sols id)
                    test-cases (:test-cases t)]]
        (cond
          (not (:has-tests? t))
          (swap! quarantine conj {:id id
                                  :source (:source t)
                                  :fn-name (:fn-name t)
                                  :reason "Malformed upstream test file or syntax error (e.g. unescaped quotes)"})

          (str/blank? code)
          (swap! quarantine conj {:id id
                                  :source (:source t)
                                  :fn-name (:fn-name t)
                                  :reason "No verified reference solution available"})

          :else
          (let [ver (verify-solution-in-sci code test-cases)]
            (if (:passed? ver)
              (swap! passed conj id)
              (swap! quarantine conj {:id id
                                      :source (:source t)
                                      :fn-name (:fn-name t)
                                      :reason (or (:error ver)
                                                  (format "Failed %d/%d assertions"
                                                          (- (:total-count ver) (:passed-count ver))
                                                          (:total-count ver)))})))))

      (let [n-total (count normalized)
            n-passed (count @passed)
            pass-rate (double (/ n-passed n-total))
            _ (println (format "SCI Sandbox Verification: %d/%d passed (%.2f%% floor)"
                               n-passed n-total (* 100.0 pass-rate)))
            _ (println (format "Quarantined Tasks        : %d (%.2f%%)"
                               (count @quarantine) (* 100.0 (/ (count @quarantine) n-total))))]

        ;; AC2 Invariant Check: >= 95% verification floor
        (when (< pass-rate 0.95)
          (throw (IllegalStateException.
                  (format "Verification floor violated: expected >= 95.0%%, achieved %.2f%% (%d/%d)"
                          (* 100.0 pass-rate) n-passed n-total))))

        ;; Strip internal helper keys from catalog schemas
        (let [clean-task (fn [t] (dissoc t :test-cases :has-tests?))
              clean-dev (mapv clean-task dev-tasks)
              clean-sealed (mapv clean-task sealed-tasks)
              catalog-f (io/file catalog-dir)]
          (.mkdirs catalog-f)

          ;; Write catalog artifacts
          (let [dev-out (io/file catalog-dir "tasks_dev.edn")
                sealed-out (io/file catalog-dir "tasks_sealed.edn")
                quarantine-out (io/file catalog-dir "quarantine.edn")
                sols-out (io/file catalog-dir "solutions.edn")]

            (spit dev-out (with-out-str (pprint/pprint clean-dev)))
            (spit sealed-out (with-out-str (pprint/pprint clean-sealed)))
            (spit quarantine-out (with-out-str (pprint/pprint (vec @quarantine))))
            (spit sols-out (with-out-str (pprint/pprint sols)))

            (println "\nWrote catalog artifacts to:" catalog-dir)
            (println "  - tasks_dev.edn    :" (count clean-dev) "tasks")
            (println "  - tasks_sealed.edn :" (count clean-sealed) "tasks")
            (println "  - solutions.edn    :" (count sols) "reference solutions")
            (println "  - quarantine.edn   :" (count @quarantine) "quarantined tasks")
            (println "\nMultiPL-E Ingestion successfully completed!"))

          {:total n-total
           :dev-count (count clean-dev)
           :sealed-count (count clean-sealed)
           :passed-count n-passed
           :pass-rate pass-rate
           :quarantine-count (count @quarantine)})))))

(defn -main
  "CLI entrypoint for MultiPL-E ingestion."
  [& args]
  (try
    (let [base-dir (or (first args) "scratch/clojure-llm")
          catalog-dir (or (second args) "resources/catalog/gate3_evals/multipl_e")]
      (run-ingest {:base-dir base-dir :catalog-dir catalog-dir})
      (System/exit 0))
    (catch Throwable e
      (println "\nIngestion Exception:" (.getMessage e))
      (.printStackTrace e)
      (System/exit 1))))
