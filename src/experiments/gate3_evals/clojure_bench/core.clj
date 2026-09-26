(ns experiments.gate3-evals.clojure-bench.core
  "Pure functional core for clojure_bench (Gate 3 Evals, Stage 2).
   Provides prompt synthesis, submission detection, hermetic SCI sandbox grading,
   public test feedback formatting, best-submission selection, and telemetry metrics calculation."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [sci.core :as sci]
            [tools.gemma4-agent :as agent])
  (:import [java.io File]
           [java.security MessageDigest]))

;; =============================================================================
;; 1. Sealed Test Integrity (SHA-256)
;; =============================================================================

(defn compute-bytes-sha256
  "Computes lowercase hexadecimal SHA-256 hash for byte array `bs`."
  [^bytes bs]
  (let [md (MessageDigest/getInstance "SHA-256")
        digest (.digest md bs)
        sb (StringBuilder.)]
    (doseq [b digest]
      (.append sb (format "%02x" (bit-and (int b) 0xff))))
    (.toString sb)))

(defn compute-file-sha256
  "Computes lowercase hexadecimal SHA-256 hash for a File or file path string using streaming buffer."
  [file-or-path]
  (let [^File f (if (instance? File file-or-path)
                  file-or-path
                  (File. (str file-or-path)))]
    (if (.exists f)
      (let [md (MessageDigest/getInstance "SHA-256")
            buffer (byte-array (* 64 1024))]
        (with-open [is (java.io.FileInputStream. f)]
          (loop []
            (let [read (.read is buffer)]
              (when (pos? read)
                (.update md buffer 0 read)
                (recur)))))
        (let [digest (.digest md)
              sb (StringBuilder.)]
          (doseq [b digest]
            (.append sb (format "%02x" (bit-and (int b) 0xff))))
          (.toString sb)))
      (throw (IllegalArgumentException. (str "File not found for sha256 computation: " (.getPath f)))))))

;; =============================================================================
;; 2. Tightened Grading SCI Context
;; =============================================================================

(defn create-tightened-grading-ctx
  "Creates a tightened, hermetic SCI evaluation context for grading Clojure benchmark tasks.
   Provides pure Clojure primitives, math, atom/swap!/deref, and macro definitions.
   Strictly isolates file I/O (no slurp, spit, list-files), system inspection, and reflection."
  []
  (sci/init
   {:classes {'Math Math}
    :bindings {'println println
               'print print
               'prn prn
               'str str}}))

;; =============================================================================
;; 3. Prompt Synthesis
;; =============================================================================

(def SINGLE-SHOT-SYSTEM-PROMPT
  "You are an expert Clojure engineer. Implement the requested function or macro in pure, idiomatic Clojure.
Output the complete definition in a ```clojure ... ``` code block without using tools or extra commentary.")

(defn render-benchmark-prompt
  "Renders the formatted user prompt for a task, embedding public test examples.
   Never includes sealed/hidden tests to preserve evaluation integrity."
  [{:keys [title fn-name prompt public-tests]}]
  (let [fn-str (str fn-name)
        tests-formatted (str/join "\n"
                                  (map (fn [{:keys [code expected]}]
                                         (format "  - Code: %s\n    Expected: %s" code expected))
                                       public-tests))]
    (format
     "Task: %s\nTarget Identifier: `%s`\n\nInstructions:\n%s\n\nPublic Examples & Unit Tests:\n%s\n\nImplement `%s` in pure, idiomatic Clojure. Use standard bindings and collections."
     title fn-str prompt tests-formatted fn-str)))

;; =============================================================================
;; 4. Submission Detection
;; =============================================================================

(defn- extract-top-level-symbols
  "Extracts all top-level symbols defined or declared in Clojure code."
  [code-str]
  (when (string? code-str)
    (let [reader-ctx (sci/init {})]
      (try
        (loop [reader (sci/source-reader code-str)
               syms []]
          (let [form (try (sci/parse-next reader-ctx reader) (catch Throwable _ nil))]
            (if (or (nil? form) (= form :sci.core/eof))
              syms
              (let [found-sym (cond
                                (seq? form)
                                (let [op (first form)
                                      target (second form)]
                                  (cond
                                    ('#{defn defn- defmacro def defmulti defmethod} op)
                                    (str target)

                                    (and ('#{def} op) (symbol? target))
                                    (str target)

                                    :else nil))
                                :else nil)]
                (recur reader (if found-sym (conj syms found-sym) syms))))))
        (catch Throwable _ [])))))

(defn submission-form?
  "Returns true if `code-str` defines or declares the target `fn-name` (e.g. defn, defmacro, or def)."
  [code-str fn-name]
  (let [target-name (name (or fn-name ""))
        top-syms (extract-top-level-symbols (or code-str ""))]
    (cond
      (some #(= % target-name) top-syms) true
      (seq top-syms) false
      ;; Fallback regex in case of partial reader errors on malformed forms
      :else (boolean (re-find (re-pattern (str "(?s)\\((?:defn|defmacro|def)\\s+"
                                               (java.util.regex.Pattern/quote target-name)
                                               "(?![\\w\\-\\?\\!\\*\\+\\/])"))
                              (or code-str ""))))))

(defn extract-markdown-code-blocks
  "Extracts all ```clojure ... ``` or ```clj ... ``` or bare ``` ... ``` code blocks from text."
  [text]
  (let [pattern #"(?s)```(?:clojure|clj)?\s*\n?(.*?)(?:```|$)"
        raw-matches (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern (or text "")))))]
    (mapv (fn [block]
            (-> block
                (str/replace #"^```[a-z]*>?" "")
                (str/replace #"```$" "")
                str/trim))
          raw-matches)))

(defn extract-candidate-code
  "Extracts candidate Clojure code for target `fn-name` from raw text.
   Checks unwrapped tool calls, markdown code blocks, and raw s-expressions,
   preferring forms that define or declare `fn-name`."
  [text fn-name]
  (let [target (name (or fn-name ""))
        stripped (agent/strip-thinking-trace (or text ""))
        tool-code (when-let [tc (agent/extract-tool-call stripped)]
                    (:code tc))
        blocks (extract-markdown-code-blocks stripped)
        raw-sexpr (agent/extract-balanced-sexpr stripped)
        candidates (vec (distinct (filter seq (concat (when tool-code [tool-code])
                                                      blocks
                                                      (when raw-sexpr [raw-sexpr])))))
        best-stripped (or (first (filter #(submission-form? % target) candidates))
                          (first candidates))]
    (if (and best-stripped (submission-form? best-stripped target))
      best-stripped
      ;; Fallback to searching unstripped text if no matching submission found outside thinking
      (let [fb-tool-code (when-let [tc (agent/extract-tool-call text)] (:code tc))
            fb-blocks (extract-markdown-code-blocks text)
            fb-sexpr (agent/extract-balanced-sexpr text)
            fb-candidates (vec (distinct (filter seq (concat (when fb-tool-code [fb-tool-code])
                                                             fb-blocks
                                                             (when fb-sexpr [fb-sexpr])))))]
        (or (first (filter #(submission-form? % target) fb-candidates))
            best-stripped
            (first fb-candidates))))))

;; =============================================================================
;; 5. Hermetic Grading Engine
;; =============================================================================

(defn grade-submission
  "Evaluates candidate `code-str` in a fresh tightened SCI context and executes `test-cases`.
   Each test case is a map `{:keys [code expected]}`.
   Returns a structured grading verdict map:
   `{:all-passed? bool :passed-count n :total-count m :results [...] :error err-msg}`."
  ([code-str test-cases]
   (grade-submission (create-tightened-grading-ctx) code-str test-cases))
  ([sci-ctx code-str test-cases]
   (let [clean-code (str/trim (or code-str ""))]
     (if (empty? clean-code)
       {:all-passed? false
        :passed-count 0
        :total-count (count test-cases)
        :results []
        :error "Empty candidate code"}
       (try
         ;; 1. Evaluate candidate definition in the tightened context
         (sci/eval-string* sci-ctx clean-code)

         ;; 2. Run each test case against the defined function
         (let [results (mapv (fn [{:keys [code expected]}]
                               (try
                                 (let [actual (sci/eval-string* sci-ctx code)
                                       expected-val (edn/read-string (str expected))
                                       passed? (= actual expected-val)]
                                   {:code code
                                    :expected expected
                                    :actual (pr-str actual)
                                    :passed? passed?})
                                 (catch Throwable te
                                   {:code code
                                    :expected expected
                                    :actual nil
                                    :error (.getMessage te)
                                    :passed? false})))
                             test-cases)
               passed-count (count (filter :passed? results))
               total-count (count test-cases)]
           {:all-passed? (= passed-count total-count)
            :passed-count passed-count
            :total-count total-count
            :results results
            :error nil})
         (catch Throwable e
           {:all-passed? false
            :passed-count 0
            :total-count (count test-cases)
            :results []
            :error (str "Compilation/Execution Exception: " (.getMessage e))}))))))

;; =============================================================================
;; 6. Public Test Feedback Formatting
;; =============================================================================

(defn format-public-feedback
  "Formats public test execution results into an informative observation string for the agent."
  [{:keys [all-passed? passed-count total-count results error]}]
  (cond
    (seq error)
    (format "[Public tests: 0/%d passed. Error encountered: %s]" total-count error)

    all-passed?
    (format "[Public tests: %d/%d passed. All public examples succeeded!]" passed-count total-count)

    :else
    (let [failures (filter #(not (:passed? %)) results)
          failure-lines (map (fn [{:keys [code expected actual error]}]
                               (if error
                                 (format "  - Code: %s\n    Error: %s" code error)
                                 (format "  - Code: %s\n    Expected: %s\n    Actual: %s" code expected actual)))
                             failures)]
      (format "[Public tests: %d/%d passed. Failing tests:\n%s\nPlease revise your implementation.]"
              passed-count total-count (str/join "\n" failure-lines)))))

;; =============================================================================
;; 7. Submission Selection
;; =============================================================================

(defn pick-best-public-submission
  "Given a sequence of submission maps `[{:code ... :public-res ...}]`,
   selects the candidate with the highest passed-count, breaking ties by later submission."
  [submissions]
  (when (seq submissions)
    (let [scored (map-indexed (fn [idx sub]
                                (let [passed (long (get-in sub [:public-res :passed-count] 0))
                                      total (long (get-in sub [:public-res :total-count] 1))
                                      ratio (if (pos? total) (double (/ passed total)) 0.0)]
                                  {:sub sub
                                   :idx idx
                                   :passed passed
                                   :ratio ratio}))
                              submissions)]
      (:sub (last (sort-by (juxt :ratio :passed :idx) scored))))))

;; =============================================================================
;; 8. Metrics & Reporting
;; =============================================================================

(defn calculate-metrics
  "Computes summary metrics across a collection of evaluation rows."
  [rows]
  (let [real-rows (filterv #(not (:dry-run? %)) rows)
        effective-rows (if (seq real-rows) real-rows (vec rows))
        total-evals (count effective-rows)
        passed-evals (count (filter :passed? effective-rows))
        mean-hidden-pass-rate (if (pos? total-evals) (double (/ passed-evals total-evals)) 0.0)

        by-mode (group-by :mode effective-rows)
        single-shot-rows (get by-mode :single-shot [])
        agentic-rows (get by-mode :agentic [])

        calc-mode-stats (fn [group-rows]
                          (let [cnt (count group-rows)
                                pass-cnt (count (filter :passed? group-rows))
                                pass-rate (if (pos? cnt) (double (/ pass-cnt cnt)) 0.0)
                                tokens-in (reduce + 0 (map #(long (or (:tokens-in %) 0)) group-rows))
                                tokens-out (reduce + 0 (map #(long (or (:tokens-out %) 0)) group-rows))
                                wall-ms (reduce + 0.0 (map #(double (or (:wall-ms %) 0.0)) group-rows))]
                            {:count cnt
                             :passed pass-cnt
                             :pass-rate pass-rate
                             :tokens-in tokens-in
                             :tokens-out tokens-out
                             :wall-ms wall-ms}))

        single-stats (calc-mode-stats single-shot-rows)
        agentic-stats (calc-mode-stats agentic-rows)

        ;; pass@1: tasks where the first submission passed
        pass-at-1-cnt (count (filter #(and (:passed? %) (<= (long (or (:n-submissions %) 1)) 1)) rows))
        pass-at-1 (if (pos? total-evals) (double (/ pass-at-1-cnt total-evals)) 0.0)

        ;; median submissions to pass among agentic passing tasks
        agentic-passed-subs (sort (map #(long (or (:n-submissions %) 1)) (filter :passed? agentic-rows)))
        median-submissions (if (seq agentic-passed-subs)
                             (let [n (count agentic-passed-subs)
                                   mid (quot n 2)]
                               (if (odd? n)
                                 (double (nth agentic-passed-subs mid))
                                 (/ (+ (double (nth agentic-passed-subs mid))
                                       (double (nth agentic-passed-subs (dec mid))))
                                    2.0)))
                             0.0)

        by-task (into (sorted-map)
                      (map (fn [[task task-rows]]
                             [task {:passed? (boolean (some :passed? task-rows))
                                    :single-shot (boolean (:passed? (first (filter #(= (:mode %) :single-shot) task-rows))))
                                    :agentic (boolean (:passed? (first (filter #(= (:mode %) :agentic) task-rows))))}])
                           (group-by :task rows)))]
    {:total-evals total-evals
     :passed-evals passed-evals
     :mean-hidden-pass-rate mean-hidden-pass-rate
     :pass-at-1 pass-at-1
     :median-submissions median-submissions
     :single-shot single-stats
     :agentic agentic-stats
     :by-task by-task}))

(defn format-results-row
  "Formats an individual task evaluation result map for results.edn, persisting
   rich diagnostics (candidate-code, error, test-summary, failure details, stop-reason)."
  [{:keys [model task mode candidate-code grade-res error n-submissions tokens-in tokens-out wall-ms sealed-sha checkpoint-sha dry-run?]}]
  (let [all-passed? (boolean (:all-passed? grade-res))
        has-error? (seq (or error (:error grade-res)))
        err-msg (or error (:error grade-res))
        has-candidate? (boolean (seq candidate-code))
        raw-failures (when-not all-passed?
                       (mapv (fn [{:keys [code expected actual error]}]
                               (let [m {:code code
                                        :expected (str expected)}]
                                 (cond-> m
                                   actual (assoc :actual (str actual))
                                   error (assoc :error (str error)))))
                             (filter #(not (:passed? %)) (or (:results grade-res) []))))
        stop-reason (cond
                      all-passed? :passed-all
                      (not has-candidate?) :no-extraction
                      (and has-error? (str/includes? (str err-msg) "Compilation/Execution Exception")) :compilation-error
                      (and has-error? (not (seq raw-failures))) :error
                      :else :test-failure)]
    {:model (str model)
     :task (str task)
     :mode (keyword mode)
     :passed? all-passed?
     :n-submissions (long (or n-submissions 1))
     :tokens-in (long (or tokens-in 0))
     :tokens-out (long (or tokens-out 0))
     :wall-ms (double (or wall-ms 0.0))
     :sealed-sha (str sealed-sha)
     :checkpoint-sha (str checkpoint-sha)
     :stop-reason stop-reason
     :dry-run? (boolean dry-run?)
     :candidate-code candidate-code
     :error err-msg
     :test-summary {:passed (long (or (:passed-count grade-res) 0))
                    :total (long (or (:total-count grade-res) 0))}
     :failures (when (seq raw-failures) raw-failures)}))

(defn read-results-edn
  "Reads all EDN rows from results-file."
  [results-file]
  (let [f (io/file results-file)]
    (if (.exists f)
      (with-open [r (java.io.PushbackReader. (io/reader f))]
        (loop [rows []]
          (let [row (edn/read {:eof ::eof} r)]
            (if (= row ::eof)
              rows
              (recur (conj rows row))))))
      [])))

(defn format-summary-csv
  "Formats summary metrics into a clean CSV string."
  [metrics]
  (let [{:keys [total-evals passed-evals mean-hidden-pass-rate pass-at-1 median-submissions single-shot agentic]} metrics
        ss-rate (:pass-rate single-shot 0.0)
        ag-rate (:pass-rate agentic 0.0)
        delta (- ag-rate ss-rate)]
    (str "metric,value,single_shot,agentic,delta_loop,target\n"
         (format "mean_hidden_pass_rate,%.4f,%.4f,%.4f,%.4f,>=0.5000\n"
                 mean-hidden-pass-rate ss-rate ag-rate delta)
         (format "pass_at_1,%.4f,N/A,N/A,N/A,>=0.4000\n" pass-at-1)
         (format "median_submissions_to_pass,%.2f,N/A,%.2f,N/A,<=3.0\n"
                 median-submissions median-submissions)
         (format "total_evals,%d,%d,%d,N/A,10 tasks\n"
                 total-evals (:count single-shot 0) (:count agentic 0))
         (format "passed_evals,%d,%d,%d,N/A,N/A\n"
                 passed-evals (:passed single-shot 0) (:passed agentic 0)))))
