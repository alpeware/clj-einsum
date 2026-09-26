(ns experiments.gate3-evals.clojure-bench.run
  "Stage 2 Implementation & Benchmark Runner for clojure_bench (Gate 3 Evals).
   Evaluates raw s-expression Clojure writing ability (single-shot) and loop ability (agentic)
   on Gemma 4 family models under OpenXLA PJRT execution and hermetic SCI grading."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [tools.cli :as cli]
            [tools.gemma4-agent :as agent]
            [tools.gemma4-inference :as gemma4-inf]))

;; =============================================================================
;; 1. Default Evaluation Configuration
;; =============================================================================

(def DEFAULT-BENCH-OPTS
  {:model ".models/gemma-4-E2B-it-int4"
   :backend :rocm
   :mode :all ;; :single-shot, :agentic, or :all
   :tasks "all"
   :system bench-core/AGENT-SYSTEM-PROMPT
   :max-turns 5
   :max-consecutive-errors 3
   :max-new-tokens 1536
   :max-seq-len 2048
   :temperature 0.0
   :top-k 10
   :repetition-penalty 1.0
   :thinking true
   :nudge-on-no-tool true
   :overwrite false
   :dry-run false
   :quiet false
   :public-tasks-file "resources/proposals/gate3_evals/clojure_bench/tasks_public.edn"
   :sealed-tasks-file "resources/proposals/gate3_evals/clojure_bench/tasks_sealed.edn"
   :results-file "resources/proposals/gate3_evals/clojure_bench/results.edn"
   :summary-file "resources/proposals/gate3_evals/clojure_bench/summary.csv"})

;; Mock reference solutions used for dry-run verification
(def MOCK-REFERENCE-SOLUTIONS
  {"first-n" "(defn first-n [coll] (vec (take 10 coll)))"
   "my-range" "(defn my-range ([end] (my-range 0 end)) ([start end] (loop [i start acc []] (if (>= i end) acc (recur (inc i) (conj acc i))))))"
   "deep-flatten" "(defn deep-flatten [coll] (reduce (fn [acc x] (if (sequential? x) (into acc (deep-flatten x)) (conj acc x))) [] coll))"
   "freqs" "(defn freqs [coll] (reduce (fn [m x] (update m x (fnil inc 0))) {} coll))"
   "partition-by-parity" "(defn partition-by-parity [coll] (reduce (fn [m x] (if (even? x) (update m :even conj x) (update m :odd conj x))) {:even [] :odd []} coll))"
   "my-comp" "(defn my-comp [& fns] (if (empty? fns) identity (let [rfns (reverse fns) r (first rfns) rest-fns (rest rfns)] (fn [& args] (reduce (fn [acc f] (f acc)) (apply r args) rest-fns)))))"
   "balanced-delims?" "(defn balanced-delims? [s] (let [matches {\\) \\( \\] \\[ \\} \\{} opens #{\\( \\[ \\{}] (loop [chars (seq s) stack ()] (if-let [c (first chars)] (cond (opens c) (recur (rest chars) (conj stack c)) (matches c) (if (= (first stack) (matches c)) (recur (rest chars) (pop stack)) false) :else (recur (rest chars) stack)) (empty? stack)))))"
   "deep-update-vals" "(defn deep-update-vals [f x] (cond (map? x) (reduce-kv (fn [m k v] (assoc m k (deep-update-vals f v))) {} x) (vector? x) (mapv #(deep-update-vals f %) x) :else (f x)))"
   "lazy-interleave" "(defn lazy-interleave [c1 c2] (lazy-seq (let [s1 (seq c1) s2 (seq c2)] (cond (and s1 s2) (cons (first s1) (cons (first s2) (lazy-interleave (rest s1) (rest s2)))) s1 s1 s2 s2 :else nil))))"
   "my-or" "(defmacro my-or ([] nil) ([x] x) ([x & next] (list (quote let) [(quote or#) x] (list (quote if) (quote or#) (quote or#) (cons (quote my-or) next)))))"})

;; =============================================================================
;; 2. Session Initialization & Weight Hash
;; =============================================================================

(def ^:private checkpoint-hash-cache (atom {}))

(defn- compute-checkpoint-hash
  "Computes SHA-256 for the model weights file (model.safetensors) with caching."
  [model-dir]
  (let [safetensors (io/file model-dir "model.safetensors")]
    (if (.exists safetensors)
      (let [cache-key [(.getAbsolutePath safetensors) (.length safetensors) (.lastModified safetensors)]]
        (if-let [cached (get @checkpoint-hash-cache cache-key)]
          cached
          (let [sha (bench-core/compute-file-sha256 safetensors)]
            (swap! checkpoint-hash-cache assoc cache-key sha)
            sha)))
      "unknown-safetensors-sha256")))

(defn init-benchmark-session
  "Initializes pinned persistent VRAM session for Gemma 4 benchmark evaluation."
  [opts]
  (if (:dry-run opts)
    {:opts opts
     :model-dir (:model opts)
     :checkpoint-sha "dry-run-checkpoint-sha"}
    (let [_ (when (= (keyword (:backend opts)) :rocm)
              (Thread/sleep 3000))
          model-dir (cli/find-model-dir (or (:model-dir opts) (:model opts)) :gemma-4)
          max-seq-len (long (or (:max-seq-len opts) 1024))
          checkpoint-sha (compute-checkpoint-hash model-dir)
          session-opts (assoc opts
                              :model-dir model-dir
                              :model model-dir
                              :mode :agent
                              :max-seq-len max-seq-len
                              :max-new-tokens (long (or (:max-new-tokens opts) 512))
                              :sandbox :benchmark)
          session (gemma4-inf/init-agent-vram-session session-opts max-seq-len)]
      (assoc session
             :model-dir model-dir
             :checkpoint-sha checkpoint-sha))))

;; =============================================================================
;; 3. Evaluation Drivers: Single-Shot & Agentic
;; =============================================================================

(defn run-single-shot-task
  "Runs single-shot Clojure generation:
   Generates once, extracts candidate code, grades against sealed tests in clean SCI sandbox."
  [session task hidden-tests sealed-sha opts]
  (let [{:keys [dry-run quiet]} opts
        task-id (:id task)
        fn-name (:fn-name task)
        prompt (bench-core/render-benchmark-prompt task)
        model-name (.getName (io/file (or (:model-dir session) (:model opts))))
        checkpoint-sha (:checkpoint-sha session)
        _ (when-not quiet (println (format "\n[Single-Shot] Task: %s (%s)..." task-id fn-name)))
        t0 (System/nanoTime)]

    (if dry-run
      ;; Dry-run mode: use verified mock code
      (let [mock-code (get MOCK-REFERENCE-SOLUTIONS task-id "(defn stub [x] x)")
            grade-res (bench-core/grade-submission mock-code hidden-tests)
            wall-ms (/ (- (System/nanoTime) t0) 1e6)]
        (bench-core/format-results-row
         {:model model-name
          :task task-id
          :mode :single-shot
          :candidate-code mock-code
          :grade-res grade-res
          :n-submissions 1
          :tokens-in 120
          :tokens-out 45
          :wall-ms wall-ms
          :sealed-sha sealed-sha
          :checkpoint-sha checkpoint-sha
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :dry-run? true}))

      ;; Actual model inference
      (let [chat-prompt (agent/format-agent-chat-prompt
                         bench-core/SINGLE-SHOT-SYSTEM-PROMPT
                         [{:role :user :content prompt}]
                         8
                         (boolean (:thinking opts))
                         nil)
            gen-res (gemma4-inf/generate-new-tokens-and-text session chat-prompt)
            wall-ms (/ (- (System/nanoTime) t0) 1e6)
            gen-text (:text gen-res)
            prompt-tokens (long (or (:prompt-tokens gen-res) 0))
            new-tokens (long (or (:new-tokens gen-res) 0))
            candidate-code (bench-core/extract-candidate-code gen-text fn-name)
            grade-res (if (seq candidate-code)
                        (bench-core/grade-submission candidate-code hidden-tests)
                        {:all-passed? false :passed-count 0 :total-count (count hidden-tests) :error "No code extracted"})
            error-msg (when-not (seq candidate-code) "No Clojure code extracted from model generation")]
        (when-not quiet
          (println (format "  ↳ Result: %s (Pass: %d/%d, %.1f ms)"
                           (if (:all-passed? grade-res) "PASS" "FAIL")
                           (:passed-count grade-res)
                           (:total-count grade-res)
                           wall-ms))
          (when-not (:all-passed? grade-res)
            (if candidate-code
              (println (format "    ↳ Candidate Code: %s" (str/replace candidate-code #"\n" " ")))
              (println "    ↳ No candidate code extracted."))))
        (bench-core/format-results-row
         {:model model-name
          :task task-id
          :mode :single-shot
          :candidate-code candidate-code
          :grade-res grade-res
          :error error-msg
          :n-submissions 1
          :tokens-in prompt-tokens
          :tokens-out new-tokens
          :wall-ms wall-ms
          :sealed-sha sealed-sha
          :checkpoint-sha checkpoint-sha
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :dry-run? false})))))

(defn run-agentic-task
  "Runs agentic evaluation with eval_clojure tool and submission hook:
   Tool calls that define the target fn execute public tests and receive public feedback.
   The best-public submission is graded against sealed hidden tests exactly once."
  [session task hidden-tests sealed-sha opts]
  (let [{:keys [dry-run quiet max-turns]} opts
        task-id (:id task)
        fn-name (:fn-name task)
        prompt (bench-core/render-benchmark-prompt task)
        model-name (.getName (io/file (or (:model-dir session) (:model opts))))
        checkpoint-sha (:checkpoint-sha session)
        submissions (atom [])
        _ (when-not quiet (println (format "\n[Agentic Loop] Task: %s (%s, max %d turns)..." task-id fn-name (long (or max-turns 5)))))
        t0 (System/nanoTime)]

    (if dry-run
      ;; Dry run mode
      (let [mock-code (get MOCK-REFERENCE-SOLUTIONS task-id "(defn stub [x] x)")
            pub-res (bench-core/grade-submission mock-code (:public-tests task))
            _ (swap! submissions conj {:code mock-code :public-res pub-res})
            best-sub (bench-core/pick-best-public-submission @submissions)
            hidden-res (bench-core/grade-submission (:code best-sub) hidden-tests)
            wall-ms (/ (- (System/nanoTime) t0) 1e6)]
        (bench-core/format-results-row
         {:model model-name
          :task task-id
          :mode :agentic
          :candidate-code (:code best-sub)
          :grade-res hidden-res
          :n-submissions (count @submissions)
          :tokens-in 240
          :tokens-out 90
          :wall-ms wall-ms
          :sealed-sha sealed-sha
          :checkpoint-sha checkpoint-sha
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :dry-run? true}))

      ;; Actual agentic loop execution
      (let [submission-tool-hook
            (fn [sci-ctx tool-code]
              (let [eval-res (agent/eval-tool-code sci-ctx tool-code)
                    candidate (bench-core/extract-candidate-code tool-code fn-name)
                    is-sub? (and (seq candidate) (bench-core/submission-form? candidate fn-name))]
                (if is-sub?
                  (let [pub-res (bench-core/grade-submission sci-ctx candidate (:public-tests task))
                        feedback (bench-core/format-public-feedback pub-res)]
                    (swap! submissions conj {:code candidate :public-res pub-res})
                    (update eval-res :output #(str % "\n" feedback)))
                  eval-res)))

            task-session (update session :opts assoc
                                 :sandbox :benchmark
                                 :tool-eval-fn submission-tool-hook
                                 :quiet quiet)
            transcript (agent/run-agent-loop task-session prompt (bench-core/create-tightened-grading-ctx) submission-tool-hook)
            telemetry (get (meta transcript) :turn-telemetry [])
            total-in (reduce + 0 (map #(long (or (:prompt-tokens %) 0)) telemetry))
            total-out (reduce + 0 (map #(long (or (:new-tokens %) 0)) telemetry))
            total-turn-ms (reduce + 0.0 (map #(double (or (:total-turn-ms %) 0.0)) telemetry))
            wall-ms (if (pos? total-turn-ms) total-turn-ms (/ (- (System/nanoTime) t0) 1e6))

            ;; Fallback if agent provided code in response text rather than tool call
            _ (when (empty? @submissions)
                (let [last-text (or (:response (last transcript)) (:raw (last transcript)) (:content (last transcript)) "")
                      candidate (bench-core/extract-candidate-code last-text fn-name)]
                  (when (and (seq candidate) (bench-core/submission-form? candidate fn-name))
                    (let [pub-res (bench-core/grade-submission candidate (:public-tests task))]
                      (swap! submissions conj {:code candidate :public-res pub-res})))))

            best-sub (bench-core/pick-best-public-submission @submissions)
            candidate-code (:code best-sub)
            hidden-res (if (seq candidate-code)
                         (bench-core/grade-submission candidate-code hidden-tests)
                         {:all-passed? false :passed-count 0 :total-count (count hidden-tests) :error "No submission found"})
            error-msg (when-not (seq candidate-code) "No candidate submission found during agent loop")]

        (when-not quiet
          (println (format "  ↳ Result: %s (Pass: %d/%d, %d submissions, %.1f ms)"
                           (if (:all-passed? hidden-res) "PASS" "FAIL")
                           (:passed-count hidden-res)
                           (:total-count hidden-res)
                           (count @submissions)
                           wall-ms))
          (when-not (:all-passed? hidden-res)
            (if candidate-code
              (println (format "    ↳ Best Candidate: %s" (str/replace candidate-code #"\n" " ")))
              (println "    ↳ No candidate submission found."))))
        (bench-core/format-results-row
         {:model model-name
          :task task-id
          :mode :agentic
          :candidate-code candidate-code
          :grade-res hidden-res
          :error error-msg
          :n-submissions (count @submissions)
          :tokens-in total-in
          :tokens-out total-out
          :wall-ms wall-ms
          :sealed-sha sealed-sha
          :checkpoint-sha checkpoint-sha
          :transcript (vec transcript)
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :dry-run? false})))))

;; =============================================================================
;; 4. Benchmark Orchestration & Reporting
;; =============================================================================

(defn- filter-tasks
  "Filters task collection based on `--tasks` flag."
  [tasks task-filter-str]
  (if (or (nil? task-filter-str) (= task-filter-str "all"))
    tasks
    (let [ids (set (str/split task-filter-str #","))]
      (filterv #(contains? ids (:id %)) tasks))))

(defn run-benchmark
  "Executes the clojure_bench evaluation suite across specified tasks and modes."
  [opts]
  (let [dry-run? (boolean (:dry-run opts))
        results-path (if (and dry-run? (= (:results-file opts) (:results-file DEFAULT-BENCH-OPTS)))
                       "resources/proposals/gate3_evals/clojure_bench/results_dry_run.edn"
                       (:results-file opts))
        summary-path (if (and dry-run? (= (:summary-file opts) (:summary-file DEFAULT-BENCH-OPTS)))
                       "resources/proposals/gate3_evals/clojure_bench/summary_dry_run.csv"
                       (:summary-file opts))
        opts (assoc opts :results-file results-path :summary-file summary-path)
        public-file (io/file (:public-tasks-file opts))
        sealed-file (io/file (:sealed-tasks-file opts))
        results-file (io/file results-path)
        summary-file (io/file summary-path)

        _ (when-not (.exists public-file)
            (throw (IllegalArgumentException. (str "Public tasks file not found: " public-file))))
        _ (when-not (.exists sealed-file)
            (throw (IllegalArgumentException. (str "Sealed tasks file not found: " sealed-file))))

        sealed-sha (bench-core/compute-file-sha256 sealed-file)
        public-tasks (edn/read-string (slurp public-file))
        sealed-map (into {} (map (juxt :id :hidden-tests) (edn/read-string (slurp sealed-file))))
        selected-tasks (filter-tasks public-tasks (:tasks opts))
        run-mode (keyword (:mode opts))

        _ (when-not (:quiet opts)
            (println "==================================================")
            (println "=== clojure_bench: Gemma 4 Clojure Eval Suite ===")
            (println "==================================================")
            (println (format "Model Checkpoint     : %s" (:model opts)))
            (println (format "Backend              : %s" (:backend opts)))
            (println (format "Sealed SHA-256       : %s" sealed-sha))
            (println (format "Evaluation Mode      : %s" run-mode))
            (println (format "Selected Tasks       : %d/%d" (count selected-tasks) (count public-tasks)))
            (println (format "Dry Run Mode         : %s" dry-run?))
            (println (format "Results File         : %s" (.getPath results-file)))
            (println (format "Summary File         : %s" (.getPath summary-file)))
            (println "=================================================="))

        _ (when (:overwrite opts)
            (spit results-file ""))

        session (init-benchmark-session opts)
        all-results (atom [])]

    (try
      (doseq [task selected-tasks]
        (let [task-id (:id task)
              hidden-tests (get sealed-map task-id [])]

          ;; 1. Single-shot evaluation
          (when (or (= run-mode :all) (= run-mode :single-shot))
            (let [ss-row (run-single-shot-task session task hidden-tests sealed-sha opts)]
              (swap! all-results conj ss-row)
              (spit results-file (str (pr-str ss-row) "\n") :append true)))

          ;; 2. Agentic evaluation
          (when (or (= run-mode :all) (= run-mode :agentic))
            (let [ag-row (run-agentic-task session task hidden-tests sealed-sha opts)]
              (swap! all-results conj ag-row)
              (spit results-file (str (pr-str ag-row) "\n") :append true)))))

      (finally
        (when (and (not (:dry-run opts)) (map? session))
          (try (gemma4-inf/close-agent-session! session) (catch Throwable _ nil)))))

    ;; Summarize cumulative metrics from results-file and output CSV
    (let [cumulative-rows (bench-core/read-results-edn results-file)
          metrics (bench-core/calculate-metrics cumulative-rows)
          summary-csv (bench-core/format-summary-csv metrics)
          run-metrics (bench-core/calculate-metrics @all-results)]
      (spit summary-file summary-csv)
      (when-not (:quiet opts)
        (println "\n==================================================")
        (println "=== clojure_bench Empirical Measurement Summary ==")
        (println "==================================================")
        (println (format "This Run Evaluations   : %d" (:total-evals run-metrics)))
        (println (format "This Run Passed        : %d" (:passed-evals run-metrics)))
        (println (format "This Run Pass Rate     : %5.1f%%" (* 100.0 (:mean-hidden-pass-rate run-metrics))))
        (println "--------------------------------------------------")
        (println (format "Cumulative Evaluations : %d" (:total-evals metrics)))
        (println (format "Cumulative Passed      : %d" (:passed-evals metrics)))
        (println (format "Cumulative Pass Rate   : %5.1f%%" (* 100.0 (:mean-hidden-pass-rate metrics))))
        (println (format "Cumulative Pass@1      : %5.1f%%" (* 100.0 (:pass-at-1 metrics))))
        (println (format "Median Submissions     : %5.2f" (:median-submissions metrics)))
        (println "--------------------------------------------------")
        (println (format "Single-Shot Pass Rate  : %5.1f%% (%d/%d)"
                         (* 100.0 (get-in metrics [:single-shot :pass-rate] 0.0))
                         (get-in metrics [:single-shot :passed] 0)
                         (get-in metrics [:single-shot :count] 0)))
        (println (format "Agentic Loop Pass Rate : %5.1f%% (%d/%d)"
                         (* 100.0 (get-in metrics [:agentic :pass-rate] 0.0))
                         (get-in metrics [:agentic :passed] 0)
                         (get-in metrics [:agentic :count] 0)))
        (let [delta (- (get-in metrics [:agentic :pass-rate] 0.0)
                       (get-in metrics [:single-shot :pass-rate] 0.0))]
          (println (format "Delta Added by Loop    : %+5.1f%%" (* 100.0 delta))))
        (println "==================================================")
        (println (format "Saved results row(s) to [%s]" (.getPath results-file)))
        (println (format "Saved cumulative summary to [%s]" (.getPath summary-file))))
      metrics)))

;; =============================================================================
;; 5. CLI Entrypoint
;; =============================================================================

(defn parse-bench-cli-args
  "Parses CLI flags for clj_bench."
  [args]
  (let [opts (cli/parse-cli-args args DEFAULT-BENCH-OPTS)]
    (cond-> opts
      (string? (:max-turns opts)) (update :max-turns #(Long/parseLong %))
      (string? (:max-consecutive-errors opts)) (update :max-consecutive-errors #(Long/parseLong %))
      (string? (:max-new-tokens opts)) (update :max-new-tokens #(Long/parseLong %))
      (string? (:max-seq-len opts)) (update :max-seq-len #(Long/parseLong %))
      (string? (:temperature opts)) (update :temperature #(Double/parseDouble %))
      (string? (:top-k opts)) (update :top-k #(Long/parseLong %))
      (string? (:repetition-penalty opts)) (update :repetition-penalty #(Double/parseDouble %))
      (string? (:dry-run opts)) (update :dry-run #(Boolean/parseBoolean %))
      (string? (:thinking opts)) (update :thinking #(Boolean/parseBoolean %))
      (string? (:overwrite opts)) (update :overwrite #(Boolean/parseBoolean %))
      (string? (:nudge-on-no-tool opts)) (update :nudge-on-no-tool #(Boolean/parseBoolean %))
      (string? (:backend opts)) (update :backend #(keyword (str/replace % #"^:+" "")))
      (string? (:mode opts)) (update :mode #(keyword (str/replace % #"^:+" ""))))))

(defn -main
  "CLI entrypoint for clojure_bench."
  [& args]
  (try
    (let [opts (parse-bench-cli-args args)]
      (run-benchmark opts)
      (System/exit 0))
    (catch Throwable e
      (println "\nclojure_bench Exception:" (.getMessage e))
      (.printStackTrace e)
      (System/exit 1))))
