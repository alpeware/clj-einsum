(ns experiments.gate3-evals.clojure-bench.run
  "Stage 2 Implementation & Benchmark Runner for clojure_bench (Gate 3 Evals).
   Evaluates raw s-expression Clojure writing ability (single-shot) and loop ability (agentic)
   on Gemma 4 family models under OpenXLA PJRT execution and hermetic SCI grading."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.agent.core :as agent]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.runtime :as gemma4-rt]
            [einsum.runtime.arena :as arena]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [experiments.gate3-evals.prompt-tuning-v1.core :as pt-core]))

;; =============================================================================
;; 1. Default Evaluation Configuration
;; =============================================================================

(def DEFAULT-BENCH-OPTS
  {:model ".models/gemma-4-E2B-it-int4"
   :backend :rocm
   :mode :all ;; :single-shot, :agentic, or :all
   :tool-syntax :native
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
   :semantic-stop false
   :early-exit false
   :nudge-short-circuit false
   :overwrite false
   :include-quarantine false
   :dry-run false
   :quiet false
   :save-transcripts false
   :public-tasks-file "resources/catalog/gate3_evals/clojure_bench/tasks_public.edn"
   :sealed-tasks-file "resources/catalog/gate3_evals/clojure_bench/tasks_sealed.edn"
   :results-file "resources/catalog/gate3_evals/clojure_bench/results.edn"
   :summary-file "resources/catalog/gate3_evals/clojure_bench/summary.csv"})

;; Mock reference solutions used for dry-run verification
(def MOCK-REFERENCE-SOLUTIONS
  (let [base {"first-n" "(defn first-n [coll] (vec (take 10 coll)))"
              "my-range" "(defn my-range ([end] (my-range 0 end)) ([start end] (loop [i start acc []] (if (>= i end) acc (recur (inc i) (conj acc i))))))"
              "deep-flatten" "(defn deep-flatten [coll] (reduce (fn [acc x] (if (sequential? x) (into acc (deep-flatten x)) (conj acc x))) [] coll))"
              "freqs" "(defn freqs [coll] (reduce (fn [m x] (update m x (fnil inc 0))) {} coll))"
              "partition-by-parity" "(defn partition-by-parity [coll] (reduce (fn [m x] (if (even? x) (update m :even conj x) (update m :odd conj x))) {:even [] :odd []} coll))"
              "my-comp" "(defn my-comp [& fns] (if (empty? fns) identity (let [rfns (reverse fns) r (first rfns) rest-fns (rest rfns)] (fn [& args] (reduce (fn [acc f] (f acc)) (apply r args) rest-fns)))))"
              "balanced-delims?" "(defn balanced-delims? [s] (let [matches {\\) \\( \\] \\[ \\} \\{} opens #{\\( \\[ \\{}] (loop [chars (seq s) stack ()] (if-let [c (first chars)] (cond (opens c) (recur (rest chars) (conj stack c)) (matches c) (if (= (first stack) (matches c)) (recur (rest chars) (pop stack)) false) :else (recur (rest chars) stack)) (empty? stack)))))"
              "deep-update-vals" "(defn deep-update-vals [f x] (cond (map? x) (reduce-kv (fn [m k v] (assoc m k (deep-update-vals f v))) {} x) (vector? x) (mapv #(deep-update-vals f %) x) :else (f x)))"
              "lazy-interleave" "(defn lazy-interleave [c1 c2] (lazy-seq (let [s1 (seq c1) s2 (seq c2)] (cond (and s1 s2) (cons (first s1) (cons (first s2) (lazy-interleave (rest s1) (rest s2)))) s1 s1 s2 s2 :else nil))))"
              "my-or" "(defmacro my-or ([] nil) ([x] x) ([x & next] (list (quote let) [(quote or#) x] (list (quote if) (quote or#) (quote or#) (cons (quote my-or) next)))))"}
        extra (try
                (if-let [res (or (io/resource "catalog/gate3_evals/multipl_e/solutions.edn")
                                 (let [f (io/file "resources/catalog/gate3_evals/multipl_e/solutions.edn")]
                                   (when (.exists f) f)))]
                  (edn/read-string (slurp res))
                  {})
                (catch Throwable _ {}))]
    (merge base extra)))

;; =============================================================================
;; 2. Session Initialization & Weight Hash
;; =============================================================================

(def ^:private checkpoint-hash-cache (atom {}))

(defn- compute-checkpoint-hash
  "Computes SHA-256 for the model weights file (model.safetensors) with caching."
  [model-dir]
  (let [safetensors (io/file model-dir "model.safetensors")
        sha-file (io/file model-dir "model.safetensors.sha256")]
    (cond
      (and (.exists sha-file) (.exists safetensors) (>= (.lastModified sha-file) (.lastModified safetensors)))
      (str/trim (slurp sha-file))

      (.exists safetensors)
      (let [cache-key [(.getAbsolutePath safetensors) (.length safetensors) (.lastModified safetensors)]]
        (if-let [cached (get @checkpoint-hash-cache cache-key)]
          cached
          (let [sha (bench-core/compute-file-sha256 safetensors)]
            (swap! checkpoint-hash-cache assoc cache-key sha)
            (try (spit sha-file sha) (catch Throwable _ nil))
            sha)))

      :else
      "unknown-safetensors-sha256")))

(defn init-benchmark-session
  "Initializes pinned persistent VRAM session for Gemma 4 benchmark evaluation."
  [opts]
  (if (:dry-run opts)
    {:opts opts
     :model-dir (:model opts)
     :checkpoint-sha "dry-run-checkpoint-sha"}
    (let [model-dir (cfg/find-model-dir (or (:model-dir opts) (:model opts)))
          max-seq-len (long (or (:max-seq-len opts) 4608))
          checkpoint-sha (compute-checkpoint-hash model-dir)
          session-opts (assoc opts
                              :model-dir model-dir
                              :model model-dir
                              :mode :agent
                              :max-seq-len max-seq-len
                              :max-new-tokens (long (or (:max-new-tokens opts) 1536))
                              :sandbox :benchmark)
          session (gemma4-rt/init-agent-vram-session session-opts max-seq-len)]
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
        prompt (bench-core/render-benchmark-prompt task (or (:prompt-template opts) :clojure-bench-v1) :single-shot)
        model-name (.getName (io/file (or (:model-dir session) (:model opts))))
        checkpoint-sha (:checkpoint-sha session)
        _ (when-not quiet (println (format "\n[Single-Shot] Task: %s (%s)..." task-id fn-name)))
        t0 (System/nanoTime)]

    (if dry-run
      ;; Dry-run mode: use verified mock code
      (let [mock-code (get MOCK-REFERENCE-SOLUTIONS task-id (get MOCK-REFERENCE-SOLUTIONS (str fn-name) "(defn stub [x] x)"))
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
          :prompt-sha (:prompt-sha opts)
          :temperature (double (or (:temperature opts) 0.0))
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :max-new-tokens (long (or (:max-new-tokens opts) 1536))
          :prompt-variant (:prompt-variant opts)
          :dry-run? true}))

      ;; Actual model inference
      (let [max-new (long (or (:single-shot-max-new-tokens opts)
                              (if (or (= (:mode opts) :all) (= (:mode opts) :single-shot))
                                4096
                                (:max-new-tokens opts))
                              4096))
            stop-pred (when (:semantic-stop opts)
                        (fn [text]
                          (agent/semantic-stop? text {:target-fn fn-name})))
            single-shot-session (assoc (update session :opts assoc
                                               :stop-predicate stop-pred
                                               :max-new-tokens max-new)
                                       :kv-state nil)
            chat-prompt (agent/format-agent-chat-prompt
                         bench-core/SINGLE-SHOT-SYSTEM-PROMPT
                         [{:role :user :content prompt}]
                         8
                         (boolean (:thinking opts))
                         nil)
            gen-res (gemma4-rt/generate-new-tokens-and-text single-shot-session chat-prompt)
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
          :prompt-sha (:prompt-sha opts)
          :temperature (double (or (:temperature opts) 0.0))
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :max-new-tokens max-new
          :prompt-variant (:prompt-variant opts)
          :dry-run? false})))))

(defn run-agentic-task
  "Runs agentic evaluation with eval_clojure tool and submission hook:
   Tool calls that define the target fn execute public tests and receive public feedback.
   The best-public submission is graded against sealed hidden tests exactly once."
  [session task hidden-tests sealed-sha opts]
  (let [{:keys [dry-run quiet max-turns]} opts
        tool-syntax (keyword (or (:tool-syntax opts) :native))
        task-id (:id task)
        fn-name (:fn-name task)
        prompt (bench-core/render-benchmark-prompt task (or (:prompt-template opts) :clojure-bench-v1) :agentic)
        model-name (.getName (io/file (or (:model-dir session) (:model opts))))
        checkpoint-sha (:checkpoint-sha session)
        submissions (atom [])
        _ (when-not quiet (println (format "\n[Agentic Loop] Task: %s (%s, max %d turns, syntax %s)..." task-id fn-name (long (or max-turns 5)) (name tool-syntax))))
        t0 (System/nanoTime)]

    (if dry-run
      ;; Dry run mode
      (let [mock-code (get MOCK-REFERENCE-SOLUTIONS task-id (get MOCK-REFERENCE-SOLUTIONS (str fn-name) "(defn stub [x] x)"))
            pub-res (bench-core/grade-submission mock-code (:public-tests task))
            _ (swap! submissions conj {:code mock-code :public-res pub-res :turn 1})
            best-sub (bench-core/pick-best-public-submission @submissions)
            hidden-res (bench-core/grade-submission (:code best-sub) hidden-tests)
            wall-ms (/ (- (System/nanoTime) t0) 1e6)]
        (bench-core/format-results-row
         {:model model-name
          :task task-id
          :mode :agentic
          :tool-syntax tool-syntax
          :candidate-code (:code best-sub)
          :grade-res hidden-res
          :n-submissions (count @submissions)
          :tokens-in 240
          :tokens-out 90
          :wall-ms wall-ms
          :sealed-sha sealed-sha
          :checkpoint-sha checkpoint-sha
          :prompt-sha (:prompt-sha opts)
          :temperature (double (or (:temperature opts) 0.0))
          :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
          :max-new-tokens (long (or (:max-new-tokens opts) 1536))
          :prompt-variant (:prompt-variant opts)
          :transcript [{:turn 1 :role :model :content (str "(defn " fn-name " ...)")}]
          :save-transcripts? (boolean (or (:save-transcripts opts) (:save-transcripts? opts)))
          :dry-run? true}))

      ;; Actual agentic loop execution
      (let [task-kv-state (atom nil)
            early-exit-opt? (get opts :early-exit false)
            nudge-sc-opt? (get opts :nudge-short-circuit false)
            submission-tool-hook
            (fn submission-tool-hook
              ([sci-ctx tool-code]
               (submission-tool-hook sci-ctx tool-code (inc (count @submissions))))
              ([sci-ctx tool-code turn]
               (let [eval-res (agent/eval-tool-code sci-ctx tool-code)
                     candidate (bench-core/extract-candidate-code tool-code fn-name)
                     is-sub? (and (seq candidate) (bench-core/submission-form? candidate fn-name))]
                 (if is-sub?
                   (let [pub-res (bench-core/grade-submission sci-ctx candidate (:public-tests task))
                         early-exit? (boolean (and early-exit-opt? (:all-passed? pub-res)))
                         feedback (bench-core/format-public-feedback pub-res)
                         failures (when-not (:all-passed? pub-res)
                                    (if (:error pub-res)
                                      (:error pub-res)
                                      (str/join "; "
                                                (map (fn [{:keys [code expected actual error]}]
                                                       (if error
                                                         (format "%s -> %s" code error)
                                                         (format "%s -> Expected %s, got %s" code expected actual)))
                                                     (filter #(not (:passed? %)) (:results pub-res))))))
                         nudge-msg (when-not (:all-passed? pub-res)
                                     "Revise your implementation to pass all public tests.")]
                     (swap! submissions conj {:code candidate :public-res pub-res :turn turn})
                     (cond-> (assoc eval-res
                                    :status (if (:all-passed? pub-res) :success (if (:error pub-res) :error :failed))
                                    :tests_passed (:passed-count pub-res)
                                    :tests_total (:total-count pub-res)
                                    :output (if (:all-passed? pub-res)
                                              (if (seq (:output eval-res))
                                                (str (:output eval-res) "\n" feedback)
                                                feedback)
                                              feedback))
                       (seq failures) (assoc :failures failures)
                       (seq nudge-msg) (assoc :nudge nudge-msg)
                       early-exit? (assoc :early-exit? true)))
                   eval-res))))

            candidate-check-fn (when nudge-sc-opt?
                                 (fn [text]
                                   (when-not (agent/inside-unclosed-thought? text)
                                     (let [clean (agent/strip-thinking-trace (or text ""))
                                           candidate (bench-core/extract-candidate-code clean fn-name)]
                                       (and (seq candidate)
                                            (bench-core/submission-form? candidate fn-name)
                                            (some? (agent/try-parse-sci-reader candidate)))))))

            stop-pred (when (:semantic-stop opts)
                        (fn [text]
                          (agent/semantic-stop? text {:target-fn fn-name :tool-syntax tool-syntax})))

            task-system (or (when-not (= (:system opts) bench-core/AGENT-SYSTEM-PROMPT)
                              (:system opts))
                            (case tool-syntax
                              :xml bench-core/AGENT-SYSTEM-PROMPT-XML
                              :fenced bench-core/AGENT-SYSTEM-PROMPT-FENCED
                              bench-core/AGENT-SYSTEM-PROMPT))

            task-tool-decl (if (or (= tool-syntax :xml) (= tool-syntax :fenced))
                             nil
                             (get opts :tool-declaration agent/DEFAULT-TOOL-DECLARATION))

            task-session (assoc (update session :opts assoc
                                        :system task-system
                                        :tool-declaration task-tool-decl
                                        :tool-syntax tool-syntax
                                        :sandbox :benchmark
                                        :tool-eval-fn submission-tool-hook
                                        :candidate-check-fn candidate-check-fn
                                        :stop-predicate stop-pred
                                        :max-new-tokens (long (or (:max-new-tokens opts) 1536))
                                        :quiet quiet)
                                :kv-state task-kv-state)]
        (try
          (let [transcript (agent/run-agent-loop task-session prompt (bench-core/create-tightened-grading-ctx) submission-tool-hook)
                telemetry (get (meta transcript) :turn-telemetry [])
                total-in (reduce + 0 (map #(long (or (:prompt-tokens %) 0)) telemetry))
                total-out (reduce + 0 (map #(long (or (:new-tokens %) 0)) telemetry))
                total-turn-ms (reduce + 0.0 (map #(double (or (:total-turn-ms %) 0.0)) telemetry))
                wall-ms (if (pos? total-turn-ms) total-turn-ms (/ (- (System/nanoTime) t0) 1e6))

                ;; Fallback / ratchet: extract and grade candidates from model turns in transcript if not already captured
                _ (doseq [t transcript
                          :when (= (:role t) :model)]
                    (let [text (or (:response t) (:raw t) (:content t) "")
                          candidate (bench-core/extract-candidate-code text fn-name)]
                      (when (and (seq candidate)
                                 (bench-core/submission-form? candidate fn-name)
                                 (not (some #(= (:code %) candidate) @submissions)))
                        (let [pub-res (bench-core/grade-submission candidate (:public-tests task))]
                          (swap! submissions conj {:code candidate :public-res pub-res :turn (or (:turn t) 1)})))))

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
              :tool-syntax tool-syntax
              :candidate-code candidate-code
              :grade-res hidden-res
              :error error-msg
              :n-submissions (count @submissions)
              :tokens-in total-in
              :tokens-out total-out
              :wall-ms wall-ms
              :sealed-sha sealed-sha
              :checkpoint-sha checkpoint-sha
              :prompt-sha (:prompt-sha opts)
              :transcript (vec transcript)
              :temperature (double (or (:temperature opts) 0.0))
              :repetition-penalty (double (or (:repetition-penalty opts) 1.0))
              :max-new-tokens (long (or (:max-new-tokens opts) 1536))
              :prompt-variant (:prompt-variant opts)
              :save-transcripts? (boolean (or (:save-transcripts opts) (:save-transcripts? opts)))
              :dry-run? false}))
          (finally
            (when-let [st @task-kv-state]
              (when-let [bufs (seq (:kv-buffers st))]
                (try (arena/destroy! (:session-arena session) bufs) (catch Throwable _ nil)))
              (reset! task-kv-state nil))))))))

;; =============================================================================
;; 4. Benchmark Orchestration & Reporting
;; =============================================================================

(defn- load-quarantine-ids
  "Loads set of quarantined task IDs from catalog if present."
  []
  (let [qf (io/file "resources/catalog/gate3_evals/multipl_e/quarantine.edn")]
    (if (.exists qf)
      (set (keep :id (try (edn/read-string (slurp qf)) (catch Throwable _ nil))))
      #{})))

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
  (let [opts (merge DEFAULT-BENCH-OPTS opts)
        dry-run? (boolean (:dry-run opts))
        results-path (if (and dry-run? (= (:results-file opts) (:results-file DEFAULT-BENCH-OPTS)))
                       "resources/catalog/gate3_evals/clojure_bench/results_dry_run.edn"
                       (:results-file opts))
        summary-path (if (and dry-run? (= (:summary-file opts) (:summary-file DEFAULT-BENCH-OPTS)))
                       "resources/catalog/gate3_evals/clojure_bench/summary_dry_run.csv"
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
        prompt-sha (if-let [pv (:prompt-variant opts)]
                     (pt-core/compute-prompt-sha pv)
                     (bench-core/compute-file-sha256 public-file))
        opts (assoc opts :prompt-sha prompt-sha)
        {:keys [harness-sha harness-dirty?]} (bench-core/harness-version-info)
        _ (when (and (not dry-run?) harness-dirty?)
            (println (str "WARNING: git tree is dirty; rows will be stamped :harness-dirty? true "
                          "under sha " harness-sha ". Record runs should use a clean tree.")))
        public-tasks (edn/read-string (slurp public-file))
        sealed-map (into {} (map (juxt :id :hidden-tests) (edn/read-string (slurp sealed-file))))
        raw-selected (filter-tasks public-tasks (:tasks opts))
        quarantine-ids (load-quarantine-ids)
        include-quarantine? (boolean (:include-quarantine opts))
        filtered-tasks (if include-quarantine?
                         raw-selected
                         (filterv #(not (contains? quarantine-ids (:id %))) raw-selected))
        stratified? (get opts :stratified (or (= (:tasks opts) "multipl-e-dev")
                                              (= (:tasks opts) "all")))
        selected-tasks (cond
                         (and (:limit opts) stratified?)
                         (bench-core/stratify-tasks filtered-tasks (:limit opts))
                         (:limit opts)
                         (vec (take (:limit opts) filtered-tasks))
                         :else
                         filtered-tasks)
        n-quarantined (- (count raw-selected) (count filtered-tasks))
        run-mode (keyword (:mode opts))

        _ (when-not (:quiet opts)
            (println "==================================================")
            (println "=== clojure_bench: Gemma 4 Clojure Eval Suite ===")
            (println "==================================================")
            (println (format "Model Checkpoint     : %s" (:model opts)))
            (println (format "Backend              : %s" (:backend opts)))
            (println (format "Prompt SHA-256       : %s" prompt-sha))
            (println (format "Sealed SHA-256       : %s" sealed-sha))
            (println (format "Evaluation Mode      : %s" run-mode))
            (println (format "Selected Tasks       : %d/%d%s"
                             (count selected-tasks)
                             (count public-tasks)
                             (if (:limit opts)
                               (str " (limit: " (:limit opts) (when stratified? ", stratified") ")")
                               "")))
            (when (pos? n-quarantined)
              (println (format "Quarantine Filter    : Excluded %d quarantined tasks (use --include-quarantine true to evaluate)" n-quarantined)))
            (println (format "Dry Run Mode         : %s" dry-run?))
            (println (format "Results File         : %s" (.getPath results-file)))
            (println (format "Summary File         : %s" (.getPath summary-file)))
            (println "=================================================="))

        _ (do
            (io/make-parents results-file)
            (io/make-parents summary-file))
        _ (when (:overwrite opts)
            (spit results-file ""))

        user-flags (or (:user-flags opts) #{})
        all-results (atom [])]

    ;; 1. Single-shot evaluation phase
    (when (or (= run-mode :all) (= run-mode :single-shot))
      (let [ss-max-seq-len (or (:single-shot-max-seq-len opts)
                               (when (contains? user-flags "max-seq-len") (:max-seq-len opts))
                               4608)
            ss-max-new (or (:single-shot-max-new-tokens opts)
                           (when (contains? user-flags "max-new-tokens") (:max-new-tokens opts))
                           4096)
            ss-opts (assoc opts :max-seq-len ss-max-seq-len :max-new-tokens ss-max-new)
            ss-session (init-benchmark-session ss-opts)]
        (try
          (doseq [task selected-tasks]
            (let [task-id (:id task)
                  hidden-tests (or (get sealed-map task-id) (:hidden-tests task) [])
                  ss-row (run-single-shot-task ss-session task hidden-tests sealed-sha ss-opts)]
              (swap! all-results conj ss-row)
              (spit results-file (str (pr-str ss-row) "\n") :append true)))
          (finally
            (when (and (not dry-run?) (map? ss-session))
              (try (gemma4-rt/close-agent-session! ss-session) (catch Throwable _ nil)))))))

    ;; 2. Agentic evaluation phase
    (when (or (= run-mode :all) (= run-mode :agentic))
      (let [ag-max-seq-len (or (:agentic-max-seq-len opts)
                               (when (contains? user-flags "max-seq-len") (:max-seq-len opts))
                               2048)
            ag-max-new (or (:agentic-max-new-tokens opts)
                           (when (contains? user-flags "max-new-tokens") (:max-new-tokens opts))
                           1536)
            ag-opts (assoc opts :max-seq-len ag-max-seq-len :max-new-tokens ag-max-new)
            ag-session (init-benchmark-session ag-opts)]
        (try
          (doseq [task selected-tasks]
            (let [task-id (:id task)
                  hidden-tests (or (get sealed-map task-id) (:hidden-tests task) [])
                  ag-row (run-agentic-task ag-session task hidden-tests sealed-sha ag-opts)]
              (swap! all-results conj ag-row)
              (spit results-file (str (pr-str ag-row) "\n") :append true)))
          (finally
            (when (and (not dry-run?) (map? ag-session))
              (try (gemma4-rt/close-agent-session! ag-session) (catch Throwable _ nil)))))))

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

(defn- parse-raw-cli-args
  [args defaults]
  (loop [rem-args (vec args)
         opts defaults]
    (if (empty? rem-args)
      opts
      (let [arg (first rem-args)]
        (cond
          (str/starts-with? arg "--")
          (let [flag (subs arg 2)
                k (keyword flag)]
            (if (or (= flag "dry-run") (= flag "quiet") (= flag "overwrite")
                    (= flag "thinking") (= flag "nudge-on-no-tool")
                    (= flag "include-quarantine"))
              (if (and (> (count rem-args) 1) (not (str/starts-with? (second rem-args) "--")))
                (recur (subvec rem-args 2) (assoc opts k (Boolean/parseBoolean (second rem-args))))
                (recur (subvec rem-args 1) (assoc opts k true)))
              (if (> (count rem-args) 1)
                (let [val (second rem-args)
                      parsed (cond
                               (or (= flag "model") (= flag "model-dir"))
                               (let [path (if (and (string? val) (not (str/starts-with? val ".")) (not (str/starts-with? val "/")))
                                            (if (.exists (io/file val)) val (str ".models/" (last (str/split val #"/"))))
                                            val)]
                                 (cond
                                   (.exists (io/file path)) path
                                   (.exists (io/file (str/lower-case path))) (str/lower-case path)
                                   :else path))
                               :else val)]
                  (recur (subvec rem-args 2) (assoc opts k parsed)))
                (recur (subvec rem-args 1) opts))))
          :else
          (recur (subvec rem-args 1) opts))))))

(defn parse-bench-cli-args
  "Parses CLI flags for clj_bench."
  [args]
  (let [user-flags (set (keep #(when (str/starts-with? % "--") (subs % 2)) args))
        raw-opts (parse-raw-cli-args args DEFAULT-BENCH-OPTS)
        raw-variant (or (:prompt-variant raw-opts)
                        (when (contains? #{"p0" ":p0" "p1" ":p1" "p2" ":p2"} (str (:prompt-template raw-opts)))
                          (keyword (str/replace (str (:prompt-template raw-opts)) #"^:+" ""))))
        variant-sys (when raw-variant (pt-core/get-system-prompt (keyword raw-variant)))
        raw-opts (cond-> raw-opts
                   raw-variant (assoc :prompt-variant (keyword raw-variant))
                   variant-sys (assoc :system variant-sys))
        normalized-tasks-opts
        (cond
          (= (:tasks raw-opts) "multipl-e-dev")
          (assoc raw-opts
                 :public-tasks-file "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"
                 :sealed-tasks-file "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"
                 :prompt-template (if (contains? #{:p0 :p1 :p2} (:prompt-template raw-opts))
                                    :multipl-e-v0
                                    (or (:prompt-template raw-opts) :multipl-e-v0))
                 :tool-syntax (if (contains? user-flags "tool-syntax")
                                (keyword (str/replace (str (:tool-syntax raw-opts)) #"^:+" ""))
                                :fenced)
                 :tasks "all")

          (= (:tasks raw-opts) "multipl-e-sealed")
          (assoc raw-opts
                 :public-tasks-file "resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn"
                 :sealed-tasks-file "resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn"
                 :prompt-template (if (contains? #{:p0 :p1 :p2} (:prompt-template raw-opts))
                                    :multipl-e-v0
                                    (or (:prompt-template raw-opts) :multipl-e-v0))
                 :tool-syntax (if (contains? user-flags "tool-syntax")
                                (keyword (str/replace (str (:tool-syntax raw-opts)) #"^:+" ""))
                                :fenced)
                 :tasks "all")

          (or (= (:tasks raw-opts) "dev-50") (= (:tasks raw-opts) "multipl-e-dev-50"))
          (assoc raw-opts
                 :public-tasks-file "resources/catalog/gate3_evals/multipl_e/dev_50_public.edn"
                 :sealed-tasks-file "resources/catalog/gate3_evals/multipl_e/dev_50_sealed.edn"
                 :tasks "all")

          :else
          raw-opts)
        opts (cond-> normalized-tasks-opts
               (string? (:limit normalized-tasks-opts)) (update :limit #(Long/parseLong %))
               (string? (:stratified normalized-tasks-opts)) (update :stratified #(Boolean/parseBoolean %))
               (string? (:prompt-variant normalized-tasks-opts)) (update :prompt-variant #(keyword (str/replace % #"^:+" "")))
               (string? (:max-turns normalized-tasks-opts)) (update :max-turns #(Long/parseLong %))
               (string? (:max-consecutive-errors normalized-tasks-opts)) (update :max-consecutive-errors #(Long/parseLong %))
               (string? (:max-new-tokens normalized-tasks-opts)) (update :max-new-tokens #(Long/parseLong %))
               (string? (:max-seq-len normalized-tasks-opts)) (update :max-seq-len #(Long/parseLong %))
               (string? (:temperature normalized-tasks-opts)) (update :temperature #(Double/parseDouble %))
               (string? (:top-k normalized-tasks-opts)) (update :top-k #(Long/parseLong %))
               (string? (:repetition-penalty normalized-tasks-opts)) (update :repetition-penalty #(Double/parseDouble %))
               (string? (:dry-run normalized-tasks-opts)) (update :dry-run #(Boolean/parseBoolean %))
               (string? (:include-quarantine normalized-tasks-opts)) (update :include-quarantine #(Boolean/parseBoolean %))
               (string? (:thinking normalized-tasks-opts)) (update :thinking #(Boolean/parseBoolean %))
               (string? (:overwrite normalized-tasks-opts)) (update :overwrite #(Boolean/parseBoolean %))
               (string? (:save-transcripts normalized-tasks-opts)) (update :save-transcripts #(Boolean/parseBoolean %))
               (string? (:nudge-on-no-tool normalized-tasks-opts)) (update :nudge-on-no-tool #(Boolean/parseBoolean %))
               (string? (:semantic-stop normalized-tasks-opts)) (update :semantic-stop #(Boolean/parseBoolean %))
               (string? (:early-exit normalized-tasks-opts)) (update :early-exit #(Boolean/parseBoolean %))
               (string? (:nudge-short-circuit normalized-tasks-opts)) (update :nudge-short-circuit #(Boolean/parseBoolean %))
               (string? (:backend normalized-tasks-opts)) (update :backend #(keyword (str/replace % #"^:+" "")))
               (string? (:tool-syntax normalized-tasks-opts)) (update :tool-syntax #(keyword (str/replace % #"^:+" "")))
               (string? (:mode normalized-tasks-opts)) (update :mode #(keyword (str/replace % #"^:+" "")))
               (string? (:prompt-template normalized-tasks-opts)) (update :prompt-template #(keyword (str/replace % #"^:+" ""))))
        opts (assoc opts :user-flags user-flags)]
    opts))

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
