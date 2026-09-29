(ns experiments.gate3-evals.prompt-tuning-v1.run
  "Stage 2 Implementation: Stratified Pilot Runner & Candidate Downselection
   for Prompt Tuning v1: Worked Agentic Trajectories (Gate 3 Evals).
   Evaluates candidate prompts (P0, P1, P2) on a 50-task stratified MultiPL-E dev pilot,
   audits all submissions for canary echoing, and applies the pre-registered m=1 downselection rule."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.models.gemma4.runtime :as gemma4-rt]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [experiments.gate3-evals.clojure-bench.run :as bench-run]
            [experiments.gate3-evals.prompt-tuning-v1.core :as pt-core]))

;; =============================================================================
;; 1. Configuration & Default Options
;; =============================================================================

(def DEFAULT-PILOT-OPTS
  {:model ".models/gemma-4-e4b-it-qat-int4"
   :backend :rocm
   :limit 50
   :variants [:p0 :p1 :p2]
   :tool-syntax :fenced
   :max-turns 5
   :max-consecutive-errors 3
   :max-new-tokens 1536
   :max-seq-len 2048
   :temperature 0.0
   :dry-run false
   :quiet false
   :stratified true
   :tasks-file "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"
   :quarantine-file "resources/catalog/gate3_evals/multipl_e/quarantine.edn"
   :output-file "resources/proposals/gate3_evals/prompt_tuning_v1/pilot_results.edn"})

;; =============================================================================
;; 2. Downselection & Decision Logic
;; =============================================================================

(defn downselect-candidate
  "Applies Phase 2 downselection rule across variant results:
   - Fatal canary leak -> KILLED.
   - Early-stop if neither P1 nor P2 beats P0 by >= 2 tasks on pilot -> REJECT.
   - Else, select winning candidate P* among {P1, P2} with highest solve rate;
     ties broken by smaller token budget (P1 < P2).
   Returns {:verdict :PROCEED-TO-PHASE-3/:REJECT/:KILLED
            :selected-candidate :p1/:p2/nil
            :early-stop? boolean
            :deltas {:p1 d1 :p2 d2}
            :rationale string}."
  [variant-stats]
  (let [canary-leaks (vec (mapcat :canary-leaks (vals variant-stats)))
        k0 (long (get-in variant-stats [:p0 :passed-count] 0))
        k1 (long (get-in variant-stats [:p1 :passed-count] 0))
        k2 (long (get-in variant-stats [:p2 :passed-count] 0))
        d1 (- k1 k0)
        d2 (- k2 k0)
        deltas {:p1 d1 :p2 d2}]
    (cond
      ;; 1. KILLED triggers dominate
      (seq canary-leaks)
      {:verdict :KILLED
       :selected-candidate nil
       :early-stop? true
       :deltas deltas
       :rationale (str "Fatal canary echoing leak detected in model submissions: " (pr-str canary-leaks))}

      ;; 2. Early-stop: neither P1 nor P2 beats P0 by >= 2 tasks
      (and (< d1 2) (< d2 2))
      {:verdict :REJECT
       :selected-candidate nil
       :early-stop? true
       :deltas deltas
       :rationale (format "Phase 2 early stop: neither P1 (Δ%+d) nor P2 (Δ%+d) beat P0 by >= 2 tasks on pilot (threshold: >= 2)" d1 d2)}

      ;; 3. Candidate selection: max solve rate, tie broken by smaller token budget (P1 wins over P2)
      :else
      (let [candidate (cond
                        (> k1 k2) :p1
                        (> k2 k1) :p2
                        :else :p1)] ;; Tie-break: P1 (356 tok) < P2 (591 tok)
        {:verdict :PROCEED-TO-PHASE-3
         :selected-candidate candidate
         :early-stop? false
         :deltas deltas
         :rationale (format "Selected %s as winning candidate P* (P0: %d, P1: %d [Δ%+d], P2: %d [Δ%+d]; %s)"
                            (name candidate) k0 k1 d1 k2 d2
                            (if (= k1 k2) "tie broken by smaller token budget P1" "highest solve rate"))}))))

;; =============================================================================
;; 3. Pilot Execution Pipeline
;; =============================================================================

(defn load-pilot-tasks
  "Loads dev tasks, filters out quarantined IDs, and selects a stratified sample of `limit` tasks."
  [tasks-path quarantine-path limit stratified?]
  (let [tasks (edn/read-string (slurp tasks-path))
        q-ids (if (.exists (io/file quarantine-path))
                (set (keep :id (edn/read-string (slurp quarantine-path))))
                #{})
        filtered (filterv #(not (contains? q-ids (:id %))) tasks)]
    (if (and limit stratified?)
      (bench-core/stratify-tasks filtered limit)
      (if limit (vec (take limit filtered)) filtered))))

(defn run-pilot
  "Executes the Phase 2 stratified pilot across candidate variants (:p0, :p1, :p2).
   Audits all generated candidate code for canary echoing leaks.
   Applies single-candidate downselection and saves detailed pilot results."
  [opts]
  (let [opts (merge DEFAULT-PILOT-OPTS opts)
        model-path (cond
                     (.exists (io/file (:model opts))) (:model opts)
                     (.exists (io/file (str/lower-case (:model opts)))) (str/lower-case (:model opts))
                     :else (:model opts))
        opts (assoc opts :model model-path)
        dry-run? (boolean (:dry-run opts))
        quiet? (boolean (:quiet opts))
        tasks-file (io/file (:tasks-file opts))
        q-file (io/file (:quarantine-file opts))
        output-file (io/file (:output-file opts))
        limit (long (or (:limit opts) 50))
        stratified? (boolean (get opts :stratified true))
        variants (vec (or (:variants opts) [:p0 :p1 :p2]))

        _ (when-not (.exists tasks-file)
            (throw (IllegalArgumentException. (str "Tasks file not found: " tasks-file))))

        pilot-tasks (load-pilot-tasks tasks-file q-file limit stratified?)
        strat-counts (frequencies (map #(first (str/split (:id %) #"-")) pilot-tasks))
        sealed-sha (bench-core/compute-file-sha256 tasks-file)
        opts (assoc opts :sealed-sha sealed-sha :prompt-sha sealed-sha)

        _ (when-not quiet?
            (println "==================================================")
            (println "=== Prompt Tuning v1: Phase 2 Stratified Pilot ===")
            (println "==================================================")
            (println (format "Model Checkpoint     : %s" (:model opts)))
            (println (format "Backend              : %s" (:backend opts)))
            (println (format "Pilot Tasks          : %d tasks %s" (count pilot-tasks) (pr-str strat-counts)))
            (println (format "Variants to Eval     : %s" (str/join ", " (map name variants))))
            (println (format "Dry Run Mode         : %s" dry-run?))
            (println (format "Output File          : %s" (.getPath output-file)))
            (println "=================================================="))

        tool-syntax (keyword (if (and (:user-flags opts) (contains? (:user-flags opts) "tool-syntax"))
                               (:tool-syntax opts)
                               :fenced))
        session-opts (assoc opts
                            :mode :agentic
                            :tool-syntax tool-syntax
                            :tasks "all")
        session (bench-run/init-benchmark-session session-opts)
        variant-results (atom {})]

    (try
      (doseq [v variants]
        (let [sys-prompt (pt-core/get-system-prompt v)
              v-prompt-sha (pt-core/compute-prompt-sha v)
              v-opts (assoc session-opts
                            :prompt-variant v
                            :prompt-sha v-prompt-sha
                            :system sys-prompt
                            :prompt-template :multipl-e-v0
                            :tool-syntax tool-syntax)
              v-rows (atom [])
              v-canary-leaks (atom [])
              _ (when-not quiet?
                  (println (format "\n--- Evaluating Variant: %s (%s, sha: %s) ---"
                                   (name v) (get-in pt-core/PROMPT-VARIANTS [v :name]) v-prompt-sha)))]

          (doseq [task pilot-tasks]
            (let [task-id (:id task)
                  fn-name (:fn-name task)
                  hidden-tests (or (seq (:hidden-tests task)) (:public-tests task) [])
                  row (bench-run/run-agentic-task session task hidden-tests sealed-sha v-opts)
                  candidate (:candidate-code row)]
              (swap! v-rows conj row)
              ;; Audit candidate submission for canary echoing
              (when (seq candidate)
                (let [echo-check (pt-core/detect-canary-echo candidate fn-name)]
                  (when (:leak? echo-check)
                    (let [leak-info {:task task-id :fn-name fn-name :variant v :details echo-check}]
                      (swap! v-canary-leaks conj leak-info)
                      (println (format "  [!] CANARY ECHO DETECTED on %s (%s): %s"
                                       task-id fn-name (str/join "; " (:reasons echo-check))))))))))

          (let [rows @v-rows
                passed-cnt (count (filter :passed? rows))
                total-cnt (count rows)
                pass-rate (if (pos? total-cnt) (/ (double passed-cnt) total-cnt) 0.0)
                tot-wall (reduce + 0.0 (map :wall-ms rows))
                tot-in (reduce + 0 (map :tokens-in rows))
                tot-out (reduce + 0 (map :tokens-out rows))
                stats {:variant v
                       :name (get-in pt-core/PROMPT-VARIANTS [v :name])
                       :prompt-sha v-prompt-sha
                       :passed-count passed-cnt
                       :total-count total-cnt
                       :pass-rate pass-rate
                       :wall-ms tot-wall
                       :tokens-in tot-in
                       :tokens-out tot-out
                       :canary-leaks @v-canary-leaks
                       :rows rows}]
            (swap! variant-results assoc v stats)
            (when-not quiet?
              (println (format ">>> Variant %s Results: %d/%d passed (%.1f%%) in %.1f ms | Canary Leaks: %d"
                               (name v) passed-cnt total-cnt (* 100.0 pass-rate) tot-wall (count @v-canary-leaks)))))))
      (finally
        (when (and (not dry-run?) (map? session))
          (try (gemma4-rt/close-agent-session! session) (catch Throwable _ nil)))))

    (let [stats-map @variant-results
          decision (downselect-candidate stats-map)
          summary-data {:timestamp (str (java.time.Instant/now))
                        :model (:model opts)
                        :backend (:backend opts)
                        :pilot-size (count pilot-tasks)
                        :stratification strat-counts
                        :task-ids (mapv :id pilot-tasks)
                        :variant-stats (into {} (map (fn [[k v]] [k (dissoc v :rows)]) stats-map))
                        :decision decision}]
      (io/make-parents output-file)
      (spit output-file (pr-str summary-data))

      (when-not quiet?
        (println "\n==================================================")
        (println "=== Prompt Tuning v1: Downselection Outcome ===")
        (println "==================================================")
        (println (format "P0 (Zero-Shot Control) : %d/%d passed"
                         (get-in stats-map [:p0 :passed-count] 0)
                         (count pilot-tasks)))
        (println (format "P1 (Single Worked Ex)  : %d/%d passed (Δ%+d)"
                         (get-in stats-map [:p1 :passed-count] 0)
                         (count pilot-tasks)
                         (get-in decision [:deltas :p1] 0)))
        (println (format "P2 (Dual Worked Exs)   : %d/%d passed (Δ%+d)"
                         (get-in stats-map [:p2 :passed-count] 0)
                         (count pilot-tasks)
                         (get-in decision [:deltas :p2] 0)))
        (println "--------------------------------------------------")
        (println (format "Verdict                : %s" (:verdict decision)))
        (println (format "Winning Candidate P*   : %s" (if-let [c (:selected-candidate decision)] (name c) "NONE")))
        (println (format "Early Stop Triggered   : %s" (:early-stop? decision)))
        (println (format "Rationale              : %s" (:rationale decision)))
        (println (format "Saved Pilot Results to : [%s]" (.getPath output-file)))
        (println "=================================================="))

      summary-data)))

;; =============================================================================
;; 4. CLI Entrypoint
;; =============================================================================

(defn -main
  "CLI entrypoint for Prompt Tuning v1 Pilot."
  [& args]
  (try
    (let [parsed-opts (bench-run/parse-bench-cli-args args)]
      (run-pilot parsed-opts)
      (System/exit 0))
    (catch Throwable e
      (println "\nPrompt Tuning Pilot Exception:" (.getMessage e))
      (.printStackTrace e)
      (System/exit 1))))
