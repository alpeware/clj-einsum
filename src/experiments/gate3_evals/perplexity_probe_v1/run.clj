(ns experiments.gate3-evals.perplexity-probe-v1.run
  "Stage 2 Implementation: Perplexity Probe v1 Runner (Gate 3 Evals).
   Scores 31B teacher agentic trajectories under student model (gemma-4-e4b-it-qat-int4)
   teacher forcing to measure perplexity on gap tasks vs reference contrast tasks."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.pprint :as pprint]
            [clojure.string :as str]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]
            [einsum.runtime.tokenizer.protocol :as tok-proto]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [experiments.gate3-evals.perplexity-probe-v1.core :as probe-core])
  (:import [java.io File PushbackReader]))

;; =============================================================================
;; 1. Default Configuration & File Paths
;; =============================================================================

(def DEFAULT-PROBE-OPTS
  {:model ".models/gemma-4-e4b-it-qat-int4"
   :teacher-model "gemma-4-31b-it-qat-int4"
   :backend :rocm
   :trajectories-file "resources/proposals/gate3_evals/perplexity_probe_v1/teacher_trajectories.edn"
   :tasks-file "resources/catalog/gate3_evals/clojure_bench/tasks_public.edn"
   :results-file "resources/proposals/gate3_evals/perplexity_probe_v1/results.edn"
   :summary-csv "resources/proposals/gate3_evals/perplexity_probe_v1/summary.csv"
   :max-seq-len 3072
   :dry-run false
   :quiet false})

(def TEACHER-CHECKPOINT-SHA "f20a02315954b540db819393224e07c3d60c1f07c9052b10a530db1556d2603c")
(def STUDENT-CHECKPOINT-SHA "33ebfa9b85f077df19c0526c9f0e039c044d4e64eef47368730208b478dc05ec")

;; =============================================================================
;; 2. File Ingestion & Data Preparation
;; =============================================================================

(defn load-all-edn-forms
  "Reads all EDN forms from a file until EOF."
  [file-or-path]
  (let [^File f (io/file file-or-path)]
    (if (.exists f)
      (with-open [r (PushbackReader. (io/reader f))]
        (loop [forms []]
          (let [form (edn/read {:eof ::eof} r)]
            (if (= form ::eof)
              forms
              (recur (conj forms form))))))
      (throw (IllegalArgumentException. (str "File not found: " (.getPath f)))))))

(defn load-trajectories-by-task
  "Loads teacher trajectories from EDN file and indexes them by string task id."
  [trajectories-file]
  (let [trajs (load-all-edn-forms trajectories-file)]
    (into {} (map (fn [t] [(str (:task t)) t]) trajs))))

(defn load-tasks-by-id
  "Loads benchmark tasks from public catalog and indexes them by string id."
  [tasks-file]
  (let [tasks (edn/read-string (slurp tasks-file))]
    (into {} (map (fn [t] [(str (:id t)) t]) tasks))))

(defn compute-model-sha
  "Computes sha256 for model weights safetensors file."
  [model-dir]
  (let [st-file (io/file model-dir "model.safetensors")]
    (if (.exists st-file)
      (bench-core/compute-file-sha256 st-file)
      "unknown-safetensors")))

(defn create-mock-tokenizer
  "Creates a pure Clojure mock tokenizer for testing and dry-run execution."
  []
  (reify tok-proto/Tokenizer
    (encode [_ text] (mapv int (str text)))
    (decode [_ _] "")
    (bos-id [_] 2)
    (eos-id [_] 1)))

;; =============================================================================
;; 3. Evaluation Execution Engine
;; =============================================================================

(defn evaluate-single-trajectory
  "Evaluates student model perplexity on a single teacher trajectory under teacher forcing.
   Constructs evaluation mask, executes forward scoring pass, and calculates segment metrics."
  [session t-def traj tokenizer opts]
  (let [task-id (str (:id t-def))
        group (if (contains? probe-core/GAP-TASKS-SET task-id) :gap :ref)
        eval-mask (probe-core/build-evaluation-mask t-def (:transcript traj) tokenizer opts)
        token-ids (:token-ids eval-mask)
        num-total (:num-total eval-mask)
        num-scored (:num-scored eval-mask)
        num-think (:num-think eval-mask)
        num-code (:num-code eval-mask)
        dry-run? (:dry-run opts)
        t0 (System/nanoTime)
        log-probs (if dry-run?
                    (into [0.0] (repeat (dec (count token-ids)) -0.693147))
                    (let [res (rt/score-sequence-log-probs session token-ids)]
                      (:log-probs res)))
        t1 (System/nanoTime)
        elapsed-sec (/ (- t1 t0) 1e9)
        tok-per-sec (if (pos? elapsed-sec) (/ num-total elapsed-sec) 0.0)
        metrics (probe-core/compute-sequence-metrics log-probs (:token-pairs eval-mask))]
    (when-not (:quiet opts)
      (println (format "  [%-4s] Task: %-20s | Total: %4d tok | Scored: %4d (Think: %3d, Code: %4d)"
                       (name group) task-id num-total num-scored num-think num-code))
      (println (format "         Scored in %.2fs (%.1f tok/s) | PPL: %7.4f (Think: %7.4f, Code: %7.4f)"
                       elapsed-sec tok-per-sec
                       (:perplexity metrics) (:think-perplexity metrics) (:code-perplexity metrics))))
    {:task-id task-id
     :group group
     :elapsed-sec elapsed-sec
     :tok-per-sec tok-per-sec
     :num-total num-total
     :num-scored num-scored
     :num-think num-think
     :num-code num-code
     :metrics metrics}))

(defn run-perplexity-probe
  "Main execution workflow: initializes model session, iterates over gap and reference trajectories,
   computes corpus-wide micro-average perplexity, and writes results.edn and summary.csv."
  [user-opts]
  (let [opts (merge DEFAULT-PROBE-OPTS user-opts)
        _ (when-not (:quiet opts)
            (println "================================================================================")
            (println "              PERPLEXITY PROBE v1: STUDENT SURPRISE EVALUATION")
            (println "================================================================================")
            (println (format "Student Model   : %s" (:model opts)))
            (println (format "Teacher Model   : %s" (:teacher-model opts)))
            (println (format "Backend Target  : %s" (:backend opts)))
            (println (format "Trajectories    : %s" (:trajectories-file opts)))
            (println (format "Tasks File      : %s" (:tasks-file opts)))
            (println (format "Max Seq Len     : %d" (:max-seq-len opts)))
            (println (format "Dry Run Mode    : %s" (:dry-run opts)))
            (println "--------------------------------------------------------------------------------"))
        trajs-by-task (load-trajectories-by-task (:trajectories-file opts))
        tasks-by-id (load-tasks-by-id (:tasks-file opts))
        _ (doseq [tid probe-core/ALL-PROBE-TASKS]
            (when-not (get trajs-by-task tid)
              (throw (ex-info (str "Missing trajectory for required task: " tid) {:task tid})))
            (when-not (get tasks-by-id tid)
              (throw (ex-info (str "Missing task definition in catalog for task: " tid) {:task tid}))))
        session (if (:dry-run opts)
                  {:tokenizer (create-mock-tokenizer)}
                  (let [base-sess (rt/init-agent-vram-session (assoc opts :method :kv-cache :max-seq-len (:max-seq-len opts)))
                        ctx (:ctx base-sess)
                        config (:config base-sess)
                        norm-dtype (if (or (:is-int8 config) (:is-int4 config) (:is-ternary config) (= (:weight-dtype config) :ternary))
                                     :bf16
                                     (or (:weight-dtype config) :f32))
                        target-exec (kernels/compile-target-log-prob-executable ctx (long (or (:vocab-size config) 262144)) norm-dtype)]
                    (assoc base-sess
                           :step-executable (or (:step-executable base-sess) (:executable base-sess))
                           :target-scoring-executable target-exec)))
        tokenizer (:tokenizer session)
        task-rows (atom [])]
    (try
      (when-not (:quiet opts)
        (println "\n[Phase 1] Evaluating Trajectories Under Teacher Forcing:"))
      (doseq [tid probe-core/ALL-PROBE-TASKS]
        (let [t-def (get tasks-by-id tid)
              traj (get trajs-by-task tid)
              row (evaluate-single-trajectory session t-def traj tokenizer opts)]
          (swap! task-rows conj row)))

      (let [rows @task-rows
            aggregates (probe-core/compute-corpus-aggregates rows)
            verdict (:verdict aggregates)
            student-sha (if (:dry-run opts) STUDENT-CHECKPOINT-SHA (compute-model-sha (:model opts)))
            report-map {:probe "perplexity_probe_v1"
                        :timestamp (str (java.time.Instant/now))
                        :student-model (:model opts)
                        :student-sha student-sha
                        :teacher-model (:teacher-model opts)
                        :teacher-sha TEACHER-CHECKPOINT-SHA
                        :backend (:backend opts)
                        :max-seq-len (:max-seq-len opts)
                        :dry-run? (:dry-run opts)
                        :eval-tasks probe-core/ALL-PROBE-TASKS
                        :gap-tasks probe-core/GAP-TASKS
                        :ref-tasks probe-core/REF-TASKS
                        :task-results rows
                        :aggregates aggregates
                        :verdict verdict}]

        ;; Format and print summary table
        (when-not (:quiet opts)
          (println "\n================================================================================")
          (println "                     PERPLEXITY PROBE v1 SUMMARY RESULTS")
          (println "================================================================================")
          (println (format "%-20s %-5s %6s %6s %6s %9s %9s %9s %6s %-6s"
                           "Task ID" "Group" "Total" "Scored" "Think" "PPL(all)" "PPL(thk)" "PPL(code)" "r_k" "SFT?"))
          (println "--------------------------------------------------------------------------------")
          (doseq [r rows]
            (let [tid (:task-id r)
                  m (:metrics r)
                  r_k (get-in aggregates [:task-ratios tid] 0.0)
                  sft? (contains? (set (:sft-candidates aggregates)) tid)]
              (println (format "%-20s %-5s %6d %6d %6d %9.4f %9.4f %9.4f %6.2f %-6s"
                               tid (name (:group r)) (:num-total r) (:num-scored r) (:num-think r)
                               (:perplexity m) (:think-perplexity m) (:code-perplexity m)
                               r_k (if sft? "YES" "NO")))))
          (println "--------------------------------------------------------------------------------")
          (println (format "Micro-Average Gap PPL       : %7.4f (Cross-Entropy: %.4f, N = %d tokens)"
                           (:gap-micro-ppl aggregates) (:gap-micro-ce aggregates) (:gap-tokens aggregates)))
          (println (format "Micro-Average Ref PPL       : %7.4f (Cross-Entropy: %.4f, N = %d tokens)"
                           (:ref-micro-ppl aggregates) (:ref-micro-ce aggregates) (:ref-tokens aggregates)))
          (println (format "Primary Ratio (Gap / Ref)   : %7.4f (Threshold: >= %.2f)"
                           (:ratio aggregates) probe-core/GO-RATIO-THRESHOLD))
          (println (format "Macro-Average Ratio         : %7.4f (Gap: %.4f / Ref: %.4f)"
                           (:macro-ratio aggregates) (:gap-macro-ppl aggregates) (:ref-macro-ppl aggregates)))
          (println (format "SFT Candidate Tasks         : %s" (str/join ", " (:sft-candidates aggregates))))
          (println (format "Flagged/Noise Tasks         : %s" (str/join ", " (:flagged-tasks aggregates))))
          (println "--------------------------------------------------------------------------------")
          (println (format "PRE-REGISTERED GATE DECISION: [%s]" (name verdict)))
          (println "================================================================================\n"))

        ;; Write results.edn
        (when-let [out-file (:results-file opts)]
          (io/make-parents (io/file out-file))
          (spit out-file (with-out-str (clojure.pprint/pprint report-map)))
          (when-not (:quiet opts)
            (println (str "Persisted full results EDN to: " out-file))))

        ;; Write summary.csv
        (when-let [csv-file (:summary-csv opts)]
          (io/make-parents (io/file csv-file))
          (let [sb (StringBuilder.)]
            (.append sb "task_id,group,num_total,num_scored,num_think,num_code,cross_entropy,perplexity,think_perplexity,code_perplexity,r_k,sft_candidate\n")
            (doseq [r rows]
              (let [tid (:task-id r)
                    m (:metrics r)
                    r_k (get-in aggregates [:task-ratios tid] 0.0)
                    sft? (contains? (set (:sft-candidates aggregates)) tid)]
                (.append sb (format "%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%s\n"
                                    tid (name (:group r)) (:num-total r) (:num-scored r) (:num-think r) (:num-code r)
                                    (:cross-entropy m) (:perplexity m) (:think-perplexity m) (:code-perplexity m)
                                    r_k (str sft?)))))
            (.append sb "# AGGREGATE SUMMARY\n")
            (.append sb (format "# Gap Micro PPL: %.6f (tokens: %d, cross-entropy: %.6f)\n"
                                (:gap-micro-ppl aggregates) (:gap-tokens aggregates) (:gap-micro-ce aggregates)))
            (.append sb (format "# Ref Micro PPL: %.6f (tokens: %d, cross-entropy: %.6f)\n"
                                (:ref-micro-ppl aggregates) (:ref-tokens aggregates) (:ref-micro-ce aggregates)))
            (.append sb (format "# Primary Ratio (Gap / Ref): %.6f (Threshold: >= %.2f)\n"
                                (:ratio aggregates) probe-core/GO-RATIO-THRESHOLD))
            (.append sb (format "# Macro Ratio: %.6f\n" (:macro-ratio aggregates)))
            (.append sb (format "# SFT Candidates: %s\n" (str/join "; " (:sft-candidates aggregates))))
            (.append sb (format "# Pre-Registered Verdict: %s\n" (name verdict)))
            (spit csv-file (.toString sb))
            (when-not (:quiet opts)
              (println (str "Persisted summary CSV to: " csv-file)))))

        report-map)
      (finally
        (when (and session (not (:dry-run opts)))
          (rt/close-agent-session! session))))))

;; =============================================================================
;; 4. CLI Entrypoint
;; =============================================================================

(defn parse-cli-opts
  "Parses command line arguments into keyword option map."
  [args]
  (loop [remaining args
         opts {}]
    (if (empty? remaining)
      opts
      (let [arg (first remaining)]
        (cond
          (str/starts-with? arg "--")
          (let [k (keyword (subs arg 2))
                v (second remaining)]
            (recur (drop 2 remaining) (assoc opts k v)))
          :else
          (recur (rest remaining) opts))))))

(defn normalize-opts
  "Normalizes CLI string options to typed values."
  [raw-opts]
  (let [opts (merge DEFAULT-PROBE-OPTS raw-opts)]
    (cond-> opts
      (string? (:backend opts)) (update :backend #(keyword (str/replace % #"^:+" "")))
      (string? (:dry-run opts)) (update :dry-run #(Boolean/parseBoolean %))
      (string? (:quiet opts)) (update :quiet #(Boolean/parseBoolean %))
      (string? (:max-seq-len opts)) (update :max-seq-len #(Long/parseLong %)))))

(defn -main
  [& args]
  (let [opts (normalize-opts (parse-cli-opts args))]
    (try
      (let [report (run-perplexity-probe opts)]
        (System/exit (if (= (:verdict report) :GO) 0 1)))
      (catch Throwable t
        (.printStackTrace t)
        (System/exit 2)))))
