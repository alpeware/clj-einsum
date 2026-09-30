(ns experiments.gate3-evals.perplexity-probe-v1.core
  "Pure functional core for Perplexity Probe v1 (Gate 3 Evals, Stage 2).
   Provides transcript tokenization, role-based structured masking,
   token-weighted micro-average corpus cross-entropy / perplexity calculation,
   pre-registered GO/NO-GO gate decision rules, and SFT task filtering."
  (:require [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [einsum.runtime.tokenizer.protocol :as tok-proto]))

;; =============================================================================
;; 1. Pinned Artifacts & Falsification Criteria
;; =============================================================================

(def TEACHER-CHECKPOINT-SHA
  "f20a02315954b540db819393224e07c3d60c1f07c9052b10a530db1556d2603c")

(def STUDENT-CHECKPOINT-SHA
  "33ebfa9b85f077df19c0526c9f0e039c044d4e64eef47368730208b478dc05ec")

(def GAP-TASKS
  ["first-n" "my-range" "deep-update-vals" "lazy-interleave"])

(def REF-TASKS
  ["freqs" "partition-by-parity"])

(def ALL-PROBE-TASKS
  (into GAP-TASKS REF-TASKS))

(def GAP-TASKS-SET
  (set GAP-TASKS))

(def REF-TASKS-SET
  (set REF-TASKS))

(def GO-RATIO-THRESHOLD 1.50)
(def SFT-TASK-THRESHOLD 1.30)

;; =============================================================================
;; 2. Structured Transcript Delineation & Evaluation Masking
;; =============================================================================

(defn build-evaluation-segments
  "Constructs structured turn segments from task prompt and teacher transcript.
   Each segment is a tuple: [text-content segment-type role mask-flag].
   Role-based masking rules:
     - Framing tokens (<bos>, <|turn>, <turn|>, <|channel>thought\n, <channel|>) receive mask 0.
     - System prompt and user prompt turns receive mask 0.
     - Tool observation turns receive mask 0.
     - Only :model turn content receives mask 1.
     - Model tokens are partitioned into :think (internal reasoning) and :code (solution/actions)."
  ([task transcript]
   (build-evaluation-segments task transcript {}))
  ([task transcript opts]
   (let [sys-prompt (or (:system opts) bench-core/AGENT-SYSTEM-PROMPT-FENCED)
         u-prompt (bench-core/render-benchmark-prompt task (or (:prompt-template opts) :clojure-bench-v1) :agentic)
         base-segments [["<bos>" :framing :framing 0]
                        ["<|turn>system\n" :framing :framing 0]
                        [(str sys-prompt "\n") :system :system 0]
                        ["<turn|>\n" :framing :framing 0]
                        ["<|turn>user\n" :framing :framing 0]
                        [(str u-prompt "\n") :user :user 0]
                        ["<turn|>\n" :framing :framing 0]]
         turn-segments
         (vec (mapcat
               (fn [t]
                 (cond
                   (= (:role t) :model)
                   (let [thought (when (seq (:thought t)) (:thought t))
                         resp (when (seq (:response t)) (:response t))]
                     (concat
                      [["<|turn>model\n" :framing :framing 0]]
                      (if thought
                        [["<|channel>thought\n" :framing :framing 0]
                         [thought :think :model 1]
                         ["<channel|>" :framing :framing 0]]
                        [["<|channel>thought\n<channel|>" :framing :framing 0]])
                      (when resp
                        [[resp :code :model 1]])
                      [["<turn|>\n" :framing :framing 0]]))

                   (= (:role t) :tool)
                   [["<|turn>user\n" :framing :framing 0]
                    [(str (:content t) "\n") :tool :tool 0]
                    ["<turn|>\n" :framing :framing 0]]

                   :else
                   []))
               transcript))]
     (into base-segments turn-segments))))

(defn build-evaluation-mask
  "Tokenizes structured transcript segments and builds token metadata vectors.
   Ensures position 0 (<bos>) strictly receives mask 0.
   Returns a map:
   {:token-pairs [{:id <int> :segment <kw> :role <kw> :mask <0|1> :pos <int>}]
    :token-ids [<int> ...]
    :mask [<0|1> ...]
    :num-total <int>
    :num-scored <int>
    :num-think <int>
    :num-code <int>}"
  [task transcript tokenizer opts]
  (let [segments (build-evaluation-segments task transcript opts)
        token-pairs (vec (mapcat
                          (fn [[text seg role mask]]
                            (let [ids (tok-proto/encode tokenizer text)]
                              (mapv (fn [id]
                                      {:id (int id)
                                       :segment seg
                                       :role role
                                       :mask (long mask)})
                                    ids)))
                          segments))
        ;; Invariant: position 0 is un-scored
        sanitized-pairs (if (seq token-pairs)
                          (assoc-in token-pairs [0 :mask] 0)
                          [])
        indexed-pairs (vec (map-indexed (fn [i m] (assoc m :pos i)) sanitized-pairs))
        token-ids (mapv :id indexed-pairs)
        mask-vec (mapv :mask indexed-pairs)
        n-total (count indexed-pairs)
        n-scored (count (filter #(= (:mask %) 1) indexed-pairs))
        n-think (count (filter #(and (= (:mask %) 1) (= (:segment %) :think)) indexed-pairs))
        n-code (count (filter #(and (= (:mask %) 1) (= (:segment %) :code)) indexed-pairs))]
    {:token-pairs indexed-pairs
     :token-ids token-ids
     :mask mask-vec
     :num-total n-total
     :num-scored n-scored
     :num-think n-think
     :num-code n-code}))

;; =============================================================================
;; 3. Perplexity & Cross-Entropy Metrics
;; =============================================================================

(defn compute-sequence-metrics
  "Computes NLL, mean cross-entropy, and perplexity over scored token positions.
   Takes:
     - `log-probs`: Vector of log-probabilities where (nth log-probs i) is log P(w_i | w_<i).
     - `token-pairs`: Metadata for each token from `build-evaluation-mask`.
   Returns a structured map of metrics overall, for thinking, and for code/action segments."
  [log-probs token-pairs]
  (let [n (min (count log-probs) (count token-pairs))
        scored-indices (filter #(= (:mask (nth token-pairs %)) 1) (range n))
        scored-lps (mapv #(double (nth log-probs %)) scored-indices)
        sum-nll (- (reduce + 0.0 scored-lps))
        n-scored (count scored-indices)
        mean-ce (if (pos? n-scored) (/ sum-nll n-scored) 0.0)
        ppl (if (pos? n-scored) (Math/exp mean-ce) 1.0)

        think-indices (filter #(and (= (:mask (nth token-pairs %)) 1)
                                    (= (:segment (nth token-pairs %)) :think))
                              (range n))
        think-lps (mapv #(double (nth log-probs %)) think-indices)
        think-nll (- (reduce + 0.0 think-lps))
        n-think (count think-indices)
        think-ce (if (pos? n-think) (/ think-nll n-think) 0.0)
        think-ppl (if (pos? n-think) (Math/exp think-ce) 1.0)

        code-indices (filter #(and (= (:mask (nth token-pairs %)) 1)
                                   (= (:segment (nth token-pairs %)) :code))
                             (range n))
        code-lps (mapv #(double (nth log-probs %)) code-indices)
        code-nll (- (reduce + 0.0 code-lps))
        n-code (count code-indices)
        code-ce (if (pos? n-code) (/ code-nll n-code) 0.0)
        code-ppl (if (pos? n-code) (Math/exp code-ce) 1.0)]
    {:num-total n
     :num-scored n-scored
     :num-think n-think
     :num-code n-code
     :sum-nll sum-nll
     :cross-entropy mean-ce
     :perplexity ppl
     :think-sum-nll think-nll
     :think-cross-entropy think-ce
     :think-perplexity think-ppl
     :code-sum-nll code-nll
     :code-cross-entropy code-ce
     :code-perplexity code-ppl}))

(defn compute-corpus-aggregates
  "Computes token-weighted micro-average corpus cross-entropy, perplexity,
   and pre-registered GO/NO-GO gate decision across all evaluated tasks.
   Takes a sequence of task result maps:
   [{:task-id \"first-n\" :group :gap :metrics {...}} ...]"
  [task-rows]
  (let [by-group (group-by :group task-rows)
        gap-rows (get by-group :gap [])
        ref-rows (get by-group :ref [])

        gap-tokens (reduce + 0 (map #(get-in % [:metrics :num-scored] 0) gap-rows))
        gap-nll (reduce + 0.0 (map #(get-in % [:metrics :sum-nll] 0.0) gap-rows))
        gap-micro-ce (if (pos? gap-tokens) (/ gap-nll gap-tokens) 0.0)
        gap-micro-ppl (if (pos? gap-tokens) (Math/exp gap-micro-ce) 1.0)
        gap-macro-ppl (if (seq gap-rows)
                        (/ (reduce + 0.0 (map #(get-in % [:metrics :perplexity] 1.0) gap-rows))
                           (count gap-rows))
                        1.0)

        ref-tokens (reduce + 0 (map #(get-in % [:metrics :num-scored] 0) ref-rows))
        ref-nll (reduce + 0.0 (map #(get-in % [:metrics :sum-nll] 0.0) ref-rows))
        ref-micro-ce (if (pos? ref-tokens) (/ ref-nll ref-tokens) 0.0)
        ref-micro-ppl (if (pos? ref-tokens) (Math/exp ref-micro-ce) 1.0)
        ref-macro-ppl (if (seq ref-rows)
                        (/ (reduce + 0.0 (map #(get-in % [:metrics :perplexity] 1.0) ref-rows))
                           (count ref-rows))
                        1.0)

        ratio (if (pos? ref-micro-ppl) (/ gap-micro-ppl ref-micro-ppl) 0.0)
        macro-ratio (if (pos? ref-macro-ppl) (/ gap-macro-ppl ref-macro-ppl) 0.0)
        verdict (if (>= ratio GO-RATIO-THRESHOLD) :GO :NO-GO)

        ;; Per-task ratio r_k = PPL_k / PPL_ref
        task-ratios
        (into {}
              (map (fn [row]
                     (let [tid (:task-id row)
                           ppl (get-in row [:metrics :perplexity] 1.0)
                           r_k (if (pos? ref-micro-ppl) (/ ppl ref-micro-ppl) 0.0)]
                       [tid r_k]))
                   task-rows))

        ;; SFT inclusion candidate tasks (r_k >= 1.30)
        sft-candidates
        (filter (fn [tid]
                  (let [r (get task-ratios tid 0.0)]
                    (>= r SFT-TASK-THRESHOLD)))
                GAP-TASKS)

        flagged-tasks
        (filter (fn [tid]
                  (let [r (get task-ratios tid 0.0)]
                    (< r SFT-TASK-THRESHOLD)))
                GAP-TASKS)]

    {:verdict verdict
     :ratio ratio
     :macro-ratio macro-ratio
     :gap-micro-ppl gap-micro-ppl
     :gap-macro-ppl gap-macro-ppl
     :gap-micro-ce gap-micro-ce
     :gap-tokens gap-tokens
     :gap-nll gap-nll
     :ref-micro-ppl ref-micro-ppl
     :ref-macro-ppl ref-macro-ppl
     :ref-micro-ce ref-micro-ce
     :ref-tokens ref-tokens
     :ref-nll ref-nll
     :task-ratios task-ratios
     :sft-candidates (vec sft-candidates)
     :flagged-tasks (vec flagged-tasks)}))

;; =============================================================================
;; 4. Reporting Artifact Formatters (CSV & EDN)
;; =============================================================================

(defn format-summary-csv
  "Formats summary table as comma-separated values matching spec requirements."
  [task-rows aggregates]
  (let [sb (StringBuilder.)
        _ (.append sb "task,group,scored_tokens,think_tokens,code_tokens,ppl_overall,ppl_think,ppl_code,ratio_to_ref,sft_include?\n")]
    (doseq [row task-rows]
      (let [tid (:task-id row)
            grp (name (:group row))
            m (:metrics row)
            r_k (get-in aggregates [:task-ratios tid] 0.0)
            inc? (if (= grp "gap") (>= r_k SFT-TASK-THRESHOLD) true)]
        (.append sb (format "%s,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%s\n"
                            tid grp
                            (long (:num-scored m))
                            (long (:num-think m))
                            (long (:num-code m))
                            (double (:perplexity m))
                            (double (:think-perplexity m))
                            (double (:code-perplexity m))
                            (double r_k)
                            (boolean inc?)))))
    ;; Append Aggregate Summary Rows
    (.append sb (format "CORPUS_GAP,gap,%d,-,-,%.4f,-,-,%.4f,-\n"
                        (long (:gap-tokens aggregates))
                        (double (:gap-micro-ppl aggregates))
                        (double (:ratio aggregates))))
    (.append sb (format "CORPUS_REF,ref,%d,-,-,%.4f,-,-,1.0000,-\n"
                        (long (:ref-tokens aggregates))
                        (double (:ref-micro-ppl aggregates))))
    (.append sb (format "VERDICT,%s,-,-,-,-,-,-,%.4f,-\n"
                        (name (:verdict aggregates))
                        (double (:ratio aggregates))))
    (.toString sb)))
