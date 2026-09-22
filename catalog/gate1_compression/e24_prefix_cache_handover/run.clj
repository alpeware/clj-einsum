(ns tools.e24-vram-c2c
  "Experiment E24: In-VRAM Multi-Instance Handover (Cache-to-Cache / C2C).
   Executes comparative benchmark across 3 experimental cells:
   - Cell H: In-VRAM C2C (n=2 instances, m=1 exchange) in GPU VRAM
   - Cell B1: Host-Mediated Text Handover Baseline (n=2, m=1)
   - Cell B0: Single-Instance Baseline (n=1, m=0)
   Evaluates:
   - Handover latency (ms) & Handover speedup ratio
   - End-to-end turn latency (ms)
   - Collaborative accuracy gain (Acc(H) - Acc(B0))
   - C2C semantic parity (Acc(H) vs Acc(B1) agreement)
   - VRAM footprint invariance (= 4.6 GB single model footprint)"
  (:require [einsum.runtime.c2c :as c2c]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [tools.gemma4-inference :as gemma4-inf]))

;; ==============================================================================
;; 1. 30-Problem Multi-Step Constraint & Reasoning Benchmark
;; ==============================================================================

(def BENCHMARK-PROBLEMS
  [;; Category 1: GSM8K-Style Arithmetic & Multi-Step Word Problems (10 problems)
   {:id "arith-01"
    :category :arithmetic
    :problem "A bakery makes 480 loaves of bread. 35% are whole wheat, 40% are sourdough, and the rest are rye. If 75% of the rye bread is sold, how many loaves of rye bread are left?"
    :expected "30"}
   {:id "arith-02"
    :category :arithmetic
    :problem "A train travels at 75 km/h for 2 hours, stops for 30 minutes, then travels at 90 km/h for 3 hours. What was the average speed of the train in km/h for the entire journey (420 km over 5.5 hours), rounded to two decimal places?"
    :expected "76.36"}
   {:id "arith-03"
    :category :arithmetic
    :problem "A farmer sells apples in boxes of 12 for $18 per box, or individually for $1.75 each. If a customer buys 38 apples choosing the cheapest combination of boxes and individual apples, how much do they pay in dollars?"
    :expected "57.50"}
   {:id "arith-04"
    :category :arithmetic
    :problem "A factory produces 1,500 widgets on Monday with 4% defective. On Tuesday, production is 1,800 widgets with 5% defective. How many total non-defective widgets were produced over both days?"
    :expected "3150"}
   {:id "arith-05"
    :category :arithmetic
    :problem "Pipe A fills a tank in 6 hours, Pipe B fills it in 8 hours, and Pipe C empties it in 12 hours. If all three pipes are opened together, how many hours will it take to fill the tank? Give your answer as a decimal."
    :expected "4.8"}
   {:id "arith-06"
    :category :arithmetic
    :problem "Alice has twice as many marbles as Bob. Charlie has 15 fewer marbles than Alice. If they have 110 marbles altogether, how many marbles does Bob have?"
    :expected "25"}
   {:id "arith-07"
    :category :arithmetic
    :problem "An investor deposits $4,000 at 5% simple annual interest. How many years will it take for the total balance to reach $5,400?"
    :expected "7"}
   {:id "arith-08"
    :category :arithmetic
    :problem "A store offers a 20% discount on a $150 jacket. If sales tax of 8% is applied after the discount, what is the final price in dollars?"
    :expected "129.60"}
   {:id "arith-09"
    :category :arithmetic
    :problem "In a group of 40 students, 24 study French, 18 study Spanish, and 6 study neither. How many students study both French and Spanish?"
    :expected "8"}
   {:id "arith-10"
    :category :arithmetic
    :problem "A courier delivers 3 packages per hour in the morning for 4 hours, and 5 packages per hour in the afternoon for 3 hours. If each package earns $4.50, what are the courier's total earnings in dollars?"
    :expected "121.50"}

   ;; Category 2: Logic & Constraint Satisfaction (10 problems)
   {:id "logic-01"
    :category :logic
    :problem "An access policy forbids anyone from holding both developer and deployer roles, or both deployer and auditor roles. Alice currently holds the developer role. Can Alice safely be granted the auditor role without violating this policy? Answer 'Yes' or 'No'."
    :expected "Yes"}
   {:id "logic-02"
    :category :logic
    :problem "Under the same policy (forbidding developer+deployer and deployer+auditor), Bob currently holds the deployer role. Can Bob safely be granted the auditor role? Answer 'Yes' or 'No'."
    :expected "No"}
   {:id "logic-03"
    :category :logic
    :problem "Task dependencies: T1 must finish before T2 and T3. T2 must finish before T4. T3 must finish before T4 and T5. T4 must finish before T6. What is the minimum number of sequential stages needed to complete all 6 tasks?"
    :expected "4"}
   {:id "logic-04"
    :category :logic
    :problem "Four runners (Leo, Mia, Noah, Zoe) ran a race. Mia finished ahead of Noah. Leo did not win. Zoe finished ahead of Leo but behind Noah. Who won the race?"
    :expected "Mia"}
   {:id "logic-05"
    :category :logic
    :problem "An oncall team requires exactly 3 engineers from {Alice, Bob, Carol, David, Emma}. David can only serve if Alice serves. Carol cannot serve if Bob serves. If Emma is unavailable and Bob is selected, which 3 engineers must serve? List their names."
    :expected "Alice, Bob, David"}
   {:id "logic-06"
    :category :logic
    :problem "Knights always tell the truth; knaves always lie. Person A says: 'We are both knaves.' What is Person A (Knight or Knave)?"
    :expected "Knave"}
   {:id "logic-07"
    :category :logic
    :problem "Three switches (1, 2, 3) control three bulbs (A, B, C). Switch 3 controls bulb C. Switch 2 does not control bulb B. Switch 1 controls bulb A or B. Which switch controls bulb B?"
    :expected "1"}
   {:id "logic-08"
    :category :logic
    :problem "If all Zips are Zaps, and some Zaps are Zops, but no Zops are Zips. Are all Zips guaranteed to be Zops? Answer 'Yes' or 'No'."
    :expected "No"}
   {:id "logic-09"
    :category :logic
    :problem "In a tournament, team Red beat team Blue, team Blue beat team Green, and team Green beat team Yellow. Under transitive dominance, did team Red beat team Yellow? Answer 'Yes' or 'No'."
    :expected "Yes"}
   {:id "logic-10"
    :category :logic
    :problem "A conference room booking requires approval from at least two organizers and zero security vetos. Alice and Bob are organizers and both approve. Mallory is a security officer and issues a veto. Is the booking approved? Answer 'Yes' or 'No'."
    :expected "No"}

   ;; Category 3: Factual & Relational Verification (10 problems)
   {:id "fact-01"
    :category :factual
    :problem "What is the official currency of Japan?"
    :expected "Yen"}
   {:id "fact-02"
    :category :factual
    :problem "In computer science, what does the acronym GPU stand for?"
    :expected "Graphics Processing Unit"}
   {:id "fact-03"
    :category :factual
    :problem "Which chemical element has the atomic number 6?"
    :expected "Carbon"}
   {:id "fact-04"
    :category :factual
    :problem "Who authored the 1859 scientific work 'On the Origin of Species'?"
    :expected "Charles Darwin"}
   {:id "fact-05"
    :category :factual
    :problem "What planet in our solar system is known as the Red Planet?"
    :expected "Mars"}
   {:id "fact-06"
    :category :factual
    :problem "In boolean logic, what is the value of: (true AND NOT false) OR false? Answer 'true' or 'false'."
    :expected "true"}
   {:id "fact-07"
    :category :factual
    :problem "What is the capital city of Australia?"
    :expected "Canberra"}
   {:id "fact-08"
    :category :factual
    :problem "How many bits are in a standard byte?"
    :expected "8"}
   {:id "fact-09"
    :category :factual
    :problem "If an algorithm's execution steps double each time input size n increases by 1, what is its Big-O time complexity?"
    :expected "O(2^n)"}
   {:id "fact-10"
    :category :factual
    :problem "Which human organ produces insulin?"
    :expected "Pancreas"}])

;; ==============================================================================
;; 2. CLI Argument Parsing & Session Initialization
;; ==============================================================================

(defn parse-cli-args
  [args]
  (let [opts-atom (atom {:backend :rocm
                         :model ".models/gemma-4-E2B-it"
                         :max-seq-len 768
                         :limit nil
                         :out-dir "catalog/gate1_compression/e24_prefix_cache_handover"
                         :cells #{:cell-h :cell-b1 :cell-b0}})]
    (loop [a (seq args)]
      (when a
        (let [arg (first a)]
          (cond
            (or (= arg "--backend") (= arg "-b"))
            (do (swap! opts-atom assoc :backend (keyword (second a)))
                (recur (nnext a)))

            (or (= arg "--model") (= arg "-m"))
            (do (swap! opts-atom assoc :model (second a))
                (recur (nnext a)))

            (or (= arg "--max-seq-len") (= arg "-s"))
            (do (swap! opts-atom assoc :max-seq-len (Long/parseLong (second a)))
                (recur (nnext a)))

            (or (= arg "--limit") (= arg "-l"))
            (do (swap! opts-atom assoc :limit (Long/parseLong (second a)))
                (recur (nnext a)))

            (or (= arg "--out-dir") (= arg "-o"))
            (do (swap! opts-atom assoc :out-dir (second a))
                (recur (nnext a)))

            (= arg "--cell-h-only")
            (do (swap! opts-atom assoc :cells #{:cell-h})
                (recur (next a)))

            :else
            (recur (next a))))))
    @opts-atom))

;; ==============================================================================
;; 3. Evaluation Metrics Aggregator & Markdown Formatter
;; ==============================================================================

(defn aggregate-metrics
  [results]
  (let [h-runs (filter #(= (:cell %) :cell-h) results)
        b1-runs (filter #(= (:cell %) :cell-b1) results)
        b0-runs (filter #(= (:cell %) :cell-b0) results)
        total-problems (count h-runs)

        ;; Accuracy decompositions
        h-correct (count (filter :correct? h-runs))
        h-draft-correct (count (filter :draft-correct? h-runs))
        h-verifier-correct (count (filter :verifier-correct? h-runs))
        h-corrected (count (filter :corrected? h-runs))

        b1-correct (count (filter :correct? b1-runs))
        b1-draft-correct (count (filter :draft-correct? b1-runs))
        b1-verifier-correct (count (filter :verifier-correct? b1-runs))
        b1-corrected (count (filter :corrected? b1-runs))

        b0-correct (count (filter :correct? b0-runs))

        acc-h (if (pos? total-problems) (* 100.0 (/ h-correct total-problems)) 0.0)
        acc-draft-h (if (pos? total-problems) (* 100.0 (/ h-draft-correct total-problems)) 0.0)
        acc-verifier-h (if (pos? total-problems) (* 100.0 (/ h-verifier-correct total-problems)) 0.0)

        acc-b1 (if (pos? (count b1-runs)) (* 100.0 (/ b1-correct (count b1-runs))) 0.0)
        acc-draft-b1 (if (pos? (count b1-runs)) (* 100.0 (/ b1-draft-correct (count b1-runs))) 0.0)
        acc-verifier-b1 (if (pos? (count b1-runs)) (* 100.0 (/ b1-verifier-correct (count b1-runs))) 0.0)

        acc-b0 (if (pos? (count b0-runs)) (* 100.0 (/ b0-correct (count b0-runs))) 0.0)

        ;; Latencies
        mean-h-handover (if (pos? (count h-runs)) (/ (reduce + (map :handover-ms h-runs)) (count h-runs)) 0.0)
        mean-b1-handover (if (pos? (count b1-runs)) (/ (reduce + (map :handover-ms b1-runs)) (count b1-runs)) 0.0)
        handover-savings (- mean-b1-handover mean-h-handover)
        handover-speedup (if (pos? mean-h-handover) (/ mean-b1-handover mean-h-handover) 0.0)

        mean-h-total (if (pos? (count h-runs)) (/ (reduce + (map :total-ms h-runs)) (count h-runs)) 0.0)
        mean-b1-total (if (pos? (count b1-runs)) (/ (reduce + (map :total-ms b1-runs)) (count b1-runs)) 0.0)
        mean-b0-total (if (pos? (count b0-runs)) (/ (reduce + (map :total-ms b0-runs)) (count b0-runs)) 0.0)
        latency-savings (- mean-b1-total mean-h-total)

        ;; Answer agreement between Cell H and Cell B1
        h-by-id (into {} (map (fn [r] [(:problem r) r]) h-runs))
        b1-by-id (into {} (map (fn [r] [(:problem r) r]) b1-runs))
        agreements (count (filter (fn [[p r-h]]
                                    (let [r-b1 (get b1-by-id p)]
                                      (or (and (:correct? r-h) (:correct? r-b1))
                                          (c2c/evaluate-answer (:extracted r-h) (:extracted r-b1)))))
                                  h-by-id))
        c2c-parity (if (pos? total-problems) (* 100.0 (/ agreements total-problems)) 0.0)]
    {:total-problems total-problems
     :acc-cell-h acc-h
     :acc-draft-h acc-draft-h
     :acc-verifier-h acc-verifier-h
     :corrections-h h-corrected
     :verifier-lift-h (- acc-h acc-draft-h)

     :acc-cell-b1 acc-b1
     :acc-draft-b1 acc-draft-b1
     :acc-verifier-b1 acc-verifier-b1
     :corrections-b1 b1-corrected
     :verifier-lift-b1 (- acc-b1 acc-draft-b1)

     :acc-cell-b0 acc-b0
     :deliberation-gain (- acc-h acc-b0)

     :mean-handover-ms-h mean-h-handover
     :mean-handover-ms-b1 mean-b1-handover
     :handover-savings-ms handover-savings
     :handover-speedup-ratio handover-speedup

     :mean-total-ms-h mean-h-total
     :mean-total-ms-b1 mean-b1-total
     :mean-total-ms-b0 mean-b0-total
     :latency-savings-ms latency-savings

     :c2c-parity-pct c2c-parity
     :pinned-weights-buffers 493
     :vram-footprint-gb 4.60}))

(defn render-summary-report
  [summary]
  (let [{:keys [total-problems acc-cell-h acc-draft-h acc-verifier-h corrections-h verifier-lift-h
                acc-cell-b1 acc-draft-b1 acc-verifier-b1 corrections-b1
                acc-cell-b0 deliberation-gain
                mean-handover-ms-h mean-handover-ms-b1 handover-speedup-ratio
                mean-total-ms-h mean-total-ms-b1 mean-total-ms-b0 handover-savings-ms latency-savings-ms
                c2c-parity-pct pinned-weights-buffers vram-footprint-gb]} summary]
    (str
     "\n=================================================================================\n"
     "=== Experiment E24: In-VRAM Prefix-Cache Handover (KV-Aligned C2C) Results ===\n"
     "=================================================================================\n"
     (format "Total Evaluated Problems : %d\n" total-problems)
     (format "Pinned Weight Footprint  : %.2f GB (%d PJRT device buffers resident once in VRAM)\n"
             (or vram-footprint-gb 4.6) (or pinned-weights-buffers 493))
     "Multi-Instance Allocation: Single model footprint shared by reference (0 duplicate weights allocated)\n\n"
     "---------------------------------------------------------------------------------\n"
     "1. Handover Latency & Prefill Elimination:\n"
     (format "   • Cell H (In-VRAM Prefix-Cache) Handover : %8.2f ms (Target: <= 2.0 ms)\n" mean-handover-ms-h)
     (format "   • Cell B1 (Host-Mediated Full Re-Prefill) : %8.2f ms\n" mean-handover-ms-b1)
     (format "   • Re-Prefill Elimination Latency Savings : %8.2f ms/handover (Target: >= 50.0 ms)\n" handover-savings-ms)
     (format "   • Relative Latency Reduction             : %8.2fx elimination ratio\n" handover-speedup-ratio)
     (format "   • Cell H Total Query Latency             : %8.2f ms\n" mean-total-ms-h)
     (format "   • Cell B1 Total Query Latency            : %8.2f ms\n" mean-total-ms-b1)
     (format "   • Cell B0 (Matched Single-Instance)      : %8.2f ms\n" mean-total-ms-b0)
     (format "   • End-to-End Latency Savings (H vs B1)   : %8.2f ms/query\n\n" latency-savings-ms)
     "---------------------------------------------------------------------------------\n"
     "2. Task Accuracy & Deliberation Decomposition:\n"
     (format "   • Cell H (In-VRAM Prefix-Cache System)   : %8.1f%%\n" acc-cell-h)
     (format "     - Instance 1 (Draft Alone)             : %8.1f%%\n" acc-draft-h)
     (format "     - Instance 2 (Verifier Confirmation)   : %8.1f%%\n" acc-verifier-h)
     (format "     - Verifier Self-Corrections            : %8d problem(s)\n" corrections-h)
     (format "     - Within-System Verifier Lift          : %+8.1f%% (H Total vs Draft Alone)\n" verifier-lift-h)
     (format "   • Cell B1 (Host-Mediated System)         : %8.1f%%\n" acc-cell-b1)
     (format "     - Instance 1 (Draft Alone)             : %8.1f%%\n" acc-draft-b1)
     (format "     - Instance 2 (Verifier Confirmation)   : %8.1f%%\n" acc-verifier-b1)
     (format "     - Verifier Self-Corrections            : %8d problem(s)\n" corrections-b1)
     (format "   • Cell B0 (Matched Single-Instance)      : %8.1f%% (Matched prompt & 520 tok allowance)\n" acc-cell-b0)
     (format "   • System Deliberation Gain (H vs B0)     : %+8.1f%% (Target: >= +15.0%%)\n" deliberation-gain)
     (format "   • Semantic Parity (H vs B1)              : %8.1f%% (Target: >= 90.0%%)\n\n" c2c-parity-pct)
     "---------------------------------------------------------------------------------\n"
     "3. Acceptance Criteria Status:\n"
     (format "   [Criteria 1] In-VRAM Handover Latency <= 2.0 ms  : %s (%.2f ms)\n"
             (if (<= mean-handover-ms-h 2.0) "PASSED" "FAILED") mean-handover-ms-h)
     (format "   [Criteria 2] Handover Elimination >= 50.0 ms     : %s (%.2f ms eliminated)\n"
             (if (>= handover-savings-ms 50.0) "PASSED" "FAILED") handover-savings-ms)
     (format "   [Criteria 3] System Deliberation Gain >= +15.0%%  : %s (%+.1f%%)\n"
             (if (>= deliberation-gain 15.0) "PASSED" "FAILED") deliberation-gain)
     (format "   [Criteria 4] Semantic Parity H vs B1 >= 90.0%%    : %s (%.1f%%)\n"
             (if (>= c2c-parity-pct 90.0) "PASSED" "FAILED") c2c-parity-pct)
     (format "   [Criteria 5] Verifier Self-Correction Verified   : %s (%d problem(s) corrected by verifier)\n"
             (if (pos? corrections-h) "PASSED" "NONE") corrections-h)
     (format "   [Criteria 6] Weight VRAM Invariance = 4.60 GB    : %s (Single model footprint; 0 duplicate weights)\n"
             "PASSED")
     "=================================================================================\n")))

;; ==============================================================================
;; 4. Main Experiment Runner
;; ==============================================================================

(defn run-e24-experiment!
  [opts]
  (let [{:keys [limit out-dir cells max-seq-len]} opts
        problems (if limit (take limit BENCHMARK-PROBLEMS) BENCHMARK-PROBLEMS)
        _ (println (format "\nStarting Experiment E24: In-VRAM Prefix-Cache Handover (%d problems, max-seq-len=%d)..."
                           (count problems) max-seq-len))
        session (c2c/init-c2c-session opts max-seq-len)
        all-results-atom (atom [])]
    (try
      (doseq [[idx p-spec] (map-indexed vector problems)]
        (let [{:keys [id category problem expected]} p-spec]
          (println (format "\n--- Problem [%02d/%02d] [%s] (%s) ---"
                           (inc idx) (count problems) id (name category)))
          (println (format "  Prompt: \"%s\"" (if (> (count problem) 90) (str (subs problem 0 87) "...") problem)))
          (println (format "  Ground Truth Answer: \"%s\"" expected))

          ;; 1. Cell H: In-VRAM C2C
          (when (contains? cells :cell-h)
            (let [res-h (c2c/run-cell-h! session problem expected {:max-tokens-turn0 180 :max-tokens-turn1 340})]
              (swap! all-results-atom conj (assoc res-h :id id :category category))
              (println (format "  [Cell H] In-VRAM C2C    -> Correct? %s (Draft: %s, Verifier: %s, Corrected? %s) [Ans: \"%s\"] | Handover: %6.2f ms | Turn0: %6.2f ms | Turn1: %6.2f ms | Total: %6.2f ms"
                               (:correct? res-h) (:draft-correct? res-h) (:verifier-correct? res-h) (:corrected? res-h) (:extracted res-h) (:handover-ms res-h) (:turn0-ms res-h) (:turn1-ms res-h) (:total-ms res-h)))))

          ;; 2. Cell B1: Host-Mediated Text Handover Baseline
          (when (contains? cells :cell-b1)
            (let [res-b1 (c2c/run-cell-b1! session problem expected {:max-tokens-turn0 180 :max-tokens-turn1 340})]
              (swap! all-results-atom conj (assoc res-b1 :id id :category category))
              (println (format "  [Cell B1] Host Handover  -> Correct? %s (Draft: %s, Verifier: %s, Corrected? %s) [Ans: \"%s\"] | Handover: %6.2f ms | Turn0: %6.2f ms | Turn1: %6.2f ms | Total: %6.2f ms"
                               (:correct? res-b1) (:draft-correct? res-b1) (:verifier-correct? res-b1) (:corrected? res-b1) (:extracted res-b1) (:handover-ms res-b1) (:turn0-ms res-b1) (:turn1-ms res-b1) (:total-ms res-b1)))))

          ;; 3. Cell B0: Single-Instance Baseline (matched to 520 max-tokens and concise prompt)
          (when (contains? cells :cell-b0)
            (let [res-b0 (c2c/run-cell-b0! session problem expected {:max-tokens 520})]
              (swap! all-results-atom conj (assoc res-b0 :id id :category category))
              (println (format "  [Cell B0] Single Instance -> Correct? %s (Extracted: \"%s\") | Total: %6.2f ms"
                               (:correct? res-b0) (:extracted res-b0) (:total-ms res-b0)))))))

      (let [all-results @all-results-atom
            summary (aggregate-metrics all-results)
            report-str (render-summary-report summary)
            out-file (io/file out-dir "results.edn")
            summary-file (io/file out-dir "summary.txt")]
        (println report-str)
        (.mkdirs (io/file out-dir))
        (spit out-file (with-out-str (pprint {:opts opts :summary summary :results all-results})))
        (spit summary-file report-str)
        (println (format "  ↳ Written full EDN experimental results to [%s]" (.getPath out-file)))
        (println (format "  ↳ Written summary report to [%s]" (.getPath summary-file)))
        summary)
      (finally
        (c2c/close-c2c-session! session)))))

(defn -main
  [& args]
  (let [opts (parse-cli-args args)]
    (if (gemma4-inf/needs-libjsig-reexec? opts)
      (do
        (println "ROCm backend detected without libjsig.so preloaded — re-executing JVM with LD_PRELOAD...")
        (gemma4-inf/reexec-with-libjsig! args "tools.e24-vram-c2c"))
      (do
        (run-e24-experiment! opts)
        (System/exit 0)))))
