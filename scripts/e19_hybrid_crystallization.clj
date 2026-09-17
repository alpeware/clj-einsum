(ns scripts.e19-hybrid-crystallization
  "Experiment E19: Hybrid Crystallization — GD Proposes, Discrete Search Disposes.
   Combines gradient descent over continuous tensor compositions (E17 A2 proposer)
   with combinatorial boolean matrix search (discrete crystallization).
   Executes Phase 0 gates (P0a Oracle ceiling, P0b Proposer quality)
   and Phase 1 factorial search (H vs B0 vs B1) across seeds 42–46."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.data.family-trees :as ft]
            [clj-xla.logic.models.predicate-invention :as pi]
            [clj-xla.logic.predicate-crystallization :as pc]
            [clojure.java.io :as io]
            [clojure.set :as set]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT-OPTS
  {:backend :cpu
   :num-entities 64
   :num-trees 8
   :dim 64
   :rank 8
   :epochs 200
   :lr 0.05
   :lr-z 0.25
   :tau-init 1.0
   :tau-min 0.05
   :lambda-l1 1e-4
   :lambda-search 0.01
   :top-k 256
   :seeds [42 43 44 45 46]
   :b1-draws 3
   :out-dir "paper-experiments/e19-hybrid/2026-09-17/"})

(defn- normalize-args [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT-OPTS]
    (if (empty? remaining)
      opts
      (let [k (first remaining)
            v (second remaining)]
        (case k
          "--backend" (recur (subvec remaining 2) (assoc opts :backend (keyword v)))
          "--num-entities" (recur (subvec remaining 2) (assoc opts :num-entities (Long/parseLong v)))
          "--num-trees" (recur (subvec remaining 2) (assoc opts :num-trees (Long/parseLong v)))
          "--dim" (recur (subvec remaining 2) (assoc opts :dim (Long/parseLong v)))
          "--epochs" (recur (subvec remaining 2) (assoc opts :epochs (Long/parseLong v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--lr-z" (recur (subvec remaining 2) (assoc opts :lr-z (Double/parseDouble v)))
          "--lambda-search" (recur (subvec remaining 2) (assoc opts :lambda-search (Double/parseDouble v)))
          "--top-k" (recur (subvec remaining 2) (assoc opts :top-k (Long/parseLong v)))
          "--out-dir" (recur (subvec remaining 2) (assoc opts :out-dir v))
          (recur (subvec remaining 1) opts))))))

(defn- mean [coll]
  (if (empty? coll) 0.0 (/ (reduce + (map double coll)) (double (count coll)))))

(defn- std-dev [coll]
  (if (< (count coll) 2)
    0.0
    (let [m (mean coll)
          variance (/ (reduce + (map #(Math/pow (- (double %) m) 2.0) coll))
                      (double (dec (count coll))))]
      (Math/sqrt variance))))

;; ==============================================================================
;; Phase 0a: Oracle Ceiling Validation
;; ==============================================================================

(defn run-p0a-oracle-ceiling
  "Runs Phase 0a oracle ceiling validation:
   Constructs a clean 8-tree forest (56 edges, 8 entities/tree, 7 edges/tree)
   with known ground truth, and runs backward elimination on C = true-parents.
   Gate requirement: must reach F1 = 1.0."
  [opts]
  (println "\n================================================================================")
  (println "  PHASE 0a: ORACLE CEILING VALIDATION (Search Machinery Validation)")
  (println "================================================================================")
  (let [lambda-val (double (:lambda-search opts))
        make-tree (fn [offset]
                    (let [o offset]
                      [[(+ o 0) (+ o 1)]
                       [(+ o 0) (+ o 2)]
                       [(+ o 1) (+ o 3)]
                       [(+ o 1) (+ o 4)]
                       [(+ o 2) (+ o 5)]
                       [(+ o 2) (+ o 6)]
                       [(+ o 2) (+ o 7)]]))
        clean-parents (set (mapcat make-tree (range 0 64 8)))
        gp-target (pc/bool-mat-or-and-sq clean-parents)
        sib-target (pc/bool-mat-trans-or-and clean-parents)
        t0 (System/nanoTime)
        search-res (pc/backward-elimination clean-parents gp-target sib-target lambda-val)
        time-ms (/ (- (System/nanoTime) t0) 1e6)
        recovered (:selected-edges search-res)
        f1-metrics (pc/pair-set-f1 recovered clean-parents)
        f1 (:f1 f1-metrics)
        pass? (== 1.0 f1)]
    (println (format "  True Parents:           %d edges (8 trees x 7 edges)" (count clean-parents)))
    (println (format "  Target Compositions:    Grandparent: %d pairs | Sibling: %d pairs" (count gp-target) (count sib-target)))
    (println (format "  Search Result:          Recovered %d edges in %.2f ms (evals: %d)"
                     (count recovered) time-ms (:eval-count search-res)))
    (println (format "  Discrete Metrics:       Precision: %.4f | Recall: %.4f | F1: %.4f"
                     (:precision f1-metrics) (:recall f1-metrics) f1))
    (println (format "  P0a Gate Verdict:       [%s] (Requirement: F1 == 1.0)" (if pass? "PASS" "FAIL")))
    {:pass? pass?
     :true-edges (count clean-parents)
     :recovered-edges (count recovered)
     :precision (:precision f1-metrics)
     :recall (:recall f1-metrics)
     :f1 f1
     :time-ms time-ms
     :eval-count (:eval-count search-res)}))

;; ==============================================================================
;; Phase 0b: Proposer Quality Audit (Retrain E17 A2 & Measure Recall@256)
;; ==============================================================================

(defn retrain-proposer-a2
  "Retrains E17 A2 full latent matrix model in OpenXLA PJRT VRAM for given seed.
   Returns {:forest ... :Z ... :final-L ... :train-ms ... :compile-ms ...}."
  [ctx forest seed opts]
  (let [n (long (:num-entities opts))
        dim (long (:dim opts))
        epochs (long (:epochs opts))
        lr (double (:lr opts))
        lr-z (double (:lr-z opts))
        tau-init (double (:tau-init opts))
        tau-min (double (:tau-min opts))
        lambda-l1 (double (:lambda-l1 opts))
        step-opts {:num-entities n
                   :num-trees (:num-trees opts)
                   :dim dim
                   :rank (:rank opts)
                   :ablation :a2
                   :lr lr
                   :lr-z lr-z}
        t0-compile (System/nanoTime)
        exec (pi/compile-predicate-invention-step ctx step-opts)
        compile-ms (/ (- (System/nanoTime) t0-compile) 1e6)
        initial-state (pi/init-parameters forest step-opts)
        t0-train (System/nanoTime)
        final-state
        (loop [epoch 0
               state initial-state]
          (if (>= epoch epochs)
            state
            (let [progress (/ (double epoch) (double (max 1 (dec epochs))))
                  tau (* tau-init (Math/pow (/ tau-min tau-init) progress))
                  batch (pi/make-step-batch forest state {:tau tau :lambda-l1 lambda-l1})
                  step-res (pi/run-predicate-invention-step! exec batch)]
              (recur (inc epoch)
                     (assoc state
                            :E (:E_new step-res)
                            :W_gp (:W_gp_new step-res)
                            :W_sib (:W_sib_new step-res)
                            :Z (:Z_new step-res))))))
        train-ms (/ (- (System/nanoTime) t0-train) 1e6)]
    {:forest forest
     :seed seed
     :compile-ms compile-ms
     :train-ms train-ms
     :Z (:Z final-state)}))

(defn extract-candidate-edges
  "Extracts top-K off-diagonal candidate edges sorted by raw logit Z descending.
   Excludes self-loops (Parent relation is irreflexive)."
  [^floats z-arr ^long n ^long top-k]
  (let [off-diag
        (for [i (range n)
              j (range n)
              :when (not= i j)]
          [[i j] (double (aget z-arr (+ (* i n) j)))])
        sorted-pairs (mapv first (take top-k (sort-by second > off-diag)))]
    (set sorted-pairs)))

(defn run-p0b-proposer-quality
  "Evaluates candidate recall@K of true Parent edges in the top-K by raw logit Z."
  [proposer-runs top-k]
  (println "\n================================================================================")
  (println (format "  PHASE 0b: PROPOSER QUALITY AUDIT (Candidate Recall@%d by Raw Logit Z)" top-k))
  (println "================================================================================")
  (let [per-seed-results
        (mapv (fn [run]
                (let [seed (:seed run)
                      forest (:forest run)
                      true-parents (set (:parent-pairs forest))
                      n (long (:num-entities forest))
                      z-arr ^floats (:Z run)
                      cands (extract-candidate-edges z-arr n top-k)
                      recalled (set/intersection cands true-parents)
                      recall (/ (double (count recalled)) (double (count true-parents)))]
                  (println (format "  [Seed %d] True: %d | Top-%d Candidates: %d | Recalled: %2d | Recall: %6.2f%%"
                                   seed (count true-parents) top-k (count cands) (count recalled) (* 100.0 recall)))
                  {:seed seed
                   :true-count (count true-parents)
                   :top-k top-k
                   :recalled-count (count recalled)
                   :recall recall
                   :candidates cands}))
              proposer-runs)
        recalls (mapv :recall per-seed-results)
        mean-recall (mean recalls)
        std-recall (std-dev recalls)
        pass? (>= mean-recall 0.90)]
    (println "--------------------------------------------------------------------------------")
    (println (format "  Mean Candidate Recall@%d:  %6.2f%% +/- %5.2f%%" top-k (* 100.0 mean-recall) (* 100.0 std-recall)))
    (println (format "  P0b Gate Verdict:          [%s] (Requirement: mean recall >= 90.00%%)" (if pass? "PASS" "FAIL")))
    (println "--------------------------------------------------------------------------------")
    {:pass? pass?
     :top-k top-k
     :mean-recall mean-recall
     :std-recall std-recall
     :per-seed per-seed-results}))

;; ==============================================================================
;; Phase 1: Combinatorial Search Sweep (H, B0, B1)
;; ==============================================================================

(defn evaluate-candidate-cell
  "Runs discrete search on a candidate set and evaluates F1 vs hidden Parent."
  [cell-name candidates forest opts]
  (let [true-parents (set (:parent-pairs forest))
        g-train (set (get-in forest [:splits :grandparent :train]))
        s-train (set (get-in forest [:splits :sibling :train]))
        lambda-val (double (:lambda-search opts))
        t0 (System/nanoTime)
        search-res (pc/discrete-crystallization-search candidates g-train s-train {:lambda lambda-val})
        time-ms (/ (- (System/nanoTime) t0) 1e6)
        recovered (:edges search-res)
        f1-metrics (pc/pair-set-f1 recovered true-parents)
        gp-pred (pc/bool-mat-or-and-sq recovered)
        sib-pred (pc/bool-mat-trans-or-and recovered)
        gp-train-f1 (:f1 (pc/pair-set-f1 gp-pred g-train))
        sib-train-f1 (:f1 (pc/pair-set-f1 sib-pred s-train))]
    {:cell cell-name
     :seed (:seed forest)
     :candidate-count (count candidates)
     :recovered-count (count recovered)
     :score (:score search-res)
     :precision (:precision f1-metrics)
     :recall (:recall f1-metrics)
     :parent-f1 (:f1 f1-metrics)
     :gp-train-f1 gp-train-f1
     :sib-train-f1 sib-train-f1
     :eval-count (:eval-count search-res)
     :time-ms time-ms
     :recovered-edges recovered}))

(defn sample-random-candidates
  "Samples top-k random off-diagonal edges."
  [^long n ^long top-k ^long seed ^long draw-idx]
  (let [rnd (java.util.Random. (+ (* seed 1000) draw-idx))
        all-off-diag (vec (for [i (range n) j (range n) :when (not= i j)] [i j]))
        shuffled (let [al (java.util.ArrayList. ^java.util.Collection all-off-diag)]
                   (java.util.Collections/shuffle al rnd)
                   (vec (take top-k al)))]
    (set shuffled)))

(defn run-phase1-sweep
  "Executes the Phase 1 search sweep across all 5 seeds for cells H, B0, and B1."
  [_proposer-runs p0b-results opts]
  (println "\n================================================================================")
  (println "  PHASE 1: DISCRETE CRYSTALLIZATION SEARCH SWEEP (H vs B0 vs B1)")
  (println "================================================================================")
  (let [n (long (:num-entities opts))
        all-off-diag (set (for [i (range n) j (range n) :when (not= i j)] [i j]))
        top-k (long (:top-k opts))
        b1-draws (long (:b1-draws opts))

        sweep-results
        (mapv (fn [p0b-seed-res]
                (let [seed (:seed p0b-seed-res)
                      forest (ft/generate-family-forest {:num-entities n :num-trees (:num-trees opts) :seed seed})
                      h-cands (:candidates p0b-seed-res)

                      ;; 1. Cell H: Hybrid (top-256 by GD logit Z)
                      h-res (evaluate-candidate-cell :H h-cands forest opts)
                      _ (println (format "  [Seed %d | H ] Recovered: %2d | Parent F1: %6.4f (P: %.4f, R: %.4f) | Evals: %5d | Time: %6.1f ms"
                                         seed (:recovered-count h-res) (:parent-f1 h-res) (:precision h-res) (:recall h-res) (:eval-count h-res) (:time-ms h-res)))

                      ;; 2. Cell B0: No Proposal (all 4032 off-diagonal edges)
                      b0-res (evaluate-candidate-cell :B0 all-off-diag forest opts)
                      _ (println (format "  [Seed %d | B0] Recovered: %2d | Parent F1: %6.4f (P: %.4f, R: %.4f) | Evals: %5d | Time: %6.1f ms"
                                         seed (:recovered-count b0-res) (:parent-f1 b0-res) (:precision b0-res) (:recall b0-res) (:eval-count b0-res) (:time-ms b0-res)))

                      ;; 3. Cell B1: Random Proposal (256 random edges, averaged across 3 draws)
                      b1-sub-results
                      (mapv (fn [draw-idx]
                              (let [b1-cands (sample-random-candidates n top-k seed draw-idx)]
                                (evaluate-candidate-cell :B1 b1-cands forest opts)))
                            (range b1-draws))
                      b1-mean-f1 (mean (mapv :parent-f1 b1-sub-results))
                      b1-mean-rec (mean (mapv :recovered-count b1-sub-results))
                      b1-mean-p (mean (mapv :precision b1-sub-results))
                      b1-mean-r (mean (mapv :recall b1-sub-results))
                      b1-mean-evals (long (Math/round (mean (mapv :eval-count b1-sub-results))))
                      b1-mean-time (mean (mapv :time-ms b1-sub-results))
                      _ (println (format "  [Seed %d | B1] Recovered: %4.1f | Parent F1: %6.4f (P: %.4f, R: %.4f) | Evals: %5d | Time: %6.1f ms"
                                         seed b1-mean-rec b1-mean-f1 b1-mean-p b1-mean-r b1-mean-evals b1-mean-time))]
                  {:seed seed
                   :H h-res
                   :B0 b0-res
                   :B1 {:parent-f1 b1-mean-f1
                        :precision b1-mean-p
                        :recall b1-mean-r
                        :recovered-count b1-mean-rec
                        :eval-count b1-mean-evals
                        :time-ms b1-mean-time
                        :sub-draws b1-sub-results}}))
              (:per-seed p0b-results))]
    sweep-results))

;; ==============================================================================
;; Criteria Evaluation & Reporting
;; ==============================================================================

(defn evaluate-criteria [p0a-res p0b-res sweep-results]
  (let [h-f1s (mapv #(get-in % [:H :parent-f1]) sweep-results)
        b0-f1s (mapv #(get-in % [:B0 :parent-f1]) sweep-results)
        b1-f1s (mapv #(get-in % [:B1 :parent-f1]) sweep-results)
        h-recs (mapv #(get-in % [:H :recovered-count]) sweep-results)
        h-times (mapv #(get-in % [:H :time-ms]) sweep-results)

        h-f1-mean (mean h-f1s)
        h-f1-std (std-dev h-f1s)
        b0-f1-mean (mean b0-f1s)
        b0-f1-std (std-dev b0-f1s)
        b1-f1-mean (mean b1-f1s)
        b1-f1-std (std-dev b1-f1s)

        delta-h-b1 (- h-f1-mean b1-f1-mean)

        ;; Criteria
        c1-crystallization (>= h-f1-mean 0.80)
        c2-attribution (>= delta-h-b1 0.30)
        c3-no-collapse (and (every? #(<= 28 % 112) h-recs)
                            (every? #(< % 300000.0) h-times))
        falsification-accepted? (or (not (:pass? p0b-res))
                                    (< h-f1-mean 0.80))]
    {:h-f1-mean h-f1-mean
     :h-f1-std h-f1-std
     :b0-f1-mean b0-f1-mean
     :b0-f1-std b0-f1-std
     :b1-f1-mean b1-f1-mean
     :b1-f1-std b1-f1-std
     :delta-h-b1 delta-h-b1
     :c1-crystallization c1-crystallization
     :c2-attribution c2-attribution
     :c3-no-collapse c3-no-collapse
     :p0a-pass (:pass? p0a-res)
     :p0b-pass (:pass? p0b-res)
     :falsification-accepted falsification-accepted?}))

(defn- write-summary-csv! [path sweep-results]
  (let [lines (concat
               ["seed,cell,candidates,recovered_edges,parent_f1,precision,recall,gp_train_f1,sib_train_f1,evals,time_ms"]
               (mapcat (fn [r]
                         (let [s (:seed r)
                               h (:H r)
                               b0 (:B0 r)
                               b1 (:B1 r)]
                           [(format "%d,H,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.1f"
                                    s (:candidate-count h) (:recovered-count h) (:parent-f1 h) (:precision h) (:recall h) (:gp-train-f1 h) (:sib-train-f1 h) (:eval-count h) (:time-ms h))
                            (format "%d,B0,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.1f"
                                    s (:candidate-count b0) (:recovered-count b0) (:parent-f1 b0) (:precision b0) (:recall b0) (:gp-train-f1 b0) (:sib-train-f1 b0) (:eval-count b0) (:time-ms b0))
                            (format "%d,B1,%d,%.1f,%.4f,%.4f,%.4f,N/A,N/A,%d,%.1f"
                                    s 256 (:recovered-count b1) (:parent-f1 b1) (:precision b1) (:recall b1) (:eval-count b1) (:time-ms b1))]))
                       sweep-results))]
    (io/make-parents path)
    (spit path (str (str/join "\n" lines) "\n"))
    (println (format "Saved summary CSV to %s" path))))

;; ==============================================================================
;; Main Entrypoint
;; ==============================================================================

(defn -main [& args]
  (let [opts (parse-cli-args args)
        out-dir (:out-dir opts)
        backend (:backend opts)
        ctx (if (= backend :rocm)
              (do (println "Initializing OpenXLA PJRT ROCm Backend...") (xla/init-backend! :rocm))
              (do (println "Initializing OpenXLA PJRT CPU Backend...") (xla/init-backend! :cpu)))
        seeds (:seeds opts)]

    (println "\n================================================================================")
    (println "  EXPERIMENT E19: HYBRID CRYSTALLIZATION (GD PROPOSES, SEARCH DISPOSES)")
    (println "================================================================================")
    (println (format "Target Hardware:    [%s]" (if (= backend :rocm) "AMD ROCm (RX 7900 XTX)" "Host CPU (PJRT CPU)")))
    (println (format "Dataset / Forest:   [64 entities, 8 family trees, seeds: %s]" (str/join ", " seeds)))
    (println "  Proposer Model:     [E17 A2 Full Latent Matrix, tau annealed 1.0 -> 0.05, 200 epochs]")
    (println (format "Discrete Search:    [Boolean OR-AND Semiring, Forward Lookahead + Backward Elimination, lambda=%.2f]"
                     (:lambda-search opts)))
    (println "================================================================================\n")

    ;; 1. Phase 0a: Oracle Ceiling
    (let [p0a-res (run-p0a-oracle-ceiling opts)

          ;; 2. Phase 0b: Retrain Proposers & Audit Recall@256
          _ (println "\n[Phase 0b] Retraining E17 A2 proposers across 5 seeds in OpenXLA PJRT VRAM...")
          proposer-runs
          (mapv (fn [s]
                  (print (format "  Retraining A2 [Seed %d] ... " s)) (flush)
                  (let [f (ft/generate-family-forest {:num-entities (:num-entities opts) :num-trees (:num-trees opts) :seed s})
                        run (retrain-proposer-a2 ctx f s opts)]
                    (println (format "Done (Compile: %.1f ms | Train: %.1f ms)" (:compile-ms run) (:train-ms run)))
                    run))
                seeds)
          p0b-res (run-p0b-proposer-quality proposer-runs (:top-k opts))

          phase0-data {:p0a p0a-res :p0b p0b-res}
          phase0-path (str out-dir "phase0.edn")
          _ (io/make-parents phase0-path)
          _ (spit phase0-path (pr-str phase0-data))
          _ (println (format "Saved Phase 0 telemetry to %s" phase0-path))

          ;; 3. Phase 1: Search Sweep (H vs B0 vs B1)
          sweep-results (run-phase1-sweep proposer-runs p0b-res opts)
          results-path (str out-dir "results.edn")
          csv-path (str out-dir "summary.csv")
          _ (spit results-path (pr-str sweep-results))
          _ (write-summary-csv! csv-path sweep-results)

          ;; 4. Criteria & Final Verdict
          crit (evaluate-criteria p0a-res p0b-res sweep-results)]

      (println "\n================================================================================")
      (println "  EXPERIMENT E19: FINAL SYNTHESIS & CRITERIA EVALUATION")
      (println "================================================================================")
      (println (format "  Cell H (Hybrid):         F1 = %6.4f +/- %6.4f" (:h-f1-mean crit) (:h-f1-std crit)))
      (println (format "  Cell B0 (No Proposal):   F1 = %6.4f +/- %6.4f" (:b0-f1-mean crit) (:b0-f1-std crit)))
      (println (format "  Cell B1 (Random 256):    F1 = %6.4f +/- %6.4f" (:b1-f1-mean crit) (:b1-f1-std crit)))
      (println (format "  Attribution Delta (H-B1): %+6.4f" (:delta-h-b1 crit)))
      (println "--------------------------------------------------------------------------------")
      (println (format "  Gate P0a (Oracle Ceiling F1 == 1.0):            [%s]" (if (:p0a-pass crit) "PASS" "FAIL")))
      (println (format "  Gate P0b (Proposer Recall@256 >= 90.0%%):       [%s] (Actual: %5.2f%%)"
                       (if (:p0b-pass crit) "PASS" "FAIL") (* 100.0 (double (:mean-recall p0b-res)))))
      (println (format "  Criterion 1 (Crystallization H F1 >= 0.80):     [%s] (Actual: %6.4f)"
                       (if (:c1-crystallization crit) "PASS" "FAIL") (:h-f1-mean crit)))
      (println (format "  Criterion 2 (Attribution H - B1 >= 0.30):       [%s] (Actual: %+6.4f)"
                       (if (:c2-attribution crit) "PASS" "FAIL") (:delta-h-b1 crit)))
      (println (format "  Criterion 3 (No Collapse 28 <= |L| <= 112):     [%s]"
                       (if (:c3-no-collapse crit) "PASS" "FAIL")))
      (println "--------------------------------------------------------------------------------")
      (println (format "  Falsification Clause Accepted:                  [%s]"
                       (if (:falsification-accepted crit) "YES (FALSIFIED)" "NO (HYPOTHESIS UPHELD)")))
      (println "================================================================================\n")
      (System/exit 0))))
