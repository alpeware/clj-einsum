(ns tools.e20-constrained-crystallization
  "Experiment E20: Constrained Crystallization — Symbolic Priors Make Factorization Identifiable.
   Tests whether symbolic domain priors (indegree in {0, 2} and acyclicity) resolve the
   compositional underdetermination that caused E17 and E19 to fail.
   Executes Phase 0 gates (P0a Oracle ceiling, P0b Mating ambiguity verification)
   and Phase 1 factorial search (H vs B0 vs T) across seeds 42–46."
  (:require [einsum.core :as xla]
            [einsum.logic.constrained-search :as cs]
            [einsum.logic.data.family-trees :as ft]
            [einsum.models.predicate-invention :as pi]
            [einsum.logic.predicate-crystallization :as pc]
            [clojure.java.io :as io]
            [clojure.set :as set]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT-OPTS
  {:backend :cpu
   :num-entities 64
   :num-trees 8
   :steps 200000
   :restarts 3
   :t-init 0.1
   :t-min 1e-4
   :seeds [42 43 44 45 46]
   :out-dir "catalog/trainable-limits/e20-constrained-crystallization/"})

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
          "--steps" (recur (subvec remaining 2) (assoc opts :steps (Long/parseLong v)))
          "--restarts" (recur (subvec remaining 2) (assoc opts :restarts (Long/parseLong v)))
          "--t-init" (recur (subvec remaining 2) (assoc opts :t-init (Double/parseDouble v)))
          "--t-min" (recur (subvec remaining 2) (assoc opts :t-min (Double/parseDouble v)))
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
  "P0a Gate: Constrained SA restricted to true parent-pairs as only assignable pairs.
   Requirement: F1* == 1.0."
  [opts]
  (println "\n================================================================================")
  (println "  PHASE 0a: ORACLE CEILING VALIDATION (Search Machinery Validation)")
  (println "================================================================================")
  (let [forest (ft/generate-family-forest {:num-entities (:num-entities opts)
                                           :num-trees (:num-trees opts)
                                           :seed 42})
        g-all (set (:grandparent-pairs forest))
        s-all (set (:sibling-pairs forest))
        t0 (System/nanoTime)
        res (cs/run-oracle-ceiling-sa g-all s-all forest {:steps 20000 :seed 42})
        time-ms (/ (- (System/nanoTime) t0) 1e6)
        pass? (== 1.0 (:f1-star res))]
    (println (format "  True Parents:           %d edges across %d family trees"
                     (count (:parent-pairs forest)) (:num-trees forest)))
    (println (format "  Search Result:          Recovered %d edges in %.2f ms (Best score: %.4f)"
                     (:recovered-count res) time-ms (:best-score res)))
    (println (format "  Discrete Metrics:       F1*: %.4f | Raw F1: %.4f"
                     (:f1-star res) (:raw-f1 res)))
    (println (format "  P0a Gate Verdict:       [%s] (Requirement: F1* == 1.0000)"
                     (if pass? "PASS" "FAIL")))
    (assoc res :pass? pass? :time-ms time-ms)))

;; ==============================================================================
;; Phase 0b: Mating Ambiguity Verification
;; ==============================================================================

(defn run-p0b-mating-ambiguity
  "P0b Gate: Verifies that the mating-swapped parent graph reproduces 100% of G and S
   across all 8 trees x 5 seeds, with raw F1 == 0.7000."
  [seeds opts]
  (println "\n================================================================================")
  (println "  PHASE 0b: MATING AMBIGUITY VERIFICATION (Identifiability Theory Audit)")
  (println "================================================================================")
  (let [per-seed-results
        (mapv (fn [s]
                (let [forest (ft/generate-family-forest {:num-entities (:num-entities opts)
                                                         :num-trees (:num-trees opts)
                                                         :seed s})
                      trees (:trees forest)
                      g-all (set (:grandparent-pairs forest))
                      s-all (set (:sibling-pairs forest))
                      g-train (set (get-in forest [:splits :grandparent :train]))
                      s-train (set (get-in forest [:splits :sibling :train]))
                      swapped-parents
                      (set
                       (mapcat
                        (fn [tree]
                          (let [true-p (set (:parent-pairs tree))
                                m (:mating-child tree)
                                o (:other-child tree)
                                g-kids (set (:gen2-children tree))]
                            (set (map (fn [[u v]] (if (and (= u m) (contains? g-kids v)) [o v] [u v])) true-p))))
                        trees))
                      g-swapped (pc/bool-mat-or-and-sq swapped-parents)
                      s-swapped (pc/bool-mat-trans-or-and swapped-parents)
                      f1-metrics (pc/pair-set-f1 swapped-parents (set (:parent-pairs forest)))
                      match? (and (= g-all g-swapped)
                                  (= s-all s-swapped)
                                  (set/subset? g-train g-swapped)
                                  (set/subset? s-train s-swapped)
                                  (== 0.70 (:f1 f1-metrics)))]
                  (println (format "  [Seed %d] G Match: %-5s | S Match: %-5s | Raw F1: %.4f | Status: %s"
                                   s (str (= g-all g-swapped)) (str (= s-all s-swapped))
                                   (:f1 f1-metrics) (if match? "VERIFIED" "MISMATCH")))
                  {:seed s
                   :g-match? (= g-all g-swapped)
                   :s-match? (= s-all s-swapped)
                   :raw-f1 (:f1 f1-metrics)
                   :pass? match?}))
              seeds)
        all-passed? (every? :pass? per-seed-results)]
    (println "--------------------------------------------------------------------------------")
    (println (format "  P0b Gate Verdict:       [%s] (Requirement: 100%% match across all 5 seeds)"
                     (if all-passed? "PASS" "FAIL")))
    (println "--------------------------------------------------------------------------------")
    {:pass? all-passed?
     :per-seed per-seed-results}))

;; ==============================================================================
;; Proposer Retraining for Tiebreaker Cell T
;; ==============================================================================

(defn train-proposer-z
  "Trains E17 A2 in PJRT VRAM to obtain raw logit matrix Z for tiebreaking."
  [ctx forest seed opts]
  (let [n (long (:num-entities opts))
        step-opts {:num-entities n
                   :num-trees (:num-trees opts)
                   :dim 64
                   :rank 8
                   :ablation :a2
                   :seed seed
                   :lr 0.05
                   :lr-z 0.25}
        exec (pi/compile-predicate-invention-step ctx step-opts)
        initial-state (pi/init-parameters forest step-opts)
        final-state
        (loop [epoch 0
               state initial-state]
          (if (>= epoch 200)
            state
            (let [progress (/ (double epoch) 199.0)
                  tau (* 1.0 (Math/pow (/ 0.05 1.0) progress))
                  batch (pi/make-step-batch forest state {:tau tau :lambda-l1 1e-4})
                  step-res (pi/run-predicate-invention-step! exec batch)]
              (recur (inc epoch)
                     (assoc state
                            :E (:E_new step-res)
                            :W_gp (:W_gp_new step-res)
                            :W_sib (:W_sib_new step-res)
                            :Z (:Z_new step-res))))))]
    (:Z final-state)))

;; ==============================================================================
;; Phase 1 Factorial Sweep: Cells H, B0, T
;; ==============================================================================

(defn run-phase1-sweep
  "Runs Phase 1 factorial evaluation across all seeds."
  [ctx seeds opts]
  (println "\n================================================================================")
  (println "  PHASE 1: CONSTRAINED CRYSTALLIZATION SEARCH SWEEP (H vs B0 vs T)")
  (println "================================================================================")
  (let [n (long (:num-entities opts))
        steps (long (:steps opts))
        restarts (long (:restarts opts))
        all-off-diag (set (for [i (range n) j (range n) :when (not= i j)] [i j]))]
    (mapv (fn [s]
            (let [forest (ft/generate-family-forest {:num-entities n
                                                     :num-trees (:num-trees opts)
                                                     :seed s})
                  true-parents (set (:parent-pairs forest))
                  g-train (set (get-in forest [:splits :grandparent :train]))
                  s-train (set (get-in forest [:splits :sibling :train]))

                  ;; 1. Cell H: Constrained SA (Indegree in {0,2} + Acyclicity)
                  sa-h (cs/run-constrained-sa g-train s-train
                                              {:num-entities n
                                               :steps steps
                                               :restarts restarts
                                               :t-init (:t-init opts)
                                               :t-min (:t-min opts)
                                               :seed s})
                  h-edges (cs/state->edges (:best-state sa-h))
                  h-non-roots (count (filter some? (:best-state sa-h)))
                  h-f1-star (cs/compute-f1-star h-edges forest)
                  h-f1-raw (:f1 (pc/pair-set-f1 h-edges true-parents))
                  h-p (:precision (pc/pair-set-f1 h-edges true-parents))
                  h-r (:recall (pc/pair-set-f1 h-edges true-parents))
                  _ (println (format "  [Seed %d | H ] F1*: %6.4f | Raw F1: %6.4f (P: %.4f, R: %.4f) | Edges: %2d (Non-roots: %2d) | Score: %.4f | Time: %6.1f ms"
                                     s h-f1-star h-f1-raw h-p h-r (count h-edges) h-non-roots (:best-score sa-h) (:time-ms sa-h)))

                  ;; 2. Cell B0: Unconstrained Edge-Greedy Search (Control)
                  t0-b0 (System/nanoTime)
                  b0-res (pc/discrete-crystallization-search all-off-diag g-train s-train {:lambda 0.01})
                  b0-time (/ (- (System/nanoTime) t0-b0) 1e6)
                  b0-edges (:edges b0-res)
                  b0-f1-star (cs/compute-f1-star b0-edges forest)
                  b0-f1-raw (:f1 (pc/pair-set-f1 b0-edges true-parents))
                  b0-p (:precision (pc/pair-set-f1 b0-edges true-parents))
                  b0-r (:recall (pc/pair-set-f1 b0-edges true-parents))
                  _ (println (format "  [Seed %d | B0] F1*: %6.4f | Raw F1: %6.4f (P: %.4f, R: %.4f) | Edges: %2d | Score: %.4f | Time: %6.1f ms"
                                     s b0-f1-star b0-f1-raw b0-p b0-r (count b0-edges) (:score b0-res) b0-time))

                  ;; 3. Cell T: Constrained SA with GD Logit Z Tiebreaker
                  z-arr (train-proposer-z ctx forest s opts)
                  sa-t (cs/run-constrained-sa g-train s-train
                                              {:num-entities n
                                               :steps steps
                                               :restarts restarts
                                               :t-init (:t-init opts)
                                               :t-min (:t-min opts)
                                               :seed s
                                               :tiebreaker-z z-arr})
                  t-edges (cs/state->edges (:best-state sa-t))
                  t-non-roots (count (filter some? (:best-state sa-t)))
                  t-f1-star (cs/compute-f1-star t-edges forest)
                  t-f1-raw (:f1 (pc/pair-set-f1 t-edges true-parents))
                  t-p (:precision (pc/pair-set-f1 t-edges true-parents))
                  t-r (:recall (pc/pair-set-f1 t-edges true-parents))
                  _ (println (format "  [Seed %d | T ] F1*: %6.4f | Raw F1: %6.4f (P: %.4f, R: %.4f) | Edges: %2d (Non-roots: %2d) | Score: %.4f | Time: %6.1f ms"
                                     s t-f1-star t-f1-raw t-p t-r (count t-edges) t-non-roots (:best-score sa-t) (:time-ms sa-t)))]

              {:seed s
               :H {:f1-star h-f1-star :raw-f1 h-f1-raw :precision h-p :recall h-r
                   :edges (count h-edges) :non-roots h-non-roots :score (:best-score sa-h) :time-ms (:time-ms sa-h)}
               :B0 {:f1-star b0-f1-star :raw-f1 b0-f1-raw :precision b0-p :recall b0-r
                    :edges (count b0-edges) :non-roots (count b0-edges) :score (:score b0-res) :time-ms b0-time}
               :T {:f1-star t-f1-star :raw-f1 t-f1-raw :precision t-p :recall t-r
                   :edges (count t-edges) :non-roots t-non-roots :score (:best-score sa-t) :time-ms (:time-ms sa-t)}}))
          seeds)))

;; ==============================================================================
;; Criteria Evaluation & Reporting
;; ==============================================================================

(defn evaluate-criteria [p0a-res p0b-res sweep-results]
  (let [h-f1-stars (mapv #(get-in % [:H :f1-star]) sweep-results)
        h-raw-f1s (mapv #(get-in % [:H :raw-f1]) sweep-results)
        h-non-roots (mapv #(get-in % [:H :non-roots]) sweep-results)
        h-times (mapv #(get-in % [:H :time-ms]) sweep-results)

        b0-f1-stars (mapv #(get-in % [:B0 :f1-star]) sweep-results)
        t-f1-stars (mapv #(get-in % [:T :f1-star]) sweep-results)

        h-star-mean (mean h-f1-stars)
        h-star-std (std-dev h-f1-stars)
        h-raw-mean (mean h-raw-f1s)
        h-raw-std (std-dev h-raw-f1s)

        b0-star-mean (mean b0-f1-stars)
        b0-star-std (std-dev b0-f1-stars)
        t-star-mean (mean t-f1-stars)
        t-star-std (std-dev t-f1-stars)

        delta-h-b0 (- h-star-mean b0-star-mean)

        c1-identifiability (>= h-star-mean 0.80)
        c2-attribution (>= delta-h-b0 0.50)
        c3-ambiguity-confirmation (<= 0.60 h-raw-mean 0.75)
        c4-no-collapse (and (every? #(<= 16 % 80) h-non-roots)
                            (every? #(< % 600000.0) h-times))
        falsification-accepted? (or (not (:pass? p0b-res))
                                    (< h-star-mean 0.80))]
    {:h-star-mean h-star-mean
     :h-star-std h-star-std
     :h-raw-mean h-raw-mean
     :h-raw-std h-raw-std
     :b0-star-mean b0-star-mean
     :b0-star-std b0-star-std
     :t-star-mean t-star-mean
     :t-star-std t-star-std
     :delta-h-b0 delta-h-b0
     :c1-identifiability c1-identifiability
     :c2-attribution c2-attribution
     :c3-ambiguity-confirmation c3-ambiguity-confirmation
     :c4-no-collapse c4-no-collapse
     :p0a-pass (:pass? p0a-res)
     :p0b-pass (:pass? p0b-res)
     :falsification-accepted falsification-accepted?}))

(defn- write-summary-csv! [path sweep-results]
  (let [lines (concat
               ["seed,cell,f1_star,raw_f1,precision,recall,recovered_edges,non_roots,score,time_ms"]
               (mapcat (fn [r]
                         (let [s (:seed r)
                               h (:H r)
                               b0 (:B0 r)
                               t (:T r)]
                           [(format "%d,H,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.1f"
                                    s (:f1-star h) (:raw-f1 h) (:precision h) (:recall h) (:edges h) (:non-roots h) (:score h) (:time-ms h))
                            (format "%d,B0,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.1f"
                                    s (:f1-star b0) (:raw-f1 b0) (:precision b0) (:recall b0) (:edges b0) (:non-roots b0) (:score b0) (:time-ms b0))
                            (format "%d,T,%.4f,%.4f,%.4f,%.4f,%d,%d,%.4f,%.1f"
                                    s (:f1-star t) (:raw-f1 t) (:precision t) (:recall t) (:edges t) (:non-roots t) (:score t) (:time-ms t))]))
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
    (println "  EXPERIMENT E20: CONSTRAINED CRYSTALLIZATION (SYMBOLIC DOMAIN PRIORS)")
    (println "================================================================================")
    (println (format "Target Hardware:    [%s]" (if (= backend :rocm) "AMD ROCm (RX 7900 XTX)" "Host CPU (PJRT CPU)")))
    (println (format "Dataset / Forest:   [64 entities, 8 family trees, seeds: %s]" (str/join ", " seeds)))
    (println "Structural Priors:  [Indegree in {0, 2} + Strict Acyclicity (DAG)]")
    (println (format "Search Algorithm:   [Pair-Assignment Simulated Annealing, %d steps x %d restarts, T=%.2f -> %.1e]"
                     (:steps opts) (:restarts opts) (:t-init opts) (:t-min opts)))
    (println "================================================================================\n")

    ;; 1. Phase 0a: Oracle Ceiling
    (let [p0a-res (run-p0a-oracle-ceiling opts)

          ;; 2. Phase 0b: Mating Ambiguity Verification
          p0b-res (run-p0b-mating-ambiguity seeds opts)

          phase0-data {:p0a p0a-res :p0b p0b-res}
          phase0-path (str out-dir "phase0.edn")
          _ (io/make-parents phase0-path)
          _ (spit phase0-path (pr-str phase0-data))
          _ (println (format "Saved Phase 0 telemetry to %s" phase0-path))

          ;; 3. Phase 1: Search Sweep (H vs B0 vs T)
          sweep-results (run-phase1-sweep ctx seeds opts)
          results-path (str out-dir "results.edn")
          csv-path (str out-dir "summary.csv")
          _ (spit results-path (pr-str sweep-results))
          _ (write-summary-csv! csv-path sweep-results)

          ;; 4. Criteria & Final Synthesis
          crit (evaluate-criteria p0a-res p0b-res sweep-results)]

      (println "\n================================================================================")
      (println "  EXPERIMENT E20: FINAL SYNTHESIS & CRITERIA EVALUATION")
      (println "================================================================================")
      (println (format "  Cell H (Constrained SA):     F1* = %6.4f +/- %6.4f | Raw F1 = %6.4f +/- %6.4f"
                       (:h-star-mean crit) (:h-star-std crit) (:h-raw-mean crit) (:h-raw-std crit)))
      (println (format "  Cell B0 (Unconstrained Edge): F1* = %6.4f +/- %6.4f"
                       (:b0-star-mean crit) (:b0-star-std crit)))
      (println (format "  Cell T (GD Logit Tiebreaker): F1* = %6.4f +/- %6.4f"
                       (:t-star-mean crit) (:t-star-std crit)))
      (println (format "  Attribution Delta (H - B0):   %+6.4f" (:delta-h-b0 crit)))
      (println "--------------------------------------------------------------------------------")
      (println (format "  Gate P0a (Oracle Ceiling F1* == 1.0):          [%s]"
                       (if (:p0a-pass crit) "PASS" "FAIL")))
      (println (format "  Gate P0b (Mating Ambiguity Verification):      [%s]"
                       (if (:p0b-pass crit) "PASS" "FAIL")))
      (println (format "  Criterion 1 (Identifiability H F1* >= 0.80):   [%s] (Actual: %6.4f)"
                       (if (:c1-identifiability crit) "PASS" "FAIL") (:h-star-mean crit)))
      (println (format "  Criterion 2 (Attribution H - B0 >= 0.50):      [%s] (Actual: %+6.4f)"
                       (if (:c2-attribution crit) "PASS" "FAIL") (:delta-h-b0 crit)))
      (println (format "  Criterion 3 (Ambiguity Confirmed Raw in [.60,.75]): [%s] (Actual: %6.4f)"
                       (if (:c3-ambiguity-confirmation crit) "PASS" "FAIL") (:h-raw-mean crit)))
      (println (format "  Criterion 4 (No Collapse 16 <= non-roots <= 80): [%s]"
                       (if (:c4-no-collapse crit) "PASS" "FAIL")))
      (println "--------------------------------------------------------------------------------")
      (println (format "  Falsification Clause Accepted:                 [%s]"
                       (if (:falsification-accepted crit) "YES (FALSIFIED)" "NO (HYPOTHESIS UPHELD)")))
      (println "================================================================================\n")
      (System/exit 0))))
