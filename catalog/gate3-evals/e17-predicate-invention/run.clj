(ns tools.poc-predicate-invention
  "Experiment E17 Driver: Gradient-Based Predicate Invention with Held-Out Evaluation and Discrete Recovery.
   Decides the fundamental question: Domingos-2020 (kernel machines / representation learning bottleneck)
   vs Domingos-2025 (emergent predicate invention via gradient descent over tensor equations).
   Trains in OpenXLA PJRT VRAM with analytical adjoints, temperature annealing, and discrete F1 evaluation."
  (:require [einsum.core :as xla]
            [einsum.logic.data.family-trees :as ft]
            [einsum.models.predicate-invention :as pi]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :num-entities 64
   :num-trees 8
   :dim 64
   :rank 8
   :epochs 200
   :lr 0.05
   :lr-z 0.25
   :z-init-mean 0.0
   :tau-init 1.0
   :tau-min 0.05
   :tau-ce 0.1
   :lambda-l1 1e-4
   :seed 42
   :ablation :a2
   :data-fraction 1.0
   :out-dir "catalog/trainable-limits/e17-predicate-invention/"})

(defn- normalize-args [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT_OPTS]
    (if (empty? remaining)
      opts
      (let [k (first remaining)
            v (second remaining)]
        (case k
          "--backend" (recur (subvec remaining 2) (assoc opts :backend (keyword v)))
          "--num-entities" (recur (subvec remaining 2) (assoc opts :num-entities (Long/parseLong v)))
          "--num-trees" (recur (subvec remaining 2) (assoc opts :num-trees (Long/parseLong v)))
          "--dim" (recur (subvec remaining 2) (assoc opts :dim (Long/parseLong v)))
          "--rank" (recur (subvec remaining 2) (assoc opts :rank (Long/parseLong v)))
          "--latent-rank" (recur (subvec remaining 2) (assoc opts :rank (Long/parseLong v)))
          "--epochs" (recur (subvec remaining 2) (assoc opts :epochs (Long/parseLong v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--lr-z" (recur (subvec remaining 2) (assoc opts :lr-z (Double/parseDouble v)))
          "--z-init-mean" (recur (subvec remaining 2) (assoc opts :z-init-mean (Double/parseDouble v)))
          "--tau-init" (recur (subvec remaining 2) (assoc opts :tau-init (Double/parseDouble v)))
          "--tau-min" (recur (subvec remaining 2) (assoc opts :tau-min (Double/parseDouble v)))
          "--tau-ce" (recur (subvec remaining 2) (assoc opts :tau-ce (Double/parseDouble v)))
          "--lambda-l1" (recur (subvec remaining 2) (assoc opts :lambda-l1 (Double/parseDouble v)))
          "--seed" (recur (subvec remaining 2) (assoc opts :seed (Long/parseLong v)))
          "--ablation" (recur (subvec remaining 2) (assoc opts :ablation (keyword v)))
          "--data-fraction" (recur (subvec remaining 2) (assoc opts :data-fraction (Double/parseDouble v)))
          "--out-dir" (recur (subvec remaining 2) (assoc opts :out-dir v))
          (recur (subvec remaining 1) opts))))))

(defn- subsample-forest [forest frac seed]
  (if (>= frac 0.999)
    forest
    (let [rnd (java.util.Random. (long seed))
          subsample (fn [pairs]
                      (let [shuffled (vec (doto (java.util.ArrayList. ^java.util.Collection pairs)
                                            (java.util.Collections/shuffle rnd)))
                            n-sub (max 1 (long (Math/round (* (double frac) (double (count pairs))))))]
                        (vec (take n-sub shuffled))))
          train-gp (get-in forest [:splits :grandparent :train])
          train-sib (get-in forest [:splits :sibling :train])]
      (-> forest
          (assoc-in [:splits :grandparent :train] (subsample train-gp))
          (assoc-in [:splits :sibling :train] (subsample train-sib))))))

(defn run-single-run [ctx user-opts]
  (let [opts (merge DEFAULT_OPTS user-opts)
        n (long (:num-entities opts))
        num-trees (long (:num-trees opts))
        seed (long (:seed opts))
        epochs (long (:epochs opts))
        ablation (:ablation opts)
        tau-init (double (:tau-init opts))
        tau-min (double (:tau-min opts))
        lambda-l1 (double (:lambda-l1 opts))
        frac (double (:data-fraction opts))

        base-forest (ft/generate-family-forest {:num-entities n :num-trees num-trees :seed seed})
        forest (subsample-forest base-forest frac seed)

        t0-compile (System/nanoTime)
        exec (pi/compile-predicate-invention-step ctx opts)
        compile-ms (/ (- (System/nanoTime) t0-compile) 1e6)

        initial-state (pi/init-parameters forest opts)
        t0-train (System/nanoTime)

        train-res
        (loop [epoch 0
               state initial-state
               last-step-res nil]
          (if (>= epoch epochs)
            {:final-state state :last-step-res last-step-res}
            (let [progress (/ (double epoch) (double (max 1 (dec epochs))))
                  ;; Geometric temperature annealing tau_init -> tau_min
                  tau (* tau-init (Math/pow (/ tau-min tau-init) progress))
                  batch (pi/make-step-batch forest state {:tau tau :lambda-l1 lambda-l1})
                  step-res (pi/run-predicate-invention-step! exec batch)
                  new-state (cond-> (assoc state
                                           :E (:E_new step-res)
                                           :W_gp (:W_gp_new step-res)
                                           :W_sib (:W_sib_new step-res))
                              (contains? step-res :Z_new)
                              (assoc :Z (:Z_new step-res))

                              (and (= ablation :a3) (contains? step-res :A_new))
                              (assoc :A (:A_new step-res) :B (:B_new step-res) :Z (:Z step-res)))]
              (recur (inc epoch) new-state step-res))))

        train-ms (/ (- (System/nanoTime) t0-train) 1e6)
        mean-step-ms (/ train-ms (double epochs))
        last-res (:last-step-res train-res)
        scores-gp ^floats (:Scores_gp last-res)
        scores-sib ^floats (:Scores_sib last-res)
        l-mat ^floats (or (:L last-res) (float-array (* n n)))

        ;; Evaluation
        held-out-gp (get-in forest [:splits :grandparent :held-out])
        held-out-sib (get-in forest [:splits :sibling :held-out])
        gp-eval (pi/eval-link-prediction scores-gp held-out-gp n)
        sib-eval (pi/eval-link-prediction scores-sib held-out-sib n)

        flat-parent (float-array (flatten (:parent-matrix forest)))
        f1-discrete (pi/compute-f1-metrics l-mat flat-parent 0.5)
        closure-val (pi/compute-two-hop-closure l-mat (:grandparent-pairs forest) n 0.5)]

    {:ablation ablation
     :seed seed
     :data-fraction frac
     :compile-ms compile-ms
     :train-ms train-ms
     :mean-step-ms mean-step-ms
     :gp-eval gp-eval
     :sib-eval sib-eval
     :discrete-recovery f1-discrete
     :two-hop-closure closure-val
     :final-L l-mat
     :scores-gp scores-gp
     :scores-sib scores-sib
     :forest forest}))

(defn- mean [coll]
  (if (empty? coll) 0.0 (/ (reduce + (map double coll)) (double (count coll)))))

(defn- std-dev [coll]
  (if (< (count coll) 2)
    0.0
    (let [m (mean coll)
          var (/ (reduce + (map #(Math/pow (- (double %) m) 2.0) coll)) (double (dec (count coll))))]
      (Math/sqrt var))))

(defn run-experiment [opts]
  (println "================================================================================")
  (println "  EXPERIMENT E17: GRADIENT-BASED PREDICATE INVENTION (PJRT)")
  (println "================================================================================")
  (println (format "Target Hardware:       [%s]" (if (= (:backend opts) :rocm)
                                                   "AMD Radeon RX 7900 XTX (ROCm)"
                                                   "Host CPU (OpenXLA PJRT CPU)")))
  (println (format "Universe:              [%d entities, %d family trees, Dimension d=%d]"
                   (:num-entities opts) (:num-trees opts) (:dim opts)))
  (println (format "Ablation Mode:         [%s] | Data Fraction: [%.1f%%]"
                   (name (:ablation opts)) (* 100.0 (double (:data-fraction opts)))))
  (println (format "Training Horizon:      [%d epochs | Annealing tau: %.2f -> %.2f | lr: %.3f]"
                   (:epochs opts) (:tau-init opts) (:tau-min opts) (:lr opts)))
  (println "================================================================================\n")

  (let [backend (:backend opts)
        ctx (if (= backend :rocm)
              (do (println "Initializing PJRT ROCm Backend...") (xla/init-backend! :rocm))
              (do (println "Initializing PJRT CPU Backend...") (xla/init-backend! :cpu)))
        mode (:ablation opts)]

    (if (#{:a0 :a1 :a2 :a3} mode)
      ;; Single ablation run
      (let [res (run-single-run ctx opts)
            gp (:gp-eval res)
            f1 (:discrete-recovery res)]
        (println "--------------------------------------------------------------------------------")
        (println (format "Run Complete: [%s | Seed %d]" (name mode) (:seed opts)))
        (println (format "  Compilation Latency:      %.2f ms" (:compile-ms res)))
        (println (format "  Total Training Time:      %.2f ms (%.3f ms/epoch)" (:train-ms res) (:mean-step-ms res)))
        (println (format "  Held-Out GP Hits@1:       %6.2f%% (MRR: %.4f, Mean Rank: %.2f)"
                         (* 100.0 (:hits-1 gp)) (:mrr gp) (:mean-rank gp)))
        (println (format "  Held-Out GP Hits@10:      %6.2f%%" (* 100.0 (:hits-10 gp))))
        (println (format "  Discrete Recovery F1:     %6.4f (Precision: %.4f, Recall: %.4f)"
                         (:f1 f1) (:precision f1) (:recall f1)))
        (println (format "  Two-Hop Ground Closure:   %6.2f%%" (* 100.0 (double (:two-hop-closure res)))))
        (println "--------------------------------------------------------------------------------\n")
        res)

      ;; Full ablation sweep (:sweep)
      (let [seeds [42 43 44 45 46]
            ablations [:a0 :a1 :a2 :a3]
            _ (println (format "Starting full pre-registered ablation sweep across %d seeds (%s)..."
                               (count seeds) (str/join ", " seeds)))
            results-by-ablation
            (into {}
                  (map (fn [abl]
                         (println (format "\n>>> Executing Ablation [%s] across %d seeds..." (name abl) (count seeds)))
                         (let [runs (mapv (fn [s]
                                            (print (format "  [Seed %d] ... " s)) (flush)
                                            (let [r (run-single-run ctx (assoc opts :ablation abl :seed s))]
                                              (println (format "Hits@1: %5.1f%% | F1: %5.3f | Step: %.2f ms"
                                                               (* 100.0 (get-in r [:gp-eval :hits-1]))
                                                               (get-in r [:discrete-recovery :f1])
                                                               (:mean-step-ms r)))
                                              r))
                                          seeds)]
                           [abl runs]))
                       ablations))

            ;; Aggregate statistics
            summary-stats
            (mapv (fn [abl]
                    (let [runs (get results-by-ablation abl)
                          h1-list (mapv #(get-in % [:gp-eval :hits-1]) runs)
                          mrr-list (mapv #(get-in % [:gp-eval :mrr]) runs)
                          f1-list (mapv #(get-in % [:discrete-recovery :f1]) runs)
                          closure-list (mapv #(:two-hop-closure %) runs)
                          step-ms-list (mapv #(:mean-step-ms %) runs)]
                      {:ablation abl
                       :h1-mean (mean h1-list) :h1-std (std-dev h1-list)
                       :mrr-mean (mean mrr-list) :mrr-std (std-dev mrr-list)
                       :f1-mean (mean f1-list) :f1-std (std-dev f1-list)
                       :closure-mean (mean closure-list) :closure-std (std-dev closure-list)
                       :step-ms-mean (mean step-ms-list)}))
                  ablations)

            ;; Paired deltas A2 over A0
            a0-runs (get results-by-ablation :a0)
            a2-runs (get results-by-ablation :a2)
            a1-runs (get results-by-ablation :a1)
            paired-deltas
            (mapv (fn [i]
                    (let [h1-a0 (get-in (nth a0-runs i) [:gp-eval :hits-1])
                          h1-a2 (get-in (nth a2-runs i) [:gp-eval :hits-1])
                          h1-a1 (get-in (nth a1-runs i) [:gp-eval :hits-1])
                          delta-a2 (- h1-a2 h1-a0)
                          delta-a1 (- h1-a1 h1-a0)]
                      {:seed (nth seeds i)
                       :a0 h1-a0
                       :a1 h1-a1
                       :a2 h1-a2
                       :delta-a2 delta-a2
                       :delta-a1 delta-a1}))
                  (range (count seeds)))

            mean-delta-a2 (mean (map :delta-a2 paired-deltas))
            mean-delta-a1 (mean (map :delta-a1 paired-deltas))
            positive-delta-count (count (filter #(pos? (:delta-a2 %)) paired-deltas))
            a2-f1-mean (:f1-mean (first (filter #(= (:ablation %) :a2) summary-stats)))
            a3-f1-mean (:f1-mean (first (filter #(= (:ablation %) :a3) summary-stats)))

            c1-pass? (and (>= mean-delta-a2 0.15) (>= positive-delta-count 4))
            c2-pass? (>= a2-f1-mean 0.80)
            c3-pass? (if (pos? mean-delta-a2) (< mean-delta-a1 (* 0.5 mean-delta-a2)) false)
            c4-pass? (>= a2-f1-mean a3-f1-mean)]

        (println "\n================================================================================")
        (println "  EXPERIMENT E17: PRE-REGISTERED SUMMARY MATRIX (5 SEEDS)")
        (println "================================================================================")
        (println " Ablation | Description        | Held-Out Hits@1   | MRR               | Discrete F1       | Closure")
        (println "----------+--------------------+-------------------+-------------------+-------------------+----------")
        (doseq [{:keys [ablation h1-mean h1-std mrr-mean mrr-std f1-mean f1-std closure-mean]} summary-stats]
          (let [desc (case ablation
                       :a0 "Rank-0 Baseline"
                       :a1 "Frozen Core"
                       :a2 "Full E17 Invention"
                       :a3 "Low-Rank (Tucker)")]
            (println (format "   %-4s   | %-18s | %5.1f%% +/- %4.1f%% | %6.4f +/- %.3f | %6.4f +/- %.3f |  %5.1f%%"
                             (name ablation) desc
                             (* 100.0 h1-mean) (* 100.0 h1-std)
                             mrr-mean mrr-std
                             f1-mean f1-std
                             (* 100.0 closure-mean)))))
        (println "--------------------------------------------------------------------------------")
        (println "\nPAIRED PER-SEED DELTAS (A2 over A0):")
        (doseq [{:keys [seed a0 a2 delta-a2]} paired-deltas]
          (println (format "  Seed %d: A0 = %5.1f%% | A2 = %5.1f%% | Delta = %+6.2f%%"
                           seed (* 100.0 a0) (* 100.0 a2) (* 100.0 delta-a2))))
        (println (format "  Mean Delta: %+6.2f%% | Positive Seeds: %d/%d"
                         (* 100.0 mean-delta-a2) positive-delta-count (count seeds)))

        (println "\nPRE-REGISTERED SUCCESS CRITERIA CHECK:")
        (println (format "  1. Predictive Lift (Delta Hits@1 >= +0.15, >= 4/5 seeds): %s (Delta: %+5.1f%%, %d/%d)"
                         (if c1-pass? "PASSED [YES]" "FAILED [NO]")
                         (* 100.0 mean-delta-a2) positive-delta-count (count seeds)))
        (println (format "  2. Discrete Recovery (F1 >= 0.80 after annealing)       : %s (F1: %.4f)"
                         (if c2-pass? "PASSED [YES]" "FAILED [NO]") a2-f1-mean))
        (println (format "  3. Learning, not Capacity (A1 Lift < 50%% of A2 Lift)   : %s (A1: %+5.1f%% vs A2: %+5.1f%%)"
                         (if c3-pass? "PASSED [YES]" "FAILED [NO]")
                         (* 100.0 mean-delta-a1) (* 100.0 mean-delta-a2)))
        (println (format "  4. Rank Sanity (Full-Matrix A2 >= Low-Rank A3 on F1)    : %s (A2: %.4f vs A3: %.4f)"
                         (if c4-pass? "PASSED [YES]" "FAILED [NO]") a2-f1-mean a3-f1-mean))
        (println "================================================================================\n")

        ;; Save artifacts
        (let [out-dir (io/file (:out-dir opts))
              _ (.mkdirs out-dir)
              edn-file (io/file out-dir "sweep_results.edn")
              csv-file (io/file out-dir "summary.csv")]
          (spit edn-file (pr-str {:opts opts
                                  :summary summary-stats
                                  :paired-deltas paired-deltas
                                  :criteria {:c1-predictive-lift c1-pass?
                                             :c2-discrete-recovery c2-pass?
                                             :c3-learning-not-capacity c3-pass?
                                             :c4-rank-sanity c4-pass?}}))
          (spit csv-file
                (str "ablation,h1_mean,h1_std,mrr_mean,mrr_std,f1_mean,f1_std,closure_mean,step_ms_mean\n"
                     (str/join "\n"
                               (map (fn [{:keys [ablation h1-mean h1-std mrr-mean mrr-std f1-mean f1-std closure-mean step-ms-mean]}]
                                      (format "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
                                              (name ablation) h1-mean h1-std mrr-mean mrr-std f1-mean f1-std closure-mean step-ms-mean))
                                    summary-stats))))
          (println (format "Results saved cleanly to %s and %s" (.getPath edn-file) (.getPath csv-file))))

        {:summary summary-stats
         :paired-deltas paired-deltas
         :criteria {:c1 c1-pass? :c2 c2-pass? :c3 c3-pass? :c4 c4-pass?}}))))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-experiment opts)
    (shutdown-agents)
    (System/exit 0)))
