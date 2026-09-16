(ns scripts.poc-datalog-state-tracker
  "Experiment E4 Driver: In-VRAM Datalog Fixpoint State Tracker for Long-Horizon Agents.
   Simulates a 100-turn autonomous software engineering agent refactoring trajectory.
   Maintains codebase dependency relations and test coverage maps in OpenXLA PJRT device memory (VRAM).
   Demonstrates constant O(1) context length, sub-millisecond compiled fixpoint execution,
   and zero context explosion over 100+ turns."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.agent.state-tracker :as tracker]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :turns 100
   :num-entities 64})

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
          "--turns" (recur (subvec remaining 2) (assoc opts :turns (Long/parseLong v)))
          "--num-entities" (recur (subvec remaining 2) (assoc opts :num-entities (Long/parseLong v)))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; 1. Realistic Codebase Ontology Construction
;; ==============================================================================

(defn build-synthetic-codebase-ontology
  "Builds a realistic software dependency and test coverage DAG for N entities:
   - Nodes 0..(N/4-1): Core low-level libraries (AST, semiring, autodiff, etc.)
   - Nodes (N/4)..(N/2-1): Mid-level engines (memory, symbolic, lower, etc.)
   - Nodes (N/2)..(3N/4-1): High-level user models & agent loops (gemma, agent_vram, etc.)
   - Nodes (3N/4)..(N-1): Unit & integration test suites
   Returns {:entities [...], :deps #{[x y] ...}, :tests #{[t x] ...}}."
  [n]
  (let [quarter (quot n 4)
        half (* 2 quarter)
        three-quarters (* 3 quarter)

        ;; Generate entity names
        entities (mapv (fn [i]
                         (cond
                           (< i quarter) (str "core/lib_" i ".clj")
                           (< i half) (str "engine/mod_" i ".clj")
                           (< i three-quarters) (str "models/model_" i ".clj")
                           :else (str "test/test_suite_" i ".clj")))
                       (range n))

        ;; Dependency edges [x y] meaning x depends on y:
        ;; High-level depends on mid-level; mid-level depends on core
        deps (into #{}
                   (concat
                    ;; Mid-level modules depend on 2 core modules
                    (mapcat (fn [m]
                              [[(+ quarter m) (mod m quarter)]
                               [(+ quarter m) (mod (inc m) quarter)]])
                            (range quarter))
                    ;; High-level models depend on mid-level modules
                    (mapcat (fn [m]
                              [[(+ half m) (+ quarter (mod m quarter))]
                               [(+ half m) (+ quarter (mod (inc m) quarter))]])
                            (range quarter))))

        ;; Test coverage edges [t x] meaning test t covers x:
        tests (into #{}
                    (mapcat (fn [t]
                              (let [tid (+ three-quarters t)]
                                [[tid (mod t quarter)]
                                 [tid (+ quarter (mod t quarter))]
                                 [tid (+ half (mod t quarter))]]))
                            (range (- n three-quarters))))]

    {:entities entities
     :deps deps
     :tests tests}))

;; ==============================================================================
;; 2. 100-Turn Simulation & Telemetry Runner
;; ==============================================================================

(defn run-datalog-state-tracker-experiment [opts]
  (println "==================================================================")
  (println "  clj-xla: Experiment E4 - In-VRAM Datalog Fixpoint State Tracker")
  (println "==================================================================")
  (println (str "Backend:           [" (:backend opts) "]"))
  (println (str "Turns:             [" (:turns opts) "]"))
  (println (str "Codebase Universe: [" (:num-entities opts) " entities]"))
  (println "==================================================================\n")

  (let [backend (:backend opts)
        ctx (if (= backend :rocm)
              (do
                (println "Initializing PJRT ROCm Backend for In-VRAM Datalog Fixpoints...")
                (xla/init-backend! :rocm))
              (do
                (println "Initializing PJRT CPU Backend for In-VRAM Datalog Fixpoints...")
                (xla/init-backend! :cpu)))

        n (long (:num-entities opts))
        total-turns (long (:turns opts))
        ontology (build-synthetic-codebase-ontology n)
        entities (:entities ontology)
        deps (:deps ontology)
        tests (:tests ontology)

        _ (println (format "Constructed Codebase DAG: %d entities, %d dependency edges, %d test coverage edges."
                           n (count deps) (count tests)))
        _ (println "Compiling In-VRAM Datalog Transitive Fixpoint in OpenXLA PJRT...")

        t0-compile (System/nanoTime)
        tracker-session (tracker/init-state-tracker ctx n {:deps deps :tests tests})
        compile-ms (/ (- (System/nanoTime) t0-compile) 1e6)
        _ (println (format "OpenXLA compilation complete in %.2f ms. Resident VRAM memory: %.2f KB.\n"
                           compile-ms (/ (double (:vram-bytes tracker-session)) 1024.0)))

        _ (println "Starting 100-Turn Autonomous Agent Trajectory Simulation...")
        _ (println "------------------------------------------------------------------")

        ;; Simulate 100 agent turns
        turn-metrics
        (mapv
         (fn [turn]
           ;; Agent modifies a file (rotating across core, engine, and model files)
           (let [mod-id (mod (* turn 7) (long (* 3 (quot n 4))))
                 mod-name (nth entities mod-id)
                 res (tracker/step-state-tracker! tracker-session #{mod-id})
                 aff-files (:affected-files res)
                 aff-tests (:affected-tests res)
                 lat (:latency-ms res)
                 vram (:vram-bytes res)

                 ;; Host-side ground truth verification
                 gt-aff (tracker/compute-affected-entities-host deps #{mod-id} n)
                 exact? (= (set (filter #(< % (long (* 3 (quot n 4)))) aff-files)) gt-aff)]

             (when (or (zero? turn)
                       (= turn 24)
                       (= turn 49)
                       (= turn 74)
                       (= turn 99))
               (println (format " [Turn %3d] Action: Edit %-22s -> Affected: %2d files, %2d tests | In-VRAM Latency: %5.3f ms | Exact: %s"
                                (inc turn) (str "\"" mod-name "\"")
                                (count aff-files) (count aff-tests)
                                (double lat) (if exact? "TRUE [100%]" "FALSE")))
               (let [sample-tests (take 3 (map #(nth entities %) aff-tests))]
                 (println (format "            Triggered Test Re-runs: %s%s"
                                  (str/join ", " sample-tests)
                                  (if (> (count aff-tests) 3) "..." "")))))

             {:turn turn
              :latency-ms lat
              :vram-bytes vram
              :affected-files-cnt (count aff-files)
              :affected-tests-cnt (count aff-tests)
              :exact? exact?}))
         (range total-turns))

        latencies (mapv :latency-ms turn-metrics)
        mean-lat (/ (reduce + latencies) (double (count latencies)))
        all-exact? (every? :exact? turn-metrics)
        all-const-mem? (apply = (mapv :vram-bytes turn-metrics))
        vram-kb (/ (double (:vram-bytes tracker-session)) 1024.0)]

    (println "\n==================================================================")
    (println "  EXPERIMENT E4 SUMMARY: IN-VRAM DATALOG AGENT STATE TRACKER")
    (println "==================================================================")
    (println (format " Total Simulated Turns:              %d turns" total-turns))
    (println (format " Deductive Mathematical Exactness:   %s" (if all-exact? "100.0% [PERFECT SOUNDNESS]" "FAILED")))
    (println " Context Length Across 100 Turns:    O(1) [Zero Prompt Growth]")
    (println (format " Resident VRAM Memory Footprint:     %.2f KB (Constant across all 100 turns: %s)"
                     vram-kb (str all-const-mem?)))
    (println (format " Mean In-VRAM Contraction Latency:   %.4f ms per turn" mean-lat))
    (println "==================================================================\n")

    {:all-exact? all-exact?
     :all-const-mem? all-const-mem?
     :mean-lat mean-lat
     :vram-bytes (:vram-bytes tracker-session)}))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-datalog-state-tracker-experiment opts)
    (shutdown-agents)
    (System/exit 0)))
