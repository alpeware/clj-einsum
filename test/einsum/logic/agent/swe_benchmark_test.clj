(ns einsum.logic.agent.swe-benchmark-test
  "Unit and generative tests for Experiment E7: Long-Horizon Software Engineering Agent Benchmark.
   Compares Arm A (Standard LLM full-history KV cache) vs Arm B (In-VRAM Datalog State Tracker + Ephemeral Memory).
   Rule 1: Strict TDD - generative tests for invariants defined before core implementation."
  (:require [einsum.core :as xla]
            [einsum.logic.agent.swe-benchmark :as swe]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property Tests: Codebase Ontology Integrity
;; ==============================================================================

(defspec prop-codebase-ontology-integrity 20
  (prop/for-all [seed gen/nat]
                (let [ontology (swe/build-codebase-ontology {:num-files 32
                                                             :num-functions 64
                                                             :num-tests 16
                                                             :seed seed})
                      {:keys [files functions tests deps func->file test->funcs total-entities]} ontology
                      file-ids (set (map :id files))
                      func-ids (set (map :id functions))
                      test-ids (set (map :id tests))]
                  (and
                   (= 32 (count files))
                   (= 64 (count functions))
                   (= 16 (count tests))
                   (= total-entities (+ 32 64 16))
       ;; Every function maps to a valid file
                   (every? #(contains? file-ids %) (vals func->file))
       ;; Every dependency connects valid functions
                   (every? (fn [[caller callee]]
                             (and (contains? func-ids caller)
                                  (contains? func-ids callee)))
                           deps)
       ;; Every test covers valid functions
                   (every? (fn [[t-id covered-funcs]]
                             (and (contains? test-ids t-id)
                                  (every? #(contains? func-ids %) covered-funcs)))
                           test->funcs)))))

;; ==============================================================================
;; 2. Generative Property: Ground-Truth Dependency Transitive Closure Soundness
;; ==============================================================================

(defspec prop-ground-truth-transitive-soundness 20
  (prop/for-all [chain-len (gen/choose 3 7)]
                (let [;; Function chain: f0 -> f1 -> f2 -> ... -> f_{chain-len - 1}
          ;; where fi depends on f_{i+1}
                      dep-pairs (set (map (fn [i] [i (inc i)]) (range (dec chain-len))))
          ;; Modify the deepest function: (dec chain-len)
                      modified-func #{(dec chain-len)}
                      closure (swe/compute-transitive-invalidations dep-pairs modified-func)]
      ;; All functions in 0..(chain-len - 1) must be in the transitive invalidation set
                  (every? #(contains? closure %) (range chain-len)))))

;; ==============================================================================
;; 3. Unit Test: Arm B In-VRAM Deductive Exactness via PJRT
;; ==============================================================================

(deftest test-arm-b-pjrt-deductive-exactness
  (testing "Arm B achieves 100% deductive exactness on a multi-turn refactoring sequence in PJRT"
    (let [ctx (xla/get-context)
          ontology (swe/build-codebase-ontology {:num-files 16
                                                 :num-functions 32
                                                 :num-tests 8
                                                 :seed 1234})
          actions (swe/generate-benchmark-actions ontology 25 1234)
          session-b (swe/init-arm-b-session ctx ontology)
          results (atom [])]
      (doseq [action actions]
        (let [step-res (swe/step-arm-b! session-b action ontology)]
          (swap! results conj step-res)))
      (let [all-exact? (every? :exact? @results)]
        (is (true? all-exact?) "Arm B must achieve 100% deductive exactness across all turns")
        (is (= 25 (count @results)))))))

;; ==============================================================================
;; 4. Generative Property: Arm B O(1) Constant Memory Invariant
;; ==============================================================================

(defspec prop-arm-b-constant-memory-invariant 15
  (prop/for-all [num-turns (gen/choose 10 30)]
                (let [ctx (xla/get-context)
                      ontology (swe/build-codebase-ontology {:num-files 8
                                                             :num-functions 16
                                                             :num-tests 4
                                                             :seed 999})
                      actions (swe/generate-benchmark-actions ontology num-turns 999)
                      session-b (swe/init-arm-b-session ctx ontology)
                      vram-measurements (mapv (fn [action]
                                                (:state-vram-bytes (swe/step-arm-b! session-b action ontology)))
                                              actions)]
      ;; All turn measurements must be identical (strictly constant O(1) memory)
                  (apply = vram-measurements))))

;; ==============================================================================
;; 5. Unit Test: Full Benchmark Simulation (Arm A vs Arm B Comparison)
;; ==============================================================================

(deftest test-full-benchmark-comparison
  (testing "SWE benchmark demonstrates Arm B superiority in memory and deductive stability"
    (let [opts {:backend :cpu
                :turns 50
                :num-files 16
                :num-functions 32
                :num-tests 8
                :seed 42
                :window-size 4096}
          benchmark-res (swe/run-swe-benchmark opts)
          summary (:summary benchmark-res)]
      (is (= 50 (:total-turns summary)))
      (is (= 1.0 (:arm-b-mean-accuracy summary)) "Arm B must maintain 100.0% deductive accuracy")
      (is (< (:arm-a-mean-accuracy summary) (:arm-b-mean-accuracy summary))
          "Arm A mean accuracy must degrade over long horizon due to context limits")
      (is (< (:arm-a-late-accuracy summary) (:arm-b-late-accuracy summary))
          "Arm A late-stage accuracy must fall below Arm B exact accuracy")
      (is (> (:arm-a-final-vram-mb summary) (:arm-b-final-vram-mb summary))
          "Arm A KV cache VRAM footprint must exceed Arm B constant footprint"))))
