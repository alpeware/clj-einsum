(ns clj-xla.logic.agent.state-tracker-test
  "Unit and generative tests for Experiment E4: In-VRAM Datalog Fixpoint State Tracker.
   Rule 1: Strict TDD - invariants and properties defined before implementation."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.agent.state-tracker :as tracker]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property: Transitive Invalidation Soundness
;; ==============================================================================

(defspec prop-state-tracker-transitive-invalidation 25
  (prop/for-all [chain-len (gen/choose 3 8)]
    ;; Build a linear dependency chain: 0 -> 1 -> 2 -> ... -> (chain-len - 1)
    ;; where i depends on (i+1). If (chain-len - 1) is modified, all prior items are affected.
                (let [n 16
                      dep-pairs (mapv (fn [i] [i (inc i)]) (range (dec chain-len)))
                      modified #{(dec chain-len)}
                      res (tracker/compute-affected-entities-host dep-pairs modified n)]
                  (every? #(contains? res %) (range chain-len)))))

;; ==============================================================================
;; 2. In-VRAM OpenXLA PJRT Datalog Compilation & Execution Check
;; ==============================================================================

(deftest test-state-tracker-pjrt-execution
  (testing "In-VRAM Datalog Fixpoint compiles to OpenXLA PJRT and tracks state exactly"
    (let [ctx (xla/get-context)
          n 8
          ;; Dependencies: file 0 depends on file 1, file 1 depends on file 2
          ;; Tests: test 6 tests file 0, test 7 tests file 1
          deps #{[0 1] [1 2]}
          tests #{[6 0] [7 1]}
          tracker-session (tracker/init-state-tracker ctx n {:deps deps :tests tests})
          ;; Turn 1: Modify file 2 (root dependency)
          turn1-res (tracker/step-state-tracker! tracker-session #{2})
          affected-files (:affected-files turn1-res)
          affected-tests (:affected-tests turn1-res)]
      ;; Modifying file 2 must transitively invalidate file 1 and file 0
      (is (contains? affected-files 2) "Modified file 2 must be affected")
      (is (contains? affected-files 1) "File 1 depends on file 2 and must be invalidated")
      (is (contains? affected-files 0) "File 0 depends on file 1 and must be invalidated")
      ;; Tests 7 (tests file 1) and 6 (tests file 0) must be flagged for re-run
      (is (contains? affected-tests 7) "Test 7 covers file 1 and must be flagged")
      (is (contains? affected-tests 6) "Test 6 covers file 0 and must be flagged"))))

;; ==============================================================================
;; 3. 100-Turn Simulation Invariant: Constant O(1) Memory Footprint
;; ==============================================================================

(deftest test-100-turn-constant-memory
  (testing "100-turn simulated agent execution retains strictly constant VRAM memory"
    (let [ctx (xla/get-context)
          n 32
          tracker-session (tracker/init-state-tracker ctx n {:deps #{[0 1] [1 2] [2 3] [3 4]}
                                                             :tests #{[30 0] [31 1]}})
          mem-sizes (atom [])]
      (dotimes [turn 100]
        (let [mod-entity (mod turn 5)
              res (tracker/step-state-tracker! tracker-session #{mod-entity})]
          (swap! mem-sizes conj (:vram-bytes res))))
      (let [first-size (first @mem-sizes)]
        (is (= 100 (count @mem-sizes)))
        (is (every? #(= % first-size) @mem-sizes)
            "State tensor memory in VRAM must remain strictly constant across 100 turns")))))
