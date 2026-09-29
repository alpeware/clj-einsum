(ns experiments.gate3-evals.prompt-tuning-v1-test
  "Unit and generative tests for Prompt Tuning v1: Worked Agentic Trajectories.
   Validates token budget invariants (P0, P1, P2 <= 600), strict catalog disjointness,
   mechanical canary echoing detection, McNemar paired tests, and complete decision precedence."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.runtime.tokenizer.core :as tok-core]
            [experiments.gate3-evals.clojure-bench.core :as bench-core]
            [experiments.gate3-evals.prompt-tuning-v1.core :as pt-core]
            [experiments.gate3-evals.prompt-tuning-v1.run :as pt-run]))

;; =============================================================================
;; 1. Token Budget Invariant Tests (Gate 1)
;; =============================================================================

(deftest test-prompt-token-budgets
  (testing "Candidate prompt variants conform to strict token budget invariants (<= 600 cap)"
    (let [model-path ".models/gemma-4-e4b-it-qat-int4"]
      (if (and (.exists (io/file model-path))
               (.exists (io/file model-path "tokenizer.json")))
        (let [tok (tok-core/from-file model-path)
              p0 pt-core/PROMPT-V0-ZERO-SHOT
              p1 pt-core/PROMPT-V1-SINGLE
              p2 pt-core/PROMPT-V2-DUAL
              t0 (pt-core/measure-prompt-tokens tok p0)
              t1 (pt-core/measure-prompt-tokens tok p1)
              t2 (pt-core/measure-prompt-tokens tok p2)]
          (is (pos? t0) "P0 token count must be positive")
          (is (< t0 200) (format "P0 (rules only) must be < 200 tokens (got %d)" t0))
          (is (< t1 400) (format "P1 (single worked example) must be < 400 tokens (got %d)" t1))
          (is (<= t2 600) (format "P2 (dual worked examples) must not exceed 600 token cap (got %d)" t2))
          (is (> t2 t1) "P2 token count must exceed P1")
          (is (> t1 t0) "P1 token count must exceed P0"))
        ;; Fallback when running in minimal CI clone without model weights: approximate char bounds
        (do
          (is (< (count pt-core/PROMPT-V0-ZERO-SHOT) 1000))
          (is (< (count pt-core/PROMPT-V1-SINGLE) 2500))
          (is (<= (count pt-core/PROMPT-V2-DUAL) 4000)))))))

(deftest test-deterministic-prompt-sha
  (testing "compute-prompt-sha renders deterministic 64-character lowercase hexadecimal SHA-256 hashes"
    (let [sha0 (pt-core/compute-prompt-sha :p0)
          sha1 (pt-core/compute-prompt-sha :p1)
          sha2 (pt-core/compute-prompt-sha :p2)]
      (is (re-matches #"^[0-9a-f]{64}$" sha0) "P0 SHA must be 64-char lowercase hex")
      (is (re-matches #"^[0-9a-f]{64}$" sha1) "P1 SHA must be 64-char lowercase hex")
      (is (re-matches #"^[0-9a-f]{64}$" sha2) "P2 SHA must be 64-char lowercase hex")
      (is (distinct? sha0 sha1 sha2) "All prompt variant SHAs must be distinct")
      ;; Determinism across calls
      (is (= sha0 (pt-core/compute-prompt-sha :p0)) "P0 SHA must be strictly deterministic")
      (is (= sha1 (pt-core/compute-prompt-sha :p1)) "P1 SHA must be strictly deterministic")
      (is (= sha2 (pt-core/compute-prompt-sha :p2)) "P2 SHA must be strictly deterministic")
      ;; Equivalence between keyword and system prompt text
      (is (= sha0 (pt-core/compute-prompt-sha pt-core/PROMPT-V0-ZERO-SHOT)))
      (is (= sha1 (pt-core/compute-prompt-sha pt-core/PROMPT-V1-SINGLE)))
      (is (= sha2 (pt-core/compute-prompt-sha pt-core/PROMPT-V2-DUAL))))))

;; =============================================================================
;; 2. Strict Catalog Disjointness Invariant (Zero Contamination)
;; =============================================================================

(deftest test-strict-catalog-disjointness
  (testing "Worked example symbols are strictly disjoint from all catalog datasets"
    (let [dev-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"))
          sealed-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn"))
          bench-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/clojure_bench/tasks_sealed.edn"))
          dev50-tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/dev_50_public.edn"))
          all-catalog-tasks (concat dev-tasks sealed-tasks bench-tasks dev50-tasks)
          disjoint-res (pt-core/check-disjointness all-catalog-tasks)]
      (is (true? (:disjoint? disjoint-res))
          (format "Disjointness violated! Conflicts: %s" (:conflicts disjoint-res)))
      (is (empty? (:conflicts disjoint-res)) "Zero conflict symbols permitted in catalog"))))

;; =============================================================================
;; 3. Mechanical Canary Echoing Detection
;; =============================================================================

(deftest test-canary-echo-detector
  (testing "detect-canary-echo correctly distinguishes clean submissions from canary leaks"
    ;; Clean submission
    (let [clean-res (pt-core/detect-canary-echo "(defn solve [xs] (filter even? xs))" 'solve)]
      (is (false? (:leak? clean-res)) "Clean code must not trigger leak")
      (is (empty? (:reasons clean-res))))

    ;; Canary symbol leak (sum-even-squares injected into unrelated task)
    (let [leak-sym-res (pt-core/detect-canary-echo "(defn solve [xs] (sum-even-squares xs))" 'solve)]
      (is (true? (:leak? leak-sym-res)) "Calling canary symbol must trigger leak")
      (is (seq (:reasons leak-sym-res))))

    ;; Canary symbol leak (word-lengths injected)
    (let [leak-word-res (pt-core/detect-canary-echo "(defn foo [s] (let [word-lengths (fn [x] x)] (word-lengths s)))" 'foo)]
      (is (true? (:leak? leak-word-res)) "Binding canary symbol must trigger leak"))

    ;; Canary literal string leak
    (let [leak-lit-res (pt-core/detect-canary-echo "(defn bar [x] (println \"Expected 20, got 10\") x)" 'bar)]
      (is (true? (:leak? leak-lit-res)) "Literal canary string must trigger leak"))

    ;; Canary target function exemption
    (let [exempt-res (pt-core/detect-canary-echo "(defn sum-even-squares [nums] nums)" 'sum-even-squares)]
      (is (false? (:leak? exempt-res)) "Target function matching canary itself must not trigger leak"))))

;; =============================================================================
;; 4. Paired McNemar Test Invariants
;; =============================================================================

(deftest test-mcnemar-test-cases
  (testing "mcnemar-test computes exact and asymptotic tests correctly"
    ;; Identical models (0 discordant pairs)
    (let [res0 (pt-core/mcnemar-test 0 0 447)]
      (is (= 0.0 (:delta res0)))
      (is (false? (:significant? res0))))

    ;; Significant positive gain (30 favorable flips vs 5 unfavorable flips)
    (let [res-pos (pt-core/mcnemar-test 30 5 447)]
      (is (> (:delta res-pos) 0.05))
      (is (true? (:significant? res-pos))))

    ;; Symmetrical discordant pairs (15 vs 15) -> p = 0.50, not significant
    (let [res-sym (pt-core/mcnemar-test 15 15 447)]
      (is (= 0.0 (:delta res-sym)))
      (is (false? (:significant? res-sym))))

    ;; Negative flips (5 favorable vs 30 unfavorable) -> not significant for positive gain
    (let [res-neg (pt-core/mcnemar-test 5 30 447)]
      (is (< (:delta res-neg) 0.0))
      (is (false? (:significant? res-neg))))))

(defspec prop-mcnemar-invariants 50
  (prop/for-all [b gen/nat
                 c gen/nat
                 n (gen/fmap #(+ % 50) gen/nat)]
                (let [res (pt-core/mcnemar-test b c n)]
                  (and (= (:delta res) (double (/ (- b c) n)))
                       (>= (:chi2 res) 0.0)
                       (boolean? (:significant? res))))))

;; =============================================================================
;; 5. Decision Precedence Hierarchy & Complete Partition
;; =============================================================================

(deftest test-decision-rules-edge-cases
  (testing "evaluate-precedence-decision eliminates Case A, Case B, and partitions cleanly"
    ;; Happy path ADOPT
    (let [adopt-res (pt-core/evaluate-precedence-decision
                     {:delta-e4b 0.06
                      :mcnemar-significant? true
                      :delta-31b 0.02
                      :sealed10-regressions 0
                      :canary-leaks 0
                      :prompt-tokens 560
                      :wall-clock-inflation 0.05
                      :phase2-early-stop? false})]
      (is (= :adopt (:verdict adopt-res))))

    ;; Case A: Delta = +6% but non-significant (p >= 0.05) -> REJECT
    (let [case-a-res (pt-core/evaluate-precedence-decision
                      {:delta-e4b 0.06
                       :mcnemar-significant? false
                       :delta-31b 0.02
                       :sealed10-regressions 0
                       :canary-leaks 0
                       :prompt-tokens 560
                       :wall-clock-inflation 0.05
                       :phase2-early-stop? false})]
      (is (= :reject (:verdict case-a-res)) "Non-significant gain must route deterministically to REJECT"))

    ;; Case B: Delta-E4B = +6% but teacher dev regresses (Delta-31B = -2%) -> REJECT
    (let [case-b-res (pt-core/evaluate-precedence-decision
                      {:delta-e4b 0.06
                       :mcnemar-significant? true
                       :delta-31b -0.02
                       :sealed10-regressions 0
                       :canary-leaks 0
                       :prompt-tokens 560
                       :wall-clock-inflation 0.05
                       :phase2-early-stop? false})]
      (is (= :reject (:verdict case-b-res)) "Teacher dev regression must route deterministically to REJECT"))

    ;; Fatal Trigger: Canary leak -> KILLED dominates
    (let [killed-canary (pt-core/evaluate-precedence-decision
                         {:delta-e4b 0.08
                          :mcnemar-significant? true
                          :delta-31b 0.04
                          :sealed10-regressions 0
                          :canary-leaks 1
                          :prompt-tokens 560
                          :wall-clock-inflation 0.05
                          :phase2-early-stop? false})]
      (is (= :killed (:verdict killed-canary)) "Canary leak must dominate and trigger KILLED"))

    ;; Fatal Trigger: Sealed-10 adverse regression -> KILLED dominates
    (let [killed-sealed (pt-core/evaluate-precedence-decision
                         {:delta-e4b 0.08
                          :mcnemar-significant? true
                          :delta-31b 0.04
                          :sealed10-regressions 1
                          :canary-leaks 0
                          :prompt-tokens 560
                          :wall-clock-inflation 0.05
                          :phase2-early-stop? false})]
      (is (= :killed (:verdict killed-sealed)) "Sealed regression must dominate and trigger KILLED"))

    ;; Fatal Trigger: Negative transfer on dev -> KILLED
    (let [killed-neg (pt-core/evaluate-precedence-decision
                      {:delta-e4b -0.01
                       :mcnemar-significant? false
                       :delta-31b 0.0
                       :sealed10-regressions 0
                       :canary-leaks 0
                       :prompt-tokens 560
                       :wall-clock-inflation 0.05
                       :phase2-early-stop? false})]
      (is (= :killed (:verdict killed-neg)) "Negative transfer must trigger KILLED"))

    ;; Phase 2 early-stop -> REJECT
    (let [early-stop-res (pt-core/evaluate-precedence-decision
                          {:delta-e4b 0.01
                           :mcnemar-significant? false
                           :delta-31b 0.0
                           :sealed10-regressions 0
                           :canary-leaks 0
                           :prompt-tokens 560
                           :wall-clock-inflation 0.05
                           :phase2-early-stop? true})]
      (is (= :reject (:verdict early-stop-res)) "Phase 2 pilot early-stop must trigger REJECT"))))

(defspec prop-decision-precedence-partition 100
  (prop/for-all [d-e4b (gen/fmap #(- (/ (double %) 100.0) 0.5) (gen/choose 0 100))
                 sig? gen/boolean
                 d-31b (gen/fmap #(- (/ (double %) 100.0) 0.5) (gen/choose 0 100))
                 adverse-flips (gen/choose 0 2)
                 leaks (gen/choose 0 2)
                 tokens (gen/choose 100 800)
                 inflation (gen/fmap #(/ (double %) 100.0) (gen/choose 0 50))
                 early-stop? gen/boolean]
                (let [res (pt-core/evaluate-precedence-decision
                           {:delta-e4b d-e4b
                            :mcnemar-significant? sig?
                            :delta-31b d-31b
                            :sealed10-regressions adverse-flips
                            :canary-leaks leaks
                            :prompt-tokens tokens
                            :wall-clock-inflation inflation
                            :phase2-early-stop? early-stop?})]
                  (and (contains? #{:adopt :reject :killed} (:verdict res))
                       (string? (:rationale res))
                       (pos? (count (:rationale res)))))))

;; =============================================================================
;; 6. Phase 2: Stratified Pilot & Candidate Downselection Tests
;; =============================================================================

(deftest test-stratify-tasks-multipl-e-dev
  (testing "stratify-tasks preserves source prefix proportions on MultiPL-E dev (447 tasks)"
    (let [tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"))
          sampled-50 (bench-core/stratify-tasks tasks 50)
          counts-50 (frequencies (map #(first (str/split (:id %) #"-")) sampled-50))]
      (is (= 50 (count sampled-50)) "Must select exactly 50 tasks")
      (is (= 14 (get counts-50 "humaneval")) "Humaneval must be 14 tasks (28% of 50)")
      (is (= 36 (get counts-50 "mbpp")) "MBPP must be 36 tasks (72% of 50)"))

    (testing "stratify-tasks with limit 10"
      (let [tasks (edn/read-string (slurp "resources/catalog/gate3_evals/multipl_e/tasks_dev.edn"))
            sampled-10 (bench-core/stratify-tasks tasks 10)
            counts-10 (frequencies (map #(first (str/split (:id %) #"-")) sampled-10))]
        (is (= 10 (count sampled-10)))
        (is (= 3 (get counts-10 "humaneval")))
        (is (= 7 (get counts-10 "mbpp")))))))

(deftest test-phase2-downselection-rules
  (testing "Downselection early-stop if neither P1 nor P2 beats P0 by >= 2 tasks"
    ;; k0 = 30, k1 = 31 (+1), k2 = 30 (+0) -> REJECT
    (let [stats {:p0 {:passed-count 30} :p1 {:passed-count 31} :p2 {:passed-count 30}}
          res (pt-run/downselect-candidate stats)]
      (is (= :REJECT (:verdict res)))
      (is (nil? (:selected-candidate res)))
      (is (true? (:early-stop? res)))
      (is (= {:p1 1 :p2 0} (:deltas res)))))

  (testing "Downselection selects P1 when P1 wins by >= 2 tasks"
    ;; k0 = 30, k1 = 33 (+3), k2 = 31 (+1) -> P1 selected
    (let [stats {:p0 {:passed-count 30} :p1 {:passed-count 33} :p2 {:passed-count 31}}
          res (pt-run/downselect-candidate stats)]
      (is (= :PROCEED-TO-PHASE-3 (:verdict res)))
      (is (= :p1 (:selected-candidate res)))
      (is (false? (:early-stop? res)))
      (is (= {:p1 3 :p2 1} (:deltas res)))))

  (testing "Downselection selects P2 when P2 wins by >= 2 tasks"
    ;; k0 = 30, k1 = 32 (+2), k2 = 34 (+4) -> P2 selected
    (let [stats {:p0 {:passed-count 30} :p1 {:passed-count 32} :p2 {:passed-count 34}}
          res (pt-run/downselect-candidate stats)]
      (is (= :PROCEED-TO-PHASE-3 (:verdict res)))
      (is (= :p2 (:selected-candidate res)))
      (is (false? (:early-stop? res)))
      (is (= {:p1 2 :p2 4} (:deltas res)))))

  (testing "Downselection tie-break favors smaller token budget (P1 < P2)"
    ;; k0 = 30, k1 = 34 (+4), k2 = 34 (+4) -> P1 selected by tie-break
    (let [stats {:p0 {:passed-count 30} :p1 {:passed-count 34} :p2 {:passed-count 34}}
          res (pt-run/downselect-candidate stats)]
      (is (= :PROCEED-TO-PHASE-3 (:verdict res)))
      (is (= :p1 (:selected-candidate res)))
      (is (false? (:early-stop? res)))
      (is (= {:p1 4 :p2 4} (:deltas res)))))

  (testing "Fatal canary leak dominates and triggers KILLED"
    (let [stats {:p0 {:passed-count 30}
                 :p1 {:passed-count 36 :canary-leaks [{:task "t1" :details "leak"}]}
                 :p2 {:passed-count 30}}
          res (pt-run/downselect-candidate stats)]
      (is (= :KILLED (:verdict res)))
      (is (nil? (:selected-candidate res)))
      (is (true? (:early-stop? res))))))

(defspec prop-downselect-candidate-invariants 100
  (prop/for-all [k0 (gen/choose 0 50)
                 k1 (gen/choose 0 50)
                 k2 (gen/choose 0 50)
                 has-leak? gen/boolean]
                (let [stats {:p0 {:passed-count k0}
                             :p1 {:passed-count k1 :canary-leaks (when has-leak? [{:leak true}])}
                             :p2 {:passed-count k2}}
                      res (pt-run/downselect-candidate stats)
                      d1 (- k1 k0)
                      d2 (- k2 k0)]
                  (cond
                    has-leak?
                    (and (= :KILLED (:verdict res))
                         (nil? (:selected-candidate res))
                         (true? (:early-stop? res)))

                    (and (< d1 2) (< d2 2))
                    (and (= :REJECT (:verdict res))
                         (nil? (:selected-candidate res))
                         (true? (:early-stop? res)))

                    :else
                    (and (= :PROCEED-TO-PHASE-3 (:verdict res))
                         (false? (:early-stop? res))
                         (contains? #{:p1 :p2} (:selected-candidate res))
                         (if (= k1 k2) (= :p1 (:selected-candidate res)) true))))))

(deftest test-phase2-pilot-dry-run-e2e
  (testing "End-to-end Phase 2 pilot dry-run executes cleanly and outputs valid summary"
    (let [tmp-file (java.io.File/createTempFile "pilot-dry-run-test" ".edn")
          tmp-path (.getAbsolutePath tmp-file)
          _ (.deleteOnExit tmp-file)
          res (pt-run/run-pilot {:limit 5
                                 :dry-run true
                                 :quiet true
                                 :output-file tmp-path})]
      (is (= 5 (:pilot-size res)))
      (is (= ["humaneval" "mbpp"] (sort (keys (:stratification res)))))
      (is (= [:p0 :p1 :p2] (keys (:variant-stats res))))
      (is (some? (:decision res)))
      (is (contains? #{:PROCEED-TO-PHASE-3 :REJECT :KILLED} (get-in res [:decision :verdict])))
      (is (.exists tmp-file))
      (is (pos? (.length tmp-file))))))

(deftest test-pilot-canary-leak-integration
  (testing "Canary echoing detection triggers and routes through pilot audit path"
    (let [leaking-code "(defn solve [x] (sum-even-squares x))"
          clean-code "(defn solve [x] (* x 2))"
          target-fn "solve"
          leaking-audit (pt-core/detect-canary-echo leaking-code target-fn)
          clean-audit (pt-core/detect-canary-echo clean-code target-fn)]
      ;; 1. Core detector contracts
      (is (true? (:leak? leaking-audit)))
      (is (true? (:echo-detected? leaking-audit)))
      (is (seq (:reasons leaking-audit)))
      (is (seq (:matches leaking-audit)))
      (is (false? (:leak? clean-audit)))
      (is (false? (:echo-detected? clean-audit)))

      ;; 2. Integration with downselect-candidate: leak forces :KILLED
      (let [stats-with-leak {:p0 {:passed-count 35}
                             :p1 {:passed-count 40
                                  :canary-leaks [{:task "mbpp-clj-001"
                                                  :fn-name "solve"
                                                  :variant :p1
                                                  :details leaking-audit}]}
                             :p2 {:passed-count 38}}
            decision (pt-run/downselect-candidate stats-with-leak)]
        (is (= :KILLED (:verdict decision)))
        (is (nil? (:selected-candidate decision)))
        (is (true? (:early-stop? decision)))
        (is (str/includes? (:rationale decision) "Fatal canary echoing leak detected"))))))
