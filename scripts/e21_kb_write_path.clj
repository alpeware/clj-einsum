(ns scripts.e21-kb-write-path
  "Experiment E21: Schema-Constrained KB Write Path.
   Executes the 7 pre-registered acceptance criteria on the E20 family-tree generator schema:
   1. Schema-checked commit (indegree <= 2, 80 true edges commit cleanly, 81st rejected)
   2. Cycle rejection (backward child -> ancestor rejected naming :acyclicity)
   3. Type rejection (domain sort violation rejected naming :domain)
   4. Ambiguity, not guessing (E20 mating ambiguity stored as disjunction, queries return :ambiguous)
   5. Derived relations (grandparent/2 materialized via Datalog rule reaches F1 = 1.0)
   6. Provenance & retraction (append-only log, supersede markers, history query)
   7. Read latency (p50/p99 microsecond benchmarking)"
  (:require [clj-xla.logic.data.family-trees :as ft]
            [clj-xla.logic.kb :as kb]
            [clj-xla.logic.predicate-crystallization :as pc]
            [clojure.java.io :as io])
  (:gen-class))

(def family-schema
  {:parent {:arity 2
            :sorts [:person :person]
            :cardinality {:target-arg 1 :max 2}
            :denial #{:acyclic}}
   :sibling {:arity 2
             :sorts [:person :person]
             :identity {:via :parent :shared-parents true}}
   :grandparent {:arity 2
                 :sorts [:person :person]
                 :derived true
                 :rule {:head [:grandparent :?x :?z]
                        :body [[:parent :?x :?y] [:parent :?y :?z]]}}})

(defn run-acceptance-criteria [opts]
  (println "\n================================================================================")
  (println "  EXPERIMENT E21: SCHEMA-CONSTRAINED KB WRITE PATH ACCEPTANCE AUDIT")
  (println "================================================================================")
  (let [forest (ft/generate-family-forest {:num-entities 64 :num-trees 8 :seed 42})
        true-parents (set (:parent-pairs forest))
        true-gp (set (:grandparent-pairs forest))
        entities (into {} (map (fn [i] [i :person]) (range 64)))
        kb0 (kb/create-kb family-schema {:entities entities})

        ;; ----------------------------------------------------------------------
        ;; 1. Criterion 1: Schema-Checked Commit
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 1] Schema-Checked Commit (80 True Edges + 81st Rejection) ---")
        commit-res
        (reduce (fn [acc [p c]]
                  (let [res (kb/assert-fact (:kb acc) [:parent p c] {:source :generator-truth})]
                    (if (:error res)
                      (reduced {:error res})
                      {:kb res :committed (inc (:committed acc))})))
                {:kb kb0 :committed 0}
                true-parents)
        c1-pass-80? (and (not (:error commit-res)) (== 80 (:committed commit-res)))
        kb-80 (:kb commit-res)
        ;; Attempt 81st edge giving child a 3rd parent
        sample-child (second (first true-parents))
        existing-p (set (map first (filter #(= (second %) sample-child) true-parents)))
        extra-parent (first (remove #(or (contains? existing-p %) (= % sample-child)) (range 64)))
        reject-81-res (kb/assert-fact kb-80 [:parent extra-parent sample-child] {:source :illegal-3rd-parent})
        c1-reject-3rd? (and (= :schema-violation (:error reject-81-res))
                            (= :cardinality (:violation reject-81-res))
                            (= sample-child (:target-arg reject-81-res)))
        c1-status (and c1-pass-80? c1-reject-3rd?)
        _ (println (format "  Assert 80 True Edges:    [%s] (Committed: %d/80)"
                           (if c1-pass-80? "PASS" "FAIL") (:committed commit-res)))
        _ (println (format "  Reject 81st Edge (Card): [%s] (Violation: %s, Target: %s, Count: %s, Max: %s)"
                           (if c1-reject-3rd? "PASS" "FAIL")
                           (:violation reject-81-res) (:target-arg reject-81-res)
                           (:current-count reject-81-res) (:max reject-81-res)))

        ;; ----------------------------------------------------------------------
        ;; 2. Criterion 2: Cycle Rejection
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 2] Denial Constraint: Acyclicity Rejection ---")
        ;; Attempt backward edge: child -> founder
        tree0 (first (:trees forest))
        founder (first (:founders tree0))
        gen2-child (first (:gen2-children tree0))
        cycle-res (kb/assert-fact kb-80 [:parent gen2-child founder] {:source :malicious-backward-edge})
        c2-status (and (= :schema-violation (:error cycle-res))
                       (= :acyclicity (:violation cycle-res))
                       (= gen2-child (:from cycle-res))
                       (= founder (:to cycle-res)))
        _ (println (format "  Backward Edge Assertion: [%s] (Violation: %s, From: %s, To: %s)"
                           (if c2-status "PASS" "FAIL")
                           (:violation cycle-res) (:from cycle-res) (:to cycle-res)))

        ;; ----------------------------------------------------------------------
        ;; 3. Criterion 3: Domain Sort Rejection
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 3] Domain Constraint: Argument Sort Rejection ---")
        type-res (kb/assert-fact kb-80 [:parent :alice 42] {:source :unregistered-sort})
        c3-status (and (= :schema-violation (:error type-res))
                       (= :domain (:violation type-res))
                       (= :alice (:value type-res))
                       (= :person (:expected type-res)))
        _ (println (format "  Typecheck Assertion:     [%s] (Violation: %s, Value: %s, Expected: %s)"
                           (if c3-status "PASS" "FAIL")
                           (:violation type-res) (:value type-res) (:expected type-res)))

        ;; ----------------------------------------------------------------------
        ;; 4. Criterion 4: Ambiguity as a First-Class Citizen
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 4] Ambiguity Store: Mating Disjunction Resolution ---")
        m (:mating-child tree0)
        o (:other-child tree0)
        s (:spouse tree0)
        g1 (first (:gen2-children tree0))
        branch-1 #{[:parent m g1] [:parent s g1]}
        branch-2 #{[:parent o g1] [:parent s g1]}
        kb-disj (kb/assert-disjunction kb0 [branch-1 branch-2] {:source :e20-mating-ambiguity})
        q-m (kb/query-fact kb-disj [:parent m g1])
        q-o (kb/query-fact kb-disj [:parent o g1])
        attempt-commit-m (kb/assert-fact kb-disj [:parent m g1] {:source :premature-guess})
        c4-query-ambig? (and (= :ambiguous (:status q-m))
                             (= :ambiguous (:status q-o))
                             (= 2 (count (:completions q-m))))
        c4-reject-guess? (and (= :schema-violation (:error attempt-commit-m))
                              (= :ambiguity (:violation attempt-commit-m)))
        c4-status (and c4-query-ambig? c4-reject-guess?)
        _ (println (format "  Query [:parent %s %s]:   [%s] (Status: %s, Completions: %d)"
                           m g1 (if c4-query-ambig? "PASS" "FAIL") (:status q-m) (count (:completions q-m))))
        _ (println (format "  Reject Definite Guess:   [%s] (Violation: %s, Message: '%s')"
                           (if c4-reject-guess? "PASS" "FAIL") (:violation attempt-commit-m) (:message attempt-commit-m)))

        ;; ----------------------------------------------------------------------
        ;; 5. Criterion 5: Derived Relations (Datalog Materialization)
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 5] Derived Relations: Grandparent/2 Fixpoint ---")
        t0-mat (System/nanoTime)
        kb-mat (kb/materialize-derived kb-80)
        mat-time-ms (/ (- (System/nanoTime) t0-mat) 1e6)
        mat-gp (set (get-in kb-mat [:facts :grandparent]))
        f1-gp (pc/pair-set-f1 mat-gp true-gp)
        c5-status (and (= mat-gp true-gp) (== 1.0 (:f1 f1-gp)))
        _ (println (format "  Materialization Fixpoint: [%s] (Derived %d pairs in %.2f ms)"
                           (if c5-status "PASS" "FAIL") (count mat-gp) mat-time-ms))
        _ (println (format "  Grandparent F1 vs Truth:  F1: %.4f | Precision: %.4f | Recall: %.4f"
                           (:f1 f1-gp) (:precision f1-gp) (:recall f1-gp)))

        ;; ----------------------------------------------------------------------
        ;; 6. Criterion 6: Provenance & Append-Only Retraction
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 6] Provenance Tracking & Append-Only Supersede Markers ---")
        fact-to-retract [:parent (first (:founders tree0)) (first (:gen1-children tree0))]
        kb-retracted (kb/retract-fact kb-80 fact-to-retract {:source :agent-repair :reason :re-orienting})
        active-after (get-in kb-retracted [:facts :parent])
        hist (kb/history kb-retracted fact-to-retract)
        c6-removed? (not (contains? active-after (vec (rest fact-to-retract))))
        c6-has-supersede? (and (= 2 (count hist))
                               (= :assert (:op (first hist)))
                               (= :retract (:op (second hist)))
                               (= (:tx-id (second hist)) (:superseded-by (first hist)))
                               (= (:tx-id (first hist)) (:supersedes (second hist))))
        c6-status (and c6-removed? c6-has-supersede?)
        _ (println (format "  Edge Removed from Facts:  [%s] (Fact: %s)"
                           (if c6-removed? "PASS" "FAIL") (pr-str fact-to-retract)))
        _ (println (format "  Supersede Marker Chain:   [%s] (Assertion tx %d -> Retraction tx %d)"
                           (if c6-has-supersede? "PASS" "FAIL")
                           (:tx-id (first hist)) (:tx-id (second hist))))

        ;; ----------------------------------------------------------------------
        ;; 7. Criterion 7: Read Latency Benchmarking
        ;; ----------------------------------------------------------------------
        _ (println "\n--- [Criterion 7] Host-Side Ground Query Read Latency ---")
        queries (vec (map (fn [[p c]] [:parent p c]) true-parents))
        bench (kb/measure-read-latency kb-80 queries 10000)
        c7-status (and (pos? (:p50-us bench)) (< (:p50-us bench) 50.0))
        _ (println (format "  Ground Query Latency:     p50: %6.2f us | p90: %6.2f us | p99: %6.2f us | Mean: %6.2f us"
                           (:p50-us bench) (:p90-us bench) (:p99-us bench) (:mean-us bench)))
        _ (println (format "  Latency Verdict:          [%s] (Requirement: Microsecond range host-side)"
                           (if c7-status "PASS" "FAIL")))

        all-passed? (and c1-status c2-status c3-status c4-status c5-status c6-status c7-status)

        report {:c1 {:status (if c1-status :pass :fail) :committed 80 :rejected-extra? c1-reject-3rd?}
                :c2 {:status (if c2-status :pass :fail) :violation (:violation cycle-res)}
                :c3 {:status (if c3-status :pass :fail) :violation (:violation type-res)}
                :c4 {:status (if c4-status :pass :fail) :query-ambiguous? c4-query-ambig? :guess-rejected? c4-reject-guess?}
                :c5 {:status (if c5-status :pass :fail) :f1 (:f1 f1-gp) :pairs (count mat-gp) :time-ms mat-time-ms}
                :c6 {:status (if c6-status :pass :fail) :removed? c6-removed? :supersede-verified? c6-has-supersede?}
                :c7 {:status (if c7-status :pass :fail) :bench bench}
                :overall (if all-passed? :pass :fail)}]

    (println "\n================================================================================")
    (println "  EXPERIMENT E21: ACCEPTANCE SUMMARY")
    (println "================================================================================")
    (println (format "  Criterion 1 (Schema-Checked Commit):           [%s]" (if c1-status "PASS" "FAIL")))
    (println (format "  Criterion 2 (Acyclicity Denial Rejection):     [%s]" (if c2-status "PASS" "FAIL")))
    (println (format "  Criterion 3 (Domain Sort Rejection):           [%s]" (if c3-status "PASS" "FAIL")))
    (println (format "  Criterion 4 (Ambiguity Store & No Guessing):   [%s]" (if c4-status "PASS" "FAIL")))
    (println (format "  Criterion 5 (Derived Datalog Grandparent F1):  [%s] (F1 = %.4f)" (if c5-status "PASS" "FAIL") (:f1 f1-gp)))
    (println (format "  Criterion 6 (Provenance & Supersede Chain):    [%s]" (if c6-status "PASS" "FAIL")))
    (println (format "  Criterion 7 (Host Read Latency p50 < 50 us):   [%s] (p50 = %.2f us)" (if c7-status "PASS" "FAIL") (:p50-us bench)))
    (println "--------------------------------------------------------------------------------")
    (println (format "  OVERALL E21 SPECIFICATION VERDICT:             [%s]" (if all-passed? "ALL CRITERIA PASSED" "FAILED")))
    (println "================================================================================\n")

    (let [out-file (str (:out-dir opts) "results.edn")]
      (io/make-parents out-file)
      (spit out-file (pr-str report))
      (println (format "Saved E21 acceptance report to %s\n" out-file)))

    report))

(defn -main [& _args]
  (let [opts {:out-dir "paper-experiments/e21-kb/2026-09-17/"}]
    (run-acceptance-criteria opts)
    (System/exit 0)))
