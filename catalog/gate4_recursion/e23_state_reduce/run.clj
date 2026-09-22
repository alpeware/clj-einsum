(ns tools.e23-reduce
  "Experiment E23: The Reduce — Verified KB Accumulation Over a Horizon (RBAC).
   Executes:
   - Phase 0 Verification Gates:
     * P0a: Oracle determinism & invariant check (T=20, T=60)
     * P0b: Offline trap rejection across all 4 trap types
     * P0c: Write-arm emission smoke test (5 dry-run observations)
   - Phase 1 Comparative Sweep over Horizons (T=20 and T=60):
     * Cell H: Verified KB Accumulation Reduce Loop
     * Cell B0: In-Context Tracking Baseline
     * Cell B1: Unverified KB Accumulation Baseline
   - Snapshot F1, Gate-Hold, Horizon Flatness, and Interrogation QA."
  (:require [einsum.kb.store :as kb]
            [einsum.kb.rbac :as rbac]
            [einsum.runtime.tokenizer.protocol :as tok]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.string :as str]
            [tools.gemma4-inference :as gemma4-inf]))

;; ==============================================================================
;; 1. Prompts & Few-Shot Worked Examples
;; ==============================================================================

(def SYSTEM-PROMPT-H
  "You maintain the organization's RBAC knowledge base. Propose fact updates based on events:
<|tool_call> propose assert has_role USER ROLE<tool_call|>
<|tool_call> propose retract has_role USER ROLE<tool_call|>
<|tool_call> propose ambiguous has_role U1 ROLE | has_role U2 ROLE<tool_call|>
<|tool_call> propose resolve 0 has_role USER ROLE<tool_call|>
Rules: SoD forbids deployer+auditor, developer+auditor, admin+deployer. Admin cap: 3.")

(def FEW-SHOT-EXAMPLES-H
  "<|turn>user
Event: grant Bob the auditor role.<turn|>
<|turn>model
<|tool_call> propose assert has_role bob auditor<tool_call|>
<|tool_response> rejected: separation-of-duty([deployer auditor])<tool_response|>
Rejected due to separation of duty.<turn|>
<|turn>user
Event: blurry ticket: oncall role granted to either Erin or Frank.<turn|>
<|turn>model
<|tool_call> propose ambiguous has_role erin oncall | has_role frank oncall<tool_call|>
<|tool_response> committed<tool_response|>
Recorded ambiguous oncall assignment.<turn|>
<|turn>user
Event: confirmed: oncall ticket was assigned to Erin.<turn|>
<|turn>model
<|tool_call> propose resolve 0 has_role erin oncall<tool_call|>
<|tool_response> committed: resolved disjunction 0<tool_response|>
Resolved to Erin.<turn|>
<|turn>user
Event: grant Grace the developer role.<turn|>
<|turn>model
<|tool_call> propose assert has_role grace developer<tool_call|>
<|tool_response> committed: has_role grace developer<tool_response|>
Granted developer role to Grace.<turn|>
")

(def SYSTEM-PROMPT-B0
  "You are an access control tracking assistant. Carefully track all role assignments and revocations described in the event notices. When asked questions, answer accurately based on the true state of role assignments.")

(defn make-h-step-prompt
  [snapshot-str obs-text]
  (str "<bos><|turn>system\n" SYSTEM-PROMPT-H "<turn|>\n"
       FEW-SHOT-EXAMPLES-H
       "<|turn>user\n"
       snapshot-str "\n"
       "Event: " obs-text "<turn|>\n"
       "<|turn>model\n"))

(defn make-b0-interrogation-prompt
  [events-history question]
  (let [history-text (str/join "\n" (map (fn [e] (format "%d. %s" (:step e) (:text e))) events-history))]
    (str "<bos><|turn>system\n" SYSTEM-PROMPT-B0 "<turn|>\n"
         "<|turn>user\n"
         "Event History:\n" history-text "\n\n"
         "Question: " question "\n"
         "Answer with just 'Yes' or 'No', or list the exact names.<turn|>\n"
         "<|turn>model\n")))

(defn make-h-interrogation-prompt
  [snapshot-str question]
  (str "<bos><|turn>system\n"
       "You are an access control assistant. Answer the question directly using ONLY the facts present in the KB snapshot. If asked a yes/no question, answer 'Yes' or 'No'.<turn|>\n"
       "<|turn>user\n"
       snapshot-str "\n\n"
       "Question: " question "<turn|>\n"
       "<|turn>model\n"))

(defn generate-model-response
  "Executes generation and returns only the newly generated model text tokens,
   preventing prompt echo or few-shot leakage."
  [session prompt-str]
  (let [{:keys [tokenizer]} session
        prompt-ids (tok/encode tokenizer prompt-str)
        final-tokens (gemma4-inf/generate-text session prompt-str)
        prompt-count (min (count prompt-ids) (max 0 (- (long (:max-seq-len session 768)) 2)))
        gen-tokens (if (> (count final-tokens) prompt-count)
                     (subvec final-tokens prompt-count)
                     [])]
    (str/trim (tok/decode tokenizer gen-tokens))))

;; ==============================================================================
;; 2. Phase 0 Verification Gates
;; ==============================================================================

(defn run-phase0-verification
  [session out-dir]
  (println "\n==================================================")
  (println "=== Phase 0 Verification Gates (E23) ===")
  (println "==================================================")

  ;; P0a: Oracle determinism
  (println "\n[Gate P0a] Testing oracle determinism & shared prefix (T=20, T=60)...")
  (let [stream-20 (rbac/generate-rbac-stream {:seed 42 :length 20 :trap-rate 0.25})
        stream-60 (rbac/generate-rbac-stream {:seed 42 :length 60 :trap-rate 0.25})
        shared-prefix? (= stream-20 (subvec stream-60 0 20))
        init-kb (rbac/create-initial-rbac-kb)
        kb-20 (rbac/fold-oracle-stream init-kb stream-20)
        kb-60 (rbac/fold-oracle-stream init-kb stream-60)
        facts-20 (get-in kb-20 [:facts :has_role])
        facts-60 (get-in kb-60 [:facts :has_role])
        admins-20 (filter #(= (second %) :admin) facts-20)
        admins-60 (filter #(= (second %) :admin) facts-60)
        admin-cap-ok? (and (<= (count admins-20) rbac/DEFAULT-ADMIN-CAP)
                           (<= (count admins-60) rbac/DEFAULT-ADMIN-CAP))
        sod-ok-20? (not-any? (fn [[u r]]
                               (some (fn [[r1 r2]]
                                       (and (or (= r r1) (= r r2))
                                            (contains? (set (map second (filter #(= (first %) u) facts-20)))
                                                       (if (= r r1) r2 r1))))
                                     rbac/DEFAULT-SOD-PAIRS))
                             facts-20)
        sod-ok-60? (not-any? (fn [[u r]]
                               (some (fn [[r1 r2]]
                                       (and (or (= r r1) (= r r2))
                                            (contains? (set (map second (filter #(= (first %) u) facts-60)))
                                                       (if (= r r1) r2 r1))))
                                     rbac/DEFAULT-SOD-PAIRS))
                             facts-60)
        p0a-pass? (and shared-prefix? admin-cap-ok? sod-ok-20? sod-ok-60? (seq facts-20) (seq facts-60))
        p0a-report {:gate :P0a
                    :status (if p0a-pass? :pass :fail)
                    :shared-prefix? shared-prefix?
                    :admin-cap-ok? admin-cap-ok?
                    :sod-ok-20? sod-ok-20?
                    :sod-ok-60? sod-ok-60?
                    :facts-20-count (count facts-20)
                    :facts-60-count (count facts-60)}]
    (println (format "  P0a Status: %s (Shared prefix: %s, Invariants preserved at T=20 and T=60)"
                     (if p0a-pass? "PASS" "FAIL") shared-prefix?))

    ;; P0b: Trap rejection
    (println "\n[Gate P0b] Testing trap rejection directly against schema gate...")
    (let [kb (rbac/create-initial-rbac-kb)
          res-sod (kb/commit-with-schema kb [:assert [:has_role :bob :auditor]])
          res-card (let [k1 (:committed (kb/commit-with-schema kb [:assert [:has_role :heidi :admin]]))
                         k2 (:committed (kb/commit-with-schema k1 [:assert [:has_role :ivan :admin]]))]
                     (kb/commit-with-schema k2 [:assert [:has_role :judy :admin]]))
          res-dom (kb/commit-with-schema kb [:assert [:has_role :eve :viewer]])
          res-noop-assert (kb/commit-with-schema kb [:assert [:has_role :alice :developer]])
          res-noop-retract (kb/commit-with-schema kb [:retract [:has_role :heidi :deployer]])
          sod-rej? (= :separation-of-duty (get-in res-sod [:rejected :violation]))
          card-rej? (= :cardinality (get-in res-card [:rejected :violation]))
          dom-rej? (= :domain (get-in res-dom [:rejected :violation]))
          noop-assert-rej? (= :no-op (get-in res-noop-assert [:rejected :violation]))
          noop-retract-rej? (= :no-op (get-in res-noop-retract [:rejected :violation]))
          p0b-pass? (and sod-rej? card-rej? dom-rej? noop-assert-rej? noop-retract-rej?)
          p0b-report {:gate :P0b
                      :status (if p0b-pass? :pass :fail)
                      :sod-rejection? sod-rej?
                      :cardinality-rejection? card-rej?
                      :domain-rejection? dom-rej?
                      :noop-assert-rejection? noop-assert-rej?
                      :noop-retract-rejection? noop-retract-rej?}]
      (println (format "  P0b Status: %s (All 4 trap categories correctly rejected naming violation)"
                       (if p0b-pass? "PASS" "FAIL")))

      ;; P0c: Write-arm emission smoke test (5 dry-run observations)
      (println "\n[Gate P0c] Running 5 dry-run prompt emissions for write-arm proposals...")
      (let [smoke-cases [{:desc "Grant developer to Judy"
                          :obs "Ticket update: grant Judy the developer role."}
                         {:desc "Grant viewer to Frank"
                          :obs "Ticket update: grant Frank the viewer role."}
                         {:desc "Revoke viewer from Erin"
                          :obs "HR update: revoke Erin from the viewer role."}
                         {:desc "Trap: grant Bob auditor"
                          :obs "Internal request: grant Bob the auditor role."}
                         {:desc "Ambiguous: oncall to Erin or Frank"
                          :obs "Ticket #412 scan is blurry: role oncall was granted to either Erin or Frank."}]
            initial-snapshot (rbac/render-snapshot kb 0)
            p0c-runs
            (mapv (fn [{:keys [desc obs]}]
                    (let [prompt-str (make-h-step-prompt initial-snapshot obs)
                          model-part (generate-model-response session prompt-str)
                          parsed (rbac/parse-proposal model-part)
                          has-proposal? (some? parsed)]
                      (println (format "  Smoke: %s -> Parsed op? %s (%s) [raw: %s]"
                                       desc has-proposal? (or (:op parsed) "none") (pr-str model-part)))
                      {:desc desc :emitted model-part :parsed parsed :has-proposal? has-proposal?}))
                  smoke-cases)
            p0c-successes (count (filter :has-proposal? p0c-runs))
            p0c-pass? (>= p0c-successes 3)
            p0c-report {:gate :P0c
                        :status (if p0c-pass? :pass :fail)
                        :successes p0c-successes
                        :total (count smoke-cases)
                        :runs p0c-runs}]
        (println (format "  P0c Status: %s (%d / %d emitted well-formed proposal syntax)"
                         (if p0c-pass? "PASS" "FAIL") p0c-successes (count smoke-cases)))

        (let [phase0-summary {:phase0-passed? (and p0a-pass? p0b-pass? p0c-pass?)
                              :gates {:P0a p0a-report
                                      :P0b p0b-report
                                      :P0c p0c-report}}]
          (.mkdirs (io/file out-dir))
          (spit (io/file out-dir "phase0.edn") (with-out-str (pprint phase0-summary)))
          (println (format "\nSaved Phase 0 telemetry to [%s/phase0.edn]" out-dir))
          phase0-summary)))))

;; ==============================================================================
;; 3. Interrogation QA Suite
;; ==============================================================================

(def SCRIPTED-INTERROGATION-QUESTIONS
  [{:id :q1 :text "Does Carol hold the auditor role?" :query [:has_role :carol :auditor]}
   {:id :q2 :text "Does Bob hold the auditor role?" :query [:has_role :bob :auditor]}
   {:id :q3 :text "Does Dan hold the admin role?" :query [:has_role :dan :admin]}
   {:id :q4 :text "Does Alice hold the developer role?" :query [:has_role :alice :developer]}
   {:id :q5 :text "Does Bob hold the deployer role?" :query [:has_role :bob :deployer]}
   {:id :q6 :text "Does Mallory hold the admin role?" :query [:has_role :mallory :admin]}
   {:id :q7 :text "Does Eve hold the developer role?" :query [:has_role :eve :developer]}
   {:id :q8 :text "Does Heidi hold the deployer role?" :query [:has_role :heidi :deployer]}
   {:id :q9 :text "Does Grace hold the developer role?" :query [:has_role :grace :developer]}
   {:id :q10 :text "Does Erin hold the oncall role?" :query [:has_role :erin :oncall]}])

(defn evaluate-interrogation-qa
  [session cell-type kb-or-history oracle-kb]
  (mapv
   (fn [{:keys [id text query]}]
     (let [oracle-has? (contains? (or (get-in oracle-kb [:facts :has_role]) #{}) (rest query))
           expected-ans (if oracle-has? "Yes" "No")
           prompt (if (= cell-type :b0)
                    (make-b0-interrogation-prompt kb-or-history text)
                    (make-h-interrogation-prompt (rbac/render-snapshot kb-or-history) text))
           clean-ans (generate-model-response session prompt)
           match? (if oracle-has?
                    (boolean (re-find #"(?i)\b(yes|true|holds)\b" clean-ans))
                    (boolean (re-find #"(?i)\b(no|false|not|neither)\b" clean-ans)))]
       {:id id
        :question text
        :expected expected-ans
        :response clean-ans
        :match? match?}))
   SCRIPTED-INTERROGATION-QUESTIONS))

;; ==============================================================================
;; 4. Horizon Reduce Loop (Cell H, Cell B0, Cell B1)
;; ==============================================================================

(defn run-cell-h
  "Runs Cell H: Verified KB Accumulation Reduce Loop.
   At each step t:
   S_{t+1} = verified_commit(S_t, proposals_t)"
  [session stream]
  (println (format "\n--- Running Cell H (Verified KB Reduce) over T=%d ---" (count stream)))
  (let [initial-kb (rbac/create-initial-rbac-kb)]
    (loop [remaining stream
           cur-kb initial-kb
           step-logs []]
      (if (empty? remaining)
        {:final-kb cur-kb
         :step-logs step-logs}
        (let [event (first remaining)
              t (:step event)
              obs (:text event)
              snapshot-str (rbac/render-snapshot cur-kb t)
              prompt-str (make-h-step-prompt snapshot-str obs)
              t0 (System/nanoTime)
              model-part (generate-model-response session prompt-str)
              t1 (System/nanoTime)
              decode-ms (/ (double (- t1 t0)) 1000000.0)
              proposal (rbac/parse-proposal model-part)
              commit-t0 (System/nanoTime)
              commit-res (when proposal
                           (kb/commit-with-schema cur-kb proposal))
              commit-t1 (System/nanoTime)
              commit-us (/ (double (- commit-t1 commit-t0)) 1000.0)
              next-kb (if-let [new-kb (:committed commit-res)]
                        new-kb
                        cur-kb)
              step-log {:step t
                        :event-kind (:kind event)
                        :valid? (:valid? event)
                        :decode-ms decode-ms
                        :commit-us commit-us
                        :proposal proposal
                        :committed? (boolean (:committed commit-res))
                        :violation (get-in commit-res [:rejected :violation])
                        :facts-count (count (get-in next-kb [:facts :has_role]))}]
          (when (or (zero? (rem t 10)) (= t (dec (count stream))))
            (println (format "  Step %2d/%2d | event: %-16s | prop: %-7s | committed: %-5s | facts: %2d | decode: %5.1fms | gate: %4.1fµs"
                             t (count stream) (name (:kind event))
                             (name (or (:op proposal) :none))
                             (:committed? step-log)
                             (:facts-count step-log)
                             decode-ms commit-us)))
          (recur (rest remaining) next-kb (conj step-logs step-log)))))))

(defn run-cell-b1
  "Runs Cell B1: Unverified KB Accumulation Baseline.
   Proposals from the model are committed raw into the KB without gate checks."
  [session stream]
  (println (format "\n--- Running Cell B1 (Unverified KB Accumulation) over T=%d ---" (count stream)))
  (let [initial-kb (rbac/create-initial-rbac-kb)]
    (loop [remaining stream
           cur-kb initial-kb
           step-logs []]
      (if (empty? remaining)
        {:final-kb cur-kb
         :step-logs step-logs}
        (let [event (first remaining)
              t (:step event)
              obs (:text event)
              snapshot-str (rbac/render-snapshot cur-kb t)
              prompt-str (make-h-step-prompt snapshot-str obs)
              model-part (generate-model-response session prompt-str)
              proposal (rbac/parse-proposal model-part)
              next-kb (if proposal
                        (case (:op proposal)
                          :assert
                          (let [fact (:fact proposal)]
                            (update-in cur-kb [:facts (first fact)] (fnil conj #{}) (vec (rest fact))))
                          :retract
                          (let [fact (:fact proposal)]
                            (update-in cur-kb [:facts (first fact)] (fnil disj #{}) (vec (rest fact))))
                          :ambiguous
                          (update cur-kb :disjunctions conj {:id (count (:disjunctions cur-kb))
                                                             :branches (vec (map (fn [b] #{b}) (:branches proposal)))})
                          :disambiguate
                          (let [fact (:chosen-fact proposal)]
                            (-> cur-kb
                                (update :disjunctions (fn [ds] (filterv #(not= (:id %) (:disjunction-id proposal)) ds)))
                                (update-in [:facts (first fact)] (fnil conj #{}) (vec (rest fact)))))
                          cur-kb)
                        cur-kb)
              step-log {:step t
                        :event-kind (:kind event)
                        :proposal proposal
                        :facts-count (count (get-in next-kb [:facts :has_role]))}]
          (when (or (zero? (rem t 10)) (= t (dec (count stream))))
            (println (format "  Step %2d/%2d | event: %-16s | prop: %-7s | facts: %2d"
                             t (count stream) (name (:kind event))
                             (name (or (:op proposal) :none))
                             (:facts-count step-log))))
          (recur (rest remaining) next-kb (conj step-logs step-log)))))))

(defn count-violating-facts
  "Verifies whether any trap facts (SoD violation, admin cap violation, undeclared entity) are present."
  [kb]
  (let [facts (or (get-in kb [:facts :has_role]) #{})
        declared (set (keys (:entities kb)))
        undeclared-count (count (remove (fn [[u r]] (and (contains? declared u) (contains? declared r))) facts))
        by-user (group-by first facts)
        sod-count (reduce (fn [acc [_ user-facts]]
                            (let [roles (set (map second user-facts))]
                              (+ acc (count (filter (fn [[r1 r2]] (and (contains? roles r1) (contains? roles r2)))
                                                    rbac/DEFAULT-SOD-PAIRS)))))
                          0
                          by-user)
        admins (count (filter #(= (second %) :admin) facts))
        card-count (if (> admins rbac/DEFAULT-ADMIN-CAP)
                     (- admins rbac/DEFAULT-ADMIN-CAP)
                     0)]
    (+ undeclared-count sod-count card-count)))

;; ==============================================================================
;; 5. Top-Level Main CLI Entrypoint
;; ==============================================================================

(defn -main [& args]
  (let [parsed-cli (gemma4-inf/parse-cli-args args)
        opts (assoc parsed-cli
                    :temperature 0.0
                    :max-new-tokens 64
                    :max-seq-len 768
                    :method :vram-loop
                    :vram-loop? true)
        out-dir (or (some (fn [[k v]] (when (= k "--out-dir") v))
                          (partition 2 (gemma4-inf/parse-cli-args args)))
                    "catalog/agent-deliberation/e23-state-reduce")
        _ (.mkdirs (io/file out-dir))
        _ (println "\n========================================================")
        _ (println "=== Experiment E23: The Reduce (RBAC Verified Loop) ===")
        _ (println "========================================================")
        session (gemma4-inf/init-agent-vram-session opts 768)]

    (try
      ;; Phase 0
      (let [p0-summary (run-phase0-verification session out-dir)]
        (when-not (:phase0-passed? p0-summary)
          (println "\n[FATAL] Phase 0 verification failed! Halting.")
          (System/exit 1))

        ;; Generate streams
        (println "\nGenerating scripted RBAC event streams (seed=42, trap-rate=0.25)...")
        (let [stream-60 (rbac/generate-rbac-stream {:seed 42 :length 60 :trap-rate 0.25})
              stream-20 (subvec stream-60 0 20)
              init-kb (rbac/create-initial-rbac-kb)
              oracle-20 (rbac/fold-oracle-stream init-kb stream-20)
              oracle-60 (rbac/fold-oracle-stream init-kb stream-60)
              stream-telemetry {:stream-20-count (count stream-20)
                                :stream-60-count (count stream-60)
                                :events-20 stream-20
                                :events-60 stream-60
                                :oracle-20-facts (vec (get-in oracle-20 [:facts :has_role]))
                                :oracle-60-facts (vec (get-in oracle-60 [:facts :has_role]))}]
          (spit (io/file out-dir "stream.edn") (with-out-str (pprint stream-telemetry)))
          (println (format "Saved event stream and oracle ground truth to [%s/stream.edn]" out-dir))

          ;; Execute Cell H (T=20 and T=60)
          (let [res-h-20 (run-cell-h session stream-20)
                res-h-60 (run-cell-h session stream-60)
                h-f1-20 (rbac/evaluate-snapshot-f1 oracle-20 (:final-kb res-h-20))
                h-f1-60 (rbac/evaluate-snapshot-f1 oracle-60 (:final-kb res-h-60))
                h-violating-20 (count-violating-facts (:final-kb res-h-20))
                h-violating-60 (count-violating-facts (:final-kb res-h-60))
                horizon-drift (Math/abs (- (:f1 h-f1-60) (:f1 h-f1-20)))

                ;; Execute Cell B1 (Unverified)
                res-b1-20 (run-cell-b1 session stream-20)
                res-b1-60 (run-cell-b1 session stream-60)
                b1-f1-20 (rbac/evaluate-snapshot-f1 oracle-20 (:final-kb res-b1-20))
                b1-f1-60 (rbac/evaluate-snapshot-f1 oracle-60 (:final-kb res-b1-60))
                b1-violating-20 (count-violating-facts (:final-kb res-b1-20))
                b1-violating-60 (count-violating-facts (:final-kb res-b1-60))

                ;; Execute Cell B0 (In-context interrogation)
                _ (println "\n--- Running Cell B0 Interrogation (In-Context Baseline) ---")
                b0-qa-20 (evaluate-interrogation-qa session :b0 stream-20 oracle-20)
                b0-qa-60 (evaluate-interrogation-qa session :b0 stream-60 oracle-60)
                b0-acc-20 (/ (double (count (filter :match? b0-qa-20))) (count b0-qa-20))
                b0-acc-60 (/ (double (count (filter :match? b0-qa-60))) (count b0-qa-60))

                ;; Interrogate Cell H
                _ (println "\n--- Running Cell H Interrogation (KB Snapshot QA) ---")
                h-qa-20 (evaluate-interrogation-qa session :h (:final-kb res-h-20) oracle-20)
                h-qa-60 (evaluate-interrogation-qa session :h (:final-kb res-h-60) oracle-60)
                h-acc-20 (/ (double (count (filter :match? h-qa-20))) (count h-qa-20))
                h-acc-60 (/ (double (count (filter :match? h-qa-60))) (count h-qa-60))

                ;; Check ambiguity lifecycle
                amb-step-log (first (filter #(= (:step %) 5) (:step-logs res-h-20)))
                disamb-step-log (first (filter #(= (:step %) 12) (:step-logs res-h-20)))
                ambiguity-lifecycle-ok? (and (= (:op (:proposal amb-step-log)) :ambiguous)
                                             (= (:op (:proposal disamb-step-log)) :disambiguate)
                                             (:committed? amb-step-log)
                                             (:committed? disamb-step-log))

                final-results
                {:experiment "E23 — The Reduce (RBAC)"
                 :date "2026-09-17"
                 :acceptance-criteria
                 {:gate-hold-h-t20 {:violating-facts h-violating-20 :pass? (zero? h-violating-20)}
                  :gate-hold-h-t60 {:violating-facts h-violating-60 :pass? (zero? h-violating-60)}
                  :snapshot-f1-h-t20 {:f1 (:f1 h-f1-20) :threshold 0.90 :pass? (>= (:f1 h-f1-20) 0.90)}
                  :snapshot-f1-h-t60 {:f1 (:f1 h-f1-60) :threshold 0.90 :pass? (>= (:f1 h-f1-60) 0.90)}
                  :horizon-flatness {:drift horizon-drift :threshold 0.05 :pass? (<= horizon-drift 0.05)}
                  :ambiguity-lifecycle {:pass? ambiguity-lifecycle-ok?}}
                 :metrics
                 {:cell-H {:t20 {:f1 h-f1-20 :qa-acc h-acc-20 :violating h-violating-20}
                           :t60 {:f1 h-f1-60 :qa-acc h-acc-60 :violating h-violating-60}}
                  :cell-B1 {:t20 {:f1 b1-f1-20 :violating b1-violating-20}
                            :t60 {:f1 b1-f1-60 :violating b1-violating-60}}
                  :cell-B0 {:t20 {:qa-acc b0-acc-20}
                            :t60 {:qa-acc b0-acc-60}}}
                 :logs {:h-t20 (:step-logs res-h-20)
                        :h-t60 (:step-logs res-h-60)
                        :b1-t20 (:step-logs res-b1-20)
                        :b1-t60 (:step-logs res-b1-60)}}]

            (spit (io/file out-dir "results.edn") (with-out-str (pprint final-results)))
            (println (format "\nSaved final experiment results to [%s/results.edn]" out-dir))

            (println "\n========================================================")
            (println "=== Pre-Registered Acceptance Criteria Evaluation ===")
            (println "========================================================")
            (println (format "1. Gate-Hold (Violating facts in H = 0): T=20: %d | T=60: %d -> %s"
                             h-violating-20 h-violating-60
                             (if (and (zero? h-violating-20) (zero? h-violating-60)) "PASS" "FAIL")))
            (println (format "2. Snapshot F1 (H >= 0.90): T=20: %.4f | T=60: %.4f -> %s"
                             (:f1 h-f1-20) (:f1 h-f1-60)
                             (if (and (>= (:f1 h-f1-20) 0.90) (>= (:f1 h-f1-60) 0.90)) "PASS" "FAIL")))
            (println (format "3. Horizon Flatness (|F1(60) - F1(20)| <= 0.05): drift = %.4f -> %s"
                             horizon-drift (if (<= horizon-drift 0.05) "PASS" "FAIL")))
            (println (format "4. B0 Degradation (Context pressure): T=20: %.1f%% -> T=60: %.1f%%"
                             (* 100.0 b0-acc-20) (* 100.0 b0-acc-60)))
            (println (format "5. Ambiguity Lifecycle: %s" (if ambiguity-lifecycle-ok? "PASS" "FAIL")))
            (println (format "6. Unverified B1 Comparison: T=60 Violating facts = %d | F1 = %.4f"
                             b1-violating-60 (:f1 b1-f1-60)))
            (println "========================================================\n"))))
      (finally
        (gemma4-inf/close-agent-session! session)))))
