(ns einsum.agent.graph
  "Declarative multi-turn coding agent and autoregressive token decoder graphs."
  (:require [clojure.string :as str]
            [einsum.agent.core :as agent-core]
            [einsum.graph.binder :as binder]
            [einsum.graph.runtime :as runtime]))

;; =============================================================================
;; 1. Multi-Turn Coding Agent Mealy Coordinator
;; =============================================================================

(defn init-turn-state
  "Initializes the Mealy turn-loop coordinator state."
  [opts]
  {:turn 1
   :max-turns (long (or (:max-turns opts) 5))
   :max-consecutive-errors (long (or (:max-consecutive-errors opts) 3))
   :consecutive-errors 0
   :system-prompt (or (:system opts) (:system-prompt opts))
   :tool-declaration (if (contains? opts :tool-declaration)
                       (:tool-declaration opts)
                       agent-core/DEFAULT-TOOL-DECLARATION)
   :tool-syntax (keyword (or (:tool-syntax opts) :native))
   :enable-thinking? (boolean (or (:enable-thinking? opts) (:thinking opts)))
   :nudge-on-blind-solution? (if (contains? opts :nudge-on-blind-solution?)
                               (boolean (:nudge-on-blind-solution? opts))
                               (if (contains? opts :nudge-on-no-tool)
                                 (boolean (:nudge-on-no-tool opts))
                                 true))
   :nudged? false
   :candidate-check-fn (or (:candidate-check-fn opts) (:candidate-fn opts))
   :compaction (:compaction opts)
   :turn-budget-warning? (boolean (:turn-budget-warning? opts))
   :history []
   :transcript []
   :turn-telemetry []
   :early-exit? false
   :pending-turn nil})

(defn turn-coordinator-step
  "Pure Mealy state reducer for the multi-turn coding agent loop:
   [:mealy {:id :turn-loop
            :ins [:prompt :model-reply :tool-obs :rejected-obs]
            :outs [:llm-req :eval-code :final :telemetry]}]."
  [state in-port payload]
  (case in-port
    :prompt
    (let [prompt-str (if (map? payload) (or (:prompt payload) "") (str (or payload "")))
          user-msg {:role :user :content prompt-str}
          history [user-msg]
          formatted-prompt (agent-core/format-agent-chat-prompt
                            (:system-prompt state)
                            history
                            8
                            (:enable-thinking? state)
                            (:tool-declaration state)
                            (:tool-syntax state))]
      [(assoc state :history history :turn 1)
       {:llm-req [formatted-prompt]}])

    :model-reply
    (let [raw-reply (if (map? payload) (or (:text payload) "") (str (or payload "")))
          model-ms (double (if (map? payload) (or (:model-ms payload) 0.0) 0.0))
          thinking (agent-core/extract-thinking-trace raw-reply)
          clean-reply (agent-core/strip-thinking-trace raw-reply)
          syntax (:tool-syntax state :native)
          tool-call (agent-core/extract-tool-call raw-reply syntax)
          tool-code (if (map? tool-call) (:code tool-call) tool-call)
          tool-name (if (map? tool-call) (or (:name tool-call) "eval_clojure") "eval_clojure")
          model-content (agent-core/sanitize-history-model-content tool-call clean-reply raw-reply)
          history-with-reply (conj (:history state) {:role :model :content model-content})
          turn (:turn state)
          max-turns (:max-turns state)
          transcript-entry {:turn turn
                            :role :model
                            :content (if (seq thinking)
                                       (str "[Thought Process]\n" thinking "\n\n[Model Response]\n" clean-reply)
                                       raw-reply)
                            :thought thinking
                            :response clean-reply
                            :raw raw-reply
                            :tool-call tool-call}
          cand-fn (or (:candidate-check-fn state) (constantly false))
          has-candidate? (boolean (cand-fn clean-reply))
          nudge? (and (nil? tool-code)
                      (agent-core/nudge-needed?
                       {:tool-code tool-code
                        :turn turn
                        :max-turns max-turns
                        :new-tokens 10
                        :opts {:nudge-on-no-tool (:nudge-on-blind-solution? state)}
                        :history (:history state)
                        :consecutive-errors (:consecutive-errors state)
                        :has-candidate? has-candidate?
                        :nudged? (:nudged? state)
                        :nudge-on-blind-solution? (:nudge-on-blind-solution? state)
                        :clean-reply clean-reply}))]
      (cond
        ;; Case 1: Final Answer (No tool call and no nudge needed)
        (and (nil? tool-code) (not nudge?))
        (let [telem-entry (cond-> {:turn turn :model-ms model-ms :tool-ms 0.0 :total-turn-ms model-ms :status :final-answer :thinking thinking :code nil}
                            (map? payload)
                            (merge (select-keys payload [:prompt-tokens :new-tokens :new-token-ids :prefill-ms :decode-ms :decode-tok-s :decode-ms-tok :total-ms])
                                   (when-let [tok-s (:decode-tok-s payload)] {:tok-per-sec tok-s})))
              final-ans (str/trim (if (seq clean-reply) clean-reply raw-reply))
              final-map {:final-answer final-ans
                         :history history-with-reply
                         :transcript (conj (:transcript state) transcript-entry)
                         :turn-telemetry (conj (:turn-telemetry state) telem-entry)
                         :early-exit? false}]
          [(assoc state
                  :history history-with-reply
                  :transcript (conj (:transcript state) transcript-entry)
                  :turn-telemetry (conj (:turn-telemetry state) telem-entry))
           {:final [final-map]
            :telemetry [telem-entry]}])

        ;; Case 3: Blind Solution Nudge
        nudge?
        (let [nudge-msg (case syntax
                          :xml "Please test your Clojure implementation by outputting code in <clojure>...</clojure>."
                          :fenced "Please test your Clojure implementation by outputting code in ```clojure ... ```."
                          "Please test your Clojure implementation by calling the eval_clojure tool.")
              telem-entry (cond-> {:turn turn :model-ms model-ms :tool-ms 0.0 :total-turn-ms model-ms :status :nudged :thinking thinking :code nil}
                            (map? payload)
                            (merge (select-keys payload [:prompt-tokens :new-tokens :new-token-ids :prefill-ms :decode-ms :decode-tok-s :decode-ms-tok :total-ms])
                                   (when-let [tok-s (:decode-tok-s payload)] {:tok-per-sec tok-s})))
              nudge-history-item {:role :user :content nudge-msg}
              nudge-transcript-item {:turn turn :role :user :content nudge-msg}
              history-with-nudge (conj history-with-reply nudge-history-item)
              compacted-history (if (:compaction state)
                                  (agent-core/compact-agent-history history-with-nudge (:compaction state))
                                  history-with-nudge)
              next-prompt (agent-core/format-agent-chat-prompt
                           (:system-prompt state)
                           compacted-history
                           8
                           (:enable-thinking? state)
                           (:tool-declaration state)
                           syntax)]
          [(assoc state
                  :history compacted-history
                  :transcript (-> (:transcript state)
                                  (conj transcript-entry)
                                  (conj nudge-transcript-item))
                  :turn (inc turn)
                  :nudged? true
                  :turn-telemetry (conj (:turn-telemetry state) telem-entry))
           {:llm-req [next-prompt]
            :telemetry [telem-entry]}])

        ;; Case 4: Execute Tool Call
        :else
        (let [pending (cond-> {:model-ms model-ms
                               :thinking thinking
                               :tool-code tool-code
                               :tool-name tool-name
                               :raw-reply raw-reply
                               :clean-reply clean-reply
                               :transcript-entry transcript-entry}
                        (map? payload)
                        (merge (select-keys payload [:prompt-tokens :new-tokens :new-token-ids :prefill-ms :decode-ms :decode-tok-s :decode-ms-tok :total-ms])
                               (when-let [tok-s (:decode-tok-s payload)] {:tok-per-sec tok-s})))]
          [(assoc state
                  :history history-with-reply
                  :pending-turn pending)
           {:eval-code [{:code tool-code :turn turn}]}])))

    (:tool-obs :rejected-obs :early-exit-obs)
    (let [eval-res (if (map? payload) payload {:status :error :output (str payload)})
          pending (:pending-turn state)
          turn (:turn state)
          early-exit? (or (= in-port :early-exit-obs) (boolean (:early-exit? eval-res)))
          err-turn? (and (not= in-port :early-exit-obs) (or (= in-port :rejected-obs) (not= :success (:status eval-res))))
          new-consec-errors (if err-turn? (inc (:consecutive-errors state)) 0)
          telem-entry (cond-> {:turn turn
                               :model-ms (or (:model-ms pending) 0.0)
                               :tool-ms (double (or (:tool-ms eval-res) 0.0))
                               :total-turn-ms (+ (double (or (:model-ms pending) 0.0)) (double (or (:tool-ms eval-res) 0.0)))
                               :status (:status eval-res)
                               :thinking (:thinking pending)
                               :code (:tool-code pending)}
                        pending
                        (merge (select-keys pending [:prompt-tokens :new-tokens :new-token-ids :prefill-ms :decode-ms :decode-tok-s :decode-ms-tok :total-ms :tok-per-sec])))
          new-telemetry (conj (:turn-telemetry state) telem-entry)
          tool-name (or (:tool-name pending) "eval_clojure")
          syntax (:tool-syntax state :fenced)
          error-budget-reached? (>= new-consec-errors (:max-consecutive-errors state))
          augmented-res (cond
                          error-budget-reached?
                          (assoc eval-res :system_note
                                 (format "Consecutive tool error limit (%d) reached. Tool execution is now disabled. Provide your final answer in plain text based on the observations collected so far without calling further tools."
                                         (:max-consecutive-errors state)))

                          (and (:turn-budget-warning? state)
                               (== turn (dec (:max-turns state)))
                               (not early-exit?))
                          (assoc eval-res :system_note "Warning: You are approaching the turn budget. Please finalize your solution in the next turn.")

                          :else
                          eval-res)
          obs-str (agent-core/format-tool-response tool-name augmented-res syntax)
          history-with-tool (conj (:history state) {:role :tool :content obs-str})
          tool-transcript-entry {:turn turn :role :tool :content obs-str}
          next-transcript (-> (:transcript state)
                              (conj (:transcript-entry pending))
                              (conj tool-transcript-entry))
          compacted-history (if (:compaction state)
                              (agent-core/compact-agent-history history-with-tool (:compaction state))
                              history-with-tool)]
      (cond
        ;; Condition 1: Early Exit (e.g. all public tests passed)
        early-exit?
        (let [final-ans (str/trim (if (seq (:clean-reply pending)) (:clean-reply pending) (or (:raw-reply pending) "")))
              final-map {:final-answer final-ans
                         :history compacted-history
                         :transcript next-transcript
                         :turn-telemetry new-telemetry
                         :early-exit? true
                         :early-exit-reason :public-pass}]
          [(assoc state
                  :history compacted-history
                  :transcript next-transcript
                  :turn-telemetry new-telemetry
                  :early-exit? true
                  :pending-turn nil)
           {:final [final-map]
            :telemetry [telem-entry]}])

        ;; Condition 2: Exceeded consecutive error budget
        error-budget-reached?
        (let [halted-ans (str "HALTED: Exceeded maximum consecutive tool errors (" (:max-consecutive-errors state) "). Last error:\n" (:output eval-res))
              final-map {:final-answer halted-ans
                         :history compacted-history
                         :transcript next-transcript
                         :turn-telemetry new-telemetry
                         :early-exit? false}]
          [(assoc state
                  :history compacted-history
                  :transcript next-transcript
                  :turn-telemetry new-telemetry
                  :consecutive-errors new-consec-errors
                  :pending-turn nil)
           {:final [final-map]
            :telemetry [telem-entry]}])

        ;; Condition 3: Reached max-turns limit
        (>= turn (:max-turns state))
        (let [final-ans (str/trim (if (seq (:clean-reply pending)) (:clean-reply pending) (or (:raw-reply pending) "")))
              final-map {:final-answer final-ans
                         :history compacted-history
                         :transcript next-transcript
                         :turn-telemetry new-telemetry
                         :early-exit? false}]
          [(assoc state
                  :history compacted-history
                  :transcript next-transcript
                  :turn-telemetry new-telemetry
                  :pending-turn nil)
           {:final [final-map]
            :telemetry [telem-entry]}])

        ;; Condition 4: Advance to next turn
        :else
        (let [next-turn (inc turn)
              next-prompt (agent-core/format-agent-chat-prompt
                           (:system-prompt state)
                           compacted-history
                           8
                           (:enable-thinking? state)
                           (:tool-declaration state)
                           syntax)]
          [(assoc state
                  :history compacted-history
                  :transcript next-transcript
                  :turn next-turn
                  :consecutive-errors new-consec-errors
                  :turn-telemetry new-telemetry
                  :pending-turn nil)
           {:llm-req [next-prompt]
            :telemetry [telem-entry]}])))))

;; =============================================================================
;; 2. Tool Observation Verification Gate
;; =============================================================================

(defn verify-tool-observation
  "Pure predicate/router for the verification gate [:verify {:id :verification-gate}].
   Routes :early-exit? true to [:early-exit obs], :status :success to [:verified obs],
   and errors/timeouts to [:rejected obs]."
  [obs]
  (cond
    (:early-exit? obs) [:early-exit obs]
    (= (:status obs) :success) [:verified obs]
    :else [:rejected obs]))

;; =============================================================================
;; 3. Declarative Coding Agent Hiccup Graph
;; =============================================================================

(defn coding-agent-graph
  "Returns the declarative Hiccup vector for the multi-turn coding agent."
  ([] (coding-agent-graph {}))
  ([opts]
   [:agent {:id :coding-agent
            :ins [:prompt-in]
            :outs [:final-out :telemetry-out]
            :requires {:llm {:type :compute/einsum}
                       :sandbox {:type :eval/sci}}}
    [:mealy {:id :turn-loop
             :init-state (init-turn-state opts)
             :ins [:prompt :model-reply :tool-obs :rejected-obs :early-exit-obs]
             :outs [:llm-req :eval-code :final :telemetry]
             :step turn-coordinator-step}]
    [:verify {:id :verification-gate
              :ins [:in]
              :outs [:verified :rejected :early-exit]
              :pred (or (:verify-fn opts) verify-tool-observation)}]
    [:conn [:agent/in :prompt-in] [:turn-loop :prompt]]
    [:conn [:turn-loop :llm-req] [:cap/llm :generate]]
    [:conn [:cap/llm :reply] [:turn-loop :model-reply]]
    [:conn [:turn-loop :eval-code] [:cap/sandbox :eval]]
    [:conn [:cap/sandbox :result] [:verification-gate :in]]
    [:conn [:verification-gate :verified] [:turn-loop :tool-obs]]
    [:conn [:verification-gate :rejected] [:turn-loop :rejected-obs]]
    [:conn [:verification-gate :early-exit] [:turn-loop :early-exit-obs]]
    [:conn [:turn-loop :final] [:agent/out :final-out]]
    [:conn [:turn-loop :telemetry] [:agent/out :telemetry-out]]]))

;; =============================================================================
;; 4. Autoregressive Decoder Fractal Sub-Agent Graph
;; =============================================================================

(defn- greedy-argmax
  [logits]
  (let [row (cond
              (and (map? logits) (contains? logits :logits)) (greedy-argmax (:logits logits))
              (and (map? logits) (contains? logits :data) (some? (:data logits))) (:data logits)
              (and (vector? logits) (seq logits) (or (vector? (first logits)) (sequential? (first logits)))) (last logits)
              :else logits)]
    (cond
      (instance? (Class/forName "[F") row)
      (let [^floats arr row
            n (alength arr)]
        (loop [i 1 max-idx 0 max-val (aget arr 0)]
          (if (< i n)
            (let [v (aget arr i)]
              (if (> v max-val)
                (recur (inc i) i v)
                (recur (inc i) max-idx max-val)))
            max-idx)))

      (sequential? row)
      (let [indexed (map-indexed vector row)]
        (first (apply max-key second indexed)))

      :else
      (long row))))

(defn autoregressive-decoder-graph
  "Returns the token-level fractal sub-agent Hiccup vector for autoregressive decoding."
  ([] (autoregressive-decoder-graph {}))
  ([opts]
   (let [eos-id (:eos-id opts)
         stop-ids (cond
                    (set? (:eos-ids opts)) (cond-> (:eos-ids opts) (some? (:eos-id opts)) (conj (:eos-id opts)))
                    (sequential? (:eos-ids opts)) (cond-> (set (:eos-ids opts)) (some? (:eos-id opts)) (conj (:eos-id opts)))
                    (some? (:eos-id opts)) #{(:eos-id opts)}
                    :else #{})
         max-new-tokens (long (or (:max-new-tokens opts) 16))]
     [:agent {:id :autoregressive-decoder
              :ins [:tokens-in]
              :outs [:tokens-out]
              :requires {:gpu {:type :compute/einsum}}}
      [:mealy {:id :seq-state
               :init-state {:pos 0
                            :max-new-tokens max-new-tokens
                            :eos-id eos-id
                            :eos-ids stop-ids
                            :generated []
                            :prompt-tokens []
                            :done? false}
               :ins [:prompt-tokens :next-token :eos-token]
               :outs [:step-in :final-seq]
               :step (fn [state in-port payload]
                       (case in-port
                         :prompt-tokens
                         (let [toks (if (vector? payload) payload (vec payload))]
                           (if (or (empty? toks) (<= (:max-new-tokens state) 0))
                             [(assoc state :prompt-tokens toks :generated [] :done? true) {:final-seq [[]]}]
                             (let [pos (count toks)]
                               [(assoc state :prompt-tokens toks :pos pos :generated [] :done? false)
                                {:step-in [{:inputs [toks] :pos pos}]}])))

                         :eos-token
                         (if (:done? state)
                           [state {}]
                           [(assoc state :done? true) {:final-seq [(:generated state)]}])

                         :next-token
                         (if (:done? state)
                           [state {}]
                           (let [token (if (number? payload) (long payload) (long (first payload)))
                                 next-gen (conj (:generated state) token)
                                 done? (>= (count next-gen) (:max-new-tokens state))]
                             (if done?
                               [(assoc state :generated next-gen :done? true)
                                {:final-seq [next-gen]}]
                               (let [next-pos (inc (:pos state))]
                                 [(assoc state :generated next-gen :pos next-pos)
                                  {:step-in [{:inputs [token] :pos next-pos}]}]))))))}]
      [:mealy {:id :logits-sampler
               :init-state {:temperature (or (:temperature opts) 0.0)
                            :seen-tokens []}
               :ins [:logits]
               :outs [:token]
               :step (fn [state _in-port payload]
                       (let [sample-fn (:sample-fn opts)
                             token (long (if sample-fn
                                           (try
                                             (sample-fn payload (:seen-tokens state))
                                             (catch clojure.lang.ArityException _
                                               (sample-fn payload)))
                                           (greedy-argmax payload)))]
                         [(update state :seen-tokens conj token) {:token [token]}]))}]
      [:verify {:id :stop-gate
                :ins [:in]
                :outs [:continue :eos]
                :pred (fn [token]
                        (if (contains? stop-ids token)
                          [:eos token]
                          [:continue token]))}]
      [:conn [:agent/in :tokens-in] [:seq-state :prompt-tokens]]
      [:conn [:seq-state :step-in] [:cap/gpu :forward]]
      [:conn [:cap/gpu :logits] [:logits-sampler :logits]]
      [:conn [:logits-sampler :token] [:stop-gate :in]]
      [:conn [:stop-gate :continue] [:seq-state :next-token]]
      [:conn [:stop-gate :eos] [:seq-state :eos-token]]
      [:conn [:seq-state :final-seq] [:agent/out :tokens-out]]])))

;; =============================================================================
;; 4.1 In-VRAM State Tracker Graph Helper
;; =============================================================================

(defn state-tracker-agent-graph
  "Returns a declarative agent graph AST for in-VRAM Datalog state tracking.
   Connects incoming modified entity sets to the :compute/state-tracker capability
   and routes invalidated entity sets to the agent output."
  ([] (state-tracker-agent-graph {}))
  ([opts]
   [:agent {:id (or (:id opts) :state-tracker-agent)
            :ins [:modified-in]
            :outs [:invalidated-out]
            :requires {:tracker (merge {:type :compute/state-tracker}
                                       (when (:tracker opts) (:tracker opts)))}}
    [:conn [:agent/in :modified-in] [:cap/tracker :step]]
    [:conn [:cap/tracker :invalidated] [:agent/out :invalidated-out]]]))

;; =============================================================================
;; 5. Default Workstation Device & Execution Helpers
;; =============================================================================

(defn default-agent-device
  "Returns a Hiccup vector defining a standard workstation device for agent execution."
  ([] (default-agent-device {}))
  ([opts]
   (let [base-caps [[:cap {:id :storage-cas
                           :type :storage/cas}]
                    [:cap {:id :einsum-llm
                           :type :compute/einsum
                           :backend (or (:backend opts) :uop-cpu)
                           :session (:session opts)
                           :generate-fn (:generate-fn opts)
                           :forward-fn (:forward-fn opts)
                           :decode (:decode opts)}]
                    [:cap {:id :sci-sandbox
                           :type :eval/sci
                           :sandbox (or (:sandbox opts) :agent)
                           :timeout-ms (or (:timeout-ms opts) 5000)
                           :eval-fn (or (:eval-fn opts) (:tool-eval-fn opts))}]
                    [:cap {:id :terminal-out
                           :type :io/terminal
                           :quiet? (if (contains? opts :quiet?) (:quiet? opts) true)}]]
         tracker-cap (when (or (:enable-state-tracker? opts) (:tracker-deps opts))
                       [:cap {:id :state-tracker
                              :type :compute/state-tracker
                              :n (or (:tracker-n opts) 16)
                              :deps (:tracker-deps opts)
                              :tests (:tracker-tests opts)
                              :backend (or (:tracker-backend opts) :interpreter)}])]
     (into [:device {:id (or (:device-id opts) :local-workstation)}]
           (if tracker-cap (conj base-caps tracker-cap) base-caps)))))

(defn run-coding-agent-graph
  "Binds and executes the declarative coding agent graph on a device.
   Returns a map containing :final-answer, :history, :transcript, :turn-telemetry, :early-exit?, and :event-log."
  ([generate-fn user-prompt]
   (run-coding-agent-graph generate-fn user-prompt {}))
  ([generate-fn user-prompt opts]
   (let [graph-ast (coding-agent-graph opts)
         dev-ast (default-agent-device (assoc opts :generate-fn generate-fn))
         bound-graph (binder/bind-agent graph-ast dev-ast opts)
         executed (-> bound-graph
                      (runtime/enqueue-input :prompt-in user-prompt)
                      runtime/run-graph!)
         final-val (first (get-in executed [:outbox :final-out]))
         telem (get-in executed [:outbox :telemetry-out])
         final-answer (or (:final-answer final-val) "")
         transcript (or (:transcript final-val) [])
         history (or (:history final-val) [])
         turn-telem (or (:turn-telemetry final-val) telem [])
         early-exit? (boolean (:early-exit? final-val))
         res-map {:final-answer final-answer
                  :transcript transcript
                  :history history
                  :turn-telemetry turn-telem
                  :early-exit? early-exit?
                  :early-exit-reason (when early-exit? :public-pass)
                  :event-log (:event-log executed)}]
     (with-meta res-map res-map))))

(defn execute-agent-session
  "Executes a session via coding-agent-graph and returns a backward-compatible transcript vector with metadata."
  [session-or-fn initial-prompt custom-sci-ctx-or-opts custom-tool-eval-fn]
  (let [loop-start-t (System/nanoTime)
        opts (cond
               (map? custom-sci-ctx-or-opts)
               (if (map? session-or-fn)
                 (merge (or (:opts session-or-fn) session-or-fn) custom-sci-ctx-or-opts)
                 custom-sci-ctx-or-opts)

               (map? session-or-fn)
               (or (:opts session-or-fn) session-or-fn)

               :else {})
        session (if (map? session-or-fn) session-or-fn {})
        scripted-atom (when-let [scripted (:scripted-responses opts)]
                        (atom (vec scripted)))
        custom-gen? (or (fn? session-or-fn)
                        (some? (:generate-fn opts))
                        (some? scripted-atom))
        gen-fn (cond
                 (fn? session-or-fn)
                 (fn [_sess prompt] (session-or-fn prompt))

                 (:generate-fn opts)
                 (:generate-fn opts)

                 scripted-atom
                 (fn [_sess _prompt]
                   (let [resp (first @scripted-atom)]
                     (when (seq @scripted-atom)
                       (swap! scripted-atom subvec 1))
                     (if (map? resp)
                       resp
                       {:text (or resp "") :prompt-tokens 10 :new-tokens 10})))

                 :else nil)
        tool-eval (or custom-tool-eval-fn (:tool-eval-fn opts) agent-core/eval-tool-code)
        sci-ctx (or (when-not (map? custom-sci-ctx-or-opts) custom-sci-ctx-or-opts)
                    (:sci-ctx opts)
                    (case (:sandbox opts)
                      :benchmark (agent-core/create-benchmark-sci-ctx)
                      (agent-core/create-agent-sci-ctx)))
        eval-wrapper (fn [& args]
                       (let [[c-str turn-num] (case (count args)
                                                3 [(nth args 1) (nth args 2)]
                                                2 (if (string? (first args))
                                                    [(first args) (second args)]
                                                    [(second args) 1])
                                                1 [(first args) 1]
                                                ["" 1])]
                         (if (= tool-eval agent-core/eval-tool-code)
                           (tool-eval sci-ctx c-str (or (:timeout-ms opts) 5000))
                           (try
                             (tool-eval sci-ctx c-str turn-num)
                             (catch clojure.lang.ArityException _
                               (try
                                 (tool-eval sci-ctx c-str)
                                 (catch clojure.lang.ArityException _
                                   (try
                                     (tool-eval c-str turn-num)
                                     (catch clojure.lang.ArityException _
                                       (tool-eval c-str))))))))))
        merged-opts (cond-> (assoc opts
                                   :eval-fn eval-wrapper
                                   :tool-eval-fn eval-wrapper)
                      custom-gen?
                      (assoc :generate-fn (fn [prompt] (gen-fn session prompt)))

                      (not custom-gen?)
                      (assoc :generate-fn nil
                             :session (assoc session :opts (merge (:opts session) opts))
                             :decode (or (:decode opts) (:decode (:opts session)) :host)))
        graph-ast (coding-agent-graph merged-opts)
        dev-ast (default-agent-device merged-opts)
        bound-graph (binder/bind-agent graph-ast dev-ast merged-opts)
        executed (-> bound-graph
                     (runtime/enqueue-input :prompt-in initial-prompt)
                     runtime/run-graph!)
        final-val (first (get-in executed [:outbox :final-out]))
        telem (get-in executed [:outbox :telemetry-out])
        transcript (vec (or (:transcript final-val) []))
        early-exit? (boolean (:early-exit? final-val))]
    (when (or (not (:quiet merged-opts)) (:profile-out merged-opts))
      (let [telem-entries (or (:turn-telemetry final-val) telem [])]
        (agent-core/print-and-save-telemetry! (atom telem-entries) loop-start-t (boolean (:quiet merged-opts)) (:profile-out merged-opts))))
    (with-meta transcript
      {:turn-telemetry (or (:turn-telemetry final-val) telem [])
       :history (or (:history final-val) [])
       :early-exit? early-exit?
       :early-exit-reason (when early-exit? :public-pass)
       :final-answer (or (:final-answer final-val) "")
       :event-log (:event-log executed)})))
