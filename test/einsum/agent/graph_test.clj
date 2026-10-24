(ns einsum.agent.graph-test
  "Unit and generative property tests for Declarative Coding Agent and Autoregressive Decoder Graphs."
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.string :as str]
            [einsum.agent.core :as agent-core]
            [einsum.agent.graph :as graph]
            [einsum.graph.binder :as binder]
            [einsum.graph.capability :as cap]
            [einsum.graph.runtime :as runtime]
            [einsum.logic.agent.state-tracker :as state-tracker]
            [einsum.models.gemma4.runtime :as gemma4-rt]
            [einsum.runtime.tokenizer.protocol :as tok-proto]))

;; -----------------------------------------------------------------------------
;; 1. Property Test: Coding Agent Turn Bounds and Event-Log Hash Chain
;; -----------------------------------------------------------------------------

(defspec prop-coding-agent-graph-turn-bounds-and-hash-chain
  40
  (prop/for-all [max-turns (gen/choose 2 5)
                 max-errors (gen/choose 1 3)
                 tool-turn-count (gen/choose 0 3)]
                (let [scripted-turns (vec (concat (repeat tool-turn-count "<|tool_call>call:eval_clojure{code:\"(+ 1 2)\"}<tool_call|>")
                                                  ["Final answer: 3"]))
                      responses-atom (atom scripted-turns)
                      mock-gen (fn [_prompt]
                                 (let [r (first @responses-atom)]
                                   (when (seq @responses-atom)
                                     (swap! responses-atom subvec 1))
                                   (or r "Default answer")))
                      mock-tool (fn [_ctx _code _turn]
                                  {:status :success :output "3"})
                      opts {:max-turns max-turns
                            :max-consecutive-errors max-errors
                            :tool-eval-fn mock-tool
                            :generate-fn mock-gen
                            :nudge-on-blind-solution? false}
                      graph-ast (graph/coding-agent-graph opts)
                      dev-ast (graph/default-agent-device opts)
                      bound (binder/bind-agent graph-ast dev-ast opts)
                      executed (-> bound
                                   (runtime/enqueue-input :prompt-in "Calculate 1+2")
                                   runtime/run-graph!)
                      final-out (first (get-in executed [:outbox :final-out]))
                      event-log (:event-log executed)]
                  (and (map? final-out)
                       (string? (:final-answer final-out))
                       (<= (count (:turn-telemetry final-out)) max-turns)
                       (runtime/verify-event-log-chain event-log)
                       (pos? (count event-log))))))

;; -----------------------------------------------------------------------------
;; 2. Property Test: Autoregressive Decoder Greedy Stopping
;; -----------------------------------------------------------------------------

(defspec prop-autoregressive-decoder-graph-greedy-stopping
  40
  (prop/for-all [vocab-size (gen/choose 5 20)
                 eos-id (gen/choose 1 4)
                 max-new-tokens (gen/choose 3 8)
                 eos-pos (gen/choose 1 10)]
                (let [step-counter (atom 0)
                      mock-gpu (fn [_in-port _payload _cas-store]
                                 (let [step (swap! step-counter inc)
                                       target-tok (if (= step eos-pos)
                                                    eos-id
                                                    (let [cand (mod (inc step) vocab-size)]
                                                      (if (= cand eos-id) (mod (+ cand 2) vocab-size) cand)))
                                       logits (float-array vocab-size)]
                                   (aset logits (int target-tok) (float 100.0))
                                   {:logits [logits] :out [logits]}))
                      opts {:max-new-tokens max-new-tokens
                            :eos-id eos-id}
                      decoder-ast (graph/autoregressive-decoder-graph opts)
                      device-ast [:device {:id :mock-gpu-dev}
                                  [:cap {:id :gpu :type :compute/einsum}]]
                      bound (binder/bind-agent decoder-ast device-ast {:capabilities {:gpu mock-gpu}})
                      executed (-> bound
                                   (runtime/enqueue-input :tokens-in [10 20])
                                   runtime/run-graph!)
                      tokens-out (first (get-in executed [:outbox :tokens-out]))
                      expected-tokens
                      (loop [step 1
                             acc []]
                        (if (> step max-new-tokens)
                          acc
                          (if (= step eos-pos)
                            acc
                            (let [cand (mod (inc step) vocab-size)
                                  tok (if (= cand eos-id) (mod (+ cand 2) vocab-size) cand)]
                              (recur (inc step) (conj acc tok))))))]
                  (and (== 1 (count (get-in executed [:outbox :tokens-out])))
                       (= expected-tokens tokens-out)))))

(deftest test-autoregressive-decoder-on-real-compute-einsum-capability
  (testing "Autoregressive decoder graph bound to real :compute/einsum capability (:forward port emitting {:logits [out] :out [out]}) emits [1 2 3 4] once"
    (let [step-counter (atom 0)
          ;; Create a real :compute/einsum capability instance whose :forward handler emits {:logits [out] :out [out]}
          real-einsum-cap (cap/start-cap!
                           (cap/build-capability
                            {:id :gpu
                             :type :compute/einsum
                             :backend :interpreter
                             :ast {:name "step_logits"
                                   :invars [[:x [:tensor [1] :f32]]]
                                   :outvars [:y]
                                   :known-shapes {}
                                   :eqns [{:op :ir/add :invars [:x :x] :outvars [:y]}]}}
                            {}))
          ;; Wrap invoke-fn to feed a step-dependent 10-element float array into the real :compute/einsum :forward port
          wrapped-cap (update real-einsum-cap :invoke-fn
                              (fn [orig-invoke]
                                (fn [this in-port _payload cas-store]
                                  (let [step (swap! step-counter inc)
                                        in-arr (float-array 10)]
                                    (aset in-arr (int step) (float 25.0))
                                    (orig-invoke this in-port {:inputs [in-arr]
                                                               :ast {:name "step_logits_10"
                                                                     :invars [[:x [:tensor [10] :f32]]]
                                                                     :outvars [:y]
                                                                     :known-shapes {}
                                                                     :eqns [{:op :ir/add :invars [:x :x] :outvars [:y]}]}}
                                                 cas-store)))))
          dev-ast [:device {:id :gpu-dev}
                   [:cap {:id :gpu :type :compute/einsum}]]
          decoder-ast (graph/autoregressive-decoder-graph {:max-new-tokens 4 :eos-id 99})
          bound (binder/bind-agent decoder-ast dev-ast {:capabilities {:gpu wrapped-cap}})
          executed (-> bound
                       (runtime/enqueue-input :tokens-in [100 200])
                       runtime/run-graph!)
          outbox (:outbox executed)
          tokens-out (:tokens-out outbox)]
      (is (= 1 (count tokens-out)))
      (is (= [1 2 3 4] (first tokens-out))))))

;; -----------------------------------------------------------------------------
;; 3. Unit Tests: Multi-Turn Tool Call, Verification Gate, and Final Answer
;; -----------------------------------------------------------------------------

(deftest test-multi-turn-tool-call-and-verification-gate
  (testing "Multi-turn tool call executes via sandbox, passes verification gate, and produces final answer"
    (let [scripted-responses (atom ["<|tool_call>call:eval_clojure{code:\"(+ 40 2)\"}<tool_call|>"
                                    "The calculated answer is 42."])
          mock-gen (fn [_prompt]
                     (let [r (first @scripted-responses)]
                       (swap! scripted-responses subvec 1)
                       r))
          mock-eval (fn [_code-str _timeout]
                      {:status :success :output "42"})
          res (graph/run-coding-agent-graph mock-gen
                                            "Calculate 40 + 2"
                                            {:eval-fn mock-eval
                                             :nudge-on-blind-solution? false
                                             :max-turns 4})]
      (is (= "The calculated answer is 42." (:final-answer res)))
      (let [meta-info (meta res)]
        (is (= 2 (count (:turn-telemetry meta-info))))
        (is (= :success (:status (first (:turn-telemetry meta-info)))))
        (is (= :final-answer (:status (second (:turn-telemetry meta-info)))))
        (is (runtime/verify-event-log-chain (:event-log meta-info)))))))

;; -----------------------------------------------------------------------------
;; 4. Unit Tests: Blind Solution Nudge Short-Circuiting
;; -----------------------------------------------------------------------------

(deftest test-blind-solution-nudge-short-circuiting
  (testing "Blind answer without tool call triggers system nudge observation"
    (let [scripted-responses (atom ["The answer is 42 without any testing."
                                    "Final verified answer after testing: 42."])
          mock-gen (fn [_prompt]
                     (let [r (first @scripted-responses)]
                       (swap! scripted-responses subvec 1)
                       r))
          res (graph/run-coding-agent-graph mock-gen
                                            "Please compute something"
                                            {:nudge-on-blind-solution? true
                                             :max-turns 4})]
      (is (= "Final verified answer after testing: 42." (:final-answer res)))
      (let [meta-info (meta res)
            telemetry (:turn-telemetry meta-info)]
        (is (>= (count telemetry) 2))
        (is (= :nudged (:status (first telemetry))))))))

;; -----------------------------------------------------------------------------
;; 5. Unit Tests: Early Exit on Public Test Pass
;; -----------------------------------------------------------------------------

(deftest test-early-exit-on-public-pass
  (testing "Early exit on public test pass terminates agent loop at Turn 1"
    (let [session {:opts {:system "System prompt"
                          :max-turns 5
                          :scripted-responses ["<|tool_call>call:eval_clojure{code:\"(defn my-inc [x] (inc x))\"}<tool_call|>"
                                               "Turn 2 that should never be executed"]
                          :tool-eval-fn (fn [_ctx _code _turn]
                                          {:status :success
                                           :output "[Public tests: 2/2 passed.]"
                                           :early-exit? true})
                          :quiet true}}
          transcript (agent-core/run-agent-loop session "Implement my-inc")]
      (is (= 2 (count transcript)))
      (is (= :model (:role (first transcript))))
      (is (= :tool (:role (second transcript))))
      (is (true? (:early-exit? (meta transcript))))
      (is (= :public-pass (:early-exit-reason (meta transcript))))
      (is (some #(and (= [:verification-gate :early-exit] (:event/source %))
                      (= [:turn-loop :early-exit-obs] (:event/target %)))
                (:event-log (meta transcript)))))))

;; -----------------------------------------------------------------------------
;; 6. Unit Tests: Fractal Sub-Agent Embedding
;; -----------------------------------------------------------------------------

(deftest test-fractal-subagent-decoder-embedding
  (testing "Outer agent graph embedding autoregressive-decoder-graph as a child [:agent ...] node"
    (let [decoder-hiccup (graph/autoregressive-decoder-graph {:max-new-tokens 3 :eos-id 99})
          outer-agent
          [:agent {:id :llm-orchestrator
                   :ins [:prompt-tokens]
                   :outs [:final-tokens]
                   :requires {:gpu {:type :compute/einsum}}}
           decoder-hiccup
           [:conn [:agent/in :prompt-tokens] [:autoregressive-decoder :tokens-in]]
           [:conn [:autoregressive-decoder :tokens-out] [:agent/out :final-tokens]]]
          step-counter (atom 0)
          mock-gpu (fn [_in-port _payload _cas-store]
                     (let [step (swap! step-counter inc)
                           logits (float-array 10)]
                       (aset logits (int (mod step 10)) (float 10.0))
                       {:logits [logits] :out [logits]}))
          dev-ast [:device {:id :gpu-dev}
                   [:cap {:id :gpu :type :compute/einsum}]]
          bound (binder/bind-agent outer-agent dev-ast {:capabilities {:gpu mock-gpu}})
          executed (-> bound
                       (runtime/enqueue-input :prompt-tokens [1 2 3])
                       runtime/run-graph!)
          out (first (get-in executed [:outbox :final-tokens]))]
      (is (vector? out))
      (is (= 3 (count out)))
      (testing "Deterministic replay-graph of outer agent embedding decoder sub-agent"
        (let [replayed (runtime/replay-graph outer-agent (:event-log executed))]
          (is (= (:outbox executed) (:outbox replayed)))
          (is (= (get-in executed [:node-states :autoregressive-decoder :outbox])
                 (get-in replayed [:node-states :autoregressive-decoder :outbox])))
          (is (runtime/verify-event-log-chain (:event-log replayed))))))))

;; -----------------------------------------------------------------------------
;; 7. Generative Property: In-VRAM State Tracker Capability Parity
;; -----------------------------------------------------------------------------

(defspec prop-state-tracker-capability-host-parity 30
  (prop/for-all [dep-count (gen/choose 1 6)
                 mod-count (gen/choose 1 4)]
                (let [n 8
                      cand-pairs (for [x (range n) y (range n) :when (not= x y)] [x y])
                      deps (set (take dep-count (shuffle cand-pairs)))
                      cand-tests (for [t (range n) x (range n) :when (not= t x)] [t x])
                      tests (set (take dep-count (shuffle cand-tests)))
                      mod-cand (shuffle (range n))
                      modified (set (take mod-count mod-cand))
                      expected-affected (state-tracker/compute-affected-entities-host deps modified n)
                      agent-ast (graph/state-tracker-agent-graph)
                      dev-ast [:device {:id :tracker-dev}
                               [:cap {:id :state-tracker
                                      :type :compute/state-tracker
                                      :n n
                                      :deps deps
                                      :tests tests
                                      :backend :interpreter}]]
                      bound (binder/bind-agent agent-ast dev-ast {})
                      executed (-> bound
                                   (runtime/enqueue-input :modified-in modified)
                                   runtime/run-graph!)
                      out (first (get-in executed [:outbox :invalidated-out]))]
                  (and (map? out)
                       (= expected-affected (:affected-files out))))))

;; -----------------------------------------------------------------------------
;; 8. Unit Tests: Coding Agent and State Tracker Composition
;; -----------------------------------------------------------------------------

(deftest test-coding-agent-with-state-tracker-composition
  (testing "Composed workflow executing coding-agent-graph and state-tracker-agent-graph"
    (let [coding-ast (graph/coding-agent-graph {:max-turns 2})
          tracker-ast (graph/state-tracker-agent-graph)
          composite-agent
          [:agent {:id :workflow-orchestrator
                   :ins [:user-prompt :changed-files]
                   :outs [:code-result :invalidated-tests]
                   :requires {:llm {:type :compute/einsum}
                              :sandbox {:type :eval/sci}
                              :tracker {:type :compute/state-tracker}}}
           coding-ast
           tracker-ast
           [:conn [:agent/in :user-prompt] [:coding-agent :prompt-in]]
           [:conn [:coding-agent :final-out] [:agent/out :code-result]]
           [:conn [:agent/in :changed-files] [:state-tracker-agent :modified-in]]
           [:conn [:state-tracker-agent :invalidated-out] [:agent/out :invalidated-tests]]]
          mock-gen (fn [_prompt] "Final answer: done")
          mock-eval (fn [_code _turn] {:status :success :output "done"})
          dev-ast [:device {:id :full-dev}
                   [:cap {:id :llm :type :compute/einsum :generate-fn mock-gen}]
                   [:cap {:id :sandbox :type :eval/sci :eval-fn mock-eval}]
                   [:cap {:id :tracker :type :compute/state-tracker
                          :n 8 :deps #{[0 1] [1 2]} :tests #{[5 0] [6 1]}
                          :backend :interpreter}]]
          bound (binder/bind-agent composite-agent dev-ast {})
          executed (-> bound
                       (runtime/enqueue-input :user-prompt "Implement feature")
                       (runtime/enqueue-input :changed-files #{2})
                       runtime/run-graph!)
          code-out (first (get-in executed [:outbox :code-result]))
          tracker-out (first (get-in executed [:outbox :invalidated-tests]))]
      (is (map? code-out))
      (is (= "Final answer: done" (:final-answer code-out)))
      (is (map? tracker-out))
      (is (contains? (:affected-files tracker-out) 0))
      (is (contains? (:affected-files tracker-out) 1))
      (is (contains? (:affected-files tracker-out) 2))
      (is (contains? (:affected-tests tracker-out) 5))
      (is (contains? (:affected-tests tracker-out) 6)))))

(deftest test-autoregressive-decoder-eos-ids-set-stopping
  (testing "Autoregressive decoder graph stops when encountering any token in :eos-ids set"
    (let [step-counter (atom 0)
          stop-set #{1 106 50}
          ;; Simulate emitting tokens: 42, 43, 106 (stop), 44
          tokens-seq [42 43 106 44 45]
          mock-gpu (fn [_in-port _payload _cas-store]
                     (let [idx (swap! step-counter inc)
                           tok (nth tokens-seq (dec idx) 99)
                           logits (float-array 200)]
                       (aset logits (int tok) (float 100.0))
                       {:logits [logits] :out [logits]}))
          opts {:max-new-tokens 10
                :eos-ids stop-set}
          decoder-ast (graph/autoregressive-decoder-graph opts)
          device-ast [:device {:id :mock-gpu-dev}
                      [:cap {:id :gpu :type :compute/einsum}]]
          bound (binder/bind-agent decoder-ast device-ast {:capabilities {:gpu mock-gpu}})
          executed (-> bound
                       (runtime/enqueue-input :tokens-in [10 20])
                       runtime/run-graph!)
          tokens-out (first (get-in executed [:outbox :tokens-out]))]
      (is (= [42 43] tokens-out))
      (is (= 3 @step-counter)))))

(deftest test-fake-session-decode-graph-and-deterministic-replay
  (testing "Fake session decode with :decode :graph produces sub-event-log and supports full deterministic replay"
    (let [step-counter (atom 0)
          tokens-seq [100 101 1]
          fake-tokenizer (reify
                           tok-proto/Tokenizer
                           (encode [_ _text] [10 20])
                           (decode [_ token-ids] (str/join " " token-ids))
                           (decode [_ token-ids _skip] (str/join " " token-ids))
                           (bos-id [_] 1)
                           (eos-id [_] 1))
          fake-fwd (fn [_session _payload]
                     (let [idx (swap! step-counter inc)
                           tok (nth tokens-seq (dec idx) 1)
                           logits (float-array 200)]
                       (aset logits (int tok) (float 100.0))
                       logits))
          fake-session {:tokenizer fake-tokenizer
                        :opts {:max-new-tokens 5}
                        :forward-fn fake-fwd}]
      ;; 1. Parity between :decode :host and :decode :graph on fake-session
      (reset! step-counter 0)
      (let [res-host (gemma4-rt/generate-new-tokens-and-text fake-session "Hello world" {:decode :host})
            _ (reset! step-counter 0)
            res-graph (gemma4-rt/generate-new-tokens-and-text fake-session "Hello world" {:decode :graph})
            sub-log (:sub-event-log res-graph)
            sub-hiccup (:sub-hiccup res-graph)]
        (is (= (:new-token-ids res-host) (:new-token-ids res-graph)))
        (is (= (:text res-host) (:text res-graph)))
        (is (= "100 101" (:text res-graph)))
        (is (= 2 (:new-tokens res-graph)))
        (is (= [100 101] (:new-token-ids res-graph)))
        (is (vector? sub-log))
        (is (pos? (count sub-log)))
        (is (runtime/verify-event-log-chain sub-log))

        ;; 2. Replay the sub-event log deterministically with zero capabilities
        (let [replayed (runtime/replay-graph sub-hiccup sub-log)]
          (is (= [100 101] (first (get-in replayed [:outbox :tokens-out]))))
          (is (= sub-log (:event-log replayed)))))

      ;; 3. Execute inside coding-agent-graph with :decode :graph
      (reset! step-counter 0)
      (let [agent-opts {:max-turns 1
                        :decode :graph
                        :nudge-on-blind-solution? false
                        :session fake-session}
            agent-ast (graph/coding-agent-graph agent-opts)
            dev-ast (graph/default-agent-device agent-opts)
            bound (binder/bind-agent agent-ast dev-ast agent-opts)
            executed (-> bound
                         (runtime/enqueue-input :prompt-in "Hello world")
                         runtime/run-graph!)
            parent-log (:event-log executed)
            final-out (first (get-in executed [:outbox :final-out]))
            ;; Verify that the capability step envelope recorded the sub-event-log
            cap-env (first (filter #(= (first (:event/target %)) :cap/llm) parent-log))
            sub-log-in-parent (get-in cap-env [:event/meta :sub-event-log])]
        (is (some? final-out))
        (is (string? (:final-answer final-out)))
        (is (= "100 101" (:final-answer final-out)))
        (is (seq sub-log-in-parent))
        (is (runtime/verify-event-log-chain parent-log))

        ;; 4. Replay the full parent graph deterministically
        (let [replayed-parent (runtime/replay-graph agent-ast parent-log)]
          (is (= (:node-states executed) (:node-states replayed-parent)))
          (is (= (:outbox executed) (:outbox replayed-parent)))
          (is (seq (:sub-replays replayed-parent)))))

      ;; 5. Two-turn tool-calling agent scenario with :decode :graph and deterministic sub-replay
      (let [turn1-resp "<|tool_call>call:eval_clojure{code:\"(* 17 23)\"}<tool_call|>"
            turn2-resp "The result is 391."
            tok-map {201 turn1-resp
                     202 turn2-resp}
            turn2-tokenizer (reify
                              tok-proto/Tokenizer
                              (encode [_ _text] [10 20])
                              (decode [_ token-ids]
                                (str/join "" (map #(get tok-map % (str %)) token-ids)))
                              (decode [_ token-ids _skip]
                                (str/join "" (map #(get tok-map % (str %)) token-ids)))
                              (bos-id [_] 2)
                              (eos-id [_] 1))
            t2-step-counter (atom 0)
            ;; Turn 1: 201 (tool call), 1 (eos); Turn 2: 202 (final answer), 1 (eos)
            t2-tokens [201 1 202 1]
            t2-fwd (fn [_session _payload]
                     (let [idx (swap! t2-step-counter inc)
                           tok (nth t2-tokens (dec idx) 1)
                           logits (float-array 300)]
                       (aset logits (int tok) (float 100.0))
                       logits))
            t2-session {:tokenizer turn2-tokenizer
                        :opts {:max-new-tokens 5}
                        :forward-fn t2-fwd}
            agent-opts {:max-turns 4
                        :decode :graph
                        :nudge-on-blind-solution? false
                        :session t2-session}
            agent-ast (graph/coding-agent-graph agent-opts)
            dev-ast (graph/default-agent-device agent-opts)
            bound (binder/bind-agent agent-ast dev-ast agent-opts)
            executed (-> bound
                         (runtime/enqueue-input :prompt-in "Please compute (* 17 23)")
                         runtime/run-graph!)
            parent-log (:event-log executed)
            final-out (first (get-in executed [:outbox :final-out]))]
        (is (some? final-out))
        (is (= "The result is 391." (:final-answer final-out)))
        (is (runtime/verify-event-log-chain parent-log))

        ;; 6. Replay the full parent graph deterministically with zero capabilities (both in-memory and after EDN serialization)
        (let [edn-str (pr-str parent-log)
              deserialized-parent-log (edn/read-string edn-str)
              replayed-parent (runtime/replay-graph agent-ast deserialized-parent-log)]
          (is (not (str/includes? edn-str "#object[")))
          (is (= (:node-states executed) (:node-states replayed-parent)))
          (is (= (:outbox executed) (:outbox replayed-parent)))
          (is (= 2 (count (:sub-replays replayed-parent))))
          (is (every? runtime/verify-event-log-chain (map :event-log (:sub-replays replayed-parent)))))

        ;; 7. Verify replay-graph fails loudly when :sub-event-log is present without :sub-hiccup
        (let [stripped-log (mapv (fn [env]
                                   (if (get-in env [:event/meta :sub-event-log])
                                     (update env :event/meta dissoc :sub-hiccup)
                                     env))
                                 parent-log)]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo
                                #"Cannot replay :sub-event-log without :sub-hiccup"
                                (runtime/replay-graph agent-ast stripped-log))))

        ;; 8. Verify unified build-decode-policy shared across host and graph decode paths
        (let [policy (gemma4-rt/build-decode-policy t2-session {:max-new-tokens 10 :eos-ids #{1 106 107}} [10 20] 8)]
          (is (= 5 (:max-new-tokens policy)))
          (is (= 1 (:eos-id policy)))
          (is (= #{1 106 107} (:eos-ids policy)))
          (is (true? ((:stop-token? policy) 106)))
          (is (false? ((:stop-token? policy) 201))))))))

(deftest test-multi-turn-nested-decoder-subagent-isolation
  (testing "Multi-turn nested autoregressive-decoder-graph resets :seen-tokens and :sub-event-log per turn and round-trips through pure EDN"
    (let [seen-history (atom [])
          step-counter (atom 0)
          ;; Turn 1 emits [11 12 99(eos)]; Turn 2 emits [21 22 99(eos)]
          seq-tokens [11 12 99 21 22 99]
          custom-sample-fn (fn [logits seen-tokens]
                             (swap! seen-history conj (vec seen-tokens))
                             (cond
                               (instance? (Class/forName "[F") logits)
                               (let [^floats arr logits]
                                 (first (apply max-key second (map-indexed vector arr))))
                               (and (map? logits) (contains? logits :cas/argmax))
                               (:cas/argmax logits)
                               :else 99))
          decoder-hiccup (graph/autoregressive-decoder-graph {:max-new-tokens 4
                                                              :eos-id 99
                                                              :sample-fn custom-sample-fn})
          outer-agent
          [:agent {:id :multi-turn-orchestrator
                   :ins [:prompt-tokens]
                   :outs [:final-tokens]
                   :requires {:gpu {:type :compute/einsum}}}
           decoder-hiccup
           [:conn [:agent/in :prompt-tokens] [:autoregressive-decoder :tokens-in]]
           [:conn [:autoregressive-decoder :tokens-out] [:agent/out :final-tokens]]]
          mock-gpu (fn [_in-port _payload _cas-store]
                     (let [idx (swap! step-counter inc)
                           tok (nth seq-tokens (dec idx) 99)
                           logits (float-array 128)]
                       (aset logits (int tok) (float 50.0))
                       {:logits [logits] :out [logits]}))
          dev-ast [:device {:id :gpu-dev}
                   [:cap {:id :gpu :type :compute/einsum}]]
          bound (binder/bind-agent outer-agent dev-ast {:capabilities {:gpu mock-gpu}})
          ;; Turn 1 invocation
          after-t1 (-> bound
                       (runtime/enqueue-input :prompt-tokens [1 2])
                       runtime/run-graph!)
          ;; Turn 2 invocation on the SAME outer graph instance
          after-t2 (-> after-t1
                       (runtime/enqueue-input :prompt-tokens [1 2 11 12])
                       runtime/run-graph!)
          outer-log (:event-log after-t2)
          sub-logs (vec (keep #(get-in % [:event/meta :sub-event-log]) outer-log))]
      ;; 1. Both turns produced their expected token sequences
      (is (= [[11 12] [21 22]] (get-in after-t2 [:outbox :final-tokens])))
      ;; 2. :logits-sampler reset :seen-tokens to [] at the start of Turn 2!
      ;; Turn 1 steps see: [], [11], [11 12]; Turn 2 steps see: [], [21], [21 22]
      (is (= [[] [11] [11 12] [] [21] [21 22]] @seen-history))
      ;; 3. Each turn's :sub-event-log is isolated and starts cleanly at tx = 1 and GENESIS-HASH
      (is (= 2 (count sub-logs)))
      (doseq [slog sub-logs]
        (is (= 1 (:event/tx (first slog))))
        (is (= runtime/GENESIS-HASH (:event/prev-hash (first slog))))
        (is (runtime/verify-event-log-chain slog)))
      ;; 4. Pure EDN serialization & zero-blob cross-process replay
      (let [edn-str (pr-str outer-log)
            deserialized-outer-log (edn/read-string edn-str)]
        (is (not (str/includes? edn-str "#object[")))
        (is (runtime/verify-event-log-chain deserialized-outer-log))
        (reset! seen-history [])
        (let [replayed (runtime/replay-graph outer-agent deserialized-outer-log)]
          (is (= (:outbox after-t2) (:outbox replayed)))
          (is (= 2 (count (:sub-replays replayed))))
          (is (= [[] [11] [11 12] [] [21] [21 22]] @seen-history)))))))
