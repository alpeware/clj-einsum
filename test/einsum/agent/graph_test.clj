(ns einsum.agent.graph-test
  "Unit and generative property tests for Declarative Coding Agent and Autoregressive Decoder Graphs."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.agent.core :as agent-core]
            [einsum.agent.graph :as graph]
            [einsum.graph.binder :as binder]
            [einsum.graph.runtime :as runtime]))

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
                      tokens-out (first (get-in executed [:outbox :tokens-out]))]
                  (and (vector? tokens-out)
                       (<= (count tokens-out) max-new-tokens)
                       (or (= (count tokens-out) max-new-tokens)
                           (and (<= 0 (count tokens-out))
                                (< eos-pos max-new-tokens)))))))

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
      (is (= :public-pass (:early-exit-reason (meta transcript)))))))

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
      (is (= 3 (count out))))))
