(ns einsum.graph.runtime-test
  "Generative property and unit tests for the pure-Clojure Agent, Device, and Capability Graph Architecture."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.graph.cas :as cas]
            [einsum.graph.runtime :as graph]
            [einsum.graph.schema :as schema]
            [malli.core :as m]))

;; -----------------------------------------------------------------------------
;; Generators
;; -----------------------------------------------------------------------------

(def gen-edn-scalar
  (gen/one-of [gen/int gen/double gen/string-alphanumeric gen/keyword gen/boolean]))

(def gen-edn-value
  (gen/recursive-gen
   (fn [inner]
     (gen/one-of [(gen/vector inner 0 5)
                  (gen/map gen/keyword inner)]))
   gen-edn-scalar))

;; -----------------------------------------------------------------------------
;; 1. Property Test: CAS Store Roundtrip & SHA-256 Digest Determinism
;; -----------------------------------------------------------------------------

(defspec prop-cas-store-roundtrip-and-digest
  50
  (prop/for-all [val gen-edn-value]
                (let [store (cas/create-cas-store)
                      ptr (cas/put! store val)
                      resolved (cas/resolve-ptr store ptr)
                      expected-bytes (if (string? val) (.getBytes ^String val "UTF-8") (.getBytes (pr-str val) "UTF-8"))]
                  (and (m/validate schema/CasPtrSchema ptr)
                       (cas/cas-ptr? ptr)
                       (= val resolved)
                       (= (:cas/cid ptr) (str "sha256:" (cas/sha256-bytes expected-bytes)))))))

;; -----------------------------------------------------------------------------
;; 2. Property Test: Mealy Hash-Chain Tamper-Evidence
;; -----------------------------------------------------------------------------

(defspec prop-mealy-hash-chain-tamper-evidence
  50
  (prop/for-all [inputs (gen/not-empty (gen/vector gen/int 1 10))
                 tamper-field (gen/elements [:event/tx :event/prev-hash :event/hash :event/payload])]
                (let [counter-hiccup
                      [:agent {:id :counter-agent :ins [:in] :outs [:out] :requires {}}
                       [:mealy {:id :counter
                                :ins [:in]
                                :outs [:out]
                                :init-state {:count 0}
                                :step (fn [state _in-port payload]
                                        (let [next-c (+ (:count state) payload)]
                                          [{:count next-c} {:out [next-c]}]))}]
                       [:conn [:agent/in :in] [:counter :in]]
                       [:conn [:counter :out] [:agent/out :out]]]

                      inst (graph/compile-graph counter-hiccup)
                      executed (reduce (fn [acc inp]
                                         (-> acc
                                             (graph/enqueue-input :in inp)
                                             graph/run-graph!))
                                       inst
                                       inputs)
                      log (:event-log executed)]
                  (and (graph/verify-event-log-chain log)
           ;; Tampering any envelope in the log causes verify-event-log-chain to throw
                       (let [idx (rand-int (count log))
                             tampered-log (update-in log [idx tamper-field]
                                                     (fn [v]
                                                       (if (int? v)
                                                         (+ v 999)
                                                         (str v "_tampered"))))]
                         (try
                           (graph/verify-event-log-chain tampered-log)
                           false
                           (catch Exception _
                             true)))))))

;; -----------------------------------------------------------------------------
;; 3. Property Test: [:verify] Node Return Contract Normalization
;; -----------------------------------------------------------------------------

(defspec prop-verify-node-return-contracts
  50
  (prop/for-all [v gen/int
                 threshold gen/int]
                (let [node-def {:id :gate :ins [:in] :outs [:verified :rejected]}
          ;; 1. Boolean return contract
                      bool-res (graph/normalize-verify-output node-def (> v threshold) v)
          ;; 2. Tuple return contract [:verified v] or [:rejected v]
                      expected-port (if (> v threshold) :verified :rejected)
                      tuple-res (graph/normalize-verify-output node-def [expected-port v] v)
          ;; 3. Map return contract {:port :verified :payload v}
                      map-port-res (graph/normalize-verify-output node-def {:port expected-port :payload v} v)
          ;; 4. Direct map return contract {:verified [v]}
                      direct-map-res (graph/normalize-verify-output node-def {expected-port [v]} v)]
                  (and (= bool-res {expected-port [v]})
                       (= tuple-res {expected-port [v]})
                       (= map-port-res {expected-port [v]})
                       (= direct-map-res {expected-port [v]})))))

;; -----------------------------------------------------------------------------
;; 4. Property Test: Deterministic Replay Identity
;; -----------------------------------------------------------------------------

(defspec prop-deterministic-replay-identity
  50
  (prop/for-all [prompts (gen/not-empty (gen/vector gen/string-alphanumeric 1 6))]
                (let [agent-hiccup
                      [:agent {:id :llm-turn-agent :ins [:user-prompt] :outs [:reply] :requires {:cap/llm :any}}
                       [:mealy {:id :turn-loop
                                :ins [:prompt :llm-resp]
                                :outs [:call-llm :final-reply]
                                :init-state {:turns 0}
                                :step (fn [state in-port payload]
                                        (case in-port
                                          :prompt
                                          [(update state :turns inc)
                                           {:call-llm [(str payload " (turn " (inc (:turns state)) ")")]}]
                                          :llm-resp
                                          [state
                                           {:final-reply [(str "Echo: " payload)]}]))}]
                       [:conn [:agent/in :user-prompt] [:turn-loop :prompt]]
                       [:conn [:turn-loop :call-llm] [:cap/llm :generate]]
                       [:conn [:cap/llm :generate] [:turn-loop :llm-resp]]
                       [:conn [:turn-loop :final-reply] [:agent/out :reply]]]

                      mock-llm (fn [_in-port payload]
                                 {:generate [(str "MockLLM[" payload "]")]})
                      inst (graph/compile-graph agent-hiccup {:capabilities {:llm mock-llm}})
                      executed (reduce (fn [acc prompt]
                                         (-> acc
                                             (graph/enqueue-input :user-prompt prompt)
                                             graph/run-graph!))
                                       inst
                                       prompts)
                      replayed (graph/replay-graph agent-hiccup (:event-log executed))]
                  (and (= (:node-states executed) (:node-states replayed))
                       (= (:outbox executed) (:outbox replayed))
                       (= (count (:event-log executed)) (count (:event-log replayed)))))))

;; -----------------------------------------------------------------------------
;; 5. Property Test: Fractal Sub-Agent Equivalence & Concurrency Parity
;; -----------------------------------------------------------------------------

(defspec prop-fractal-subagent-equivalence
  40
  (prop/for-all [nums (gen/not-empty (gen/vector (gen/choose 1 100) 1 6))]
                (let [sub-agent-hiccup
                      [:agent {:id :adder-subagent :ins [:x] :outs [:y] :requires {}}
                       [:mealy {:id :sub-adder
                                :ins [:in]
                                :outs [:out]
                                :init-state {:sum 0}
                                :step (fn [state _in-port payload]
                                        (let [ns (+ (:sum state) payload)]
                                          [{:sum ns} {:out [ns]}]))}]
                       [:conn [:agent/in :x] [:sub-adder :in]]
                       [:conn [:sub-adder :out] [:agent/out :y]]]

                      outer-agent-hiccup
                      [:agent {:id :outer-agent :ins [:in] :outs [:out] :requires {}}
                       sub-agent-hiccup
                       [:conn [:agent/in :in] [:adder-subagent :x]]
                       [:conn [:adder-subagent :y] [:agent/out :out]]]

          ;; 1. Deterministic mode run
                      inst-det (graph/compile-graph outer-agent-hiccup {:mode :deterministic})
                      executed-det (reduce (fn [acc n]
                                             (-> acc
                                                 (graph/enqueue-input :in n)
                                                 graph/run-graph!))
                                           inst-det
                                           nums)

          ;; 2. Concurrent (Virtual-Thread) mode run
                      inst-conc (graph/compile-graph outer-agent-hiccup {:mode :concurrent})
                      executed-conc (reduce (fn [acc n]
                                              (-> acc
                                                  (graph/enqueue-input :in n)
                                                  graph/run-graph!))
                                            inst-conc
                                            nums)]
                  (and (= (:outbox executed-det) (:outbox executed-conc))
                       (= (get-in executed-det [:node-states :adder-subagent :node-states :sub-adder :sum])
                          (get-in executed-conc [:node-states :adder-subagent :node-states :sub-adder :sum]))))))

;; -----------------------------------------------------------------------------
;; Unit Tests: Schema, Validation & CAS Helpers
;; -----------------------------------------------------------------------------

(deftest test-schema-validation-helpers
  (testing "validate-agent-graph! checks valid and rejects invalid Hiccup ASTs"
    (let [valid-agent
          [:agent {:id :test-agent :ins [:in] :outs [:out] :requires {}}
           [:mealy {:id :node1 :ins [:in] :outs [:out] :init-state {} :step (fn [s _ _] [s {}])}]
           [:conn [:agent/in :in] [:node1 :in]]
           [:conn [:node1 :out] [:agent/out :out]]]]
      (is (= valid-agent (schema/validate-agent-graph! valid-agent))))

    (testing "Rejects invalid tags or missing required attributes"
      (is (thrown? Exception (schema/validate-agent-graph! [:invalid-tag {}])))
      (is (thrown? Exception (schema/validate-agent-graph! [:agent {}])))
      (is (thrown? Exception (schema/validate-agent-graph! [:agent {:id :a :ins [:in]}]))))))

(deftest test-cas-externalization
  (testing "maybe-externalize! externalizes strings exceeding inline threshold"
    (let [store (cas/create-cas-store)
          short-str "hello"
          long-str (apply str (repeat 200 "x"))
          ext-short (cas/maybe-externalize! store short-str 50)
          ext-long (cas/maybe-externalize! store long-str 50)]
      (is (= short-str ext-short))
      (is (cas/cas-ptr? ext-long))
      (is (= long-str (cas/resolve-ptr store ext-long))))))

(deftest test-primitive-array-cas-externalization-and-edn-roundtrip
  (testing "Primitive arrays in graph messages are externalized to CAS pointers, producing pure EDN event logs and deterministic hashes across JVM instances"
    (let [f-arr-1 (float-array [1.25 -3.5 9.75 4.0])
          f-arr-2 (float-array [1.25 -3.5 9.75 4.0])
          i-arr-1 (int-array [10 42 7])
          ;; Two distinct JVM float-array instances with identical values produce identical event hashes
          h1 (graph/compute-event-hash graph/GENESIS-HASH [:cap/gpu :logits] [:sampler :logits] f-arr-1)
          h2 (graph/compute-event-hash graph/GENESIS-HASH [:cap/gpu :logits] [:sampler :logits] f-arr-2)]
      (is (not (identical? f-arr-1 f-arr-2)))
      (is (= h1 h2))

      (let [agent-hiccup
            [:agent {:id :prim-array-agent
                     :ins [:trigger]
                     :outs [:max-idx]
                     :requires {:gpu :any}}
             [:mealy {:id :requester
                      :ins [:in]
                      :outs [:fwd]
                      :init-state {}
                      :step (fn [s _ p] [s {:fwd [p]}])}]
             [:mealy {:id :consumer
                      :ins [:logits]
                      :outs [:idx]
                      :init-state {:seen []}
                      :step (fn [s _ payload]
                              (let [idx (cond
                                          (instance? (Class/forName "[F") payload)
                                          (let [^floats arr payload]
                                            (first (apply max-key second (map-indexed vector arr))))
                                          (and (map? payload) (contains? payload :cas/argmax))
                                          (:cas/argmax payload)
                                          :else -1)]
                                [(update s :seen conj idx) {:idx [idx]}]))}]
             [:conn [:agent/in :trigger] [:requester :in]]
             [:conn [:requester :fwd] [:cap/gpu :forward]]
             [:conn [:cap/gpu :logits] [:consumer :logits]]
             [:conn [:consumer :idx] [:agent/out :max-idx]]]
            mock-gpu (fn [_port _payload]
                       {:logits [(float-array [1.25 -3.5 9.75 4.0])]})
            inst (graph/compile-graph agent-hiccup {:capabilities {:gpu mock-gpu}})
            executed (-> inst
                         (graph/enqueue-input :trigger i-arr-1)
                         graph/run-graph!)
            event-log (:event-log executed)
            serialized-edn (pr-str event-log)
            deserialized-log (edn/read-string serialized-edn)]
        (is (= [2] (get-in executed [:outbox :max-idx])))
        ;; 1. Zero #object[...] tags in serialized EDN
        (is (not (str/includes? serialized-edn "#object[")))
        ;; 2. All envelopes in deserialized event-log validate against EventEnvelopeSchema and hash chain
        (is (every? schema/valid-envelope? deserialized-log))
        (is (graph/verify-event-log-chain deserialized-log))
        ;; 3. Replay on deserialized EDN log with an EMPTY CAS store (simulating remote bitecho peer)
        (let [replayed-unhydrated (graph/replay-graph agent-hiccup deserialized-log)]
          (is (= (:outbox executed) (:outbox replayed-unhydrated)))
          (is (= (:node-states executed) (:node-states replayed-unhydrated))))
        ;; 4. Replay with hydrated CAS store
        (let [replayed-hydrated (graph/replay-graph agent-hiccup deserialized-log {:cas-store (:cas-store executed)})]
          (is (= (:outbox executed) (:outbox replayed-hydrated)))
          (is (= (:node-states executed) (:node-states replayed-hydrated))))))))
