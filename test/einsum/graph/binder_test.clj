(ns einsum.graph.binder-test
  "Unit and generative property tests for Capability SPI and Single-Device Binder (einsum.graph.capability and einsum.graph.binder)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as core]
            [einsum.graph.binder :as binder]
            [einsum.graph.capability :as cap]
            [einsum.graph.cas :as cas]
            [clojure.string :as str]))

;; -----------------------------------------------------------------------------
;; Fixtures & Standard Devices
;; -----------------------------------------------------------------------------

(def standard-device-hiccup
  [:device {:id :standard-dev}
   [:cap {:id :sci-eval :type :eval/sci :sandbox :agent}]
   [:cap {:id :cas-store :type :storage/cas}]
   [:cap {:id :term-out :type :io/terminal :quiet? true}]
   [:cap {:id :einsum-llm :type :compute/einsum :backend :uop-cpu}]])

;; -----------------------------------------------------------------------------
;; 1. Property Test: Binder Satisfies Compatible Device
;; -----------------------------------------------------------------------------

(def gen-req-spec
  (gen/elements
   [{:key :eval :spec :eval/sci}
    {:key :storage :spec :storage/cas}
    {:key :terminal :spec :io/terminal}
    {:key :llm :spec :compute/einsum}
    {:key :cap/llm :spec :any}
    {:key :eval :spec {:type :eval/sci}}
    {:key :storage :spec {:type :storage/cas}}
    {:key :opt-cap :spec {:type :unknown/cap :optional? true}}]))

(defspec prop-binder-satisfies-compatible-device
  50
  (prop/for-all [req-items (gen/not-empty (gen/vector gen-req-spec 1 4))]
                (let [reqs (into {} (map (fn [{:keys [key spec]}] [key spec])) req-items)
                      agent-hiccup
                      [:agent {:id :test-agent :ins [:in] :outs [:out] :requires reqs}
                       [:mealy {:id :echo :ins [:in] :outs [:out] :init-state {} :step (fn [s _ p] [s {:out [p]}])}]
                       [:conn [:agent/in :in] [:echo :in]]
                       [:conn [:echo :out] [:agent/out :out]]]
                      bound (binder/bind-agent agent-hiccup standard-device-hiccup)]
                  (and (map? bound)
                       (some? (:device bound))
                       (every? (fn [[k req-spec]]
                                 (if (and (map? req-spec) (:optional? req-spec))
                                   true
                                   (let [norm-k (keyword (name k))]
                                     (contains? (:capabilities bound) norm-k))))
                               reqs)))))

;; -----------------------------------------------------------------------------
;; 2. Property Test: Binder Rejects Missing Capabilities
;; -----------------------------------------------------------------------------

(def gen-missing-key
  (gen/elements [:quantum-unit :non-existent-cap :telepathic-link :alien-compute]))

(defspec prop-binder-rejects-missing-capabilities
  50
  (prop/for-all [missing-k gen-missing-key]
                (let [agent-hiccup
                      [:agent {:id :test-agent :ins [:in] :outs [:out] :requires {missing-k (keyword "missing" (name missing-k))}}
                       [:mealy {:id :echo :ins [:in] :outs [:out] :init-state {} :step (fn [s _ p] [s {:out [p]}])}]
                       [:conn [:agent/in :in] [:echo :in]]
                       [:conn [:echo :out] [:agent/out :out]]]]
                  (try
                    (binder/bind-agent agent-hiccup standard-device-hiccup)
                    false
                    (catch clojure.lang.ExceptionInfo e
                      (let [data (ex-data e)]
                        (and (= "Unsatisfied capability requirement" (.getMessage e))
                             (= (:agent data) :test-agent)
                             (= (:requirement data) missing-k)
                             (vector? (:available-caps data)))))))))

;; -----------------------------------------------------------------------------
;; 3. Property Test: SCI Capability Evaluation Parity
;; -----------------------------------------------------------------------------

(defspec prop-sci-capability-evaluation-parity
  40
  (prop/for-all [a gen/small-integer
                 b gen/small-integer
                 c (gen/choose 1 100)]
                (let [code-str (format "(+ %d (* %d %d))" a b c)
                      sci-agent (cap/start-cap! (cap/build-capability {:id :sci-a :type :eval/sci :sandbox :agent} {}))
                      sci-bench (cap/start-cap! (cap/build-capability {:id :sci-b :type :eval/sci :sandbox :benchmark} {}))
                      res-agent (cap/invoke-cap! sci-agent :eval code-str nil)
                      res-bench (cap/invoke-cap! sci-bench :eval code-str nil)
                      item-a (first (:result res-agent))
                      item-b (first (:result res-bench))]
                  (and (= (:status item-a) :success)
                       (= (:status item-b) :success)
                       (= (:output item-a) (:output item-b))
                       (number? (:tool-ms item-a))
                       (number? (:tool-ms item-b))))))

;; -----------------------------------------------------------------------------
;; 4. Unit Tests: Compute Einsum Capability Both Modes
;; -----------------------------------------------------------------------------

(deftest test-compute-einsum-capability-both-modes
  (testing ":generate mode with mock prompt completion"
    (let [mock-gen (fn [prompt] {:text (str "Echo: " prompt) :tokens 4})
          cap-inst (cap/start-cap! (cap/build-capability {:id :llm
                                                          :type :compute/einsum
                                                          :generate-fn mock-gen}
                                                         {}))
          out (cap/invoke-cap! cap-inst :generate "Hello UOp" nil)
          reply (first (:reply out))]
      (is (= "Echo: Hello UOp" (:text reply)))
      (is (= 4 (:tokens reply)))
      (is (number? (:model-ms reply)))))

  (testing ":forward mode with Tensor Logic graph compilation and execution"
    (let [graph {:name "mul_test"
                 :invars [[:x [:tensor [2] :f32]]
                          [:y [:tensor [2] :f32]]]
                 :outvars [:z]
                 :known-shapes {}
                 :eqns [{:op :ir/multiply :invars [:x :y] :outvars [:z]}]}
          cap-inst (cap/start-cap! (cap/build-capability {:id :fwd
                                                          :type :compute/einsum
                                                          :backend :interpreter
                                                          :ast graph}
                                                         {}))
          x-arr (float-array [2.0 3.0])
          y-arr (float-array [4.0 5.0])
          out (cap/invoke-cap! cap-inst :forward {:inputs [x-arr y-arr]} nil)]
      (is (map? out))
      (is (contains? out :out))
      (let [res (first (:out out))
            slice (core/to-host-slice res 0 2 2 :f32)]
        (is (= [8.0 15.0] (vec slice))))))

  (testing ":forward mode with CAS store resolved pointer inputs"
    (let [graph {:name "add_test"
                 :invars [[:x [:tensor [2] :f32]]
                          [:y [:tensor [2] :f32]]]
                 :outvars [:z]
                 :known-shapes {}
                 :eqns [{:op :ir/add :invars [:x :y] :outvars [:z]}]}
          cap-inst (cap/start-cap! (cap/build-capability {:id :fwd-cas
                                                          :type :compute/einsum
                                                          :backend :interpreter
                                                          :ast graph}
                                                         {}))
          store (cas/create-cas-store)
          ptr-x (cas/put! store (float-array [10.0 20.0]))
          ptr-y (cas/put! store (float-array [30.0 40.0]))
          out (cap/invoke-cap! cap-inst :forward {:inputs [ptr-x ptr-y]} store)]
      (is (map? out))
      (let [res (first (:out out))
            slice (core/to-host-slice res 0 2 2 :f32)]
        (is (= [40.0 60.0] (vec slice)))))))

;; -----------------------------------------------------------------------------
;; 5. Unit Tests: Device Lifecycle, Storage CAS, and Terminal IO
;; -----------------------------------------------------------------------------

(deftest test-device-lifecycle-and-stopping
  (testing "compile-device and stop-device! cleanly lifecycle capabilities"
    (let [dev (binder/compile-device standard-device-hiccup)]
      (is (map? dev))
      (is (= :standard-dev (:id dev)))
      (is (contains? (:caps dev) :sci-eval))
      (is (contains? (:caps dev) :cas-store))
      (is (contains? (:caps dev) :term-out))
      (is (contains? (:caps dev) :einsum-llm))
      ;; Stopping device calls stop-cap! on all capabilities
      (is (nil? (binder/stop-device! dev))))))

(deftest test-storage-cas-capability
  (testing ":storage/cas put, get, and write-edn operations"
    (let [store (cas/create-cas-store)
          cap-inst (cap/start-cap! (cap/build-capability {:id :cas-cap :type :storage/cas} {:cas-store store}))
          ;; Put
          put-res (cap-inst :put {:hello "world"} store)
          ptr (first (:ptr put-res))]
      (is (cas/cas-ptr? ptr))
      ;; Get
      (let [get-res (cap-inst :get ptr store)]
        (is (= {:hello "world"} (first (:value get-res)))))
      ;; Write EDN to temp file
      (let [tmp-f (java.io.File/createTempFile "cas_test_" ".edn")
            path (.getAbsolutePath tmp-f)
            _ (.deleteOnExit tmp-f)
            write-res (cap-inst :write-edn {:path path :data {:answer 42}} store)
            stored-ptr (first (:stored write-res))]
        (is (cas/cas-ptr? stored-ptr))
        (is (= "{:answer 42}" (slurp path)))
        (is (= {:answer 42} (cas/resolve-ptr store stored-ptr)))))))

(deftest test-io-terminal-capability
  (testing ":io/terminal print appends to printed atom and responds with ack"
    (let [cap-inst (cap/start-cap! (cap/build-capability {:id :term :type :io/terminal :quiet? true} {}))
          ack1 (cap-inst :print "Line 1")
          ack2 (cap-inst :stream "Line 2")]
      (is (= {:ack ["Line 1"]} ack1))
      (is (= {:ack ["Line 2"]} ack2))
      (is (= ["Line 1" "Line 2"] @(:printed cap-inst))))))

(deftest test-sci-benchmark-sandbox-isolation
  (testing ":benchmark sandbox strictly forbids file I/O reflection like slurp"
    (let [sci-bench (cap/start-cap! (cap/build-capability {:id :sci-bench :type :eval/sci :sandbox :benchmark} {}))
          res (sci-bench :eval "(slurp \"/tmp/foo\")" nil)
          item (first (:result res))]
      (is (= :error (:status item)))
      (is (str/includes? (:output item) "resolve symbol: slurp")))))

(deftest test-single-root-binding-no-duplicate-start
  (testing "Root capability requirements with overrides are started exactly once across nested sub-agents and stopped cleanly"
    (let [dev-ast [:device {:id :counter-dev}
                   [:cap {:id :sci-eval :type :eval/sci :sandbox :agent}]]
          sub-agent
          [:agent {:id :child-worker
                   :ins [:sub-in]
                   :outs [:sub-out]
                   :requires {:eval {:type :eval/sci :timeout-ms 1234}}}
           [:mealy {:id :sub-echo :ins [:sub-in] :outs [:sub-out] :init-state {} :step (fn [s _ p] [s {:sub-out [p]}])}]
           [:conn [:agent/in :sub-in] [:sub-echo :sub-in]]
           [:conn [:sub-echo :sub-out] [:agent/out :sub-out]]]
          root-agent
          [:agent {:id :root-orchestrator
                   :ins [:root-in]
                   :outs [:root-out]
                   :requires {:eval {:type :eval/sci :timeout-ms 1234}}}
           sub-agent
           [:conn [:agent/in :root-in] [:child-worker :sub-in]]
           [:conn [:child-worker :sub-out] [:agent/out :root-out]]]
          bound (binder/bind-agent root-agent dev-ast)
          device-inst (:device bound)
          specialized-insts @(:specialized-instances device-inst)]
      (is (= 1 (count specialized-insts)))
      (is (nil? (binder/stop-device! device-inst))))))
