(ns einsum.graph.capability
  "Extensible Capability Service Provider Interface (SPI) and built-in single-device capabilities:
   :eval/sci, :storage/cas, :io/terminal, and :compute/einsum."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.agent.core :as agent-core]
            [einsum.core :as core]
            [einsum.graph.cas :as cas]
            [einsum.logic.agent.state-tracker :as state-tracker]
            [einsum.models.gemma4.runtime :as gemma4-rt]))

;; =============================================================================
;; 1. Capability Protocol & Reusable Instance Wrapper
;; =============================================================================

(defprotocol ICapability
  "Extensible Capability Service Provider Interface (SPI) for single-device and distributed nodes."
  (start-cap! [this] "Initializes resources (e.g. SCI sandbox, VRAM session, thread pool) and returns started instance.")
  (invoke-cap! [this in-port payload cas-store] "Executes capability on incoming port and payload with CAS store access, returning an output map.")
  (stop-cap! [this] "Tears down resources and returns nil."))

(defrecord CapabilityInstance [id type attrs state printed start-fn invoke-fn stop-fn]
  ICapability
  (start-cap! [this]
    (if start-fn
      (or (start-fn this) this)
      this))
  (invoke-cap! [this in-port payload cas-store]
    (invoke-fn this in-port payload cas-store))
  (stop-cap! [this]
    (when stop-fn
      (stop-fn this))
    nil)

  clojure.lang.IFn
  (invoke [this in-port payload]
    (invoke-cap! this in-port payload nil))
  (invoke [this in-port payload cas-store]
    (invoke-cap! this in-port payload cas-store))
  (applyTo [this args]
    (clojure.lang.AFn/applyToHelper this args)))

(defn make-capability-instance
  "Constructs a standard CapabilityInstance with default state and printed atoms."
  [{:keys [id type attrs state printed start-fn invoke-fn stop-fn]}]
  (map->CapabilityInstance
   {:id id
    :type type
    :attrs (or attrs {})
    :state (or state (atom {}))
    :printed (or printed (atom []))
    :start-fn start-fn
    :invoke-fn invoke-fn
    :stop-fn stop-fn}))

;; =============================================================================
;; 2. Extensible Capability Builder Multi-Method
;; =============================================================================

(defmulti build-capability
  "Builds an unstarted capability instance from its Hiccup attributes map and device context.
   Extensible via standard `defmethod build-capability <type> [cap-attrs device-ctx]`."
  (fn [cap-attrs _device-ctx] (:type cap-attrs)))

;; -----------------------------------------------------------------------------
;; 2.1 :eval/sci (SCI Clojure Sandbox Execution)
;; -----------------------------------------------------------------------------

(defmethod build-capability :eval/sci
  [cap-attrs device-ctx]
  (let [id (:id cap-attrs)
        state (atom {:device-ctx device-ctx})
        eval-fn (or (:eval-fn cap-attrs) agent-core/eval-tool-code)
        timeout-ms (:timeout-ms cap-attrs 5000)
        start-fn (fn [this]
                   (let [ctx (or (:sci-ctx cap-attrs)
                                 (case (:sandbox cap-attrs :agent)
                                   :benchmark (agent-core/create-benchmark-sci-ctx)
                                   :agent (agent-core/create-agent-sci-ctx)
                                   (agent-core/create-agent-sci-ctx)))]
                     (swap! (:state this) assoc :sci-ctx ctx)
                     this))
        invoke-fn (fn [this _in-port payload cas-store]
                    (let [sci-ctx (or (get @(:state this) :sci-ctx)
                                      (:sci-ctx cap-attrs)
                                      (agent-core/create-agent-sci-ctx))
                          effective-store (or cas-store (get @(:state this) :cas-store) (:cas-store device-ctx))
                          resolved-payload (if (and effective-store (cas/cas-ptr? payload))
                                             (cas/resolve-ptr effective-store payload)
                                             payload)
                          [code-str turn] (cond
                                            (string? resolved-payload)
                                            [resolved-payload nil]

                                            (map? resolved-payload)
                                            (let [code (:code resolved-payload)
                                                  code-resolved (if (and effective-store (cas/cas-ptr? code))
                                                                  (cas/resolve-ptr effective-store code)
                                                                  code)]
                                              [(str (or code-resolved "")) (:turn resolved-payload)])

                                            :else
                                            [(str (or resolved-payload "")) nil])
                          t0 (System/nanoTime)
                          res (if (and (= eval-fn agent-core/eval-tool-code) (some? turn))
                                (eval-fn sci-ctx code-str timeout-ms)
                                (try
                                  (eval-fn sci-ctx code-str (or turn timeout-ms))
                                  (catch clojure.lang.ArityException _
                                    (try
                                      (eval-fn code-str (or turn timeout-ms))
                                      (catch clojure.lang.ArityException _
                                        (eval-fn code-str))))))
                          tool-ms (/ (- (System/nanoTime) t0) 1e6)
                          ret (assoc res :tool-ms tool-ms :code code-str :turn turn)]
                      {:result [ret]}))
        stop-fn (fn [this]
                  (swap! (:state this) dissoc :sci-ctx)
                  nil)]
    (make-capability-instance
     {:id id
      :type :eval/sci
      :attrs cap-attrs
      :state state
      :start-fn start-fn
      :invoke-fn invoke-fn
      :stop-fn stop-fn})))

;; -----------------------------------------------------------------------------
;; 2.2 :storage/cas (Content-Addressed Storage)
;; -----------------------------------------------------------------------------

(defmethod build-capability :storage/cas
  [cap-attrs device-ctx]
  (let [id (:id cap-attrs)
        state (atom {:device-ctx device-ctx})
        invoke-fn (fn [this in-port payload cas-store]
                    (let [effective-store (or cas-store
                                              (get @(:state this) :cas-store)
                                              (:cas-store device-ctx)
                                              (:cas-store cap-attrs))]
                      (when-not effective-store
                        (throw (ex-info "No CAS store available for :storage/cas capability"
                                        {:id id :attrs cap-attrs})))
                      (case in-port
                        :put
                        (let [ptr (cas/put! effective-store payload)]
                          {:ptr [ptr]})

                        :get
                        (let [val (cas/resolve-ptr effective-store payload)]
                          {:value [val]})

                        :write-edn
                        (let [[path data] (cond
                                            (and (map? payload) (or (contains? payload :path) (contains? payload :data)))
                                            [(:path payload) (:data payload)]

                                            (string? (:out cap-attrs))
                                            [(:out cap-attrs) payload]

                                            :else
                                            [nil payload])
                              _ (when (and path (not (str/blank? (str path))))
                                  (when-let [p (io/file (str path))]
                                    (when-let [parent (.getParentFile p)]
                                      (.mkdirs parent)))
                                  (spit (str path) (pr-str data)))
                              ptr (cas/put! effective-store data)]
                          {:stored [ptr]})

                        ;; Default fallback
                        (if (cas/cas-ptr? payload)
                          {:value [(cas/resolve-ptr effective-store payload)]}
                          {:ptr [(cas/put! effective-store payload)]}))))]
    (make-capability-instance
     {:id id
      :type :storage/cas
      :attrs cap-attrs
      :state state
      :invoke-fn invoke-fn})))

;; -----------------------------------------------------------------------------
;; 2.3 :io/terminal (Terminal / Stdout Output)
;; -----------------------------------------------------------------------------

(defmethod build-capability :io/terminal
  [cap-attrs device-ctx]
  (let [id (:id cap-attrs)
        printed (or (:printed cap-attrs) (atom []))
        state (atom {:device-ctx device-ctx})
        invoke-fn (fn [this _in-port payload _cas-store]
                    (let [quiet? (or (:quiet? cap-attrs) (:quiet cap-attrs))]
                      (when-not quiet?
                        (println payload))
                      (swap! (:printed this) conj payload)
                      {:ack [payload]}))]
    (make-capability-instance
     {:id id
      :type :io/terminal
      :attrs cap-attrs
      :state state
      :printed printed
      :invoke-fn invoke-fn})))

;; -----------------------------------------------------------------------------
;; 2.4 :compute/einsum (Declarative Tensor Logic & Model Execution)
;; -----------------------------------------------------------------------------

(defn- call-generate-fn
  [gen-fn session prompt opts]
  (cond
    (and session opts)
    (try (gen-fn session prompt opts)
         (catch clojure.lang.ArityException _
           (try (gen-fn session prompt)
                (catch clojure.lang.ArityException _
                  (gen-fn prompt)))))
    session
    (try (gen-fn session prompt)
         (catch clojure.lang.ArityException _
           (gen-fn prompt)))
    :else
    (try (gen-fn prompt)
         (catch clojure.lang.ArityException _
           (gen-fn session prompt)))))

(defmethod build-capability :compute/einsum
  [cap-attrs device-ctx]
  (let [id (:id cap-attrs)
        backend (or (:backend cap-attrs) :uop-cpu)
        state (atom {:device-ctx device-ctx
                     :backend backend
                     :kernel-cache (atom {})})
        start-fn (fn [this]
                   (let [b (or (:backend (:attrs this)) :uop-cpu)]
                     (swap! (:state this) assoc :backend b :kernel-cache (atom {}))
                     this))
        invoke-fn (fn [this in-port payload cas-store]
                    (let [effective-store (or cas-store (get @(:state this) :cas-store) (:cas-store device-ctx))]
                      (case in-port
                        :generate
                        (let [prompt (if (map? payload) (:prompt payload) payload)
                              session (or (when (map? payload) (:session payload))
                                          (:session cap-attrs)
                                          (get @(:state this) :session))
                              opts (when (map? payload) (:opts payload))
                              effective-opts (merge (when session (:opts session))
                                                    (select-keys cap-attrs [:decode :max-new-tokens :temperature :top-k :repetition-penalty])
                                                    (select-keys @(:state this) [:decode :max-new-tokens :temperature :top-k :repetition-penalty])
                                                    opts)
                              gen-fn (or (:generate-fn cap-attrs)
                                         (get @(:state this) :generate-fn))
                              t0 (System/nanoTime)
                              gen-res (if gen-fn
                                        (call-generate-fn gen-fn session prompt effective-opts)
                                        (gemma4-rt/generate-new-tokens-and-text session prompt effective-opts))
                              model-ms (/ (- (System/nanoTime) t0) 1e6)
                              reply-val (if (map? gen-res)
                                          (assoc gen-res :model-ms (double (or (:total-ms gen-res) model-ms)))
                                          {:text gen-res :model-ms model-ms})]
                          {:reply [reply-val]})

                        :forward
                        (let [fwd-fn (or (when (map? payload) (:forward-fn payload))
                                         (:forward-fn cap-attrs)
                                         (get @(:state this) :forward-fn))
                              session (or (when (map? payload) (:session payload))
                                          (:session cap-attrs)
                                          (get @(:state this) :session))]
                          (if fwd-fn
                            (let [out (fwd-fn session payload)]
                              {:logits [out] :out [out]})
                            (let [ast (or (when (map? payload) (:ast payload))
                                          (:ast cap-attrs)
                                          (get @(:state this) :ast))
                                  _ (when-not ast
                                      (throw (ex-info "Missing AST for :forward compute/einsum invocation"
                                                      {:payload payload :cap-attrs cap-attrs})))
                                  raw-inputs (cond
                                               (and (map? payload) (contains? payload :inputs)) (:inputs payload)
                                               (and (map? payload) (contains? payload :args)) (:args payload)
                                               (vector? payload) payload
                                               (sequential? payload) (vec payload)
                                               :else [payload])
                                  resolved-inputs (mapv (fn [x]
                                                          (if (and effective-store (cas/cas-ptr? x))
                                                            (cas/resolve-ptr effective-store x)
                                                            x))
                                                        raw-inputs)
                                  b (or (:backend cap-attrs) (get @(:state this) :backend) :uop-cpu)
                                  cache (or (get @(:state this) :kernel-cache) (atom {}))
                                  cache-key [ast b]
                                  kernel (or (get @cache cache-key)
                                             (let [ctx (core/init-backend! b)
                                                   k (core/compile-kernel ctx "forward_kernel" ast)]
                                               (swap! cache assoc cache-key k)
                                               k))
                                  out (apply core/execute kernel resolved-inputs)]
                              {:logits [out] :out [out]})))

                        ;; Default: if :ast is present, default to forward; else generate
                        (if (or (:ast cap-attrs) (and (map? payload) (:ast payload)))
                          (let [ast (or (when (map? payload) (:ast payload)) (:ast cap-attrs))
                                raw-inputs (cond
                                             (and (map? payload) (contains? payload :inputs)) (:inputs payload)
                                             (and (map? payload) (contains? payload :args)) (:args payload)
                                             (vector? payload) payload
                                             (sequential? payload) (vec payload)
                                             :else [payload])
                                resolved-inputs (mapv (fn [x]
                                                        (if (and effective-store (cas/cas-ptr? x))
                                                          (cas/resolve-ptr effective-store x)
                                                          x))
                                                      raw-inputs)
                                b (or (:backend cap-attrs) :uop-cpu)
                                ctx (core/init-backend! b)
                                kernel (core/compile-kernel ctx "forward_kernel" ast)
                                out (apply core/execute kernel resolved-inputs)]
                            {:logits [out] :out [out]})
                          (let [prompt (if (map? payload) (:prompt payload) (str payload))
                                t0 (System/nanoTime)
                                gen-fn (or (:generate-fn cap-attrs) (constantly {:text prompt}))
                                gen-res (try (gen-fn prompt) (catch clojure.lang.ArityException _ {:text prompt}))
                                model-ms (/ (- (System/nanoTime) t0) 1e6)]
                            {:reply [(if (map? gen-res) (assoc gen-res :model-ms model-ms) {:text gen-res :model-ms model-ms})]})))))]
    (make-capability-instance
     {:id id
      :type :compute/einsum
      :attrs cap-attrs
      :state state
      :start-fn start-fn
      :invoke-fn invoke-fn})))

;; -----------------------------------------------------------------------------
;; 2.5 :compute/state-tracker (In-VRAM Datalog Fixpoint State Tracker)
;; -----------------------------------------------------------------------------

(defmethod build-capability :compute/state-tracker
  [cap-attrs device-ctx]
  (let [id (:id cap-attrs)
        tracker-atom (atom (:tracker-session cap-attrs))
        state (atom {:device-ctx device-ctx
                     :tracker-atom tracker-atom})
        start-fn (fn [this]
                   (when-not @tracker-atom
                     (let [b (or (:backend cap-attrs) :interpreter)
                           ctx (core/create-context {:backend b})
                           n (or (:n cap-attrs) 16)
                           session (state-tracker/init-state-tracker ctx n cap-attrs)]
                       (reset! tracker-atom session)))
                   (swap! (:state this) assoc :tracker-session @tracker-atom)
                   this)
        invoke-fn (fn [this _in-port payload cas-store]
                    (let [effective-store (or cas-store (get @(:state this) :cas-store) (:cas-store device-ctx))
                          resolved-payload (if (and effective-store (cas/cas-ptr? payload))
                                             (cas/resolve-ptr effective-store payload)
                                             payload)
                          session (or @tracker-atom
                                      (let [b (or (:backend cap-attrs) :interpreter)
                                            ctx (core/create-context {:backend b})
                                            n (or (:n cap-attrs) 16)
                                            s (state-tracker/init-state-tracker ctx n cap-attrs)]
                                        (reset! tracker-atom s)
                                        (swap! (:state this) assoc :tracker-session s)
                                        s))
                          modified-set (cond
                                         (and (map? resolved-payload) (contains? resolved-payload :modified))
                                         (set (:modified resolved-payload))

                                         (set? resolved-payload)
                                         resolved-payload

                                         (coll? resolved-payload)
                                         (set resolved-payload)

                                         (number? resolved-payload)
                                         #{(long resolved-payload)}

                                         :else
                                         #{})
                          res (state-tracker/step-state-tracker! session modified-set)]
                      {:invalidated [res] :out [res]}))
        stop-fn (fn [this]
                  (reset! tracker-atom nil)
                  (swap! (:state this) dissoc :tracker-session)
                  nil)]
    (make-capability-instance
     {:id id
      :type :compute/state-tracker
      :attrs cap-attrs
      :state state
      :start-fn start-fn
      :invoke-fn invoke-fn
      :stop-fn stop-fn})))

(defmethod build-capability :default
  [cap-attrs device-ctx]
  (throw (ex-info (str "Unknown capability type: " (pr-str (:type cap-attrs)))
                  {:cap-attrs cap-attrs :device-ctx device-ctx})))
