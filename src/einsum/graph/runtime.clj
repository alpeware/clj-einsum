(ns einsum.graph.runtime
  "First-principles zero-core.async deterministic and virtual-thread graph engine."
  (:require [einsum.graph.cas :as cas]
            [einsum.graph.schema :as schema]))

;; -----------------------------------------------------------------------------
;; 1. Cryptographic Hash-Chain Middleware
;; -----------------------------------------------------------------------------

(def GENESIS-HASH
  "Canonical 64-character zero-hash representing the root of every event chain."
  "0000000000000000000000000000000000000000000000000000000000000000")

(defn namespace-or-name
  "Renders a keyword with namespace as 'ns/name' (e.g. :agent/in -> 'agent/in')
   or keyword without namespace as 'name' (e.g. :counter -> 'counter')."
  [x]
  (cond
    (keyword? x) (if (namespace x)
                   (str (namespace x) "/" (name x))
                   (name x))
    (string? x) x
    (nil? x) ""
    :else (str x)))

(defn compute-event-hash
  "Computes a deterministic SHA-256 event digest over the previous hash,
   source [src-node src-port], destination [dst-node dst-port], and payload."
  [prev-hash [src-node src-port] [dst-node dst-port] payload]
  (cas/sha256-str
   (str prev-hash ":"
        (namespace-or-name src-node) ":" (name src-port) "->"
        (namespace-or-name dst-node) ":" (name dst-port) ":"
        (pr-str payload))))

(defn verify-event-log-chain
  "Pure function verifying that event-log forms a valid, tamper-evident cryptographic hash chain.
   Throws ex-info if any envelope is tampered or sequence is broken; returns true if valid."
  [event-log]
  (loop [envs event-log
         expected-tx 1
         expected-prev GENESIS-HASH]
    (if (empty? envs)
      true
      (let [env (first envs)
            tx (:event/tx env)
            prev-h (:event/prev-hash env)
            h (:event/hash env)
            [src-n src-p] (:event/source env)
            [dst-n dst-p] (:event/target env)
            payload (:event/payload env)]
        (when-not (= tx expected-tx)
          (throw (ex-info "Invalid transaction index in event chain"
                          {:expected-tx expected-tx :actual-tx tx :envelope env})))
        (when-not (= prev-h expected-prev)
          (throw (ex-info "Previous hash mismatch in event chain"
                          {:expected-prev expected-prev :actual-prev prev-h :tx tx :envelope env})))
        (let [calc-h (compute-event-hash prev-h [src-n src-p] [dst-n dst-p] payload)]
          (when-not (= h calc-h)
            (throw (ex-info "Event hash mismatch (tampered payload or envelope)"
                            {:expected-hash calc-h :actual-hash h :tx tx :envelope env}))))
        (recur (rest envs) (inc expected-tx) h)))))

;; -----------------------------------------------------------------------------
;; 2. Idiomatic [:verify] Return Contract Normalization
;; -----------------------------------------------------------------------------

(defn normalize-verify-output
  "Normalizes the return value of a verify predicate into {out-port [payloads]}."
  [verify-node raw-res fallback-payload]
  (let [outs (set (:outs verify-node))
        first-out (first (:outs verify-node))
        second-out (second (:outs verify-node))]
    (cond
      ;; 1. Boolean return
      (boolean? raw-res)
      {(if raw-res first-out second-out) [fallback-payload]}

      ;; 2. Tuple return [port payload']
      (and (vector? raw-res)
           (== (count raw-res) 2)
           (keyword? (first raw-res))
           (contains? outs (first raw-res)))
      {(first raw-res) [(second raw-res)]}

      ;; 3. Map return with :port and :payload
      (and (map? raw-res) (contains? raw-res :port) (contains? raw-res :payload))
      {(:port raw-res) [(:payload raw-res)]}

      ;; 4. Direct map {out-port payloads}
      (map? raw-res)
      (into {}
            (map (fn [[k v]]
                   [k (if (vector? v) v [v])]))
            raw-res)

      :else
      (throw (ex-info "Invalid return contract from verify predicate"
                      {:node verify-node :raw-res raw-res})))))

;; -----------------------------------------------------------------------------
;; 3. Graph Compilation
;; -----------------------------------------------------------------------------

(declare compile-graph)

(defn compile-graph
  "Compiles an [:agent attrs & children] Hiccup AST into an executable GraphInstance.
   Options can specify:
     :capabilities - map of capability handlers {cap-key fn}
     :cas-store    - custom CAS store (defaults to fresh memory CAS store)
     :mode         - :deterministic (default) or :concurrent (virtual threads)"
  ([agent-hiccup]
   (compile-graph agent-hiccup {}))
  ([agent-hiccup opts]
   (schema/validate-agent-graph! agent-hiccup)
   (let [{:keys [attrs children]} (schema/parse-hiccup-node agent-hiccup)
         conns-atom (atom {})
         nodes-atom (atom {})
         node-states-atom (atom {})]
     (doseq [child children]
       (let [child-tag (first child)]
         (case child-tag
           :mealy
           (let [node-def (second child)
                 node-id (:id node-def)]
             (swap! nodes-atom assoc node-id (assoc node-def :type :mealy))
             (swap! node-states-atom assoc node-id (:init-state node-def)))

           :verify
           (let [node-def (second child)
                 node-id (:id node-def)]
             (swap! nodes-atom assoc node-id (assoc node-def :type :verify))
             (when (contains? node-def :init-state)
               (swap! node-states-atom assoc node-id (:init-state node-def))))

           :agent
           (let [sub-attrs (second child)
                 node-id (:id sub-attrs)
                 sub-compiled (compile-graph child opts)]
             (swap! nodes-atom assoc node-id (assoc sub-attrs :type :agent :hiccup child))
             (swap! node-states-atom assoc node-id sub-compiled))

           :conn
           (let [src (nth child 1)
                 dst (nth child 2)]
             (swap! conns-atom update src (fnil conj []) dst)))))

     {:attrs attrs
      :nodes @nodes-atom
      :node-states @node-states-atom
      :conns-by-source @conns-atom
      :queue clojure.lang.PersistentQueue/EMPTY
      :tx 0
      :prev-hash GENESIS-HASH
      :event-log []
      :outbox (zipmap (:outs attrs) (repeat []))
      :capabilities (or (:capabilities opts) {})
      :cas-store (or (:cas-store opts) (cas/create-cas-store))
      :mode (or (:mode opts) :deterministic)
      :hiccup agent-hiccup})))

;; -----------------------------------------------------------------------------
;; 4. Execution Engine
;; -----------------------------------------------------------------------------

(defn- make-envelope
  [tx prev-hash src-tuple dst-tuple payload]
  (let [h (compute-event-hash prev-hash src-tuple dst-tuple payload)]
    {:event/id (java.util.UUID/randomUUID)
     :event/tx tx
     :event/prev-hash prev-hash
     :event/hash h
     :event/source src-tuple
     :event/target dst-tuple
     :event/payload payload
     :event/meta {}}))

(defn- enqueue-message
  [graph-inst src-tuple dst-tuple payload]
  (let [next-tx (inc (:tx graph-inst))
        env (make-envelope next-tx (:prev-hash graph-inst) src-tuple dst-tuple payload)]
    (-> graph-inst
        (assoc :tx next-tx
               :prev-hash (:event/hash env))
        (update :queue conj env)
        (update :event-log conj env))))

(defn enqueue-input
  "Enqueues an external input into the graph on [:agent/in in-port], routing it
   across all configured connections from that input port."
  [graph-inst in-port payload]
  (let [src-tuple [:agent/in in-port]
        destinations (get-in graph-inst [:conns-by-source src-tuple] [])]
    (reduce (fn [inst dst-tuple]
              (enqueue-message inst src-tuple dst-tuple payload))
            graph-inst
            destinations)))

(defn- invoke-cap-fn
  [cap-fn in-port payload cas-store]
  (try
    (cap-fn in-port payload cas-store)
    (catch clojure.lang.ArityException _
      (cap-fn in-port payload))))

(defn invoke-capability
  "Invokes a capability handler. If mode is :concurrent, executes in a Java virtual thread."
  [cap-handler in-port payload cas-store mode]
  (if (= mode :concurrent)
    (let [result (promise)]
      (Thread/startVirtualThread
       (fn []
         (try
           (deliver result {:ok (invoke-cap-fn cap-handler in-port payload cas-store)})
           (catch Throwable t
             (deliver result {:err t})))))
      (let [res @result]
        (if (:err res)
          (throw (:err res))
          (:ok res))))
    (invoke-cap-fn cap-handler in-port payload cas-store)))

(declare run-graph!)

(defn step-graph!
  "Performs a single deterministic step of the graph, processing the head envelope
   from :queue and routing generated outputs across downstream connections."
  [graph-inst]
  (let [q (:queue graph-inst)]
    (if (empty? q)
      graph-inst
      (let [env (peek q)
            graph-inst-popped (assoc graph-inst :queue (pop q))
            [dst-node dst-port] (:event/target env)
            payload (:event/payload env)]
        (cond
          ;; Case 1: External Agent Output
          (= dst-node :agent/out)
          (update-in graph-inst-popped [:outbox dst-port] conj payload)

          ;; Case 2: Capability Invocation (:cap/<name>)
          (= (namespace dst-node) "cap")
          (let [cap-key (keyword (name dst-node))
                cap-handler (get-in graph-inst [:capabilities cap-key])
                _ (when-not cap-handler
                    (throw (ex-info (str "Missing capability handler for: " (pr-str cap-key))
                                    {:capability cap-key :target (:event/target env)})))
                out-map (invoke-capability cap-handler dst-port payload (:cas-store graph-inst) (:mode graph-inst))]
            (reduce-kv
             (fn [inst out-p msgs]
               (let [src-t [dst-node out-p]
                     targets (get-in inst [:conns-by-source src-t] [])
                     msg-seq (if (vector? msgs) msgs [msgs])]
                 (reduce
                  (fn [i msg]
                    (reduce (fn [i2 dst-t] (enqueue-message i2 src-t dst-t msg))
                            i
                            targets))
                  inst
                  msg-seq)))
             graph-inst-popped
             out-map))

          ;; Case 3: Mealy Machine Node
          (let [node (get-in graph-inst [:nodes dst-node])]
            (= (:type node) :mealy))
          (let [node (get-in graph-inst [:nodes dst-node])
                curr-st (get-in graph-inst [:node-states dst-node])
                [next-st out-map] ((:step node) curr-st dst-port payload)
                inst-updated (assoc-in graph-inst-popped [:node-states dst-node] next-st)]
            (reduce-kv
             (fn [inst out-p msgs]
               (let [src-t [dst-node out-p]
                     targets (get-in inst [:conns-by-source src-t] [])
                     msg-seq (if (vector? msgs) msgs [msgs])]
                 (reduce
                  (fn [i msg]
                    (reduce (fn [i2 dst-t] (enqueue-message i2 src-t dst-t msg))
                            i
                            targets))
                  inst
                  msg-seq)))
             inst-updated
             out-map))

          ;; Case 4: Verify Node
          (let [node (get-in graph-inst [:nodes dst-node])]
            (= (:type node) :verify))
          (let [node (get-in graph-inst [:nodes dst-node])
                has-st? (contains? node :init-state)
                curr-st (when has-st? (get-in graph-inst [:node-states dst-node]))
                raw-res (if has-st?
                          ((:pred node) curr-st dst-port payload)
                          ((:pred node) payload))
                [next-st verify-out] (if (and has-st? (vector? raw-res) (== (count raw-res) 2) (map? (first raw-res)))
                                       raw-res
                                       [curr-st raw-res])
                out-map (normalize-verify-output node verify-out payload)
                inst-updated (if has-st?
                               (assoc-in graph-inst-popped [:node-states dst-node] next-st)
                               graph-inst-popped)]
            (reduce-kv
             (fn [inst out-p msgs]
               (let [src-t [dst-node out-p]
                     targets (get-in inst [:conns-by-source src-t] [])
                     msg-seq (if (vector? msgs) msgs [msgs])]
                 (reduce
                  (fn [i msg]
                    (reduce (fn [i2 dst-t] (enqueue-message i2 src-t dst-t msg))
                            i
                            targets))
                  inst
                  msg-seq)))
             inst-updated
             out-map))

          ;; Case 5: Nested Agent Node
          (let [node (get-in graph-inst [:nodes dst-node])]
            (= (:type node) :agent))
          (let [sub-inst (get-in graph-inst [:node-states dst-node])
                sub-inst-cleared (assoc sub-inst :outbox (zipmap (get-in sub-inst [:attrs :outs]) (repeat [])))
                sub-inst-stepped (-> sub-inst-cleared
                                     (enqueue-input dst-port payload)
                                     (run-graph! {:mode (:mode graph-inst)}))
                inst-updated (assoc-in graph-inst-popped [:node-states dst-node] sub-inst-stepped)]
            (reduce-kv
             (fn [inst out-p msgs]
               (let [src-t [dst-node out-p]
                     targets (get-in inst [:conns-by-source src-t] [])
                     msg-seq (if (vector? msgs) msgs [msgs])]
                 (reduce
                  (fn [i msg]
                    (reduce (fn [i2 dst-t] (enqueue-message i2 src-t dst-t msg))
                            i
                            targets))
                  inst
                  msg-seq)))
             inst-updated
             (:outbox sub-inst-stepped)))

          :else
          (throw (ex-info (str "Unknown destination node in event target: " (pr-str dst-node))
                          {:target (:event/target env) :envelope env})))))))

(defn run-graph!
  "Repeatedly steps graph-inst until the message queue is empty.
   Options can specify:
     :max-steps - maximum queue transitions before aborting (default 10000)."
  ([graph-inst]
   (run-graph! graph-inst {}))
  ([graph-inst opts]
   (let [max-steps (long (or (:max-steps opts) 10000))]
     (loop [inst graph-inst
            step 0]
       (if (empty? (:queue inst))
         inst
         (if (>= step max-steps)
           (throw (ex-info "Exceeded max-steps in run-graph!"
                           {:steps step :max-steps max-steps}))
           (recur (step-graph! inst) (inc step))))))))

;; -----------------------------------------------------------------------------
;; 5. Deterministic Replay
;; -----------------------------------------------------------------------------

(defn replay-graph
  "Replays a recorded event-log against agent-hiccup with zero capabilities.
   Validates the cryptographic hash chain, walks all node transitions deterministically,
   and reproduces the exact final :node-states and :outbox."
  [agent-hiccup event-log]
  (verify-event-log-chain event-log)
  (let [initial-inst (compile-graph agent-hiccup {:capabilities {}})]
    (loop [envs event-log
           inst initial-inst]
      (if (empty? envs)
        (assoc inst
               :event-log event-log
               :tx (count event-log)
               :prev-hash (if (seq event-log) (:event/hash (last event-log)) GENESIS-HASH))
        (let [env (first envs)
              [dst-node dst-port] (:event/target env)
              payload (:event/payload env)]
          (cond
            (= dst-node :agent/out)
            (recur (rest envs)
                   (update-in inst [:outbox dst-port] conj payload))

            (= (namespace dst-node) "cap")
            (recur (rest envs) inst)

            (let [node (get-in inst [:nodes dst-node])]
              (= (:type node) :mealy))
            (let [node (get-in inst [:nodes dst-node])
                  curr-st (get-in inst [:node-states dst-node])
                  [next-st _] ((:step node) curr-st dst-port payload)]
              (recur (rest envs)
                     (assoc-in inst [:node-states dst-node] next-st)))

            (let [node (get-in inst [:nodes dst-node])]
              (= (:type node) :verify))
            (let [node (get-in inst [:nodes dst-node])
                  has-st? (contains? node :init-state)
                  curr-st (when has-st? (get-in inst [:node-states dst-node]))
                  raw-res (if has-st?
                            ((:pred node) curr-st dst-port payload)
                            ((:pred node) payload))
                  [next-st _] (if (and has-st? (vector? raw-res) (== (count raw-res) 2) (map? (first raw-res)))
                                raw-res
                                [curr-st raw-res])]
              (recur (rest envs)
                     (if has-st?
                       (assoc-in inst [:node-states dst-node] next-st)
                       inst)))

            (let [node (get-in inst [:nodes dst-node])]
              (= (:type node) :agent))
            (let [sub-inst (get-in inst [:node-states dst-node])
                  sub-inst-cleared (assoc sub-inst :outbox (zipmap (get-in sub-inst [:attrs :outs]) (repeat [])))
                  sub-inst-stepped (-> sub-inst-cleared
                                       (enqueue-input dst-port payload)
                                       (run-graph! {:capabilities {}}))]
              (recur (rest envs)
                     (assoc-in inst [:node-states dst-node] sub-inst-stepped)))

            :else
            (recur (rest envs) inst)))))))
