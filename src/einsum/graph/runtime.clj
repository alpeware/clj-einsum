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

(defn- canonicalize-for-hash
  "Recursively replaces any raw Java primitive array in `payload` with its canonical
   `{:cas/cid \"sha256:...\"}` map so `compute-event-hash` never depends on JVM object identity."
  [payload]
  (cond
    (cas/primitive-array? payload)
    {:cas/cid (cas/primitive-array-cid payload)}

    (vector? payload)
    (mapv canonicalize-for-hash payload)

    (and (map? payload) (not (cas/cas-ptr? payload)))
    (persistent!
     (reduce-kv (fn [acc k v]
                  (assoc! acc k (canonicalize-for-hash v)))
                (transient {})
                payload))

    :else
    payload))

(defn compute-event-hash
  "Computes a deterministic SHA-256 event digest over the previous hash,
   source [src-node src-port], destination [dst-node dst-port], and payload."
  [prev-hash [src-node src-port] [dst-node dst-port] payload]
  (cas/sha256-str
   (str prev-hash ":"
        (namespace-or-name src-node) ":" (name src-port) "->"
        (namespace-or-name dst-node) ":" (name dst-port) ":"
        (pr-str (canonicalize-for-hash payload)))))

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
      (let [target-port (if raw-res first-out second-out)]
        (if target-port
          {target-port [fallback-payload]}
          {}))

      ;; 2. Tuple return [port payload']
      (and (vector? raw-res)
           (== (count raw-res) 2)
           (keyword? (first raw-res)))
      (if (contains? outs (first raw-res))
        {(first raw-res) [(second raw-res)]}
        (throw (ex-info "Verify predicate returned undeclared output port"
                        {:node verify-node :port (first raw-res) :valid-outs outs})))

      ;; 3. Map return with :port and :payload
      (and (map? raw-res) (contains? raw-res :port) (contains? raw-res :payload))
      (if (contains? outs (:port raw-res))
        {(:port raw-res) [(:payload raw-res)]}
        (throw (ex-info "Verify predicate returned undeclared output port"
                        {:node verify-node :port (:port raw-res) :valid-outs outs})))

      ;; 4. Direct map {out-port payloads}
      (map? raw-res)
      (do
        (doseq [k (keys raw-res)]
          (when-not (contains? outs k)
            (throw (ex-info "Verify predicate returned undeclared output port"
                            {:node verify-node :port k :valid-outs outs}))))
        (into {}
              (map (fn [[k v]]
                     [k (if (vector? v) v [v])]))
              raw-res))

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
                 sub-opts (if-let [sub-caps (get-in opts [:subagent-capabilities node-id])]
                            (assoc opts :capabilities sub-caps)
                            opts)
                 sub-compiled (compile-graph child sub-opts)]
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
        clean-payload (cas/externalize-primitives! (:cas-store graph-inst) payload)
        env (make-envelope next-tx (:prev-hash graph-inst) src-tuple dst-tuple clean-payload)]
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
                out-map (invoke-capability cap-handler dst-port payload (:cas-store graph-inst) (:mode graph-inst))
                first-reply (first (:reply out-map))
                has-sub-log? (and (map? first-reply) (:sub-event-log first-reply))
                sub-log (when has-sub-log? (:sub-event-log first-reply))
                raw-sub-hiccup (when has-sub-log? (:sub-hiccup first-reply))
                sub-decoder-opts (when has-sub-log? (:sub-decoder-opts first-reply))
                sub-hiccup-meta (when raw-sub-hiccup
                                  (let [ptr (cas/put! (:cas-store graph-inst) raw-sub-hiccup)]
                                    (cond-> ptr
                                      (map? sub-decoder-opts)
                                      (assoc :cas/decoder-opts sub-decoder-opts))))
                tx-idx (dec (:event/tx env))
                parent-log (get graph-inst-popped :event-log)
                parent-log-updated (if (and has-sub-log? (vector? parent-log) (<= 0 tx-idx) (< tx-idx (count parent-log)))
                                     (update-in parent-log [tx-idx :event/meta]
                                                merge
                                                (cond-> {:sub-event-log sub-log}
                                                  sub-hiccup-meta (assoc :sub-hiccup sub-hiccup-meta)))
                                     parent-log)
                inst-with-log (if has-sub-log?
                                (assoc graph-inst-popped :event-log parent-log-updated)
                                graph-inst-popped)
                clean-out-map (if has-sub-log?
                                (update out-map :reply (fn [replies]
                                                         (mapv (fn [r]
                                                                 (if (map? r)
                                                                   (dissoc r :sub-event-log :sub-hiccup :sub-decoder-opts)
                                                                   r))
                                                               replies)))
                                out-map)]
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
             inst-with-log
             clean-out-map))

          ;; Case 3: Mealy Machine Node
          (let [node (get-in graph-inst [:nodes dst-node])]
            (= (:type node) :mealy))
          (let [node (get-in graph-inst [:nodes dst-node])
                curr-st (get-in graph-inst [:node-states dst-node])
                resolved-payload (cas/resolve-payload (:cas-store graph-inst) payload)
                [next-st out-map] ((:step node) curr-st dst-port resolved-payload)
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
                resolved-payload (cas/resolve-payload (:cas-store graph-inst) payload)
                raw-res (if has-st?
                          ((:pred node) curr-st dst-port resolved-payload)
                          ((:pred node) resolved-payload))
                [next-st verify-out] (if (and has-st? (vector? raw-res) (== (count raw-res) 2) (map? (first raw-res)))
                                       raw-res
                                       [curr-st raw-res])
                out-map (normalize-verify-output node verify-out resolved-payload)
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
                sub-inst-cleared (assoc sub-inst
                                        :outbox (zipmap (get-in sub-inst [:attrs :outs]) (repeat []))
                                        :queue clojure.lang.PersistentQueue/EMPTY
                                        :tx 0
                                        :prev-hash GENESIS-HASH
                                        :event-log [])
                sub-inst-stepped (-> sub-inst-cleared
                                     (enqueue-input dst-port payload)
                                     (run-graph! {:mode (:mode graph-inst)}))
                tx-idx (dec (:event/tx env))
                parent-log (get graph-inst-popped :event-log)
                parent-log-updated (if (and (vector? parent-log) (<= 0 tx-idx) (< tx-idx (count parent-log)))
                                     (assoc-in parent-log [tx-idx :event/meta :sub-event-log] (:event-log sub-inst-stepped))
                                     parent-log)
                inst-updated (-> graph-inst-popped
                                 (assoc :event-log parent-log-updated)
                                 (assoc-in [:node-states dst-node] sub-inst-stepped))]
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
   and reproduces the exact final :node-states and :outbox.
   Optional `opts` map may specify:
     :cas-store           - CAS store for hydrating externalized primitive array pointers
     :initial-node-states - map of prior node states (used when replaying multi-turn nested agents)"
  ([agent-hiccup event-log]
   (replay-graph agent-hiccup event-log {}))
  ([agent-hiccup event-log opts]
   (verify-event-log-chain event-log)
   (let [base-inst (compile-graph agent-hiccup {:capabilities {}
                                                :cas-store (:cas-store opts)})
         initial-inst (if-let [init-states (:initial-node-states opts)]
                        (update base-inst :node-states merge init-states)
                        base-inst)]
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
             (let [sub-event-log (get-in env [:event/meta :sub-event-log])
                   raw-sub-hiccup (get-in env [:event/meta :sub-hiccup])
                   resolved-sub-hiccup (cond
                                         (and (vector? raw-sub-hiccup) (= :agent (first raw-sub-hiccup)))
                                         raw-sub-hiccup

                                         (cas/cas-ptr? raw-sub-hiccup)
                                         (or (cas/resolve-ptr (:cas-store inst) raw-sub-hiccup)
                                             (when-let [d-opts (:cas/decoder-opts raw-sub-hiccup)]
                                               ((requiring-resolve 'einsum.agent.graph/autoregressive-decoder-graph) d-opts)))

                                         :else nil)
                   inst-with-sub (if (seq sub-event-log)
                                   (do
                                     (when (nil? resolved-sub-hiccup)
                                       (throw (ex-info "Cannot replay :sub-event-log without :sub-hiccup in :event/meta"
                                                       {:tx (:event/tx env)
                                                        :target (:event/target env)
                                                        :meta (:event/meta env)})))
                                     (let [sub-replayed (replay-graph resolved-sub-hiccup
                                                                      sub-event-log
                                                                      {:cas-store (:cas-store inst)})]
                                       (update inst :sub-replays (fnil conj []) sub-replayed)))
                                   inst)]
               (recur (rest envs) inst-with-sub))

             (let [node (get-in inst [:nodes dst-node])]
               (= (:type node) :mealy))
             (let [node (get-in inst [:nodes dst-node])
                   curr-st (get-in inst [:node-states dst-node])
                   resolved-payload (cas/resolve-payload (:cas-store inst) payload)
                   [next-st _] ((:step node) curr-st dst-port resolved-payload)]
               (recur (rest envs)
                      (assoc-in inst [:node-states dst-node] next-st)))

             (let [node (get-in inst [:nodes dst-node])]
               (= (:type node) :verify))
             (let [node (get-in inst [:nodes dst-node])
                   has-st? (contains? node :init-state)
                   curr-st (when has-st? (get-in inst [:node-states dst-node]))
                   resolved-payload (cas/resolve-payload (:cas-store inst) payload)
                   raw-res (if has-st?
                             ((:pred node) curr-st dst-port resolved-payload)
                             ((:pred node) resolved-payload))
                   [next-st _] (if (and has-st? (vector? raw-res) (== (count raw-res) 2) (map? (first raw-res)))
                                 raw-res
                                 [curr-st raw-res])]
               (recur (rest envs)
                      (if has-st?
                        (assoc-in inst [:node-states dst-node] next-st)
                        inst)))

             (let [node (get-in inst [:nodes dst-node])]
               (= (:type node) :agent))
             (let [node (get-in inst [:nodes dst-node])
                   sub-event-log (get-in env [:event/meta :sub-event-log])
                   prior-sub-inst (get-in inst [:node-states dst-node])
                   sub-inst-stepped (if (seq sub-event-log)
                                      (replay-graph (:hiccup node)
                                                    sub-event-log
                                                    {:cas-store (:cas-store inst)
                                                     :initial-node-states (:node-states prior-sub-inst)})
                                      (let [sub-inst-cleared (assoc prior-sub-inst
                                                                    :outbox (zipmap (get-in prior-sub-inst [:attrs :outs]) (repeat []))
                                                                    :queue clojure.lang.PersistentQueue/EMPTY
                                                                    :tx 0
                                                                    :prev-hash GENESIS-HASH
                                                                    :event-log [])]
                                        (-> sub-inst-cleared
                                            (enqueue-input dst-port payload)
                                            (run-graph! {:capabilities {}}))))
                   inst-with-sub (cond-> (assoc-in inst [:node-states dst-node] sub-inst-stepped)
                                   (seq sub-event-log)
                                   (update :sub-replays (fnil conj []) sub-inst-stepped))]
               (recur (rest envs) inst-with-sub))

             :else
             (recur (rest envs) inst))))))))
