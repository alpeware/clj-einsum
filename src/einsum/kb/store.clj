(ns einsum.kb.store
  "Experiment E21: Schema-Constrained Knowledge Base with Verified Write Path.
   Tier 1 committed store conforming to Repository Rules 1 and 2 (Pure Functions, Sans-IO).
   Enforces:
   - Schema declaration (arity, entity sorts, cardinality, acyclicity denial, identity)
   - Verified commit pipeline: typecheck -> constraint check -> ambiguity check -> provenance stamp
   - First-class disjunction store for underdetermined relational ambiguities
   - Append-only provenance log with supersede markers for retractions
   - Host-side Datalog rule materialization (semi-naive / forward chaining)
   - Microsecond ground query read path")

;; ==============================================================================
;; 1. KB Construction & Entity Sort Registry
;; ==============================================================================

(defn create-kb
  "Initializes an immutable schema-constrained Knowledge Base.
   Opts may provide:
   - :entities (map of entity-id to sort keyword)"
  ([schema] (create-kb schema nil))
  ([schema opts]
   {:schema schema
    :entities (or (:entities opts) {})
    :facts (into {} (map (fn [[k _]] [k #{}]) schema))
    :disjunctions []
    :log []
    :tx-counter 0}))

(defn register-entity
  "Registers an entity with an associated domain sort keyword."
  [kb entity sort-kw]
  (assoc-in kb [:entities entity] sort-kw))

(defn register-entities
  "Registers a collection of entities with an associated domain sort keyword."
  [kb entities sort-kw]
  (reduce (fn [k e] (register-entity k e sort-kw)) kb entities))

;; ==============================================================================
;; 2. Graph Algorithms: Incremental Reachability & Cycle Detection
;; ==============================================================================

(defn reachable?
  "Checks whether target is reachable from start in a directed edge set.
   edges is a collection/set of [u v] vectors."
  [edges start target]
  (if (= start target)
    true
    (let [adj (group-by first edges)]
      (loop [queue (into clojure.lang.PersistentQueue/EMPTY (map second (get adj start)))
             visited #{start}]
        (if (empty? queue)
          false
          (let [curr (peek queue)
                rest-q (pop queue)]
            (cond
              (= curr target) true
              (contains? visited curr) (recur rest-q visited)
              :else
              (let [neighbors (map second (get adj curr))
                    new-neighbors (remove visited neighbors)]
                (recur (into rest-q new-neighbors)
                       (conj visited curr))))))))))

;; ==============================================================================
;; 3. Constraint Checkers (Pure Validation)
;; ==============================================================================

(defn check-domain
  "Validates that relation exists in schema, arity matches, and all arguments
   conform to the declared per-argument sorts."
  [kb rel args]
  (if-let [rel-def (get-in kb [:schema rel])]
    (let [arity (:arity rel-def)
          sorts (:sorts rel-def)]
      (cond
        (not= (count args) arity)
        {:error :schema-violation
         :violation :domain
         :relation rel
         :message (format "Arity mismatch for relation %s: expected %d, got %d" rel arity (count args))}

        :else
        (loop [idx 0]
          (if (>= idx arity)
            nil
            (let [val (nth args idx)
                  expected-sort (nth sorts idx)
                  actual-sort (get-in kb [:entities val])]
              (if (or (nil? actual-sort) (not= actual-sort expected-sort))
                {:error :schema-violation
                 :violation :domain
                 :relation rel
                 :arg-index idx
                 :value val
                 :expected expected-sort
                 :actual actual-sort}
                (recur (inc idx))))))))
    {:error :schema-violation
     :violation :domain
     :relation rel
     :message (format "Undeclared relation in schema: %s" rel)}))

(defn check-cardinality
  "Validates cardinality constraints, e.g. indegree <= max on child argument or role cap."
  [kb rel args]
  (if-let [card (get-in kb [:schema rel :cardinality])]
    (let [target-arg (long (:target-arg card))
          target-val (nth args target-arg)
          existing-facts (or (get-in kb [:facts rel]) #{})]
      (if (contains? existing-facts (vec args))
        nil ;; Idempotent commit of already present fact
        (cond
          ;; Value-specific caps, e.g. {:caps {:admin 3}}
          (and (:caps card) (contains? (:caps card) target-val))
          (let [max-allowed (long (get (:caps card) target-val))
                current-matches (filter #(= (nth % target-arg) target-val) existing-facts)
                current-count (count current-matches)]
            (if (>= current-count max-allowed)
              {:error :schema-violation
               :violation :cardinality
               :relation rel
               :target-arg target-val
               :current-count current-count
               :max max-allowed
               :message (format "Cardinality cap exceeded for %s: current %d, max %d" target-val current-count max-allowed)}
              nil))

          ;; General max cap, e.g. {:target-arg 1 :max 2}
          (:max card)
          (let [max-allowed (long (:max card))
                current-matches (filter #(= (nth % target-arg) target-val) existing-facts)
                current-count (count current-matches)]
            (if (>= current-count max-allowed)
              {:error :schema-violation
               :violation :cardinality
               :relation rel
               :target-arg target-val
               :current-count current-count
               :max max-allowed}
              nil))

          :else nil)))
    nil))

(defn check-denial
  "Validates denial constraints such as acyclicity on directed relations or separation-of-duty."
  [kb rel args]
  (if-let [denial (get-in kb [:schema rel :denial])]
    (cond
      ;; 1. Acyclicity denial
      (and (set? denial) (contains? denial :acyclic))
      (let [from (nth args 0)
            to (nth args 1)]
        (if (= from to)
          {:error :schema-violation
           :violation :acyclicity
           :relation rel
           :from from
           :to to
           :message "Self-loop creates a directed cycle"}
          (let [existing-edges (or (get-in kb [:facts rel]) #{})]
            (if (reachable? existing-edges to from)
              {:error :schema-violation
               :violation :acyclicity
               :relation rel
               :from from
               :to to}
              nil))))

      ;; 2. Separation-of-Duty denial: {:sod-pairs #{[:deployer :auditor] ...}}
      (or (:sod-pairs denial) (:sod denial))
      (let [sod-pairs (or (:sod-pairs denial) (:sod denial))
            u (nth args 0)
            r (nth args 1)
            existing-roles (set (map second (filter #(= (first %) u) (or (get-in kb [:facts rel]) #{}))))]
        (if-let [conflicting-pair
                 (some (fn [[r1 r2]]
                         (when (or (and (= r r1) (contains? existing-roles r2))
                                   (and (= r r2) (contains? existing-roles r1)))
                           [r1 r2]))
                       sod-pairs)]
          {:error :schema-violation
           :violation :separation-of-duty
           :relation rel
           :pair conflicting-pair
           :user u
           :message (format "Separation-of-duty violation: user %s cannot hold both %s and %s"
                            u (first conflicting-pair) (second conflicting-pair))}
          nil))

      :else nil)
    nil))

(defn check-identity
  "Validates identity constraints, such as sibling pair requiring identical parent pairs."
  [kb rel args]
  (if-let [identity-def (get-in kb [:schema rel :identity])]
    (if (= :parent (:via identity-def))
      (let [a (nth args 0)
            b (nth args 1)
            parent-facts (or (get-in kb [:facts :parent]) #{})
            parents-a (set (map first (filter #(= (second %) a) parent-facts)))
            parents-b (set (map first (filter #(= (second %) b) parent-facts)))]
        ;; When both have 2 known parents, verify equality
        (if (and (== 2 (count parents-a))
                 (== 2 (count parents-b))
                 (not= parents-a parents-b))
          {:error :schema-violation
           :violation :identity
           :relation rel
           :a a
           :b b
           :parents-a parents-a
           :parents-b parents-b}
          nil))
      nil)
    nil))

(defn check-ambiguity
  "Verifies that candidate fact is not an unresolved branch in an active disjunction."
  [kb rel args]
  (let [candidate-fact (vec (cons rel args))
        disjunctions (:disjunctions kb)]
    (if-let [disj (some (fn [d]
                          (when (some #(contains? % candidate-fact) (:branches d))
                            d))
                        disjunctions)]
      {:error :schema-violation
       :violation :ambiguity
       :relation rel
       :fact candidate-fact
       :disjunction-id (:id disj)
       :message "Cannot commit ambiguous candidate as definite fact without resolving disjunction"}
      nil)))

;; ==============================================================================
;; 4. Verified Commit & Provenance-Tracked Retraction
;; ==============================================================================

(defn assert-fact
  "Commits a single fact through the verified verification pipeline:
   typecheck -> constraint check -> ambiguity check -> provenance stamp.
   Returns the updated KB value, or an error map naming the exact violation."
  ([kb fact] (assert-fact kb fact nil))
  ([kb fact provenance]
   (let [rel (first fact)
         args (vec (rest fact))]
     (or (check-domain kb rel args)
         (check-cardinality kb rel args)
         (check-denial kb rel args)
         (check-identity kb rel args)
         (check-ambiguity kb rel args)
         (let [new-tx (inc (:tx-counter kb))
               fact-vec (vec (cons rel args))
               prov (or provenance {:source :assert})
               log-entry {:tx-id new-tx
                          :op :assert
                          :fact fact-vec
                          :provenance prov
                          :timestamp (System/currentTimeMillis)
                          :superseded-by nil}
               existing-facts (or (get-in kb [:facts rel]) #{})]
           (-> kb
               (assoc :tx-counter new-tx)
               (update :log conj log-entry)
               (assoc-in [:facts rel] (conj existing-facts args))))))))

(defn assert-facts
  "Commits a collection of facts sequentially.
   Aborts and returns the violation map upon the first failure."
  ([kb facts] (assert-facts kb facts nil))
  ([kb facts provenance]
   (reduce (fn [k f]
             (let [res (assert-fact k f provenance)]
               (if (:error res)
                 (reduced res)
                 res)))
           kb
           facts)))

(defn retract-fact
  "Retracts a previously asserted fact via an append-only supersede marker.
   Does not delete from history; stamps the supersede pointer into the log."
  ([kb fact] (retract-fact kb fact nil))
  ([kb fact provenance]
   (let [rel (first fact)
         args (vec (rest fact))
         fact-vec (vec (cons rel args))
         existing-facts (or (get-in kb [:facts rel]) #{})]
     (if-not (contains? existing-facts args)
       {:error :not-found
        :fact fact-vec
        :message "Fact not found in active KB facts"}
       (let [new-tx (inc (:tx-counter kb))
             prov (or provenance {:source :retract})
             ;; Find the latest non-superseded assertion transaction in the log
             latest-assert-idx
             (loop [idx (dec (count (:log kb)))]
               (if (< idx 0)
                 nil
                 (let [entry (nth (:log kb) idx)]
                   (if (and (= (:fact entry) fact-vec)
                            (= :assert (:op entry))
                            (nil? (:superseded-by entry)))
                     idx
                     (recur (dec idx))))))
             superseded-tx (when latest-assert-idx
                             (:tx-id (nth (:log kb) latest-assert-idx)))
             retract-entry {:tx-id new-tx
                            :op :retract
                            :fact fact-vec
                            :supersedes superseded-tx
                            :provenance prov
                            :timestamp (System/currentTimeMillis)}
             updated-log (if latest-assert-idx
                           (assoc-in (:log kb) [latest-assert-idx :superseded-by] new-tx)
                           (:log kb))]
         (-> kb
             (assoc :tx-counter new-tx)
             (assoc :log (conj updated-log retract-entry))
             (assoc-in [:facts rel] (disj existing-facts args))))))))

;; ==============================================================================
;; 5. First-Class Disjunction Store (Mating Ambiguity Representation)
;; ==============================================================================

(defn assert-disjunction
  "Stores a set of mutually exclusive relational completions explicitly.
   Each branch is a collection/set of candidate facts [rel & args].
   Rejects guessing or premature commitment."
  ([kb branches] (assert-disjunction kb branches nil))
  ([kb branches provenance]
   (let [disj-id (count (:disjunctions kb))
         prov (or provenance {:source :disjunction})
         branches-set (mapv (fn [b]
                              (if (and (vector? b) (keyword? (first b)))
                                #{b}
                                (set b)))
                            branches)
         entry {:id disj-id
                :branches branches-set
                :provenance prov
                :timestamp (System/currentTimeMillis)}]
     (update kb :disjunctions conj entry))))

(defn resolve-disjunction
  "Resolves an active disjunction to a definite fact, removing the disjunction from active store
   and asserting the chosen fact with :op :resolve-disjunction in the transaction log."
  ([kb disj-id chosen-fact] (resolve-disjunction kb disj-id chosen-fact nil))
  ([kb disj-id chosen-fact provenance]
   (let [disjunctions (:disjunctions kb)
         disj (some #(when (= (:id %) disj-id) %) disjunctions)]
     (cond
       (nil? disj)
       {:error :not-found :message (format "Disjunction id %s not found" disj-id)}

       (not (some #(contains? % chosen-fact) (:branches disj)))
       {:error :schema-violation :violation :invalid-branch :message "Chosen fact is not in disjunction branches"}

       :else
       (let [kb-without-disj (assoc kb :disjunctions (filterv #(not= (:id %) disj-id) disjunctions))
             rel (first chosen-fact)
             args (vec (rest chosen-fact))]
         (or (check-domain kb-without-disj rel args)
             (check-cardinality kb-without-disj rel args)
             (check-denial kb-without-disj rel args)
             (check-identity kb-without-disj rel args)
             (check-ambiguity kb-without-disj rel args)
             (let [new-tx (inc (:tx-counter kb-without-disj))
                   prov (or provenance {:source :resolve-disjunction :disjunction-id disj-id})
                   log-entry {:tx-id new-tx
                              :op :resolve-disjunction
                              :fact chosen-fact
                              :disjunction-id disj-id
                              :provenance prov
                              :timestamp (System/currentTimeMillis)
                              :superseded-by nil}
                   existing-facts (or (get-in kb-without-disj [:facts rel]) #{})]
               (-> kb-without-disj
                   (assoc :tx-counter new-tx)
                   (update :log conj log-entry)
                   (assoc-in [:facts rel] (conj existing-facts args))))))))))

(defn commit-with-schema
  "Commits an agent proposal through the verified schema pipeline:
   - :assert -> check no-op, validate schema (domain, cardinality, denial, identity, ambiguity), append log
   - :retract -> check no-op, validate held, stamp supersede marker
   - :ambiguous -> record first-class disjunction
   - :disambiguate -> resolve disjunction to definite fact
   Returns {:committed updated-kb} on success, or {:rejected violation-map} on failure."
  ([kb proposal]
   (let [[op payload prov]
         (cond
           (vector? proposal)
           [(first proposal) (rest proposal) nil]

           (map? proposal)
           [(:op proposal)
            (case (:op proposal)
              :assert [(:fact proposal)]
              :retract [(:fact proposal)]
              :ambiguous [(:branches proposal)]
              :disambiguate [(:disjunction-id proposal) (:chosen-fact proposal)]
              nil)
            (:provenance proposal)]

           :else
           [:unknown nil nil])]
     (case op
       :assert
       (let [fact (first payload)
             rel (first fact)
             args (vec (rest fact))
             existing (or (get-in kb [:facts rel]) #{})]
         (if (contains? existing args)
           {:rejected {:error :schema-violation
                       :violation :no-op
                       :fact fact
                       :message "Fact already held in KB"}}
           (let [res (assert-fact kb fact prov)]
             (if (:error res)
               {:rejected res}
               {:committed res}))))

       :retract
       (let [fact (first payload)
             rel (first fact)
             args (vec (rest fact))
             existing (or (get-in kb [:facts rel]) #{})]
         (if-not (contains? existing args)
           {:rejected {:error :schema-violation
                       :violation :no-op
                       :fact fact
                       :message "Fact not held in KB"}}
           (let [res (retract-fact kb fact prov)]
             (if (:error res)
               {:rejected res}
               {:committed res}))))

       :ambiguous
       (let [branches (first payload)
             res (assert-disjunction kb branches prov)]
         (if (:error res)
           {:rejected res}
           {:committed res}))

       :disambiguate
       (let [disj-id (first payload)
             chosen-fact (second payload)
             res (resolve-disjunction kb disj-id chosen-fact prov)]
         (if (:error res)
           {:rejected res}
           {:committed res}))

       {:rejected {:error :invalid-op :proposal proposal}})))
  ([schema kb proposal]
   (commit-with-schema (assoc kb :schema schema) proposal)))

;; ==============================================================================
;; 6. Host-Side Read Path & Datalog Materialization
;; ==============================================================================

(defn query-fact
  "Evaluates a ground query fact [rel & args].
   Returns:
   - {:status :true, :fact fact} if committed definitely
   - {:status :ambiguous, :fact fact, :completions [...], :disjunction-id id} if underdetermined
   - {:status :false, :fact fact} otherwise."
  [kb fact]
  (let [rel (first fact)
        args (vec (rest fact))
        fact-vec (vec (cons rel args))
        committed? (contains? (or (get-in kb [:facts rel]) #{}) args)]
    (if committed?
      {:status :true :fact fact-vec}
      ;; Check active disjunctions
      (if-let [matching-disj
               (some (fn [d]
                       (when (some #(contains? % fact-vec) (:branches d))
                         d))
                     (:disjunctions kb))]
        {:status :ambiguous
         :fact fact-vec
         :completions (:branches matching-disj)
         :disjunction-id (:id matching-disj)}
        {:status :false :fact fact-vec}))))

(defn materialize-derived
  "Computes host-side forward-chaining fixpoint for derived relations in schema.
   Currently implements 2-hop compositions such as grandparent/2."
  [kb]
  (let [schema (:schema kb)
        derived-rels (filter (fn [[_ def]] (:derived def)) schema)]
    (reduce (fn [cur-kb [rel rel-def]]
              (let [rule (:rule rel-def)]
                (if (and rule (= [[:parent :?x :?y] [:parent :?y :?z]] (:body rule)))
                  ;; Standard 2-hop relational join
                  (let [parents (or (get-in cur-kb [:facts :parent]) #{})
                        by-p (group-by first parents)
                        gp-pairs
                        (set
                         (for [[p c] parents
                               [_ g] (get by-p c)]
                           [p g]))]
                    (assoc-in cur-kb [:facts rel] gp-pairs))
                  cur-kb)))
            kb
            derived-rels)))

(defn history
  "Returns the full causal append-only transaction history for a fact or relation."
  [kb query]
  (cond
    (vector? query)
    (filterv #(= (:fact %) query) (:log kb))

    (keyword? query)
    (filterv #(= (first (:fact %)) query) (:log kb))

    :else
    (:log kb)))

;; ==============================================================================
;; 7. Read Latency Benchmarking
;; ==============================================================================

(defn measure-read-latency
  "Measures ground query latency across repeated trials in microseconds."
  [kb queries ^long n-trials]
  (let [n-q (count queries)
        ;; Warm up JIT
        _ (dotimes [_ 1000]
            (query-fact kb (nth queries (rem _ n-q))))
        timings (double-array n-trials)]
    (dotimes [i n-trials]
      (let [q (nth queries (rem i n-q))
            t0 (System/nanoTime)
            _ (query-fact kb q)
            t1 (System/nanoTime)
            us (/ (double (- t1 t0)) 1000.0)]
        (aset timings i us)))
    (java.util.Arrays/sort timings)
    (let [p50 (aget timings (int (* 0.50 n-trials)))
          p90 (aget timings (int (* 0.90 n-trials)))
          p99 (aget timings (int (* 0.99 n-trials)))
          sum (areduce timings idx s 0.0 (+ s (aget timings idx)))
          mean (/ sum (double n-trials))]
      {:p50-us p50
       :p90-us p90
       :p99-us p99
       :mean-us mean
       :min-us (aget timings 0)
       :max-us (aget timings (dec n-trials))})))
