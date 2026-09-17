(ns clj-xla.logic.rbac
  "Role-Based Access Control (RBAC) domain specification, event generation,
   and snapshot management for Experiment E23: The Reduce.
   Pure Sans-IO logic (Repository Rule 2)."
  (:require [clj-xla.logic.kb :as kb]
            [clojure.set :as set]
            [clojure.string :as str]))

;; ==============================================================================
;; 1. Domain Constants & Schema
;; ==============================================================================

(def DEFAULT-USERS
  [:alice :bob :carol :dan :erin :frank :grace :heidi :ivan :judy :mallory :oscar])

(def DEFAULT-ROLES
  [:admin :developer :deployer :auditor :oncall :viewer])

(def DEFAULT-PERMS
  [:read_code :write_code :deploy_prod :view_logs :audit_access :restart_services :manage_users :view_metrics])

(def DEFAULT-ROLE-PERMS
  {:developer #{:read_code :write_code :view_logs}
   :deployer #{:deploy_prod :restart_services :view_logs :view_metrics}
   :auditor #{:audit_access :view_logs :view_metrics}
   :admin #{:manage_users :view_logs :view_metrics :restart_services}
   :oncall #{:view_logs :restart_services :view_metrics}
   :viewer #{:view_logs :view_metrics}})

(def DEFAULT-SOD-PAIRS
  #{[:deployer :auditor]
    [:developer :auditor]
    [:admin :deployer]})

(def DEFAULT-ADMIN-CAP 3)

(def rbac-schema
  {:has_role {:arity 2
              :sorts [:user :role]
              :cardinality {:target-arg 1 :caps {:admin DEFAULT-ADMIN-CAP}}
              :denial {:sod-pairs DEFAULT-SOD-PAIRS}}
   :role_perm {:arity 2
               :sorts [:role :perm]}})

(defn make-entities
  ([] (make-entities DEFAULT-USERS DEFAULT-ROLES DEFAULT-PERMS))
  ([users roles perms]
   (into {}
         (concat
          (map (fn [u] [u :user]) users)
          (map (fn [r] [r :role]) roles)
          (map (fn [p] [p :perm]) perms)))))

(defn create-initial-rbac-kb
  "Creates the initial RBAC KB with declared entities and initial assignments:
   - Alice: developer
   - Bob: deployer
   - Carol: auditor
   - Dan: admin
   - Erin: viewer"
  ([] (create-initial-rbac-kb {}))
  ([opts]
   (let [users (or (:users opts) DEFAULT-USERS)
         roles (or (:roles opts) DEFAULT-ROLES)
         perms (or (:perms opts) DEFAULT-PERMS)
         sod-pairs (or (:sod-pairs opts) DEFAULT-SOD-PAIRS)
         admin-cap (or (:admin-cap opts) DEFAULT-ADMIN-CAP)
         schema (assoc-in rbac-schema [:has_role :cardinality :caps :admin] admin-cap)
         schema (assoc-in schema [:has_role :denial :sod-pairs] sod-pairs)
         entities (make-entities users roles perms)
         init-kb (kb/create-kb schema {:entities entities})
         initial-facts [[:has_role :alice :developer]
                        [:has_role :bob :deployer]
                        [:has_role :carol :auditor]
                        [:has_role :dan :admin]
                        [:has_role :erin :viewer]]]
     (reduce (fn [k f]
               (let [res (kb/assert-fact k f {:source :init})]
                 (if (:error res)
                   (throw (ex-info "Failed to assert initial RBAC fact" {:fact f :error res}))
                   res)))
             init-kb
             initial-facts))))

;; ==============================================================================
;; 2. Snapshot Rendering
;; ==============================================================================

(defn render-snapshot
  "Renders a bounded text view of the KB snapshot for model prompting (§7).
   Only includes active facts and open disjunctions, never full causal history."
  ([kb] (render-snapshot kb nil))
  ([kb step]
   (let [facts (sort-by (juxt first second) (get-in kb [:facts :has_role]))
         disjunctions (:disjunctions kb)
         header (if step
                  (format ";; KB snapshot (t=%d)" step)
                  ";; KB snapshot")
         fact-lines (map (fn [[u r]] (format "(has_role %s %s)" (pr-str u) (pr-str r))) facts)
         disj-lines (when (seq disjunctions)
                      (concat
                       ["" ";; ambiguous"]
                       (map (fn [d]
                              (let [branches (:branches d)
                                    branch-strs (map (fn [b]
                                                       (let [f (first b)]
                                                         (format "(has_role %s %s)"
                                                                 (pr-str (nth f 1))
                                                                 (pr-str (nth f 2)))))
                                                     branches)]
                                (format "(one-of %s)" (str/join " " branch-strs))))
                            disjunctions)))]
     (str/join "\n" (concat [header] fact-lines disj-lines)))))

;; ==============================================================================
;; 3. Proposal Parsing & Tool Responses
;; ==============================================================================

(defn parse-proposal
  "Parses model proposal syntax emitted within native tool call tags:
   - <|tool_call> propose assert has_role alice developer<tool_call|>
   - <|tool_call> propose retract has_role bob deployer<tool_call|>
   - <|tool_call> propose ambiguous has_role erin oncall | has_role frank oncall<tool_call|>
   - <|tool_call> propose resolve 0 has_role erin oncall<tool_call|>
   Also tolerates parenthesized or Clojure vector syntax."
  [text]
  (when (string? text)
    (let [call-match (or (second (re-find #"(?s)<\|tool_call\|?>(.*?)(?:<tool_call\|?>|$)" text))
                         text)
          clean-text (-> call-match
                         (str/replace #"[()]" " ")
                         (str/replace #"[\",]" " ")
                         str/trim)]
      (cond
        ;; 1. Ambiguous: has | separating branches or begins with ambiguous
        (or (str/includes? clean-text "|")
            (re-find #"(?:propose\s+)?ambiguous\s+" clean-text))
        (let [branches-str (str/replace clean-text #"^.*?(?:propose\s+)?(?:ambiguous|resolve(?:\s+ID|\s+\d+)?)\s+" "")
              parts (str/split branches-str #"\|")
              branches (keep (fn [part]
                               (when-let [[_ _rel u r] (re-find #"(has_role|role)\s+:?([a-zA-Z0-9_-]+)\s+:?([a-zA-Z0-9_-]+)" part)]
                                 [:has_role (keyword u) (keyword r)]))
                             parts)]
          (when (>= (count branches) 2)
            {:op :ambiguous :branches (vec branches)}))

        ;; 2. Resolve/disambiguate: [propose] resolve [ID] [<id>] [has_role] u r
        (re-find #"(?i)(?:propose\s+)?(?:resolve|resolved|disambiguate)\b" clean-text)
        (if-let [[_ _id-str u r] (re-find #"(?i)(?:propose\s+)?(?:resolve|resolved|disambiguate)\s+(?:ID\s+)?(\d+)\s+(?:has_role\s+)?:?([a-zA-Z0-9_-]+)\s+:?([a-zA-Z0-9_-]+)" clean-text)]
          {:op :disambiguate
           :disjunction-id (Long/parseLong _id-str)
           :chosen-fact [:has_role (keyword (str/lower-case u)) (keyword (str/lower-case r))]}
          (when-let [[_ u r] (re-find #"(?i)(?:propose\s+)?(?:resolve|resolved|disambiguate)\s+(?:ID\s+)?(?:has_role\s+)?:?([a-zA-Z0-9_-]+)\s+:?([a-zA-Z0-9_-]+)" clean-text)]
            {:op :disambiguate
             :disjunction-id 0
             :chosen-fact [:has_role (keyword (str/lower-case u)) (keyword (str/lower-case r))]}))

        ;; 3. Assert: [propose] assert has_role u r
        (re-find #"(?:propose\s+)?assert\s+" clean-text)
        (when-let [[_ u r] (re-find #"(?:propose\s+)?assert\s+(?:has_role\s+)?:?([a-zA-Z0-9_-]+)\s+:?([a-zA-Z0-9_-]+)" clean-text)]
          {:op :assert :fact [:has_role (keyword u) (keyword r)]})

        ;; 4. Retract: [propose] retract has_role u r
        (re-find #"(?:propose\s+)?retract\s+" clean-text)
        (when-let [[_ u r] (re-find #"(?:propose\s+)?retract\s+(?:has_role\s+)?:?([a-zA-Z0-9_-]+)\s+:?([a-zA-Z0-9_-]+)" clean-text)]
          {:op :retract :fact [:has_role (keyword u) (keyword r)]})

        :else nil))))

(defn format-proposal-response
  "Formats the gate observation feedback to the model within native tool response tokens."
  [result]
  (cond
    (:committed result)
    (let [kb (:committed result)
          last-entry (last (:log kb))
          op (:op last-entry)
          fact (:fact last-entry)]
      (case op
        :assert (format "<|tool_response> committed: %s %s %s<tool_response|>"
                        (name (first fact)) (name (nth fact 1)) (name (nth fact 2)))
        :retract (format "<|tool_response> committed: retracted %s %s %s<tool_response|>"
                         (name (first fact)) (name (nth fact 1)) (name (nth fact 2)))
        :resolve-disjunction (format "<|tool_response> committed: resolved disjunction %s to %s %s %s<tool_response|>"
                                     (:disjunction-id last-entry)
                                     (name (first fact)) (name (nth fact 1)) (name (nth fact 2)))
        "<|tool_response> committed<tool_response|>"))

    (:rejected result)
    (let [rej (:rejected result)
          violation (:violation rej)]
      (case violation
        :separation-of-duty
        (let [[r1 r2] (:pair rej)]
          (format "<|tool_response> rejected: separation-of-duty([%s %s])<tool_response|>"
                  (name r1) (name r2)))

        :cardinality
        (format "<|tool_response> rejected: cardinality(%s cap %d)<tool_response|>"
                (name (or (:target-arg rej) :admin))
                (long (or (:max rej) DEFAULT-ADMIN-CAP)))

        :domain
        (format "<|tool_response> rejected: domain(undeclared entity %s)<tool_response|>"
                (pr-str (:value rej)))

        :no-op
        "<|tool_response> rejected: no-op<tool_response|>"

        :ambiguity
        "<|tool_response> rejected: ambiguity(unresolved branch)<tool_response|>"

        (format "<|tool_response> rejected: %s<tool_response|>" (name (or violation :invalid)))))

    :else
    "<|tool_response> error: unrecognized response<tool_response|>"))

;; ==============================================================================
;; 4. Seeded Event Stream Generator & Oracle Applier
;; ==============================================================================

(defn generate-rbac-stream
  "Generates a scripted stream of T natural-language observations.
   Guarantees deterministic event generation and shared prefix between horizons (e.g. T=20 and T=60)."
  [{:keys [seed length trap-rate]
    :or {seed 42 length 20 trap-rate 0.25}}]
  (let [rnd (java.util.Random. (long seed))
        pool-valid-grants [[:grace :developer]
                           [:heidi :viewer]
                           [:ivan :viewer]
                           [:judy :developer]
                           [:frank :viewer]
                           [:alice :oncall]
                           [:oscar :developer]
                           [:mallory :viewer]]
        pool-valid-revokes [[:erin :viewer]
                            [:grace :developer]
                            [:oscar :developer]]
        pool-sod-traps [[:bob :auditor]     ;; Bob is deployer -> violates [:deployer :auditor]
                        [:carol :developer]  ;; Carol is auditor -> violates [:developer :auditor]
                        [:dan :deployer]]    ;; Dan is admin -> violates [:admin :deployer]
        pool-cardinality-traps [[:mallory :admin]
                                [:judy :admin]
                                [:oscar :admin]]
        pool-domain-traps [[:eve :developer]
                           [:trent :viewer]
                           [:sybil :oncall]
                           [:mallory :root]]
        pool-noop-traps [[:alice :developer] ;; already held
                         [:dan :admin]       ;; already held
                         [:heidi :deployer]] ;; revoke not held
        ]
    (mapv
     (fn [t]
       (cond
         ;; Ambiguity lifecycle at steps 5 and 12
         (= t 5)
         {:step t
          :kind :ambiguous
          :valid? true
          :text "Ticket #412 scan is blurry: role oncall was granted to either Erin or Frank."
          :proposal [:ambiguous [[:has_role :erin :oncall] [:has_role :frank :oncall]]]}

         (= t 12)
         {:step t
          :kind :disambiguate
          :valid? true
          :text "HR confirmed: oncall ticket #412 was assigned to Erin."
          :proposal [:disambiguate 0 [:has_role :erin :oncall]]}

         ;; Traps (25% rate default)
         (< (.nextDouble rnd) trap-rate)
         (let [trap-choice (.nextInt rnd 4)]
           (case trap-choice
             ;; 1. SoD trap
             0 (let [pair (nth pool-sod-traps (.nextInt rnd (count pool-sod-traps)))]
                 {:step t
                  :kind :trap-sod
                  :valid? false
                  :text (format "Internal request: grant %s the %s role."
                                (str/capitalize (name (first pair))) (name (second pair)))
                  :proposal [:assert [:has_role (first pair) (second pair)]]})

             ;; 2. Cardinality trap
             1 (let [pair (nth pool-cardinality-traps (.nextInt rnd (count pool-cardinality-traps)))]
                 {:step t
                  :kind :trap-cardinality
                  :valid? false
                  :text (format "Escalation ticket: grant %s the admin role."
                                (str/capitalize (name (first pair))))
                  :proposal [:assert [:has_role (first pair) :admin]]})

             ;; 3. Domain trap
             2 (let [pair (nth pool-domain-traps (.nextInt rnd (count pool-domain-traps)))]
                 {:step t
                  :kind :trap-domain
                  :valid? false
                  :text (format "External request: grant %s the %s role."
                                (str/capitalize (name (first pair))) (name (second pair)))
                  :proposal [:assert [:has_role (first pair) (second pair)]]})

             ;; 4. No-op trap
             3 (let [pair (nth pool-noop-traps (.nextInt rnd (count pool-noop-traps)))]
                 (if (= (first pair) :heidi)
                   {:step t
                    :kind :trap-noop
                    :valid? false
                    :text "Revocation notice: revoke Heidi's deployer role."
                    :proposal [:retract [:has_role :heidi :deployer]]}
                   {:step t
                    :kind :trap-noop
                    :valid? false
                    :text (format "Routine access review: grant %s the %s role."
                                  (str/capitalize (name (first pair))) (name (second pair)))
                    :proposal [:assert [:has_role (first pair) (second pair)]]}))))

         ;; Valid event
         :else
         (let [revoke? (and (> t 15) (< (.nextDouble rnd) 0.30))]
           (if revoke?
             (let [pair (nth pool-valid-revokes (.nextInt rnd (count pool-valid-revokes)))]
               {:step t
                :kind :revoke
                :valid? true
                :text (format "HR update: revoke %s from the %s role."
                              (str/capitalize (name (first pair))) (name (second pair)))
                :proposal [:retract [:has_role (first pair) (second pair)]]})
             (let [pair (nth pool-valid-grants (.nextInt rnd (count pool-valid-grants)))]
               {:step t
                :kind :grant
                :valid? true
                :text (format "Ticket update: grant %s the %s role."
                              (str/capitalize (name (first pair))) (name (second pair)))
                :proposal [:assert [:has_role (first pair) (second pair)]]})))))
     (range length))))

(defn fold-oracle-stream
  "Folds the event stream deterministically through commit-with-schema.
   Only schema-valid proposals are committed, maintaining ground truth."
  [initial-kb stream]
  (reduce (fn [kb event]
            (if-let [prop (:proposal event)]
              (let [res (kb/commit-with-schema kb prop)]
                (if-let [new-kb (:committed res)]
                  new-kb
                  kb))
              kb))
          initial-kb
          stream))

;; ==============================================================================
;; 5. Interrogation & Metrics Evaluation
;; ==============================================================================

(defn evaluate-snapshot-f1
  "Calculates Precision, Recall, and F1 score of a target KB snapshot against oracle KB ground truth."
  [oracle-kb target-kb]
  (let [oracle-facts (or (get-in oracle-kb [:facts :has_role]) #{})
        target-facts (or (get-in target-kb [:facts :has_role]) #{})
        tp (count (set/intersection oracle-facts target-facts))
        fp (count (set/difference target-facts oracle-facts))
        fn-cnt (count (set/difference oracle-facts target-facts))
        precision (if (zero? (+ tp fp)) 1.0 (/ (double tp) (double (+ tp fp))))
        recall (if (zero? (+ tp fn-cnt)) 1.0 (/ (double tp) (double (+ tp fn-cnt))))
        f1 (if (zero? (+ precision recall))
             0.0
             (/ (* 2.0 precision recall) (+ precision recall)))
        oracle-disjs (set (:disjunctions oracle-kb))
        target-disjs (set (:disjunctions target-kb))
        disj-match? (= oracle-disjs target-disjs)]
    {:precision precision
     :recall recall
     :f1 f1
     :tp tp
     :fp fp
     :fn fn-cnt
     :disj-match? disj-match?
     :oracle-count (count oracle-facts)
     :target-count (count target-facts)}))
