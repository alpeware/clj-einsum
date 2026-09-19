(ns einsum.kb.rbac-test
  "Unit and generative property tests for Experiment E23: The Reduce (RBAC).
   Repository Rule 1: Generative tests for invariants written FIRST before implementation.
   Verifies:
   - Schema-driven verified commit pipeline (commit-with-schema)
   - Totality of integrity constraints: domain, separation-of-duty, cardinality, no-op, ambiguity
   - Oracle accumulator invariance: zero violating facts ever committed
   - Disjunction and disambiguation lifecycle
   - Full backward compatibility with E21 family schema
   - Gate P0a (oracle determinism) and Gate P0b (trap rejection)"
  (:require [einsum.kb.store :as kb]
            [einsum.kb.rbac :as rbac]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; ==============================================================================
;; 1. Generative Property Tests (Repository Rule 1)
;; ==============================================================================

(def gen-rbac-user
  (gen/elements [:alice :bob :carol :dan :erin :frank :grace :heidi :ivan :judy :mallory :oscar]))

(def gen-rbac-role
  (gen/elements [:admin :developer :deployer :auditor :oncall :viewer]))

(def gen-invalid-user
  (gen/elements [:eve :mallory_clone :root :trent :sybil :stranger]))

(def gen-invalid-role
  (gen/elements [:superadmin :root :guest :owner :hacker]))

(defspec prop-schema-commit-totality 50
  (prop/for-all [user gen-rbac-user
                 role gen-rbac-role
                 invalid-u? gen/boolean
                 invalid-r? gen/boolean]
                (let [kb (rbac/create-initial-rbac-kb)
                      u (if invalid-u? :eve user)
                      r (if invalid-r? :superadmin role)
                      proposal [:assert [:has_role u r]]
                      result (kb/commit-with-schema kb proposal)
                      existing-roles (set (map second (filter #(= (first %) u) (get-in kb [:facts :has_role]))))
                      admin-count (count (filter #(= (second %) :admin) (get-in kb [:facts :has_role])))]
                  (cond
        ;; Domain violation
                    (or invalid-u? invalid-r?)
                    (and (contains? result :rejected)
                         (= :domain (get-in result [:rejected :violation])))

        ;; No-op (already held)
                    (contains? existing-roles r)
                    (and (contains? result :rejected)
                         (= :no-op (get-in result [:rejected :violation])))

        ;; Separation-of-Duty violation
                    (some (fn [[r1 r2]]
                            (or (and (= r r1) (contains? existing-roles r2))
                                (and (= r r2) (contains? existing-roles r1))))
                          rbac/DEFAULT-SOD-PAIRS)
                    (and (contains? result :rejected)
                         (= :separation-of-duty (get-in result [:rejected :violation])))

        ;; Cardinality violation (admin cap)
                    (and (= r :admin) (>= admin-count rbac/DEFAULT-ADMIN-CAP))
                    (and (contains? result :rejected)
                         (= :cardinality (get-in result [:rejected :violation])))

        ;; Valid commit
                    :else
                    (and (contains? result :committed)
                         (contains? (get-in result [:committed :facts :has_role]) [u r]))))))

(defspec prop-oracle-accumulator-invariance 30
  (prop/for-all [seed gen/nat
                 stream-len (gen/choose 10 30)]
                (let [stream (rbac/generate-rbac-stream {:seed seed :length stream-len :trap-rate 0.30})
                      initial-kb (rbac/create-initial-rbac-kb)
                      final-kb (rbac/fold-oracle-stream initial-kb stream)
                      facts (get-in final-kb [:facts :has_role])
                      by-user (group-by first facts)
                      admins (filter #(= (second %) :admin) facts)]
                  (and
       ;; Invariant 1: Admin cardinality never exceeded
                   (<= (count admins) rbac/DEFAULT-ADMIN-CAP)

       ;; Invariant 2: Zero SoD violations for any user
                   (every? (fn [[_u user-facts]]
                             (let [roles (set (map second user-facts))]
                               (not-any? (fn [[r1 r2]]
                                           (and (contains? roles r1) (contains? roles r2)))
                                         rbac/DEFAULT-SOD-PAIRS)))
                           by-user)

       ;; Invariant 3: Zero undeclared entities
                   (every? (fn [[u r]]
                             (and (contains? (get-in final-kb [:entities]) u)
                                  (contains? (get-in final-kb [:entities]) r)))
                           facts)))))

(defspec prop-ambiguity-lifecycle-soundness 25
  (prop/for-all [u1 gen-rbac-user
                 u2 gen-rbac-user
                 role gen-rbac-role]
                (let [initial-kb (rbac/create-initial-rbac-kb)
                      eligible? (contains? (kb/commit-with-schema initial-kb [:assert [:has_role u1 role]]) :committed)]
                  (if (or (= u1 u2) (not eligible?))
                    true
                    (let [branches [[:has_role u1 role] [:has_role u2 role]]
              ;; 1. Propose ambiguous event
                          res1 (kb/commit-with-schema initial-kb [:ambiguous branches])
                          kb1 (:committed res1)]
                      (and
                       (some? kb1)
                       (= 1 (count (:disjunctions kb1)))
           ;; Querying branch returns :ambiguous
                       (= :ambiguous (:status (kb/query-fact kb1 [:has_role u1 role])))
           ;; Direct commit of candidate branch without resolution is rejected
                       (= :ambiguity (get-in (kb/commit-with-schema kb1 [:assert [:has_role u1 role]])
                                             [:rejected :violation]))
           ;; 2. Disambiguate to u1
                       (let [res2 (kb/commit-with-schema kb1 [:disambiguate 0 [:has_role u1 role]])
                             kb2 (:committed res2)]
                         (and
                          (some? kb2)
                          (empty? (:disjunctions kb2))
                          (= :true (:status (kb/query-fact kb2 [:has_role u1 role])))
                          (= :resolve-disjunction (:op (last (:log kb2))))))))))))

;; ==============================================================================
;; 2. Backward Compatibility with E21 Family Tree Schema
;; ==============================================================================

(deftest test-family-schema-backward-compatibility
  (testing "commit-with-schema preserves full E21 family tree semantics"
    (let [schema {:parent {:arity 2 :sorts [:person :person] :cardinality {:target-arg 1 :max 2} :denial #{:acyclic}}
                  :sibling {:arity 2 :sorts [:person :person] :identity {:via :parent :shared-parents true}}}
          entities {0 :person, 1 :person, 2 :person, 3 :person}
          kb (kb/create-kb schema {:entities entities})
          ;; Commit parent 0 -> 2
          res1 (kb/commit-with-schema kb [:assert [:parent 0 2]])
          kb1 (:committed res1)
          ;; Commit parent 1 -> 2
          res2 (kb/commit-with-schema kb1 [:assert [:parent 1 2]])
          kb2 (:committed res2)
          ;; Attempt 3rd parent 3 -> 2 (cardinality violation)
          res3 (kb/commit-with-schema kb2 [:assert [:parent 3 2]])
          ;; Attempt cycle 2 -> 0 (acyclicity violation)
          res4 (kb/commit-with-schema kb2 [:assert [:parent 2 0]])]
      (is (some? kb1))
      (is (some? kb2))
      (is (= 2 (count (get-in kb2 [:facts :parent]))))
      (is (= :cardinality (get-in res3 [:rejected :violation])))
      (is (= :acyclicity (get-in res4 [:rejected :violation]))))))

;; ==============================================================================
;; 3. Gate P0a: Oracle Determinism
;; ==============================================================================

(deftest test-gate-p0a-oracle-determinism
  (testing "Gate P0a: Stream generation and oracle applier are fully deterministic"
    (let [stream-20 (rbac/generate-rbac-stream {:seed 42 :length 20 :trap-rate 0.25})
          stream-60 (rbac/generate-rbac-stream {:seed 42 :length 60 :trap-rate 0.25})
          initial-kb (rbac/create-initial-rbac-kb)]
      ;; Assert shared prefix: first 20 events of stream-60 must equal stream-20
      (is (= stream-20 (subvec stream-60 0 20))
          "T=60 stream must share exact prefix with T=20 stream")

      ;; Fold oracle on T=20
      (let [kb-20 (rbac/fold-oracle-stream initial-kb stream-20)
            facts-20 (get-in kb-20 [:facts :has_role])]
        (is (seq facts-20))
        (is (not-any? (fn [[u r]]
                        (some (fn [[r1 r2]]
                                (and (or (= r r1) (= r r2))
                                     (contains? (set (map second (filter #(= (first %) u) facts-20)))
                                                (if (= r r1) r2 r1))))
                              rbac/DEFAULT-SOD-PAIRS))
                      facts-20)
            "Oracle KB at T=20 must have 0 SoD violations")
        (is (<= (count (filter #(= (second %) :admin) facts-20)) rbac/DEFAULT-ADMIN-CAP)
            "Oracle KB at T=20 must satisfy admin cap"))

      ;; Fold oracle on T=60
      (let [kb-60 (rbac/fold-oracle-stream initial-kb stream-60)
            facts-60 (get-in kb-60 [:facts :has_role])]
        (is (seq facts-60))
        (is (<= (count (filter #(= (second %) :admin) facts-60)) rbac/DEFAULT-ADMIN-CAP)
            "Oracle KB at T=60 must satisfy admin cap")))))

;; ==============================================================================
;; 4. Gate P0b: Trap Rejection (Offline Gate)
;; ==============================================================================

(deftest test-gate-p0b-trap-rejection
  (testing "Gate P0b: Every trap event is rejected naming its violation keyword"
    (let [kb (rbac/create-initial-rbac-kb)]
      ;; 1. SoD trap: Bob holds deployer; granting auditor must be rejected
      (let [res (kb/commit-with-schema kb [:assert [:has_role :bob :auditor]])]
        (is (contains? res :rejected))
        (is (= :separation-of-duty (get-in res [:rejected :violation]))))

      ;; 2. No-op trap: Alice already holds developer; duplicate grant must be rejected
      (let [res (kb/commit-with-schema kb [:assert [:has_role :alice :developer]])]
        (is (contains? res :rejected))
        (is (= :no-op (get-in res [:rejected :violation]))))

      ;; 3. No-op trap: Revoking role not held must be rejected
      (let [res (kb/commit-with-schema kb [:retract [:has_role :heidi :deployer]])]
        (is (contains? res :rejected))
        (is (= :no-op (get-in res [:rejected :violation]))))

      ;; 4. Domain trap: Undeclared user eve must be rejected
      (let [res (kb/commit-with-schema kb [:assert [:has_role :eve :viewer]])]
        (is (contains? res :rejected))
        (is (= :domain (get-in res [:rejected :violation]))))

      ;; 5. Cardinality trap: Dan is admin. Add Heidi and Ivan as admins (total 3). 4th admin must be rejected
      (let [kb1 (:committed (kb/commit-with-schema kb [:assert [:has_role :heidi :admin]]))
            kb2 (:committed (kb/commit-with-schema kb1 [:assert [:has_role :ivan :admin]]))
            res-4th (kb/commit-with-schema kb2 [:assert [:has_role :judy :admin]])]
        (is (= 3 (count (filter #(= (second %) :admin) (get-in kb2 [:facts :has_role])))))
        (is (contains? res-4th :rejected))
        (is (= :cardinality (get-in res-4th [:rejected :violation])))))))

;; ==============================================================================
;; 5. Snapshot Rendering & Proposal Parsing
;; ==============================================================================

(deftest test-snapshot-rendering-and-proposal-parsing
  (testing "Snapshot rendering produces clean bounded text view and proposals parse accurately"
    (let [kb (rbac/create-initial-rbac-kb)
          rendered (rbac/render-snapshot kb)]
      (is (str/includes? rendered "(has_role :alice :developer)"))
      (is (str/includes? rendered "(has_role :bob :deployer)"))
      (is (str/includes? rendered "(has_role :dan :admin)")))

    (testing "Proposals parse correctly from tool call syntax"
      (is (= {:op :assert :fact [:has_role :alice :developer]}
             (rbac/parse-proposal "<|tool_call> propose assert has_role alice developer<tool_call|>")))
      (is (= {:op :retract :fact [:has_role :bob :deployer]}
             (rbac/parse-proposal "<|tool_call> propose retract has_role bob deployer<tool_call|>")))
      (is (= {:op :ambiguous :branches [[:has_role :erin :oncall] [:has_role :frank :oncall]]}
             (rbac/parse-proposal "<|tool_call> propose ambiguous has_role erin oncall | has_role frank oncall<tool_call|>")))
      (is (= {:op :disambiguate :disjunction-id 0 :chosen-fact [:has_role :erin :oncall]}
             (rbac/parse-proposal "<|tool_call> propose resolve 0 has_role erin oncall<tool_call|>"))))))
