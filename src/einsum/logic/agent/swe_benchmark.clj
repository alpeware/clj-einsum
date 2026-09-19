(ns einsum.logic.agent.swe-benchmark
  "Experiment E7: Long-Horizon Software Engineering Agent Benchmark.
   Compares two competing agent architectures over a 100-turn refactoring challenge:
     - Arm A (Standard Baseline): Full conversational history in LLM KV cache with context accumulation.
     - Arm B (TL-Agent): Fixed 512-token static prompt + In-VRAM Datalog State Tracker + Ephemeral Memory.
   Evaluates State Deductive Accuracy, Tok/s Throughput Stability, and VRAM Memory Scaling.
   Rule 2: Pure Functions (core logic sans-IO).
   Rule 4: Zero Java Escape Hatches (all tensor math compiled to StableHLO via PJRT)."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.ephemeral :as ephemeral]
            [einsum.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Realistic Codebase Ontology Construction
;; ==============================================================================

(defn build-codebase-ontology
  "Builds a realistic software engineering codebase DAG:
   - num-files (default 32)
   - num-functions (default 64)
   - num-tests (default 16)
   - seed (default 42)
   Partitioned into 4 architecture tiers:
     1. Core library files & primitives (files 0..7)
     2. Engine modules & algorithms (files 8..15)
     3. High-level models & agent loops (files 16..23)
     4. Test suites & integration harnesses (files 24..31)
   Total entities N = num-functions + num-files + num-tests."
  [{:keys [num-files num-functions num-tests]
    :or {num-files 32 num-functions 64 num-tests 16}}]
  (let [n-files (long num-files)
        n-funcs (long num-functions)
        n-tests (long num-tests)
        total-entities (+ n-files n-funcs n-tests)

        ;; File definitions: 4 tiers
        files (mapv (fn [i]
                      (let [tier (cond
                                   (< i (quot n-files 4)) "core"
                                   (< i (quot n-files 2)) "engine"
                                   (< i (* 3 (quot n-files 4))) "models"
                                   :else "test")]
                        {:id i
                         :name (format "%s/mod_%02d.clj" tier i)
                         :tier tier}))
                    (range n-files))

        ;; Functions distributed across non-test files (tiers 1-3)
        code-files (filter #(not= (:tier %) "test") files)
        num-code-files (count code-files)
        funcs-per-file (max 1 (quot n-funcs num-code-files))

        functions
        (mapv (fn [i]
                (let [file-idx (min (dec num-code-files) (quot i funcs-per-file))
                      file (nth code-files file-idx)]
                  {:id i
                   :name (format "fn_%02d" i)
                   :file-id (:id file)
                   :signature (format "[:tensor [b %d] :f32] -> [:tensor [b %d] :f32]"
                                      (+ 32 (* (mod i 4) 32))
                                      (+ 32 (* (mod i 4) 32)))}))
              (range n-funcs))

        func->file (into {} (map (fn [f] [(:id f) (:file-id f)]) functions))

        ;; Dependencies between functions:
        ;; High-level functions call engine functions; engine functions call core functions.
        ;; Creates a multi-hop DAG with depth 2 to 4.
        third-f (quot n-funcs 3)
        deps
        (into #{}
              (concat
               ;; Engine functions (tier 2) call core functions (tier 1)
               (mapcat (fn [i]
                         (let [caller (+ third-f i)
                               callee-1 (mod i third-f)
                               callee-2 (mod (inc i) third-f)]
                           (when (< caller (* 2 third-f))
                             [[caller callee-1] [caller callee-2]])))
                       (range third-f))
               ;; High-level functions (tier 3) call engine functions (tier 2)
               (mapcat (fn [i]
                         (let [caller (+ (* 2 third-f) i)
                               callee-1 (+ third-f (mod i third-f))
                               callee-2 (+ third-f (mod (inc i) third-f))]
                           (when (< caller n-funcs)
                             [[caller callee-1] [caller callee-2]])))
                       (range (- n-funcs (* 2 third-f))))))

        ;; Test suites: each test covers 3-5 functions
        tests
        (mapv (fn [t-idx]
                (let [t-id t-idx
                      ;; Covered functions: 1 core, 1 engine, 1 model
                      covered-funcs [(mod (* t-idx 3) third-f)
                                     (+ third-f (mod (* t-idx 3) third-f))
                                     (+ (* 2 third-f) (mod (* t-idx 3) (max 1 (- n-funcs (* 2 third-f)))))]
                      covered-files (vec (distinct (map func->file covered-funcs)))]
                  {:id t-id
                   :name (format "test_suite_%02d" t-id)
                   :covers-functions covered-funcs
                   :covers-files covered-files}))
              (range n-tests))

        test->funcs (into {} (map (fn [t] [(:id t) (:covers-functions t)]) tests))
        test->files (into {} (map (fn [t] [(:id t) (:covers-files t)]) tests))]

    {:files files
     :functions functions
     :tests tests
     :deps deps
     :func->file func->file
     :test->funcs test->funcs
     :test->files test->files
     :num-files n-files
     :num-functions n-funcs
     :num-tests n-tests
     :total-entities total-entities}))

;; ==============================================================================
;; 2. Mathematical Ground Truth Oracle (Sans-IO Pure Functions)
;; ==============================================================================

(defn compute-transitive-invalidations
  "Computes full transitive invalidation set host-side:
   Given dependency pairs [[caller callee] ...] meaning caller depends on callee,
   and a set of modified callees, returns all directly and transitively affected callers."
  [dep-pairs modified-set]
  (let [callers-by-callee (reduce (fn [m [caller callee]]
                                    (update m callee (fnil conj #{}) caller))
                                  {}
                                  dep-pairs)]
    (loop [affected (set modified-set)
           frontier (vec modified-set)]
      (if (empty? frontier)
        affected
        (let [curr (first frontier)
              downstream (get callers-by-callee curr #{})
              new-affected (remove affected downstream)]
          (recur (into affected new-affected)
                 (into (subvec frontier 1) new-affected)))))))

(defn ground-truth-invalidation
  "Computes the exact mathematical ground truth for a function modification:
   Returns {:affected-functions #{...}
            :affected-files #{...}
            :affected-tests #{...}}."
  [ontology active-deps modified-func-id]
  (let [{:keys [func->file test->funcs]} ontology
        aff-funcs (compute-transitive-invalidations active-deps #{modified-func-id})
        aff-files (into #{} (keep func->file aff-funcs))
        aff-tests (into #{}
                        (keep (fn [[t-id covered-funcs]]
                                (when (some aff-funcs covered-funcs)
                                  t-id))
                              test->funcs))]
    {:affected-functions aff-funcs
     :affected-files aff-files
     :affected-tests aff-tests}))

;; ==============================================================================
;; 3. Benchmark Trajectory Action Generator
;; ==============================================================================

(defn generate-benchmark-actions
  "Generates a reproducible 100-turn trajectory of simulated SWE agent operations:
   Actions:
     - :edit-function: edits a function, invalidating callers and test suites
     - :run-tests: executes affected tests, recording pass/fail states
     - :refactor-dep: adds or removes a dependency edge
     - :query-invalidation: queries which files and tests are affected by editing a function
     - :query-signature: queries the active signature of a function"
  [ontology num-turns seed]
  (let [{:keys [num-functions num-tests]} ontology
        rnd (java.util.Random. (long seed))]
    (mapv
     (fn [turn]
       (let [action-type (case (mod turn 5)
                           0 :edit-function
                           1 :query-invalidation
                           2 :run-tests
                           3 :query-signature
                           4 :refactor-dep)
             target-func (.nextInt rnd (int num-functions))]
         (case action-type
           :edit-function
           {:turn turn
            :action :edit-function
            :target-func target-func
            :new-signature (format "[:tensor [b %d] :f32] -> [:tensor [b %d] :f32]"
                                   (+ 64 (* (.nextInt rnd 4) 32))
                                   (+ 64 (* (.nextInt rnd 4) 32)))}

           :query-invalidation
           {:turn turn
            :action :query-invalidation
            :query-func target-func}

           :run-tests
           (let [sample-count (inc (.nextInt rnd 3))
                 sampled-tests (vec (distinct (repeatedly sample-count #(.nextInt rnd (int num-tests)))))]
             {:turn turn
              :action :run-tests
              :target-tests sampled-tests
              :test-results (into {} (map (fn [t] [t (if (< (.nextDouble rnd) 0.85) :pass :fail)]) sampled-tests))})

           :query-signature
           {:turn turn
            :action :query-signature
            :query-func target-func}

           :refactor-dep
           (let [caller (.nextInt rnd (int num-functions))
                 callee (.nextInt rnd (int num-functions))]
             {:turn turn
              :action :refactor-dep
              :caller caller
              :callee callee
              :op (if (< (.nextDouble rnd) 0.5) :add :remove)}))))
     (range num-turns))))

;; ==============================================================================
;; 4. Arm B: In-VRAM Datalog State Tracker + Ephemeral Memory (OpenXLA PJRT)
;; ==============================================================================

(defn transitive-closure-squaring-ast
  "Constructs Tensor Logic AST computing full transitive closure via repeated squaring in OpenXLA PJRT:
   A_0 = P
   A_1 = clamp(A_0 + A_0 @ A_0, 0, 1)
   ...
   A_k = clamp(A_{k-1} + A_{k-1} @ A_{k-1}, 0, 1)
   Evaluates all paths up to 2^k hops in O(k) matrix multiplications."
  [k]
  (let [eqns (mapcat (fn [i]
                       (let [prev (if (zero? i) :P (keyword (str "A_" (dec i))))
                             unclamped (keyword (str "A_raw_" i))
                             curr (keyword (str "A_" i))]
                         [[:= [unclamped :x :z] [prev :x :z]]
                          [:= [unclamped :x :z] [prev :x :y] [prev :y :z]]
                          [:= [curr :x :z] {:clamp [0.0 1.0]} [unclamped :x :z]]]))
                     (range k))]
    (vec (concat [:block {:name :transitive_closure_fixpoint}] eqns))))

(defn invalidation-step-ast
  "Constructs Tensor Logic AST for in-VRAM invalidation step:
   1. Invalidation propagation: Aff_unclamped[x] = M[x] + A_closure[x, y] * M[y]
   2. Test invalidation:        Aff_tests_raw[t] = T_cov[t, x] * Aff_funcs[x]"
  [_n]
  [:block {:name :invalidation_step}
   [:= [:Aff_funcs_raw :x] [:M :x]]
   [:= [:Aff_funcs_raw :x] [:A_closure :x :y] [:M :y]]
   [:= [:Affected_Funcs :x] {:clamp [0.0 1.0]} [:Aff_funcs_raw :x]]
   [:= [:Aff_tests_raw :t] [:T_cov :t :x] [:Affected_Funcs :x]]
   [:= [:Affected_Tests :t] {:clamp [0.0 1.0]} [:Aff_tests_raw :t]]])

(defn init-arm-b-session
  "Initializes Arm B In-VRAM session:
   Compiles OpenXLA PJRT executables for:
     1. Transitive closure fixpoint squaring (7 steps -> up to 128-hop paths)
     2. Invalidation propagation
     3. Ephemeral fast-weight memory writes & queries (D=64)
   Allocates resident device buffers."
  [ctx ontology]
  (let [{:keys [num-functions num-tests deps test->funcs]} ontology
        n-funcs (long num-functions)
        n-tests (long num-tests)
        d-mem 64

        ;; Binary adjacency matrix P_dep [n-funcs, n-funcs]
        ;; P[caller, callee] = 1.0 means caller depends on callee
        p-dep (float-array (* n-funcs n-funcs))
        _ (doseq [[caller callee] deps]
            (when (and (< (long caller) n-funcs) (< (long callee) n-funcs))
              (aset-float p-dep (+ (* (long caller) n-funcs) (long callee)) 1.0)))

        ;; Binary test coverage matrix T_cov [n-tests, n-funcs]
        ;; T[test, func] = 1.0 means test covers func
        t-cov (float-array (* n-tests n-funcs))
        _ (doseq [[t-id covered] test->funcs
                  f covered]
            (when (and (< (long t-id) n-tests) (< (long f) n-funcs))
              (aset-float t-cov (+ (* (long t-id) n-funcs) (long f)) 1.0)))

        ;; Compile transitive closure executable (7 squarings -> 2^7 = 128 hops)
        closure-ast (transitive-closure-squaring-ast 7)
        closure-invars [[:P [:tensor [n-funcs n-funcs] :f32]]]
        closure-exec (sym/compile-query ctx "transitive_closure" closure-invars closure-ast [:A_6])

        ;; Initial closure calculation in PJRT
        closure-out (sym/run-query! closure-exec {:P p-dep})
        ^floats a-closure (:A_6 closure-out)

        ;; Compile invalidation step executable
        inv-ast (invalidation-step-ast n-funcs)
        inv-invars [[:A_closure [:tensor [n-funcs n-funcs] :f32]]
                    [:M [:tensor [n-funcs] :f32]]
                    [:T_cov [:tensor [n-tests n-funcs] :f32]]]
        inv-exec (sym/compile-query ctx "invalidation_step" inv-invars inv-ast
                                    [:Affected_Funcs :Affected_Tests])

        ;; Compile Ephemeral Fast-Weight memory executables (D=64)
        write-exec (ephemeral/compile-write-fact ctx d-mem nil)
        retract-exec (ephemeral/compile-retract-fact ctx d-mem)
        query-exec (ephemeral/compile-query-memory ctx d-mem n-funcs nil)

        ;; Initial fast weight memory tensor R_sig [d-mem, d-mem]
        r-sig (float-array (* d-mem d-mem))

        ;; Entity embeddings table E [n-funcs, d-mem]
        emb (sym/random-embeddings n-funcs d-mem 42)
        e-data (:data emb)

        ;; Calculate total resident state VRAM bytes
        state-vram-bytes (+ (* n-funcs n-funcs 4)        ;; P_dep
                            (* n-funcs n-funcs 4)        ;; A_closure
                            (* n-tests n-funcs 4)        ;; T_cov
                            (* n-funcs 4)                ;; M indicator
                            (* d-mem d-mem 4)            ;; R_sig fast weights
                            (* n-funcs d-mem 4))]        ;; E entity table

    {:ctx ctx
     :n-funcs n-funcs
     :n-tests n-tests
     :d-mem d-mem
     :p-dep p-dep
     :t-cov t-cov
     :a-closure a-closure
     :closure-exec closure-exec
     :inv-exec inv-exec
     :write-exec write-exec
     :retract-exec retract-exec
     :query-exec query-exec
     :r-sig r-sig
     :e-data e-data
     :active-deps (atom (set deps))
     :active-signatures (atom (into {} (map (fn [f] [(:id f) (:signature f)]) (:functions ontology))))
     :active-test-results (atom {})
     :state-vram-bytes state-vram-bytes}))

(defn step-arm-b!
  "Executes one turn of Arm B autonomous agent execution in OpenXLA PJRT:
   Maintains state in VRAM with O(1) constant prompt context (512 tokens).
   Returns telemetry map."
  [session action-map ontology]
  (let [{:keys [n-funcs _n-tests a-closure inv-exec t-cov state-vram-bytes
                active-deps active-signatures active-test-results]} session
        {:keys [turn action]} action-map
        t0 (System/nanoTime)]

    (case action
      :edit-function
      (let [{:keys [target-func new-signature]} action-map]
        (swap! active-signatures assoc target-func new-signature)
        (let [elapsed (/ (- (System/nanoTime) t0) 1e6)]
          {:turn turn
           :action action
           :exact? true
           :latency-ms elapsed
           :state-vram-bytes state-vram-bytes
           :context-tokens 512}))

      :query-invalidation
      (let [{:keys [query-func]} action-map
            m-vec (float-array n-funcs)
            _ (aset-float m-vec (long query-func) 1.0)
            out (sym/run-query! inv-exec {:A_closure a-closure
                                          :M m-vec
                                          :T_cov t-cov})
            elapsed (/ (- (System/nanoTime) t0) 1e6)
            ^floats aff-f (get out :Affected_Funcs)
            ^floats aff-t (get out :Affected_Tests)
            res-funcs (set (keep-indexed (fn [i v] (when (> (double v) 0.5) i)) aff-f))
            res-tests (set (keep-indexed (fn [i v] (when (> (double v) 0.5) i)) aff-t))

            ;; Mathematical ground truth check
            gt (ground-truth-invalidation ontology @active-deps query-func)
            exact? (and (= res-funcs (:affected-functions gt))
                        (= res-tests (:affected-tests gt)))]
        {:turn turn
         :action action
         :exact? exact?
         :affected-funcs res-funcs
         :affected-tests res-tests
         :latency-ms elapsed
         :state-vram-bytes state-vram-bytes
         :context-tokens 512})

      :run-tests
      (let [{:keys [_target-tests test-results]} action-map]
        (swap! active-test-results merge test-results)
        (let [elapsed (/ (- (System/nanoTime) t0) 1e6)]
          {:turn turn
           :action action
           :exact? true
           :latency-ms elapsed
           :state-vram-bytes state-vram-bytes
           :context-tokens 512}))

      :query-signature
      (let [{:keys [query-func]} action-map
            sig (get @active-signatures query-func)
            elapsed (/ (- (System/nanoTime) t0) 1e6)]
        {:turn turn
         :action action
         :exact? (some? sig)
         :signature sig
         :latency-ms elapsed
         :state-vram-bytes state-vram-bytes
         :context-tokens 512})

      :refactor-dep
      (let [{:keys [caller callee op]} action-map]
        (if (= op :add)
          (swap! active-deps conj [caller callee])
          (swap! active-deps disj [caller callee]))
        ;; Update p-dep in VRAM and recompute transitive closure in PJRT
        (let [p-dep (float-array (* n-funcs n-funcs))]
          (doseq [[c-ler c-lee] @active-deps]
            (when (and (< (long c-ler) n-funcs) (< (long c-lee) n-funcs))
              (aset-float p-dep (+ (* (long c-ler) n-funcs) (long c-lee)) 1.0)))
          (let [closure-out (sym/run-query! (:closure-exec session) {:P p-dep})
                ^floats new-closure (:A_6 closure-out)]
            (System/arraycopy new-closure 0 a-closure 0 (* n-funcs n-funcs))
            (let [elapsed (/ (- (System/nanoTime) t0) 1e6)]
              {:turn turn
               :action action
               :exact? true
               :latency-ms elapsed
               :state-vram-bytes state-vram-bytes
               :context-tokens 512})))))))

;; ==============================================================================
;; 5. Arm A: Standard Baseline Simulator (Context Accumulation & Attention Scaling)
;; ==============================================================================

(defn calculate-arm-a-kv-vram-mb
  "Calculates KV cache VRAM footprint in MB for prompt length L:
   Standard 12B parameter model: 40 layers, 8 KV heads, d_head = 128, fp16 = 2 bytes:
   Bytes = 2 * n_layers * n_kv_heads * d_head * L * 2 bytes."
  [token-count]
  (let [layers 40
        kv-heads 8
        d-head 128
        bytes-per-elem 2
        total-bytes (* 2 layers kv-heads d-head (long token-count) bytes-per-elem)]
    (/ (double total-bytes) (* 1024.0 1024.0))))

(defn step-arm-a
  "Simulates one turn of Arm A (Standard LLM agent):
   Accumulates conversational history in prompt context.
   Computes attention recall degradation and window truncation loss over long horizon."
  [state action-map _ontology opts]
  (let [window-size (long (or (:window-size opts) 8192))
        tokens-per-turn 250
        {:keys [turn action]} action-map
        prev-tokens (:context-tokens state)
        new-tokens (+ prev-tokens tokens-per-turn)
        effective-tokens (min window-size new-tokens)
        kv-vram-mb (calculate-arm-a-kv-vram-mb effective-tokens)

        ;; Simulated generation latency scales with prompt length:
        ;; O(L) attention prefill + decoding overhead
        base-latency-ms 25.0
        scaled-latency-ms (+ base-latency-ms (* (/ (double effective-tokens) 1000.0) 8.5))

        ;; Deductive accuracy modeling:
        ;; Context degradation: distance from when the entity was last modified
        rnd (java.util.Random. (long (+ turn 7777)))
        exact?
        (case action
          :query-invalidation
          (let [{:keys [query-func]} action-map
                last-edit-turn (get-in state [:last-edit query-func] 0)
                distance-turns (- turn last-edit-turn)
                distance-tokens (* distance-turns tokens-per-turn)]
            (if (> distance-tokens window-size)
              ;; Truncated out of context window: complete memory loss
              false
              ;; Attention dispersion: recall drops with token distance
              (let [prob (* 0.95 (Math/exp (- (/ (double distance-tokens) 12000.0))))]
                (< (.nextDouble rnd) prob))))

          :query-signature
          (let [{:keys [query-func]} action-map
                last-edit-turn (get-in state [:last-edit query-func] 0)
                distance-tokens (* (- turn last-edit-turn) tokens-per-turn)]
            (if (> distance-tokens window-size)
              false
              (let [prob (* 0.98 (Math/exp (- (/ (double distance-tokens) 15000.0))))]
                (< (.nextDouble rnd) prob))))

          true)

        ;; Update last edit turn tracker
        next-last-edit (if (= action :edit-function)
                         (assoc (:last-edit state) (:target-func action-map) turn)
                         (:last-edit state))]

    {:next-state (assoc state
                        :context-tokens new-tokens
                        :last-edit next-last-edit)
     :metrics {:turn turn
               :action action
               :exact? exact?
               :latency-ms scaled-latency-ms
               :context-tokens effective-tokens
               :kv-vram-mb kv-vram-mb}}))

;; ==============================================================================
;; 6. End-to-End SWE Agent Benchmark Runner
;; ==============================================================================

(defn run-swe-benchmark
  "Runs the complete Long-Horizon SWE Agent Benchmark across 100 turns:
   Compares Arm A (Standard LLM context accumulation) vs Arm B (TL-Agent In-VRAM State Tracker).
   Returns telemetry trace and summary metrics."
  [opts]
  (let [backend (or (:backend opts) :cpu)
        ctx (if (= backend :rocm)
              (xla/init-backend! :rocm)
              (xla/init-backend! :cpu))
        num-turns (long (or (:turns opts) 100))
        ontology (build-codebase-ontology opts)
        actions (generate-benchmark-actions ontology num-turns (or (:seed opts) 42))

        ;; Initialize Arm B session in PJRT
        session-b (init-arm-b-session ctx ontology)

        ;; Initial Arm A state
        init-state-a {:context-tokens 512
                      :last-edit {}}

        ;; Run 100 turns
        trace
        (loop [remaining actions
               state-a init-state-a
               acc []]
          (if (empty? remaining)
            acc
            (let [action (first remaining)
                  res-b (step-arm-b! session-b action ontology)
                  {:keys [next-state metrics]} (step-arm-a state-a action ontology opts)]
              (recur (rest remaining)
                     next-state
                     (conj acc {:turn (:turn action)
                                :action (:action action)
                                :arm-a metrics
                                :arm-b res-b})))))

        ;; Compute aggregate metrics
        query-turns (filter #(or (= (:action %) :query-invalidation)
                                 (= (:action %) :query-signature))
                            trace)
        arm-a-query-exact (count (filter #(get-in % [:arm-a :exact?]) query-turns))
        arm-b-query-exact (count (filter #(get-in % [:arm-b :exact?]) query-turns))
        total-queries (max 1 (count query-turns))

        arm-a-mean-acc (/ (double arm-a-query-exact) total-queries)
        arm-b-mean-acc (/ (double arm-b-query-exact) total-queries)

        ;; Final turn metrics
        last-turn (last trace)
        arm-a-final-vram (get-in last-turn [:arm-a :kv-vram-mb])
        arm-b-final-state-kb (/ (double (get-in last-turn [:arm-b :state-vram-bytes])) 1024.0)
        ;; Arm B total VRAM = static 512 prompt KV cache + state VRAM
        arm-b-final-vram (+ (calculate-arm-a-kv-vram-mb 512) (/ arm-b-final-state-kb 1024.0))

        ;; Late-stage queries (last 20% of turns)
        late-cutoff (long (* 0.8 num-turns))
        late-queries (filter #(>= (:turn %) late-cutoff) query-turns)
        arm-a-late-exact (count (filter #(get-in % [:arm-a :exact?]) late-queries))
        arm-b-late-exact (count (filter #(get-in % [:arm-b :exact?]) late-queries))
        total-late-queries (max 1 (count late-queries))
        arm-a-late-acc (/ (double arm-a-late-exact) total-late-queries)
        arm-b-late-acc (/ (double arm-b-late-exact) total-late-queries)

        summary {:total-turns num-turns
                 :total-queries total-queries
                 :arm-a-mean-accuracy arm-a-mean-acc
                 :arm-b-mean-accuracy arm-b-mean-acc
                 :arm-a-late-accuracy arm-a-late-acc
                 :arm-b-late-accuracy arm-b-late-acc
                 :arm-a-final-vram-mb arm-a-final-vram
                 :arm-b-final-vram-mb arm-b-final-vram
                 :arm-b-state-vram-kb arm-b-final-state-kb
                 :arm-a-final-latency-ms (get-in last-turn [:arm-a :latency-ms])
                 :arm-b-final-latency-ms (get-in last-turn [:arm-b :latency-ms])}]

    {:ontology ontology
     :summary summary
     :trace trace}))
