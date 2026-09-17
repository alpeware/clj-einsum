(ns clj-xla.logic.symbolic
  "Embedding-Space Symbolic Reasoning for Tensor Logic (Section 5 of Pedro Domingos, arXiv:2510.12269).
   Implements relation embedding via superposed tensor products, rule embedding via contraction joins,
   fact decoding, tempered analogical queries, and host-driven forward-chaining fixpoints."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.ast :as ast]
            [clj-xla.logic.core :as logic]
            [clj-xla.logic.dce :as dce]
            [clj-xla.pjrt :as pjrt]
            [clojure.set :as set]
            [clojure.string :as str]))

;; ==============================================================================
;; 3.1 Data Preparation (Pure, Host-Side, Sans-IO)
;; ==============================================================================

(defn fact-tensor
  "Encodes a set or sequence of fact index tuples into a dense indicator float tensor.
   `facts`: set of [x y] (binary) or [x y z ...] (n-ary).
   `n`: integer total object count (domain size).
   Returns {:shape [n ... n] :data float-array} with 1.0f at true fact coordinates.
   Validates that indices are within [0, n-1]. Silently deduplicates duplicate facts."
  [facts n]
  (let [n-long (long n)
        facts-seq (vec (into #{} facts))
        arity (if (seq facts-seq) (count (first facts-seq)) 2)
        shape (vec (repeat arity n-long))
        total-size (reduce * 1 shape)
        data (float-array total-size)]
    (doseq [tuple facts-seq]
      (when-not (= (count tuple) arity)
        (throw (ex-info "Arity mismatch in fact tuple" {:tuple tuple :expected-arity arity})))
      (let [idx (loop [dims tuple
                       stride-shape (vec (rest shape))
                       offset 0]
                  (if (empty? dims)
                    offset
                    (let [coord (long (first dims))]
                      (when-not (and (>= coord 0) (< coord n-long))
                        (throw (ex-info "Fact coordinate out of bounds" {:coord coord :n n-long :tuple tuple})))
                      (let [stride (reduce * 1 stride-shape)]
                        (recur (rest dims) (if (seq stride-shape) (vec (rest stride-shape)) []) (+ offset (* coord stride)))))))]
        (aset-float data (int idx) (float 1.0))))
    {:shape shape :data data}))

(defn random-embeddings
  "Generates an N x D embedding matrix where each row is a unit vector sampled
   independently from a standard normal distribution and normalized to Euclidean length 1.0.
   Returns {:shape [n d] :data float-array}. Deterministic for a given seed."
  [n d seed]
  (let [n-long (long n)
        d-long (long d)
        rnd (java.util.Random. (long seed))
        total (* n-long d-long)
        data (float-array total)]
    (dotimes [i n-long]
      (let [row-offset (* i d-long)]
        ;; 1. Sample standard normals
        (dotimes [j d-long]
          (aset-float data (int (+ row-offset j)) (float (.nextGaussian rnd))))
        ;; 2. Compute L2 norm
        (let [norm (loop [j 0 sum 0.0]
                     (if (>= j d-long)
                       (Math/sqrt sum)
                       (let [v (double (aget data (int (+ row-offset j))))]
                         (recur (inc j) (+ sum (* v v))))))]
          ;; 3. Normalize row to unit vector
          (when (pos? norm)
            (let [inv-norm (float (/ 1.0 norm))]
              (dotimes [j d-long]
                (let [idx (int (+ row-offset j))]
                  (aset-float data idx (* (aget data idx) inv-norm)))))))))
    {:shape [n-long d-long] :data data}))

(defn correlate-embeddings
  "Creates a copy of embedding matrix `emb` where entity row `j` is replaced by
   normalize(rho * row_i + sqrt(1 - rho^2) * fresh_gaussian).
   Used for analogical-regime reasoning (Section 5: similarity-weighted inference).
   Validates 0.0 <= rho <= 1.0."
  [emb i j rho]
  (let [rho-d (double rho)]
    (when-not (and (>= rho-d 0.0) (<= rho-d 1.0))
      (throw (ex-info "Correlation coefficient rho must be in [0.0, 1.0]"
                      {:type :invalid-correlation :rho rho})))
    (let [{:keys [shape data]} emb
          [_n d] shape
          i-long (long i)
          j-long (long j)
          d-long (long d)
          ^floats orig-data data
          new-data (float-array (alength orig-data))
          _ (System/arraycopy orig-data 0 new-data 0 (alength orig-data))
          rnd (java.util.Random. (long (+ (* i-long 31) j-long 7)))
          i-offset (* i-long d-long)
          j-offset (* j-long d-long)
          sqrt-term (Math/sqrt (- 1.0 (* rho-d rho-d)))]
      ;; Generate blended vector for row j
      (dotimes [k d-long]
        (let [vi (double (aget orig-data (int (+ i-offset k))))
              noise (.nextGaussian rnd)
              blended (+ (* rho-d vi) (* sqrt-term noise))]
          (aset-float new-data (int (+ j-offset k)) (float blended))))
      ;; Normalize row j to unit length
      (let [norm (loop [k 0 sum 0.0]
                   (if (>= k d-long)
                     (Math/sqrt sum)
                     (let [v (double (aget new-data (int (+ j-offset k))))]
                       (recur (inc k) (+ sum (* v v))))))]
        (when (pos? norm)
          (let [inv-norm (float (/ 1.0 norm))]
            (dotimes [k d-long]
              (let [idx (int (+ j-offset k))]
                (aset-float new-data idx (* (aget new-data idx) inv-norm)))))))
      {:shape shape :data new-data})))

(defn decode-pairs
  "Decodes a flat N x N score array into a set of entity pairs with score >= threshold.
   `scores`: flat float-array of length N*N.
   `n`: integer entity count.
   `threshold`: float cut-off (default 0.5, per Section 5).
   `objects`: optional vector of entity identifiers (e.g. [:alice :bob ...]).
   Returns #{[obj-a obj-b]}."
  ([scores n]
   (decode-pairs scores n 0.5 nil))
  ([scores n threshold]
   (decode-pairs scores n threshold nil))
  ([scores n threshold objects]
   (let [n-long (long n)
         thresh (float (or threshold 0.5))
         ^floats arr (if (instance? (Class/forName "[F") scores)
                       scores
                       (float-array scores))
         res (transient #{})]
     (dotimes [i n-long]
       (dotimes [j n-long]
         (let [idx (+ (* i n-long) j)
               score (aget arr (int idx))]
           (when (>= score thresh)
             (let [obj-a (if objects (nth objects i) i)
                   obj-b (if objects (nth objects j) j)]
               (conj! res [obj-a obj-b]))))))
     (persistent! res))))

;; ==============================================================================
;; 3.2 Pure AST Constructors
;; ==============================================================================

(defn embed-relation
  "Constructs Tensor Logic equation for embedding relation `rel-name` using `emb-name`:
   EmbR[i, j] = sum_{x,y} R[x, y] * E[x, i] * E[y, j].
   Arities:
   [rel-emb-name rel-name emb-name] -> default binary [:x :y] [:i :j]
   [rel-emb-name rel-name emb-name obj-idxs emb-idxs]"
  ([rel-emb-name rel-name emb-name]
   (embed-relation rel-emb-name rel-name emb-name [:x :y] [:i :j]))
  ([rel-emb-name rel-name emb-name obj-idxs emb-idxs]
   (let [emb-terms (mapv (fn [obj-idx emb-idx] [emb-name obj-idx emb-idx]) obj-idxs emb-idxs)
         body (into [(into [rel-name] obj-idxs)] emb-terms)
         node (into [:= (into [rel-emb-name] emb-idxs)] body)]
     (assert (ast/valid-node? node) (str "Invalid AST node in embed-relation: " node))
     node)))

(defn query-relation
  "Constructs Tensor Logic equation for querying an embedded relation:
   Dq[a, b] = sum_{i,j} EmbR[i, j] * E[a, i] * E[b, j].
   Arities:
   [q-name rel-emb-name emb-name] -> default binary [:a :b] [:i :j]
   [q-name rel-emb-name emb-name query-idxs emb-idxs]
   [q-name rel-emb-name emb-name query-idxs emb-idxs attrs]"
  ([q-name rel-emb-name emb-name]
   (query-relation q-name rel-emb-name emb-name [:a :b] [:i :j]))
  ([q-name rel-emb-name emb-name query-idxs emb-idxs]
   (query-relation q-name rel-emb-name emb-name query-idxs emb-idxs nil))
  ([q-name rel-emb-name emb-name query-idxs emb-idxs attrs]
   (let [emb-terms (mapv (fn [q-idx emb-idx] [emb-name q-idx emb-idx]) query-idxs emb-idxs)
         body (into [(into [rel-emb-name] emb-idxs)] emb-terms)
         node (cond-> [:= (into [q-name] query-idxs)]
                attrs (conj attrs)
                true (into body))]
     (assert (ast/valid-node? node) (str "Invalid AST node in query-relation: " node))
     node)))

(defn symbol->emb-head
  "Converts a logic rule head symbol/keyword to its deterministic embedded tensor name.
   E.g., :gp -> :GP, :grandparent -> :GRANDPARENT."
  [head-sym]
  (keyword (str/upper-case (name head-sym))))

(defn- se-index-collision?
  "Checks if a user-supplied logic variable matches the reserved internal :se<N> scheme."
  [v]
  (boolean (re-matches #"^se\d+$" (name v))))

(defn embed-rule
  "Embeds a Horn clause rule into a Tensor Logic contraction equation.
   `rule`: map {:head [:gp :x :z]
                :body [[:parent :x :y] [:parent :y :z]]
                :rel-embs {:parent :EmbR}}
   Emits: [:= [:GP :se0 :se1] [:EmbR :se0 :se2] [:EmbR :se2 :se1]].
   Throws ex-info on:
   - :index-collision: logic variable matches reserved :se<N> scheme.
   - :unbound-head-var: head variable not bound in any body atom.
   - :unknown-relation: body relation not found in :rel-embs."
  [{:keys [head body rel-embs] :as _rule}]
  (let [head-rel (first head)
        head-vars (vec (rest head))
        body-atoms body
        ;; 1. Check index collision
        all-vars (into (vec head-vars) (mapcat rest body-atoms))]
    (doseq [v all-vars]
      (when (se-index-collision? v)
        (throw (ex-info (str "User logic variable collisions with reserved :se<N> namespace: " v)
                        {:type :index-collision :var v}))))
    ;; 2. Check unbound head variables
    (let [body-var-set (set (mapcat rest body-atoms))]
      (doseq [hv head-vars]
        (when-not (contains? body-var-set hv)
          (throw (ex-info (str "Unbound head variable: " hv " in rule head " head)
                          {:type :unbound-head-var :var hv :head head})))))
    ;; 3. Deterministic embedding index allocation (:se0, :se1, :se2, ...)
    (let [var-seq (loop [vars all-vars
                         seen #{}
                         ordered []]
                    (if (empty? vars)
                      ordered
                      (let [v (first vars)]
                        (if (contains? seen v)
                          (recur (rest vars) seen ordered)
                          (recur (rest vars) (conj seen v) (conj ordered v))))))
          var->idx (into {} (map-indexed (fn [i v] [v (keyword (str "se" i))]) var-seq))
          ;; 4. Head tensor name
          head-name (symbol->emb-head head-rel)
          head-term (into [head-name] (mapv var->idx head-vars))
          ;; 5. Body terms
          body-terms (mapv (fn [atom]
                             (let [rel (first atom)
                                   args (rest atom)
                                   emb-rel (get rel-embs rel)]
                               (when-not emb-rel
                                 (throw (ex-info (str "Unknown relation in rule body: " rel)
                                                 {:type :unknown-relation :rel rel :rel-embs rel-embs})))
                               (into [emb-rel] (mapv var->idx args))))
                           body-atoms)
          node (into [:= head-term] body-terms)]
      (assert (ast/valid-node? node) (str "Invalid AST node in embed-rule: " node))
      node)))

(defn rule-program
  "Constructs a container block containing the embedded rule equations for `rules`.
   Multiple rules sharing the same head relation receive implicit sum accumulation,
   functioning as logical disjunction."
  ([rules]
   (rule-program rules {}))
  ([rules rel-embs]
   (let [rule-nodes (mapv (fn [r]
                            (embed-rule (assoc r :rel-embs (merge rel-embs (:rel-embs r)))))
                          rules)
         node (into [:block {:name :symbolic-program}] rule-nodes)]
     (assert (ast/valid-node? node) (str "Invalid AST node in rule-program: " node))
     node)))

(defn gram-matrix
  "Constructs Tensor Logic equation for computing the Gram matrix (entity similarity):
   S[a, b] = sum_d E[a, d] * E[b, d]."
  ([sim-name emb-name]
   (gram-matrix sim-name emb-name [:a :b] :d))
  ([sim-name emb-name [qa qb] emb-d]
   (let [node [:= [sim-name qa qb] [emb-name qa emb-d] [emb-name qb emb-d]]]
     (assert (ast/valid-node? node) (str "Invalid AST node in gram-matrix: " node))
     node)))

(defn tempered-query
  "Constructs a query-relation equation with a tempered sigmoid nonlinearity:
   sigma(x, T) = 1 / (1 + exp(-x / T)).
   Validates temperature > 0 (T=0 limit is the untempered crisp query)."
  [q-name rel-emb-name emb-name idxs temperature]
  (let [t (double temperature)]
    (when-not (pos? t)
      (throw (ex-info "Temperature must be strictly positive (> 0)"
                      {:type :invalid-temperature :temperature temperature})))
    (let [[qa qb] (take 2 idxs)
          [emb-i emb-j] (drop 2 idxs)]
      (query-relation q-name rel-emb-name emb-name [qa qb] [emb-i emb-j]
                      {:act :sigmoid :scale (/ 1.0 t)}))))

;; ==============================================================================
;; 3.3 Execution Boundary (Impure - PJRT Execution)
;; ==============================================================================

(defn compile-query
  "Compiles a Tensor Logic symbolic AST into an OpenXLA PJRT loaded executable."
  ([graph-name invars ast target-heads]
   (logic/compile-ast graph-name invars ast target-heads))
  ([ctx graph-name invars ast target-heads]
   (logic/compile-ast ctx graph-name invars ast target-heads)))

(defn run-query!
  "Executes compiled symbolic query `executable` with map of {invar-kw float-array}.
   Returns map of {target-head-kw float-array}.
   Explicitly destroys native device buffers before returning."
  [executable inputs]
  (let [ctx (or (:ctx executable) (xla/get-context))
        invars (or (get-in executable [:graph :invars]) [])
        outvars (or (get-in executable [:graph :outvars]) [])
        ephemeral-buffers (atom [])
        input-buffers
        (mapv (fn [[invar-name [_kw shape dtype]]]
                (let [data (get inputs invar-name)
                      _ (when-not data
                          (throw (ex-info "Missing input data for invar"
                                          {:invar invar-name :available (keys inputs)})))]
                  (if (instance? java.lang.foreign.MemorySegment data)
                    data
                    (let [dtype-enum (case dtype :f32 11 :bf16 13 :i32 4 :i8 2 11)
                          buf (pjrt/buffer-from-host-buffer ctx (:client ctx) data shape dtype-enum)]
                      (swap! ephemeral-buffers conj buf)
                      buf))))
              invars)]
    (try
      (let [num-outs (count outvars)
            raw-out (pjrt/execute-executable ctx (or (:handle executable) executable) input-buffers num-outs)
            out-buffers (if (= num-outs 1) [raw-out] raw-out)
            results (into {}
                          (map-indexed
                           (fn [idx target-head]
                             (let [buf (nth out-buffers idx)
                                   sh (or (get-in executable [:graph :known-shapes target-head])
                                          [1])
                                   total-elems (reduce * 1 sh)
                                   floats (pjrt/buffer-to-host-buffer ctx buf (long total-elems) :f32)]
                               [target-head floats]))
                           outvars))]
        ;; Destroy device output buffers
        (doseq [b out-buffers]
          (try (xla/destroy-buffer! b) (catch Exception _ nil)))
        results)
      (finally
        ;; Destroy only ephemeral device input buffers
        (doseq [b @ephemeral-buffers]
          (try (xla/destroy-buffer! b) (catch Exception _ nil)))))))

;; ==============================================================================
;; 3.4 Forward-Chaining Driver (Host-Side, Phase 1)
;; ==============================================================================

(defn forward-chain
  "Executes host-driven per-relation forward-chaining fixpoint reasoning.
   Compiles execution graph once and iterates host-side fact updates until fixpoint or max-iters.
   `opts`: {:rules [...]
            :facts {rel-kw #{[0 1] ...} ...}
            :n 5
            :d 512
            :seed 42
            :max-iters 10 (optional, default 10)
            :threshold 0.5 (optional, default 0.5)
            :objects [:alice :bob ...] (optional)}
   Returns {:facts {rel-kw #{...}} :head-facts {head-kw #{...}} :iterations k :trace [{:iter k :new {rel-kw #{...}}}]}."
  [{:keys [rules facts n d seed max-iters threshold objects] :as _opts}]
  (let [n-long (long n)
        d-long (long d)
        thresh (float (or threshold 0.5))
        max-it (long (or max-iters 10))]

    ;; 1. Validations: binary arity, duplicate heads, and valid facts map
    (when-not (map? facts)
      (throw (ex-info "Facts must be a map of {relation-kw #{[x y] ...}}"
                      {:type :invalid-facts :facts facts})))
    (doseq [r rules]
      (let [h (:head r)
            h-args (rest h)]
        (when-not (= (count h-args) 2)
          (throw (ex-info (str "Unsupported rule head arity (must be binary): " h)
                          {:type :unsupported-arity :head h}))))
      (doseq [atom (:body r)]
        (let [atom-args (rest atom)]
          (when-not (= (count atom-args) 2)
            (throw (ex-info (str "Unsupported rule body atom arity (must be binary): " atom)
                            {:type :unsupported-arity :atom atom}))))))
    (doseq [[rel tuples] facts
            tuple tuples]
      (when-not (= (count tuple) 2)
        (throw (ex-info (str "Unsupported fact tuple arity (must be binary): " tuple " in relation " rel)
                        {:type :unsupported-arity :rel rel :tuple tuple}))))
    (let [head-rels (mapv (comp first :head) rules)]
      (when-not (= (count head-rels) (count (set head-rels)))
        (throw (ex-info (str "Duplicate rule heads detected: " head-rels)
                        {:type :duplicate-rule-head :heads head-rels}))))

    ;; 2. Determine all distinct relations (EDB and IDB)
    (let [input-relations (set (keys facts))
          body-relations (set (mapcat (fn [r] (map first (:body r))) rules))
          head-relations (set (map (comp first :head) rules))
          all-relations (vec (distinct (concat input-relations body-relations head-relations)))
          rel->emb (into {} (map (fn [r] [r (keyword (str "Emb_" (name r)))]) all-relations))

          ;; Embed equations: one per relation
          embed-eqns (mapv (fn [r]
                             (let [emb-name (get rel->emb r)
                                   invar-name (keyword (str "R_" (name r)))]
                               (embed-relation emb-name invar-name :E [:x :y] [:i :j])))
                           all-relations)

          ;; Rule nodes with rel-embs covering every relation
          rule-nodes (mapv (fn [r]
                             (embed-rule (assoc r :rel-embs rel->emb)))
                           rules)

          ;; Query nodes: one per rule head
          query-nodes (mapv (fn [r]
                              (let [head-rel (first (:head r))
                                    q-name (keyword (str "Dq_" (name head-rel)))
                                    h-emb (symbol->emb-head head-rel)]
                                (query-relation q-name h-emb :E [:a :b] [:se0 :se1])))
                            rules)

          target-heads (mapv (fn [r] (keyword (str "Dq_" (name (first (:head r)))))) rules)
          invars (into [[:E [:tensor [n-long d-long] :f32]]]
                       (map (fn [r] [(keyword (str "R_" (name r))) [:tensor [n-long n-long] :f32]])
                            all-relations))

          chain-ast (vec (concat
                          [:block {:name :forward-chain-step}]
                          embed-eqns
                          rule-nodes
                          query-nodes))

          exec (compile-query "forward_chain_graph" invars chain-ast target-heads)
          emb (random-embeddings n-long d-long (or seed 42))
          e-data (:data emb)

          obj->idx (when objects (into {} (map-indexed (fn [i o] [o i]) objects)))
          init-facts (into {}
                           (map (fn [r]
                                  [r (set (map (fn [tuple]
                                                 (if (and obj->idx (not (number? (first tuple))))
                                                   (mapv obj->idx tuple)
                                                   (mapv long tuple)))
                                               (get facts r #{})))])
                                all-relations))]

      ;; 3. Host-side fixpoint loop
      (loop [curr-facts init-facts
             iter 1
             trace []]
        (let [inputs (into {:E e-data}
                           (map (fn [r]
                                  [(keyword (str "R_" (name r)))
                                   (:data (fact-tensor (get curr-facts r #{}) n-long))])
                                all-relations))
              outputs (run-query! exec inputs)

              ;; Decode newly inferred facts from all head queries (as integer pairs)
              new-facts-by-head (into {}
                                      (map (fn [r]
                                             (let [head-rel (first (:head r))
                                                   q-name (keyword (str "Dq_" (name head-rel)))
                                                   scores (get outputs q-name)
                                                   pairs (decode-pairs scores n-long thresh nil)]
                                               [head-rel pairs]))
                                           rules))
              truly-new-by-head (into {}
                                      (map (fn [[head-rel pairs]]
                                             [head-rel (set/difference pairs (get curr-facts head-rel #{}))])
                                           new-facts-by-head))
              updated-facts (reduce (fn [acc [head-rel pairs]]
                                      (update acc head-rel (fnil set/union #{}) pairs))
                                    curr-facts
                                    new-facts-by-head)
              step-trace (conj trace {:iter iter
                                      :new truly-new-by-head})]
          (if (or (every? empty? (vals truly-new-by-head)) (>= iter max-it))
            (let [map-obj-pair (if objects
                                 (fn [[i j]] [(nth objects i) (nth objects j)])
                                 identity)
                  map-obj-set (fn [s] (into #{} (map map-obj-pair) s))
                  map-rel-map (fn [m] (into {} (map (fn [[k v]] [k (map-obj-set v)]) m)))]
              {:facts (map-rel-map updated-facts)
               :head-facts (into {} (map (fn [r]
                                           (let [hr (first (:head r))]
                                             [hr (map-obj-set (get updated-facts hr #{}))]))
                                         rules))
               :iterations iter
               :trace (mapv (fn [t]
                              (update t :new map-rel-map))
                            step-trace)})
            (recur updated-facts (inc iter) step-trace)))))))

;; ==============================================================================
;; 3.5 Backward Chaining Query Planner (Helper)
;; ==============================================================================

(defn query-plan
  "Plans a query over rule set `rules` for target head `query-head`.
   Applies backward-chaining dead code elimination (DCE) to prune unused intermediate rules."
  [rules rel-embs query-head]
  (let [prog (rule-program rules rel-embs)
        eqns (filterv ast/eqn? (rest (rest prog)))
        targets (if (set? query-head) query-head #{query-head})
        pruned-eqns (dce/prune-ast eqns targets)
        pruned-block (into [:block {:name :symbolic-program}] pruned-eqns)]
    {:targets targets
     :pruned-ast pruned-block}))

;; ==============================================================================
;; Pure Clojure Symbolic Deductive Oracle & Denoising Helper (for Verification)
;; ==============================================================================

(defn- match-atom [atom fact sub]
  (when (= (first atom) (first fact))
    (let [atom-args (rest atom)
          fact-args (rest fact)]
      (when (= (count atom-args) (count fact-args))
        (reduce (fn [s [var val]]
                  (if (nil? s)
                    nil
                    (if (contains? s var)
                      (if (= (get s var) val) s nil)
                      (assoc s var val))))
                sub
                (map vector atom-args fact-args))))))

(defn- join-body [body facts sub]
  (if (empty? body)
    [sub]
    (let [atom (first body)
          rem (rest body)]
      (mapcat (fn [fact]
                (if-let [new-sub (match-atom atom fact sub)]
                  (join-body rem facts new-sub)
                  []))
              facts))))

(defn symbolic-oracle-forward-chain
  "Pure Clojure reference implementation (oracle) of deductive closure.
   Computes the fixpoint of Horn clause rules over a set of relational facts.
   `rules`: seq of {:head [:rel-name & vars] :body [[:rel-name & vars] ...]}
   `facts`: set of [:rel-name & args] (e.g. #{[:parent 0 1] [:parent 1 2]})."
  [rules facts]
  (loop [curr-facts (set facts)]
    (let [new-facts (into curr-facts
                          (for [rule rules
                                sub (join-body (:body rule) curr-facts {})]
                            (into [(first (:head rule))]
                                  (mapv #(get sub %) (rest (:head rule))))))]
      (if (= curr-facts new-facts)
        curr-facts
        (recur new-facts)))))

(defn denoise-scores
  "Denoises raw continuous query scores by thresholding at `threshold` and returning
   the binary indicator float array."
  [scores n threshold]
  (let [n-long (long n)
        thresh (float (or threshold 0.5))
        total (* n-long n-long)
        ^floats src scores
        dst (float-array total)]
    (dotimes [i total]
      (aset-float dst i (if (>= (aget src i) thresh) (float 1.0) (float 0.0))))
    dst))

;; ==============================================================================
;; 4. Scalability Improvements (Shah & Zadrozny, arXiv:2601.17188v1)
;; ==============================================================================

;; 4.1 Direct Symbolic Datalog via Tensor Contraction (Paper Exp 1, §3.1 & §5.1)

(defn datalog-transitive-step-ast
  "Constructs a Tensor Logic AST block for one iterative step of transitive closure:
   A_next = clamp(A + A x P, 0.0, 1.0).
   Pure boolean/indicator contraction without embedding noise (Shah & Zadrozny, arXiv:2601.17188v1 Exp 1, §3.1 & §5.1)."
  ([a-name p-name next-name]
   (datalog-transitive-step-ast a-name p-name next-name [:x :z] :y))
  ([a-name p-name next-name [x z] y]
   (let [unclamped-head (keyword (str (name next-name) "_unclamped"))]
     [:block {:name :datalog-transitive-step}
      [:= [unclamped-head x z] [a-name x z]]
      [:= [unclamped-head x z] [a-name x y] [p-name y z]]
      [:= [next-name x z] {:clamp [0.0 1.0]} [unclamped-head x z]]])))

(defn containment?
  "Verification Check 1 (Paper §3.1.4): P subset-of A.
   All direct parent edges must appear in the transitive closure.
   Accepts sets of index pairs #{[x y] ...} or result map."
  ([p a]
   (let [p-set (if (map? p) (:facts p) (set p))
         a-set (if (map? a) (:facts a) (set a))]
     (set/subset? p-set a-set)))
  ([result]
   (containment? (:p-facts result) (:facts result))))

(defn closure?
  "Verification Check 2 (Paper §3.1.4): A x P adds no new edges to A.
   One additional step of chaining does not discover any new relationships.
   Accepts sets of index pairs #{[x y] ...} or result map."
  ([p a]
   (let [p-set (if (map? p) (:facts p) (set p))
         a-set (if (map? a) (:facts a) (set a))
         p-by-y (reduce (fn [m [y z]] (update m y (fnil conj #{}) z)) {} p-set)]
     (every? (fn [[x y]]
               (let [zs (get p-by-y y #{})]
                 (every? (fn [z] (contains? a-set [x z])) zs)))
             a-set)))
  ([p a _n]
   (closure? p a))
  ([result]
   (closure? (:p-facts result) (:facts result))))

(defn acyclic?
  "Verification Check 3 (Paper §3.1.4): diag(A) = 0.
   No individual is their own ancestor.
   Accepts sets of index pairs #{[x y] ...} or result map."
  ([a]
   (let [a-set (if (map? a) (:facts a) (set a))]
     (every? (fn [[x y]] (not= x y)) a-set))))

(defn lineage-stats
  "Computes ancestor and descendant counts for an individual (Paper §4.1.3).
   `a-facts`: set of [ancestor descendant] pairs.
   `person`: individual identifier.
   Returns {:ancestors int :descendants int}."
  [a-facts person]
  (let [facts-set (if (map? a-facts) (:facts a-facts) (set a-facts))
        ancestors (count (filter (fn [[_x z]] (= z person)) facts-set))
        descendants (count (filter (fn [[x _z]] (= x person)) facts-set))]
    {:ancestors ancestors :descendants descendants}))

(defn datalog-transitive-closure
  "Computes transitive closure via direct Boolean/indicator tensor contraction in OpenXLA PJRT
   (Shah & Zadrozny, arXiv:2601.17188v1 Exp 1, §3.1 & §5.1).
   `opts`: {:facts #{[0 1] ...}
            :n integer entity count
            :max-iters integer (default 100)
            :objects optional vector of entity labels}
   Returns {:facts #{...} :iterations int :converged? bool :matrix float-array :n int :p-facts #{...}}."
  [{:keys [facts n max-iters objects]}]
  (let [n-long (long n)
        max-it (long (or max-iters 100))
        facts-set (set facts)
        p-tensor (fact-tensor facts-set n-long)
        p-data (:data p-tensor)
        step-ast (datalog-transitive-step-ast :A :P :A_next)
        invars [[:A [:tensor [n-long n-long] :f32]]
                [:P [:tensor [n-long n-long] :f32]]]
        exec (compile-query "datalog_tc_step" invars step-ast [:A_next])]
    (loop [a-curr p-data
           iter 1]
      (let [out (run-query! exec {:A a-curr :P p-data})
            ^floats a-next (get out :A_next)
            ^floats a-prev a-curr
            total (* n-long n-long)
            changed? (loop [idx 0]
                       (if (>= idx total)
                         false
                         (if (and (> (aget a-next idx) 0.5)
                                  (<= (aget a-prev idx) 0.5))
                           true
                           (recur (inc idx)))))]
        (if (or (not changed?) (>= iter max-it))
          (let [decoded (decode-pairs a-next n-long 0.5 objects)]
            {:facts decoded
             :p-facts (if objects
                        (set (map (fn [[i j]] [(nth objects i) (nth objects j)]) facts-set))
                        facts-set)
             :iterations iter
             :converged? (not changed?)
             :matrix a-next
             :n n-long})
          (recur a-next (inc iter)))))))

;; 4.2 Compact Transformation Matrices & Composition (Paper Exp 2, §3.2)

(defn compose-matrices-ast
  "Constructs Tensor Logic equation for composing two relation transformation matrices:
   M_comp[i, j] = sum_k M1[i, k] * M2[k, j] (Paper Exp 2, §3.2 & Exp 3, §3.3.6).
   Arities:
   [comp-name m1-name m2-name] -> default [:i :j] :k
   [comp-name m1-name m2-name [i j] k]"
  ([comp-name m1-name m2-name]
   (compose-matrices-ast comp-name m1-name m2-name [:i :j] :k))
  ([comp-name m1-name m2-name [i j] k]
   (let [node [:= [comp-name i j] [m1-name i k] [m2-name k j]]]
     (assert (ast/valid-node? node) (str "Invalid AST in compose-matrices-ast: " node))
     node)))

(defn transform-entity-ast
  "Constructs Tensor Logic equation for mapping an entity vector through a transformation matrix:
   V_out[d] = sum_i V_in[i] * M[i, d] (Paper Exp 2, §3.2).
   Arities:
   [out-name in-name m-name] -> default :d :i
   [out-name in-name m-name d-idx in-idx]"
  ([out-name in-name m-name]
   (transform-entity-ast out-name in-name m-name :d :i))
  ([out-name in-name m-name d-idx in-idx]
   (let [node [:= [out-name d-idx] [in-name in-idx] [m-name in-idx d-idx]]]
     (assert (ast/valid-node? node) (str "Invalid AST in transform-entity-ast: " node))
     node)))

(defn score-candidates-ast
  "Constructs Tensor Logic equation for scoring candidate entities via dot product with temperature:
   Scores[n] = (1 / T) * sum_d V_pred[d] * E[n, d] (Paper Exp 2, §3.2).
   Arities:
   [scores-name v-name e-name temperature] -> default :n :d
   [scores-name v-name e-name n-idx d-idx temperature]"
  ([scores-name v-name e-name temperature]
   (score-candidates-ast scores-name v-name e-name :n :d temperature))
  ([scores-name v-name e-name n-idx d-idx temperature]
   (let [t (double (or temperature 1.0))
         node [:= [scores-name n-idx] {:scale (/ 1.0 t)} [v-name d-idx] [e-name n-idx d-idx]]]
     (assert (ast/valid-node? node) (str "Invalid AST in score-candidates-ast: " node))
     node)))

(defn predict-compositional-query
  "Executes zero-shot multi-hop compositional query via chained relation matrices in OpenXLA PJRT
   (Shah & Zadrozny, arXiv:2601.17188v1 Exp 2, §3.2).
   `opts`: {:subject-idx int
            :relation-matrices [float-array ...]
            :entity-embeddings {:shape [n d] :data float-array}
            :d int
            :temperature double (optional, default 1.0)}
   Returns {:scores float-array :top-entity-idx int :sorted-entities [int ...]}."
  [{:keys [subject-idx relation-matrices entity-embeddings d temperature]}]
  (let [d-long (long d)
        {:keys [shape data]} entity-embeddings
        [n _] shape
        n-long (long n)
        temp (double (or temperature 1.0))
        ^floats e-arr data
        s-long (long subject-idx)
        s-vec (float-array d-long)]
    (System/arraycopy e-arr (int (* s-long d-long)) s-vec 0 (int d-long))
    (let [num-m (count relation-matrices)
          m-vars (mapv #(keyword (str "M" %)) (range num-m))
          comp-ast
          (cond
            (= num-m 1)
            [:block {:name :comp-pred}
             (transform-entity-ast :Vpred :Vs (first m-vars))
             (score-candidates-ast :Scores :Vpred :E temp)]

            (= num-m 2)
            [:block {:name :comp-pred}
             (compose-matrices-ast :Mcomp (first m-vars) (second m-vars))
             (transform-entity-ast :Vpred :Vs :Mcomp)
             (score-candidates-ast :Scores :Vpred :E temp)]

            :else
            (let [steps (mapv (fn [idx]
                                (let [in-m (if (zero? idx) (first m-vars) (keyword (str "Mcomp" idx)))
                                      next-m (nth m-vars (inc idx))
                                      out-m (if (= idx (- num-m 2)) :Mcomp (keyword (str "Mcomp" (inc idx))))]
                                  (compose-matrices-ast out-m in-m next-m)))
                              (range (dec num-m)))]
              (vec (concat [:block {:name :comp-pred}]
                           steps
                           [(transform-entity-ast :Vpred :Vs :Mcomp)
                            (score-candidates-ast :Scores :Vpred :E temp)]))))
          invars (into [[:Vs [:tensor [d-long] :f32]]
                        [:E [:tensor [n-long d-long] :f32]]]
                       (map (fn [v] [v [:tensor [d-long d-long] :f32]]) m-vars))
          exec (compile-query "comp_pred_graph" invars comp-ast [:Scores])
          inputs (into {:Vs s-vec :E e-arr}
                       (map vector m-vars relation-matrices))
          out (run-query! exec inputs)
          ^floats scores (get out :Scores)
          sorted-entities (vec (sort-by (fn [idx] (- (aget scores idx))) (range n-long)))]
      {:scores scores
       :top-entity-idx (first sorted-entities)
       :sorted-entities sorted-entities})))

;; 4.3 Superposition Construction & Filtered Ranking Evaluation (Paper Exp 3, §3.3)

(defn superposition-relation-matrix
  "Computes relation transformation matrix via Tensor Logic superposition construction:
   R_r = E^T * A_r * E = sum_{(h, t) in facts(r)} e_h (x) e_t (Shah & Zadrozny, arXiv:2601.17188v1 Exp 3, §3.3.2; Domingos §5).
   Compiled through OpenXLA PJRT into optimized StableHLO dot_general contractions (Rule 4).
   `entity-embeddings`: {:shape [n d] :data float-array}.
   `facts`: sequence or set of [h t] integer entity index pairs.
   Returns {:shape [d d] :data float-array}."
  ([entity-embeddings facts]
   (superposition-relation-matrix (xla/get-context) entity-embeddings facts))
  ([ctx entity-embeddings facts]
   (let [{:keys [shape data]} entity-embeddings
         [n d] shape
         n-long (long n)
         d-long (long d)
         a-tensor (fact-tensor facts n-long)
         ast [:block {:name :superposition-matrix}
              (embed-relation :R :A :E [:h :t] [:i :j])]
         invars [[:A [:tensor [n-long n-long] :f32]]
                 [:E [:tensor [n-long d-long] :f32]]]
         exec (compile-query ctx "superposition_rel_matrix" invars ast [:R])
         out (run-query! exec {:A (:data a-tensor) :E data})]
     {:shape [d-long d-long] :data (get out :R)})))

(defn predict-tail-query
  "Evaluates tail prediction query (h, r, ?) via OpenXLA PJRT (Paper Exp 3, §3.3.3):
   v_pred = e_h * R_r, scores = (v_pred * E^T) / T.
   `opts`: {:head-idx int
            :relation-matrix {:shape [d d] :data float-array}
            :entity-embeddings {:shape [n d] :data float-array}
            :temperature double (optional, default 0.1)}
   Returns {:scores float-array :top-entity-idx int :sorted-entities [int ...]}."
  [{:keys [head-idx relation-matrix entity-embeddings temperature]}]
  (let [{:keys [shape data]} entity-embeddings
        [n d] shape
        n-long (long n)
        d-long (long d)
        temp (double (or temperature 0.1))
        h-long (long head-idx)
        ^floats e-arr data
        eh-vec (float-array d-long)]
    (System/arraycopy e-arr (int (* h-long d-long)) eh-vec 0 (int d-long))
    (let [ast [:block {:name :tail-pred}
               [:= [:Vpred :j] [:Eh :i] [:R :i :j]]
               [:= [:Scores :n] {:scale (/ 1.0 temp)} [:Vpred :j] [:E :n :j]]]
          invars [[:Eh [:tensor [d-long] :f32]]
                  [:R [:tensor [d-long d-long] :f32]]
                  [:E [:tensor [n-long d-long] :f32]]]
          exec (compile-query "tail_pred_graph" invars ast [:Scores])
          out (run-query! exec {:Eh eh-vec :R (:data relation-matrix) :E e-arr})
          ^floats scores (get out :Scores)
          sorted-entities (vec (sort-by (fn [idx] (- (aget scores idx))) (range n-long)))]
      {:scores scores
       :top-entity-idx (first sorted-entities)
       :sorted-entities sorted-entities})))

(defn predict-head-query
  "Evaluates head prediction query (?, r, t) via OpenXLA PJRT (Paper Exp 3, §3.3.3):
   v_pred = e_t * R_r^T, scores = (v_pred * E^T) / T.
   `opts`: {:tail-idx int
            :relation-matrix {:shape [d d] :data float-array}
            :entity-embeddings {:shape [n d] :data float-array}
            :temperature double (optional, default 0.1)}
   Returns {:scores float-array :top-entity-idx int :sorted-entities [int ...]}."
  [{:keys [tail-idx relation-matrix entity-embeddings temperature]}]
  (let [{:keys [shape data]} entity-embeddings
        [n d] shape
        n-long (long n)
        d-long (long d)
        temp (double (or temperature 0.1))
        t-long (long tail-idx)
        ^floats e-arr data
        et-vec (float-array d-long)]
    (System/arraycopy e-arr (int (* t-long d-long)) et-vec 0 (int d-long))
    (let [ast [:block {:name :head-pred}
               [:= [:Vpred :i] [:Et :j] [:R :i :j]]
               [:= [:Scores :n] {:scale (/ 1.0 temp)} [:Vpred :i] [:E :n :i]]]
          invars [[:Et [:tensor [d-long] :f32]]
                  [:R [:tensor [d-long d-long] :f32]]
                  [:E [:tensor [n-long d-long] :f32]]]
          exec (compile-query "head_pred_graph" invars ast [:Scores])
          out (run-query! exec {:Et et-vec :R (:data relation-matrix) :E e-arr})
          ^floats scores (get out :Scores)
          sorted-entities (vec (sort-by (fn [idx] (- (aget scores idx))) (range n-long)))]
      {:scores scores
       :top-entity-idx (first sorted-entities)
       :sorted-entities sorted-entities})))

(defn evaluate-filtered-ranking
  "Evaluates ranking metrics under canonical filtered ranking protocol (Paper Exp 3, §3.3.5 & §3.3.6).
   For each test query, masks out all other known true entities for the query (h, r, ?) so they
   do not artificially penalize the target entity's rank.
   `queries`: seq of {:head h :rel r :tail t :scores float-array}
   `all-known-triples`: set of [h r t] known true facts across all splits.
   Returns {:mrr double :hits@1 double :hits@3 double :hits@10 double :count int}."
  [queries all-known-triples]
  (let [triples-set (set all-known-triples)
        q-list (vec queries)
        num-q (count q-list)]
    (if (zero? num-q)
      {:mrr 0.0 :hits-1 0.0 :hits-3 0.0 :hits-10 0.0 (keyword "hits@1") 0.0 (keyword "hits@3") 0.0 (keyword "hits@10") 0.0 :count 0}
      (let [ranks
            (mapv (fn [{:keys [head rel tail scores]}]
                    (let [h head
                          r rel
                          target-t tail
                          ^floats arr (if (instance? (Class/forName "[F") scores)
                                        scores
                                        (float-array scores))
                          target-score (aget arr (int target-t))
                          num-entities (alength arr)]
                      (loop [e 0
                             better-count 0]
                        (if (>= e num-entities)
                          (inc better-count)
                          (if (or (= e target-t)
                                  (contains? triples-set [h r e]))
                            (recur (inc e) better-count)
                            (let [s (aget arr (int e))]
                              (if (> s target-score)
                                (recur (inc e) (inc better-count))
                                (recur (inc e) better-count))))))))
                  q-list)
            mrr (/ (reduce + 0.0 (map (fn [^long r] (/ 1.0 (double r))) ranks)) (double num-q))
            hits-1 (/ (double (count (filter #(= % 1) ranks))) (double num-q))
            hits-3 (/ (double (count (filter #(<= % 3) ranks))) (double num-q))
            hits-10 (/ (double (count (filter #(<= % 10) ranks))) (double num-q))]
        {:mrr mrr
         :hits-1 hits-1
         :hits-3 hits-3
         :hits-10 hits-10
         (keyword "hits@1") hits-1
         (keyword "hits@3") hits-3
         (keyword "hits@10") hits-10
         :count num-q}))))
