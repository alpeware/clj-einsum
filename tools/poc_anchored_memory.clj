(ns tools.poc-anchored-memory
  "QR-orthonormalized LLM-anchored entity embeddings (disentangling cross-talk from distribution shift).
   Evaluates whether grounding Pedro Domingos' symbolic relational memory cores directly
   in Gemma 4's tied token embedding space achieves zero-shot retrieval with W = I (identity).
   Compares:
     Arm 1 (Control):   Random entity table @ D=1536, W = I (isolates D=1536 effect)
     Arm 2 (Re-run):    LLM-anchored raw @ D=1536, W = I (replicates Task C 0/7)
     Arm 4 (Test):      LLM-anchored QR-orthonormalized @ D=1536, W = I (surgically eliminates cross-talk)
     Arm 3 (Reference): De-oracled baseline @ D=256, W = random (0/7 chance)"
  (:require [einsum.core :as xla]
            [einsum.logic.memory.relation :as mem]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.protocol :refer [bos-id encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [tools.gemma4-inference :as g4]
            [tools.poc-fact-grounding :as poc-base]
            [tools.train-mem-projection :as train-base])
  (:import [java.lang.foreign MemorySegment])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :model ".models/gemma-4-E2B-it"
   :triples "data/wiki_recent_triples.edn"
   :dim 1536
   :threshold 0.5
   :memory-seed 2026
   :entity-mode :anchored-qr
   :train-refinement false})

(defn- normalize-args [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT_OPTS]
    (if (empty? remaining)
      opts
      (let [k (first remaining)
            v (second remaining)]
        (case k
          "--backend" (recur (subvec remaining 2) (assoc opts :backend (keyword v)))
          "--model" (recur (subvec remaining 2) (assoc opts :model v))
          "--triples" (recur (subvec remaining 2) (assoc opts :triples v))
          "--dim" (recur (subvec remaining 2) (assoc opts :dim (Long/parseLong v)))
          "--threshold" (recur (subvec remaining 2) (assoc opts :threshold (Double/parseDouble v)))
          "--memory-seed" (recur (subvec remaining 2) (assoc opts :memory-seed (Long/parseLong v)))
          "--entity-mode" (recur (subvec remaining 2) (assoc opts :entity-mode (keyword v)))
          "--train-refinement" (recur (subvec remaining 2) (assoc opts :train-refinement (Boolean/parseBoolean v)))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; 1. Pure Linear Algebra & Table Construction Helpers
;; ==============================================================================

(defn build-identity-matrix
  "Returns a flat float array of shape [dim, dim] representing the identity matrix I."
  [dim]
  (let [d (long dim)
        arr (float-array (* d d))]
    (dotimes [i d]
      (aset-float arr (+ (* i d) i) (float 1.0)))
    arr))

(defn mean-pool-embeddings
  "Computes element-wise arithmetic mean across a list of equal-length float arrays."
  [vectors hidden-dim]
  (let [cnt (count vectors)
        h-dim (long hidden-dim)
        out (float-array h-dim)]
    (when (pos? cnt)
      (dotimes [i h-dim]
        (let [sum (loop [idx 0 s 0.0]
                    (if (>= idx cnt)
                      s
                      (recur (inc idx) (+ s (double (aget ^floats (nth vectors idx) i))))))]
          (aset-float out i (float (/ sum (double cnt)))))))
    out))

(defn l2-normalize
  "Normalizes float vector to Euclidean length 1.0. Zero vector remains zero."
  [^floats v]
  (let [len (alength v)
        sum-sq (loop [i 0 s 0.0]
                 (if (>= i len)
                   s
                   (let [x (double (aget v i))]
                     (recur (inc i) (+ s (* x x))))))
        norm (Math/sqrt sum-sq)
        scale (if (> norm 1e-12) (/ 1.0 norm) 0.0)
        out (float-array len)]
    (dotimes [i len]
      (aset-float out i (float (* (double (aget v i)) scale))))
    out))

(defn read-embed-row-floats
  "Reads a single token embedding row from an off-heap embed-tokens MemorySegment or reader function."
  [weights-source token-id hidden-dim]
  (cond
    (fn? weights-source)
    (weights-source token-id)

    (instance? MemorySegment weights-source)
    (let [h-dim (long hidden-dim)
          offset (* (long token-id) h-dim 2)
          ^MemorySegment row-seg (.asSlice ^MemorySegment weights-source (long offset) (long (* h-dim 2)))
          sa (short-array h-dim)
          ^MemorySegment target (MemorySegment/ofArray sa)
          _ (.copyFrom target row-seg)
          fa (float-array h-dim)]
      (dotimes [i h-dim]
        (let [s (int (aget sa i))
              bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
          (aset-float fa i (Float/intBitsToFloat bits))))
      fa)

    :else
    (throw (ex-info "Unsupported weights-source type" {:type (type weights-source)}))))

(defn build-llm-anchored-table
  "Builds relation memory map initialized with LLM-anchored entity embeddings derived from
   the model's tied language_model.embed_tokens.weight.
   For each entity:
     1. Tokenize name with leading space: (encode tokenizer (str \" \" name))
     2. Read each token embedding row from embed_tokens slice
     3. Mean-pool rows -> one 1536-dim vector per entity
     4. L2-normalize to unit sphere
   Returns identical map structure to mem/init-relation-memory."
  ([weights-source prefix-base tokenizer entities hidden-dim]
   (build-llm-anchored-table weights-source prefix-base tokenizer entities hidden-dim 3))
  ([weights-source prefix-base tokenizer entities hidden-dim relation-count]
   (let [n (count entities)
         d (long hidden-dim)
         k (long relation-count)
         entity-table (float-array (* n d))
         embed-slice (cond
                       (instance? MemorySegment weights-source)
                       weights-source

                       (fn? weights-source)
                       weights-source

                       (map? weights-source)
                       (let [p-base (or prefix-base
                                        (if (contains? (:header weights-source) "model.language_model.embed_tokens.weight")
                                          "model.language_model."
                                          "model."))
                             tname (str p-base "embed_tokens.weight")]
                         (st/get-tensor-slice weights-source tname))

                       :else
                       (throw (ex-info "Invalid weights-source for anchored table" {:source weights-source})))]

     (dotimes [idx n]
       (let [entity-name (nth entities idx)
             b-id (bos-id tokenizer)
             raw-ids (encode tokenizer (str " " (str/trim entity-name)))
             clean-ids (vec (remove #(= % b-id) raw-ids))
             token-vectors (mapv #(read-embed-row-floats embed-slice % d) clean-ids)
             pooled (mean-pool-embeddings token-vectors d)
             normed (l2-normalize pooled)
             row-offset (* idx d)]
         (dotimes [col d]
           (aset-float entity-table (+ row-offset col) (aget ^floats normed col)))))

     {:entity-table entity-table
      :entity-shape [n d]
      :cores (float-array (* k d d))
      :core-shape [k d d]
      :entity-count n
      :dim d
      :relation-count k})))

(defn orthonormalize-table
  "Applies Modified Gram-Schmidt (MGS) in f64 to orthonormalize the rows of an [N, D] table.
   Input: ^floats table (flat float-array of length N*D), n (row count), d (vector dim).
   Returns: flat float-array of length N*D containing N orthonormal d-vectors Q (in f32).
   Handles degenerate inputs (duplicate/parallel/zero rows) via a deterministic canonical basis
   search policy to guarantee no NaNs and strictly orthonormal rows."
  [^floats table n d]
  (let [n-long (long n)
        d-long (long d)
        q-rows (object-array n-long)
        eps 1e-12]
    (dotimes [i n-long]
      (let [v (double-array d-long)
            offset (* i d-long)]
        ;; 1. Copy row i in double precision
        (dotimes [k d-long]
          (aset v k (double (aget table (+ offset k)))))

        ;; 2. Modified Gram-Schmidt: sequentially project out prior q_j
        (dotimes [j i]
          (let [^doubles q-j (aget q-rows j)
                dot (loop [k 0 s 0.0]
                      (if (>= k d-long)
                        s
                        (recur (inc k) (+ s (* (aget v k) (aget q-j k))))))]
            (dotimes [k d-long]
              (aset v k (- (aget v k) (* dot (aget q-j k)))))))

        ;; 3. Compute Euclidean norm
        (let [norm (Math/sqrt (loop [k 0 s 0.0]
                                (if (>= k d-long)
                                  s
                                  (let [val (aget v k)]
                                    (recur (inc k) (+ s (* val val)))))))]
          ;; 4. Check for degeneracy (norm < eps)
          (if (< norm eps)
            ;; Degenerate/parallel row: search canonical basis vectors e_m
            (let [resolved (double-array d-long)]
              (loop [m 0]
                (if (>= m d-long)
                  (throw (ex-info "Failed to resolve degenerate row in MGS" {:row i :d d-long}))
                  (do
                    ;; Try canonical unit vector e_m
                    (dotimes [k d-long]
                      (aset resolved k (if (= k m) 1.0 0.0)))
                    ;; Project out all prior q_j
                    (dotimes [j i]
                      (let [^doubles q-j (aget q-rows j)
                            dot (aget q-j m)]
                        (dotimes [k d-long]
                          (aset resolved k (- (aget resolved k) (* dot (aget q-j k)))))))
                    (let [c-norm (Math/sqrt (loop [k 0 s 0.0]
                                              (if (>= k d-long)
                                                s
                                                (let [val (aget resolved k)]
                                                  (recur (inc k) (+ s (* val val)))))))]
                      (if (> c-norm 1e-6)
                        ;; Found valid orthogonal basis direction
                        (dotimes [k d-long]
                          (aset v k (/ (aget resolved k) c-norm)))
                        (recur (inc m))))))))
            ;; Non-degenerate: normalize directly
            (dotimes [k d-long]
              (aset v k (/ (aget v k) norm)))))

        (aset q-rows i v)))

    ;; 5. Convert to output flat float-array
    (let [out (float-array (* n-long d-long))]
      (dotimes [i n-long]
        (let [^doubles q-i (aget q-rows i)
              row-offset (* i d-long)]
          (dotimes [k d-long]
            (aset-float out (+ row-offset k) (float (aget q-i k))))))
      out)))

(defn init-qr-anchored-memory
  "Builds relation memory map with the same shape as mem/init-relation-memory,
   where :entity-table is the orthonormalized table Q derived from raw-anchored-mem."
  [raw-anchored-mem]
  (let [n (long (:entity-count raw-anchored-mem))
        d (long (:dim raw-anchored-mem))
        k (long (:relation-count raw-anchored-mem))
        q-table (orthonormalize-table (:entity-table raw-anchored-mem) n d)]
    {:entity-table q-table
     :entity-shape [n d]
     :cores (float-array (* k d d))
     :core-shape [k d d]
     :entity-count n
     :dim d
     :relation-count k}))

;; ==============================================================================
;; 2. Diagnostics: Pairwise Cosine and Margin Statistics
;; ==============================================================================

(defn compute-pairwise-cosine-stats
  "Computes mean, max, and min pairwise cosine similarities across all pairs in table [N, D]."
  [^floats table n dim]
  (let [n-long (long n)
        d-long (long dim)
        norms (float-array n-long)
        _ (dotimes [i n-long]
            (let [offset (* i d-long)
                  sum-sq (loop [k 0 s 0.0]
                           (if (>= k d-long)
                             s
                             (let [v (double (aget table (+ offset k)))]
                               (recur (inc k) (+ s (* v v))))))]
              (aset-float norms i (float (Math/sqrt sum-sq)))))
        cosines (atom [])]
    (dotimes [i n-long]
      (let [i-offset (* i d-long)
            norm-i (double (aget norms i))]
        (loop [j (inc i)]
          (when (< j n-long)
            (let [j-offset (* j d-long)
                  norm-j (double (aget norms j))
                  dot (loop [k 0 s 0.0]
                        (if (>= k d-long)
                          s
                          (let [vi (double (aget table (+ i-offset k)))
                                vj (double (aget table (+ j-offset k)))]
                            (recur (inc k) (+ s (* vi vj))))))
                  denom (* norm-i norm-j)
                  cos-sim (if (> denom 1e-12) (/ dot denom) 0.0)]
              (swap! cosines conj cos-sim)
              (recur (inc j)))))))
    (let [vals @cosines
          cnt (count vals)]
      (if (zero? cnt)
        {:mean 0.0 :max 0.0 :min 0.0}
        (let [sorted-vals (sort vals)
              mean-val (/ (reduce + vals) (double cnt))
              max-val (last sorted-vals)
              min-val (first sorted-vals)]
          {:mean (double mean-val)
           :max (double max-val)
           :min (double min-val)})))))

(defn compute-margin-stats
  "Computes mean and median margin (top-1 score - top-2 score) across query evaluations."
  [margins]
  (let [cnt (count margins)]
    (if (zero? cnt)
      {:mean 0.0 :median 0.0}
      (let [sorted-m (sort margins)
            mean-m (/ (reduce + margins) (double cnt))
            median-m (if (odd? cnt)
                       (nth sorted-m (quot cnt 2))
                       (/ (+ (nth sorted-m (dec (quot cnt 2)))
                             (nth sorted-m (quot cnt 2)))
                          2.0))]
        {:mean (double mean-m)
         :median (double median-m)}))))

(defn compute-score-scale-stats
  "Computes score-scale statistics across a sequence of query evaluation results:
     - mean top-1 score
     - std dev of top-1 score
     - fixed threshold (e.g. 0.5) pass count and rate
     - calibrated threshold (0.5 * mean top-1 score) pass count and rate"
  [results fixed-threshold]
  (let [scores (mapv (fn [r] (double (or (:top-1-score r) 0.0))) results)
        cnt (count scores)]
    (if (zero? cnt)
      {:mean 0.0 :std 0.0
       :fixed-threshold (double fixed-threshold) :fixed-passes 0 :fixed-pass-rate 0.0
       :calibrated-threshold 0.0 :calibrated-passes 0 :calibrated-pass-rate 0.0}
      (let [mean (/ (reduce + scores) (double cnt))
            var-sum (reduce + (map (fn [s] (let [diff (- s mean)] (* diff diff))) scores))
            std (Math/sqrt (/ var-sum (double cnt)))
            cal-thresh (* 0.5 mean)
            f-passes (count (filter #(> % (double fixed-threshold)) scores))
            c-passes (count (filter #(> % cal-thresh) scores))]
        {:mean (double mean)
         :std (double std)
         :fixed-threshold (double fixed-threshold)
         :fixed-passes f-passes
         :fixed-pass-rate (/ (double f-passes) (double cnt))
         :calibrated-threshold (double cal-thresh)
         :calibrated-passes c-passes
         :calibrated-pass-rate (/ (double c-passes) (double cnt))}))))

;; ==============================================================================
;; 3. Evaluation Harness Across Arms
;; ==============================================================================

(defn evaluate-memory-arm
  "Evaluates one experimental arm (entity table + W_mem_proj) across all 7 queries.
   Returns per-query results, accuracy, margin stats, score stats, and gate pass-rate."
  [arm-name session exec device-weights mem w-proj kb-data opts]
  (let [ctx (:ctx session)
        tokenizer (:tokenizer session)
        entities (:entities kb-data)
        id->entity (into {} (map-indexed (fn [idx name] [idx name]) entities))
        rel->id (into {} (map-indexed (fn [idx name] [name idx]) (:relations kb-data)))
        n (count entities)
        vocab-size (long (or (:vocab-size (:config session)) 262144))
        w-vocab (poc-base/build-entity-token-table tokenizer entities vocab-size)
        triples (:triples kb-data)
        num-queries (count triples)
        threshold (double (or (:threshold opts) 0.5))
        max-seq-len 48]

    (println (format "\n--- Evaluating Arm: %s ---" arm-name))
    (let [results
          (mapv
           (fn [idx [head rel expected-tail]]
             (let [head-clean (str/trim head)
                   query-prompt (str "Who is the CEO of " head-clean "?")
                   raw-q-ids (encode tokenizer query-prompt)
                   clean-q-ids (if (= (first raw-q-ids) (bos-id tokenizer))
                                 (vec (rest raw-q-ids))
                                 (vec raw-q-ids))
                   prefix [(bos-id tokenizer) 105 2364 107]
                   suffix [106 107 105 4368 107]
                   q-prompt-ids (vec (concat prefix clean-q-ids suffix))
                   q-prompt-len (count q-prompt-ids)
                   q-rel-id (get rel->id rel 0)

                   ;; Allocate relational buffers with arm's W_mem_proj and memory table:
                   grounded-bufs (g4/allocate-relational-buffers session mem
                                                                 {:rel-id q-rel-id
                                                                  :w-mem-proj w-proj
                                                                  :w-entity-to-vocab w-vocab})

                   in-arr (int-array max-seq-len)
                   _ (dotimes [i q-prompt-len] (aset in-arr i (int (nth q-prompt-ids i))))
                   in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                   pos-arr (int-array [(dec q-prompt-len)])
                   pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                   args (into [in-b pos-b] (concat device-weights grounded-bufs))
                   out (xla/execute exec args)
                   ^floats scores (xla/to-host-slice (nth out 1) 0 n n :bf16)
                   sorted-entities (vec (sort-by (fn [[_idx sc]] (- sc))
                                                 (map-indexed (fn [i sc] [i (double sc)]) scores)))
                   top-1-id (first (first sorted-entities))
                   top-1-name (get id->entity top-1-id)
                   top-1-score (double (second (first sorted-entities)))
                   top-2-score (double (second (second sorted-entities)))
                   margin (- top-1-score top-2-score)
                   top-3 (mapv (fn [[i sc]] [(get id->entity i) (double sc)]) (take 3 sorted-entities))
                   hit? (= top-1-name expected-tail)
                   gate-passed? (> top-1-score threshold)]

               (xla/destroy-buffer! ctx in-b)
               (xla/destroy-buffer! ctx pos-b)
               (doseq [b out] (xla/destroy-buffer! ctx b))
               (g4/destroy-relational-buffers! ctx grounded-bufs)

               (println (format " [Query %d/%d] \"%s\"" (inc idx) num-queries query-prompt))
               (println (format "   Expected: %-16s | Top-1: %-16s (score: %6.2f, margin: %6.2f) -> %s"
                                (str "\"" expected-tail "\"") (str "\"" top-1-name "\"")
                                top-1-score margin (if hit? "HIT [CORRECT]" "MISS")))
               (println (format "   Top-3: %s"
                                (str/join ", " (map (fn [[ename sc]] (format "\"%s\":%.2f" ename sc)) top-3))))

               {:head head
                :expected expected-tail
                :top-1 top-1-name
                :top-1-score top-1-score
                :margin margin
                :hit? hit?
                :gate-passed? gate-passed?}))
           (range num-queries)
           triples)

          hits (count (filter :hit? results))
          accuracy (/ (double hits) (double num-queries))
          gate-passes (count (filter :gate-passed? results))
          gate-pass-rate (/ (double gate-passes) (double num-queries))
          margins (mapv :margin results)
          margin-stats (compute-margin-stats margins)
          score-stats (compute-score-scale-stats results threshold)]

      {:arm arm-name
       :results results
       :hits hits
       :accuracy accuracy
       :gate-pass-rate gate-pass-rate
       :margin-stats margin-stats
       :score-stats score-stats})))

;; ==============================================================================
;; 4. Main Experiment Pipeline
;; ==============================================================================

(defn run-anchored-memory-experiment [opts]
  (println "==================================================================")
  (println "  clj-xla: LLM-Anchored Entity Embeddings (Zero-Shot Bridge)")
  (println "==================================================================")
  (println (str "Backend:           [" (:backend opts) "]"))
  (println (str "Model:             [" (:model opts) "]"))
  (println (str "Triples File:      [" (:triples opts) "]"))
  (println (str "Memory Dim D:      [" (:dim opts) "] (1536 = Gemma hidden-dim)"))
  (println (str "Threshold:         [" (:threshold opts) "]"))
  (println (str "Train Refinement:  [" (:train-refinement opts) "]"))
  (println "==================================================================\n")

  (let [triples-file (io/file (:triples opts))]
    (when-not (.exists triples-file)
      (throw (ex-info (str "Triples file not found: " (:triples opts)) {:file (:triples opts)})))

    (let [kb-data (edn/read-string (slurp triples-file))
          entities (vec (:entities kb-data))
          n (count entities)
          d (long (:dim opts))
          k (count (:relations kb-data))
          w-identity (build-identity-matrix d)
          max-seq-len 48
          base-session (g4/init-inference-session (assoc opts :max-seq-len max-seq-len))
          ctx (:ctx base-session)
          tokenizer (:tokenizer base-session)
          config (assoc (:config base-session)
                        :relational-memory {:dim d :entity-count n}
                        :max-seq-len max-seq-len
                        :last-token-only? true)
          session (assoc base-session :config config)
          device-weights (g4/allocate-device-weights session)

          ;; Compile Gemma 4 with active relational grounding block @ D=1536
          exec (g4/compile-tensor-logic-executable
                (assoc session :opts (assoc opts :targets [:logits :entity_scores :normed_last]))
                max-seq-len)

          ;; --------------------------------------------------------------------
          ;; ARM 1: Random Entity Table @ D=1536, W = I (Control)
          ;; --------------------------------------------------------------------
          mem-arm1 (mem/init-relation-memory n d k (or (:memory-seed opts) 2026))
          _ (let [entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
                  rel->id (into {} (map-indexed (fn [idx name] [name idx]) (:relations kb-data)))]
              (doseq [[h r t] (:triples kb-data)]
                (let [h-i (get entity->id h)
                      r-i (get rel->id r)
                      t-i (get entity->id t)]
                  (when (and h-i r-i t-i)
                    (mem/accumulate-fact! (:cores mem-arm1) (:entity-table mem-arm1) h-i r-i t-i d)))))
          arm1-res (evaluate-memory-arm "Arm 1 (Random Table @ D=1536, W = I)"
                                        session exec device-weights mem-arm1 w-identity kb-data opts)

          ;; --------------------------------------------------------------------
          ;; ARM 2: LLM-Anchored Raw @ D=1536, W = I (Replicates Task C 0/7)
          ;; --------------------------------------------------------------------
          weights-mmap (:weights-mmap base-session)
          prefix-base (:prefix-base base-session)
          _ (println "\nBuilding LLM-anchored raw entity table from Gemma 4 embed_tokens.weight...")
          mem-arm2 (build-llm-anchored-table weights-mmap prefix-base tokenizer entities d k)
          _ (let [entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
                  rel->id (into {} (map-indexed (fn [idx name] [name idx]) (:relations kb-data)))]
              (doseq [[h r t] (:triples kb-data)]
                (let [h-i (get entity->id h)
                      r-i (get rel->id r)
                      t-i (get entity->id t)]
                  (when (and h-i r-i t-i)
                    (mem/accumulate-fact! (:cores mem-arm2) (:entity-table mem-arm2) h-i r-i t-i d)))))
          arm2-res (evaluate-memory-arm "Arm 2 (LLM-Anchored Raw @ D=1536, W = I)"
                                        session exec device-weights mem-arm2 w-identity kb-data opts)

          ;; --------------------------------------------------------------------
          ;; ARM 4: LLM-Anchored QR @ D=1536, W = I (The Test: Orthonormalized)
          ;; --------------------------------------------------------------------
          _ (println "\nBuilding LLM-anchored QR-orthonormalized entity table (MGS in f64)...")
          mem-arm4 (init-qr-anchored-memory mem-arm2)
          _ (let [entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
                  rel->id (into {} (map-indexed (fn [idx name] [name idx]) (:relations kb-data)))]
              (doseq [[h r t] (:triples kb-data)]
                (let [h-i (get entity->id h)
                      r-i (get rel->id r)
                      t-i (get entity->id t)]
                  (when (and h-i r-i t-i)
                    (mem/accumulate-fact! (:cores mem-arm4) (:entity-table mem-arm4) h-i r-i t-i d)))))
          arm4-res (evaluate-memory-arm "Arm 4 (LLM-Anchored QR @ D=1536, W = I)"
                                        session exec device-weights mem-arm4 w-identity kb-data opts)

          ;; --------------------------------------------------------------------
          ;; Pairwise Cosine Diagnostics (Off-Diagonal Similarity)
          ;; --------------------------------------------------------------------
          cos-arm1 (compute-pairwise-cosine-stats (:entity-table mem-arm1) n d)
          cos-arm2 (compute-pairwise-cosine-stats (:entity-table mem-arm2) n d)
          cos-arm4 (compute-pairwise-cosine-stats (:entity-table mem-arm4) n d)]

      ;; --------------------------------------------------------------------
      ;; Per-Entity Breakdown & Fragmentation Analysis
      ;; --------------------------------------------------------------------
      (println "\n------------------------------------------------------------------")
      (println " PER-ENTITY BREAKDOWN & SUB-TOKEN FRAGMENTATION:")
      (println "------------------------------------------------------------------")
      (doseq [idx (range (count (:triples kb-data)))]
        (let [r1 (nth (:results arm1-res) idx)
              r2 (nth (:results arm2-res) idx)
              r4 (nth (:results arm4-res) idx)
              head (:head r1)
              exp (:expected r1)
              rare? (contains? #{"Alpeware" "Simon Pure"} head)]
          (println (format " [%s%s] Head: %-12s | Expected: %-14s"
                           (if rare? "RARE " "COMMON")
                           (if rare? "(!)" "   ")
                           head (str "\"" exp "\"")))
          (println (format "   Arm 1 (Random):      Top-1: %-14s (margin: %5.2f) -> %s"
                           (str "\"" (:top-1 r1) "\"") (:margin r1) (if (:hit? r1) "HIT" "MISS")))
          (println (format "   Arm 2 (AnchoredRaw): Top-1: %-14s (margin: %5.2f) -> %s"
                           (str "\"" (:top-1 r2) "\"") (:margin r2) (if (:hit? r2) "HIT" "MISS")))
          (println (format "   Arm 4 (AnchoredQR):  Top-1: %-14s (margin: %5.2f) -> %s"
                           (str "\"" (:top-1 r4) "\"") (:margin r4) (if (:hit? r4) "HIT" "MISS")))))

      ;; --------------------------------------------------------------------
      ;; Summary Comparison Report
      ;; --------------------------------------------------------------------
      (println "\n==================================================================")
      (println " SUMMARY: QR-Orthonormalized Anchored Memory Experiment (Task D)")
      (println "==================================================================")
      (println (format " Entity Universe:   %d entities" n))
      (println " Interpretation Guide (n=7, chance=1/14=7.1%):")
      (println "   0–1 / 7 ( 0.0% – 14.3%): Chance-consistent")
      (println "     2 / 7 ( 28.6%):        Suggestive (p ≈ 0.09)")
      (println "   ≥ 3 / 7 (≥ 42.9%):       Significant (p ≈ 0.01)")
      (println "------------------------------------------------------------------")
      (println " PRIMARY RETRIEVAL ACCURACY (Scale-free, pre-gate :entity_scores):")
      (println "   Arm 3 (Reference): 0 / 7 ( 0.0%) [Random @ D=256, W=random]")
      (println (format "   Arm 1 (Control):   %d / 7 (%5.1f%%) [Random @ D=1536, W=I]"
                       (:hits arm1-res) (* 100.0 (double (:accuracy arm1-res)))))
      (println (format "   Arm 2 (Re-run):    %d / 7 (%5.1f%%) [Anchored Raw @ D=1536, W=I]"
                       (:hits arm2-res) (* 100.0 (double (:accuracy arm2-res)))))
      (println (format "   Arm 4 (Test):      %d / 7 (%5.1f%%) [Anchored QR @ D=1536, W=I]"
                       (:hits arm4-res) (* 100.0 (double (:accuracy arm4-res)))))
      (println "------------------------------------------------------------------")
      (println " MARGIN STATISTICS (Top-1 - Top-2 Score):")
      (println (format "   Arm 1 (Random):      Mean: %6.2f | Median: %6.2f"
                       (:mean (:margin-stats arm1-res)) (:median (:margin-stats arm1-res))))
      (println (format "   Arm 2 (AnchoredRaw): Mean: %6.2f | Median: %6.2f"
                       (:mean (:margin-stats arm2-res)) (:median (:margin-stats arm2-res))))
      (println (format "   Arm 4 (AnchoredQR):  Mean: %6.2f | Median: %6.2f"
                       (:mean (:margin-stats arm4-res)) (:median (:margin-stats arm4-res))))
      (println "------------------------------------------------------------------")
      (println " TABLE CROSS-TALK DIAGNOSTICS (Off-Diagonal Pairwise Cosines across 14 entities):")
      (println (format "   Arm 1 (Random):      Mean: %9.6f | Max: %9.6f | Min: %9.6f"
                       (:mean cos-arm1) (:max cos-arm1) (:min cos-arm1)))
      (println (format "   Arm 2 (AnchoredRaw): Mean: %9.6f | Max: %9.6f | Min: %9.6f"
                       (:mean cos-arm2) (:max cos-arm2) (:min cos-arm2)))
      (println (format "   Arm 4 (AnchoredQR):  Mean: %9.6f | Max: %9.6f | Min: %9.6f"
                       (:mean cos-arm4) (:max cos-arm4) (:min cos-arm4)))
      (println "------------------------------------------------------------------")
      (println " SCORE SCALE & GATE PASS-RATES:")
      (println " [Note: Cross-arm comparison at fixed threshold 0.5 is invalid due to score scale;")
      (println "        verdict leads with scale-free accuracy.]")
      (let [s1 (:score-stats arm1-res)
            s2 (:score-stats arm2-res)
            s4 (:score-stats arm4-res)]
        (println (format "   Arm 1: Mean Top-1: %6.2f (std: %5.2f) | Fixed @ 0.5: %d/7 (%5.1f%%) | Calibrated @ %5.2f: %d/7 (%5.1f%%)"
                         (:mean s1) (:std s1) (:fixed-passes s1) (* 100.0 (:fixed-pass-rate s1))
                         (:calibrated-threshold s1) (:calibrated-passes s1) (* 100.0 (:calibrated-pass-rate s1))))
        (println (format "   Arm 2: Mean Top-1: %6.2f (std: %5.2f) | Fixed @ 0.5: %d/7 (%5.1f%%) | Calibrated @ %5.2f: %d/7 (%5.1f%%)"
                         (:mean s2) (:std s2) (:fixed-passes s2) (* 100.0 (:fixed-pass-rate s2))
                         (:calibrated-threshold s2) (:calibrated-passes s2) (* 100.0 (:calibrated-pass-rate s2))))
        (println (format "   Arm 4: Mean Top-1: %6.2f (std: %5.2f) | Fixed @ 0.5: %d/7 (%5.1f%%) | Calibrated @ %5.2f: %d/7 (%5.1f%%)"
                         (:mean s4) (:std s4) (:fixed-passes s4) (* 100.0 (:fixed-pass-rate s4))
                         (:calibrated-threshold s4) (:calibrated-passes s4) (* 100.0 (:calibrated-pass-rate s4)))))
      (println "==================================================================")

      ;; Verdict Paragraph
      (println "\nVERDICT & MECHANISTIC INTERPRETATION:")
      (let [hits4 (:hits arm4-res)
            margin4 (:mean (:margin-stats arm4-res))
            margin1 (:mean (:margin-stats arm1-res))
            margin2 (:mean (:margin-stats arm2-res))
            margin-recovered? (> margin4 (* 1.5 margin2))]
        (cond
          (>= hits4 3)
          (do
            (println " [Branch (a)]: Arm 4 achieved ≥ 3/7 retrieval with recovered margins.")
            (println " Causal Attribution: Entity-vector cross-talk was the primary blocker.")
            (println " Preserving Gemma's 14-dim entity subspace via QR-orthonormalization restored clean addressing.")
            (println " The zero-shot bridge hypothesis lives! Proceed to refinement/LOO and scale-up."))

          (and (<= hits4 1) margin-recovered?)
          (do
            (println (format " [Branch (b)]: Arm 4 retrieval is chance-consistent (%d/7), but margins recovered (%.2f vs %.2f raw, vs %.2f control)."
                             hits4 margin4 margin2 margin1))
            (println " Causal Attribution: Addressing is clean (cross-talk eliminated, max off-diagonal pairwise cosine ~ 1e-6),")
            (println " but the subspace is wrong. Distribution shift between contextual question-probe hidden states and")
            (println " token-embedding means is the fundamental blocker, NOT cross-talk.")
            (println " The remaining lever is a learned linear projection map (Stage 3) or a span-pooled probe."))

          :else
          (do
            (println (format " [Branch (c)]: Arm 4 retrieval is chance-consistent (%d/7) and margins remain compressed (%.2f vs %.2f raw, vs %.2f control)."
                             hits4 margin4 margin2 margin1))
            (println " Causal Attribution: Something beyond cross-talk and subspace orientation (likely probe-side geometry")
            (println " or token fragmentation) is dominating query addressing."))))
      (println "==================================================================\n")

      ;; Stage 3 Refinement (if requested)
      (when (:train-refinement opts)
        (println "\n------------------------------------------------------------------")
        (println " STAGE 3: Conditional Refinement (1536x1536 map init=I on Arm 4 QR Table)")
        (println "------------------------------------------------------------------")
        (let [dataset (train-base/collect-frozen-probes session exec device-weights kb-data opts max-seq-len)
              _ (println " Training 1536x1536 refinement map starting from W = Identity on Arm 4 table...")
              refine-res (train-base/train-projection dataset w-identity (:entity-table mem-arm4) d d n
                                                      {:lr 0.01 :steps 300
                                                       :on-step (fn [s loss acc]
                                                                  (when (or (zero? s) (zero? (mod s 50)))
                                                                    (println (format "  Step %3d: Loss=%.4f, Acc=%.1f%%"
                                                                                     s (double loss) (* 100.0 (double acc))))))})
              _ (println (format " Refinement Training Finished: Final Loss=%.4f, Acc=%.1f%%"
                                 (double (:final-loss refine-res)) (* 100.0 (double (:final-acc refine-res)))))
              _ (println " Evaluating 7-fold Leave-One-Out CV on Arm 4 QR table...")
              loo-res (train-base/run-leave-one-out-cv dataset w-identity (:entity-table mem-arm4) d d n
                                                       {:lr 0.01 :steps 300})]
          (println (format " Arm 4 QR Space LOO Mean Accuracy: %.1f%% (%d/%d)"
                           (* 100.0 (double (:mean-acc loo-res))) (:total-hits loo-res) n))
          (doseq [{:keys [fold target pred-id expected-id hit?]} (:fold-results loo-res)]
            (println (format "   Fold %d: Held out \"%-12s\" -> Pred: %-14s, True: %-14s -> %s"
                             (inc fold) target (nth entities pred-id) (nth entities expected-id)
                             (if hit? "HIT [CORRECT]" "MISS"))))))

      ;; Cleanup device weights
      (doseq [w device-weights]
        (xla/destroy-buffer! ctx w))

      {:arm1 arm1-res
       :arm2 arm2-res
       :arm4 arm4-res
       :cos-arm1 cos-arm1
       :cos-arm2 cos-arm2
       :cos-arm4 cos-arm4})))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (if (g4/needs-libjsig-reexec? opts)
      (g4/reexec-with-libjsig! args "tools.poc-anchored-memory")
      (do
        (run-anchored-memory-experiment opts)
        (System/exit 0)))))
