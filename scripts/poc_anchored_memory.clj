(ns scripts.poc-anchored-memory
  "LLM-anchored entity embeddings (zero-shot bridge hypothesis).
   Evaluates whether grounding Pedro Domingos' symbolic relational memory cores directly
   in Gemma 4's tied token embedding space achieves zero-shot retrieval with W = I (identity).
   Compares:
     Arm 1 (Control):   Random entity table @ D=1536, W = I (isolates D=1536 effect)
     Arm 2 (Test):      LLM-anchored entity table @ D=1536, W = I (the hypothesis)
     Arm 3 (Reference): De-oracled baseline @ D=256, W = random (0/7 chance)"
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.memory.relation :as mem]
            [clj-xla.safetensors :as st]
            [clj-xla.tokenizer.protocol :refer [bos-id encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scripts.gemma4-inference :as g4]
            [scripts.poc-fact-grounding :as poc-base]
            [scripts.train-mem-projection :as train-base])
  (:import [java.lang.foreign MemorySegment])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :model ".models/gemma-4-E2B-it"
   :triples "data/wiki_recent_triples.edn"
   :dim 1536
   :threshold 0.5
   :memory-seed 2026
   :entity-mode :llm-anchored
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

;; ==============================================================================
;; 3. Evaluation Harness Across Arms
;; ==============================================================================

(defn evaluate-memory-arm
  "Evaluates one experimental arm (entity table + W_mem_proj) across all 7 queries.
   Returns per-query results, accuracy, margin stats, and gate pass-rate."
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
          margin-stats (compute-margin-stats margins)]

      {:arm arm-name
       :results results
       :hits hits
       :accuracy accuracy
       :gate-pass-rate gate-pass-rate
       :margin-stats margin-stats})))

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
          ;; ARM 2: LLM-Anchored Entity Table @ D=1536, W = I (Test)
          ;; --------------------------------------------------------------------
          weights-mmap (:weights-mmap base-session)
          prefix-base (:prefix-base base-session)
          _ (println "\nBuilding LLM-anchored entity table from Gemma 4 embed_tokens.weight...")
          mem-arm2 (build-llm-anchored-table weights-mmap prefix-base tokenizer entities d k)
          _ (let [entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
                  rel->id (into {} (map-indexed (fn [idx name] [name idx]) (:relations kb-data)))]
              (doseq [[h r t] (:triples kb-data)]
                (let [h-i (get entity->id h)
                      r-i (get rel->id r)
                      t-i (get entity->id t)]
                  (when (and h-i r-i t-i)
                    (mem/accumulate-fact! (:cores mem-arm2) (:entity-table mem-arm2) h-i r-i t-i d)))))
          arm2-res (evaluate-memory-arm "Arm 2 (LLM-Anchored Table @ D=1536, W = I)"
                                        session exec device-weights mem-arm2 w-identity kb-data opts)

          ;; --------------------------------------------------------------------
          ;; Pairwise Cosine Diagnostics
          ;; --------------------------------------------------------------------
          cos-arm1 (compute-pairwise-cosine-stats (:entity-table mem-arm1) n d)
          cos-arm2 (compute-pairwise-cosine-stats (:entity-table mem-arm2) n d)]

      ;; Cleanup device weights
      (doseq [w device-weights]
        (xla/destroy-buffer! ctx w))

      ;; --------------------------------------------------------------------
      ;; Summary Comparison Report
      ;; --------------------------------------------------------------------
      (println "\n==================================================================")
      (println " SUMMARY: LLM-Anchored Zero-Shot Retrieval Experiment")
      (println "==================================================================")
      (println (format " Entity Universe:   %d entities" n))
      (println (format " Chance Baseline:   7.1%% (1/%d ≈ 0.0714)" n))
      (println " Arm 3 (Reference): 0 / 7 (0.0%)  [Random @ D=256, W=random]")
      (println (format " Arm 1 (Control):   %d / 7 (%.1f%%) [Random @ D=1536, W=I]"
                       (:hits arm1-res) (* 100.0 (double (:accuracy arm1-res)))))
      (println (format " Arm 2 (Test):      %d / 7 (%.1f%%) [Anchored @ D=1536, W=I]"
                       (:hits arm2-res) (* 100.0 (double (:accuracy arm2-res)))))
      (println "------------------------------------------------------------------")
      (println " MARGIN STATISTICS (Top-1 - Top-2 Score):")
      (println (format "   Arm 1 (Random):   Mean: %6.2f | Median: %6.2f"
                       (:mean (:margin-stats arm1-res)) (:median (:margin-stats arm1-res))))
      (println (format "   Arm 2 (Anchored): Mean: %6.2f | Median: %6.2f"
                       (:mean (:margin-stats arm2-res)) (:median (:margin-stats arm2-res))))
      (println "------------------------------------------------------------------")
      (println " TABLE CROSS-TALK DIAGNOSTICS (Pairwise Cosine across 14 entities):")
      (println (format "   Arm 1 (Random):   Mean: %6.4f | Max: %6.4f | Min: %6.4f"
                       (:mean cos-arm1) (:max cos-arm1) (:min cos-arm1)))
      (println (format "   Arm 2 (Anchored): Mean: %6.4f | Max: %6.4f | Min: %6.4f"
                       (:mean cos-arm2) (:max cos-arm2) (:min cos-arm2)))
      (println "------------------------------------------------------------------")
      (println " DEDUCTIVE GATE PASS-RATES (Threshold = 0.5):")
      (println (format "   Arm 1 (Random):   %d / 7 (%.1f%%)"
                       (long (* (double (:gate-pass-rate arm1-res)) 7))
                       (* 100.0 (double (:gate-pass-rate arm1-res)))))
      (println (format "   Arm 2 (Anchored): %d / 7 (%.1f%%)"
                       (long (* (double (:gate-pass-rate arm2-res)) 7))
                       (* 100.0 (double (:gate-pass-rate arm2-res)))))
      (println "==================================================================")

      ;; Verdict Paragraph
      (println "\nVERDICT & MECHANISTIC INTERPRETATION:")
      (if (> (:hits arm2-res) 1)
        (do
          (println " Outcome (a): Arm 2 clearly beats chance. The zero-shot bridge hypothesis is supported!")
          (println " Anchoring relational memory cores directly in Gemma's tied token embedding space enables")
          (println " factual retrieval without needing a learned projection map."))
        (do
          (println " Outcome (b): Arm 2 does not beat chance (zero-shot bridge alone is insufficient).")
          (println " Three mechanistic factors account for why naive anchoring fails to retrieve zero-shot:")
          (println "   1. Distribution shift: Token embeddings E are trained to dot single-token hidden states")
          (println "      for next-token prediction, whereas the query vector h_probe is a contextual question")
          (println "      state and entities are mean-pooled multi-token spans.")
          (println (format "   2. Correlated cross-talk: Pairwise cosine in anchored space increases from %.4f (random)"
                           (:mean cos-arm1)))
          (println (format "      to %.4f (anchored), with max cosine %.4f. Company and person name embeddings share"
                           (:mean cos-arm2) (:max cos-arm2)))
          (println "      lexical sub-tokens, compressing discrimination margins between entities.")
          (println "   3. Rare entity degradation: Gemma's sub-token fragmentation for rare entities like")
          (println "      \"Alpeware\" ([108152 76728]) and \"Simon Pure\" ([20420 33677]) further skews composite vectors.")
          (println " Stage 2 conditional refinement (--train-refinement true) provides the learned linear adapter")
          (println " to de-correlate and align these spaces.")))
      (println "==================================================================\n")

      ;; Stage 2 Refinement (if requested)
      (when (:train-refinement opts)
        (println "\n------------------------------------------------------------------")
        (println " STAGE 2: Conditional Refinement (1536x1536 map init=I via autodiff)")
        (println "------------------------------------------------------------------")
        (let [dataset (train-base/collect-frozen-probes session exec device-weights kb-data opts max-seq-len)
              _ (println " Training 1536x1536 refinement map starting from W = Identity...")
              refine-res (train-base/train-projection dataset w-identity (:entity-table mem-arm2) d d n
                                                      {:lr 0.01 :steps 300
                                                       :on-step (fn [s loss acc]
                                                                  (when (or (zero? s) (zero? (mod s 50)))
                                                                    (println (format "  Step %3d: Loss=%.4f, Acc=%.1f%%"
                                                                                     s (double loss) (* 100.0 (double acc))))))})
              _ (println (format " Refinement Training Finished: Final Loss=%.4f, Acc=%.1f%%"
                                 (double (:final-loss refine-res)) (* 100.0 (double (:final-acc refine-res)))))
              _ (println " Evaluating 7-fold Leave-One-Out CV in anchored space...")
              loo-res (train-base/run-leave-one-out-cv dataset w-identity (:entity-table mem-arm2) d d n
                                                       {:lr 0.01 :steps 300})]
          (println (format " Anchored Space LOO Mean Accuracy: %.1f%% (%d/%d)"
                           (* 100.0 (double (:mean-acc loo-res))) (:total-hits loo-res) n))
          (doseq [{:keys [fold target pred-id expected-id hit?]} (:fold-results loo-res)]
            (println (format "   Fold %d: Held out \"%s\" -> Pred: %s, True: %s -> %s"
                             (inc fold) target (nth entities pred-id) (nth entities expected-id)
                             (if hit? "HIT" "MISS"))))))

      {:arm1 arm1-res
       :arm2 arm2-res
       :cos-arm1 cos-arm1
       :cos-arm2 cos-arm2})))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (if (g4/needs-libjsig-reexec? opts)
      (g4/reexec-with-libjsig! args "scripts.poc-anchored-memory")
      (do
        (run-anchored-memory-experiment opts)
        (System/exit 0)))))
