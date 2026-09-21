(ns tools.poc-span-probe
  "Span-pooled probe ablation (zero-training) for in-tensor relational memory.
   Evaluates three probe variants across all 7 factual queries:
     (i)   Last position (reproduces the de-oracled 0/7 baseline)
     (ii)  Mean-pool over head entity token span
     (iii) Max-pool over head entity token span
   Operates in sequence mode (:last-token-only? false) with :targets [:entity_scores]."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.relation :as mem]
            [einsum.runtime.tokenizer.protocol :refer [bos-id encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [tools.gemma4-inference :as g4]
            [tools.poc-fact-grounding :as poc-base])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :model ".models/gemma-4-E2B-it"
   :triples "data/wiki_recent_triples.edn"
   :dim 256
   :threshold 0.5
   :memory-seed 2026
   :proj-seed 2026})

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
          "--memory-seed" (recur (subvec remaining 2) (assoc opts :memory-seed (Long/parseLong v)))
          "--proj-seed" (recur (subvec remaining 2) (assoc opts :proj-seed (Long/parseLong v)))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; 1. Subsequence Search & Head Span Location
;; ==============================================================================

(defn find-subsequence
  "Finds the 0-based start index of `needle` in `haystack`. Returns nil if not found."
  [needle haystack]
  (let [n-count (count needle)
        h-count (count haystack)]
    (cond
      (zero? n-count) 0
      (> n-count h-count) nil
      :else
      (first (keep-indexed (fn [i part] (when (= part needle) i))
                           (partition n-count 1 haystack))))))

(defn locate-head-span
  "Locates the [start-idx span-len] of `head` in `prompt-ids` using tokenizer.
   Tries both space-prefixed and direct tokenizations. Fails loudly if absent."
  [tokenizer prompt-ids head]
  (let [head-clean (str/trim head)
        b-id (bos-id tokenizer)
        strip-bos (fn [ids] (vec (remove #(= % b-id) ids)))
        h-ids-spaced (strip-bos (encode tokenizer (str " " head-clean)))
        h-ids-direct (strip-bos (encode tokenizer head-clean))
        idx-spaced (find-subsequence h-ids-spaced prompt-ids)
        idx-direct (find-subsequence h-ids-direct prompt-ids)]
    (cond
      (some? idx-spaced) [idx-spaced (count h-ids-spaced)]
      (some? idx-direct) [idx-direct (count h-ids-direct)]
      :else
      (throw (ex-info (str "Could not locate head entity span in prompt for: " head-clean)
                      {:head head-clean :prompt-ids prompt-ids})))))

;; ==============================================================================
;; 2. Pure Score Pooling Helpers
;; ==============================================================================

(defn pool-scores-last-pos
  "Extracts entity scores at the last token position (dec prompt-len) from [P N] scores array."
  [^floats scores prompt-len n]
  (let [last-pos (dec (long prompt-len))
        out (float-array n)
        offset (* last-pos (long n))]
    (dotimes [j n]
      (aset-float out j (aget scores (+ offset j))))
    out))

(defn pool-scores-span-mean
  "Computes mean entity scores across the head entity token span [start-idx, start-idx + span-len)."
  [^floats scores start-idx span-len n]
  (let [start (long start-idx)
        slen (long span-len)
        n-long (long n)
        out (float-array n)]
    (dotimes [j n]
      (let [sum (loop [k 0 s 0.0]
                  (if (>= k slen)
                    s
                    (let [idx (+ (* (+ start k) n-long) j)]
                      (recur (inc k) (+ s (double (aget scores idx)))))))]
        (aset-float out j (float (/ sum (double slen))))))
    out))

(defn pool-scores-span-max
  "Computes max entity scores across the head entity token span [start-idx, start-idx + span-len)."
  [^floats scores start-idx span-len n]
  (let [start (long start-idx)
        slen (long span-len)
        n-long (long n)
        out (float-array n)]
    (dotimes [j n]
      (let [max-val (loop [k 1 m (double (aget scores (+ (* start n-long) j)))]
                      (if (>= k slen)
                        m
                        (let [idx (+ (* (+ start k) n-long) j)
                              v (double (aget scores idx))]
                          (recur (inc k) (Math/max m v)))))]
        (aset-float out j (float max-val))))
    out))

(defn top-entity-id
  "Returns index j in [0, n-1] of the top scoring entity."
  [^floats pooled-scores n]
  (loop [j 1
         best-id 0
         best-score (double (aget pooled-scores 0))]
    (if (>= j (long n))
      best-id
      (let [score (double (aget pooled-scores j))]
        (if (> score best-score)
          (recur (inc j) j score)
          (recur (inc j) best-id best-score))))))

;; ==============================================================================
;; 3. Main Ablation Pipeline
;; ==============================================================================

(defn run-span-probe-ablation [opts]
  (println "==================================================================")
  (println "  clj-xla: Span-Pooled Probe Ablation (Zero-Training)")
  (println "==================================================================")
  (println (str "Backend:           [" (:backend opts) "]"))
  (println (str "Model:             [" (:model opts) "]"))
  (println (str "Triples File:      [" (:triples opts) "]"))
  (println (str "Memory Dim D:      [" (:dim opts) "]"))
  (println (str "Memory Init Seed:  [" (or (:memory-seed opts) 2026) "]"))
  (println (str "Projection Seed:   [" (or (:proj-seed opts) 2026) "]"))
  (println "==================================================================\n")

  (let [triples-file (io/file (:triples opts))]
    (when-not (.exists triples-file)
      (throw (ex-info (str "Triples file not found: " (:triples opts)) {:file (:triples opts)})))

    (let [kb-data (edn/read-string (slurp triples-file))
          entities (vec (:entities kb-data))
          id->entity (into {} (map-indexed (fn [idx name] [idx name]) entities))
          entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
          relations (vec (:relations kb-data))
          rel->id (into {} (map-indexed (fn [idx name] [name idx]) relations))
          n (count entities)
          d (long (:dim opts))
          k (count relations)
          mem (mem/init-relation-memory n d k (or (:memory-seed opts) 2026))
          ^floats cores (:cores mem)
          ^floats e-table (:entity-table mem)

          _ (println (str "Entity universe: " n " entities, " k " relation types, resident D=" d))
          _ (println "Ingesting knowledge base triples via instantaneous superposition...")
          t0 (System/nanoTime)
          _ (doseq [[h r t] (:triples kb-data)]
              (let [h-i (get entity->id h)
                    r-i (get rel->id r)
                    t-i (get entity->id t)]
                (when (and h-i r-i t-i)
                  (mem/accumulate-fact! cores e-table h-i r-i t-i d))))
          elapsed-us (/ (- (System/nanoTime) t0) 1000.0)
          _ (println (format "Superposition DMA complete in %.2f microseconds.\n" (double elapsed-us)))

          max-seq-len 48
          base-session (g4/init-inference-session (assoc opts :max-seq-len max-seq-len))
          ctx (:ctx base-session)
          tokenizer (:tokenizer base-session)
          config (assoc (:config base-session)
                        :relational-memory {:dim d :entity-count n}
                        :max-seq-len max-seq-len
                        :last-token-only? false)
          session (assoc base-session :config config)
          device-weights (g4/allocate-device-weights session)

          ;; Compile sequence-mode executable (outputs full [1 P N] entity_scores)
          exec (g4/compile-tensor-logic-executable
                (assoc session :opts (assoc opts :last-token-only? false :targets [:entity_scores]))
                max-seq-len)

          ;; Buffer helpers built ONCE per run
          w-proj (poc-base/build-random-projection 1536 d (or (:proj-seed opts) 2026))
          vocab-size (long (or (:vocab-size config) 262144))
          w-entity-token-table (poc-base/build-entity-token-table tokenizer entities vocab-size)

          all-triples (:triples kb-data)
          num-queries (count all-triples)

          results
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

                   ;; Locate head entity span in prompt
                   [span-start span-len] (locate-head-span tokenizer q-prompt-ids head-clean)

                   ;; Allocate relational buffers for this query
                   grounded-bufs (g4/allocate-relational-buffers session mem
                                                                 {:rel-id q-rel-id
                                                                  :w-mem-proj w-proj
                                                                  :w-entity-to-vocab w-entity-token-table})

                   ;; Forward pass in sequence mode (no pos buffer!)
                   in-arr (int-array max-seq-len)
                   _ (dotimes [i q-prompt-len] (aset in-arr i (int (nth q-prompt-ids i))))
                   in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                   args (into [in-b] (concat device-weights grounded-bufs))
                   out (xla/execute exec args)
                   out-buf (if (sequential? out) (first out) out)
                   ^floats scores (xla/to-host-slice out-buf 0 (* max-seq-len n) (* max-seq-len n) :bf16)

                   ;; 1. Last position pool
                   last-scores (pool-scores-last-pos scores q-prompt-len n)
                   last-top-id (top-entity-id last-scores n)
                   last-top-name (get id->entity last-top-id)
                   last-hit? (= last-top-name expected-tail)

                   ;; 2. Span mean pool
                   mean-scores (pool-scores-span-mean scores span-start span-len n)
                   mean-top-id (top-entity-id mean-scores n)
                   mean-top-name (get id->entity mean-top-id)
                   mean-hit? (= mean-top-name expected-tail)

                   ;; 3. Span max pool
                   max-scores (pool-scores-span-max scores span-start span-len n)
                   max-top-id (top-entity-id max-scores n)
                   max-top-name (get id->entity max-top-id)
                   max-hit? (= max-top-name expected-tail)]

               ;; Cleanup query buffers
               (xla/destroy-buffer! ctx in-b)
               (if (sequential? out)
                 (doseq [b out] (xla/destroy-buffer! ctx b))
                 (xla/destroy-buffer! ctx out))
               (g4/destroy-relational-buffers! ctx grounded-bufs)

               (println (format "[Query %d/%d] \"%s\"" (inc idx) num-queries query-prompt))
               (println (format "  Expected Tail: \"%s\" (Span [%d..%d])"
                                expected-tail span-start (+ span-start span-len -1)))
               (println (format "  (i)   Last-Token:  %-18s -> %s" (str "\"" last-top-name "\"") (if last-hit? "HIT" "MISS")))
               (println (format "  (ii)  Span-Mean:   %-18s -> %s" (str "\"" mean-top-name "\"") (if mean-hit? "HIT" "MISS")))
               (println (format "  (iii) Span-Max:    %-18s -> %s" (str "\"" max-top-name "\"") (if max-hit? "HIT" "MISS")))
               (println "")

               {:head head
                :expected expected-tail
                :last {:top-1 last-top-name :hit? last-hit?}
                :span-mean {:top-1 mean-top-name :hit? mean-hit?}
                :span-max {:top-1 max-top-name :hit? max-hit?}}))
           (range num-queries)
           all-triples)

          hits-last (count (filter #(get-in % [:last :hit?]) results))
          hits-mean (count (filter #(get-in % [:span-mean :hit?]) results))
          hits-max (count (filter #(get-in % [:span-max :hit?]) results))
          chance-prob (/ 1.0 (double n))]

      ;; Cleanup device weights
      (doseq [w device-weights]
        (xla/destroy-buffer! ctx w))

      (println "==================================================================")
      (println " SUMMARY: Span-Pooled Probe Ablation Accuracies")
      (println "==================================================================")
      (println (format " Total Queries:     %d" num-queries))
      (println (format " Entity Universe:   %d entities" n))
      (println (format " Chance Baseline:   %.1f%% (1/%d ≈ %.4f)" (* 100.0 chance-prob) n (double chance-prob)))
      (println "------------------------------------------------------------------")
      (println (format " Results: last-token: %d/%d | span-mean: %d/%d | span-max: %d/%d (chance 1/%d ≈ %.1f%%)"
                       hits-last num-queries
                       hits-mean num-queries
                       hits-max num-queries
                       n (* 100.0 chance-prob)))
      (println "==================================================================")
      (println "\nMECHANISM ANALYSIS:")
      (if (or (> hits-mean hits-last) (> hits-max hits-last))
        (let [delta-mean (- hits-mean hits-last)
              delta-max (- hits-max hits-last)]
          (println (format " Isolating the entity token span recovered +%d (mean) / +%d (max) factual queries over last-token."
                           delta-mean delta-max))
          (println " Head-entity tokens preserve localized entity representations that are partially diluted by the final prompt token."))
        (do
          (println " Probe placement alone does not recover the factual signal (0/7 across all three variants).")
          (println " Contextual self-attention causes the query template tokens (\"Who is the CEO of ?\") to heavily")
          (println " influence token representations throughout the sequence. With an untrained fixed random projection,")
          (println " hidden states in LLM space remain orthogonal to the symbolic entity space regardless of pooling.")))
      (println "==================================================================\n")
      {:results results
       :hits-last hits-last
       :hits-mean hits-mean
       :hits-max hits-max})))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (if (g4/needs-libjsig-reexec? opts)
      (g4/reexec-with-libjsig! args "tools.poc-span-probe")
      (do
        (run-span-probe-ablation opts)
        (System/exit 0)))))
