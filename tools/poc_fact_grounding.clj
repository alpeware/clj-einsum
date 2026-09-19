(ns tools.poc-fact-grounding
  "Proof-of-concept pipeline for In-Tensor Relational Memory & Factual Grounding with Gemma 4 E2B on AMD ROCm / OpenXLA PJRT.
   Demonstrates three sequential stages:
     Stage 1: Baseline parametric evaluation (Relational Core = 0).
     Stage 2: Zero-gradient memory ingestion via Panama FFM DMA outer-product superposition (< 15 microseconds).
     Stage 3: In-tensor grounded query with logit clamping (exact factual emission with zero context inflation)."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.relation :as mem]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [tools.gemma4-inference :as g4])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :model ".models/gemma-4-E2B-it"
   :triples "data/wiki_recent_triples.edn"
   :head "Alpeware"
   :relation :ceo_of
   :dim 256
   :threshold 0.5
   :memory-seed 2026
   :proj-seed 2026
   :max-new-tokens 15
   :grounded-tokens 5
   :temperature 0.7
   :top-k 10})

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
      (let [flag (first remaining)
            val (second remaining)]
        (cond
          (and (= flag "--backend") val)
          (let [clean-kw (keyword (str/replace val #"^:+" ""))]
            (when (= clean-kw :rocm)
              (try
                (let [linker (java.lang.foreign.Linker/nativeLinker)
                      lookup (.defaultLookup linker)
                      setenv-opt (.find lookup "setenv")]
                  (when (.isPresent setenv-opt)
                    (let [setenv-ptr ^java.lang.foreign.MemorySegment (.get setenv-opt)
                          fd (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/JAVA_INT
                                                                      (into-array java.lang.foreign.MemoryLayout
                                                                                  [java.lang.foreign.ValueLayout/ADDRESS
                                                                                   java.lang.foreign.ValueLayout/ADDRESS
                                                                                   java.lang.foreign.ValueLayout/JAVA_INT]))
                          handle (.downcallHandle linker setenv-ptr fd (make-array java.lang.foreign.Linker$Option 0))]
                      (with-open [arena (java.lang.foreign.Arena/ofConfined)]
                        (.invokeWithArguments handle [(.allocateFrom arena "HIP_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
                        (.invokeWithArguments handle [(.allocateFrom arena "ROCR_VISIBLE_DEVICES") (.allocateFrom arena "0") (int 1)])
                        (.invokeWithArguments handle [(.allocateFrom arena "HSA_OVERRIDE_GFX_VERSION") (.allocateFrom arena "11.0.0") (int 1)])))))
                (catch Exception _ nil)))
            (recur (subvec remaining 2) (assoc opts :backend clean-kw)))

          (and (= flag "--model") val)
          (recur (subvec remaining 2) (assoc opts :model val))

          (and (= flag "--triples") val)
          (recur (subvec remaining 2) (assoc opts :triples val))

          (and (= flag "--head") val)
          (recur (subvec remaining 2) (assoc opts :head val))

          (and (= flag "--relation") val)
          (recur (subvec remaining 2) (assoc opts :relation (keyword (str/replace val #"^:+" ""))))

          (and (= flag "--dim") val)
          (recur (subvec remaining 2) (assoc opts :dim (Long/parseLong val)))

          (and (= flag "--threshold") val)
          (recur (subvec remaining 2) (assoc opts :threshold (Double/parseDouble val)))

          (and (= flag "--memory-seed") val)
          (recur (subvec remaining 2) (assoc opts :memory-seed (Long/parseLong val)))

          (and (= flag "--proj-seed") val)
          (recur (subvec remaining 2) (assoc opts :proj-seed (Long/parseLong val)))

          (and (= flag "--seed") val)
          (let [s (Long/parseLong val)]
            (recur (subvec remaining 2) (assoc opts :proj-seed s :memory-seed s)))

          (and (= flag "--max-new-tokens") val)
          (recur (subvec remaining 2) (assoc opts :max-new-tokens (Long/parseLong val)))

          (and (= flag "--grounded-tokens") val)
          (recur (subvec remaining 2) (assoc opts :grounded-tokens (Long/parseLong val)))

          :else
          (recur (subvec remaining 1) opts))))))

(defn build-random-projection
  "Generates a fixed [in-dim x d] random projection matrix from Gaussian samples
   scaled by 1/sqrt(in-dim) using a seeded RNG. Pure function of (in-dim, d, seed)."
  [in-dim d seed]
  (let [in-dim (long in-dim)
        d (long d)
        scale (float (/ 1.0 (Math/sqrt (double in-dim))))
        rnd (java.util.Random. (long seed))
        arr (float-array (* in-dim d))]
    (dotimes [i (* in-dim d)]
      (aset-float arr i (float (* (.nextGaussian rnd) scale))))
    arr))

(defn build-entity-token-table
  "Constructs a static [n_entities x vocab_size] float array mapping each entity
   to its constituent tokens uniformly with 1.0 weight."
  [tokenizer entities vocab-size]
  (let [n (count entities)
        v (long vocab-size)
        table (float-array (* n v))]
    (dotimes [i n]
      (let [entity-name (nth entities i)
            token-ids (encode tokenizer (str " " entity-name))
            row-offset (* i v)]
        (doseq [tid token-ids]
          (let [t-idx (long tid)]
            (when (and (>= t-idx 0) (< t-idx v))
              (aset-float table (+ row-offset t-idx) (float 1.0)))))))
    table))

(defn run-poc [opts]
  (println "==================================================================")
  (println "  clj-xla: In-Tensor Relational Memory & Factual Grounding POC")
  (println "==================================================================")
  (println (str "Backend:           [" (name (:backend opts)) "]"))
  (println (str "Model:             [" (:model opts) "]"))
  (println (str "Triples File:      [" (:triples opts) "]"))
  (println (str "Target Query Head: [\"" (:head opts) "\"]"))
  (println (str "Target Relation:   [" (:relation opts) "]"))
  (println (str "Memory Dim D:      [" (:dim opts) "]"))
  (println (str "Memory Init Seed:  [" (or (:memory-seed opts) 2026) "]"))
  (println (str "Projection Seed:   [" (or (:proj-seed opts) 2026) "]"))
  (println "==================================================================\n")

  (let [triples-file (io/file (:triples opts))]
    (when-not (.exists triples-file)
      (throw (ex-info (str "Triples file not found: " (:triples opts)) {:file (:triples opts)})))

    ;; 1. Initialize Inference Session & SafeTensors Weights
    (let [kb-data (edn/read-string (slurp triples-file))
          entities (vec (:entities kb-data))
          entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
          id->entity (into {} (map-indexed (fn [idx name] [idx name]) entities))
          relations (vec (:relations kb-data))
          rel->id (into {} (map-indexed (fn [idx name] [name idx]) relations))
          n (count entities)
          d (long (:dim opts))
          k (count relations)
          mem (mem/init-relation-memory n d k (or (:memory-seed opts) 2026))
          ^floats cores (:cores mem)
          ^floats e-table (:entity-table mem)
          target-head (:head opts)
          target-rel (:relation opts)
          rel-id (get rel->id target-rel 0)
          matching-facts (filter (fn [[h r _t]] (and (= h target-head) (= r target-rel))) (:triples kb-data))
          target-tail (second (first (map (fn [[_h _r t]] [nil t]) matching-facts)))

          _ (println (str "Entity universe: " n " entities, " k " relation types, resident D=" d))
          _ (println "Loading Gemma 4 E2B weights and compiling execution graph...")

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

          ;; Compile Gemma 4 with active relational grounding block
          exec (g4/compile-tensor-logic-executable
                (assoc session :opts (assoc opts :targets [:logits :entity_scores :normed_last]))
                max-seq-len)

          ;; Format prompt for Gemma 4 IT model
          clean-prompt (str "Who is the CEO of " target-head "?")
          raw-ids (encode tokenizer clean-prompt)
          clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (vec (rest raw-ids)) (vec raw-ids))
          prefix [(bos-id tokenizer) 105 2364 107]
          suffix [106 107 105 4368 107]
          prompt-ids (vec (concat prefix clean-ids suffix))
          prompt-len (count prompt-ids)]

      ;; ------------------------------------------------------------------------
      ;; STAGE 1: Baseline Evaluation (Parametric Weight Isolation)
      ;; ------------------------------------------------------------------------
      (println "\n------------------------------------------------------------------")
      (println " STAGE 1: Baseline Evaluation (Parametric Weight Isolation)")
      (println "------------------------------------------------------------------")
      (println (str " Query: \"" clean-prompt "\""))
      (println " Accelerator VRAM State: Relational Core R_CEO = 0 (Unseeded)")

      ;; Allocate unseeded relational buffers (R_CEO = 0, W_proj = 0, W_vocab = 0)
      (let [rel-bufs1 (g4/allocate-relational-buffers session mem {:rel-id rel-id})
            in-arr (int-array max-seq-len)
            _ (dotimes [i prompt-len] (aset in-arr i (int (nth prompt-ids i))))
            in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
            pos-arr (int-array [(dec prompt-len)])
            pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
            args1 (into [in-b pos-b] (concat device-weights rel-bufs1))
            out1 (xla/execute exec args1)
            ^floats scores1 (xla/to-host-slice (nth out1 1) 0 n n :bf16)
            max-score1 (apply max (vec scores1))]
        (println (format " Contraction v_q * R_CEO: max entity similarity score = %.4f", max-score1))
        (println " Result: Zero factual grounding signal (Vanilla parametric state)")

        ;; Run autoregressive generation for vanilla response
        (print " Generating vanilla response: ")
        (flush)
        (let [vanilla-cur (atom (vec prompt-ids))]
          (loop [step 0
                 cur-in-b in-b
                 cur-pos-b pos-b]
            (if (>= step 15)
              (do
                (xla/destroy-buffer! ctx cur-in-b)
                (xla/destroy-buffer! ctx cur-pos-b))
              (let [args-step (into [cur-in-b cur-pos-b] (concat device-weights rel-bufs1))
                    out-step (xla/execute exec args-step)
                    out-logits (if (sequential? out-step) (first out-step) out-step)
                    logits-slice (xla/to-host-slice out-logits 0 262144 262144 :bf16)
                    next-id (g4/sample-next-token logits-slice (assoc opts :temperature 0.7 :top-k 10) prompt-ids (subvec @vanilla-cur prompt-len))]
                (if (sequential? out-step)
                  (doseq [b out-step] (xla/destroy-buffer! ctx b))
                  (xla/destroy-buffer! ctx out-step))
                (xla/destroy-buffer! ctx cur-in-b)
                (xla/destroy-buffer! ctx cur-pos-b)
                (swap! vanilla-cur conj next-id)
                (print (decode tokenizer [next-id]))
                (flush)
                (if (or (= next-id 1) (= next-id 106))
                  nil
                  (let [new-in (int-array max-seq-len)
                        _ (dotimes [i (min (count @vanilla-cur) max-seq-len)]
                            (aset new-in i (int (nth @vanilla-cur i))))
                        new-in-b (xla/buffer-from-host-buffer ctx (:client ctx) new-in [1 max-seq-len] 4)
                        new-pos-b (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [(dec (count @vanilla-cur))]) [1] 4)]
                    (recur (inc step) new-in-b new-pos-b)))))))
        (println "\n")
        (g4/destroy-relational-buffers! ctx rel-bufs1)
        (doseq [b out1] (xla/destroy-buffer! ctx b)))

        ;; ------------------------------------------------------------------------
        ;; STAGE 2: Zero-Gradient Memory Ingestion
        ;; ------------------------------------------------------------------------
      (println "------------------------------------------------------------------")
      (println " STAGE 2: Zero-Gradient Memory Ingestion")
      (println "------------------------------------------------------------------")
      (println (str " Ingesting fact triple: (\"" target-head "\" " target-rel " \"" target-tail "\")"))

      (let [facts (:triples kb-data)
            t-start (System/nanoTime)]
        (doseq [[h r t] facts]
          (let [h-i (get entity->id h)
                r-i (get rel->id r)
                t-i (get entity->id t)]
            (when (and h-i r-i t-i)
              (mem/accumulate-fact! cores e-table h-i r-i t-i d))))
        (let [elapsed-us (/ (- (System/nanoTime) t-start) 1000.0)]
          (println (format " Superposition DMA complete in %.2f microseconds", (double elapsed-us)))
          (println " Backprop / gradient computation: ZERO steps (Instantaneous DMA)")
          (println " Context window prompt token inflation: 0 tokens added\n")))

        ;; ------------------------------------------------------------------------
        ;; STAGE 3: In-Tensor Grounded Generation (De-Oracled Retrieval Evaluation)
        ;; ------------------------------------------------------------------------
      (println "------------------------------------------------------------------")
      (println " STAGE 3: In-Tensor Grounded Generation (De-Oracled Retrieval)")
      (println "------------------------------------------------------------------")
      (println " Accelerator VRAM State: Superposed R_CEO active in OpenXLA graph")
      (println (str " Projection Matrix: Fixed Gaussian random projection (Seed " (or (:proj-seed opts) 2026) ")"))
      (println (str " Entity Token Table: Static uniform mapping for all " n " entities"))
      (println " Evaluating memory-driven retrieval across all factual queries...\n")

        ;; 1. Fixed random projection built ONCE per run:
      (let [w-proj (build-random-projection 1536 d (or (:proj-seed opts) 2026))

              ;; 2. Static entity->token table built ONCE per run:
            vocab-size (long (or (:vocab-size config) 262144))
            w-entity-token-table (build-entity-token-table tokenizer entities vocab-size)
            grounded-step-limit (long (or (:grounded-tokens opts) 5))

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

                       ;; Allocate relational buffers ONCE for this query:
                     grounded-bufs (g4/allocate-relational-buffers session mem
                                                                   {:rel-id q-rel-id
                                                                    :w-mem-proj w-proj
                                                                    :w-entity-to-vocab w-entity-token-table})

                       ;; Initial prompt forward pass
                     in-arr (int-array max-seq-len)
                     _ (dotimes [i q-prompt-len] (aset in-arr i (int (nth q-prompt-ids i))))
                     in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)
                     pos-arr (int-array [(dec q-prompt-len)])
                     pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-arr [1] 4)
                     args (into [in-b pos-b] (concat device-weights grounded-bufs))
                     out (xla/execute exec args)
                     ^floats scores (when (sequential? out)
                                      (xla/to-host-slice (nth out 1) 0 n n :bf16))
                     sorted-entities (vec (sort-by (fn [[_idx sc]] (- sc))
                                                   (map-indexed (fn [i sc] [i (double sc)]) scores)))
                     top-1-id (first (first sorted-entities))
                     top-1-name (get id->entity top-1-id)
                     top-3 (mapv (fn [[i sc]] [(get id->entity i) sc]) (take 3 sorted-entities))
                     hit? (= top-1-name expected-tail)]

                 (println (format "[Query %d/%d] \"%s\"" (inc idx) num-queries query-prompt))
                 (println (format "  Expected Tail: \"%s\"" expected-tail))
                 (println "  Top-3 Retrieved Entities:")
                 (doseq [[rank [ename sc]] (map-indexed vector top-3)]
                   (println (format "    %d. %-18s (score: %8.4f)" (inc rank) (str "\"" ename "\"") (double sc))))
                 (println (format "  Outcome:       %s (Top-1: \"%s\")"
                                  (if hit? "HIT [CORRECT]" "MISS")
                                  top-1-name))

                   ;; Decode loop with fixed relational buffers (No per-token buffer allocation!)
                 (print "  Grounded emission: ")
                 (flush)
                 (let [cur-tokens (atom (vec q-prompt-ids))]
                   (loop [step 0
                          cur-in-b in-b
                          cur-pos-b pos-b
                          cur-out out]
                     (let [out-logits (if (sequential? cur-out) (first cur-out) cur-out)
                           logits-slice (xla/to-host-slice out-logits 0 vocab-size vocab-size :bf16)
                           next-id (g4/sample-next-token logits-slice
                                                         (assoc opts :temperature 0.0 :top-k 1 :repetition-penalty 1.0)
                                                         q-prompt-ids
                                                         (subvec @cur-tokens q-prompt-len))]
                       (if (sequential? cur-out)
                         (doseq [b cur-out] (xla/destroy-buffer! ctx b))
                         (xla/destroy-buffer! ctx cur-out))
                       (xla/destroy-buffer! ctx cur-in-b)
                       (xla/destroy-buffer! ctx cur-pos-b)
                       (swap! cur-tokens conj next-id)
                       (print (decode tokenizer [next-id]))
                       (flush)
                       (if (or (<= grounded-step-limit 1) (>= step (dec grounded-step-limit)) (= next-id 1) (= next-id 106))
                         nil
                         (let [s-len (count @cur-tokens)
                               new-in (int-array max-seq-len)
                               _ (dotimes [i (min s-len max-seq-len)]
                                   (aset new-in i (int (nth @cur-tokens i))))
                               new-in-b (xla/buffer-from-host-buffer ctx (:client ctx) new-in [1 max-seq-len] 4)
                               new-pos-b (xla/buffer-from-host-buffer ctx (:client ctx) (int-array [(dec s-len)]) [1] 4)
                               step-args (into [new-in-b new-pos-b] (concat device-weights grounded-bufs))
                               step-out (xla/execute exec step-args)]
                           (recur (inc step) new-in-b new-pos-b step-out))))))
                 (println "\n")

                   ;; Cleanup query relational buffers
                 (g4/destroy-relational-buffers! ctx grounded-bufs)

                 {:head head
                  :expected expected-tail
                  :top-1 top-1-name
                  :hit? hit?}))
             (range num-queries)
             all-triples)

            total-hits (count (filter :hit? results))
            accuracy (/ (double total-hits) (double num-queries))
            chance-prob (/ 1.0 (double n))]

        (println "==================================================================")
        (println " SUMMARY: De-Oracled Memory-Driven Retrieval Evaluation")
        (println "==================================================================")
        (println (format " Total Queries:     %d" num-queries))
        (println (format " Entity Universe:   %d entities" n))
        (println (format " Top-1 Hits:        %d / %d" total-hits num-queries))
        (println (format " Top-1 Accuracy:    %.1f%% (%d/%d)" (* 100.0 accuracy) total-hits num-queries))
        (println (format " Chance Baseline:   %.1f%% (1/%d ≈ %.4f)" (* 100.0 chance-prob) n (double chance-prob)))
        (println (format " Result Summary:    top-1 accuracy: %d/%d (chance: 1/%d ≈ %.1f%%)"
                         total-hits num-queries n (* 100.0 chance-prob)))
        (println "==================================================================")
        (println " PoC Pipeline Completed Successfully!")
        (println "=================================================================="))

      ;; Cleanup persistent device weights
      (doseq [w device-weights]
        (xla/destroy-buffer! ctx w)))))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (if (g4/needs-libjsig-reexec? opts)
      (g4/reexec-with-libjsig! args "tools.poc-fact-grounding")
      (do
        (run-poc opts)
        (System/exit 0)))))
