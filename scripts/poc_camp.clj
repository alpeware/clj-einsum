(ns scripts.poc-camp
  "Experiment E1: Cross-Attention Memory Probe (CAMP) Driver Script.
   Evaluates whether a single-layer learned cross-attention query token can overcome
   prompt template dominance by learning to attend to head entity spans across prompt tokens.
   Runs Gemma 4 forward passes to collect sequence hidden states H in [1, L, 1536],
   trains CAMP parameters with exact symbolic adjoints, inspects token-level attention distributions,
   and benchmarks 7-fold Leave-One-Out Cross-Validation."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.memory.camp :as camp]
            [clj-xla.logic.memory.relation :as mem]
            [clj-xla.tokenizer.protocol :refer [bos-id decode encode]]
            [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scripts.gemma4-inference :as g4]
            [scripts.poc-anchored-memory :as anc-base]
            [scripts.poc-fact-grounding :as poc-base]
            [scripts.poc-span-probe :as span-base])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :model ".models/gemma-4-E2B-it"
   :triples "data/wiki_recent_triples.edn"
   :dim 1536
   :entity-mode :anchored-qr
   :lr 0.05
   :steps 300
   :tau 1.0})

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
          "--entity-mode" (recur (subvec remaining 2) (assoc opts :entity-mode (keyword v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--steps" (recur (subvec remaining 2) (assoc opts :steps (Long/parseLong v)))
          "--tau" (recur (subvec remaining 2) (assoc opts :tau (Double/parseDouble v)))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; 1. Sequence Hidden States Extraction from Gemma 4 Forward Pass
;; ==============================================================================

(defn extract-sequence-hidden-states
  "Executes Gemma 4 in sequence mode (:last-token-only? false) with :targets [:normed],
   extracting the full prompt contextual hidden state tensor H [L_prompt, 1536] for each query."
  [session exec device-weights kb-data _opts max-seq-len]
  (let [ctx (:ctx session)
        tokenizer (:tokenizer session)
        entities (vec (:entities kb-data))
        entity->id (into {} (map-indexed (fn [idx name] [name idx]) entities))
        triples (:triples kb-data)
        num-queries (count triples)
        hidden-dim 1536]
    (println (format "Extracting sequence representations H [L, %d] across %d queries..."
                     hidden-dim num-queries))
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
             tail-id (get entity->id expected-tail)
             [span-start span-len] (span-base/locate-head-span tokenizer q-prompt-ids head-clean)

             ;; Decode individual token strings for display
             token-strs (mapv (fn [tid] (decode tokenizer [tid])) q-prompt-ids)

             in-arr (int-array max-seq-len)
             _ (dotimes [i q-prompt-len] (aset in-arr i (int (nth q-prompt-ids i))))
             in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 max-seq-len] 4)

             args (into [in-b] device-weights)
             out (xla/execute exec args)
             out-buf (if (sequential? out) (first out) out)

             ;; Slice out only the valid prompt tokens [q-prompt-len, hidden-dim]
             ^floats full-h (xla/to-host-slice out-buf 0 (* max-seq-len hidden-dim) (* max-seq-len hidden-dim) :bf16)
             H (float-array (* q-prompt-len hidden-dim))
             _ (System/arraycopy full-h 0 H 0 (* q-prompt-len hidden-dim))

             ;; Cleanup step buffers
             _ (xla/destroy-buffer! ctx in-b)
             _ (if (sequential? out)
                 (doseq [b out] (xla/destroy-buffer! ctx b))
                 (xla/destroy-buffer! ctx out))

             prefix-len (count prefix)
             q-len (count clean-q-ids)
             q-mask (mapv (fn [t] (and (>= t prefix-len) (< t (+ prefix-len q-len)))) (range q-prompt-len))

             ;; Normalize H across token rows to prevent attention sink norm dominance
             H-normed (camp/normalize-sequence-h H q-prompt-len hidden-dim)]

         (println (format "  [%d/%d] \"%s\" -> L=%d tokens, Head span [%d..%d)"
                          (inc idx) num-queries query-prompt q-prompt-len span-start (+ span-start span-len)))

         {:idx idx
          :head head-clean
          :rel rel
          :expected expected-tail
          :y tail-id
          :seq-len q-prompt-len
          :H H-normed
          :mask q-mask
          :token-strs token-strs
          :span-start span-start
          :span-len span-len}))
     (range num-queries)
     triples)))

;; ==============================================================================
;; 2. Attention Breakdown & Diagnostics
;; ==============================================================================

(defn analyze-attention-distribution
  "Computes what percentage of attention weight lands on:
     - Head Entity span [span-start, span-start + span-len)
     - Template prefix / suffix scaffolding tokens"
  [^floats alpha span-start span-len l-cnt]
  (let [start (long span-start)
        slen (long span-len)
        l (long l-cnt)
        head-attn (loop [t start s 0.0]
                    (if (>= t (+ start slen))
                      s
                      (recur (inc t) (+ s (double (aget alpha t))))))
        total-attn (loop [t 0 s 0.0]
                     (if (>= t l)
                       s
                       (recur (inc t) (+ s (double (aget alpha t))))))
        template-attn (- total-attn head-attn)]
    {:head-attn head-attn
     :template-attn template-attn
     :head-pct (* 100.0 (/ head-attn (Math/max 1e-12 total-attn)))}))

;; ==============================================================================
;; 3. Experiment Runner
;; ==============================================================================

(defn run-camp-experiment [opts]
  (println "==================================================================")
  (println "  clj-xla: Experiment E1 - Cross-Attention Memory Probe (CAMP)")
  (println "==================================================================")
  (println (str "Backend:           [" (:backend opts) "]"))
  (println (str "Model:             [" (:model opts) "]"))
  (println (str "Triples File:      [" (:triples opts) "]"))
  (println (str "Memory Dim D:      [" (:dim opts) "]"))
  (println (str "Entity Mode:       [" (:entity-mode opts) "]"))
  (println (str "Learning Rate:     [" (:lr opts) "]"))
  (println (str "Steps:             [" (:steps opts) "]"))
  (println (str "Attention Tau:     [" (:tau opts) "]"))
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
          hidden-dim 1536

          ;; Build relation memory table and relation cores
          max-seq-len 48
          base-session (g4/init-inference-session (assoc opts :max-seq-len max-seq-len))
          _ctx (:ctx base-session)
          tokenizer (:tokenizer base-session)
          weights-mmap (:weights-mmap base-session)
          prefix-base (:prefix-base base-session)

          mem (if (= (:entity-mode opts) :anchored-qr)
                (do
                  (println "Building LLM-anchored QR-orthonormalized table (D=1536)...")
                  (let [raw (anc-base/build-llm-anchored-table weights-mmap prefix-base tokenizer entities d k)]
                    (anc-base/init-qr-anchored-memory raw)))
                (do
                  (println (str "Building Random normalized table (D=" d ")..."))
                  (mem/init-relation-memory n d k (or (:memory-seed opts) 2026))))

          _ (doseq [[h r t] (:triples kb-data)]
              (let [h-i (get entity->id h)
                    r-i (get rel->id r)
                    t-i (get entity->id t)]
                (when (and h-i r-i t-i)
                  (mem/accumulate-fact! (:cores mem) (:entity-table mem) h-i r-i t-i d))))

          ;; Extract active relation core R (CEO_Of)
          rel-idx 0
          R-core (let [arr (float-array (* d d))]
                   (System/arraycopy ^floats (:cores mem) (int (* rel-idx d d)) arr 0 (int (* d d)))
                   arr)
          E-table ^floats (:entity-table mem)

          ;; Configure Gemma 4 in sequence mode with :targets [:normed] (NO pos buffer)
          config (assoc (:config base-session)
                        :max-seq-len max-seq-len
                        :last-token-only? false)
          session (assoc base-session :config config)
          device-weights (g4/allocate-device-weights session)
          exec (g4/compile-tensor-logic-executable
                (assoc session :opts (assoc opts :last-token-only? false :targets [:normed]))
                max-seq-len)

          ;; Extract contextual sequence hidden states H
          dataset (extract-sequence-hidden-states session exec device-weights kb-data opts max-seq-len)

          ;; Initial parameters for CAMP:
          ;; k_attn initialized with small random values
          rng (java.util.Random. 2026)
          k-attn-init (let [arr (float-array hidden-dim)]
                        (dotimes [i hidden-dim]
                          (aset-float arr i (float (* 0.02 (- (.nextDouble rng) 0.5)))))
                        arr)
          ;; W initialized to identity if d == hidden-dim, or random projection
          W-init (if (= d hidden-dim)
                   (anc-base/build-identity-matrix d)
                   (poc-base/build-random-projection hidden-dim d 2026))

          tau (double (or (:tau opts) 1.0))
          cfg {:hidden-dim hidden-dim
               :d-mem d
               :n-entities n
               :tau tau
               :R R-core
               :E E-table}
          train-opts {:lr (:lr opts)
                      :steps (:steps opts)
                      :on-step (fn [step loss acc]
                                 (when (or (zero? step) (zero? (mod step 50)))
                                   (println (format "  Step %3d: Loss = %7.4f, Batch Accuracy = %5.1f%%"
                                                    step (double loss) (* 100.0 (double acc))))))}]

      ;; ------------------------------------------------------------------------
      ;; STAGE 1: Full-Batch Memorization & Attention Pattern Inspection
      ;; ------------------------------------------------------------------------
      (println "\n------------------------------------------------------------------")
      (println " STAGE 1: Full Dataset Training & Attention Distribution Analysis")
      (println "------------------------------------------------------------------")
      (let [trained (camp/train-camp dataset {:k-attn (aclone k-attn-init) :W (aclone W-init)} cfg train-opts)
            final-k (:k-attn trained)
            final-W (:W trained)]
        (println (format "\nTraining Complete: Initial Loss = %.4f -> Final Loss = %.4f, Accuracy = %.1f%%"
                         (double (:initial-loss trained)) (double (:final-loss trained))
                         (* 100.0 (double (:final-accuracy trained)))))

        (println "\nLearned Attention Weight Breakdown Across All Queries:")
        (doseq [{:keys [idx head expected seq-len H token-strs span-start span-len]} dataset]
          (let [fwd (camp/forward-camp-single H seq-len hidden-dim final-k final-W R-core E-table d n tau)
                ^floats alpha (:alpha fwd)
                attn-stats (analyze-attention-distribution alpha span-start span-len seq-len)
                pred-y (loop [i 1 best-i 0 best-sc (double (aget ^floats (:p fwd) 0))]
                         (if (>= i n)
                           best-i
                           (let [sc (double (aget ^floats (:p fwd) i))]
                             (if (> sc best-sc)
                               (recur (inc i) i sc)
                               (recur (inc i) best-i best-sc)))))
                pred-name (get id->entity pred-y)
                hit? (= pred-name expected)
                top-tokens (vec (sort-by (fn [[_t _tok wt]] (- wt))
                                         (map-indexed (fn [t tok] [t tok (double (aget alpha t))]) token-strs)))]
            (println (format " [Query %d] Head: %-15s | Target: %-15s | Pred: %-15s -> %s"
                             (inc idx) (str "\"" head "\"") (str "\"" expected "\"") (str "\"" pred-name "\"")
                             (if hit? "HIT [CORRECT]" "MISS")))
            (println (format "   Head Entity Attention: %5.1f%% | Template Scaffolding: %5.1f%%"
                             (:head-pct attn-stats) (- 100.0 (:head-pct attn-stats))))
            (println (format "   Top-3 Attended Tokens: %s"
                             (str/join ", " (map (fn [[_t tok wt]] (format "\"%s\": %.3f" (str/trim tok) wt)) (take 3 top-tokens))))))))

      ;; ------------------------------------------------------------------------
      ;; STAGE 2: 7-fold Leave-One-Out Cross-Validation (LOO-CV)
      ;; ------------------------------------------------------------------------
      (println "\n------------------------------------------------------------------")
      (println " STAGE 2: 7-Fold Leave-One-Out Cross-Validation (Honest Generalization)")
      (println "------------------------------------------------------------------")
      (let [loo-res (camp/run-camp-leave-one-out-cv dataset {:k-attn (aclone k-attn-init) :W (aclone W-init)} cfg train-opts)]
        (doseq [{:keys [test-idx target predicted hit? test-loss test-alpha]} (:fold-results loo-res)]
          (let [sample (nth dataset test-idx)
                head (:head sample)
                expected (get id->entity target)
                pred-name (get id->entity predicted)
                attn-stats (analyze-attention-distribution test-alpha (:span-start sample) (:span-len sample) (:seq-len sample))]
            (println (format " Fold %d: Head: %-15s | Expected: %-15s | Pred: %-15s -> %s (Loss: %.4f, Head Attn: %.1f%%)"
                             (inc test-idx) (str "\"" head "\"") (str "\"" expected "\"") (str "\"" pred-name "\"")
                             (if hit? "HIT [CORRECT]" "MISS") (double test-loss) (:head-pct attn-stats)))))

        (println "\n==================================================================")
        (println "  EXPERIMENT E1 COMPARATIVE SUMMARY")
        (println "==================================================================")
        (println " Baseline (Random Table + Random W):        0 / 7 (  0.0%)")
        (println " Task A (Span-Mean Pooling):                 0 / 7 (  0.0%)")
        (println " Task B (Learned Linear Probe W):            0 / 7 (  0.0%) [100% Train, 0% LOO]")
        (println " Task C (LLM-Anchored Raw W=I):              0 / 7 (  0.0%)")
        (println " Task D (QR-Orthonormalized W=I):            1 / 7 ( 14.3%)")
        (println (format " Experiment E1 (Cross-Attention Probe):      %d / 7 (%5.1f%%) [LOO-CV]"
                         (:hits loo-res) (* 100.0 (double (:mean-acc loo-res)))))
        (println "==================================================================\n")

        loo-res))))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-camp-experiment opts)
    (shutdown-agents)
    (System/exit 0)))
