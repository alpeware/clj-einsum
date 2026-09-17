(ns scripts.eval-gemma4-webnlg-relational
  "Grafts relational memory unbinding onto frozen Gemma 4 E2B contextual representations
   and evaluates the Pointwise Fact Selectivity vs. Neighborhood Booster hypothesis on WebNLG.
   Pre-trains W_mem and relation cores R_r 100% in-VRAM via OpenXLA PJRT, executes Gemma 4 forward
   passes to extract contextual hidden states h_normed in VRAM, and evaluates causal logit shifts
   against distractors across frequency tiers (Head, Mid, Tail, Unseen)."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.memory.contrastive :as contrastive]
            [clj-xla.logic.models.gemma :as gemma]
            [clj-xla.logic.symbolic :as sym]
            [clj-xla.pjrt :as pjrt]
            [clj-xla.tokenizer.protocol :as proto]
            [scripts.gemma4-inference :as g4]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :rocm
   :model-dir ".models/gemma-4-E2B-it"
   :train-file ".dataset/webnlg/train.edn"
   :test-seen ".dataset/webnlg/test_seen.edn"
   :test-unseen ".dataset/webnlg/test_unseen.edn"
   :checkpoint-out ".dataset/webnlg/checkpoint_gemma4_relational.edn"
   :epochs 5
   :lr 0.05
   :tau 0.1
   :lambda-mem 1.0
   :dim-mem 128
   :k-triples 16
   :max-eval 40
   :skip-train false})

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
          "--model-dir" (recur (subvec remaining 2) (assoc opts :model-dir v))
          "--train-file" (recur (subvec remaining 2) (assoc opts :train-file v))
          "--test-seen" (recur (subvec remaining 2) (assoc opts :test-seen v))
          "--test-unseen" (recur (subvec remaining 2) (assoc opts :test-unseen v))
          "--checkpoint-out" (recur (subvec remaining 2) (assoc opts :checkpoint-out v))
          "--epochs" (recur (subvec remaining 2) (assoc opts :epochs (Long/parseLong v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--tau" (recur (subvec remaining 2) (assoc opts :tau (Double/parseDouble v)))
          "--lambda-mem" (recur (subvec remaining 2) (assoc opts :lambda-mem (Double/parseDouble v)))
          "--dim-mem" (recur (subvec remaining 2) (assoc opts :dim-mem (Long/parseLong v)))
          "--k-triples" (recur (subvec remaining 2) (assoc opts :k-triples (Long/parseLong v)))
          "--max-eval" (recur (subvec remaining 2) (assoc opts :max-eval (Long/parseLong v)))
          "--skip-train" (recur (subvec remaining 1) (assoc opts :skip-train true))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; Evaluation Helper Pure Functions
;; ==============================================================================

(defn compute-rank
  "Computes 1-based rank of target value in array slice."
  [^floats logits target-idx vocab-size]
  (let [target-l (aget logits target-idx)]
    (loop [vi 0 rank 1]
      (if (>= vi vocab-size)
        rank
        (recur (inc vi) (if (> (aget logits vi) target-l) (inc rank) rank))))))

(defn compute-cand-rank
  "Computes 1-based rank of target value among candidate target entities."
  [^floats logits target-tok candidate-targets]
  (let [target-l (aget logits target-tok)]
    (loop [ci 0 rank 1]
      (if (>= ci (count candidate-targets))
        rank
        (let [act (:tok (nth candidate-targets ci))]
          (recur (inc ci) (if (> (aget logits act) target-l) (inc rank) rank)))))))

(defn top-candidate
  "Finds the candidate entity with the highest logit in candidate pool."
  [^floats logits candidate-targets]
  (apply max-key (fn [c] (aget logits (:tok c))) candidate-targets))

;; ==============================================================================
;; In-VRAM Contrastive Pre-Training on Gemma 4 Embeddings
;; ==============================================================================

(defn- sample-gaussian-floats [^java.util.Random rnd n scale]
  (let [^floats arr (float-array n)]
    (dotimes [i n]
      (aset arr i (float (* (.nextGaussian rnd) scale))))
    arr))

(defn- init-relation-core-floats [^java.util.Random rnd dm]
  (let [^floats arr (sample-gaussian-floats rnd (* dm dm) 0.01)]
    (dotimes [i dm]
      (let [idx (+ (* i dm) i)]
        (aset arr idx (float (+ (aget arr idx) 1.0)))))
    arr))

(defn train-gemma4-relational-cores!
  "Pre-trains W_mem and relation cores R_r on WebNLG triples using Gemma 4 resident embeddings."
  [ctx w-embed-buf train-triples rel-counts opts]
  (let [{:keys [epochs lr tau dim-mem k-triples]} opts
        v 262144
        din 1536
        dm (long dim-mem)
        k (long k-triples)
        rnd (java.util.Random. 42)
        _ (println (format "\nCompiling 100%% In-VRAM Contrastive Step Executable (V=%d, Din=%d, K=%d, Dm=%d)..."
                           v din k dm))
        t-c0 (System/nanoTime)
        step-exec (contrastive/compile-in-vram-contrastive-step ctx v din k dm {:dtype :bf16 :tau tau :lr lr :lambda-tl 0.3})
        t-c1 (System/nanoTime)
        _ (println (format "Compilation complete in %.2f ms." (/ (- t-c1 t-c0) 1e6)))

        ;; Initialize host parameter weights
        initial-w-mem (sample-gaussian-floats rnd (* din dm) 0.02)
        w-mem-shorts (g4/floats->bf16-shorts initial-w-mem)
        w-mem-buf-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [din dm] 13))

        ;; Triples by relation
        triples-by-rel (group-by :rel train-triples)
        relations (vec (sort-by #(get rel-counts % 0) > (keys triples-by-rel)))
        all-tails (vec (distinct (map :t-tok train-triples)))

        ;; Initialize relation cores on device
        r-bufs-atom (atom {})
        _ (doseq [r relations]
            (let [r-arr (init-relation-core-floats rnd dm)
                  r-shorts (g4/floats->bf16-shorts r-arr)
                  r-b (pjrt/buffer-from-host-buffer ctx (:client ctx) r-shorts [dm dm] 13)]
              (swap! r-bufs-atom assoc r r-b)))

        ;; Pre-allocate target identity and mask scale
        target-eye (let [^floats fa (float-array (* k k) 0.0)]
                     (dotimes [i k] (aset fa (+ (* i k) i) (float 1.0)))
                     fa)
        target-shorts (g4/floats->bf16-shorts target-eye)
        target-b (pjrt/buffer-from-host-buffer ctx (:client ctx) target-shorts [k k] 13)

        mask-scale (let [fa (float-array (* k k) (float (/ 1.0 (* (double k) (double tau)))))]
                     fa)
        mask-shorts (g4/floats->bf16-shorts mask-scale)
        mask-b (pjrt/buffer-from-host-buffer ctx (:client ctx) mask-shorts [k k] 13)]

    (println (format "Pre-training relational cores across %d relations for %d epochs in VRAM..."
                     (count relations) epochs))
    (dotimes [epoch epochs]
      (let [t-e0 (System/nanoTime)
            loss-sum (atom 0.0)
            step-count (atom 0)]
        (doseq [r relations]
          (let [r-triples (get triples-by-rel r)
                chunks (partition-all k r-triples)]
            (doseq [chunk chunks]
              (let [r-b (get @r-bufs-atom r)
                    actual-k (count chunk)
                    padded-chunk (if (< actual-k k)
                                   (let [n-pad (- k actual-k)
                                         pad-heads (take n-pad (cycle (map :h-tok chunk)))
                                         pad-tails (take n-pad (shuffle all-tails))]
                                     (into (vec chunk) (map (fn [h t] {:h-tok h :t-tok t}) pad-heads pad-tails)))
                                   (vec chunk))
                    ih-arr (int-array (map :h-tok padded-chunk))
                    it-arr (int-array (map :t-tok padded-chunk))
                    ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) ih-arr [k] 4)
                    it-b (pjrt/buffer-from-host-buffer ctx (:client ctx) it-arr [k] 4)
                    cur-w @w-mem-buf-atom
                    input-bufs [w-embed-buf ih-b it-b cur-w r-b target-b mask-b]
                    outs (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) input-bufs 4)
                    _ (xla/destroy-buffer! ctx ih-b)
                    _ (xla/destroy-buffer! ctx it-b)
                    w-new (nth outs 0)
                    r-new (nth outs 1)
                    p-buf (nth outs 2)
                    scores-buf (nth outs 3)
                    _ (xla/destroy-buffer! ctx scores-buf)
                    ;; Extract diagonal loss from P
                    p-floats (pjrt/buffer-to-host-buffer ctx p-buf (* k k) :bf16)
                    _ (xla/destroy-buffer! ctx p-buf)
                    batch-loss (loop [i 0 s 0.0]
                                 (if (>= i k)
                                   (/ s (double k))
                                   (let [prob (Math/max 1e-12 (double (aget p-floats (+ (* i k) i))))]
                                     (recur (inc i) (+ s (- (Math/log prob)))))))]
                ;; Update resident buffers
                (xla/destroy-buffer! ctx cur-w)
                (xla/destroy-buffer! ctx r-b)
                (reset! w-mem-buf-atom w-new)
                (swap! r-bufs-atom assoc r r-new)
                (swap! loss-sum + batch-loss)
                (swap! step-count inc)))))
        (let [t-e1 (System/nanoTime)
              avg-loss (/ @loss-sum (double (max 1 @step-count)))
              dur-ms (/ (- t-e1 t-e0) 1e6)]
          (println (format "  Epoch %d/%d | Steps: %,d | Mean InfoNCE Loss: %.4f | Latency: %.2f ms (%.2f ms/step)"
                           (inc epoch) epochs @step-count avg-loss dur-ms (/ dur-ms (double (max 1 @step-count))))))))

    (xla/destroy-buffer! ctx target-b)
    (xla/destroy-buffer! ctx mask-b)

    ;; Convert final resident buffers to host float arrays for inference session
    (let [w-buf @w-mem-buf-atom
          w-floats (pjrt/buffer-to-host-buffer ctx w-buf (* din dm) :bf16)
          _ (xla/destroy-buffer! ctx w-buf)
          r-maps (into {} (map (fn [[r b]]
                                 (let [fa (pjrt/buffer-to-host-buffer ctx b (* dm dm) :bf16)]
                                   (xla/destroy-buffer! ctx b)
                                   [r fa]))
                               @r-bufs-atom))]
      {:w-mem w-floats
       :r-maps r-maps})))

;; ==============================================================================
;; Main Evaluation Runner
;; ==============================================================================

(defn run-gemma4-relational-evaluation [opts]
  (let [{:keys [backend model-dir train-file test-seen test-unseen checkpoint-out
                lambda-mem dim-mem max-eval skip-train]} opts
        _ (println "\n================================================================================")
        _ (println "🧪 EXPERIMENT E15: GRAFTING RELATIONAL MEMORY ONTO FROZEN GEMMA 4 E2B")
        _ (println "================================================================================")

        ;; 1. Initialize Gemma 4 Session & Weights
        _ (println (format "Initializing Gemma 4 E2B session on backend [%s]..." (name backend)))
        session (g4/init-inference-session {:model-dir model-dir
                                            :backend backend
                                            :precision :bf16
                                            :max-seq-len 64
                                            :targets [:logits :normed_last]})
        ctx (:ctx session)
        tokenizer (:tokenizer session)
        config (:config session)
        vocab-size (long (:vocab-size config))
        hidden-dim (long (:hidden-dim config))
        dm (long dim-mem)
        dev-weights (g4/allocate-device-weights session)
        w-embed-buf (first dev-weights)

        ;; 2. Load and Prepare WebNLG Training Data
        _ (println (format "\nLoading WebNLG training triples from %s..." train-file))
        train-entries (:entries (read-string (slurp (io/file train-file))))
        raw-triples (distinct (mapcat :triples train-entries))
        rel-counts (frequencies (map second raw-triples))
        valid-train-triples
        (vec (keep (fn [[h r t]]
                     (let [h-toks (proto/encode tokenizer (str " " h) false)
                           t-toks (proto/encode tokenizer (str " " t) false)]
                       (when (and (seq h-toks) (seq t-toks))
                         {:h-str h :rel r :t-str t
                          :h-tok (first h-toks)
                          :t-tok (first t-toks)})))
                   raw-triples))
        _ (println (format "Processed %,d valid training triples across %d relations."
                           (count valid-train-triples) (count rel-counts)))

        ;; 3. Train or Load Relational Cores
        ckpt-f (io/file checkpoint-out)
        {:keys [w-mem r-maps]}
        (if (and skip-train (.exists ckpt-f))
          (do (println (format "Loading cached relational weights from %s..." checkpoint-out))
              (read-string (slurp ckpt-f)))
          (let [trained (train-gemma4-relational-cores! ctx w-embed-buf valid-train-triples rel-counts opts)]
            (when checkpoint-out
              (println (format "Saving trained relational weights to %s..." checkpoint-out))
              (spit ckpt-f (pr-str trained)))
            trained))

        ;; 4. Compile Gemma 4 Forward Executable (targets: [:logits :normed_last])
        _ (println "\nCompiling Gemma 4 Forward Executable with dual targets [:logits :normed_last]...")
        fwd-exec (g4/compile-tensor-logic-executable
                  (assoc session :opts {:targets [:logits :normed_last] :last-token-only? true}) 64)

        ;; 5. Compile Grounded Logit Unbinding Executables
        _ (println "Compiling In-VRAM Relational Logit Grounding Executables...")
        unbinding-invars [[:h [:tensor [1 1 hidden-dim] :bf16]]
                          [:logits_base [:tensor [1 1 vocab-size] :bf16]]
                          [:W_mem [:tensor [hidden-dim dm] :bf16]]
                          [:R [:tensor [dm dm] :bf16]]
                          [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]]
        unbinding-ast (gemma/gemma4-unbinding-ast hidden-dim dm vocab-size lambda-mem)
        unbinding-exec (sym/compile-query ctx "gemma_unbinding_logits" unbinding-invars unbinding-ast
                                          [:logits_grounded :delta_logits])

        entity-invars [[:i_h [:tensor [1 1] :i32]]
                       [:logits_base [:tensor [1 1 vocab-size] :bf16]]
                       [:W_mem [:tensor [hidden-dim dm] :bf16]]
                       [:R [:tensor [dm dm] :bf16]]
                       [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]]
        entity-ast (gemma/gemma4-entity-unbinding-ast hidden-dim dm vocab-size lambda-mem)
        entity-exec (sym/compile-query ctx "gemma_entity_unbinding_logits" entity-invars entity-ast
                                       [:logits_grounded :delta_logits])

        ;; Allocate W_mem buffer on device
        w-mem-shorts (g4/floats->bf16-shorts ^floats w-mem)
        w-mem-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [hidden-dim dm] 13)

        ;; 6. Build Candidate Target Entity Pool from Evaluation Sets
        load-eval-triples (fn [path]
                            (let [f (io/file path)]
                              (when (.exists f)
                                (distinct (mapcat :triples (:entries (read-string (slurp f))))))))
        seen-triples (load-eval-triples test-seen)
        unseen-triples (load-eval-triples test-unseen)
        all-eval-targets (distinct (concat (map #(nth % 2) seen-triples)
                                           (map #(nth % 2) unseen-triples)
                                           (map :t-str (take 100 valid-train-triples))))
        candidate-targets (vec (distinct (keep (fn [t]
                                                 (let [toks (proto/encode tokenizer (str " " t) false)]
                                                   (when (seq toks)
                                                     {:str t :tok (first toks)})))
                                               all-eval-targets)))
        _ (println (format "Formed candidate target evaluation pool of %,d entities." (count candidate-targets)))

        ;; Evaluation helper function for a split
        eval-split-fn
        (fn [split-name test-path]
          (println "\n================================================================================")
          (println (format "🔍 EVALUATING SPLIT: %s (%s)" split-name test-path))
          (println "================================================================================")
          (let [triples (load-eval-triples test-path)
                cloze-prompts
                (vec (keep (fn [[h rel t]]
                             (let [t-toks (proto/encode tokenizer (str " " t) false)
                                   h-toks (proto/encode tokenizer (str " " h) false)]
                               (when (and (seq t-toks) (seq h-toks))
                                 (let [cnt (get rel-counts rel 0)
                                       tier (cond
                                              (zero? cnt) :unseen
                                              (>= cnt 50) :head
                                              (>= cnt 10) :mid
                                              :else :tail)]
                                   {:prompt (str "The " (name rel) " of " h " is")
                                    :head h
                                    :head-tok (first h-toks)
                                    :rel rel
                                    :target t
                                    :target-tok (first t-toks)
                                    :train-count cnt
                                    :tier tier}))))
                           triples))
                selected-evals (vec (take (long max-eval) (distinct cloze-prompts)))]
            (println (format "Formulated %,d valid cloze test prompts (evaluating top %d):\n"
                             (count cloze-prompts) (count selected-evals)))
            (println "Prompt (truncated)             | Target          | Tier   | Base Top-1     | Active Top-1   | Δ_target | Mean Δ_dist | Max Δ_dist | Pointwise? | Neigh?")
            (println "-------------------------------+-----------------+--------+----------------+----------------+----------+-------------+------------+------------+-------")

            (let [results
                  (mapv
                   (fn [ep]
                     (let [{:keys [prompt head-tok rel target-tok tier target]} ep
                           ;; 1. Prepare prompt tokens
                           raw-ids (proto/encode tokenizer prompt false)
                           prompt-ids (vec (cons (proto/bos-id tokenizer) raw-ids))
                           p-len (count prompt-ids)
                           in-arr (int-array 64)
                           _ (dotimes [i p-len] (aset in-arr i (int (nth prompt-ids i))))
                           pos-p (int-array [(dec p-len)])
                           in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 64] 4)
                           pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                           step-inputs (into [in-b pos-b] dev-weights)

                           ;; 2. Gemma 4 forward pass (returns [logits normed_last])
                           fwd-outs (pjrt/execute-executable ctx (or (:handle fwd-exec) fwd-exec) step-inputs 2)
                           _ (xla/destroy-buffer! ctx in-b)
                           _ (xla/destroy-buffer! ctx pos-b)
                           base-logits-b (nth fwd-outs 0)
                           normed-last-b (nth fwd-outs 1)

                           ;; Read base logits to host
                           base-logits-fa (pjrt/buffer-to-host-buffer ctx base-logits-b vocab-size :bf16)
                           base-top-cand (top-candidate base-logits-fa candidate-targets)
                           base-top-str (:str base-top-cand)
                           base-top-tok (:tok base-top-cand)
                           base-cand-rank (compute-cand-rank base-logits-fa target-tok candidate-targets)
                           base-vocab-rank (compute-rank base-logits-fa target-tok vocab-size)

                           ;; 3. Prepare Relational Core R_rel
                           r-floats (get r-maps rel (float-array (* dm dm) 0.0))
                           r-shorts (g4/floats->bf16-shorts r-floats)
                           r-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) r-shorts [dm dm] 13)

                           ;; 4. Run Contextual Unbinding Executable
                           ground-outs (pjrt/execute-executable ctx (or (:handle unbinding-exec) unbinding-exec)
                                                                [normed-last-b base-logits-b w-mem-dev r-dev w-embed-buf] 2)
                           logits-grounded-b (nth ground-outs 0)
                           delta-logits-b (nth ground-outs 1)

                           grounded-fa (pjrt/buffer-to-host-buffer ctx logits-grounded-b vocab-size :bf16)
                           delta-fa (pjrt/buffer-to-host-buffer ctx delta-logits-b vocab-size :bf16)

                           ;; 5. Run Direct Entity Unbinding Executable (for comparison)
                           ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (int-array [head-tok]) [1 1] 4)
                           entity-outs (pjrt/execute-executable ctx (or (:handle entity-exec) entity-exec)
                                                                [ih-b base-logits-b w-mem-dev r-dev w-embed-buf] 2)
                           _ (xla/destroy-buffer! ctx ih-b)
                           _ (xla/destroy-buffer! ctx (nth entity-outs 0))
                           delta-ent-b (nth entity-outs 1)
                           delta-ent-fa (pjrt/buffer-to-host-buffer ctx delta-ent-b vocab-size :bf16)
                           _ (xla/destroy-buffer! ctx delta-ent-b)

                           _ (xla/destroy-buffer! ctx base-logits-b)
                           _ (xla/destroy-buffer! ctx normed-last-b)
                           _ (xla/destroy-buffer! ctx r-dev)
                           _ (xla/destroy-buffer! ctx logits-grounded-b)
                           _ (xla/destroy-buffer! ctx delta-logits-b)

                           ;; 6. Compute Active Ranks & Top Token
                           active-top-cand (top-candidate grounded-fa candidate-targets)
                           active-top-str (:str active-top-cand)
                           active-top-tok (:tok active-top-cand)
                           active-cand-rank (compute-cand-rank grounded-fa target-tok candidate-targets)
                           active-vocab-rank (compute-rank grounded-fa target-tok vocab-size)

                           ;; 7. Compute Distractor Metrics (Contextual)
                           delta-target (double (aget delta-fa target-tok))
                           distractor-cands (filterv #(not= (:tok %) target-tok) candidate-targets)
                           distractor-deltas (mapv #(double (aget delta-fa (:tok %))) distractor-cands)
                           mean-dist-delta (if (seq distractor-deltas)
                                             (/ (reduce + distractor-deltas) (double (count distractor-deltas)))
                                             0.0)
                           max-dist-delta (if (seq distractor-deltas)
                                            (apply max distractor-deltas)
                                            0.0)
                           top-base-cand-dist (apply max-key (fn [c] (aget base-logits-fa (:tok c))) distractor-cands)
                           top-base-delta (double (aget delta-fa (:tok top-base-cand-dist)))

                           pointwise? (> delta-target max-dist-delta)
                           neigh? (> delta-target mean-dist-delta)
                           causal-conversion? (and (not= base-top-tok target-tok) (= active-top-tok target-tok))

                           ;; Direct entity unbinding metrics
                           delta-ent-target (double (aget delta-ent-fa target-tok))
                           distractor-ent-deltas (mapv #(double (aget delta-ent-fa (:tok %))) distractor-cands)
                           max-dist-ent-delta (if (seq distractor-ent-deltas) (apply max distractor-ent-deltas) 0.0)
                           mean-dist-ent-delta (if (seq distractor-ent-deltas) (/ (reduce + distractor-ent-deltas) (double (count distractor-ent-deltas))) 0.0)
                           ent-pointwise? (> delta-ent-target max-dist-ent-delta)
                           ent-neigh? (> delta-ent-target mean-dist-ent-delta)

                           p-disp (if (> (count prompt) 30) (str (subs prompt 0 27) "...") prompt)
                           t-disp (if (> (count target) 15) (str (subs target 0 12) "...") target)
                           tier-str (case tier :head "HEAD" :mid "MID " :tail "TAIL" :unseen "UNSEEN")]

                       (println (format "%-30s | %-15s | %-6s | %-14s | %-14s | %+8.4f | %+11.4f | %+10.4f | %-10s | %-5s"
                                        p-disp t-disp tier-str
                                        (if (= base-top-tok target-tok) (str base-top-str " ✓") base-top-str)
                                        (if (= active-top-tok target-tok) (str active-top-str " ✓") active-top-str)
                                        delta-target mean-dist-delta max-dist-delta
                                        (if pointwise? "YES (Fact)" "NO (Noise)")
                                        (if neigh? "YES" "NO")))

                       {:prompt prompt
                        :target target
                        :rel rel
                        :tier tier
                        :base-match? (= base-top-tok target-tok)
                        :active-match? (= active-top-tok target-tok)
                        :causal-conversion? causal-conversion?
                        :base-cand-rank base-cand-rank
                        :active-cand-rank active-cand-rank
                        :base-vocab-rank base-vocab-rank
                        :active-vocab-rank active-vocab-rank
                        :delta-target delta-target
                        :mean-dist-delta mean-dist-delta
                        :max-dist-delta max-dist-delta
                        :top-base-delta top-base-delta
                        :pointwise? pointwise?
                        :neigh? neigh?
                        :ent-delta-target delta-ent-target
                        :ent-mean-dist mean-dist-ent-delta
                        :ent-max-dist max-dist-ent-delta
                        :ent-pointwise? ent-pointwise?
                        :ent-neigh? ent-neigh?}))
                   selected-evals)

                  total-n (count results)
                  n-pointwise (count (filter :pointwise? results))
                  n-neigh (count (filter :neigh? results))
                  n-ent-pointwise (count (filter :ent-pointwise? results))
                  n-ent-neigh (count (filter :ent-neigh? results))
                  base-acc (* 100.0 (/ (double (count (filter :base-match? results))) (double total-n)))
                  active-acc (* 100.0 (/ (double (count (filter :active-match? results))) (double total-n)))
                  mean-tgt (if (pos? total-n) (/ (reduce + (map :delta-target results)) (double total-n)) 0.0)
                  mean-dist (if (pos? total-n) (/ (reduce + (map :mean-dist-delta results)) (double total-n)) 0.0)
                  mean-max-dist (if (pos? total-n) (/ (reduce + (map :max-dist-delta results)) (double total-n)) 0.0)
                  mean-top-base (if (pos? total-n) (/ (reduce + (map :top-base-delta results)) (double total-n)) 0.0)
                  mean-ent-tgt (if (pos? total-n) (/ (reduce + (map :ent-delta-target results)) (double total-n)) 0.0)
                  mean-ent-dist (if (pos? total-n) (/ (reduce + (map :ent-mean-dist results)) (double total-n)) 0.0)
                  mean-ent-max (if (pos? total-n) (/ (reduce + (map :ent-max-dist results)) (double total-n)) 0.0)]

              (println "\n================================================================================")
              (println (format "🎯 GEMMA 4 DISTRACTOR SELECTIVITY DIAGNOSTIC: %s (N=%d)" split-name total-n))
              (println "================================================================================")
              (println (format "Top-1 Accuracy Baseline (Zero R_mem)  : %.1f%% (%d/%d)"
                               base-acc (count (filter :base-match? results)) total-n))
              (println (format "Top-1 Accuracy Grounded (Active R_mem): %.1f%% (%d/%d) [Causal Lift: %+d]"
                               active-acc (count (filter :active-match? results)) total-n
                               (- (count (filter :active-match? results)) (count (filter :base-match? results)))))
              (println (format "Contextual Pointwise Selectivity (Δ_tgt > max Δ_dist): %d/%d (%.1f%%)"
                               n-pointwise total-n (* 100.0 (/ (double n-pointwise) (double total-n)))))
              (println (format "Contextual Neighborhood Selectivity (Δ_tgt > mean Δ): %d/%d (%.1f%%)"
                               n-neigh total-n (* 100.0 (/ (double n-neigh) (double total-n)))))
              (println (format "Direct Entity Pointwise Selectivity (Δ_tgt > max Δ_dist): %d/%d (%.1f%%)"
                               n-ent-pointwise total-n (* 100.0 (/ (double n-ent-pointwise) (double total-n)))))
              (println (format "Direct Entity Neighborhood Selectivity (Δ_tgt > mean Δ): %d/%d (%.1f%%)"
                               n-ent-neigh total-n (* 100.0 (/ (double n-ent-neigh) (double total-n)))))
              (println (format "Contextual Shifts : Target Δ: %+6.4f | Mean Dist Δ: %+6.4f | Max Dist Δ: %+6.4f | Top-Base Δ: %+6.4f"
                               mean-tgt mean-dist mean-max-dist mean-top-base))
              (println (format "Direct Entity     : Target Δ: %+6.4f | Mean Dist Δ: %+6.4f | Max Dist Δ: %+6.4f"
                               mean-ent-tgt mean-ent-dist mean-ent-max))
              (println "--------------------------------------------------------------------------------")
              (println "Frequency Tier     | Eval | Base Acc | Act Acc | Δ_target | Mean Δ_dist | Max Δ_dist | Target > Mean? | Target > Max? (Pointwise)")
              (println "-------------------+------+----------+---------+----------+-------------+------------+----------------+--------------------------")
              (doseq [t [:head :mid :tail :unseen]]
                (let [t-res (filter #(= (:tier %) t) results)
                      tn (count t-res)]
                  (when (pos? tn)
                    (let [t-base (* 100.0 (/ (double (count (filter :base-match? t-res))) (double tn)))
                          t-act (* 100.0 (/ (double (count (filter :active-match? t-res))) (double tn)))
                          t-tgt (/ (reduce + (map :delta-target t-res)) (double tn))
                          t-mdist (/ (reduce + (map :mean-dist-delta t-res)) (double tn))
                          t-maxdist (/ (reduce + (map :max-dist-delta t-res)) (double tn))
                          t-neigh (* 100.0 (/ (double (count (filter :neigh? t-res))) (double tn)))
                          t-point (* 100.0 (/ (double (count (filter :pointwise? t-res))) (double tn)))
                          label (case t :head "Head (>= 50)      " :mid "Mid (10 - 49)     " :tail "Tail (< 10)       " :unseen "Unseen (0 Core)   ")]
                      (println (format "%s | %-4d | %5.1f%%   | %5.1f%%  | %+8.4f | %+11.4f | %+10.4f |     %5.1f%%    |          %5.1f%%"
                                       label tn t-base t-act t-tgt t-mdist t-maxdist t-neigh t-point))))))
              (println "================================================================================\n")
              results)))

        ;; Run evaluation on test_seen.edn and test_unseen.edn
        seen-res (eval-split-fn "TEST_SEEN (Held-out triples from trained relations)" test-seen)
        unseen-res (eval-split-fn "TEST_UNSEEN (Triples from zero-shot unseen relations)" test-unseen)
        _ (xla/destroy-buffer! ctx w-mem-dev)]
    {:seen seen-res
     :unseen unseen-res}))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-gemma4-relational-evaluation opts)
    (System/exit 0)))
