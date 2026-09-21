(ns tools.eval-e18-resolver-sweep
  "Experiment E18 Phase 1: 2x2 Factorial Sweep over Sampler Hygiene (Factor A)
   and Query Distribution Alignment (Factor B) across multiple random seeds.
   Evaluates against 80 WebNLG cloze queries and 747 candidate entities on ROCm via OpenXLA PJRT."
  (:require [einsum.core :as xla]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.runtime.tokenizer.protocol :as proto]
            [einsum.logic.symbolic :as sym]
            [models.gemma :as gemma]
            [einsum.logic.memory.contrastive :as contrastive]
            [einsum.logic.memory.sampler :as sampler]
            [tools.gemma4-inference :as g4]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.nio ByteBuffer ByteOrder]
           [java.nio.channels FileChannel]
           [java.nio.file Path StandardOpenOption]
           [java.util Random]))

(defn init-rocm-c-env! []
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

(defn- sample-gaussian-floats [^Random rnd n scale]
  (let [^floats arr (float-array n)]
    (dotimes [i n]
      (aset arr i (float (* (.nextGaussian rnd) scale))))
    arr))

(defn top-candidate [^floats logits candidate-targets]
  (apply max-key (fn [c] (aget logits (:tok c))) candidate-targets))

(defn load-hidden-state-cache [bin-path total-shorts]
  (let [buf (ByteBuffer/allocateDirect (* total-shorts 2))]
    (.order buf ByteOrder/LITTLE_ENDIAN)
    (with-open [fc (FileChannel/open (Path/of bin-path (into-array String []))
                                     (into-array [StandardOpenOption/READ]))]
      (.read fc buf))
    (.flip buf)
    (let [^shorts sa (short-array total-shorts)]
      (dotimes [i total-shorts]
        (aset-short sa i (.getShort buf)))
      sa)))

(defn- mean [coll]
  (if (seq coll)
    (/ (reduce + coll) (double (count coll)))
    0.0))

(defn- std-dev [coll]
  (if (> (count coll) 1)
    (let [m (mean coll)
          variance (/ (reduce + (map #(Math/pow (- (double %) m) 2) coll))
                      (double (dec (count coll))))]
      (Math/sqrt variance))
    0.0))

(defn run-e18-sweep! [opts]
  (init-rocm-c-env!)
  (let [model-dir (or (:model-dir opts) ".models/gemma-4-E2B-it")
        train-meta-file (or (:train-meta-file opts) ".dataset/webnlg/train_h_ctx_meta.edn")
        cache-bin-file (or (:cache-bin-file opts) ".dataset/webnlg/train_h_ctx_bf16.bin")
        test-seen (or (:test-seen opts) ".dataset/webnlg/test_seen.edn")
        test-unseen (or (:test-unseen opts) ".dataset/webnlg/test_unseen.edn")
        checkpoint-stage1 (or (:checkpoint-stage1 opts) ".dataset/webnlg/checkpoint_gemma4_relational.edn")
        out-edn (or (:out-edn opts) "catalog/trainable-limits/e18-diagnostics/phase1_results.edn")
        out-csv (or (:out-csv opts) "catalog/trainable-limits/e18-diagnostics/summary.csv")
        backend (or (:backend opts) :rocm)
        seeds (or (:seeds opts) [42 43 44])
        epochs (long (or (:epochs opts) 5))
        k-triples 16
        tau 0.1
        lr 0.05
        lambda-mem 0.3
        lambda-resolve 1.0
        dim-mem 128
        max-eval 40]

    (println "================================================================================")
    (println "🔬 EXPERIMENT E18 PHASE 1: RESOLVER 2x2 FACTORIAL SWEEP")
    (println "================================================================================")
    (println "Factors: [A0 (Replication) vs A1 (Hygienic)] x [B0 (Token Embed) vs B1 (Contextual Hidden State)]")
    (println (format "Seeds: %s | Epochs: %d | K: %d | Tau: %.2f | LR: %.2f | Dm: %d"
                     (pr-str seeds) epochs k-triples tau lr dim-mem))

    ;; 1. Initialize Inference Session & Allocations
    (println "\n[1/6] Initializing Gemma 4 E2B session on ROCm...")
    (let [session (g4/init-inference-session {:model-dir model-dir
                                              :backend backend
                                              :precision :bf16
                                              :max-seq-len 64
                                              :targets [:logits :normed_last]})
          ctx (:ctx session)
          tokenizer (:tokenizer session)
          config (:config session)
          vocab-size (long (:vocab-size config))
          hidden-dim (long (:hidden-dim config))
          din hidden-dim
          dm (long dim-mem)
          dev-weights (g4/allocate-device-weights session)
          w-embed-buf (first dev-weights)

          ;; 2. Load Stage 1 Checkpoint
          _ (println (format "\n[2/6] Loading frozen Stage 1 relational weights from %s..." checkpoint-stage1))
          ckpt (read-string (slurp checkpoint-stage1))
          w-mem-floats (float-array (:w-mem ckpt))
          w-mem-shorts (g4/floats->bf16-shorts w-mem-floats)
          w-mem-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [din dm] 13)
          r-maps (into {} (map (fn [[k v]] [k (float-array v)]) (:r-maps ckpt)))
          r-dev-map (into {} (keep (fn [[r fa]]
                                     (when (seq fa)
                                       [r (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                        (g4/floats->bf16-shorts fa)
                                                                        [dm dm] 13)]))
                                   r-maps))
          zero-r-buf (pjrt/buffer-from-host-buffer ctx (:client ctx) (short-array (* dm dm) (short 0)) [dm dm] 13)

          ;; 3. Load Training Triples & Hidden State Cache
          _ (println "\n[3/6] Loading training triples and contextual hidden state cache...")
          train-meta (read-string (slurp train-meta-file))
          valid-train-triples (:triples train-meta)
          n-cache (long (:n-triples train-meta))
          total-cache-shorts (* n-cache din)
          _ (println (format "  Loading binary cache %s (%,d elements)..." cache-bin-file total-cache-shorts))
          h-cache-shorts (load-hidden-state-cache cache-bin-file total-cache-shorts)
          h-cache-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) h-cache-shorts [n-cache din] 13)
          by-rel (group-by :rel valid-train-triples)
          rel-counts (frequencies (map :rel valid-train-triples))
          known-triples-set (set (map (fn [tr] [(:h-tok tr) (:rel tr) (:t-tok tr)]) valid-train-triples))

          ;; 4. Build Candidate Pool and Cloze Prompts
          _ (println "\n[4/6] Formulating candidate pool and cloze evaluation prompts...")
          load-eval-triples (fn [path]
                              (distinct (mapcat :triples (:entries (read-string (slurp path))))))
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
          k-cands (count candidate-targets)
          cand-ids-arr (int-array (map :tok candidate-targets))
          cand-ids-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) cand-ids-arr [k-cands] 4)
          _ (println (format "  Evaluator pool: %,d candidate entities." k-cands))

          build-prompts (fn [triples]
                          (vec (take max-eval
                                     (distinct
                                      (keep (fn [[h rel t]]
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
                                            triples)))))
          eval-prompts-seen (build-prompts seen-triples)
          eval-prompts-unseen (build-prompts unseen-triples)
          _ (println (format "  Cloze prompts: %d test_seen, %d test_unseen."
                             (count eval-prompts-seen) (count eval-prompts-unseen)))

          ;; 5. Compile Executables
          _ (println "\n[5/6] Compiling PJRT executables (forward, Stage 1 unbinding, Stage 2 resolver)...")
          fwd-exec (g4/compile-tensor-logic-executable
                    (assoc session :opts {:targets [:logits :normed_last] :last-token-only? true}) 64)

          unbinding-invars [[:h [:tensor [1 1 hidden-dim] :bf16]]
                            [:logits_base [:tensor [1 1 vocab-size] :bf16]]
                            [:W_mem [:tensor [hidden-dim dm] :bf16]]
                            [:R [:tensor [dm dm] :bf16]]
                            [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]]
          unbinding-ast (gemma/gemma4-unbinding-ast hidden-dim dm vocab-size lambda-mem)
          unbinding-exec (sym/compile-query ctx "gemma_unbinding_logits" unbinding-invars unbinding-ast
                                            [:logits_grounded :delta_logits :u_norm])

          entity-invars [[:i_h [:tensor [1 1] :i32]]
                         [:logits_base [:tensor [1 1 vocab-size] :bf16]]
                         [:W_mem [:tensor [hidden-dim dm] :bf16]]
                         [:R [:tensor [dm dm] :bf16]]
                         [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]]
          entity-ast (gemma/gemma4-entity-unbinding-ast hidden-dim dm vocab-size lambda-mem)
          entity-exec (sym/compile-query ctx "gemma_entity_unbinding_logits" entity-invars entity-ast
                                         [:logits_grounded :delta_logits :u_norm])

          resolver-invars [[:h [:tensor [1 1 hidden-dim] :bf16]]
                           [:u_norm [:tensor [1 1 dm] :bf16]]
                           [:cand_ids [:tensor [k-cands] :i32]]
                           [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]
                           [:W_Q [:tensor [hidden-dim dm] :bf16]]
                           [:W_K [:tensor [hidden-dim dm] :bf16]]]
          resolver-ast (gemma/gemma4-two-stage-resolver-ast hidden-dim dm vocab-size k-cands)
          resolver-exec (sym/compile-query ctx "gemma_two_stage_resolver" resolver-invars resolver-ast [:scores])

          ent-resolver-invars [[:i_h [:tensor [1 1] :i32]]
                               [:u_norm [:tensor [1 1 dm] :bf16]]
                               [:cand_ids [:tensor [k-cands] :i32]]
                               [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]
                               [:W_Q [:tensor [hidden-dim dm] :bf16]]
                               [:W_K [:tensor [hidden-dim dm] :bf16]]]
          ent-resolver-ast (gemma/gemma4-two-stage-entity-resolver-ast hidden-dim dm vocab-size k-cands)
          ent-resolver-exec (sym/compile-query ctx "gemma_two_stage_entity_resolver" ent-resolver-invars ent-resolver-ast [:scores])

          step-exec-b0 (contrastive/compile-in-vram-resolver-step ctx vocab-size din k-triples dm
                                                                  {:dtype :bf16 :lr lr :tau tau :lambda-tl 1.0 :query-source :embed})
          step-exec-b1 (contrastive/compile-in-vram-resolver-step ctx vocab-size din k-triples dm
                                                                  {:dtype :bf16 :lr lr :tau tau :lambda-tl 1.0 :query-source :cache :n-cache n-cache})

          ;; Pre-compute and cache Gemma forward pass + Stage 1 unbinding for the 80 eval prompts
          _ (println "\n[5b/6] Pre-computing frozen backbone forward passes and Stage 1 unbindings...")
          precompute-eval-prompt
          (fn [ep]
            (let [{:keys [prompt head-tok rel target-tok]} ep
                  raw-ids (proto/encode tokenizer prompt false)
                  prompt-ids (vec (cons (proto/bos-id tokenizer) raw-ids))
                  p-len (count prompt-ids)
                  in-arr (int-array 64)
                  _ (dotimes [i p-len] (aset in-arr i (int (nth prompt-ids i))))
                  pos-p (int-array [(dec p-len)])
                  in-b (xla/buffer-from-host-buffer ctx (:client ctx) in-arr [1 64] 4)
                  pos-b (xla/buffer-from-host-buffer ctx (:client ctx) pos-p [1] 4)
                  step-inputs (into [in-b pos-b] dev-weights)
                  fwd-outs (pjrt/execute-executable ctx (or (:handle fwd-exec) fwd-exec) step-inputs 2)
                  _ (xla/destroy-buffer! ctx in-b)
                  _ (xla/destroy-buffer! ctx pos-b)
                  base-logits-b (nth fwd-outs 0)
                  normed-last-b (nth fwd-outs 1)
                  base-logits-fa (pjrt/buffer-to-host-buffer ctx base-logits-b vocab-size :bf16)
                  base-top-cand (top-candidate base-logits-fa candidate-targets)
                  target-ci (first (keep-indexed (fn [idx c] (when (= (:tok c) target-tok) idx)) candidate-targets))

                  r-buf (or (get r-dev-map rel) zero-r-buf)
                  ground-outs (pjrt/execute-executable ctx (or (:handle unbinding-exec) unbinding-exec)
                                                       [normed-last-b base-logits-b w-mem-dev r-buf w-embed-buf] 3)
                  logits-grounded-b (nth ground-outs 0)
                  delta-logits-b (nth ground-outs 1)
                  u-norm-b (nth ground-outs 2)
                  grounded-fa (pjrt/buffer-to-host-buffer ctx logits-grounded-b vocab-size :bf16)
                  delta-fa (pjrt/buffer-to-host-buffer ctx delta-logits-b vocab-size :bf16)

                  ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (int-array [head-tok]) [1 1] 4)
                  entity-outs (pjrt/execute-executable ctx (or (:handle entity-exec) entity-exec)
                                                       [ih-b base-logits-b w-mem-dev r-buf w-embed-buf] 3)
                  delta-ent-b (nth entity-outs 1)
                  ent-u-norm-b (nth entity-outs 2)
                  delta-ent-fa (pjrt/buffer-to-host-buffer ctx delta-ent-b vocab-size :bf16)

                  _ (xla/destroy-buffer! ctx (nth entity-outs 0))
                  _ (xla/destroy-buffer! ctx delta-ent-b)
                  _ (xla/destroy-buffer! ctx logits-grounded-b)
                  _ (xla/destroy-buffer! ctx delta-logits-b)
                  _ (xla/destroy-buffer! ctx base-logits-b)]
              (assoc ep
                     :normed-last-b normed-last-b
                     :u-norm-b u-norm-b
                     :ih-b ih-b
                     :ent-u-norm-b ent-u-norm-b
                     :base-logits-fa base-logits-fa
                     :base-top-cand base-top-cand
                     :target-ci target-ci
                     :grounded-fa grounded-fa
                     :delta-fa delta-fa
                     :delta-ent-fa delta-ent-fa)))

          cached-seen (mapv precompute-eval-prompt eval-prompts-seen)
          cached-unseen (mapv precompute-eval-prompt eval-prompts-unseen)
          _ (println "  Backbone states pre-computed for all 80 queries.")]

      ;; 6. Execute 2x2 Factorial Sweep across Seeds
      (println "\n================================================================================")
      (println "[6/6] EXECUTING 2x2 FACTORIAL SWEEP")
      (println "================================================================================")
      (let [cells [:A0B0 :A1B0 :A0B1 :A1B1]
            all-results (atom [])
            summary-rows (atom [])]

        (doseq [cell cells]
          (let [factor-a (if (or (= cell :A0B0) (= cell :A0B1)) :A0 :A1)
                factor-b (if (or (= cell :A0B0) (= cell :A1B0)) :B0 :B1)
                step-exec (if (= factor-b :B0) step-exec-b0 step-exec-b1)]
            (println (format "\n>>> RUNNING CELL: %s (Factor A: %s, Factor B: %s) <<<"
                             (name cell) (name factor-a) (name factor-b)))

            (doseq [seed seeds]
              (let [t-cell-0 (System/nanoTime)
                    rnd-init (Random. (long seed))
                    rnd-batch (Random. (long seed))

                    ;; Initialize W_Q and W_K
                    wq-floats (sample-gaussian-floats rnd-init (* din dm) 0.02)
                    wk-floats (sample-gaussian-floats rnd-init (* din dm) 0.02)
                    wq-shorts (g4/floats->bf16-shorts wq-floats)
                    wk-shorts (g4/floats->bf16-shorts wk-floats)
                    wq-dev-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) wq-shorts [din dm] 13))
                    wk-dev-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) wk-shorts [din dm] 13))

                    ;; Static Target & Mask for A0
                    static-target-mat (float-array (* k-triples k-triples) 0.0)
                    _ (dotimes [i k-triples] (aset static-target-mat (+ (* i k-triples) i) 1.0))
                    static-target-b (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                  (g4/floats->bf16-shorts static-target-mat)
                                                                  [k-triples k-triples] 13)
                    static-mask-mat (float-array (* k-triples k-triples) (float (/ 1.0 (* (double k-triples) tau))))
                    static-mask-b (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                (g4/floats->bf16-shorts static-mask-mat)
                                                                [k-triples k-triples] 13)
                    loss-history (atom [])]

                ;; Training Loop (5 Epochs)
                (dotimes [_epoch epochs]
                  (let [loss-sum (atom 0.0)
                        step-count (atom 0)]
                    (doseq [[r triples] by-rel]
                      (let [r-buf (get r-dev-map r)
                            n (count triples)
                            triples-vec (vec triples)]
                        (when (and r-buf (>= n 2))
                          (let [batch (if (= factor-a :A0)
                                        (mapv (fn [i] (nth triples-vec (mod i n))) (range k-triples))
                                        (sampler/sample-relation-batch triples-vec k-triples rnd-batch))
                                target-b (if (= factor-a :A0)
                                           static-target-b
                                           (let [t-arr (sampler/build-multi-positive-target batch k-triples)]
                                             (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                           (g4/floats->bf16-shorts t-arr)
                                                                           [k-triples k-triples] 13)))
                                mask-b (if (= factor-a :A0)
                                         static-mask-b
                                         (let [m-arr (sampler/build-hygienic-mask-scale batch known-triples-set k-triples tau)]
                                           (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                         (g4/floats->bf16-shorts m-arr)
                                                                         [k-triples k-triples] 13)))
                                it-arr (int-array k-triples)
                                _ (dotimes [i k-triples] (aset it-arr i (int (:t-tok (nth batch i)))))
                                it-b (pjrt/buffer-from-host-buffer ctx (:client ctx) it-arr [k-triples] 4)

                                inputs (if (= factor-b :B0)
                                         (let [ih-arr (int-array k-triples)
                                               _ (dotimes [i k-triples] (aset ih-arr i (int (:h-tok (nth batch i)))))
                                               ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) ih-arr [k-triples] 4)]
                                           [w-embed-buf ih-b it-b w-mem-dev r-buf @wq-dev-atom @wk-dev-atom target-b mask-b])
                                         (let [ic-arr (int-array k-triples)
                                               _ (dotimes [i k-triples] (aset ic-arr i (int (:idx (nth batch i)))))
                                               ic-b (pjrt/buffer-from-host-buffer ctx (:client ctx) ic-arr [k-triples] 4)]
                                           [h-cache-dev w-embed-buf ic-b it-b w-mem-dev r-buf @wq-dev-atom @wk-dev-atom target-b mask-b]))
                                outs (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) inputs 4)
                                wq-new (nth outs 0)
                                wk-new (nth outs 1)
                                p-buf (nth outs 2)
                                sc-buf (nth outs 3)
                                p-floats (pjrt/buffer-to-host-buffer ctx p-buf (* k-triples k-triples) :bf16)
                                batch-loss (loop [i 0 l 0.0]
                                             (if (>= i k-triples)
                                               (/ l (double k-triples))
                                               (recur (inc i) (- l (Math/log (Math/max 1e-6 (double (aget p-floats (+ (* i k-triples) i)))))))))]
                            (xla/destroy-buffer! ctx @wq-dev-atom)
                            (xla/destroy-buffer! ctx @wk-dev-atom)
                            (reset! wq-dev-atom wq-new)
                            (reset! wk-dev-atom wk-new)
                            (xla/destroy-buffer! ctx p-buf)
                            (xla/destroy-buffer! ctx sc-buf)
                            (xla/destroy-buffer! ctx (nth inputs (if (= factor-b :B0) 1 2))) ; ih-b or ic-b
                            (xla/destroy-buffer! ctx it-b)
                            (when (= factor-a :A1)
                              (xla/destroy-buffer! ctx target-b)
                              (xla/destroy-buffer! ctx mask-b))
                            (swap! loss-sum + batch-loss)
                            (swap! step-count inc)))))
                    (swap! loss-history conj (/ @loss-sum (double @step-count)))))
                (xla/destroy-buffer! ctx static-target-b)
                (xla/destroy-buffer! ctx static-mask-b)

                ;; Evaluation on Seen and Unseen Cloze Queries
                (let [eval-fn
                      (fn [cached-prompts]
                        (mapv
                         (fn [ep]
                           (let [{:keys [normed-last-b u-norm-b ih-b ent-u-norm-b
                                         base-logits-fa target-ci delta-fa delta-ent-fa target-tok]} ep

                                 ;; Contextual Resolver
                                 scores-b (pjrt/execute-executable ctx (or (:handle resolver-exec) resolver-exec)
                                                                   [normed-last-b u-norm-b cand-ids-dev w-embed-buf @wq-dev-atom @wk-dev-atom] 1)
                                 scores-fa (pjrt/buffer-to-host-buffer ctx scores-b k-cands :bf16)
                                 _ (xla/destroy-buffer! ctx scores-b)

                                 ;; Direct Entity Resolver
                                 ent-scores-b (pjrt/execute-executable ctx (or (:handle ent-resolver-exec) ent-resolver-exec)
                                                                       [ih-b ent-u-norm-b cand-ids-dev w-embed-buf @wq-dev-atom @wk-dev-atom] 1)
                                 ent-scores-fa (pjrt/buffer-to-host-buffer ctx ent-scores-b k-cands :bf16)
                                 _ (xla/destroy-buffer! ctx ent-scores-b)

                                 ;; Contextual Stage 1 + Stage 2 Metrics
                                 s1-delta-target (if target-ci (double (aget delta-fa target-tok)) 0.0)
                                 s1-dist-deltas (keep-indexed (fn [i c] (when (not= i target-ci) (double (aget delta-fa (:tok c))))) candidate-targets)
                                 s1-mean-dist (mean s1-dist-deltas)
                                 s1-max-dist (if (seq s1-dist-deltas) (apply max s1-dist-deltas) 0.0)
                                 s1-pointwise? (> s1-delta-target s1-max-dist)
                                 s1-neigh? (> s1-delta-target s1-mean-dist)

                                 s2-target-score (if target-ci (double (aget scores-fa target-ci)) 0.0)
                                 s2-delta-target (+ s1-delta-target (* lambda-resolve s2-target-score))
                                 s2-dist-deltas (keep-indexed (fn [i c]
                                                                (when (not= i target-ci)
                                                                  (+ (double (aget delta-fa (:tok c)))
                                                                     (* lambda-resolve (aget scores-fa i)))))
                                                              candidate-targets)
                                 s2-max-dist (if (seq s2-dist-deltas) (apply max s2-dist-deltas) 0.0)
                                 s2-mean-dist (mean s2-dist-deltas)
                                 s2-pointwise? (> s2-delta-target s2-max-dist)
                                 s2-neigh? (> s2-delta-target s2-mean-dist)

                                 s2-best-idx (apply max-key (fn [i]
                                                              (let [c (nth candidate-targets i)]
                                                                (+ (double (aget base-logits-fa (:tok c)))
                                                                   (double (aget delta-fa (:tok c)))
                                                                   (* lambda-resolve (aget scores-fa i)))))
                                                    (range k-cands))
                                 s2-top-tok (:tok (nth candidate-targets s2-best-idx))
                                 s2-top1? (= s2-top-tok target-tok)

                                 ;; Direct Entity Stage 2 Metrics
                                 ent-s1-delta-target (if target-ci (double (aget delta-ent-fa target-tok)) 0.0)
                                 ent-s1-dist-deltas (keep-indexed (fn [i c] (when (not= i target-ci) (double (aget delta-ent-fa (:tok c))))) candidate-targets)
                                 ent-s1-mean-dist (mean ent-s1-dist-deltas)
                                 ent-s1-max-dist (if (seq ent-s1-dist-deltas) (apply max ent-s1-dist-deltas) 0.0)
                                 ent-s1-pointwise? (> ent-s1-delta-target ent-s1-max-dist)

                                 ent-s2-target-score (if target-ci (double (aget ent-scores-fa target-ci)) 0.0)
                                 ent-s2-delta-target (+ ent-s1-delta-target (* lambda-resolve ent-s2-target-score))
                                 ent-s2-dist-deltas (keep-indexed (fn [i c]
                                                                    (when (not= i target-ci)
                                                                      (+ (double (aget delta-ent-fa (:tok c)))
                                                                         (* lambda-resolve (aget ent-scores-fa i)))))
                                                                  candidate-targets)
                                 ent-s2-max-dist (if (seq ent-s2-dist-deltas) (apply max ent-s2-dist-deltas) 0.0)
                                 ent-s2-mean-dist (mean ent-s2-dist-deltas)
                                 ent-s2-pointwise? (> ent-s2-delta-target ent-s2-max-dist)
                                 ent-s2-neigh? (> ent-s2-delta-target ent-s2-mean-dist)

                                 ent-s2-best-idx (apply max-key (fn [i]
                                                                  (let [c (nth candidate-targets i)]
                                                                    (+ (double (aget base-logits-fa (:tok c)))
                                                                       (double (aget delta-ent-fa (:tok c)))
                                                                       (* lambda-resolve (aget ent-scores-fa i)))))
                                                        (range k-cands))
                                 ent-s2-top-tok (:tok (nth candidate-targets ent-s2-best-idx))
                                 ent-s2-top1? (= ent-s2-top-tok target-tok)]
                             {:s1-top1? (= (:tok (:base-top-cand ep)) target-tok)
                              :s2-top1? s2-top1?
                              :s1-pointwise? s1-pointwise?
                              :s2-pointwise? s2-pointwise?
                              :s1-neigh? s1-neigh?
                              :s2-neigh? s2-neigh?
                              :s1-delta-target s1-delta-target
                              :s2-delta-target s2-delta-target
                              :s1-mean-dist s1-mean-dist
                              :s2-mean-dist s2-mean-dist
                              :s2-max-dist s2-max-dist
                              :distractor-boost (- s2-mean-dist s1-mean-dist)
                              :ent-s1-pointwise? ent-s1-pointwise?
                              :ent-s2-top1? ent-s2-top1?
                              :ent-s2-pointwise? ent-s2-pointwise?
                              :ent-s2-neigh? ent-s2-neigh?
                              :ent-distractor-boost (- ent-s2-mean-dist ent-s1-mean-dist)}))
                         cached-prompts))

                      aggregate-metrics
                      (fn [results]
                        {:s2-top1 (count (filter :s2-top1? results))
                         :s2-pointwise (count (filter :s2-pointwise? results))
                         :s2-neigh (count (filter :s2-neigh? results))
                         :mean-target-delta (mean (map :s2-delta-target results))
                         :mean-dist-delta (mean (map :s2-mean-dist results))
                         :max-dist-delta (mean (map :s2-max-dist results))
                         :distractor-boost (mean (map :distractor-boost results))
                         :ent-s2-top1 (count (filter :ent-s2-top1? results))
                         :ent-s2-pointwise (count (filter :ent-s2-pointwise? results))
                         :ent-s2-neigh (count (filter :ent-s2-neigh? results))
                         :ent-distractor-boost (mean (map :ent-distractor-boost results))})

                      seen-eval (aggregate-metrics (eval-fn cached-seen))
                      unseen-eval (aggregate-metrics (eval-fn cached-unseen))
                      elapsed-s (/ (- (System/nanoTime) t-cell-0) 1e9)]

                  (xla/destroy-buffer! ctx @wq-dev-atom)
                  (xla/destroy-buffer! ctx @wk-dev-atom)

                  (println (format "  Seed %d [%.1fs]: Final Loss=%.4f | Seen: Top1=%d/40, Pointwise=%d/40, Neigh=%d/40, DistBoost=%+.4f | Unseen: Top1=%d/40, Pointwise=%d/40"
                                   seed elapsed-s (last @loss-history)
                                   (:s2-top1 seen-eval) (:s2-pointwise seen-eval) (:s2-neigh seen-eval) (:distractor-boost seen-eval)
                                   (:s2-top1 unseen-eval) (:s2-pointwise unseen-eval)))

                  (let [run-record {:cell cell
                                    :factor-a factor-a
                                    :factor-b factor-b
                                    :seed seed
                                    :loss-history @loss-history
                                    :seen seen-eval
                                    :unseen unseen-eval
                                    :elapsed-s elapsed-s}]
                    (swap! all-results conj run-record)
                    (swap! summary-rows conj
                           {:cell (name cell)
                            :seed seed
                            :loss (last @loss-history)
                            :seen-top1 (:s2-top1 seen-eval)
                            :seen-pointwise (:s2-pointwise seen-eval)
                            :seen-neigh (:s2-neigh seen-eval)
                            :seen-dist-boost (:distractor-boost seen-eval)
                            :unseen-top1 (:s2-top1 unseen-eval)
                            :unseen-pointwise (:s2-pointwise unseen-eval)
                            :unseen-neigh (:s2-neigh unseen-eval)
                            :unseen-dist-boost (:distractor-boost unseen-eval)})))))))

        ;; Clean up cached prompt device buffers
        (doseq [ep cached-seen]
          (xla/destroy-buffer! ctx (:normed-last-b ep))
          (xla/destroy-buffer! ctx (:u-norm-b ep))
          (xla/destroy-buffer! ctx (:ih-b ep))
          (xla/destroy-buffer! ctx (:ent-u-norm-b ep)))
        (doseq [ep cached-unseen]
          (xla/destroy-buffer! ctx (:normed-last-b ep))
          (xla/destroy-buffer! ctx (:u-norm-b ep))
          (xla/destroy-buffer! ctx (:ih-b ep))
          (xla/destroy-buffer! ctx (:ent-u-norm-b ep)))

        ;; 7. Summarize and Print Table
        (println "\n================================================================================")
        (println "📊 EXPERIMENT E18 PHASE 1: 2x2 FACTORIAL SUMMARY")
        (println "================================================================================")
        (println "Cell   | Seed | Loss   | Seen Top-1 | Seen Point | Seen Neigh | Seen Boost | Unseen Top-1 | Unseen Point | Unseen Boost")
        (println "-------+------+--------+------------+------------+------------+------------+--------------+--------------+-------------")
        (doseq [r @summary-rows]
          (println (format "%-6s | %-4d | %-6.4f | %2d/40      | %2d/40      | %2d/40      | %-+10.4f | %2d/40       | %2d/40       | %-+10.4f"
                           (:cell r) (:seed r) (:loss r)
                           (:seen-top1 r) (:seen-pointwise r) (:seen-neigh r) (:seen-dist-boost r)
                           (:unseen-top1 r) (:unseen-pointwise r) (:unseen-dist-boost r))))

        ;; Aggregate across seeds per cell
        (println "\n--------------------------------------------------------------------------------")
        (println "CELL AGGREGATES (Mean ± Std across 3 seeds):")
        (println "--------------------------------------------------------------------------------")
        (println "Cell   | Description                               | Seen Top-1   | Seen Pointwise | Seen Distractor Boost | Unseen Pointwise")
        (println "-------+-------------------------------------------+--------------+----------------+-----------------------+-----------------")
        (let [by-cell (group-by :cell @all-results)
              cell-stats
              (into {}
                    (map (fn [c]
                           (let [runs (get by-cell c)
                                 seen-t1 (map (comp :s2-top1 :seen) runs)
                                 seen-pw (map (comp :s2-pointwise :seen) runs)
                                 seen-db (map (comp :distractor-boost :seen) runs)
                                 unseen-pw (map (comp :s2-pointwise :unseen) runs)
                                 desc (case c
                                        :A0B0 "A0B0 (E16 Replication: Cycling + Embeds)"
                                        :A1B0 "A1B0 (Hygienic + Embeds)"
                                        :A0B1 "A0B1 (Cycling + Contextual Cache)"
                                        :A1B1 "A1B1 (Hygienic + Contextual Cache)")]
                             (println (format "%-6s | %-41s | %4.1f ± %-4.1f | %4.1f ± %-4.1f | %+10.4f ± %-6.4f   | %4.1f ± %-4.1f"
                                              (name c) desc
                                              (mean seen-t1) (std-dev seen-t1)
                                              (mean seen-pw) (std-dev seen-pw)
                                              (mean seen-db) (std-dev seen-db)
                                              (mean unseen-pw) (std-dev unseen-pw)))
                             [c {:desc desc
                                 :seen-top1 {:mean (mean seen-t1) :std (std-dev seen-t1)}
                                 :seen-pointwise {:mean (mean seen-pw) :std (std-dev seen-pw)}
                                 :seen-distractor-boost {:mean (mean seen-db) :std (std-dev seen-db)}
                                 :unseen-pointwise {:mean (mean unseen-pw) :std (std-dev unseen-pw)}}]))
                         cells))]

          ;; Write JSON/EDN
          (.mkdirs (.getParentFile (io/file out-edn)))
          (spit out-edn (pr-str {:metadata {:date "2026-09-17"
                                            :experiment :e18-phase1
                                            :backend backend
                                            :k 16 :epochs epochs :tau tau :lr lr
                                            :seeds seeds}
                                 :cell-stats cell-stats
                                 :individual-runs @all-results}))
          (println (format "\nSaved full results EDN to %s" out-edn))

          ;; Write CSV
          (.mkdirs (.getParentFile (io/file out-csv)))
          (let [csv-header "cell,seed,loss,seen_top1,seen_pointwise,seen_neigh,seen_dist_boost,unseen_top1,unseen_pointwise,unseen_neigh,unseen_dist_boost\n"
                csv-lines (map (fn [r]
                                 (str/join "," [(:cell r) (:seed r) (format "%.4f" (:loss r))
                                                (:seen-top1 r) (:seen-pointwise r) (:seen-neigh r) (format "%.4f" (:seen-dist-boost r))
                                                (:unseen-top1 r) (:unseen-pointwise r) (:unseen-neigh r) (format "%.4f" (:unseen-dist-boost r))]))
                               @summary-rows)
                csv-content (str csv-header (str/join "\n" csv-lines) "\n")]
            (spit out-csv csv-content)
            (println (format "Saved summary CSV to %s" out-csv))))))))

(defn -main [& _args]
  (run-e18-sweep! {:backend :rocm})
  (System/exit 0))
