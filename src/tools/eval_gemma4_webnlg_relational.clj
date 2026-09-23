(ns tools.eval-gemma4-webnlg-relational
  "Grafts relational memory unbinding onto frozen Gemma 4 E2B contextual representations
   and evaluates the Pointwise Fact Selectivity vs. Neighborhood Booster hypothesis on WebNLG.
   Pre-trains W_mem, relation cores R_r, and Stage 2 Non-Linear Resolver (W_Q, W_K) 100% in-VRAM
   via OpenXLA PJRT, executes Gemma 4 forward passes in VRAM, and evaluates causal logit shifts
   against distractors across frequency tiers (Head, Mid, Tail, Unseen)."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.contrastive :as contrastive]
            [einsum.models.gemma :as gemma]
            [einsum.logic.symbolic :as sym]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.runtime.tokenizer.protocol :as proto]
            [tools.gemma4-inference :as g4]
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
   :epochs-resolver 5
   :lr 0.05
   :lr-resolver 0.05
   :tau 0.1
   :lambda-mem 1.0
   :lambda-resolve 1.0
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
          "--epochs-resolver" (recur (subvec remaining 2) (assoc opts :epochs-resolver (Long/parseLong v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--lr-resolver" (recur (subvec remaining 2) (assoc opts :lr-resolver (Double/parseDouble v)))
          "--tau" (recur (subvec remaining 2) (assoc opts :tau (Double/parseDouble v)))
          "--lambda-mem" (recur (subvec remaining 2) (assoc opts :lambda-mem (Double/parseDouble v)))
          "--lambda-resolve" (recur (subvec remaining 2) (assoc opts :lambda-resolve (Double/parseDouble v)))
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
  [ctx w-embed-buf valid-train-triples rel-counts opts]
  (let [{:keys [epochs lr tau dim-mem k-triples]} opts
        v 262144
        din 1536
        dm (long dim-mem)
        k (long k-triples)
        rnd (java.util.Random. 42)
        _ (println (format "\nCompiling 100%% In-VRAM Contrastive Step Executable (V=%d, Din=%d, K=%d, Dm=%d)..."
                           v din k dm))
        step-exec (contrastive/compile-in-vram-contrastive-step ctx v din k dm
                                                                {:dtype :bf16
                                                                 :lr lr
                                                                 :tau tau
                                                                 :lambda-tl 1.0})

        ;; 1. Initialize W_mem resident in VRAM
        w-mem-floats (sample-gaussian-floats rnd (* din dm) 0.02)
        w-mem-shorts (g4/floats->bf16-shorts w-mem-floats)
        w-mem-buf-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [din dm] 13))

        ;; 2. Initialize per-relation cores resident in VRAM
        r-bufs-atom (atom {})
        _ (doseq [[r _] rel-counts]
            (let [r-floats (init-relation-core-floats rnd dm)
                  r-shorts (g4/floats->bf16-shorts r-floats)
                  r-buf (pjrt/buffer-from-host-buffer ctx (:client ctx) r-shorts [dm dm] 13)]
              (swap! r-bufs-atom assoc r r-buf)))

        ;; 3. Target Identity Matrix and Mask Scale
        target-mat (float-array (* k k) 0.0)
        _ (dotimes [i k] (aset target-mat (+ (* i k) i) 1.0))
        target-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (g4/floats->bf16-shorts target-mat) [k k] 13)

        mask-mat (float-array (* k k) (float (/ 1.0 (* (double k) (double tau)))))
        mask-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (g4/floats->bf16-shorts mask-mat) [k k] 13)

        by-rel (group-by :rel valid-train-triples)]

    (println (format "Training %d relation cores and W_mem (%d x %d) for %d epochs over %,d triples..."
                     (count rel-counts) din dm epochs (count valid-train-triples)))

    (dotimes [epoch epochs]
      (let [t-e0 (System/nanoTime)
            loss-sum (atom 0.0)
            step-count (atom 0)]
        (doseq [[r triples] by-rel]
          (let [r-buf (get @r-bufs-atom r)
                n (count triples)
                triples-vec (vec triples)]
            (when (and r-buf (>= n 2))
              (let [ih-arr (int-array k 0)
                    it-arr (int-array k 0)]
                (dotimes [i k]
                  (let [tr (nth triples-vec (mod i n))]
                    (aset ih-arr i (int (:h-tok tr)))
                    (aset it-arr i (int (:t-tok tr)))))
                (let [ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) ih-arr [k] 4)
                      it-b (pjrt/buffer-from-host-buffer ctx (:client ctx) it-arr [k] 4)
                      inputs [w-embed-buf ih-b it-b @w-mem-buf-atom r-buf target-b mask-b]
                      outs (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) inputs 4)
                      w-new (nth outs 0)
                      r-new (nth outs 1)
                      p-buf (nth outs 2)
                      sc-buf (nth outs 3)
                      p-floats (pjrt/buffer-to-host-buffer ctx p-buf (* k k) :bf16)
                      batch-loss (loop [i 0 l 0.0]
                                   (if (>= i k)
                                     (/ l (double k))
                                     (recur (inc i) (- l (Math/log (Math/max 1e-6 (double (aget p-floats (+ (* i k) i)))))))))]
                  (xla/destroy-buffer! ctx ih-b)
                  (xla/destroy-buffer! ctx it-b)
                  (xla/destroy-buffer! ctx p-buf)
                  (xla/destroy-buffer! ctx sc-buf)
                  (xla/destroy-buffer! ctx @w-mem-buf-atom)
                  (xla/destroy-buffer! ctx r-buf)
                  (reset! w-mem-buf-atom w-new)
                  (swap! r-bufs-atom assoc r r-new)
                  (swap! loss-sum + batch-loss)
                  (swap! step-count inc))))))
        (let [t-e1 (System/nanoTime)
              avg-loss (/ @loss-sum (double (max 1 @step-count)))
              dur-ms (/ (- t-e1 t-e0) 1e6)]
          (println (format "  Epoch %d/%d | Steps: %,d | Mean InfoNCE Loss: %.4f | Latency: %.2f ms (%.2f ms/step)"
                           (inc epoch) epochs @step-count avg-loss dur-ms (/ dur-ms (double (max 1 @step-count))))))))

    (xla/destroy-buffer! ctx target-b)
    (xla/destroy-buffer! ctx mask-b)

    ;; Convert final resident buffers to host float arrays
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

(defn train-gemma4-resolver!
  "Pre-trains Stage 2 Non-Linear Resolver weights (W_Q and W_K) in-VRAM via InfoNCE on WebNLG triples."
  [ctx w-embed-buf w-mem-floats r-maps valid-train-triples opts]
  (let [{:keys [epochs-resolver lr-resolver tau dim-mem k-triples]} opts
        v 262144
        din 1536
        dm (long dim-mem)
        k (long k-triples)
        epochs (long (or epochs-resolver 5))
        lr (double (or lr-resolver 0.05))
        rnd (java.util.Random. 43)
        _ (println (format "\nCompiling 100%% In-VRAM Stage 2 Resolver Step Executable (V=%d, Din=%d, K=%d, Dm=%d)..."
                           v din k dm))
        step-exec (contrastive/compile-in-vram-resolver-step ctx v din k dm
                                                             {:dtype :bf16
                                                              :lr lr
                                                              :tau tau
                                                              :lambda-tl 1.0})
        w-mem-shorts (g4/floats->bf16-shorts ^floats w-mem-floats)
        w-mem-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [din dm] 13)

        wq-floats (sample-gaussian-floats rnd (* din dm) 0.02)
        wk-floats (sample-gaussian-floats rnd (* din dm) 0.02)
        wq-shorts (g4/floats->bf16-shorts wq-floats)
        wk-shorts (g4/floats->bf16-shorts wk-floats)
        wq-dev-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) wq-shorts [din dm] 13))
        wk-dev-atom (atom (pjrt/buffer-from-host-buffer ctx (:client ctx) wk-shorts [din dm] 13))

        target-mat (float-array (* k k) 0.0)
        _ (dotimes [i k] (aset target-mat (+ (* i k) i) 1.0))
        target-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (g4/floats->bf16-shorts target-mat) [k k] 13)

        mask-mat (float-array (* k k) (float (/ 1.0 (* (double k) (double tau)))))
        mask-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (g4/floats->bf16-shorts mask-mat) [k k] 13)

        by-rel (group-by :rel valid-train-triples)
        r-dev-map (into {} (keep (fn [[r fa]]
                                   (when (seq fa)
                                     [r (pjrt/buffer-from-host-buffer ctx (:client ctx)
                                                                      (g4/floats->bf16-shorts fa)
                                                                      [dm dm] 13)]))
                                 r-maps))]

    (println (format "Training Stage 2 Non-Linear Resolver (W_Q, W_K: %d x %d) for %d epochs over %,d triples..."
                     din dm epochs (count valid-train-triples)))
    (dotimes [epoch epochs]
      (let [t-e0 (System/nanoTime)
            loss-sum (atom 0.0)
            step-count (atom 0)]
        (doseq [[r triples] by-rel]
          (let [r-buf (get r-dev-map r)
                n (count triples)
                triples-vec (vec triples)]
            (when (and r-buf (>= n 2))
              (let [ih-arr (int-array k 0)
                    it-arr (int-array k 0)]
                (dotimes [i k]
                  (let [tr (nth triples-vec (mod i n))]
                    (aset ih-arr i (int (:h-tok tr)))
                    (aset it-arr i (int (:t-tok tr)))))
                (let [ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) ih-arr [k] 4)
                      it-b (pjrt/buffer-from-host-buffer ctx (:client ctx) it-arr [k] 4)
                      inputs [w-embed-buf ih-b it-b w-mem-dev r-buf @wq-dev-atom @wk-dev-atom target-b mask-b]
                      outs (pjrt/execute-executable ctx (or (:handle step-exec) step-exec) inputs 4)
                      wq-new (nth outs 0)
                      wk-new (nth outs 1)
                      p-buf (nth outs 2)
                      sc-buf (nth outs 3)
                      p-floats (pjrt/buffer-to-host-buffer ctx p-buf (* k k) :bf16)
                      batch-loss (loop [i 0 l 0.0]
                                   (if (>= i k)
                                     (/ l (double k))
                                     (recur (inc i) (- l (Math/log (Math/max 1e-6 (double (aget p-floats (+ (* i k) i)))))))))]
                  (xla/destroy-buffer! ctx ih-b)
                  (xla/destroy-buffer! ctx it-b)
                  (xla/destroy-buffer! ctx p-buf)
                  (xla/destroy-buffer! ctx sc-buf)
                  (xla/destroy-buffer! ctx @wq-dev-atom)
                  (xla/destroy-buffer! ctx @wk-dev-atom)
                  (reset! wq-dev-atom wq-new)
                  (reset! wk-dev-atom wk-new)
                  (swap! loss-sum + batch-loss)
                  (swap! step-count inc))))))
        (let [t-e1 (System/nanoTime)
              avg-loss (/ @loss-sum (double (max 1 @step-count)))
              dur-ms (/ (- t-e1 t-e0) 1e6)]
          (println (format "  [Resolver] Epoch %d/%d | Steps: %,d | Mean InfoNCE Loss: %.4f | Latency: %.2f ms (%.2f ms/step)"
                           (inc epoch) epochs @step-count avg-loss dur-ms (/ dur-ms (double (max 1 @step-count))))))))

    (xla/destroy-buffer! ctx target-b)
    (xla/destroy-buffer! ctx mask-b)
    (xla/destroy-buffer! ctx w-mem-dev)
    (doseq [[_ b] r-dev-map] (xla/destroy-buffer! ctx b))

    (let [wq-buf @wq-dev-atom
          wk-buf @wk-dev-atom
          wq-floats (pjrt/buffer-to-host-buffer ctx wq-buf (* din dm) :bf16)
          wk-floats (pjrt/buffer-to-host-buffer ctx wk-buf (* din dm) :bf16)
          _ (xla/destroy-buffer! ctx wq-buf)
          _ (xla/destroy-buffer! ctx wk-buf)]
      {:w-q wq-floats
       :w-k wk-floats})))

;; ==============================================================================
;; Main Evaluation Runner
;; ==============================================================================

(defn run-gemma4-relational-evaluation [opts]
  (let [{:keys [backend model-dir train-file test-seen test-unseen checkpoint-out
                lambda-mem lambda-resolve dim-mem max-eval skip-train]} opts
        _ (println "\n================================================================================")
        _ (println "🧪 EXPERIMENT E16: TWO-STAGE NON-LINEAR RELATIONAL RETRIEVAL ON GEMMA 4 E2B")
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

        ;; 3. Train or Load Relational Cores & Stage 2 Resolver
        ckpt-f (io/file checkpoint-out)
        serialize-weights (fn [w-map]
                            (into {} (map (fn [[k v]]
                                            (cond
                                              (and (map? v) (= k :r-maps))
                                              [k (into {} (map (fn [[rk rv]] [rk (vec rv)]) v))]
                                              (instance? (Class/forName "[F") v)
                                              [k (vec v)]
                                              :else [k v]))
                                          w-map)))
        deserialize-weights (fn [w-map]
                              (into {} (map (fn [[k v]]
                                              (cond
                                                (and (map? v) (= k :r-maps))
                                                [k (into {} (map (fn [[rk rv]] [rk (float-array rv)]) v))]
                                                (sequential? v)
                                                [k (float-array v)]
                                                :else [k v]))
                                            w-map)))
        existing-ckpt (when (.exists ckpt-f) (deserialize-weights (read-string (slurp ckpt-f))))

        {:keys [w-mem r-maps]}
        (if (and skip-train (:w-mem existing-ckpt))
          (do (println (format "Loading cached relational weights from %s..." checkpoint-out))
              existing-ckpt)
          (train-gemma4-relational-cores! ctx w-embed-buf valid-train-triples rel-counts opts))

        {:keys [w-q w-k]}
        (if (and skip-train (:w-q existing-ckpt) (:w-k existing-ckpt))
          (do (println (format "Loading cached Stage 2 resolver weights from %s..." checkpoint-out))
              existing-ckpt)
          (let [res-weights (train-gemma4-resolver! ctx w-embed-buf w-mem r-maps valid-train-triples opts)]
            (when checkpoint-out
              (println (format "Saving complete relational and resolver checkpoint to %s..." checkpoint-out))
              (spit ckpt-f (pr-str (serialize-weights {:w-mem w-mem
                                                       :r-maps r-maps
                                                       :w-q (:w-q res-weights)
                                                       :w-k (:w-k res-weights)}))))
            res-weights))

        ;; 4. Compile Gemma 4 Forward Executable (targets: [:logits :normed_last])
        _ (println "\nCompiling Gemma 4 Forward Executable with dual targets [:logits :normed_last]...")
        fwd-exec (g4/compile-tensor-logic-executable
                  (assoc session :opts {:targets [:logits :normed_last] :last-token-only? true}) 64)

        ;; 5. Compile Grounded Logit Unbinding Executables
        _ (println "Compiling In-VRAM Relational Logit Grounding Executables (Stage 1)...")
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
        k-cands (count candidate-targets)
        _ (println (format "Formed candidate target evaluation pool of %,d entities." k-cands))

        ;; 7. Compile Stage 2 Non-Linear Resolver Executables
        _ (println "Compiling In-VRAM Non-Linear Candidate Resolver Executables (Stage 2)...")
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

        ;; Allocate Resident Device Buffers
        w-mem-shorts (g4/floats->bf16-shorts ^floats w-mem)
        w-mem-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [hidden-dim dm] 13)

        wq-shorts (g4/floats->bf16-shorts ^floats w-q)
        wk-shorts (g4/floats->bf16-shorts ^floats w-k)
        wq-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) wq-shorts [hidden-dim dm] 13)
        wk-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) wk-shorts [hidden-dim dm] 13)

        cand-ids-arr (int-array (map :tok candidate-targets))
        cand-ids-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) cand-ids-arr [k-cands] 4)

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
            (println "Prompt (truncated)             | Target          | Tier   | Base Top-1 | S1 (Lin) | S2 (TwoStage) | S1 Point? | S2 Point? | S1 Neigh? | S2 Neigh?")
            (println "-------------------------------+-----------------+--------+------------+----------+---------------+-----------+-----------+-----------+----------")

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
                           target-ci (first (keep-indexed (fn [idx c] (when (= (:tok c) target-tok) idx)) candidate-targets))

                           ;; 3. Prepare Relational Core R_rel
                           r-floats (get r-maps rel (float-array (* dm dm) 0.0))
                           r-shorts (g4/floats->bf16-shorts r-floats)
                           r-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) r-shorts [dm dm] 13)

                           ;; 4. Run Stage 1 Contextual Unbinding Executable
                           ground-outs (pjrt/execute-executable ctx (or (:handle unbinding-exec) unbinding-exec)
                                                                [normed-last-b base-logits-b w-mem-dev r-dev w-embed-buf] 3)
                           logits-grounded-b (nth ground-outs 0)
                           delta-logits-b (nth ground-outs 1)
                           u-norm-b (nth ground-outs 2)

                           grounded-fa (pjrt/buffer-to-host-buffer ctx logits-grounded-b vocab-size :bf16)
                           delta-fa (pjrt/buffer-to-host-buffer ctx delta-logits-b vocab-size :bf16)

                           ;; 5. Run Stage 1 Direct Entity Unbinding Executable
                           ih-b (pjrt/buffer-from-host-buffer ctx (:client ctx) (int-array [head-tok]) [1 1] 4)
                           entity-outs (pjrt/execute-executable ctx (or (:handle entity-exec) entity-exec)
                                                                [ih-b base-logits-b w-mem-dev r-dev w-embed-buf] 3)
                           _ (xla/destroy-buffer! ctx (nth entity-outs 0))
                           delta-ent-b (nth entity-outs 1)
                           ent-u-norm-b (nth entity-outs 2)
                           delta-ent-fa (pjrt/buffer-to-host-buffer ctx delta-ent-b vocab-size :bf16)
                           _ (xla/destroy-buffer! ctx delta-ent-b)

                            ;; 6. Run Stage 2 Non-Linear Resolver (Contextual)
                           scores-b (pjrt/execute-executable ctx (or (:handle resolver-exec) resolver-exec)
                                                             [normed-last-b u-norm-b cand-ids-dev w-embed-buf wq-dev wk-dev] 1)
                           scores-fa (pjrt/buffer-to-host-buffer ctx scores-b k-cands :bf16)
                           _ (xla/destroy-buffer! ctx scores-b)

                            ;; 7. Run Stage 2 Non-Linear Resolver (Direct Entity)
                           ent-scores-b (pjrt/execute-executable ctx (or (:handle ent-resolver-exec) ent-resolver-exec)
                                                                 [ih-b ent-u-norm-b cand-ids-dev w-embed-buf wq-dev wk-dev] 1)
                           ent-scores-fa (pjrt/buffer-to-host-buffer ctx ent-scores-b k-cands :bf16)
                           _ (xla/destroy-buffer! ctx ent-scores-b)

                           _ (xla/destroy-buffer! ctx ih-b)
                           _ (xla/destroy-buffer! ctx u-norm-b)
                           _ (xla/destroy-buffer! ctx ent-u-norm-b)
                           _ (xla/destroy-buffer! ctx base-logits-b)
                           _ (xla/destroy-buffer! ctx normed-last-b)
                           _ (xla/destroy-buffer! ctx r-dev)
                           _ (xla/destroy-buffer! ctx logits-grounded-b)
                           _ (xla/destroy-buffer! ctx delta-logits-b)

                           ;; 8. Stage 1 Metrics (Linear alone)
                           s1-top-cand (top-candidate grounded-fa candidate-targets)
                           s1-top-str (:str s1-top-cand)
                           s1-top-tok (:tok s1-top-cand)
                           s1-delta-target (double (aget delta-fa target-tok))
                           distractor-cands (filterv #(not= (:tok %) target-tok) candidate-targets)
                           s1-dist-deltas (mapv #(double (aget delta-fa (:tok %))) distractor-cands)
                           s1-mean-dist (if (seq s1-dist-deltas) (/ (reduce + s1-dist-deltas) (double (count s1-dist-deltas))) 0.0)
                           s1-max-dist (if (seq s1-dist-deltas) (apply max s1-dist-deltas) 0.0)
                           s1-pointwise? (> s1-delta-target s1-max-dist)
                           s1-neigh? (> s1-delta-target s1-mean-dist)

                           ;; Direct Entity Stage 1 Metrics
                           ent-s1-delta-target (double (aget delta-ent-fa target-tok))
                           ent-s1-dist-deltas (mapv #(double (aget delta-ent-fa (:tok %))) distractor-cands)
                           ent-s1-max-dist (if (seq ent-s1-dist-deltas) (apply max ent-s1-dist-deltas) 0.0)
                           ent-s1-mean-dist (if (seq ent-s1-dist-deltas) (/ (reduce + ent-s1-dist-deltas) (double (count ent-s1-dist-deltas))) 0.0)
                           ent-s1-pointwise? (> ent-s1-delta-target ent-s1-max-dist)
                           ent-s1-neigh? (> ent-s1-delta-target ent-s1-mean-dist)

                           ;; 9. Stage 2 Metrics (Two-Stage Non-Linear Resolver)
                           s2-target-score (if target-ci (double (aget scores-fa target-ci)) 0.0)
                           s2-delta-target (+ s1-delta-target (* (double lambda-resolve) s2-target-score))
                           s2-dist-deltas (keep-indexed (fn [i c]
                                                          (when (not= i target-ci)
                                                            (+ (double (aget delta-fa (:tok c)))
                                                               (* (double lambda-resolve) (double (aget scores-fa i))))))
                                                        candidate-targets)
                           s2-max-dist (if (seq s2-dist-deltas) (apply max s2-dist-deltas) 0.0)
                           s2-mean-dist (if (seq s2-dist-deltas) (/ (reduce + s2-dist-deltas) (double (count s2-dist-deltas))) 0.0)
                           s2-pointwise? (> s2-delta-target s2-max-dist)
                           s2-neigh? (> s2-delta-target s2-mean-dist)

                           s2-best-idx (apply max-key (fn [i]
                                                        (let [c (nth candidate-targets i)]
                                                          (+ (double (aget base-logits-fa (:tok c)))
                                                             (double (aget delta-fa (:tok c)))
                                                             (* (double lambda-resolve) (double (aget scores-fa i))))))
                                              (range k-cands))
                           s2-top-cand (nth candidate-targets s2-best-idx)
                           s2-top-str (:str s2-top-cand)
                           s2-top-tok (:tok s2-top-cand)

                           ;; Direct Entity Stage 2 Metrics
                           ent-s2-target-score (if target-ci (double (aget ent-scores-fa target-ci)) 0.0)
                           ent-s2-delta-target (+ ent-s1-delta-target (* (double lambda-resolve) ent-s2-target-score))
                           ent-s2-dist-deltas (keep-indexed (fn [i c]
                                                              (when (not= i target-ci)
                                                                (+ (double (aget delta-ent-fa (:tok c)))
                                                                   (* (double lambda-resolve) (double (aget ent-scores-fa i))))))
                                                            candidate-targets)
                           ent-s2-max-dist (if (seq ent-s2-dist-deltas) (apply max ent-s2-dist-deltas) 0.0)
                           ent-s2-mean-dist (if (seq ent-s2-dist-deltas) (/ (reduce + ent-s2-dist-deltas) (double (count ent-s2-dist-deltas))) 0.0)
                           ent-s2-pointwise? (> ent-s2-delta-target ent-s2-max-dist)
                           ent-s2-neigh? (> ent-s2-delta-target ent-s2-mean-dist)

                           p-disp (if (> (count prompt) 30) (str (subs prompt 0 27) "...") prompt)
                           t-disp (if (> (count target) 15) (str (subs target 0 12) "...") target)
                           tier-str (case tier :head "HEAD" :mid "MID " :tail "TAIL" :unseen "UNSEEN")]

                       (println (format "%-30s | %-15s | %-6s | %-10s | %-8s | %-13s | %-9s | %-9s | %-9s | %-9s"
                                        p-disp t-disp tier-str
                                        (if (= base-top-tok target-tok) (str base-top-str " ✓") base-top-str)
                                        (if (= s1-top-tok target-tok) (str s1-top-str " ✓") s1-top-str)
                                        (if (= s2-top-tok target-tok) (str s2-top-str " ✓") s2-top-str)
                                        (if s1-pointwise? "YES" "NO")
                                        (if s2-pointwise? "YES (Fact)" "NO")
                                        (if s1-neigh? "YES" "NO")
                                        (if s2-neigh? "YES" "NO")))

                       {:prompt prompt
                        :target target
                        :rel rel
                        :tier tier
                        :base-match? (= base-top-tok target-tok)
                        :s1-match? (= s1-top-tok target-tok)
                        :s2-match? (= s2-top-tok target-tok)
                        :s1-delta-target s1-delta-target
                        :s1-mean-dist s1-mean-dist
                        :s1-max-dist s1-max-dist
                        :s1-pointwise? s1-pointwise?
                        :s1-neigh? s1-neigh?
                        :s2-delta-target s2-delta-target
                        :s2-mean-dist s2-mean-dist
                        :s2-max-dist s2-max-dist
                        :s2-pointwise? s2-pointwise?
                        :s2-neigh? s2-neigh?
                        :ent-s1-pointwise? ent-s1-pointwise?
                        :ent-s1-neigh? ent-s1-neigh?
                        :ent-s2-pointwise? ent-s2-pointwise?
                        :ent-s2-neigh? ent-s2-neigh?
                        :ent-s1-delta-target ent-s1-delta-target
                        :ent-s2-delta-target ent-s2-delta-target}))
                   selected-evals)

                  total-n (count results)
                  base-acc (* 100.0 (/ (double (count (filter :base-match? results))) (double total-n)))
                  s1-acc (* 100.0 (/ (double (count (filter :s1-match? results))) (double total-n)))
                  s2-acc (* 100.0 (/ (double (count (filter :s2-match? results))) (double total-n)))

                  s1-pointwise (count (filter :s1-pointwise? results))
                  s2-pointwise (count (filter :s2-pointwise? results))
                  s1-neigh (count (filter :s1-neigh? results))
                  s2-neigh (count (filter :s2-neigh? results))

                  ent-s1-pointwise (count (filter :ent-s1-pointwise? results))
                  ent-s2-pointwise (count (filter :ent-s2-pointwise? results))
                  ent-s1-neigh (count (filter :ent-s1-neigh? results))
                  ent-s2-neigh (count (filter :ent-s2-neigh? results))

                  mean-s1-tgt (/ (reduce + (map :s1-delta-target results)) (double (max 1 total-n)))
                  mean-s1-dist (/ (reduce + (map :s1-mean-dist results)) (double (max 1 total-n)))
                  mean-s1-max (/ (reduce + (map :s1-max-dist results)) (double (max 1 total-n)))

                  mean-s2-tgt (/ (reduce + (map :s2-delta-target results)) (double (max 1 total-n)))
                  mean-s2-dist (/ (reduce + (map :s2-mean-dist results)) (double (max 1 total-n)))
                  mean-s2-max (/ (reduce + (map :s2-max-dist results)) (double (max 1 total-n)))]

              (println "\n================================================================================")
              (println (format "🎯 GEMMA 4 TWO-STAGE RETRIEVAL DIAGNOSTIC: %s (N=%d)" split-name total-n))
              (println "================================================================================")
              (println (format "Top-1 Accuracy Baseline (Zero R_mem)       : %5.1f%% (%d/%d)"
                               base-acc (count (filter :base-match? results)) total-n))
              (println (format "Top-1 Accuracy Stage 1 (Linear Unbinding)  : %5.1f%% (%d/%d) [Lift: %+d]"
                               s1-acc (count (filter :s1-match? results)) total-n
                               (- (count (filter :s1-match? results)) (count (filter :base-match? results)))))
              (println (format "Top-1 Accuracy Stage 2 (NonLinear Resolver): %5.1f%% (%d/%d) [Lift: %+d]"
                               s2-acc (count (filter :s2-match? results)) total-n
                               (- (count (filter :s2-match? results)) (count (filter :base-match? results)))))
              (println "--------------------------------------------------------------------------------")
              (println (format "Stage 1 Contextual Pointwise Selectivity (Δ > max Δ): %d/%d (%.1f%%)"
                               s1-pointwise total-n (* 100.0 (/ (double s1-pointwise) (double total-n)))))
              (println (format "Stage 2 Contextual Pointwise Selectivity (Δ > max Δ): %d/%d (%.1f%%)  <-- [THE HYPOTHESIS TEST]"
                               s2-pointwise total-n (* 100.0 (/ (double s2-pointwise) (double total-n)))))
              (println (format "Stage 1 Neighborhood Selectivity (Δ > mean Δ)       : %d/%d (%.1f%%)"
                               s1-neigh total-n (* 100.0 (/ (double s1-neigh) (double total-n)))))
              (println (format "Stage 2 Neighborhood Selectivity (Δ > mean Δ)       : %d/%d (%.1f%%)"
                               s2-neigh total-n (* 100.0 (/ (double s2-neigh) (double total-n)))))
              (println "--------------------------------------------------------------------------------")
              (println (format "Stage 1 Direct-Entity Pointwise Selectivity         : %d/%d (%.1f%%)"
                               ent-s1-pointwise total-n (* 100.0 (/ (double ent-s1-pointwise) (double total-n)))))
              (println (format "Stage 2 Direct-Entity Pointwise Selectivity         : %d/%d (%.1f%%)  <-- [ENTITY HYPOTHESIS TEST]"
                               ent-s2-pointwise total-n (* 100.0 (/ (double ent-s2-pointwise) (double total-n)))))
              (println (format "Stage 1 Direct-Entity Neighborhood Selectivity      : %d/%d (%.1f%%)"
                               ent-s1-neigh total-n (* 100.0 (/ (double ent-s1-neigh) (double total-n)))))
              (println (format "Stage 2 Direct-Entity Neighborhood Selectivity      : %d/%d (%.1f%%)"
                               ent-s2-neigh total-n (* 100.0 (/ (double ent-s2-neigh) (double total-n)))))
              (println "--------------------------------------------------------------------------------")
              (println (format "Stage 1 Shifts : Target Δ: %+6.4f | Mean Dist Δ: %+6.4f | Max Dist Δ: %+6.4f"
                               mean-s1-tgt mean-s1-dist mean-s1-max))
              (println (format "Stage 2 Shifts : Target Δ: %+6.4f | Mean Dist Δ: %+6.4f | Max Dist Δ: %+6.4f"
                               mean-s2-tgt mean-s2-dist mean-s2-max))
              (println "--------------------------------------------------------------------------------")
              (println "Frequency Tier     | Eval | Base | S1 Acc | S2 Acc | S1 Point? | S2 Point? | S1 Neigh? | S2 Neigh?")
              (println "-------------------+------+------+--------+--------+-----------+-----------+-----------+----------")
              (doseq [t [:head :mid :tail :unseen]]
                (let [t-res (filter #(= (:tier %) t) results)
                      tn (count t-res)]
                  (when (pos? tn)
                    (let [t-base (* 100.0 (/ (double (count (filter :base-match? t-res))) (double tn)))
                          t-s1 (* 100.0 (/ (double (count (filter :s1-match? t-res))) (double tn)))
                          t-s2 (* 100.0 (/ (double (count (filter :s2-match? t-res))) (double tn)))
                          t-s1-p (* 100.0 (/ (double (count (filter :s1-pointwise? t-res))) (double tn)))
                          t-s2-p (* 100.0 (/ (double (count (filter :s2-pointwise? t-res))) (double tn)))
                          t-s1-n (* 100.0 (/ (double (count (filter :s1-neigh? t-res))) (double tn)))
                          t-s2-n (* 100.0 (/ (double (count (filter :s2-neigh? t-res))) (double tn)))
                          label (case t :head "Head (>= 50)      " :mid "Mid (10 - 49)     " :tail "Tail (< 10)       " :unseen "Unseen (0 Core)   ")]
                      (println (format "%s | %-4d | %4.1f%%| %5.1f%% | %5.1f%% |   %5.1f%%  |   %5.1f%%  |   %5.1f%%  |   %5.1f%%"
                                       label tn t-base t-s1 t-s2 t-s1-p t-s2-p t-s1-n t-s2-n))))))
              (println "================================================================================\n")
              results)))

        ;; Run evaluation on test_seen.edn and test_unseen.edn
        seen-res (eval-split-fn "TEST_SEEN (Held-out triples from trained relations)" test-seen)
        unseen-res (eval-split-fn "TEST_UNSEEN (Triples from zero-shot unseen relations)" test-unseen)
        _ (xla/destroy-buffer! ctx w-mem-dev)
        _ (xla/destroy-buffer! ctx wq-dev)
        _ (xla/destroy-buffer! ctx wk-dev)
        _ (xla/destroy-buffer! ctx cand-ids-dev)]
    {:seen seen-res
     :unseen unseen-res}))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-gemma4-relational-evaluation opts)
    (System/exit 0)))
