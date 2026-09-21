(ns tools.e18-phase0-diagnostics
  "Experiment E18 Phase 0 Diagnostics:
   1. Batch composition audit (frequencies, duplicate-positive and false-negative rates).
   2. Query distribution alignment (cosine similarity of V_h vs h_ctx).
   3. Key-space geometry (post-W_K keys pairwise cosine for co-typed vs random candidates).
   4. Training dynamics (Stage-2 InfoNCE loss trajectory across epochs 1..5).
   5. Amplification decomposition (raw score_resolve vs Delta_type across targets and distractors)."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.contrastive :as contrastive]
            [models.gemma :as gemma]
            [einsum.logic.symbolic :as sym]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto]
            [tools.gemma4-inference :as g4]
            [clojure.java.io :as io])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

;; ==============================================================================
;; Math & Geometry Helpers
;; ==============================================================================

(defn- sample-gaussian-floats [^java.util.Random rnd n scale]
  (let [^floats arr (float-array n)]
    (dotimes [i n]
      (aset arr i (float (* (.nextGaussian rnd) scale))))
    arr))

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

(defn cosine-sim
  "Computes cosine similarity between two float vectors."
  [^floats a ^floats b]
  (let [n (count a)]
    (loop [i 0 dot 0.0 norm-a 0.0 norm-b 0.0]
      (if (>= i n)
        (let [denom (* (Math/sqrt norm-a) (Math/sqrt norm-b))]
          (if (< denom 1e-12) 0.0 (/ dot denom)))
        (let [ai (double (aget a i))
              bi (double (aget b i))]
          (recur (inc i)
                 (+ dot (* ai bi))
                 (+ norm-a (* ai ai))
                 (+ norm-b (* bi bi))))))))

(defn mean-and-std [vals]
  (let [n (count vals)]
    (if (zero? n)
      {:mean 0.0 :std 0.0 :min 0.0 :max 0.0 :n 0}
      (let [mean (/ (double (reduce + vals)) (double n))
            var (/ (reduce + (map #(let [d (- (double %) mean)] (* d d)) vals)) (double n))
            std (Math/sqrt var)]
        {:mean mean
         :std std
         :min (double (apply min vals))
         :max (double (apply max vals))
         :n n}))))

(defn rms-norm-vec
  "Gemma RMS norm: x / sqrt(mean(x^2) + eps)"
  [^floats x eps]
  (let [n (count x)
        ss (loop [i 0 s 0.0]
             (if (>= i n) s (let [v (double (aget x i))] (recur (inc i) (+ s (* v v))))))
        rms (Math/sqrt (+ (/ ss (double n)) (double eps)))
        out (float-array n)]
    (dotimes [i n]
      (aset out i (float (/ (double (aget x i)) rms))))
    out))

(defn mat-vec-mul
  "Computes x * W where x is [din], W is [din dm] row-major -> out [dm]."
  [^floats x ^floats w din dm]
  (let [out (float-array dm)]
    (dotimes [j dm]
      (let [sum (loop [i 0 s 0.0]
                  (if (>= i din)
                    s
                    (recur (inc i) (+ s (* (double (aget x i))
                                           (double (aget w (+ (* i dm) j))))))))]
        (aset out j (float sum))))
    out))

(defn extract-token-embedding
  "Reads a 1536-dim BF16 embedding row from mapped MemorySegment into a Java float array."
  [^MemorySegment slice token-idx din]
  (let [byte-offset (* (long token-idx) (long din) 2)
        arr (float-array din)
        val-layout (.. ValueLayout/JAVA_SHORT (withByteAlignment 1))]
    (dotimes [i din]
      (let [s (.get slice val-layout (+ byte-offset (* (long i) 2)))
            bits (unchecked-int (bit-shift-left (long (bit-and (int s) 0xffff)) 16))]
        (aset arr i (Float/intBitsToFloat bits))))
    arr))

;; ==============================================================================
;; Phase 0 Item 1: Batch Composition Audit
;; ==============================================================================

(defn run-batch-composition-audit [train-file k]
  (let [data (read-string (slurp train-file))
        entries (:entries data)
        triples (distinct (mapcat :triples entries))
        by-rel (group-by second triples)
        rel-counts (into {} (map (fn [[k-rel v]] [k-rel (count v)]) by-rel))
        total-rels (count rel-counts)
        rels-lt-16 (count (filter #(< (val %) k) rel-counts))
        rels-lt-2 (count (filter #(< (val %) 2) rel-counts))
        rels-ge-2 (count (filter #(>= (val %) 2) rel-counts))
        counts-sorted (sort (vals rel-counts))
        n-triples (count triples)
        trained-rels (filter (fn [[_ trs]] (>= (count trs) 2)) by-rel)
        all-triples-set (set triples)
        stats
        (mapv (fn [[r trs]]
                (let [n (count trs)
                      trs-vec (vec trs)
                      batch (mapv #(nth trs-vec (mod % n)) (range k))
                      row-stats
                      (mapv (fn [i]
                              (let [tr-i (nth batch i)
                                    h-i (first tr-i)
                                    t-i (nth tr-i 2)
                                    off-diags (keep-indexed (fn [j tr-j] (when (not= i j) tr-j)) batch)
                                    dup-pos (count (filter (fn [tr-j] (= (nth tr-j 2) t-i)) off-diags))
                                    false-negs (count (filter (fn [tr-j]
                                                                (let [t-j (nth tr-j 2)]
                                                                  (and (not= t-j t-i)
                                                                       (contains? all-triples-set [h-i r t-j]))))
                                                              off-diags))]
                                {:dup-pos dup-pos
                                 :false-negs false-negs
                                 :total-off (count off-diags)}))
                            (range k))]
                  {:r r
                   :n n
                   :total-dups (reduce + (map :dup-pos row-stats))
                   :total-fn (reduce + (map :false-negs row-stats))
                   :total-off (* k (dec k))}))
              trained-rels)
        total-off (reduce + (map :total-off stats))
        total-dups (reduce + (map :total-dups stats))
        total-fn (reduce + (map :total-fn stats))
        lt-16 (filter #(< (:n %) k) stats)
        ge-16 (filter #(>= (:n %) k) stats)
        lt-off (reduce + (map :total-off lt-16))
        lt-dups (reduce + (map :total-dups lt-16))
        ge-off (reduce + (map :total-off ge-16))
        ge-dups (reduce + (map :total-dups ge-16))]
    {:total-entries (count entries)
     :distinct-triples n-triples
     :total-relations total-rels
     :triples-per-rel {:min (first counts-sorted)
                       :max (last counts-sorted)
                       :median (nth counts-sorted (quot total-rels 2))
                       :mean (/ (double n-triples) total-rels)}
     :rels-lt-16 {:count rels-lt-16 :pct (* 100.0 (/ rels-lt-16 (double total-rels)))}
     :rels-lt-2 {:count rels-lt-2 :pct (* 100.0 (/ rels-lt-2 (double total-rels)))}
     :rels-ge-2-trained {:count rels-ge-2 :pct (* 100.0 (/ rels-ge-2 (double total-rels)))}
     :in-batch-contamination
     {:total-off-diagonals total-off
      :duplicate-positive {:count total-dups :pct (* 100.0 (/ total-dups (double total-off)))}
      :false-negative {:count total-fn :pct (* 100.0 (/ total-fn (double total-off)))}
      :total-contaminated {:count (+ total-dups total-fn)
                           :pct (* 100.0 (/ (+ total-dups total-fn) (double total-off)))}
      :lt-16-subgroup {:relations (count lt-16)
                       :off-diagonals lt-off
                       :duplicates lt-dups
                       :pct (* 100.0 (/ lt-dups (double (max 1 lt-off))))}
      :ge-16-subgroup {:relations (count ge-16)
                       :off-diagonals ge-off
                       :duplicates ge-dups
                       :pct (* 100.0 (/ ge-dups (double (max 1 ge-off))))}}}))

;; ==============================================================================
;; Phase 0 Item 3: Key-Space Geometry
;; ==============================================================================

(defn run-key-space-geometry [embed-slice candidate-targets checkpoint all-triples]
  (let [wk-floats (float-array (:w-k checkpoint))
        din 1536
        dm 128
        k-cands (count candidate-targets)
        ;; 1. Compute raw embeddings e_c and post-W_K keys k_c for each candidate
        cands-data
        (mapv (fn [c]
                (let [tok (:tok c)
                      e-c (extract-token-embedding embed-slice tok din)
                      z-c (mat-vec-mul e-c wk-floats din dm)
                      k-c (rms-norm-vec z-c 1e-6)]
                  {:cand c
                   :e e-c
                   :k k-c}))
              candidate-targets)

        ;; 2. Determine co-typed pairs (sharing at least one relation tail in all-triples)
        cand-str->idx (into {} (map-indexed (fn [i c] [(:str c) i]) candidate-targets))
        by-rel (group-by second all-triples)
        co-typed-pairs-set
        (reduce
         (fn [acc [_ trs]]
           (let [tails (distinct (map #(nth % 2) trs))
                 cand-indices (keep cand-str->idx tails)
                 pairs (for [i cand-indices j cand-indices :when (< i j)] [i j])]
             (into acc pairs)))
         #{}
         by-rel)

        all-pairs (for [i (range k-cands) j (range k-cands) :when (< i j)] [i j])
        co-typed-pairs (vec co-typed-pairs-set)
        non-co-typed-pairs (filterv #(not (contains? co-typed-pairs-set %)) all-pairs)
        ;; Sample 20,000 non-co-typed pairs for stable random baseline
        sampled-non-co-typed (take 20000 (shuffle non-co-typed-pairs))

        ;; 3. Compute cosine similarities in Key space
        co-typed-k-cosines
        (mapv (fn [[i j]]
                (cosine-sim (:k (nth cands-data i)) (:k (nth cands-data j))))
              co-typed-pairs)

        random-k-cosines
        (mapv (fn [[i j]]
                (cosine-sim (:k (nth cands-data i)) (:k (nth cands-data j))))
              sampled-non-co-typed)

        ;; 4. Compute cosine similarities in Embedding space
        co-typed-e-cosines
        (mapv (fn [[i j]]
                (cosine-sim (:e (nth cands-data i)) (:e (nth cands-data j))))
              co-typed-pairs)

        random-e-cosines
        (mapv (fn [[i j]]
                (cosine-sim (:e (nth cands-data i)) (:e (nth cands-data j))))
              sampled-non-co-typed)]

    {:candidate-count k-cands
     :co-typed-pair-count (count co-typed-pairs)
     :non-co-typed-pair-count (count non-co-typed-pairs)
     :key-space
     {:co-typed (mean-and-std co-typed-k-cosines)
      :random (mean-and-std random-k-cosines)
      :delta-mean (- (:mean (mean-and-std co-typed-k-cosines))
                     (:mean (mean-and-std random-k-cosines)))}
     :embed-space
     {:co-typed (mean-and-std co-typed-e-cosines)
      :random (mean-and-std random-e-cosines)
      :delta-mean (- (:mean (mean-and-std co-typed-e-cosines))
                     (:mean (mean-and-std random-e-cosines)))}}))

;; ==============================================================================
;; Phase 0 Items 2 & 5: Inference, Alignment & Amplification Decomposition
;; ==============================================================================

(defn deserialize-weights [w-map]
  (into {} (map (fn [[k v]]
                  (cond
                    (and (map? v) (= k :r-maps))
                    [k (into {} (map (fn [[rk rv]] [rk (float-array rv)]) v))]
                    (sequential? v)
                    [k (float-array v)]
                    :else [k v]))
                w-map)))

(defn run-evaluation-diagnostics
  [session dev-weights candidate-targets checkpoint opts]
  (let [{:keys [test-seen test-unseen dim-mem lambda-mem lambda-resolve max-eval]} opts
        ctx (:ctx session)
        tokenizer (:tokenizer session)
        config (:config session)
        vocab-size (long (:vocab-size config))
        hidden-dim (long (:hidden-dim config))
        dm (long dim-mem)
        k-cands (count candidate-targets)
        w-embed-buf (first dev-weights)

        w-mem-floats (float-array (:w-mem checkpoint))
        r-maps (:r-maps checkpoint)
        wq-floats (float-array (:w-q checkpoint))
        wk-floats (float-array (:w-k checkpoint))

        w-mem-shorts (g4/floats->bf16-shorts w-mem-floats)
        wq-shorts (g4/floats->bf16-shorts wq-floats)
        wk-shorts (g4/floats->bf16-shorts wk-floats)

        w-mem-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) w-mem-shorts [hidden-dim dm] 13)
        wq-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) wq-shorts [hidden-dim dm] 13)
        wk-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) wk-shorts [hidden-dim dm] 13)

        cand-ids-arr (int-array (map :tok candidate-targets))
        cand-ids-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) cand-ids-arr [k-cands] 4)

        ;; Compilations
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

        resolver-invars [[:h [:tensor [1 1 hidden-dim] :bf16]]
                         [:u_norm [:tensor [1 1 dm] :bf16]]
                         [:cand_ids [:tensor [k-cands] :i32]]
                         [:W_embed [:tensor [vocab-size hidden-dim] :bf16]]
                         [:W_Q [:tensor [hidden-dim dm] :bf16]]
                         [:W_K [:tensor [hidden-dim dm] :bf16]]]
        resolver-ast (gemma/gemma4-two-stage-resolver-ast hidden-dim dm vocab-size k-cands)
        resolver-exec (sym/compile-query ctx "gemma_two_stage_resolver" resolver-invars resolver-ast [:scores])

        ;; Safetensors arena for V_h reading
        arena (Arena/ofConfined)
        mapped-st (st/map-safetensors-weights ".models/gemma-4-E2B-it/model.safetensors" arena)
        embed-slice (st/get-tensor-slice mapped-st "model.language_model.embed_tokens.weight")

        ;; Diagnostic evaluation over a split
        eval-split-diagnostics
        (fn [split-path n-eval]
          (let [entries (:entries (read-string (slurp split-path)))
                triples (distinct (mapcat :triples entries))
                cloze-prompts
                (vec (keep (fn [[h rel t]]
                             (let [t-toks (proto/encode tokenizer (str " " t) false)
                                   h-toks (proto/encode tokenizer (str " " h) false)]
                               (when (and (seq t-toks) (seq h-toks))
                                 {:prompt (str "The " (name rel) " of " h " is")
                                  :head h
                                  :head-tok (first h-toks)
                                  :rel rel
                                  :target t
                                  :target-tok (first t-toks)})))
                           triples))
                selected (take n-eval (distinct cloze-prompts))]
            (mapv
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

                     normed-last-fa (pjrt/buffer-to-host-buffer ctx normed-last-b hidden-dim :bf16)
                     vh-fa (extract-token-embedding embed-slice head-tok hidden-dim)
                     query-cos (cosine-sim vh-fa normed-last-fa)

                     r-floats (or (get r-maps rel) (float-array (* dm dm) 0.0))
                     r-shorts (g4/floats->bf16-shorts r-floats)
                     r-dev (pjrt/buffer-from-host-buffer ctx (:client ctx) r-shorts [dm dm] 13)

                     ground-outs (pjrt/execute-executable ctx (or (:handle unbinding-exec) unbinding-exec)
                                                          [normed-last-b base-logits-b w-mem-dev r-dev w-embed-buf] 3)
                     logits-grounded-b (nth ground-outs 0)
                     delta-logits-b (nth ground-outs 1)
                     u-norm-b (nth ground-outs 2)
                     delta-fa (pjrt/buffer-to-host-buffer ctx delta-logits-b vocab-size :bf16)

                     scores-b (pjrt/execute-executable ctx (or (:handle resolver-exec) resolver-exec)
                                                       [normed-last-b u-norm-b cand-ids-dev w-embed-buf wq-dev wk-dev] 1)
                     scores-fa (pjrt/buffer-to-host-buffer ctx scores-b k-cands :bf16)

                     _ (xla/destroy-buffer! ctx logits-grounded-b)
                     _ (xla/destroy-buffer! ctx delta-logits-b)
                     _ (xla/destroy-buffer! ctx u-norm-b)
                     _ (xla/destroy-buffer! ctx scores-b)
                     _ (xla/destroy-buffer! ctx base-logits-b)
                     _ (xla/destroy-buffer! ctx normed-last-b)
                     _ (xla/destroy-buffer! ctx r-dev)

                     target-ci (first (keep-indexed (fn [idx c] (when (= (:tok c) target-tok) idx)) candidate-targets))

                     ;; Stage 1 Linear Shifts
                     s1-target-shift (double (aget delta-fa target-tok))
                     s1-distractor-shifts (keep-indexed (fn [idx c]
                                                          (when (not= idx target-ci)
                                                            (double (aget delta-fa (:tok c)))))
                                                        candidate-targets)
                     s1-dist-mean (if (seq s1-distractor-shifts) (/ (reduce + s1-distractor-shifts) (double (count s1-distractor-shifts))) 0.0)
                     s1-dist-max (if (seq s1-distractor-shifts) (apply max s1-distractor-shifts) 0.0)

                     ;; Stage 2 Resolver Shifts
                     s2-res-target-score (if target-ci (double (aget scores-fa target-ci)) 0.0)
                     s2-res-distractor-scores (keep-indexed (fn [idx _]
                                                              (when (not= idx target-ci)
                                                                (double (aget scores-fa idx))))
                                                            candidate-targets)
                     s2-res-dist-mean (if (seq s2-res-distractor-scores) (/ (reduce + s2-res-distractor-scores) (double (count s2-res-distractor-scores))) 0.0)
                     s2-res-dist-max (if (seq s2-res-distractor-scores) (apply max s2-res-distractor-scores) 0.0)

                     ;; Combined Stage 2 Shifts
                     s2-target-total (+ s1-target-shift (* (double lambda-resolve) s2-res-target-score))
                     s2-dist-total-shifts (mapv + s1-distractor-shifts (mapv #(* (double lambda-resolve) %) s2-res-distractor-scores))
                     s2-dist-total-mean (if (seq s2-dist-total-shifts) (/ (reduce + s2-dist-total-shifts) (double (count s2-dist-total-shifts))) 0.0)
                     s2-dist-total-max (if (seq s2-dist-total-shifts) (apply max s2-dist-total-shifts) 0.0)]
                 {:prompt prompt
                  :rel rel
                  :head (:head ep)
                  :target (:target ep)
                  :query-cosine query-cos
                  :s1 {:target s1-target-shift :dist-mean s1-dist-mean :dist-max s1-dist-max}
                  :s2-resolve {:target s2-res-target-score :dist-mean s2-res-dist-mean :dist-max s2-res-dist-max}
                  :s2-total {:target s2-target-total :dist-mean s2-dist-total-mean :dist-max s2-dist-total-max}}))
             selected)))

        seen-diag (eval-split-diagnostics test-seen (or max-eval 40))
        unseen-diag (eval-split-diagnostics test-unseen (or max-eval 40))

        _ (xla/destroy-buffer! ctx w-mem-dev)
        _ (xla/destroy-buffer! ctx wq-dev)
        _ (xla/destroy-buffer! ctx wk-dev)
        _ (xla/destroy-buffer! ctx cand-ids-dev)
        _ (.close arena)]

    {:seen-diagnostics seen-diag
     :unseen-diagnostics unseen-diag
     :seen-query-alignment (mean-and-std (map :query-cosine seen-diag))
     :unseen-query-alignment (mean-and-std (map :query-cosine unseen-diag))
     :seen-amplification-decomposition
     {:s1-target-shift (mean-and-std (map #(get-in % [:s1 :target]) seen-diag))
      :s1-dist-mean (mean-and-std (map #(get-in % [:s1 :dist-mean]) seen-diag))
      :s1-dist-max (mean-and-std (map #(get-in % [:s1 :dist-max]) seen-diag))
      :s2-res-target (mean-and-std (map #(get-in % [:s2-resolve :target]) seen-diag))
      :s2-res-dist-mean (mean-and-std (map #(get-in % [:s2-resolve :dist-mean]) seen-diag))
      :s2-res-dist-max (mean-and-std (map #(get-in % [:s2-resolve :dist-max]) seen-diag))
      :s2-total-target (mean-and-std (map #(get-in % [:s2-total :target]) seen-diag))
      :s2-total-dist-mean (mean-and-std (map #(get-in % [:s2-total :dist-mean]) seen-diag))
      :s2-total-dist-max (mean-and-std (map #(get-in % [:s2-total :dist-max]) seen-diag))}}))

;; ==============================================================================
;; Phase 0 Item 4: Training Dynamics (Re-run 5 Epochs with Loss Logging)
;; ==============================================================================

(defn run-training-dynamics [ctx dev-weights valid-train-triples checkpoint opts]
  (let [{:keys [epochs-resolver lr-resolver tau dim-mem k-triples]} opts
        v 262144
        din 1536
        dm (long dim-mem)
        k (long k-triples)
        epochs (long (or epochs-resolver 5))
        lr (double (or lr-resolver 0.05))
        rnd (java.util.Random. 43)
        w-embed-buf (first dev-weights)
        w-mem-floats (float-array (:w-mem checkpoint))
        r-maps (:r-maps checkpoint)

        step-exec (contrastive/compile-in-vram-resolver-step ctx v din k dm
                                                             {:dtype :bf16
                                                              :lr lr
                                                              :tau tau
                                                              :lambda-tl 1.0})
        w-mem-shorts (g4/floats->bf16-shorts w-mem-floats)
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
                                                                      (g4/floats->bf16-shorts (float-array fa))
                                                                      [dm dm] 13)]))
                                 r-maps))
        loss-history (atom [])]

    (println "\nMeasuring Stage 2 Resolver InfoNCE Training Dynamics across 5 epochs (seed 43)...")
    (dotimes [epoch epochs]
      (let [loss-sum (atom 0.0)
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
        (let [avg-loss (/ @loss-sum (double (max 1 @step-count)))]
          (println (format "  Epoch %d/%d: Mean InfoNCE Loss = %.4f" (inc epoch) epochs avg-loss))
          (swap! loss-history conj {:epoch (inc epoch) :loss avg-loss :steps @step-count}))))

    (xla/destroy-buffer! ctx target-b)
    (xla/destroy-buffer! ctx mask-b)
    (xla/destroy-buffer! ctx w-mem-dev)
    (xla/destroy-buffer! ctx @wq-dev-atom)
    (xla/destroy-buffer! ctx @wk-dev-atom)
    (doseq [[_ b] r-dev-map] (xla/destroy-buffer! ctx b))

    @loss-history))

;; ==============================================================================
;; Top-Level Diagnostic Suite Entrypoint
;; ==============================================================================

(defn run-phase0-suite [opts]
  (println "================================================================================")
  (println "🔬 EXPERIMENT E18: PHASE 0 DIAGNOSTIC MEASUREMENTS")
  (println "================================================================================")
  (init-rocm-c-env!)

  (let [train-file (:train-file opts)
        checkpoint-file (:checkpoint-out opts)
        k (long (:k-triples opts))

        ;; 1. Batch Composition Audit
        _ (println "\n--- Step 1: Batch Composition Audit ---")
        audit-res (run-batch-composition-audit train-file k)
        _ (println (format "Total Triples: %,d across %d relations."
                           (:distinct-triples audit-res) (:total-relations audit-res)))
        _ (println (format "Relations with n < 16: %d (%.1f%%)"
                           (get-in audit-res [:rels-lt-16 :count]) (get-in audit-res [:rels-lt-16 :pct])))
        _ (println (format "Relations with n >= 2 (trained): %d"
                           (get-in audit-res [:rels-ge-2-trained :count])))
        _ (println (format "Total InfoNCE Off-Diagonal Positions: %,d"
                           (get-in audit-res [:in-batch-contamination :total-off-diagonals])))
        _ (println (format "  Duplicate-positive rate (self-as-negative): %.2f%%"
                           (get-in audit-res [:in-batch-contamination :duplicate-positive :pct])))
        _ (println (format "  False-negative rate (known triples as negatives): %.2f%%"
                           (get-in audit-res [:in-batch-contamination :false-negative :pct])))
        _ (println (format "  Total Contaminated Rate: %.2f%%"
                           (get-in audit-res [:in-batch-contamination :total-contaminated :pct])))

        ;; Load Checkpoint
        _ (println (format "\nLoading checkpoint from %s..." checkpoint-file))
        ckpt (deserialize-weights (read-string (slurp checkpoint-file)))

        ;; 2. Key-Space Geometry
        _ (println "\n--- Step 2: Key-Space Geometry ---")
        arena (Arena/ofConfined)
        mapped-st (st/map-safetensors-weights ".models/gemma-4-E2B-it/model.safetensors" arena)
        embed-slice (st/get-tensor-slice mapped-st "model.language_model.embed_tokens.weight")

        tokenizer (tok/from-file ".models/gemma-4-E2B-it/tokenizer.json")
        all-entries (mapcat :triples (mapcat (fn [p] (:entries (read-string (slurp p))))
                                             [(:train-file opts) (:test-seen opts) (:test-unseen opts)]))
        all-eval-targets (distinct (concat (map #(nth % 2) (mapcat :triples (:entries (read-string (slurp (:test-seen opts))))))
                                           (map #(nth % 2) (mapcat :triples (:entries (read-string (slurp (:test-unseen opts))))))
                                           (map #(nth % 2) (take 100 all-entries))))
        candidate-targets (vec (distinct (keep (fn [t]
                                                 (let [toks (proto/encode tokenizer (str " " t) false)]
                                                   (when (seq toks)
                                                     {:str t :tok (first toks)})))
                                               all-eval-targets)))
        key-geom-res (run-key-space-geometry embed-slice candidate-targets ckpt all-entries)
        _ (.close arena)

        _ (println (format "Candidate pool: %d entities | Co-typed pairs: %,d | Random pairs: %,d"
                           (:candidate-count key-geom-res)
                           (:co-typed-pair-count key-geom-res)
                           (:non-co-typed-pair-count key-geom-res)))
        _ (println (format "Pre-W_K Embeddings:  Co-typed cos = %.4f ± %.4f | Random cos = %.4f ± %.4f (Δ = %+.4f)"
                           (get-in key-geom-res [:embed-space :co-typed :mean])
                           (get-in key-geom-res [:embed-space :co-typed :std])
                           (get-in key-geom-res [:embed-space :random :mean])
                           (get-in key-geom-res [:embed-space :random :std])
                           (get-in key-geom-res [:embed-space :delta-mean])))
        _ (println (format "Post-W_K Keys:       Co-typed cos = %.4f ± %.4f | Random cos = %.4f ± %.4f (Δ = %+.4f)"
                           (get-in key-geom-res [:key-space :co-typed :mean])
                           (get-in key-geom-res [:key-space :co-typed :std])
                           (get-in key-geom-res [:key-space :random :mean])
                           (get-in key-geom-res [:key-space :random :std])
                           (get-in key-geom-res [:key-space :delta-mean])))

        ;; 3. Inference Session for Steps 3, 4, 5
        _ (println "\nInitializing Gemma 4 E2B session on ROCm for query alignment, training dynamics, and amplification decomposition...")
        session (g4/init-inference-session {:model-dir (:model-dir opts)
                                            :backend (:backend opts)
                                            :precision :bf16
                                            :max-seq-len 64
                                            :targets [:logits :normed_last]})
        dev-weights (g4/allocate-device-weights session)

        ;; Training Dynamics
        _ (println "\n--- Step 3: Training Dynamics ---")
        train-entries (:entries (read-string (slurp train-file)))
        raw-triples (distinct (mapcat :triples train-entries))
        valid-train-triples
        (vec (keep (fn [[h r t]]
                     (let [h-toks (proto/encode tokenizer (str " " h) false)
                           t-toks (proto/encode tokenizer (str " " t) false)]
                       (when (and (seq h-toks) (seq t-toks))
                         {:h-str h :rel r :t-str t
                          :h-tok (first h-toks)
                          :t-tok (first t-toks)})))
                   raw-triples))
        dyn-res (run-training-dynamics (:ctx session) dev-weights valid-train-triples ckpt opts)

        ;; Query Distribution Alignment and Amplification Decomposition
        _ (println "\n--- Step 4 & 5: Query Alignment and Amplification Decomposition ---")
        eval-diag-res (run-evaluation-diagnostics session dev-weights candidate-targets ckpt opts)

        _ (println (format "Seen Query Cosine Alignment (V_h vs h_ctx):   %.4f ± %.4f [min: %.4f, max: %.4f]"
                           (get-in eval-diag-res [:seen-query-alignment :mean])
                           (get-in eval-diag-res [:seen-query-alignment :std])
                           (get-in eval-diag-res [:seen-query-alignment :min])
                           (get-in eval-diag-res [:seen-query-alignment :max])))
        _ (println (format "Unseen Query Cosine Alignment (V_h vs h_ctx): %.4f ± %.4f [min: %.4f, max: %.4f]"
                           (get-in eval-diag-res [:unseen-query-alignment :mean])
                           (get-in eval-diag-res [:unseen-query-alignment :std])
                           (get-in eval-diag-res [:unseen-query-alignment :min])
                           (get-in eval-diag-res [:unseen-query-alignment :max])))

        _ (println "\n--- Amplification Decomposition on TEST_SEEN (N=40) ---")
        decomp (:seen-amplification-decomposition eval-diag-res)
        _ (println (format "Stage 1 (Linear):   Target Shift: %+.4f | Mean Distractor: %+.4f | Max Distractor: %+.4f"
                           (get-in decomp [:s1-target-shift :mean])
                           (get-in decomp [:s1-dist-mean :mean])
                           (get-in decomp [:s1-dist-max :mean])))
        _ (println (format "Resolver Alone:     Target Score: %+.4f | Mean Distractor: %+.4f | Max Distractor: %+.4f"
                           (get-in decomp [:s2-res-target :mean])
                           (get-in decomp [:s2-res-dist-mean :mean])
                           (get-in decomp [:s2-res-dist-max :mean])))
        _ (println (format "Stage 2 Combined:   Target Shift: %+.4f | Mean Distractor: %+.4f | Max Distractor: %+.4f"
                           (get-in decomp [:s2-total-target :mean])
                           (get-in decomp [:s2-total-dist-mean :mean])
                           (get-in decomp [:s2-total-dist-max :mean])))

        full-results
        {:date "2026-09-17"
         :batch-composition-audit audit-res
         :key-space-geometry key-geom-res
         :training-dynamics dyn-res
         :seen-query-alignment (:seen-query-alignment eval-diag-res)
         :unseen-query-alignment (:unseen-query-alignment eval-diag-res)
         :seen-amplification-decomposition decomp}

        out-dir (io/file "catalog/trainable-limits/e18-diagnostics")]
    (.mkdirs out-dir)
    (spit (io/file out-dir "phase0_diagnostics.edn") (pr-str full-results))
    (println (format "\nSaved Phase 0 Diagnostics to %s/phase0_diagnostics.edn" (.getPath out-dir)))
    full-results))

(defn -main [& _args]
  (let [opts {:backend :rocm
              :model-dir ".models/gemma-4-E2B-it"
              :train-file ".dataset/webnlg/train.edn"
              :test-seen ".dataset/webnlg/test_seen.edn"
              :test-unseen ".dataset/webnlg/test_unseen.edn"
              :checkpoint-out ".dataset/webnlg/checkpoint_gemma4_relational.edn"
              :epochs-resolver 5
              :lr-resolver 0.05
              :tau 0.1
              :lambda-mem 1.0
              :lambda-resolve 1.0
              :dim-mem 128
              :k-triples 16
              :max-eval 40}]
    (run-phase0-suite opts)
    (System/exit 0)))
