(ns scripts.train-tl-nano-webnlg
  "Pre-training Driver for TL-Nano on the WebNLG Benchmark.
   Loads .dataset/webnlg/train.edn, constructs an active GPT-2 BPE sub-vocabulary,
   compiles OpenXLA PJRT forward executable on AMD Radeon RX 7900 XTX (ROCm) or CPU,
   executes joint LM + InfoNCE pre-training, and saves model weights via checkpoint.clj."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.models.checkpoint :as ckpt]
            [clj-xla.logic.models.tl-nano :as nano]
            [clj-xla.tokenizer.core :as tok]
            [clj-xla.tokenizer.protocol :as proto]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :rocm
   :epochs 15
   :batch-size 16
   :seq-len 24
   :vocab-size 4096
   :hidden-dim 256
   :num-layers 4
   :num-heads 4
   :head-dim 64
   :intermediate-dim 512
   :dim-mem 64
   :lr 0.03
   :lambda-tl 0.35
   :gamma 5.0
   :lambda-mem 0.8
   :threshold 0.2
   :max-entries 3500
   :data-file ".dataset/webnlg/train.edn"
   :checkpoint-file ".dataset/webnlg/checkpoint_tl_nano.bin"
   :model-dir ".models/gpt2"
   :seed 42})

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
          "--epochs" (recur (subvec remaining 2) (assoc opts :epochs (Long/parseLong v)))
          "--batch-size" (recur (subvec remaining 2) (assoc opts :batch-size (Long/parseLong v)))
          "--seq-len" (recur (subvec remaining 2) (assoc opts :seq-len (Long/parseLong v)))
          "--vocab-size" (recur (subvec remaining 2) (assoc opts :vocab-size (Long/parseLong v)))
          "--hidden-dim" (recur (subvec remaining 2) (assoc opts :hidden-dim (Long/parseLong v)))
          "--num-layers" (recur (subvec remaining 2) (assoc opts :num-layers (Long/parseLong v)))
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--lambda-tl" (recur (subvec remaining 2) (assoc opts :lambda-tl (Double/parseDouble v)))
          "--max-entries" (recur (subvec remaining 2) (assoc opts :max-entries (Long/parseLong v)))
          "--data-file" (recur (subvec remaining 2) (assoc opts :data-file v))
          "--checkpoint" (recur (subvec remaining 2) (assoc opts :checkpoint-file v))
          "--seed" (recur (subvec remaining 2) (assoc opts :seed (Long/parseLong v)))
          (recur (subvec remaining 1) opts))))))

(defn build-active-vocabulary
  "Extracts top `vocab-size` most frequent BPE tokens across corpus sentences and entity texts."
  [tokenizer entries vocab-size]
  (let [sentences (mapcat :sentences entries)
        triples (distinct (mapcat :triples entries))
        entities (distinct (concat (map first triples) (map #(nth % 2) triples)))
        all-texts (concat sentences entities)
        token-freqs (frequencies (mapcat #(proto/encode tokenizer % false) all-texts))
        sorted-bpe (map first (take (dec (long vocab-size)) (sort-by val > token-freqs)))
        pad-id 0
        bpe->active (into {pad-id 0} (map-indexed (fn [idx tok] [tok (inc idx)]) sorted-bpe))
        active->bpe (into {0 pad-id} (map (fn [[b a]] [a b]) bpe->active))]
    {:bpe->active bpe->active
     :active->bpe active->bpe
     :active-count (count bpe->active)}))

(defn prepare-batched-tensors
  "Partitions tokenized sentences into fixed-shape [batch-size, seq-len] tensors."
  [entries tokenizer bpe->active batch-size seq-len]
  (let [l (long seq-len)
        bs (long batch-size)
        sentences (vec (distinct (mapcat :sentences entries)))
        encoded
        (mapv (fn [s]
                (let [raw (proto/encode tokenizer s false)
                      active (mapv #(get bpe->active % 0) raw)
                      len (min l (count active))
                      sub (subvec active 0 len)]
                  (into sub (repeat (- l len) 0))))
              sentences)]
    (mapv (fn [chunk]
            (let [b-tokens (int-array (* bs l))
                  b-targets (int-array (* bs l))]
              (dotimes [bi (count chunk)]
                (let [sent (nth chunk bi)]
                  (dotimes [pos l]
                    (let [tok (nth sent pos)
                          next-tok (if (< pos (dec l)) (nth sent (inc pos)) 0)]
                      (aset b-tokens (+ (* bi l) pos) (int tok))
                      (aset b-targets (+ (* bi l) pos) (int next-tok))))))
              {:tokens b-tokens
               :targets b-targets
               :batch-size (count chunk)}))
          (partition bs encoded))))

(defn run-webnlg-pretraining
  [opts]
  (let [{:keys [backend epochs batch-size seq-len vocab-size hidden-dim
                num-layers num-heads head-dim intermediate-dim dim-mem
                lr lambda-tl gamma lambda-mem threshold max-entries
                data-file checkpoint-file model-dir seed]} opts
        _ (println "\n================================================================================")
        _ (println "🌐 TL-NANO WEBNLG BENCHMARK PRE-TRAINING (AMD ROCm / OpenXLA)")
        _ (println "================================================================================")
        _ (println (format "Backend: [%s] | Epochs: %d | Batch: %d | SeqLen: %d | Vocab: %d | LR: %.3f"
                           (name backend) epochs batch-size seq-len vocab-size lr))
        _ (println (format "Architecture: %d Layers | D=%d | H=%d | dh=%d | D_ff=%d | D_mem=%d"
                           num-layers hidden-dim num-heads head-dim intermediate-dim dim-mem))

        ;; 1. Load WebNLG Data
        data-f (io/file data-file)
        _ (when-not (.exists data-f)
            (throw (ex-info (str "WebNLG data file not found: " data-file) {:path data-file})))
        raw-data (read-string (slurp data-f))
        all-entries (:entries raw-data)
        selected-entries (if max-entries (vec (take max-entries all-entries)) all-entries)
        tokenizer (tok/from-file model-dir)

        _ (println (format "Loaded %,d WebNLG entries (using %,d entries, %,d sentences)."
                           (count all-entries) (count selected-entries)
                           (count (mapcat :sentences selected-entries))))

        ;; 2. Active Vocabulary & Batches
        {:keys [bpe->active active->bpe active-count]}
        (build-active-vocabulary tokenizer selected-entries vocab-size)
        _ (println (format "Active Sub-Vocabulary built: %d unique BPE tokens (space: [0, %d))."
                           active-count vocab-size))

        batches (prepare-batched-tensors selected-entries tokenizer bpe->active batch-size seq-len)
        _ (println (format "Constructed %,d training batches of size %d (seq-len=%d)."
                           (count batches) batch-size seq-len))

        ;; 3. Candidate Entities & Relational Triples
        raw-triples (distinct (mapcat :triples selected-entries))
        relations (vec (distinct (map second raw-triples)))
        entity-strings (vec (distinct (concat (map first raw-triples) (map #(nth % 2) raw-triples))))
        ne (count entity-strings)

        candidate-targets
        (vec (distinct
              (keep (fn [ent]
                      (let [toks (proto/encode tokenizer (str " " ent) false)
                            first-tok (first toks)
                            act (get bpe->active first-tok)]
                        (when act
                          {:str ent :bpe first-tok :active act})))
                    (take 64 (distinct (map #(nth % 2) raw-triples))))))
        nc (count candidate-targets)
        _ (println (format "Ground-Truth Triples: %,d | Relations: %d | Candidate Targets: %d"
                           (count raw-triples) (count relations) nc))

        triples-by-rel
        (reduce (fn [acc [h rel t]]
                  (let [h-bpe (first (proto/encode tokenizer (str " " h) false))
                        t-bpe (first (proto/encode tokenizer (str " " t) false))
                        h-act (get bpe->active h-bpe)
                        t-act (get bpe->active t-bpe)]
                    (if (and h-act t-act)
                      (update acc rel (fnil conj []) {:head h-act :tail t-act :head-str h :tail-str t})
                      acc)))
                {}
                raw-triples)

        ;; 4. OpenXLA PJRT Compilation
        ctx (xla/init-backend! backend)
        _ (println (format "OpenXLA PJRT Context initialized on %s."
                           (name (or (:backend ctx) (:platform-name ctx) backend))))

        cfg (nano/tl-nano-config
             {:vocab-size vocab-size
              :hidden-dim hidden-dim
              :intermediate-dim intermediate-dim
              :num-layers num-layers
              :num-heads num-heads
              :head-dim head-dim
              :dim-mem dim-mem
              :entity-count ne
              :cand-count nc
              :hybrid-layers (if (>= num-layers 4) #{1 3} #{0})
              :max-seq-len seq-len
              :lambda-tl lambda-tl
              :gamma gamma
              :lambda-mem lambda-mem
              :threshold threshold})

        _ (println "\nCompiling OpenXLA PJRT Training Executables...")
        t-c0 (System/nanoTime)
        fwd-train-exec (nano/compile-tl-nano-forward ctx batch-size seq-len cfg)
        embed-update-exec (nano/compile-embedding-update ctx batch-size seq-len cfg)
        t-comp (/ (- (System/nanoTime) t-c0) 1e6)
        _ (println (format "OpenXLA Graph Compilation completed in %.2f ms." t-comp))

        ;; 5. Initialize Parameters & Relational Transition Matrices R_r
        rnd (java.util.Random. (long seed))
        base-params (nano/init-tl-nano-params cfg rnd)
        d (long hidden-dim)
        dm (long dim-mem)

        r-maps (atom (into {} (map (fn [r]
                                     [r (let [arr (float-array (* dm dm))]
                                          (dotimes [i dm]
                                            (aset arr (+ (* i dm) i) (float 1.0)))
                                          (dotimes [i (* dm dm)]
                                            (aset arr i (float (+ (aget arr i) (* (.nextGaussian rnd) 0.05)))))
                                          arr)])
                                   relations)))

        sync-e-cand (fn [params]
                      (let [^floats w-emb (:W_embed params)
                            e-arr (float-array (* nc d))]
                        (dotimes [ci nc]
                          (let [c-act (:active (nth candidate-targets ci))
                                src-off (* (long c-act) d)
                                dst-off (* ci d)]
                            (System/arraycopy w-emb src-off e-arr dst-off d)))
                        (assoc params :E_cand e-arr)))

        ;; 6. Pre-training Loop
        _ (println (format "\n--- Commencing WebNLG Pre-training (%d Epochs, %,d Batches/Epoch) ---"
                           epochs (count batches)))
        _ (println "Epoch | L_total | L_LM   | L_InfoNCE | Epoch Time | Throughput")
        _ (println "------+---------+--------+-----------+------------+-----------")

        t-train-start (System/nanoTime)
        trace-acc (atom [])
        total-toks-per-epoch (* (long (count batches)) (long batch-size) (long seq-len))

        _ (println "Pinning static layer weights into PJRT device VRAM...")
        pinned-base (nano/pin-params-in-vram ctx (sync-e-cand base-params) cfg)

        final-params
        (loop [epoch 1
               curr-params pinned-base]
          (if (> epoch epochs)
            curr-params
            (let [t-ep0 (System/nanoTime)
                  ep-tot (atom 0.0)
                  ep-lm (atom 0.0)
                  ep-info (atom 0.0)

                  updated-params
                  (reduce
                   (fn [p-acc [b-idx b]]
                     (let [rel (nth relations (mod (+ (* epoch (count batches)) b-idx) (count relations)))
                           rel-triples (get triples-by-rel rel [])
                           num-pos (count rel-triples)
                           effective-triples
                           (cond
                             (zero? num-pos) []
                             (>= num-pos 4) (vec (take 16 rel-triples))
                             :else
                             (let [first-t (first rel-triples)
                                   h (:head first-t)
                                   t (:tail first-t)
                                   neg-pool (filterv #(not= (:active %) t) candidate-targets)
                                   num-needed (- 4 num-pos)
                                   neg-triples (if (pos? (count neg-pool))
                                                 (mapv (fn [_] {:head h :tail (:active (nth neg-pool (.nextInt rnd (count neg-pool))))})
                                                       (range num-needed))
                                                 [])]
                               (into (vec rel-triples) neg-triples)))
                           curr-R (get @r-maps rel)
                           p-with-r (assoc p-acc :R_mem curr-R)
                           batch-with-tr (assoc b :triples effective-triples :pos-count num-pos)
                           next-p (nano/train-step fwd-train-exec p-with-r batch-with-tr cfg lr embed-update-exec)
                           loss-info (:loss next-p)]
                       (when (:R_mem next-p)
                         (swap! r-maps assoc rel (:R_mem next-p)))
                       (swap! ep-tot + (:total-loss loss-info))
                       (swap! ep-lm + (:lm-loss loss-info))
                       (swap! ep-info + (:infonce-loss loss-info))
                       (sync-e-cand next-p)))
                   curr-params
                   (map-indexed vector batches))

                  t-ep (/ (- (System/nanoTime) t-ep0) 1e6)
                  n-b (double (count batches))
                  avg-tot (/ @ep-tot n-b)
                  avg-lm (/ @ep-lm n-b)
                  avg-info (/ @ep-info n-b)
                  tok-s (* (/ (double total-toks-per-epoch) t-ep) 1000.0)]

              (when (or (<= epoch 5) (zero? (mod epoch 5)) (= epoch epochs))
                (println (format "%5d | %7.4f | %6.4f | %9.4f | %7.1f ms | %7.0f tok/s"
                                 epoch avg-tot avg-lm avg-info t-ep tok-s)))
              (swap! trace-acc conj {:epoch epoch
                                     :loss avg-tot
                                     :lm-loss avg-lm
                                     :info-loss avg-info
                                     :time-ms t-ep
                                     :tok-s tok-s})
              (recur (inc epoch) updated-params))))

        t-train-total (/ (- (System/nanoTime) t-train-start) 1e6)
        train-seconds (/ t-train-total 1000.0)
        trace @trace-acc
        init-loss (:loss (first trace))
        final-loss (:loss (last trace))
        loss-drop (- init-loss final-loss)
        avg-throughput (/ (reduce + (map :tok-s trace)) (double (count trace)))

        _ (println (format "\nPre-training completed in %.2f s (%.2f ms) | Mean Throughput: %.0f tok/s"
                           train-seconds t-train-total avg-throughput))
        _ (println (format "Loss descent: %.4f --> %.4f (Drop: %.4f)" init-loss final-loss loss-drop))

        ;; 7. Save Checkpoint
        _ (println (format "\nSaving model checkpoint to %s..." checkpoint-file))
        ckpt-data {:config cfg
                   :metadata {:dataset "WebNLG v3.0 (en)"
                              :entries-count (count selected-entries)
                              :epochs epochs
                              :train-time-s train-seconds
                              :init-loss init-loss
                              :final-loss final-loss
                              :loss-drop loss-drop
                              :throughput-tok-s avg-throughput
                              :timestamp (str (java.time.Instant/now))}
                   :vocab {:bpe->active bpe->active
                           :active->bpe active->bpe}
                   :entities candidate-targets
                   :relations relations
                   :r-maps @r-maps
                   :params (merge base-params (select-keys final-params [:W_embed :W_mem :R_mem :E_cand]))}
        _ (nano/free-pinned-params! final-params)
        bytes-written (ckpt/save-checkpoint checkpoint-file ckpt-data)]

    (println (format "✅ Checkpoint successfully written: %,d bytes (%.2f MB)"
                     bytes-written (/ (double bytes-written) 1048576.0)))
    (println "Pre-training phase complete. Ready for evaluation via scripts.eval-tl-nano-webnlg.")
    {:train-seconds train-seconds
     :init-loss init-loss
     :final-loss final-loss
     :loss-drop loss-drop
     :avg-throughput avg-throughput
     :checkpoint checkpoint-file
     :bytes-written bytes-written}))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-webnlg-pretraining opts)
    (System/exit 0)))
