(ns tools.poc-tl-nano-real-data
  "Real-World Data Pre-training & Neuro-Symbolic Cloze Evaluation of TL-Nano.
   Trains TL-Nano on curated real-world knowledge (Wikipedia facts, tech entities, relations)
   pairing natural language autoregressive LM modeling with in-graph InfoNCE relational unbinding
   using OpenXLA PJRT on AMD Radeon RX 7900 XTX (ROCm) or CPU.
   Evaluates zero-shot cloze question answering with vs without relational memory unbinding."
  (:require [einsum.core :as xla]
            [einsum.models.tl-nano :as nano]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :rocm
   :epochs 25
   :batch-size 3
   :seq-len 24
   :vocab-size 512
   :hidden-dim 256
   :num-layers 4
   :num-heads 4
   :head-dim 64
   :intermediate-dim 512
   :dim-mem 64
   :lr 0.03
   :lambda-tl 0.4
   :gamma 5.0
   :lambda-mem 0.8
   :threshold 0.2
   :corpus-file "data/wikifacts_corpus.edn"
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
          "--lr" (recur (subvec remaining 2) (assoc opts :lr (Double/parseDouble v)))
          "--lambda-tl" (recur (subvec remaining 2) (assoc opts :lambda-tl (Double/parseDouble v)))
          "--gamma" (recur (subvec remaining 2) (assoc opts :gamma (Double/parseDouble v)))
          "--lambda-mem" (recur (subvec remaining 2) (assoc opts :lambda-mem (Double/parseDouble v)))
          "--threshold" (recur (subvec remaining 2) (assoc opts :threshold (Double/parseDouble v)))
          "--corpus-file" (recur (subvec remaining 2) (assoc opts :corpus-file v))
          "--model-dir" (recur (subvec remaining 2) (assoc opts :model-dir v))
          "--seed" (recur (subvec remaining 2) (assoc opts :seed (Long/parseLong v)))
          (recur (subvec remaining 1) opts))))))

(defn build-active-vocab
  "Constructs dense active-token mapping for all tokens present across sentences, prompts, and entities."
  [tokenizer corpus-data]
  (let [all-texts (concat (:sentences corpus-data)
                          (map :prompt (:eval_prompts corpus-data))
                          (map :target (:eval_prompts corpus-data))
                          (:entities corpus-data))
        all-bpe (distinct (mapcat #(proto/encode tokenizer % false) all-texts))
        pad-id 0
        bpe->active (into {pad-id 0} (map-indexed (fn [idx bpe-tok] [bpe-tok (inc idx)]) all-bpe))
        active->bpe (into {0 pad-id} (map (fn [[b a]] [a b]) bpe->active))]
    {:bpe->active bpe->active
     :active->bpe active->bpe
     :active-count (count bpe->active)}))

(defn prepare-batches
  "Tokenizes sentences, pads to seq-len, and constructs batched tensors."
  [sentences tokenizer bpe->active batch-size seq-len]
  (let [l (long seq-len)
        bs (long batch-size)
        encoded-sentences
        (mapv (fn [s]
                (let [toks (mapv bpe->active (proto/encode tokenizer s false))
                      len (min l (count toks))
                      sub (subvec toks 0 len)]
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
          (partition bs encoded-sentences))))

(defn run-real-data-training
  [opts]
  (let [{:keys [backend epochs batch-size seq-len vocab-size hidden-dim
                num-layers num-heads head-dim intermediate-dim dim-mem
                lr lambda-tl gamma lambda-mem threshold corpus-file model-dir seed]} opts
        _ (println "\n================================================================================")
        _ (println "🧠 TL-NANO REAL-WORLD DATA PRE-TRAINING & NEURO-SYMBOLIC BENCHMARK")
        _ (println "================================================================================")
        _ (println (format "Backend: [%s] | Epochs: %d | Batch: %d | SeqLen: %d | LR: %.3f | Lambda-TL: %.2f"
                           (name backend) epochs batch-size seq-len lr lambda-tl))
        _ (println (format "Deductive Params: gamma=%.1f | lambda-mem=%.2f | threshold=%.2f"
                           gamma lambda-mem threshold))

        ;; 1. Load Data & Tokenizer
        corpus-data (read-string (slurp corpus-file))
        tokenizer (tok/from-file model-dir)
        {:keys [bpe->active active->bpe active-count]} (build-active-vocab tokenizer corpus-data)
        _ (println (format "Loaded corpus: %d sentences, %d entities, %d relations, %d triples, %d eval prompts."
                           (count (:sentences corpus-data)) (count (:entities corpus-data))
                           (count (:relations corpus-data)) (count (:triples corpus-data))
                           (count (:eval_prompts corpus-data))))
        _ (println (format "Real GPT-2 Active Sub-Vocabulary: %d unique BPE tokens (dense space: [0, %d))."
                           active-count vocab-size))

        ;; 2. Partition Batches & Triples
        batches (prepare-batches (:sentences corpus-data) tokenizer bpe->active batch-size seq-len)
        relations (:relations corpus-data)
        entity-list (:entities corpus-data)
        ne (count entity-list)

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
                (:triples corpus-data))

        eval-prompts (:eval_prompts corpus-data)
        unique-cands
        (vec (distinct
              (mapv (fn [ep]
                      (let [t-bpe (first (proto/encode tokenizer (str " " (:target ep)) false))]
                        {:str (:target ep)
                         :bpe t-bpe
                         :active (get bpe->active t-bpe)}))
                    eval-prompts)))
        nc (count unique-cands)
        _ (println (format "Candidate Entity Set: %d target concepts for closed-world deductive grounding." nc))

        ;; 3. OpenXLA PJRT Compilation
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

        _ (println "\nCompiling OpenXLA PJRT Executables...")
        t-c0 (System/nanoTime)
        fwd-train-exec (nano/compile-tl-nano-forward ctx batch-size seq-len cfg)
        fwd-eval-exec (nano/compile-tl-nano-forward ctx 1 seq-len cfg)
        t-comp (/ (- (System/nanoTime) t-c0) 1e6)
        _ (println (format "OpenXLA Graph Compilation (Train B=%d + Eval B=1) completed in %.2f ms."
                           batch-size t-comp))

        ;; 4. Initialize Trainable Weights
        rnd (java.util.Random. (long seed))
        base-params (nano/init-tl-nano-params cfg rnd)
        d (long hidden-dim)
        dm (long dim-mem)

        ;; Relational Memory Operators R_rel for each relation
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
                          (let [c-act (:active (nth unique-cands ci))
                                src-off (* (long c-act) d)
                                dst-off (* ci d)]
                            (System/arraycopy w-emb src-off e-arr dst-off d)))
                        (assoc params :E_cand e-arr)))

        ;; 5. Pre-training Loop
        _ (println (format "\n--- Commencing Pre-training (%d Epochs, %d Batches/Epoch) ---"
                           epochs (count batches)))
        _ (println "Epoch | L_total | L_LM   | L_InfoNCE | Epoch Time | Throughput")
        _ (println "------+---------+--------+-----------+------------+-----------")

        t-train-start (System/nanoTime)
        trace-acc (atom [])
        total-toks-per-epoch (* (long (count batches)) (long batch-size) (long seq-len))

        final-params
        (loop [epoch 1
               curr-params (sync-e-cand base-params)]
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
                           curr-R (get @r-maps rel)
                           p-with-r (assoc p-acc :R_mem curr-R)
                           batch-with-tr (assoc b :triples rel-triples)
                           loss-info (nano/compute-joint-loss fwd-train-exec p-with-r batch-with-tr cfg)
                           next-p (nano/train-step fwd-train-exec p-with-r batch-with-tr cfg lr)]
                       (swap! r-maps assoc rel (:R_mem next-p))
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

        _ (println (format "\nPre-training finished in %.2f s (%.2f ms) | Mean Throughput: %.0f tok/s"
                           train-seconds t-train-total avg-throughput))
        _ (println (format "Loss descent: %.4f --> %.4f (Drop: %.4f)" init-loss final-loss loss-drop))

        ;; 6. Cloze Evaluation across Held-out Prompts
        _ (println "\n================================================================================")
        _ (println "🎯 HELD-OUT CLOZE QUESTION ANSWERING EVALUATION (18 Prompts)")
        _ (println "================================================================================")
        _ (println "Prompt                         | Target          | Zero Top-1     | Active Top-1   | Deductive? | Delta Logit")
        _ (println "-------------------------------+-----------------+----------------+----------------+------------+------------")

        eval-results
        (mapv (fn [ep]
                (let [p-str (:prompt ep)
                      rel (:rel ep)
                      head-str (:head ep)
                      t-str (:target ep)
                      head-idx (.indexOf entity-list head-str)

                      p-bpe (proto/encode tokenizer p-str false)
                      p-act (mapv bpe->active p-bpe)
                      p-len (count p-act)
                      padded-p (into p-act (repeat (- seq-len p-len) 0))
                      p-arr (int-array seq-len)
                      _ (dotimes [i seq-len] (aset p-arr i (int (nth padded-p i))))

                      t-arr (float-array (* seq-len ne) (float 0.0))
                      radj-arr (float-array (* ne ne) (float 0.0))
                      _ (when (>= head-idx 0)
                          (aset radj-arr (+ (* head-idx ne) head-idx) (float 1.0))
                          (aset t-arr (+ (* (dec p-len) ne) head-idx) (float 1.0))
                          (let [head-bpe (proto/encode tokenizer (str " " head-str) false)
                                head-act (set (mapv bpe->active head-bpe))]
                            (dotimes [pos (dec p-len)]
                              (when (head-act (nth p-act pos))
                                (aset t-arr (+ (* pos ne) head-idx) (float 1.0))))))

                      tgt-bpe (first (proto/encode tokenizer (str " " t-str) false))
                      tgt-act (get bpe->active tgt-bpe)
                      row-off (* (dec p-len) vocab-size)

                      ;; Baseline (Zero R_mem, No Knowledge Grounding)
                      z-params (assoc final-params
                                      :R_mem (float-array (* dm dm) (float 0.0))
                                      :x p-arr
                                      :T (float-array (* seq-len ne) (float 0.0))
                                      :R_adj (float-array (* ne ne) (float 0.0)))
                      z-out (nano/run-tl-nano-forward! fwd-eval-exec z-params)
                      ^floats z-logits (:logits z-out)
                      z-tgt-l (aget z-logits (+ row-off tgt-act))
                      z-cand-act (:active (apply max-key (fn [c] (aget z-logits (+ row-off (:active c)))) unique-cands))
                      z-cand-str (proto/decode tokenizer [(get active->bpe z-cand-act)])

                      ;; Tensor Logic Deductive Grounding (Active R_mem + KG Attention)
                      rel-R (get @r-maps rel)
                      a-params (assoc final-params
                                      :R_mem rel-R
                                      :x p-arr
                                      :T t-arr
                                      :R_adj radj-arr)
                      a-out (nano/run-tl-nano-forward! fwd-eval-exec a-params)
                      ^floats a-logits (:logits a-out)
                      a-tgt-l (aget a-logits (+ row-off tgt-act))
                      a-cand-act (:active (apply max-key (fn [c] (aget a-logits (+ row-off (:active c)))) unique-cands))
                      a-cand-str (proto/decode tokenizer [(get active->bpe a-cand-act)])

                      delta-l (- (double a-tgt-l) (double z-tgt-l))
                      match? (= a-cand-act tgt-act)]

                  (println (format "%-30s | %-15s | %-14s | %-14s | %-10s | %+.4f"
                                   (if (> (count p-str) 30) (str (subs p-str 0 27) "...") p-str)
                                   t-str z-cand-str a-cand-str (if match? "YES ✅" "NO ❌") delta-l))
                  {:prompt p-str :target t-str :match? match? :delta-logit delta-l}))
              eval-prompts)

        correct-count (count (filter :match? eval-results))
        total-eval (count eval-results)
        acc-pct (* 100.0 (/ (double correct-count) total-eval))
        avg-delta (/ (reduce + (map :delta-logit eval-results)) (double total-eval))]

    (println "------------------------------------------------------------------------------------------------")
    (println (format "Deductive Cloze QA Accuracy: %d / %d (%.1f%%)" correct-count total-eval acc-pct))
    (println (format "Mean Target Logit Shift: %+.4f" avg-delta))
    (println (format "Total Benchmark Training Time: %.2f seconds (%.2f ms)" train-seconds t-train-total))
    (println "================================================================================\n")

    {:backend backend
     :epochs epochs
     :train-time-s train-seconds
     :train-time-ms t-train-total
     :avg-throughput avg-throughput
     :initial-loss init-loss
     :final-loss final-loss
     :loss-drop loss-drop
     :accuracy-pct acc-pct
     :mean-delta-logit avg-delta
     :eval-results eval-results}))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-real-data-training opts)
    (System/exit 0)))
