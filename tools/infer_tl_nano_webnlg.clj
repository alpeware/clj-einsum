(ns tools.infer-tl-nano-webnlg
  "Interactive and One-Shot Inference Driver for TL-Nano (WebNLG Checkpoint).
   Loads saved model weights, compiles OpenXLA PJRT execution graph, and evaluates
   prompt completions in both Pure Neural Mode (Zero Memory) and Grounded Mode (Active Memory)."
  (:require [einsum.core :as xla]
            [models.checkpoint :as ckpt]
            [models.tl-nano :as nano]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :rocm
   :checkpoint-file ".dataset/webnlg/checkpoint_tl_nano.bin"
   :prompt "The birthPlace of Alan Bean is"
   :relation "birthPlace"
   :interactive false
   :model-dir ".models/gpt2"})

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
          "--checkpoint" (recur (subvec remaining 2) (assoc opts :checkpoint-file v))
          "--prompt" (recur (subvec remaining 2) (assoc opts :prompt v))
          "--relation" (recur (subvec remaining 2) (assoc opts :relation v))
          "--interactive" (recur (subvec remaining 1) (assoc opts :interactive true))
          (recur (subvec remaining 1) opts))))))

(defn run-inference-query
  [fwd-exec params cfg bpe->active _active->bpe tokenizer r-maps candidate-targets prompt-str rel-kw]
  (let [seq-len (:max-seq-len cfg)
        vocab-size (:vocab-size cfg)
        dm (:dim-mem cfg)
        ne (:entity-count cfg)

        p-bpe (proto/encode tokenizer prompt-str false)
        p-act (mapv #(get bpe->active % 0) p-bpe)
        p-len (min (dec seq-len) (count p-act))
        sub-act (subvec p-act 0 p-len)
        padded (into sub-act (repeat (- seq-len p-len) 0))
        p-arr (int-array seq-len)
        _ (dotimes [i seq-len] (aset p-arr i (int (nth padded i))))

        t-arr (float-array (* seq-len ne) (float 0.0))
        radj-arr (float-array (* ne ne) (float 0.0))
        row-off (* (dec p-len) vocab-size)

        ;; 1. Zero Memory Neural Baseline
        z-params (assoc params
                        :R_mem (float-array (* dm dm) (float 0.0))
                        :x p-arr
                        :T t-arr
                        :R_adj radj-arr)
        z-out (nano/run-tl-nano-forward! fwd-exec z-params)
        ^floats z-logits (:logits z-out)

        ;; 2. Active Deductive Unbinding
        rel-R (get r-maps rel-kw)
        a-params (assoc params
                        :R_mem (or rel-R (float-array (* dm dm) (float 0.0)))
                        :x p-arr
                        :T t-arr
                        :R_adj radj-arr)
        a-out (nano/run-tl-nano-forward! fwd-exec a-params)
        ^floats a-logits (:logits a-out)

        top-k 5
        extract-top-k
        (fn [^floats logits]
          (let [scored (map (fn [c]
                              {:cand (:str c)
                               :active (:active c)
                               :logit (aget logits (+ row-off (:active c)))})
                            candidate-targets)]
            (take top-k (sort-by :logit > scored))))

        z-top (extract-top-k z-logits)
        a-top (extract-top-k a-logits)]

    {:prompt prompt-str
     :relation rel-kw
     :zero-top z-top
     :active-top a-top
     :zero-best (:cand (first z-top))
     :active-best (:cand (first a-top))}))

(defn display-inference-result [res]
  (println (format "\nPrompt: \"%s\"" (:prompt res)))
  (println (format "Relation: :%s" (name (:relation res))))
  (println "--------------------------------------------------------------------------------")
  (println "Mode                       | Top Prediction     | Top 3 Candidates")
  (println "---------------------------+--------------------+---------------------------------------")
  (println (format "Pure Neural (Zero R_mem)   | %-18s | %s"
                   (:zero-best res)
                   (str/join ", " (map #(format "%s (%.2f)" (:cand %) (:logit %)) (take 3 (:zero-top res))))))
  (println (format "Deductive Grounded (Active)| %-18s | %s"
                   (:active-best res)
                   (str/join ", " (map #(format "%s (%.2f)" (:cand %) (:logit %)) (take 3 (:active-top res))))))
  (println "--------------------------------------------------------------------------------"))

(defn run-webnlg-inference
  [opts]
  (let [{:keys [backend checkpoint-file prompt relation interactive model-dir]} opts
        _ (println "\n================================================================================")
        _ (println "🔮 TL-NANO WEBNLG INFERENCE RUNNER (Dual Neural & Symbolic Reasoning)")
        _ (println "================================================================================")

        ;; 1. Load Model Checkpoint
        _ (println (format "Loading weights from %s..." checkpoint-file))
        ckpt-data (ckpt/load-checkpoint checkpoint-file)
        cfg (:config ckpt-data)
        vocab (:vocab ckpt-data)
        bpe->active (:bpe->active vocab)
        active->bpe (:active->bpe vocab)
        candidate-targets (:entities ckpt-data)
        r-maps (:r-maps ckpt-data)
        params (:params ckpt-data)
        relations (:relations ckpt-data)
        tokenizer (tok/from-file model-dir)

        _ (println (format "Model loaded: %d Layers | Vocab=%d | %d Supported Relations | %d Candidates"
                           (:num-layers cfg) (:vocab-size cfg) (count r-maps) (count candidate-targets)))

        ;; 2. Compile OpenXLA Inference Graph
        ctx (xla/init-backend! backend)
        seq-len (:max-seq-len cfg)
        _ (println "Compiling OpenXLA PJRT Inference Executable (Batch=1)...")
        t0 (System/nanoTime)
        fwd-exec (nano/compile-tl-nano-forward ctx 1 seq-len cfg)
        _ (println (format "Compilation completed in %.2f ms." (/ (- (System/nanoTime) t0) 1e6)))]

    (if-not interactive
      ;; One-shot CLI Mode
      (let [rel-kw (keyword relation)
            res (run-inference-query fwd-exec params cfg bpe->active active->bpe tokenizer r-maps candidate-targets prompt rel-kw)]
        (display-inference-result res))

      ;; Interactive REPL Loop
      (do
        (println "\n💡 Interactive REPL Mode Active.")
        (println "Type a prompt to predict the continuation (e.g. 'The almaMater of Alan Bean is').")
        (println "Commands: ':relations' (list relations), ':exit' (quit)\n")
        (loop []
          (print "\ntlnano> ")
          (flush)
          (let [line (read-line)]
            (when (and line (not= (str/trim line) ":exit"))
              (cond
                (= (str/trim line) ":relations")
                (do
                  (println "\nAvailable Relations:")
                  (doseq [r (sort relations)]
                    (println (format "  :%s" (name r))))
                  (recur))

                (str/blank? line)
                (recur)

                :else
                (do
                  (print "relation (e.g. birthPlace, almaMater, club) [default: first]: ")
                  (flush)
                  (let [rel-input (read-line)
                        rel-kw (if (str/blank? rel-input)
                                 (first relations)
                                 (keyword (str/replace (str/trim rel-input) #"^:" "")))]
                    (try
                      (let [res (run-inference-query fwd-exec params cfg bpe->active active->bpe tokenizer r-maps candidate-targets line rel-kw)]
                        (display-inference-result res))
                      (catch Exception e
                        (println "Inference error:" (.getMessage e))))
                    (recur)))))))))))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (run-webnlg-inference opts)
    (System/exit 0)))
