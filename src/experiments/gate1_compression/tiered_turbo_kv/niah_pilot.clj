(ns experiments.gate1-compression.tiered-turbo-kv.niah-pilot
  "Model-in-the-loop NIAH pilot at 16k and 32k context on Gemma 4 E4B INT4 with Fast-TurboQuant 2-Bit.
   Evaluates empirical needle retrieval accuracy across 10 depth bins (10% to 100%) at 16k and 32k lengths."
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.string :as str]
            [einsum.models.gemma4.runtime :as runtime]
            [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as tok-proto]))

(def TEST-NEEDLES
  [{:depth-bin 0 :depth 0.10 :key "CRIMSON-741"   :needle "NOTE: The special secret code is CRIMSON-741."}
   {:depth-bin 1 :depth 0.20 :key "EMERALD-825"   :needle "NOTE: The special secret code is EMERALD-825."}
   {:depth-bin 2 :depth 0.30 :key "SAPPHIRE-394"  :needle "NOTE: The special secret code is SAPPHIRE-394."}
   {:depth-bin 3 :depth 0.40 :key "OBSIDIAN-612"  :needle "NOTE: The special secret code is OBSIDIAN-612."}
   {:depth-bin 4 :depth 0.50 :key "PLATINUM-508"  :needle "NOTE: The special secret code is PLATINUM-508."}
   {:depth-bin 5 :depth 0.60 :key "TITANIUM-937"  :needle "NOTE: The special secret code is TITANIUM-937."}
   {:depth-bin 6 :depth 0.70 :key "AMETHYST-246"  :needle "NOTE: The special secret code is AMETHYST-246."}
   {:depth-bin 7 :depth 0.80 :key "TURQUOISE-159" :needle "NOTE: The special secret code is TURQUOISE-159."}
   {:depth-bin 8 :depth 0.90 :key "DIAMOND-483"   :needle "NOTE: The special secret code is DIAMOND-483."}
   {:depth-bin 9 :depth 1.00 :key "MALACHITE-720" :needle "NOTE: The special secret code is MALACHITE-720."}])

(def QUERY-PROMPT "Question: What is the special secret code?\nAnswer: The special secret code is ")

(defn build-haystack-prompt
  "Constructs a deterministic synthetic haystack text of `target-len` tokens
   with the needle inserted at `needle-depth` fraction [0.0, 1.0]."
  [tokenizer target-len needle-depth needle query]
  (let [filler "The quick brown fox jumps over the lazy dog. Immutable persistent data structures ensure safe concurrency in modern computing systems across distributed architectures. "
        filler-ids (vec (tok-proto/encode tokenizer filler))
        filler-count (count filler-ids)
        target-fill-tokens (max 100 (- target-len 200))
        num-reps (+ 50 (quot target-fill-tokens filler-count))
        full-text (apply str (repeat num-reps filler))
        all-ids (vec (tok-proto/encode tokenizer full-text))
        actual-fill (min (count all-ids) target-fill-tokens)
        clamped-ids (subvec all-ids 0 actual-fill)
        split-point (min (count clamped-ids) (long (* (double needle-depth) (double actual-fill))))
        pre-ids (subvec clamped-ids 0 split-point)
        post-ids (subvec clamped-ids split-point)
        pre-text (tok-proto/decode tokenizer pre-ids)
        post-text (tok-proto/decode tokenizer post-ids)]
    (str pre-text "\n" needle "\n" post-text "\n" query)))

(defn evaluate-needle-output
  "Evaluates generated text for retrieval of needle key.
   Returns map with :exact-match?, :prefix-match?, and normalized output strings."
  [gen-text needle-key]
  (let [clean-gen (str/replace (or gen-text "") "<turn|>" "")
        gen-upper (str/upper-case clean-gen)
        key-upper (str/upper-case needle-key)
        prefix (first (str/split key-upper #"-"))
        suffix (second (str/split key-upper #"-"))
        exact? (str/includes? gen-upper key-upper)
        prefix? (or (str/includes? gen-upper prefix)
                    (when suffix (str/includes? gen-upper suffix)))]
    {:exact-match? exact?
     :prefix-match? prefix?
     :pass? exact?
     :output-clean (str/trim clean-gen)}))

(defn run-niah-pilot-length
  "Runs 10 depth bin samples for a single context length."
  [session tokenizer seq-len]
  (let [target-tokens (- seq-len 500)
        _ (println (format "\n>>> Starting NIAH Pilot for Length %d (Target Prompt: ~%d tokens) <<<" seq-len target-tokens))]
    (mapv
     (fn [{:keys [depth-bin depth key needle]}]
       (let [prompt (build-haystack-prompt tokenizer target-tokens depth needle QUERY-PROMPT)
             prompt-len (count (tok-proto/encode tokenizer prompt))
             t0 (System/nanoTime)
             res (runtime/generate-new-tokens-and-text session prompt)
             t1 (System/nanoTime)
             elapsed-s (/ (- t1 t0) 1e9)
             gen-text (or (:text res) "")
             eval-res (evaluate-needle-output gen-text key)]
         (println (format "  [Bin %d | Depth %.2f | %5d tok] Key: %-13s -> Match: %-5s | Output: %s (%.1fs)"
                          depth-bin depth prompt-len key (str (:pass? eval-res)) (pr-str (:output-clean eval-res)) elapsed-s))
         (merge {:seq-len seq-len
                 :depth-bin depth-bin
                 :depth depth
                 :key key
                 :prompt-tokens prompt-len
                 :elapsed-s elapsed-s
                 :prefill-ms (double (or (:prefill-ms res) 0.0))
                 :decode-ms (double (or (:decode-ms res) 0.0))
                 :decode-tok-s (double (or (:decode-tok-s res) 0.0))
                 :raw-output gen-text}
                eval-res)))
     TEST-NEEDLES)))

(defn run-niah-pilot
  "Runs the complete 20-sample model-in-the-loop NIAH pilot at 16k and 32k."
  ([] (run-niah-pilot {}))
  ([user-opts]
   (println "==================================================================")
   (println "=== Model-in-the-Loop NIAH Pilot on AMD RX 7900 XTX (E4B+TQ) ===")
   (println "==================================================================")
   (let [lengths (or (:lengths user-opts) [16384 32768])
         model-path (or (:model user-opts) ".models/gemma-4-e4b-it-qat-int4")
         tok (tok/from-file model-path)
         results-by-len
         (mapv
          (fn [len]
            (let [session-opts (merge {:backend :rocm
                                       :target :rocm
                                       :model model-path
                                       :kv-quant :turboquant
                                       :turboquant-kv? true
                                       :max-seq-len len
                                       :max-new-tokens 15}
                                      user-opts)
                  session (runtime/init-agent-vram-session session-opts len)
                  samples (run-niah-pilot-length session tok len)
                  exact-passes (count (filter :exact-match? samples))
                  prefix-passes (count (filter :prefix-match? samples))
                  exact-acc (/ (double exact-passes) (double (count samples)))
                  prefix-acc (/ (double prefix-passes) (double (count samples)))]
              {:seq-len len
               :samples-count (count samples)
               :exact-passes exact-passes
               :exact-accuracy exact-acc
               :prefix-passes prefix-passes
               :prefix-accuracy prefix-acc
               :samples samples}))
          lengths)
         total-samples (reduce + (map :samples-count results-by-len))
         total-exact (reduce + (map :exact-passes results-by-len))
         total-prefix (reduce + (map :prefix-passes results-by-len))
         summary {:pilot "model-in-the-loop-niah-16k-32k"
                  :model model-path
                  :kv-cache "Fast-TurboQuant 2-Bit"
                  :total-samples total-samples
                  :total-exact-passes total-exact
                  :overall-exact-accuracy (/ (double total-exact) (double total-samples))
                  :total-prefix-passes total-prefix
                  :overall-prefix-accuracy (/ (double total-prefix) (double total-samples))
                  :results-by-length results-by-len}]
     (println "\n==================================================================")
     (println "=== Model-in-the-Loop NIAH Pilot Summary ===")
     (println "==================================================================")
     (doseq [r results-by-len]
       (println (format "  • %5d Tokens: %2d/%2d Exact Match (%.1f%%) | %2d/%2d Prefix/Component Match (%.1f%%)"
                        (:seq-len r) (:exact-passes r) (:samples-count r) (* 100.0 (:exact-accuracy r))
                        (:prefix-passes r) (:samples-count r) (* 100.0 (:prefix-accuracy r)))))
     (println (format "Overall: %d/%d Exact Match (%.1f%%) | %d/%d Prefix/Component Match (%.1f%%)"
                      total-exact total-samples (* 100.0 (/ (double total-exact) (double total-samples)))
                      total-prefix total-samples (* 100.0 (/ (double total-prefix) (double total-samples)))))
     (println "==================================================================")
     summary)))

(defn -main [& _args]
  (let [summary (run-niah-pilot)
        out-file (io/file "resources/proposals/gate1_compression/tiered_turbo_kv/niah_pilot_results.edn")]
    (.mkdirs (.getParentFile out-file))
    (spit out-file (with-out-str (pprint summary)))
    (println (format "Saved pilot results to [%s]" (.getPath out-file))))
  (System/exit 0))
