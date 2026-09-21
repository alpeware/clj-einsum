(ns tools.align-webnlg-entities
  "Token-to-Entity Alignment Pipeline & Verification Inspector for WebNLG.
   Maps human reference sentences to knowledge graph triples to construct
   token-to-entity projection matrix T [L, E] and entity adjacency matrix R_adj [E, E]
   for Pedro Domingos' KG-Attention routing (gamma * T @ R_adj @ T^T)."
  (:require [einsum.runtime.tokenizer.core :as tok]
            [einsum.runtime.tokenizer.protocol :as proto]
            [clojure.string :as str])
  (:gen-class))

(defn normalize-entity-text
  "Strips trailing qualifiers, underscores, and punctuation from entity string."
  [s]
  (let [cleaned (-> (or s "")
                    (str/replace #"_" " ")
                    (str/replace #"\s*\([^)]*\)" "") ;; Remove (director), (actor), etc.
                    (str/replace #"\s*\"" "")
                    str/trim)]
    cleaned))

(defn find-char-spans
  "Finds start and end character offsets of needle in haystack (case-insensitive)."
  [^String haystack ^String needle]
  (if (or (str/blank? haystack) (str/blank? needle))
    []
    (let [lower-hay (str/lower-case haystack)
          lower-needle (str/lower-case needle)
          n-len (.length lower-needle)]
      (loop [from 0 spans []]
        (let [idx (.indexOf lower-hay lower-needle (int from))]
          (if (neg? idx)
            spans
            (recur (+ idx n-len) (conj spans [idx (+ idx n-len)]))))))))

(defn tokenize-with-spans
  "Tokenizes text with BPE tokenizer and reconstructs character spans for each token."
  [tokenizer text]
  (let [tokens (proto/encode tokenizer text false)
        decoded-tokens (mapv #(proto/decode tokenizer [%]) tokens)]
    (loop [tok-idx 0
           char-cursor 0
           result []]
      (if (>= tok-idx (count decoded-tokens))
        result
        (let [tok-str (nth decoded-tokens tok-idx)
              ;; Find where this token string begins at or after char-cursor
              trimmed-tok (str/trim tok-str)
              found-idx (if (empty? trimmed-tok)
                          char-cursor
                          (let [idx (.indexOf text trimmed-tok (int char-cursor))]
                            (if (neg? idx) char-cursor idx)))
              end-idx (+ found-idx (count trimmed-tok))]
          (recur (inc tok-idx)
                 end-idx
                 (conj result {:token-id (nth tokens tok-idx)
                               :token-str tok-str
                               :start found-idx
                               :end end-idx
                               :idx tok-idx})))))))

(defn align-entities-to-tokens
  "Aligns triple entities (heads and tails) to token index spans in tokenized sentence."
  [tokenizer sentence entities]
  (let [token-info (tokenize-with-spans tokenizer sentence)]
    (mapv
     (fn [ent]
       (let [raw-ent ent
             norm-ent (normalize-entity-text ent)
             base-ent (first (str/split norm-ent #",")) ;; e.g. "Andrews" from "Andrews, Texas"

             ;; Try matches in descending order of specificity
             char-spans (let [s1 (find-char-spans sentence raw-ent)]
                          (if (seq s1) s1
                              (let [s2 (find-char-spans sentence norm-ent)]
                                (if (seq s2) s2
                                    (find-char-spans sentence base-ent)))))]
         (if (empty? char-spans)
           {:entity raw-ent :matched? false :tokens [] :method :none}
           (let [[c-start c-end] (first char-spans)
                 ;; Find all tokens that overlap [c-start, c-end)
                 overlapping (filterv (fn [t]
                                        (and (< (:start t) c-end)
                                             (> (:end t) c-start)))
                                      token-info)
                 matched-tok-indices (mapv :idx overlapping)
                 matched-str (str/join "" (map :token-str overlapping))]
             {:entity raw-ent
              :matched? (seq matched-tok-indices)
              :char-span [c-start c-end]
              :token-indices matched-tok-indices
              :matched-text matched-str
              :method (cond
                        (seq (find-char-spans sentence raw-ent)) :exact
                        (seq (find-char-spans sentence norm-ent)) :normalized
                        :else :alias)}))))
     entities)))

(defn build-t-and-radj
  "Constructs T matrix [L, E] and R_adj matrix [E, E] from aligned entities and triples."
  [seq-len entities triples alignments]
  (let [l (long seq-len)
        ne (long (count entities))
        ent->idx (into {} (map-indexed (fn [i e] [e i]) entities))
        t-arr (float-array (* l ne) (float 0.0))
        radj-arr (float-array (* ne ne) (float 0.0))]

    ;; 1. Populate T matrix: T[pos, ent-idx] = 1.0
    (doseq [al alignments]
      (when (:matched? al)
        (let [ent-idx (get ent->idx (:entity al))]
          (when ent-idx
            (doseq [tok-idx (:token-indices al)]
              (when (< tok-idx l)
                (aset-float t-arr (+ (* tok-idx ne) ent-idx) (float 1.0))))))))

    ;; 2. Populate R_adj matrix: R_adj[h_idx, t_idx] = 1.0 (symmetric graph attention)
    (doseq [[h _rel t] triples]
      (let [h-idx (get ent->idx h)
            t-idx (get ent->idx t)]
        (when (and h-idx t-idx)
          (aset-float radj-arr (+ (* h-idx ne) t-idx) (float 1.0))
          (aset-float radj-arr (+ (* t-idx ne) h-idx) (float 1.0)))))

    {:T t-arr
     :R_adj radj-arr}))

(defn spot-check-alignment
  "Runs spot-check inspection across a sample of WebNLG entries and reports alignment stats."
  [entries tokenizer sample-size]
  (println "\n================================================================================")
  (println (format "🔍 T-MATRIX ENTITY ALIGNMENT INSPECTOR (Spot-Checking %d Examples)" sample-size))
  (println "================================================================================")

  (let [sampled (take (long sample-size) entries)
        total-entities (atom 0)
        matched-entities (atom 0)
        exact-matches (atom 0)
        norm-matches (atom 0)
        alias-matches (atom 0)
        display-limit 8
        displayed (atom 0)]

    (doseq [[i entry] (map-indexed vector sampled)]
      (let [triples (:triples entry)
            sentences (:sentences entry)
            first-sent (first sentences)
            entities (vec (distinct (concat (map first triples) (map #(nth % 2) triples))))
            alignments (align-entities-to-tokens tokenizer first-sent entities)]

        (swap! total-entities + (count entities))
        (doseq [al alignments]
          (when (:matched? al)
            (swap! matched-entities inc)
            (case (:method al)
              :exact (swap! exact-matches inc)
              :normalized (swap! norm-matches inc)
              :alias (swap! alias-matches inc))))

        (when (< @displayed display-limit)
          (swap! displayed inc)
          (println (format "\n--- Sample #%d [Category: %s | Triples: %d] ---"
                           (inc i) (:category entry) (count triples)))
          (println (format "Sentence: \"%s\"" first-sent))
          (doseq [al alignments]
            (if (:matched? al)
              (println (format "  ✅ Matched [%-10s]: \"%s\" -> tokens %s (\"%s\")"
                               (name (:method al)) (:entity al)
                               (:token-indices al) (:matched-text al)))
              (println (format "  ❌ Missed  : \"%s\" (Not found verbatim in sentence)"
                               (:entity al))))))))

    (let [tot (double @total-entities)
          matched (double @matched-entities)
          recall-pct (* 100.0 (/ matched (Math/max 1.0 tot)))]
      (println "\n================================================================================")
      (println "📊 T-MATRIX ALIGNMENT PRECISION & RECALL TELEMETRY")
      (println "================================================================================")
      (println (format "Total Entities Evaluated : %,d" (long tot)))
      (println (format "Successfully Aligned     : %,d (%.1f%% Recall)" (long matched) recall-pct))
      (println (format "  - Exact String Match   : %,d (%.1f%%)" @exact-matches (* 100.0 (/ (double @exact-matches) matched))))
      (println (format "  - Normalized Match     : %,d (%.1f%%)" @norm-matches (* 100.0 (/ (double @norm-matches) matched))))
      (println (format "  - Canonical Alias Match: %,d (%.1f%%)" @alias-matches (* 100.0 (/ (double @alias-matches) matched))))
      (println (format "Unmatched / Missing Spans: %,d (%.1f%%)" (- (long tot) (long matched)) (- 100.0 recall-pct)))
      (println "================================================================================\n")
      {:recall-pct recall-pct
       :total-entities (long tot)
       :matched-entities (long matched)})))

(defn -main [& args]
  (let [data-file (or (first args) ".dataset/webnlg/train.edn")
        model-dir (or (second args) ".models/gpt2")
        sample-size (Long/parseLong (or (nth args 2 nil) "150"))
        _ (println (format "Loading %s..." data-file))
        data (read-string (slurp data-file))
        tokenizer (tok/from-file model-dir)]
    (spot-check-alignment (:entries data) tokenizer sample-size)
    (System/exit 0)))
