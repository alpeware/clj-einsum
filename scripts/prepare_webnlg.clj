(ns scripts.prepare-webnlg
  "Parser and preprocessor for the WebNLG (v3.0 English) dataset.
   Reads raw XML files from release_v3.0/en, normalizes entities, predicates, and sentences,
   and writes parsed .dataset/webnlg/train.edn and .dataset/webnlg/dev.edn."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.xml :as xml])
  (:gen-class))

(defn clean-text [s]
  (let [trimmed (str/trim (or s ""))
        unquoted (if (and (str/starts-with? trimmed "\"") (str/ends-with? trimmed "\"") (> (count trimmed) 1))
                   (subs trimmed 1 (dec (count trimmed)))
                   trimmed)]
    (-> unquoted
        (str/replace #"_" " ")
        (str/replace #"\s+" " ")
        str/trim)))

(defn parse-file [f]
  (try
    (let [parsed (xml/parse f)]
      (vec
       (for [node (:content parsed) :when (= :entries (:tag node))
             entry (:content node) :when (= :entry (:tag entry))
             :let [content (:content entry)
                   eid (get-in entry [:attrs :eid])
                   cat (get-in entry [:attrs :category])
                   mtriples (vec
                             (for [n content :when (= :modifiedtripleset (:tag n))
                                   trip (:content n) :when (= :mtriple (:tag trip))
                                   :let [text (first (:content trip))
                                         parts (mapv str/trim (str/split (or text "") #"\|"))]
                                   :when (= 3 (count parts))]
                               [(clean-text (nth parts 0))
                                (keyword (str/replace (clean-text (nth parts 1)) #"\s+" "_"))
                                (clean-text (nth parts 2))]))
                   lexes (vec
                          (distinct
                           (for [n content :when (= :lex (:tag n))
                                 :let [t (clean-text (first (:content n)))]
                                 :when (pos? (count t))]
                             t)))]
             :when (and (seq mtriples) (seq lexes))]
         {:eid eid
          :category cat
          :triples mtriples
          :sentences lexes})))
    (catch Exception e
      (println "Warning: failed to parse" (.getName ^java.io.File f) (.getMessage e))
      [])))

(defn parse-directory [dir-path max-triples]
  (let [dir (io/file dir-path)
        all-files (filter #(and (.isFile ^java.io.File %)
                                (.endsWith (.getName ^java.io.File %) ".xml"))
                          (file-seq dir))
        files-to-parse (if max-triples
                         (filter (fn [^java.io.File f]
                                   (let [p (.getAbsolutePath f)]
                                     (some #(str/includes? p (str % "triples")) (range 1 (inc max-triples)))))
                                 all-files)
                         all-files)]
    (vec (mapcat parse-file files-to-parse))))

(defn extract-corpus-summary [entries]
  (let [triples (distinct (mapcat :triples entries))
        sentences (distinct (mapcat :sentences entries))
        subjects (distinct (map first triples))
        predicates (distinct (map second triples))
        objects (distinct (map #(nth % 2) triples))
        entities (distinct (concat subjects objects))]
    {:total-entries (count entries)
     :total-sentences (count sentences)
     :unique-triples (count triples)
     :unique-relations (count predicates)
     :unique-entities (count entities)}))

(defn -main [& args]
  (let [repo-dir (or (first args) "/tmp/webnlg_repo/release_v3.0/en")
        out-dir (io/file ".dataset/webnlg")
        _ (.mkdirs out-dir)
        train-path (str repo-dir "/train")
        dev-path (str repo-dir "/dev")

        _ (println "Parsing WebNLG 1-3 triples subset for optimal training density...")
        ;; Use 1-3 triples for focused relational grounding density
        train-entries (parse-directory train-path 3)
        dev-entries (parse-directory dev-path 3)

        train-summary (extract-corpus-summary train-entries)
        dev-summary (extract-corpus-summary dev-entries)

        train-out (io/file out-dir "train.edn")
        dev-out (io/file out-dir "dev.edn")]

    (println "\n=== WebNLG Dataset Processing Complete ===")
    (println (format "Train: %,d entries | %,d sentences | %,d triples | %,d entities | %,d relations"
                     (:total-entries train-summary) (:total-sentences train-summary)
                     (:unique-triples train-summary) (:unique-entities train-summary)
                     (:unique-relations train-summary)))
    (println (format "Dev:   %,d entries | %,d sentences | %,d triples | %,d entities | %,d relations"
                     (:total-entries dev-summary) (:total-sentences dev-summary)
                     (:unique-triples dev-summary) (:unique-entities dev-summary)
                     (:unique-relations dev-summary)))

    (spit train-out (pr-str {:summary train-summary :entries train-entries}))
    (spit dev-out (pr-str {:summary dev-summary :entries dev-entries}))
    (println (format "Saved train to %s (%.2f MB)" (.getPath train-out) (/ (.length train-out) 1048576.0)))
    (println (format "Saved dev to %s (%.2f MB)" (.getPath dev-out) (/ (.length dev-out) 1048576.0)))))
