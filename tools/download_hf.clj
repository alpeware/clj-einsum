(ns tools.download-hf
  "Download Hugging Face model weights using pure Clojure and JDK HttpClient."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.net URI]
           [java.net.http HttpClient HttpClient$Redirect HttpRequest HttpResponse$BodyHandlers]
           [java.nio.file Files Paths]))

(defn load-hf-token
  "Loads HF API token from .env map or HF_TOKEN environment variable."
  []
  (let [env-file (io/file ".env")]
    (if (.exists env-file)
      (let [data (edn/read-string (slurp env-file))]
        (or (:hf-token data) (:token data) (:HF_TOKEN data)))
      (System/getenv "HF_TOKEN"))))

(defn- http-client []
  (.. (HttpClient/newBuilder)
      (followRedirects HttpClient$Redirect/NORMAL)
      build))

(defn parse-manifest-rpaths
  "Extracts file rpaths or rfilenames from Hugging Face model API body."
  [body]
  (mapv second (re-seq #"\"r(?:filename|path)\":\"([^\"]+)\"" body)))

(defn filter-relevant-files
  "Filters repository files to target model weights, tokenizers, and configs, prioritizing metadata and tokenizer files."
  [files]
  (let [matched (filterv #(re-find #"(safetensors|json|txt|model)$" %) files)]
    (sort-by (fn [f] (if (str/includes? f "safetensors") 1 0)) matched)))

(defn- fetch-repo-files
  ([client repo token] (fetch-repo-files client repo token nil))
  ([client repo token revision]
   (let [url (str "https://huggingface.co/api/models/" repo (when revision (str "/revision/" revision)))
         req (cond-> (HttpRequest/newBuilder)
               true (.uri (URI/create url))
               token (.header "Authorization" (str "Bearer " token))
               true (.GET)
               true (.build))
         resp (.send client req (HttpResponse$BodyHandlers/ofString))
         body (.body resp)]
     (parse-manifest-rpaths body))))

(defn- download-file
  ([client repo filename target-dir token] (download-file client repo filename target-dir token nil))
  ([client repo filename target-dir token revision]
   (let [branch (or revision "main")
         url (str "https://huggingface.co/" repo "/resolve/" branch "/" filename)
         out-path (Paths/get target-dir (into-array String [(last (str/split filename #"/"))]))
         parent-file (.toFile (.getParent out-path))
         _ (when parent-file (.mkdirs parent-file))
         existing-size (if (Files/exists out-path (into-array java.nio.file.LinkOption []))
                         (Files/size out-path)
                         0)
         req (cond-> (HttpRequest/newBuilder)
               true (.uri (URI/create url))
               token (.header "Authorization" (str "Bearer " token))
               (pos? existing-size) (.header "Range" (str "bytes=" existing-size "-"))
               true (.GET)
               true (.build))]
     (println (format "  --> Downloading %s (resume from %d MB)..." filename (quot existing-size (* 1024 1024))))
     (let [resp (.send client req (HttpResponse$BodyHandlers/ofInputStream))
           in-stream (.body resp)
           status (.statusCode resp)]
       (if (or (= status 200) (= status 206))
         (let [open-options (if (= status 206)
                              [java.nio.file.StandardOpenOption/APPEND java.nio.file.StandardOpenOption/WRITE]
                              [java.nio.file.StandardOpenOption/CREATE java.nio.file.StandardOpenOption/TRUNCATE_EXISTING java.nio.file.StandardOpenOption/WRITE])]
           (with-open [out-stream (Files/newOutputStream out-path (into-array java.nio.file.OpenOption open-options))]
             (io/copy in-stream out-stream))
           (println (format "      Done: %s (%d MB)"
                            (.getFileName out-path)
                            (quot (Files/size out-path) (* 1024 1024)))))
         (println (format "      Error downloading %s: HTTP %d" filename status)))))))

(defn parse-cli-args
  "Parses command-line arguments into {:repo ... :revision ... :target-dir ...}."
  [args]
  (loop [remaining args
         res {:repo "google/gemma-4-E2B-it" :revision nil :target-dir nil :explicit-repo false}]
    (if (empty? remaining)
      (let [repo (:repo res)
            [clean-repo repo-rev] (if (str/includes? repo "@") (str/split repo #"@" 2) [repo nil])
            final-rev (or (:revision res) repo-rev)
            model-name (last (str/split clean-repo #"/"))
            target-dir (or (:target-dir res) (str ".models/" model-name))]
        {:repo clean-repo
         :revision final-rev
         :target-dir target-dir})
      (let [arg (first remaining)]
        (cond
          (or (= arg "--revision") (= arg "--branch") (= arg "-b"))
          (recur (drop 2 remaining) (assoc res :revision (second remaining)))

          (or (= arg "--target-dir") (= arg "--dir") (= arg "-o"))
          (recur (drop 2 remaining) (assoc res :target-dir (second remaining)))

          (not (:explicit-repo res))
          (recur (rest remaining) (assoc res :repo arg :explicit-repo true))

          :else
          (recur (rest remaining) res))))))

(defn -main [& args]
  (let [{:keys [repo revision target-dir]} (parse-cli-args args)
        token (load-hf-token)
        client (http-client)
        rev-desc (if revision (str " (revision: " revision ")") "")
        _ (println (format "Fetching file manifest for [%s]%s..." repo rev-desc))
        files (fetch-repo-files client repo token revision)
        relevant-files (filter-relevant-files files)]
    (if (empty? relevant-files)
      (println (format "No files found for [%s]%s. If restricted, ensure HF_TOKEN is set." repo rev-desc))
      (do
        (println (format "Found %d model files. Downloading to [%s]..." (count relevant-files) target-dir))
        (doseq [f relevant-files]
          (download-file client repo f target-dir token revision))
        (println "\n=== Download Complete! ===")))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
