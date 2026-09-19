(ns einsum.compiler.compile
  (:require [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.java.io :as io])
  (:import [java.io File]
           [java.nio.file CopyOption Files OpenOption StandardCopyOption StandardOpenOption]
           [java.security MessageDigest]))

(defonce ^:private exec-cache (atom {}))
(defonce ^:private custom-cache-dir (atom nil))

(defn- sha256-hash [^String s]
  (let [digest (MessageDigest/getInstance "SHA-256")
        bytes (.digest digest (.getBytes s "UTF-8"))]
    (apply str (map #(format "%02x" %) bytes))))

(defn get-cache-dir
  "Returns the root File directory for PJRT compiled executable binary caching."
  []
  (or @custom-cache-dir
      (if-let [env-dir (or (System/getenv "CLJ_XLA_CACHE_DIR") (System/getProperty "clj-xla.cache-dir"))]
        (io/file env-dir)
        (io/file (System/getProperty "user.home") ".cache" "clj_xla" "executables"))))

(defn set-cache-dir!
  "Overrides the cache directory for compiled executables."
  [dir]
  (reset! custom-cache-dir (if (string? dir) (io/file dir) dir)))

(defn cache-disabled?
  "Returns true if disk caching is explicitly disabled."
  []
  (let [v (or (System/getenv "CLJ_XLA_DISABLE_CACHE") (System/getProperty "clj-xla.disable-cache"))]
    (boolean (and v (re-matches #"(?i)true|1|yes" v)))))

(defn- write-atomic! [^File target-file ^bytes data]
  (let [parent (.getParentFile target-file)]
    (when parent (.mkdirs parent))
    (let [temp-file (File/createTempFile "pjrt_exec_" ".tmp" parent)]
      (try
        (Files/write (.toPath temp-file) data (into-array OpenOption [StandardOpenOption/WRITE]))
        (Files/move (.toPath temp-file)
                    (.toPath target-file)
                    (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING StandardCopyOption/ATOMIC_MOVE]))
        (catch Exception _
          (try
            (Files/move (.toPath temp-file)
                        (.toPath target-file)
                        (into-array CopyOption [StandardCopyOption/REPLACE_EXISTING]))
            (catch Exception _ nil)))
        (finally
          (when (.exists temp-file)
            (.delete temp-file)))))))

(defn compile-graph
  "Serializes `graph` to StableHLO MLIR text, checks in-memory and on-disk binary cache,
   compiles if necessary, and returns the compiled PjRtLoadedExecutable handle map."
  [api-ctx client graph]
  (let [mlir-text (shlo/graph->mlir-text graph)
        mlir-hash (sha256-hash mlir-text)
        cli (or client (:client api-ctx))
        hash-key [cli mlir-hash]]
    ;; 1. Check in-memory cache
    (if-let [cached-exec (get @exec-cache hash-key)]
      (assoc cached-exec :f (:f graph) :graph graph)
      ;; 2. Check persistent disk cache
      (let [platform (try (pjrt/platform-name api-ctx cli) (catch Exception _ "unknown"))
            cache-d (get-cache-dir)
            cache-file (io/file cache-d (format "%s_%s.bin" platform mlir-hash))
            disk-exec (when (and (not (cache-disabled?)) (.exists cache-file) (.isFile cache-file))
                        (try
                          (let [t0 (System/nanoTime)
                                data (Files/readAllBytes (.toPath cache-file))
                                loaded (pjrt/deserialize-and-load api-ctx cli data)
                                dur-ms (/ (- (System/nanoTime) t0) 1e6)]
                            (when-not (Boolean/getBoolean "clj-xla.quiet")
                              (println (format "  ↳ Loaded cached PJRT executable [%s] (%.2f KB in %.2f ms)"
                                               (.getName cache-file) (/ (alength data) 1024.0) dur-ms)))
                            {:handle loaded :hash hash-key :f (:f graph) :graph graph :from-disk-cache? true})
                          (catch Exception _
                            (try (.delete cache-file) (catch Exception _ nil))
                            nil)))]
        (if disk-exec
          (do
            (swap! exec-cache assoc hash-key disk-exec)
            disk-exec)
          ;; 3. Fresh compilation & serialize to disk cache
          (let [exec (pjrt/compile-mlir api-ctx cli mlir-text)
                exec-obj {:handle exec :hash hash-key :f (:f graph) :graph graph :from-disk-cache? false}]
            (when (not (cache-disabled?))
              (try
                (when-let [ser-bytes (pjrt/serialize-executable api-ctx exec)]
                  (write-atomic! cache-file ser-bytes)
                  (when-not (Boolean/getBoolean "clj-xla.quiet")
                    (println (format "  ↳ Cached compiled PJRT executable to [%s] (%.2f KB)"
                                     (.getName cache-file) (/ (alength ser-bytes) 1024.0)))))
                (catch Exception _ nil)))
            (swap! exec-cache assoc hash-key exec-obj)
            exec-obj))))))

(defn clear-disk-cache!
  "Deletes all cached executable binary files from the disk cache directory."
  []
  (let [dir (get-cache-dir)]
    (when (.exists dir)
      (doseq [f (.listFiles dir)]
        (when (.isFile f)
          (.delete f))))))

(defn clear-cache!
  "Clears the in-memory compilation cache."
  []
  (reset! exec-cache {}))

