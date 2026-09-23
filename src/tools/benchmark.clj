(ns tools.benchmark
  "State-of-the-Art (SOTA) Multi-Backend Kernel & Transformer Block Benchmark Suite."
  (:require [einsum.runtime.benchmark.core :as bcore]
            [einsum.runtime.benchmark.runner :as runner]
            [einsum.compiler.pjrt.version :as v]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(defn parse-cli-opts
  "Parses command line arguments into benchmark configuration map."
  [args]
  (loop [remaining args
         opts {:backend :auto
               :warmup-iters 5
               :measure-iters 25}]
    (if (empty? remaining)
      opts
      (let [arg (first remaining)]
        (cond
          (or (= arg "--backend") (= arg "-b"))
          (recur (drop 2 remaining) (assoc opts :backend (keyword (second remaining))))

          (str/starts-with? arg "--backend=")
          (recur (rest remaining) (assoc opts :backend (keyword (subs arg (count "--backend=")))))

          (or (= arg "--warmup") (= arg "-w"))
          (recur (drop 2 remaining) (assoc opts :warmup-iters (Long/parseLong (second remaining))))

          (or (= arg "--iters") (= arg "-i") (= arg "--measure"))
          (recur (drop 2 remaining) (assoc opts :measure-iters (Long/parseLong (second remaining))))

          :else
          (recur (rest remaining) opts))))))

(defn save-benchmark-report!
  "Writes structured EDN report to target/benchmark-reports/benchmark_matrix.edn."
  [report-data]
  (let [dir (io/file "target/benchmark-reports")
        file (io/file dir "benchmark_matrix.edn")]
    (.mkdirs dir)
    (spit file (pr-str report-data))
    (println (format "\nBenchmark matrix EDN report saved to: %s" (.getAbsolutePath file)))))

(defn -main
  "Main benchmark suite entry point."
  [& args]
  (let [opts (parse-cli-opts args)
        probe (v/probe-system-driver)
        detected (:detected-backends probe)
        target-backend (:backend opts)
        backends-to-run (cond
                          (= target-backend :auto) detected
                          (contains? detected target-backend) #{target-backend}
                          :else (do
                                  (println (format "Warning: Target backend %s not detected on host. Falling back to CPU." target-backend))
                                  #{:cpu}))]
    (println "==========================================================================")
    (println "                clj-xla SOTA Kernel Benchmark Suite                       ")
    (println "==========================================================================")
    (println (format "Host System: %s (%s, %s)"
                     (get-in probe [:details :cpu :java-version] "JVM")
                     (System/getProperty "os.name")
                     (System/getProperty "os.arch")))
    (println (format "Detected Backends: %s" (str/join ", " (map name detected))))
    (println (format "Benchmarking Backends: %s" (str/join ", " (map name backends-to-run))))
    (println "--------------------------------------------------------------------------\n")

    (let [all-results (atom [])]
      (doseq [b (sort (vec backends-to-run))]
        (println (format ">>> Running benchmarks for backend: [%s]..." (name b)))
        (try
          (let [b-results (runner/run-backend-benchmarks b opts)]
            (swap! all-results concat b-results))
          (catch Exception e
            (println (format "Error running benchmark for [%s]: %s" (name b) (.getMessage e))))))

      (let [final-results @all-results
            report-map {:telemetry probe
                        :opts opts
                        :timestamp (str (java.time.Instant/now))
                        :results final-results}]
        (save-benchmark-report! report-map)
        (println "\n==========================================================================")
        (println "                    Benchmark Comparative Results Matrix                    ")
        (println "==========================================================================")
        (println (bcore/render-markdown-table final-results))
        (println "==========================================================================\n")))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
