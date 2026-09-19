(ns einsum.test-helpers.telemetry
  "System telemetry collection, hardware capability inspection, and structured EDN report generation."
  (:require [einsum.compiler.pjrt.version :as v]
            [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]))

(defn collect-system-telemetry
  "Collects host platform, JVM runtime, driver diagnostic info, and detected backend capabilities."
  []
  (let [runtime (Runtime/getRuntime)
        driver-info (v/probe-system-driver)]
    {:timestamp (str (java.time.Instant/now))
     :host {:os-name (System/getProperty "os.name")
            :os-arch (System/getProperty "os.arch")
            :os-version (System/getProperty "os.version")
            :java-version (System/getProperty "java.version")
            :max-memory-bytes (.maxMemory runtime)
            :total-memory-bytes (.totalMemory runtime)
            :free-memory-bytes (.freeMemory runtime)}
     :backends driver-info}))

(defn generate-edn-report
  "Generates a complete hardware test matrix EDN report combining telemetry and test suite `results`.
   Writes output to `out-file` (default: `target/test-reports/hardware_matrix.edn`)."
  ([results] (generate-edn-report results (io/file "target/test-reports/hardware_matrix.edn")))
  ([results out-file]
   (let [telemetry (collect-system-telemetry)
         report {:telemetry telemetry
                 :test-suite results}
         out-file (io/file out-file)]
     (.mkdirs (.getParentFile out-file))
     (spit out-file (with-out-str (pprint report)))
     report)))
