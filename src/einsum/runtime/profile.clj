(ns einsum.runtime.profile
  "High-precision telemetry profiling spans, latency percentiles, and Chrome trace export."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]))

(def ^:dynamic *active-trace-spans* nil)

(defn now-microseconds
  "Returns current system time in microseconds."
  []
  (quot (System/nanoTime) 1000))

(defn compute-latency-summary
  "Calculates summary statistics (min, max, mean, p50, p95, p99) from a sequence of latency numbers."
  [latencies]
  (if (empty? latencies)
    {:count 0 :min 0.0 :max 0.0 :mean 0.0 :p50 0.0 :p95 0.0 :p99 0.0}
    (let [sorted (vec (sort latencies))
          n (count sorted)
          sum (reduce + sorted)
          mean (/ (double sum) n)
          percentile (fn [p] (nth sorted (min (dec n) (int (Math/floor (* (/ p 100.0) n))))))]
      {:count n
       :min (double (first sorted))
       :max (double (last sorted))
       :mean (double mean)
       :p50 (double (percentile 50))
       :p95 (double (percentile 95))
       :p99 (double (percentile 99))})))

(defmacro with-profile
  "Executes `body`, measures duration in milliseconds, updates `metrics-atom` map under `span-name`,
   and records a trace span event if `*active-trace-spans*` is bound."
  [metrics-atom span-name & body]
  `(let [start-ns# (System/nanoTime)
         start-us# (now-microseconds)
         res# (do ~@body)
         dur-ns# (- (System/nanoTime) start-ns#)
         dur-ms# (/ (double dur-ns#) 1000000.0)
         dur-us# (quot dur-ns# 1000)]
     (when (instance? clojure.lang.IAtom ~metrics-atom)
       (swap! ~metrics-atom assoc ~span-name {:duration-ms dur-ms#
                                              :duration-us dur-us#}))
     (when *active-trace-spans*
       (swap! *active-trace-spans* conj {"name" ~span-name
                                         "cat" "clj-xla"
                                         "ph" "X"
                                         "ts" start-us#
                                         "dur" dur-us#
                                         "pid" 1
                                         "tid" 1}))
     res#))

(defn export-chrome-trace
  "Exports sequence of span maps into Chrome Tracing JSON format string viewable in `chrome://tracing` or `ui.perfetto.dev`."
  [spans]
  (json/write-str spans))

(defn save-chrome-trace!
  "Writes Chrome Tracing JSON event list to `out-file`."
  [spans out-file]
  (let [f (io/file out-file)]
    (.mkdirs (.getParentFile f))
    (spit f (export-chrome-trace spans))
    out-file))
