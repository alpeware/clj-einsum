(ns einsum.runtime.benchmark.core
  "Pure Clojure telemetry and metrics calculation engine for benchmarking PJRT workloads."
  (:require [clojure.string :as str]))

(defn calculate-latency-stats
  "Calculates latency summary statistics (mean, p50, p90, p99, min, max) for given latency measurements."
  [latencies]
  (let [sorted (sort latencies)
        len (count sorted)]
    (if (zero? len)
      {:mean 0.0 :p50 0.0 :p90 0.0 :p99 0.0 :min-lat 0.0 :max-lat 0.0}
      (let [sum (reduce + sorted)
            mean (double (/ sum len))
            percentile (fn [p]
                         (let [idx (min (dec len) (max 0 (int (Math/floor (* (/ p 100.0) len)))))]
                           (double (nth sorted idx))))
            p50 (percentile 50)
            p90 (percentile 90)
            p99 (percentile 99)
            min-lat (double (first sorted))
            max-lat (double (last sorted))]
        {:mean mean
         :p50 p50
         :p90 p90
         :p99 p99
         :min-lat min-lat
         :max-lat max-lat}))))

(defn gemm-flops
  "Calculates total floating point operations for matrix multiplication M x K * K x N."
  [m n k]
  (* 2 (long m) (long n) (long k)))

(defn calculate-gflops
  "Calculates GFLOPS (Billion FLOPs / sec) given total FLOPs and latency in milliseconds."
  [flops lat-ms]
  (if (or (nil? lat-ms) (zero? lat-ms))
    0.0
    (double (/ flops (* lat-ms 1000000.0)))))

(defn calculate-tflops
  "Calculates TFLOPS (Trillion FLOPs / sec) given total FLOPs and latency in milliseconds."
  [flops lat-ms]
  (if (or (nil? lat-ms) (zero? lat-ms))
    0.0
    (double (/ flops (* lat-ms 1000000000.0)))))

(defn calculate-gbps
  "Calculates effective memory bandwidth in GB/s given transferred bytes and latency in milliseconds."
  [bytes lat-ms]
  (if (or (nil? lat-ms) (zero? lat-ms))
    0.0
    (double (/ bytes (* lat-ms 1000000.0)))))

(defn render-markdown-table
  "Renders a formatted Markdown table comparing benchmark results across backends and workloads."
  [results]
  (let [headers ["Kernel / Workload" "Backend" "Cold JIT (ms)" "Mean (ms)" "p50 (ms)" "p99 (ms)" "TFLOPS" "Bandwidth (GB/s)"]
        header-line (str "| " (str/join " | " headers) " |")
        divider-line (str "| " (str/join " | " (repeat (count headers) "---")) " |")
        rows (mapv (fn [r]
                     (let [kname (name (or (:kernel r) (:name r) "unknown"))
                           bname (name (or (:backend r) :unknown))
                           cold (if-let [c (:cold-ms r)] (format "%.2f" (double c)) "N/A")
                           mean (if-let [m (or (:mean-ms r) (:mean r))] (format "%.3f" (double m)) "N/A")
                           p50 (if-let [p (:p50-ms r)] (format "%.3f" (double p)) "N/A")
                           p99 (if-let [p (:p99-ms r)] (format "%.3f" (double p)) "N/A")
                           tflops (if-let [tf (:tflops r)] (format "%.3f" (double tf)) "N/A")
                           gbps (if-let [bw (:gbps r)] (format "%.2f" (double bw)) "N/A")]
                       (str "| " (str/join " | " [kname bname cold mean p50 p99 tflops gbps]) " |")))
                   results)]
    (str/join "\n" (concat [header-line divider-line] rows))))
