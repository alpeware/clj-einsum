(ns einsum.runtime.benchmark.runner
  "Execution engine for measuring JIT compilation throughput, execution latency, FLOPS, and memory bandwidth across PJRT backends."
  (:require [einsum.runtime.benchmark.core :as bcore]
            [einsum.runtime.benchmark.workloads :as bw]
            [einsum.core :as xla]
            [einsum.compiler.pjrt :as pjrt]))

(defn benchmark-workload
  "Measures cold compilation, warm compilation, latency percentiles, FLOPS, and bandwidth for `workload` on `ctx`."
  ([ctx workload] (benchmark-workload ctx workload {}))
  ([ctx workload opts]
   (binding [xla/*default-context* ctx]
     (let [{:keys [id name category flops bytes build-graph-fn make-inputs-fn]} workload
           warmup-iters (get opts :warmup-iters 5)
           measure-iters (get opts :measure-iters 25)
           target (or (:target ctx) :cpu)

           ;; 1. Measure Cold JIT compilation
           t0 (System/nanoTime)
           graph (build-graph-fn)
           exec1 (xla/compile-graph ctx graph)
           t1 (System/nanoTime)
           cold-ms (/ (- t1 t0) 1000000.0)

           ;; 2. Measure Warm cached compilation
           t2 (System/nanoTime)
           _exec2 (xla/compile-graph ctx graph)
           t3 (System/nanoTime)
           warm-us (/ (- t3 t2) 1000.0)

           ;; 3. Transfer input data to resident device memory buffers
           inputs (make-inputs-fn)
           invars (:invars graph)
           raw-args (mapv (fn [idx [_var-name _]]
                            (get inputs _var-name (nth (vals inputs) idx)))
                          (range (count invars))
                          invars)
           device-args (mapv (fn [idx input-data]
                               (let [[_var-name [_kw shape dtype]] (nth invars idx)
                                     dtype-enum (case dtype :i8 2 :i32 4 :f32 11 :bf16 13 :f16 10 11)]
                                 (pjrt/buffer-from-host-buffer ctx (:client ctx) input-data shape dtype-enum)))
                             (range (count invars))
                             raw-args)

           ;; 4. Warmup passes
           _ (dotimes [_ warmup-iters]
               (let [out (xla/execute exec1 device-args)]
                 (if (vector? out)
                   (doseq [b out] (pjrt/destroy-buffer! ctx b))
                   (pjrt/destroy-buffer! ctx out))))

           ;; 5. Measured passes
           latencies (mapv (fn [_]
                             (let [start (System/nanoTime)
                                   out (xla/execute exec1 device-args)
                                   end (System/nanoTime)
                                   lat (/ (- end start) 1000000.0)]
                               (if (vector? out)
                                 (doseq [b out] (pjrt/destroy-buffer! ctx b))
                                 (pjrt/destroy-buffer! ctx out))
                               lat))
                           (range measure-iters))

           _ (doseq [b device-args] (pjrt/destroy-buffer! ctx b))

           stats (bcore/calculate-latency-stats latencies)
           mean-lat (:mean stats)
           total-flops (if (fn? flops) (flops) (or flops 0))
           total-bytes (if (fn? bytes) (bytes) (or bytes 0))
           gflops (bcore/calculate-gflops total-flops mean-lat)
           tflops (bcore/calculate-tflops total-flops mean-lat)
           gbps (bcore/calculate-gbps total-bytes mean-lat)]
       (merge {:kernel id
               :name name
               :category category
               :backend target
               :cold-ms cold-ms
               :warm-us warm-us
               :mean-ms mean-lat
               :p50-ms (:p50 stats)
               :p90-ms (:p90 stats)
               :p99-ms (:p99 stats)
               :flops total-flops
               :bytes total-bytes
               :gflops gflops
               :tflops tflops
               :gbps gbps}
              stats)))))

(defn run-backend-benchmarks
  "Initializes `target-backend` and benchmarks all registered workloads."
  ([target-backend] (run-backend-benchmarks target-backend {}))
  ([target-backend opts]
   (let [ctx (xla/init-backend! target-backend)]
     (mapv (fn [[_id wl]]
             (benchmark-workload ctx wl opts))
           bw/WORKLOADS))))
