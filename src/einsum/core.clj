(ns einsum.core
  "High-level Public Clojure API for OpenXLA PJRT backend initialization, compilation, and execution."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.compiler.compile :as compile]
            [einsum.compiler.fuse :as fuse]
            [einsum.compiler.kernel :as kernel]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.pjrt.version :as v]
            [einsum.logic.interpret :as interpret]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.weights :as weights])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(def ^:dynamic *default-context* nil)

(def BACKEND-LIBRARY-MAP
  {:cpu    {:default "bin/libpjrt_cpu.so"  :env "PJRT_CPU_LIBRARY_PATH"}
   :sycl   {:default "bin/libpjrt_sycl.so" :env "PJRT_SYCL_LIBRARY_PATH"}
   :rocm   {:default "bin/libpjrt_rocm.so" :env "PJRT_ROCM_LIBRARY_PATH"}
   :cuda12 {:default "bin/libpjrt_cuda.so" :env "PJRT_CUDA_LIBRARY_PATH"}})

(defn- setenv-native [^String k ^String v]
  (try
    (let [linker (java.lang.foreign.Linker/nativeLinker)
          default-lookup (.defaultLookup linker)
          setenv-opt (.find default-lookup "setenv")]
      (when (.isPresent setenv-opt)
        (let [setenv-ptr ^java.lang.foreign.MemorySegment (.get setenv-opt)
              fd (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/JAVA_INT
                                                          (into-array java.lang.foreign.MemoryLayout
                                                                      [java.lang.foreign.ValueLayout/ADDRESS
                                                                       java.lang.foreign.ValueLayout/ADDRESS
                                                                       java.lang.foreign.ValueLayout/JAVA_INT]))
              handle (.downcallHandle linker setenv-ptr fd (make-array java.lang.foreign.Linker$Option 0))]
          (with-open [arena (java.lang.foreign.Arena/ofConfined)]
            (let [k-seg (.allocateFrom arena k)
                  v-seg (.allocateFrom arena v)]
              (.invokeWithArguments handle [k-seg v-seg (int 1)]))))))
    (catch Exception _ nil)))

(defn- getenv-native [^String k]
  (try
    (let [linker (java.lang.foreign.Linker/nativeLinker)
          default-lookup (.defaultLookup linker)
          getenv-opt (.find default-lookup "getenv")]
      (when (.isPresent getenv-opt)
        (let [getenv-ptr ^java.lang.foreign.MemorySegment (.get getenv-opt)
              fd (java.lang.foreign.FunctionDescriptor/of java.lang.foreign.ValueLayout/ADDRESS
                                                          (into-array java.lang.foreign.MemoryLayout
                                                                      [java.lang.foreign.ValueLayout/ADDRESS]))
              handle (.downcallHandle linker getenv-ptr fd (make-array java.lang.foreign.Linker$Option 0))]
          (with-open [arena (java.lang.foreign.Arena/ofConfined)]
            (let [k-seg (.allocateFrom arena k)
                  res-ptr ^java.lang.foreign.MemorySegment (.invokeWithArguments handle [k-seg])]
              (when (and res-ptr (not (.equals res-ptr java.lang.foreign.MemorySegment/NULL)))
                (.getString (.reinterpret res-ptr Long/MAX_VALUE) 0)))))))
    (catch Exception _ nil)))

(defn determine-optimal-xla-flags
  "Pure function computing optimal XLA compiler flags and environment variables based on hardware target and user options."
  ([target] (determine-optimal-xla-flags target {} {}))
  ([target probe-info] (determine-optimal-xla-flags target probe-info {}))
  ([target _probe-info opts]
   (let [disable-defaults? (boolean (:disable-defaults? opts))
         user-xla-flags (or (:xla-flags opts) "")
         autotune-level (or (:autotune-level opts) 4)
         triton? (get opts :triton? true)
         sdpa? (get opts :sdpa? true)
         latency-hiding? (get opts :latency-hiding? true)
         async-stream? (get opts :async-stream? true)
         cpus (.availableProcessors (Runtime/getRuntime))

         target-kw (if (string? target)
                     (cond
                       (str/includes? target "cpu") :cpu
                       (str/includes? target "rocm") :rocm
                       (str/includes? target "cuda") :cuda12
                       (str/includes? target "sycl") :sycl
                       :else :cpu)
                     target)]
     (if disable-defaults?
       {:xla-flags user-xla-flags
        :env-vars {}
        :autotune-level autotune-level
        :cache-dir (:cache-dir opts)}
       (let [existing-xla-flags (or (System/getenv "XLA_FLAGS") "")
             xla-cache-dir (try
                             (let [dir (io/file (System/getProperty "user.home") ".cache" "xla")]
                               (.mkdirs dir)
                               (.getAbsolutePath dir))
                             (catch Exception _ nil))
             base-flags (cond-> []
                          (and (some? xla-cache-dir) (contains? #{:rocm :cuda12 :sycl} target-kw))
                          (into [(str "--xla_gpu_per_fusion_autotune_cache_dir=" xla-cache-dir)
                                 (str "--xla_gpu_experimental_autotuner_cache_dir=" xla-cache-dir)
                                 (str "--xla_gpu_kernel_cache_file=" xla-cache-dir "/kernel.cache")])

                          (= target-kw :cuda12)
                          (conj "--xla_gpu_enable_cublaslt=true")

                          (contains? #{:rocm :cuda12 :sycl} target-kw)
                          (conj (str "--xla_gpu_autotune_level=" autotune-level))

                          (and triton? (= target-kw :cuda12))
                          (conj "--xla_gpu_triton_gemm_any=true")

                          (and sdpa? (= target-kw :cuda12))
                          (conj "--xla_gpu_enable_sdpa=true")

                          (and latency-hiding? (contains? #{:rocm :cuda12} target-kw))
                          (conj "--xla_gpu_enable_latency_hiding_scheduler=true")

                          (and async-stream? (contains? #{:cuda12 :sycl} target-kw))
                          (conj "--xla_gpu_enable_highest_priority_async_stream=true")

                          (= target-kw :rocm)
                          (conj "--xla_gpu_enable_command_buffer=")

                          (contains? #{:rocm :cuda12 :sycl} target-kw)
                          (conj (str "--xla_gpu_force_compilation_parallelism=" cpus))

                          (= target-kw :cpu)
                          (into ["--xla_cpu_multi_thread_eigen=true"
                                 (str "--xla_gpu_force_compilation_parallelism=" cpus)]))

             all-flags-str (str/join " " (distinct (remove str/blank? (concat (str/split existing-xla-flags #"\s+")
                                                                              base-flags
                                                                              (str/split user-xla-flags #"\s+")))))
             cache-dir (or (:cache-dir opts)
                           (when (= target-kw :rocm)
                             (try
                               (let [dir (io/file (System/getProperty "user.home") ".cache" "hsa_cache")]
                                 (.mkdirs dir)
                                 (.getAbsolutePath dir))
                               (catch Exception _ nil))))

             env-vars (cond-> {"TF_CPP_MIN_LOG_LEVEL" "3"
                               "GLOG_minloglevel" "3"}
                        (= target-kw :rocm)
                        (assoc "HSA_OVERRIDE_GFX_VERSION" (or (:hsa-override-gfx-version opts)
                                                              (:gfx-version opts)
                                                              (getenv-native "HSA_OVERRIDE_GFX_VERSION")
                                                              (System/getenv "HSA_OVERRIDE_GFX_VERSION")
                                                              "11.0.0")
                               "ROCR_VISIBLE_DEVICES" (or (:rocr-visible-devices opts)
                                                          (:gpu-device opts)
                                                          (getenv-native "ROCR_VISIBLE_DEVICES")
                                                          (System/getenv "ROCR_VISIBLE_DEVICES")
                                                          "0")
                               "HIP_VISIBLE_DEVICES" (or (:hip-visible-devices opts)
                                                         (:gpu-device opts)
                                                         (getenv-native "HIP_VISIBLE_DEVICES")
                                                         (System/getenv "HIP_VISIBLE_DEVICES")
                                                         "0"))
                        (some? cache-dir)
                        (assoc "TF_XLA_HSACO_CACHE_DIR" cache-dir)
                        (some? xla-cache-dir)
                        (assoc "XLA_PERSISTENT_COMPILATION_CACHE_DIR" xla-cache-dir))]
         (doseq [[k v] env-vars]
           (System/setProperty k v)
           (setenv-native k v))
         (when-not (str/blank? all-flags-str)
           (System/setProperty "XLA_FLAGS" all-flags-str)
           (setenv-native "XLA_FLAGS" all-flags-str))
         {:xla-flags all-flags-str
          :env-vars env-vars
          :autotune-level autotune-level
          :cache-dir cache-dir})))))

(defn init-backend!
  "Initializes PJRT C API client runtime for specified target (:cpu, :sycl, :rocm, :cuda12, :interpreter, :interp, or a custom string path).
   Accepts optional `client-opts` map (defaults to `{:allocator \"platform\"}`).
   Sets and returns the thread-root default context *default-context*."
  ([] (init-backend! :cpu))
  ([target] (init-backend! target {:allocator "platform"}))
  ([target client-opts]
   (cond
     (contains? #{:interpreter :interp} target)
     (let [ctx {:backend :interpreter
                :target :interpreter
                :platform "interpreter"
                :client nil
                :destroy-fn (fn [_ _] nil)}]
       (alter-var-root #'*default-context* (constantly ctx))
       (when-not (or (Boolean/getBoolean "clj-xla.quiet") (:quiet? client-opts))
         (println "clj-xla initialized Pure-JVM Interpreter Backend: [interpreter] (no native PJRT required)"))
       ctx)

     :else
     (let [probe-info (try (v/probe-system-driver) (catch Exception _ {}))
           flag-config (determine-optimal-xla-flags target probe-info client-opts)
           {:keys [xla-flags env-vars]} flag-config]
       (doseq [[k v] env-vars]
         (System/setProperty k v)
         (setenv-native k v))
       (when-not (str/blank? xla-flags)
         (System/setProperty "XLA_FLAGS" xla-flags)
         (setenv-native "XLA_FLAGS" xla-flags))

       (let [lib-path (cond
                        (string? target) target
                        (keyword? target) (let [{:keys [default env]} (get BACKEND-LIBRARY-MAP target)]
                                            (or (when env (System/getenv env))
                                                default
                                                (throw (ex-info "Unknown backend target" {:target target}))))
                        :else (throw (ex-info "Invalid backend target specifier" {:target target})))
             api-ctx (pjrt/load-plugin! lib-path)
             client (pjrt/create-client api-ctx (or client-opts {}))
             pname (pjrt/platform-name api-ctx client)
             ctx (assoc api-ctx
                        :client client
                        :platform pname
                        :target target
                        :probe probe-info
                        :xla-flags flag-config)]
         (alter-var-root #'*default-context* (constantly ctx))
         (when-not (or (Boolean/getBoolean "clj-xla.quiet") (:quiet? client-opts))
           (println (format "clj-xla initialized PJRT Backend: [%s] via plugin [%s]" pname lib-path))
           (when-not (str/blank? xla-flags)
             (println (format "  ↳ Autotuned XLA_FLAGS: %s" xla-flags))))
         ctx)))))

(defn init-cpu! [] (init-backend! :cpu))
(defn init-sycl! [] (init-backend! :sycl))
(defn init-rocm! [] (init-backend! :rocm))
(defn init-cuda! [] (init-backend! :cuda12))
(defn init-interpreter!
  "Initializes the pure-JVM StableHLO interpreter backend (zero native PJRT required).
   Sets and returns the thread-root default context *default-context*."
  ([] (init-backend! :interpreter))
  ([opts] (init-backend! :interpreter opts)))

(defn get-context
  "Returns current default PJRT context or initializes CPU client."
  []
  (or *default-context* (init-cpu!)))

(defn compile-graph
  "Compiles EDN SSA graph to a native PJRT loaded executable or pure-JVM interpreter executable."
  ([graph]
   (compile-graph (get-context) graph))
  ([ctx graph]
   (if (= (:backend ctx) :interpreter)
     (let [fused (fuse/fuse-graph graph)]
       {:handle fused
        :graph fused
        :backend :interpreter
        :ctx ctx})
     (let [exec (compile/compile-graph ctx (:client ctx) graph)]
       (if (map? exec)
         (assoc exec :ctx ctx)
         exec)))))

(defn buffer-from-host-buffer
  "Transfers host primitive array/buffer into native PJRT device memory buffer."
  ([host-data shape dtype-enum]
   (let [ctx (get-context)]
     (pjrt/buffer-from-host-buffer ctx (:client ctx) host-data shape dtype-enum)))
  ([ctx client host-data shape dtype-enum]
   (pjrt/buffer-from-host-buffer ctx client host-data shape dtype-enum)))

(defn execute
  "Executes a compiled StableHLO graph executable on the PJRT device runtime or pure-JVM interpreter."
  [exec & inputs]
  (let [flat-inputs (if (and (= 1 (count inputs)) (vector? (first inputs)))
                      (first inputs)
                      inputs)
        ctx (or (when (map? exec) (:ctx exec)) (get-context))
        exec-handle (cond
                      (map? exec) (or (:handle exec) exec)
                      :else exec)]
    (if-not exec-handle
      (throw (ex-info "Invalid executable handle" {:exec exec}))
      (if (= (:backend ctx) :interpreter)
        (let [invars (or (get-in exec [:graph :invars]) [])
              input-tensors (mapv (fn [idx input-data]
                                    (if (and (map? input-data) (contains? input-data :data))
                                      input-data
                                      (let [[_var-name [_kw shape dtype]] (nth invars idx)
                                            shape-vec (vec shape)
                                            dt (or dtype :f32)]
                                        {:dtype (weights/normalize-dtype dt)
                                         :shape shape-vec
                                         :data input-data})))
                                  (range (count flat-inputs))
                                  flat-inputs)
              bindings (zipmap (map first invars) input-tensors)
              out-map (interpret/execute (:graph exec) bindings)]
          (if (= 1 (count (get-in exec [:graph :outvars])))
            (val (first out-map))
            out-map))
        (let [invars (or (get-in exec [:graph :invars]) [])
              all-segs? (every? #(instance? MemorySegment %) flat-inputs)
              device-buffers (if all-segs?
                               flat-inputs
                               (mapv (fn [idx input-data]
                                       (if (instance? MemorySegment input-data)
                                         input-data
                                         (let [[_var-name [_kw shape dtype]] (nth invars idx)
                                               dtype-enum (case dtype :i1 1 :bool 1 :pred 1 :i8 2 :i32 4 :f32 11 :bf16 13 :f16 10 11)]
                                           (pjrt/buffer-from-host-buffer ctx (:client ctx) input-data shape dtype-enum))))
                                     (range (count flat-inputs))
                                     flat-inputs))
              num-outputs (if (map? exec) (count (get-in exec [:graph :outvars] [1])) 1)
              out-buf (pjrt/execute-executable ctx exec-handle device-buffers num-outputs)]
          out-buf)))))

(def ^:private ^Class FLOAT-ARRAY-CLASS (Class/forName "[F"))
(def ^:private ^Class SHORT-ARRAY-CLASS (Class/forName "[S"))

(defn to-host-slice
  "Transfers a slice of PJRT output device buffer or interpreter tensor back to host float array.
   Supports optional `dtype` parameter (:f32 or :bf16), converting bfloat16 to float32 on transfer."
  ([out-buf]
   (to-host-slice out-buf 0 256000 (* 128 256000) :f32))
  ([out-buf slice-idx]
   (to-host-slice out-buf slice-idx 256000 (* 128 256000) :f32))
  ([out-buf slice-idx vocab-size]
   (to-host-slice out-buf slice-idx vocab-size (* 128 vocab-size) :f32))
  ([out-buf slice-idx vocab-size total-elements]
   (to-host-slice out-buf slice-idx vocab-size total-elements :f32))
  ([out-buf slice-idx vocab-size total-elements dtype]
   (let [ctx (get-context)
         n-elements (max (long total-elements) (long (* (inc slice-idx) vocab-size)))
         all-floats (cond
                      ;; 1. Direct float array
                      (instance? FLOAT-ARRAY-CLASS out-buf)
                      ^floats out-buf

                      ;; 2. Interpreter tensor map {:dtype ... :shape ... :data ...}
                      (and (map? out-buf) (contains? out-buf :data))
                      (let [data (:data out-buf)
                            dt (or (:dtype out-buf) dtype :f32)]
                        (cond
                          (instance? FLOAT-ARRAY-CLASS data)
                          ^floats data

                          (instance? SHORT-ARRAY-CLASS data)
                          (let [^shorts s-arr data
                                len (alength s-arr)
                                f-arr (float-array len)]
                            (dotimes [i len]
                              (aset f-arr i (interpret/bf16->f32 (aget s-arr i))))
                            f-arr)

                          (instance? MemorySegment data)
                          (let [^MemorySegment seg data
                                elem-size (if (= dt :bf16) 2 4)
                                len (long (min (quot (.byteSize seg) elem-size) n-elements))
                                f-arr (float-array len)]
                            (if (= dt :bf16)
                              (dotimes [i len]
                                (let [s (.getAtIndex seg ValueLayout/JAVA_SHORT (long i))]
                                  (aset f-arr i (interpret/bf16->f32 s))))
                              (dotimes [i len]
                                (aset f-arr i (.getAtIndex seg ValueLayout/JAVA_FLOAT (long i)))))
                            f-arr)

                          :else
                          (float-array (map float data))))

                      ;; 3. MemorySegment in interpreter mode
                      (and (instance? MemorySegment out-buf) (= (:backend ctx) :interpreter))
                      (let [^MemorySegment seg out-buf
                            len (long (min (quot (.byteSize seg) 4) n-elements))
                            f-arr (float-array len)]
                        (dotimes [i len]
                          (aset f-arr i (.getAtIndex seg ValueLayout/JAVA_FLOAT (long i))))
                        f-arr)

                      ;; 4. PJRT native device buffer
                      :else
                      (pjrt/buffer-to-host-buffer ctx out-buf n-elements (or dtype :f32)))
         offset (* (long slice-idx) (long vocab-size))]
     (if (and (>= offset 0) (<= (+ offset vocab-size) (alength ^floats all-floats)))
       (let [slice (float-array vocab-size)]
         (System/arraycopy all-floats offset slice 0 vocab-size)
         slice)
       all-floats))))

(defn destroy-buffer!
  "Frees native device PJRT_Buffer `buffer-handle`."
  ([buffer-handle]
   (pjrt/destroy-buffer! (get-context) buffer-handle))
  ([ctx buffer-handle]
   (pjrt/destroy-buffer! ctx buffer-handle)))

;; -----------------------------------------------------------------------------
;; Scoped Device Arenas & Memory Management (Proposal 1)
;; -----------------------------------------------------------------------------

(def create-arena arena/create-arena)
(def arena? arena/arena?)
(def track! arena/track!)
(def disown! arena/disown!)
(def promote! arena/promote!)
(def destroy! arena/destroy!)
(def close-arena! arena/close!)
(def device-buffer arena/device-buffer)
(defmacro with-device-arena
  "Executes `body` in a scoped DeviceArena. Automatically destroys all tracked
   device buffers when the block exits (even on unhandled exception)."
  [& args]
  `(arena/with-device-arena ~@args))

;; -----------------------------------------------------------------------------
;; WeightStore & Automated Invar Inference (Proposal 2)
;; -----------------------------------------------------------------------------

(def create-weight-store weights/create-weight-store)
(def infer-invars weights/infer-invars)
(def get-device-buffer weights/get-device-buffer)
(def load-device-buffers! weights/load-device-buffers!)

;; -----------------------------------------------------------------------------
;; Callable Compiled Kernels (Proposal 3)
;; -----------------------------------------------------------------------------

(def compile-kernel kernel/compile-kernel)
(defmacro defkernel
  "Defines a var bound to a compiled kernel function."
  [& args]
  `(kernel/defkernel ~@args))
