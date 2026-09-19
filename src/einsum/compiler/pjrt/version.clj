(ns einsum.compiler.pjrt.version
  "PJRT API and driver version inspection, compatibility checking, and backend telemetry."
  (:require [einsum.compiler.pjrt :as pjrt]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout ValueLayout]))

(def MINIMUM_SUPPORTED_PJRT_MINOR
  "Minimum required PJRT C API minor version supported by clj-xla."
  10)

(defn set-c-env!
  "Sets a native C process environment variable via Panama setenv downcall."
  [k v]
  (try
    (let [linker (Linker/nativeLinker)
          stdlib (.defaultLookup linker)]
      (when-let [setenv-seg (.orElse nil (.find stdlib "setenv"))]
        (let [fd (FunctionDescriptor/of ValueLayout/JAVA_INT
                                        (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
              handle (.downcallHandle linker setenv-seg fd (into-array Linker$Option []))]
          (with-open [arena (Arena/ofConfined)]
            (let [k-seg (.allocateFrom arena ^String k)
                  v-seg (.allocateFrom arena ^String v)]
              (.invokeWithArguments handle [k-seg v-seg (Integer/valueOf 0)]))))))
    (catch Exception _ nil)))

(defn ensure-hsaco-cache-dir!
  "Ensures TF_XLA_HSACO_CACHE_DIR environment variable is configured to enable OpenXLA HSACO on-disk binary caching."
  []
  (let [cache-dir (io/file (System/getProperty "user.home") ".cache" "hsa_cache")
        abs-path (.getAbsolutePath cache-dir)]
    (.mkdirs cache-dir)
    (System/setProperty "TF_XLA_HSACO_CACHE_DIR" abs-path)
    (set-c-env! "TF_XLA_HSACO_CACHE_DIR" abs-path)
    abs-path))

(defn validate-version-compatibility
  "Validates if PJRT C API `major` and `minor` numbers meet framework requirements."
  ([major minor] (validate-version-compatibility major minor MINIMUM_SUPPORTED_PJRT_MINOR))
  ([major minor min-minor]
   (let [target-major 0
         major-ok (= major target-major)
         minor-ok (>= minor min-minor)
         compat (and major-ok minor-ok)
         reason (cond
                  (not major-ok) (format "Incompatible PJRT C API major version %d (expected %d)." major target-major)
                  (not minor-ok) (format "PJRT C API minor version %d is below minimum required %d." minor min-minor)
                  :else "Compatible PJRT C API version.")]
     {:compatible? compat
      :reason reason
      :major major
      :minor minor
      :min-minor min-minor})))

(defn parse-plugin-attributes
  "Parses raw string/integer attribute map from `pjrt/plugin-attributes` into structured telemetry."
  [attrs]
  (let [attrs (or attrs {})
        driver-ver (or (get attrs "rocm_version")
                       (get attrs "cuda_version")
                       (get attrs "driver_version")
                       (get attrs "version"))
        platform (or (get attrs "platform_name")
                     (get attrs "platform")
                     "unknown")]
    {:raw-attributes attrs
     :driver-version (when driver-ver (str driver-ver))
     :platform-name (str platform)}))

(defn- exec-cmd-string
  "Executes shell command `cmd` returning non-empty stdout line or nil."
  [cmd]
  (try
    (let [pb (ProcessBuilder. ["sh" "-c" cmd])
          proc (.start pb)
          out (str/trim (slurp (.getInputStream proc)))]
      (.waitFor proc)
      (when-not (str/blank? out) out))
    (catch Exception _ nil)))

(defn probe-system-driver
  "Probes system drivers for ROCm, CUDA, SYCL, and CPU support."
  []
  (let [amdgpu-ver-file (io/file "/sys/module/amdgpu/version")
        rocm-ver-file   (io/file "/opt/rocm/version")
        hsa-lib64       (io/file "/usr/lib64/libhsa-runtime64.so")
        hsa-lib-opt     (io/file "/opt/rocm/lib/libhsa-runtime64.so")
        nvidia-ver-file (io/file "/proc/driver/nvidia/version")
        cuda-lib64      (io/file "/usr/local/cuda/lib64/libcudart.so")
        sycl-lib64-1    (io/file "/usr/lib64/libze_loader.so")
        sycl-lib64-2    (io/file "/usr/lib/x86_64-linux-gnu/libze_loader.so")
        sycl-lib64-3    (io/file "/usr/lib/libze_loader.so")
        sycl-pjrt-so    (io/file "bin/libpjrt_sycl.so")

        cmd-rocm-ver    (or (exec-cmd-string "hipconfig --version 2>/dev/null")
                            (exec-cmd-string "rocminfo 2>/dev/null | grep -i 'ROCm Version' | head -n1 | awk '{print $3}'"))
        cmd-sycl-drv    (or (exec-cmd-string "clinfo 2>/dev/null | grep -i 'Driver Version' | head -n1 | awk '{print $3}'")
                            (exec-cmd-string "sycl-ls --version 2>/dev/null"))
        cmd-ze-ver      (or (exec-cmd-string "ls -la /usr/lib*/libze_loader.so.1 2>/dev/null | awk '{print $NF}' | sed 's/.*libze_loader.so.//'")
                            (exec-cmd-string "ls -la /usr/lib*/libze_intel_gpu.so.1 2>/dev/null | awk '{print $NF}' | sed 's/.*libze_intel_gpu.so.//'"))

        rocm-detected? (or (.exists amdgpu-ver-file)
                           (.exists rocm-ver-file)
                           (.exists hsa-lib64)
                           (.exists hsa-lib-opt)
                           (some? cmd-rocm-ver))

        cuda-detected? (or (.exists nvidia-ver-file)
                           (.exists cuda-lib64))

        sycl-detected? (or (.exists sycl-lib64-1)
                           (.exists sycl-lib64-2)
                           (.exists sycl-lib64-3)
                           (.exists sycl-pjrt-so)
                           (some? cmd-sycl-drv)
                           (some? cmd-ze-ver))

        rocm-ver (or cmd-rocm-ver
                     (cond
                       (.exists amdgpu-ver-file) (try (str/trim (slurp amdgpu-ver-file)) (catch Exception _ nil))
                       (.exists rocm-ver-file)   (try (str/trim (slurp rocm-ver-file)) (catch Exception _ nil))
                       :else nil))

        cuda-ver (cond
                   (.exists nvidia-ver-file) (try
                                               (let [line (first (str/split-lines (slurp nvidia-ver-file)))]
                                                 (second (re-find #"Kernel Module\s+([\d.]+)" line)))
                                               (catch Exception _ nil))
                   :else nil)

        sycl-ver (cond
                   (and cmd-sycl-drv cmd-ze-ver) (str "Driver " cmd-sycl-drv " (Level-Zero v" cmd-ze-ver ")")
                   cmd-sycl-drv (str "Driver " cmd-sycl-drv)
                   cmd-ze-ver (str "Level-Zero v" cmd-ze-ver)
                   sycl-detected? "Level-Zero/oneAPI"
                   :else nil)

        detected (cond-> #{:cpu}
                   rocm-detected? (conj :rocm)
                   cuda-detected? (conj :cuda12)
                   sycl-detected? (conj :sycl))]
    {:detected-backends detected
     :details {:rocm {:detected? rocm-detected? :version rocm-ver}
               :cuda {:detected? cuda-detected? :version cuda-ver}
               :sycl {:detected? sycl-detected? :version sycl-ver}
               :cpu  {:detected? true :java-version (System/getProperty "java.version")}}}))

(defn inspect-plugin
  "Queries `api-ctx` for PJRT C API version, validates compatibility, and retrieves telemetry."
  [api-ctx]
  (let [[major minor] (pjrt/api-version api-ctx)
        compat-map (validate-version-compatibility major minor)
        attrs (try (pjrt/plugin-attributes api-ctx) (catch Exception _ {}))
        parsed-attrs (parse-plugin-attributes attrs)]
    (merge compat-map
           parsed-attrs
           {:system-driver (probe-system-driver)})))
