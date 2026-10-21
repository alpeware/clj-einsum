(ns einsum.hardware
  "Hardware capability detection and optimal quantization strategy selection."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

;; --- Pure Quantization Strategy Selection ---

(defn estimate-model-footprint-bytes
  "Estimates memory footprint in bytes for a model with `param-count` at `precision` (:bf16, :int8, :int4)."
  [param-count precision]
  (let [params (double param-count)]
    (case precision
      :bf16 (long (* params 2.0))
      :int8 (long (* params 1.05))   ; 1 byte per weight + per-row scales + unquantized norms/embeddings
      :int4 (long (* params 0.55))   ; 0.5 byte per packed weight + scales + unquantized embeddings
      (long (* params 2.0)))))

(defn select-quant-strategy
  "Pure function: determines the optimal quantization strategy (:bf16, :int8, or :int4)
   given a `device-profile` (containing at least `:vram-bytes`) and `model-info` (containing `:param-count`).
   Ensures model weights fit comfortably in VRAM while leaving headroom for KV cache and scratchpad."
  [device-profile model-info]
  (let [vram-bytes (double (get device-profile :vram-bytes 0))
        param-count (get model-info :param-count 0)
        ;; Budget weights at up to 75% of total VRAM, leaving >= 25% for KV cache and scratchpad
        weight-budget (* vram-bytes 0.75)
        bf16-bytes (estimate-model-footprint-bytes param-count :bf16)
        int8-bytes (estimate-model-footprint-bytes param-count :int8)]
    (cond
      (<= (double bf16-bytes) weight-budget) :bf16
      (<= (double int8-bytes) weight-budget) :int8
      :else :int4)))

;; --- Hardware Telemetry & Detection (Boundary Shell) ---

(defn- exec-sh [cmd]
  (try
    (let [pb (ProcessBuilder. ["sh" "-c" cmd])
          proc (.start pb)
          out (str/trim (slurp (.getInputStream proc)))]
      (.waitFor proc)
      (when-not (str/blank? out) out))
    (catch Exception _ nil)))

(defn- detect-rocm-devices []
  (let [cards (filter #(and (.isDirectory %) (.startsWith (.getName %) "card"))
                      (or (.listFiles (io/file "/sys/class/drm")) []))
        devices (keep (fn [card-dir]
                        (let [vram-file (io/file card-dir "device" "mem_info_vram_total")]
                          (when (.exists vram-file)
                            (let [vram-str (str/trim (slurp vram-file))
                                  vram-bytes (Long/parseLong vram-str)
                                  card-name (.getName card-dir)]
                              {:card card-name
                               :vram-bytes vram-bytes}))))
                      cards)]
    (when (seq devices)
      ;; Sort descending by VRAM (primary discrete GPU has the highest VRAM)
      (let [sorted-devs (vec (sort-by :vram-bytes > devices))
            rocminfo-out (exec-sh "rocminfo 2>/dev/null")
            gpu-match (when rocminfo-out
                        (re-find #"Name:\s+(gfx\d+)[\s\S]*?Marketing Name:\s+([^\n]+)" rocminfo-out))
            gfx-target (or (when gpu-match (nth gpu-match 1))
                           (System/getenv "HSA_OVERRIDE_GFX_VERSION")
                           "gfx1100")
            marketing-name (or (when gpu-match (str/trim (nth gpu-match 2)))
                               "AMD Radeon Discrete GPU")
            arch (cond
                   (str/starts-with? gfx-target "gfx11") :rdna3
                   (str/starts-with? gfx-target "gfx10") :rdna2
                   (str/starts-with? gfx-target "gfx9") :cdna
                   :else :rocm)]
        (mapv (fn [idx dev]
                (assoc dev
                       :id idx
                       :name (if (zero? idx) marketing-name (str "AMD GPU " idx))
                       :arch arch
                       :gfx gfx-target))
              (range)
              sorted-devs)))))

(defn- detect-nvidia-devices []
  (let [smi-out (exec-sh "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null")]
    (when (and smi-out (not (str/blank? smi-out)))
      (vec (for [line (str/split-lines smi-out)]
             (let [[idx-s name-s mem-s] (map str/trim (str/split line #","))
                   vram-mb (Long/parseLong mem-s)]
               {:id (Integer/parseInt idx-s)
                :name name-s
                :arch :cuda
                :vram-bytes (* vram-mb 1024 1024)}))))))

(defn- detect-host-memory-bytes []
  (try
    (let [meminfo (java.nio.file.Files/readString (java.nio.file.Path/of "/proc/meminfo" (into-array String [])))
          match (re-find #"MemTotal:\s+(\d+)\s+kB" meminfo)]
      (if match
        (* (Long/parseLong (second match)) 1024)
        (.maxMemory (Runtime/getRuntime))))
    (catch Exception _
      (.maxMemory (Runtime/getRuntime)))))

(defn ensure-amdgpu-active!
  "Ensures the AMD ROCm GPU is in runtime PM `active` state and resets its autosuspend
   timer by touching `/dev/dri/renderD128`, polling `/sys/class/drm/renderD128/device/power/runtime_status`
   synchronously for up to 2 seconds."
  []
  (try
    (let [dev-render (io/file "/dev/dri/renderD128")
          status-file (io/file "/sys/class/drm/renderD128/device/power/runtime_status")]
      (when (and (.exists dev-render) (.exists status-file))
        (loop [attempt 0]
          (let [st (str/trim (slurp status-file))]
            (cond
              (= st "active")
              (try (with-open [_is (io/input-stream dev-render)] nil) (catch Exception _ nil))

              (< attempt 40)
              (do
                (when (= st "suspended")
                  (try (with-open [_is (io/input-stream dev-render)] nil) (catch Exception _ nil)))
                (Thread/sleep 50)
                (recur (inc attempt))))))))
    (catch Throwable _ nil)))

(defn detect-hardware
  "Inspects host system and discrete accelerators, returning a structured hardware profile.
   Detects ROCm (RDNA3/CDNA), NVIDIA CUDA, SYCL, or falls back to CPU."
  []
  (let [rocm-devs (detect-rocm-devices)
        nv-devs (detect-nvidia-devices)
        host-ram (detect-host-memory-bytes)]
    (cond
      (seq rocm-devs)
      {:backend :rocm
       :devices rocm-devs
       :primary-device (first rocm-devs)
       :host-ram-bytes host-ram}

      (seq nv-devs)
      {:backend :cuda12
       :devices nv-devs
       :primary-device (first nv-devs)
       :host-ram-bytes host-ram}

      :else
      {:backend :cpu
       :devices [{:id 0 :name "Host CPU" :arch :cpu :vram-bytes host-ram}]
       :primary-device {:id 0 :name "Host CPU" :arch :cpu :vram-bytes host-ram}
       :host-ram-bytes host-ram})))
