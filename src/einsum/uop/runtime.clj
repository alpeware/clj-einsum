(ns einsum.uop.runtime
  "Native Project Panama FFM Runtime for the `:uop` backend (`:uop`, `:uop-cpu`, `:uop-rocm`).
   Modeled on tinygrad's `runtime/ops_cpu.py` and `runtime/ops_hip.py` (Weeks 2 & 8 of Compilers for ML).
   - `:uop-cpu`: Compiles deduplicated C99 kernels via `gcc -O3 -march=native -ffast-math -shared -fPIC`
     into `~/.cache/clj_einsum/uop/`, loads via Panama `SymbolLookup/libraryLookup`, and invokes via `Linker`.
   - `:uop-rocm` / `:uop`: Compiles deduplicated HIP C++ kernels in-process via `libhiprtc.so` into HSACO
     binaries (cached on disk in `~/.cache/clj_einsum/uop/`), manages VRAM via `libamdhip64.so`, and dispatches
     kernels back-to-back on the default HIP stream with zero intermediate host-device synchronization."
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [einsum.logic.interpret :as interpret]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.opt :as opt]
            [einsum.uop.render :as render]
            [einsum.uop.schedule :as schedule])
  (:import [java.io File]
           [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]
           [java.nio.file Files StandardCopyOption]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.locks ReentrantLock]))

;; -----------------------------------------------------------------------------
;; 1. Data Type Sizes & Host Memory Conversions
;; -----------------------------------------------------------------------------

(def ^:private ^Class FLOAT-ARRAY-CLASS (Class/forName "[F"))
(def ^:private ^Class INT-ARRAY-CLASS (Class/forName "[I"))
(def ^:private ^Class SHORT-ARRAY-CLASS (Class/forName "[S"))
(def ^:private ^Class BYTE-ARRAY-CLASS (Class/forName "[B"))
(def ^:private ^Class DOUBLE-ARRAY-CLASS (Class/forName "[D"))
(def ^:private ^Class LONG-ARRAY-CLASS (Class/forName "[J"))

(defn dtype-byte-size
  "Returns byte width for a UOp compute dtype."
  ^long [dtype]
  (case dtype
    (:bool :i1 :i8 :u8 :byte) 1
    (:i16 :u16 :f16 :bf16 :short) 2
    (:i32 :u32 :f32 :int :float) 4
    (:i64 :u64 :f64 :long :double) 8
    4))

(defn normalize-compute-dtype
  "Normalizes tensor dtype keyword to UOp compute dtype (`:bf16` and `:f16` compute as `:f32`)."
  [dtype]
  (let [dt (or dtype :f32)]
    (if (contains? #{:bf16 :f16 :bfloat16 :float} dt)
      :f32
      (case dt
        :int :i32
        :byte :i8
        :bool :i32
        :i1 :i32
        dt))))

(defn- sha256-hex
  "Computes lower-case hex SHA-256 of string `s`."
  ^String [^String s]
  (let [md (java.security.MessageDigest/getInstance "SHA-256")
        bytes (.digest md (.getBytes s "UTF-8"))
        sb (StringBuilder. 64)]
    (doseq [b bytes]
      (.append sb (format "%02x" (bit-and (int b) 0xff))))
    (.toString sb)))

(defn- ensure-cache-dir!
  "Ensures `~/.cache/clj_einsum/uop/` exists and returns its `java.io.File`."
  ^File []
  (let [dir (io/file (System/getProperty "user.home") ".cache" "clj_einsum" "uop")]
    (.mkdirs dir)
    dir))

;; -----------------------------------------------------------------------------
;; 2. HIP & HIPRTC Project Panama FFM Bindings (`/usr/lib64/libamdhip64.so`)
;; -----------------------------------------------------------------------------

(defonce ^:private hip-runtime-state (atom nil))

(defn- make-downcall
  ^MethodHandle [^Linker linker ^SymbolLookup lookup ^String sym-name ^FunctionDescriptor desc]
  (let [opt (.find lookup sym-name)]
    (when-not (.isPresent opt)
      (throw (ex-info (str "Native symbol not found: " sym-name) {:symbol sym-name})))
    (.downcallHandle linker ^MemorySegment (.get opt) desc (make-array Linker$Option 0))))

(defn- init-hip-bindings!
  "Loads `libamdhip64.so` and `libhiprtc.so` via Project Panama FFM and initializes HIP device 0.
   Returns the HIP bindings map or nil if ROCm libraries / GPU hardware are unavailable."
  []
  (if-let [cached @hip-runtime-state]
    (when (:available? cached) cached)
    (let [state
          (try
            (let [hip-path (or (System/getenv "HIP_LIBRARY_PATH") "/usr/lib64/libamdhip64.so")
                  rtc-path (or (System/getenv "HIPRTC_LIBRARY_PATH") "/usr/lib64/libhiprtc.so")
                  hip-file (io/file hip-path)
                  rtc-file (io/file rtc-path)]
              (if-not (and (.exists hip-file) (.exists rtc-file))
                {:available? false :reason "ROCm HIP/HIPRTC shared libraries not found"}
                (let [lib-arena (Arena/ofShared)
                      linker (Linker/nativeLinker)
                      hip-lookup (SymbolLookup/libraryLookup (.getAbsolutePath hip-file) lib-arena)
                      rtc-lookup (SymbolLookup/libraryLookup (.getAbsolutePath rtc-file) lib-arena)
                      addr ValueLayout/ADDRESS
                      jint ValueLayout/JAVA_INT
                      jlong ValueLayout/JAVA_LONG

                      ;; hipInit(unsigned int flags) -> hipError_t
                      hipInit (make-downcall linker hip-lookup "hipInit"
                                             (FunctionDescriptor/of jint (into-array MemoryLayout [jint])))
                      ;; hipGetDeviceCount(int* count) -> hipError_t
                      hipGetDeviceCount (make-downcall linker hip-lookup "hipGetDeviceCount"
                                                       (FunctionDescriptor/of jint (into-array MemoryLayout [addr])))
                      ;; hipMalloc(void** ptr, size_t size) -> hipError_t
                      hipMalloc (make-downcall linker hip-lookup "hipMalloc"
                                               (FunctionDescriptor/of jint (into-array MemoryLayout [addr jlong])))
                      ;; hipFree(void* ptr) -> hipError_t
                      hipFree (make-downcall linker hip-lookup "hipFree"
                                             (FunctionDescriptor/of jint (into-array MemoryLayout [addr])))
                      ;; hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) -> hipError_t
                      hipMemcpy (make-downcall linker hip-lookup "hipMemcpy"
                                               (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr jlong jint])))
                      ;; hipModuleLoadData(hipModule_t* module, const void* image) -> hipError_t
                      hipModuleLoadData (make-downcall linker hip-lookup "hipModuleLoadData"
                                                       (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr])))
                      ;; hipModuleUnload(hipModule_t module) -> hipError_t
                      hipModuleUnload (make-downcall linker hip-lookup "hipModuleUnload"
                                                     (FunctionDescriptor/of jint (into-array MemoryLayout [addr])))
                      ;; hipModuleGetFunction(hipFunction_t* function, hipModule_t module, const char* kname) -> hipError_t
                      hipModuleGetFunction (make-downcall linker hip-lookup "hipModuleGetFunction"
                                                          (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr addr])))
                      ;; hipModuleLaunchKernel(hipFunction_t f, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint sharedMem, hipStream_t stream, void** params, void** extra)
                      hipModuleLaunchKernel (make-downcall linker hip-lookup "hipModuleLaunchKernel"
                                                           (FunctionDescriptor/of jint (into-array MemoryLayout
                                                                                                   [addr jint jint jint jint jint jint jint addr addr addr])))
                      ;; hipDeviceSynchronize() -> hipError_t
                      hipDeviceSynchronize (make-downcall linker hip-lookup "hipDeviceSynchronize"
                                                          (FunctionDescriptor/of jint (into-array MemoryLayout [])))

                      ;; HIPRTC bindings
                      ;; hiprtcCreateProgram(hiprtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames)
                      hiprtcCreateProgram (make-downcall linker rtc-lookup "hiprtcCreateProgram"
                                                         (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr addr jint addr addr])))
                      ;; hiprtcCompileProgram(hiprtcProgram prog, int numOptions, const char* const* options)
                      hiprtcCompileProgram (make-downcall linker rtc-lookup "hiprtcCompileProgram"
                                                          (FunctionDescriptor/of jint (into-array MemoryLayout [addr jint addr])))
                      ;; hiprtcGetProgramLogSize(hiprtcProgram prog, size_t* logSizeRet)
                      hiprtcGetProgramLogSize (make-downcall linker rtc-lookup "hiprtcGetProgramLogSize"
                                                             (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr])))
                      ;; hiprtcGetProgramLog(hiprtcProgram prog, char* log)
                      hiprtcGetProgramLog (make-downcall linker rtc-lookup "hiprtcGetProgramLog"
                                                         (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr])))
                      ;; hiprtcGetCodeSize(hiprtcProgram prog, size_t* codeSizeRet)
                      hiprtcGetCodeSize (make-downcall linker rtc-lookup "hiprtcGetCodeSize"
                                                       (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr])))
                      ;; hiprtcGetCode(hiprtcProgram prog, char* code)
                      hiprtcGetCode (make-downcall linker rtc-lookup "hiprtcGetCode"
                                                   (FunctionDescriptor/of jint (into-array MemoryLayout [addr addr])))
                      ;; hiprtcDestroyProgram(hiprtcProgram* prog)
                      hiprtcDestroyProgram (make-downcall linker rtc-lookup "hiprtcDestroyProgram"
                                                          (FunctionDescriptor/of jint (into-array MemoryLayout [addr])))]
                  (with-open [tmp (Arena/ofConfined)]
                    (let [init-rc (int (.invokeWithArguments hipInit (into-array Object [(int 0)])))
                          cnt-seg (.allocate tmp ValueLayout/JAVA_INT)
                          cnt-rc (int (.invokeWithArguments hipGetDeviceCount (into-array Object [cnt-seg])))
                          dev-count (if (zero? cnt-rc) (.get cnt-seg ValueLayout/JAVA_INT 0) 0)]
                      (if (and (zero? init-rc) (zero? cnt-rc) (pos? dev-count))
                        {:available? true
                         :device-count dev-count
                         :arch (or (System/getenv "HSA_GPU_ARCH") "gfx1100")
                         :hipInit hipInit
                         :hipGetDeviceCount hipGetDeviceCount
                         :hipMalloc hipMalloc
                         :hipFree hipFree
                         :hipMemcpy hipMemcpy
                         :hipModuleLoadData hipModuleLoadData
                         :hipModuleUnload hipModuleUnload
                         :hipModuleGetFunction hipModuleGetFunction
                         :hipModuleLaunchKernel hipModuleLaunchKernel
                         :hipDeviceSynchronize hipDeviceSynchronize
                         :hiprtcCreateProgram hiprtcCreateProgram
                         :hiprtcCompileProgram hiprtcCompileProgram
                         :hiprtcGetProgramLogSize hiprtcGetProgramLogSize
                         :hiprtcGetProgramLog hiprtcGetProgramLog
                         :hiprtcGetCodeSize hiprtcGetCodeSize
                         :hiprtcGetCode hiprtcGetCode
                         :hiprtcDestroyProgram hiprtcDestroyProgram}
                        {:available? false
                         :reason (format "hipInit=%d, hipGetDeviceCount=%d, count=%d" init-rc cnt-rc dev-count)}))))))
            (catch Throwable t
              {:available? false :reason (.getMessage t)}))]
      (reset! hip-runtime-state state)
      (when (:available? state) state))))

(defn- check-hip! [^long rc ^String op-name]
  (when-not (zero? rc)
    (throw (ex-info (format "HIP runtime error in %s: error code %d" op-name rc)
                    {:op op-name :hip-error rc}))))

;; -----------------------------------------------------------------------------
;; 3. Device Buffer Allocation, Host <-> Device Transfers, & Cleanup
;; -----------------------------------------------------------------------------

(defn uop-buffer?
  "Returns true if `x` is a UOp device buffer map."
  [x]
  (and (map? x) (true? (:uop-buffer? x))))

(defn allocate-device-buffer
  "Allocates an uninitialized device buffer of `elem-count` elements of `dtype` on `ctx`."
  [ctx shape dtype]
  (let [target (or (:uop-target ctx) :cpu)
        compute-dt (normalize-compute-dtype dtype)
        elem-count (long (reduce * 1 shape))
        byte-size (max 8 (* elem-count (dtype-byte-size compute-dt)))]
    (if (= target :rocm)
      (let [hip (or (:hip ctx) (init-hip-bindings!))
            ^MethodHandle hipMalloc (:hipMalloc hip)]
        (with-open [tmp (Arena/ofConfined)]
          (let [ptr-slot (.allocate tmp ValueLayout/ADDRESS)
                rc (int (.invokeWithArguments hipMalloc (into-array Object [ptr-slot (long byte-size)])))
                _ (check-hip! rc "hipMalloc")
                dptr (.get ptr-slot ValueLayout/ADDRESS 0)]
            {:uop-buffer? true
             :target :rocm
             :dtype compute-dt
             :orig-dtype (or dtype :f32)
             :shape (vec shape)
             :size elem-count
             :byte-size byte-size
             :ptr dptr
             :owned? true})))
      (let [buf-arena (Arena/ofShared)
            seg (.allocate buf-arena (long byte-size) 64)]
        {:uop-buffer? true
         :target :cpu
         :dtype compute-dt
         :orig-dtype (or dtype :f32)
         :shape (vec shape)
         :size elem-count
         :byte-size byte-size
         :ptr seg
         :arena buf-arena
         :owned? true}))))

(defn destroy-buffer!
  "Frees a UOp device buffer (`hipFree` on `:rocm` or closes shared `Arena` on `:cpu`)."
  [ctx buf]
  (when (and (uop-buffer? buf) (:owned? buf))
    (try
      (if (= :rocm (:target buf))
        (when-let [hip (or (:hip ctx) @hip-runtime-state)]
          (let [^MethodHandle hipFree (:hipFree hip)
                ^MemorySegment dptr (:ptr buf)]
            (when (and dptr (not= dptr MemorySegment/NULL))
              (.invokeWithArguments hipFree (into-array Object [dptr])))))
        (when-let [^Arena a (:arena buf)]
          (when (.isAlive (.scope a))
            (.close a))))
      (catch Throwable _ nil)))
  nil)

(defn- populate-host-segment!
  "Writes `host-data` into off-heap `MemorySegment` `dst-seg` of `compute-dt` (`:f32`, `:i32`, `:i8`, `:f64`).
   Transparently upcasts `:bf16` `short[]` or `:bf16` `MemorySegment` slices into `:f32`."
  [^MemorySegment dst-seg host-data elem-count orig-dtype compute-dt]
  (let [elem-count (long elem-count)]
    (cond
    ;; 1. Already a MemorySegment
      (instance? MemorySegment host-data)
      (let [^MemorySegment src-seg host-data
            src-bytes (.byteSize src-seg)
            dst-bytes (* elem-count (dtype-byte-size compute-dt))]
        (if (and (= orig-dtype :bf16) (= compute-dt :f32) (== src-bytes (* 2 elem-count)))
        ;; Upcast bf16 MemorySegment to f32 in bulk
          (dotimes [i elem-count]
            (let [s (.get src-seg ValueLayout/JAVA_SHORT_UNALIGNED (* i 2))]
              (.setAtIndex dst-seg ValueLayout/JAVA_FLOAT (long i) (interpret/bf16->f32 s))))
          (MemorySegment/copy src-seg 0 dst-seg 0 (min src-bytes dst-bytes))))

    ;; 2. Float array
      (instance? FLOAT-ARRAY-CLASS host-data)
      (let [^floats arr host-data
            n (min elem-count (alength arr))]
        (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_FLOAT 0 n))

    ;; 3. Int array
      (instance? INT-ARRAY-CLASS host-data)
      (let [^ints arr host-data
            n (min elem-count (alength arr))]
        (if (= compute-dt :f32)
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_FLOAT (long i) (float (aget arr i))))
          (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_INT 0 n)))

    ;; 4. Short array (e.g. bf16)
      (instance? SHORT-ARRAY-CLASS host-data)
      (let [^shorts arr host-data
            n (min elem-count (alength arr))]
        (if (= compute-dt :f32)
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_FLOAT (long i) (interpret/bf16->f32 (aget arr i))))
          (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_SHORT 0 n)))

    ;; 5. Byte array
      (instance? BYTE-ARRAY-CLASS host-data)
      (let [^bytes arr host-data
            n (min elem-count (alength arr))]
        (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_BYTE 0 n))

    ;; 6. Double array
      (instance? DOUBLE-ARRAY-CLASS host-data)
      (let [^doubles arr host-data
            n (min elem-count (alength arr))]
        (if (= compute-dt :f32)
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_FLOAT (long i) (float (aget arr i))))
          (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_DOUBLE 0 n)))

    ;; 7. Long array
      (instance? LONG-ARRAY-CLASS host-data)
      (let [^longs arr host-data
            n (min elem-count (alength arr))]
        (if (= compute-dt :i32)
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_INT (long i) (int (aget arr i))))
          (MemorySegment/copy arr 0 dst-seg ValueLayout/JAVA_LONG 0 n)))

    ;; 8. Interpreter-style tensor map {:data ...}
      (and (map? host-data) (contains? host-data :data))
      (populate-host-segment! dst-seg (:data host-data) elem-count (or (:dtype host-data) orig-dtype) compute-dt)

    ;; 9. Sequential Clojure collection
      (sequential? host-data)
      (let [v (vec host-data)
            n (min elem-count (count v))]
        (if (= compute-dt :i32)
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_INT (long i) (int (nth v i))))
          (dotimes [i n]
            (.setAtIndex dst-seg ValueLayout/JAVA_FLOAT (long i) (float (nth v i))))))

      :else
      (throw (ex-info "Unsupported host-data type in uop.runtime/host->device"
                      {:type (class host-data) :dtype orig-dtype})))))

(defn host->device
  "Transfers `host-data` into a UOp device buffer (`:cpu` off-heap `MemorySegment` or `:rocm` VRAM pointer)."
  [ctx host-data shape dtype]
  (if (uop-buffer? host-data)
    host-data
    (let [target (or (:uop-target ctx) :cpu)
          orig-dt (or dtype :f32)
          compute-dt (normalize-compute-dtype orig-dt)
          elem-count (long (reduce * 1 shape))
          byte-size (max 8 (* elem-count (dtype-byte-size compute-dt)))]
      ;; Fast path on CPU if `host-data` is already an aligned off-heap MemorySegment of matching dtype
      (if (and (= target :cpu)
               (instance? MemorySegment host-data)
               (= orig-dt compute-dt)
               (>= (.byteSize ^MemorySegment host-data) (* elem-count (dtype-byte-size compute-dt))))
        {:uop-buffer? true
         :target :cpu
         :dtype compute-dt
         :orig-dtype orig-dt
         :shape (vec shape)
         :size elem-count
         :byte-size (.byteSize ^MemorySegment host-data)
         :ptr ^MemorySegment host-data
         :owned? false}
        (let [dev-buf (allocate-device-buffer ctx shape orig-dt)]
          (if (= target :rocm)
            (let [hip (or (:hip ctx) (init-hip-bindings!))
                  ^MethodHandle hipMemcpy (:hipMemcpy hip)]
              ;; If host-data is already a MemorySegment of matching dtype, copy directly HtoD without staging!
              (if (and (instance? MemorySegment host-data)
                       (= orig-dt compute-dt)
                       (>= (.byteSize ^MemorySegment host-data) (* elem-count (dtype-byte-size compute-dt))))
                (let [rc (int (.invokeWithArguments hipMemcpy
                                                    (into-array Object [(:ptr dev-buf)
                                                                        ^MemorySegment host-data
                                                                        (long (* elem-count (dtype-byte-size compute-dt)))
                                                                        (int 1)])))]
                  (check-hip! rc "hipMemcpy(HtoD direct)"))
                (with-open [staging (Arena/ofConfined)]
                  (let [stage-seg (.allocate staging (long byte-size) 64)
                        _ (populate-host-segment! stage-seg host-data elem-count orig-dt compute-dt)
                        rc (int (.invokeWithArguments hipMemcpy
                                                      (into-array Object [(:ptr dev-buf)
                                                                          stage-seg
                                                                          (long (* elem-count (dtype-byte-size compute-dt)))
                                                                          (int 1)])))]
                    (check-hip! rc "hipMemcpy(HtoD staged)")))))
            (populate-host-segment! (:ptr dev-buf) host-data elem-count orig-dt compute-dt))
          dev-buf)))))

(defn device->host-floats
  "Transfers `n-elements` from `uop-buf` back to a host `float[]` array."
  ^floats [ctx uop-buf ^long n-elements]
  (let [elem-count (min (long (:size uop-buf n-elements)) n-elements)
        dt (:dtype uop-buf :f32)
        out (float-array elem-count)]
    (if (= :rocm (:target uop-buf))
      (let [hip (or (:hip ctx) (init-hip-bindings!))
            ^MethodHandle hipMemcpy (:hipMemcpy hip)
            byte-len (* elem-count (dtype-byte-size dt))]
        (with-open [staging (Arena/ofConfined)]
          (let [stage-seg (.allocate staging (long (max 8 byte-len)) 64)
                rc (int (.invokeWithArguments hipMemcpy
                                              (into-array Object [stage-seg
                                                                  (:ptr uop-buf)
                                                                  (long byte-len)
                                                                  (int 2)])))]
            (check-hip! rc "hipMemcpy(DtoH)")
            (if (= dt :i32)
              (dotimes [i elem-count]
                (aset out i (float (.getAtIndex stage-seg ValueLayout/JAVA_INT (long i)))))
              (MemorySegment/copy stage-seg ValueLayout/JAVA_FLOAT 0 out 0 (int elem-count))))))
      (let [^MemorySegment seg (:ptr uop-buf)]
        (if (= dt :i32)
          (dotimes [i elem-count]
            (aset out i (float (.getAtIndex seg ValueLayout/JAVA_INT (long i)))))
          (MemorySegment/copy seg ValueLayout/JAVA_FLOAT 0 out 0 (int elem-count)))))
    out))

(defn device->host-ints
  "Transfers `n-elements` from `uop-buf` back to a host `int[]` array."
  ^ints [ctx uop-buf ^long n-elements]
  (let [elem-count (min (long (:size uop-buf n-elements)) n-elements)
        out (int-array elem-count)]
    (if (= :rocm (:target uop-buf))
      (let [hip (or (:hip ctx) (init-hip-bindings!))
            ^MethodHandle hipMemcpy (:hipMemcpy hip)
            byte-len (* elem-count 4)]
        (with-open [staging (Arena/ofConfined)]
          (let [stage-seg (.allocate staging (long (max 8 byte-len)) 64)
                rc (int (.invokeWithArguments hipMemcpy
                                              (into-array Object [stage-seg
                                                                  (:ptr uop-buf)
                                                                  (long byte-len)
                                                                  (int 2)])))]
            (check-hip! rc "hipMemcpy(DtoH int)")
            (MemorySegment/copy stage-seg ValueLayout/JAVA_INT 0 out 0 (int elem-count)))))
      (let [^MemorySegment seg (:ptr uop-buf)]
        (MemorySegment/copy seg ValueLayout/JAVA_INT 0 out 0 (int elem-count))))
    out))

;; -----------------------------------------------------------------------------
;; 4. Native Code Compilation & Disk Caching (GCC for CPU, HIPRTC for ROCm)
;; -----------------------------------------------------------------------------

(defonce ^:private compiled-module-cache (ConcurrentHashMap.))

(defn- compile-cpu-module!
  "Compiles a batched C99 translation unit containing all unique kernels for a schedule
   into a shared library `.so` via `gcc`, caching the `.so` in `~/.cache/clj_einsum/uop/`."
  [unique-rendered-kernels]
  (let [sb (StringBuilder.)
        _ (.append sb "// Generated by clj-einsum UOp Batched C99 Module\n")
        _ (.append sb "#include <math.h>\n#include <stdint.h>\n\n")
        _ (doseq [rk unique-rendered-kernels]
            (.append sb (:canonical-body rk))
            (.append sb "\n"))
        full-src (.toString sb)
        mod-hash (subs (sha256-hex full-src) 0 24)]
    (or (.get compiled-module-cache mod-hash)
        (let [cache-dir (ensure-cache-dir!)
              so-file (io/file cache-dir (str "libuop_" mod-hash ".so"))]
          (when-not (.exists so-file)
            (let [tmp-c (io/file cache-dir (str "tmp_" mod-hash "_" (System/nanoTime) ".c"))
                  tmp-so (io/file cache-dir (str "tmp_" mod-hash "_" (System/nanoTime) ".so"))]
              (spit tmp-c full-src)
              (let [res (sh/sh "gcc" "-O3" "-march=native" "-ffast-math" "-shared" "-fPIC"
                               (.getAbsolutePath tmp-c) "-lm" "-o" (.getAbsolutePath tmp-so))]
                (.delete tmp-c)
                (when-not (zero? (:exit res))
                  (.delete tmp-so)
                  (throw (ex-info (str "UOp C99 gcc compilation failed: " (:err res))
                                  {:exit (:exit res) :err (:err res) :source full-src})))
                (Files/move (.toPath tmp-so) (.toPath so-file)
                            (into-array java.nio.file.CopyOption [StandardCopyOption/REPLACE_EXISTING])))))
          (let [lib-arena (Arena/ofShared)
                lookup (SymbolLookup/libraryLookup (.getAbsolutePath so-file) lib-arena)
                linker (Linker/nativeLinker)
                desc (FunctionDescriptor/ofVoid (into-array MemoryLayout [ValueLayout/ADDRESS]))
                fn-handles (into {}
                                 (map (fn [{:keys [canonical-fn-name]}]
                                        (let [opt (.find lookup canonical-fn-name)]
                                          (when-not (.isPresent opt)
                                            (throw (ex-info (str "Symbol not found in UOp CPU module: " canonical-fn-name)
                                                            {:symbol canonical-fn-name :so (.getAbsolutePath so-file)})))
                                          [canonical-fn-name
                                           (.downcallHandle linker ^MemorySegment (.get opt) desc (make-array Linker$Option 0))]))
                                      unique-rendered-kernels))
                mod-entry {:target :cpu
                           :hash mod-hash
                           :so-path (.getAbsolutePath so-file)
                           :arena lib-arena
                           :fn-handles fn-handles}]
            (.put compiled-module-cache mod-hash mod-entry)
            mod-entry)))))

(defn- compile-rocm-module!
  "Compiles a batched HIP C++ translation unit containing all unique kernels for a schedule
   into an HSACO code object via `libhiprtc.so` (cached on disk in `~/.cache/clj_einsum/uop/`),
   loads it via `hipModuleLoadData`, and resolves `hipFunction_t` handles."
  [ctx unique-rendered-kernels]
  (let [hip (or (:hip ctx) (init-hip-bindings!))
        arch (or (:arch hip) "gfx1100")
        sb (StringBuilder.)
        _ (.append sb "// Generated by clj-einsum UOp Batched HIP Module\n")
        _ (.append sb "#ifndef INFINITY\n#define INFINITY (1.0f/0.0f)\n#endif\n\n")
        _ (doseq [rk unique-rendered-kernels]
            (.append sb (:canonical-body rk))
            (.append sb "\n"))
        full-src (.toString sb)
        mod-hash (subs (sha256-hex (str arch ":" full-src)) 0 24)]
    (or (.get compiled-module-cache mod-hash)
        (let [cache-dir (ensure-cache-dir!)
              hsaco-file (io/file cache-dir (str "uop_" arch "_" mod-hash ".hsaco"))
              ^bytes hsaco-bytes
              (if (.exists hsaco-file)
                (Files/readAllBytes (.toPath hsaco-file))
                (with-open [arena (Arena/ofConfined)]
                  (let [^MethodHandle hiprtcCreateProgram (:hiprtcCreateProgram hip)
                        ^MethodHandle hiprtcCompileProgram (:hiprtcCompileProgram hip)
                        ^MethodHandle hiprtcGetProgramLogSize (:hiprtcGetProgramLogSize hip)
                        ^MethodHandle hiprtcGetProgramLog (:hiprtcGetProgramLog hip)
                        ^MethodHandle hiprtcGetCodeSize (:hiprtcGetCodeSize hip)
                        ^MethodHandle hiprtcGetCode (:hiprtcGetCode hip)
                        ^MethodHandle hiprtcDestroyProgram (:hiprtcDestroyProgram hip)
                        prog-slot (.allocate arena ValueLayout/ADDRESS)
                        src-seg (.allocateFrom arena full-src)
                        name-seg (.allocateFrom arena (str "uop_" mod-hash ".cu"))
                        rc-create (int (.invokeWithArguments hiprtcCreateProgram
                                                             (into-array Object [prog-slot src-seg name-seg (int 0)
                                                                                 MemorySegment/NULL MemorySegment/NULL])))
                        _ (check-hip! rc-create "hiprtcCreateProgram")
                        prog (.get prog-slot ValueLayout/ADDRESS 0)
                        opts [(str "--offload-arch=" arch) "-O3" "-ffast-math"]
                        opts-seg (.allocate arena ValueLayout/ADDRESS (count opts))]
                    (doseq [[idx opt-str] (map-indexed vector opts)]
                      (.setAtIndex opts-seg ValueLayout/ADDRESS (long idx) (.allocateFrom arena ^String opt-str)))
                    (let [rc-comp (int (.invokeWithArguments hiprtcCompileProgram
                                                             (into-array Object [prog (int (count opts)) opts-seg])))]
                      (when-not (zero? rc-comp)
                        (let [log-sz-slot (.allocate arena ValueLayout/JAVA_LONG)
                              _ (.invokeWithArguments hiprtcGetProgramLogSize (into-array Object [prog log-sz-slot]))
                              log-sz (.get log-sz-slot ValueLayout/JAVA_LONG 0)
                              log-seg (.allocate arena (max 1 log-sz))
                              _ (.invokeWithArguments hiprtcGetProgramLog (into-array Object [prog log-seg]))
                              log-str (.getString log-seg 0)]
                          (.invokeWithArguments hiprtcDestroyProgram (into-array Object [prog-slot]))
                          (throw (ex-info (str "HIPRTC kernel compilation failed:\n" log-str)
                                          {:rc rc-comp :log log-str :source full-src}))))
                      (let [code-sz-slot (.allocate arena ValueLayout/JAVA_LONG)
                            rc-sz (int (.invokeWithArguments hiprtcGetCodeSize (into-array Object [prog code-sz-slot])))
                            _ (check-hip! rc-sz "hiprtcGetCodeSize")
                            code-sz (.get code-sz-slot ValueLayout/JAVA_LONG 0)
                            code-seg (.allocate arena code-sz)
                            rc-code (int (.invokeWithArguments hiprtcGetCode (into-array Object [prog code-seg])))
                            _ (check-hip! rc-code "hiprtcGetCode")
                            _ (.invokeWithArguments hiprtcDestroyProgram (into-array Object [prog-slot]))
                            bytes-arr (byte-array code-sz)]
                        (MemorySegment/copy code-seg ValueLayout/JAVA_BYTE 0 bytes-arr 0 (int code-sz))
                        (let [tmp-hsaco (io/file cache-dir (str "tmp_" mod-hash "_" (System/nanoTime) ".hsaco"))]
                          (Files/write (.toPath tmp-hsaco) bytes-arr (make-array java.nio.file.OpenOption 0))
                          (Files/move (.toPath tmp-hsaco) (.toPath hsaco-file)
                                      (into-array java.nio.file.CopyOption [StandardCopyOption/REPLACE_EXISTING])))
                        bytes-arr)))))]
          (with-open [arena (Arena/ofConfined)]
            (let [^MethodHandle hipModuleLoadData (:hipModuleLoadData hip)
                  ^MethodHandle hipModuleGetFunction (:hipModuleGetFunction hip)
                  image-seg (.allocate arena (alength hsaco-bytes) 64)
                  _ (MemorySegment/copy hsaco-bytes 0 image-seg ValueLayout/JAVA_BYTE 0 (alength hsaco-bytes))
                  mod-slot (.allocate arena ValueLayout/ADDRESS)
                  rc-load (int (.invokeWithArguments hipModuleLoadData (into-array Object [mod-slot image-seg])))
                  _ (check-hip! rc-load "hipModuleLoadData")
                  mod-handle (.get mod-slot ValueLayout/ADDRESS 0)
                  fn-handles (into {}
                                   (map (fn [{:keys [canonical-fn-name]}]
                                          (let [fn-slot (.allocate arena ValueLayout/ADDRESS)
                                                fn-name-seg (.allocateFrom arena ^String canonical-fn-name)
                                                rc-fn (int (.invokeWithArguments hipModuleGetFunction
                                                                                 (into-array Object [fn-slot mod-handle fn-name-seg])))]
                                            (check-hip! rc-fn (str "hipModuleGetFunction(" canonical-fn-name ")"))
                                            [canonical-fn-name (.get fn-slot ValueLayout/ADDRESS 0)]))
                                        unique-rendered-kernels))
                  mod-entry {:target :rocm
                             :hash mod-hash
                             :hsaco-path (.getAbsolutePath hsaco-file)
                             :module-handle mod-handle
                             :fn-handles fn-handles}]
              (.put compiled-module-cache mod-hash mod-entry)
              mod-entry))))))

;; -----------------------------------------------------------------------------
;; 5. UOp Graph Compilation & High-Throughput Execution
;; -----------------------------------------------------------------------------

(defn init-uop-context!
  "Initializes a UOp backend context for `:uop`, `:uop-cpu`, or `:uop-rocm`."
  ([target]
   (init-uop-context! target {}))
  ([target opts]
   (let [hip (when (contains? #{:uop :uop-rocm} target)
               (init-hip-bindings!))
         uop-target (cond
                      (= target :uop-cpu) :cpu
                      (= target :uop-rocm) (if (:available? hip)
                                             :rocm
                                             (throw (ex-info "Requested :uop-rocm backend, but ROCm HIP/HIPRTC is unavailable"
                                                             {:hip hip})))
                      (= target :uop) (if (:available? hip) :rocm :cpu)
                      :else :cpu)
         platform (if (= uop-target :rocm)
                    (str "uop-rocm:" (:arch hip "gfx1100"))
                    "uop-cpu:c99")
         ctx {:backend :uop
              :target target
              :uop-target uop-target
              :platform platform
              :hip hip
              :client nil
              :opts opts
              :destroy-fn destroy-buffer!}]
     ctx)))

(defn compile-uop-executable
  "Compiles a Tensor Logic AST (or pre-lowered UOp graph) into a native `UOpExecutable`
   with deduplicated native kernels and pre-allocated intermediate stage buffers."
  [ctx name-str in-spec ast out-spec]
  (let [uop-target (or (:uop-target ctx) :cpu)
        uop-graph (uop-lower/ast->uop-graph name-str in-spec ast out-spec)
        sched (schedule/schedule-graph uop-graph)
        opt-kernels (mapv #(opt/optimize-kernel % {:target uop-target}) (:kernels sched))
        rendered-kernels (mapv #(render/render-kernel % {:target uop-target}) opt-kernels)
        unique-kernels (->> rendered-kernels
                            (reduce (fn [[seen acc] rk]
                                      (if (contains? seen (:canonical-fn-name rk))
                                        [seen acc]
                                        [(conj seen (:canonical-fn-name rk)) (conj acc rk)]))
                                    [#{} []])
                            second)
        compiled-mod (if (= uop-target :rocm)
                       (compile-rocm-module! ctx unique-kernels)
                       (compile-cpu-module! unique-kernels))
        fn-handles (:fn-handles compiled-mod)
        ;; Pre-allocate intermediate stage buffers once per executable for zero-allocation step execution
        intermediate-buffers (into {}
                                   (keep (fn [[slot buf-info]]
                                           (when (= :intermediate (:role buf-info))
                                             [slot (allocate-device-buffer ctx (:shape buf-info) (:dtype buf-info))]))
                                         (:buffers sched)))
        ;; Pre-allocate native argument pointer tables per kernel in a shared arena
        exec-arena (Arena/ofShared)
        prepared-kernels (mapv (fn [rk]
                                 (let [slots (:buffer-slots rk)
                                       n-args (count slots)
                                       ;; arg-vals holds the actual buffer pointers (void* / hipDeviceptr_t)
                                       arg-vals-seg (.allocate exec-arena ValueLayout/ADDRESS n-args)
                                       ;; kernel-params holds pointers TO each entry in arg-vals-seg (for hipModuleLaunchKernel)
                                       kernel-params-seg (.allocate exec-arena ValueLayout/ADDRESS n-args)]
                                   (dotimes [i n-args]
                                     (.setAtIndex kernel-params-seg ValueLayout/ADDRESS (long i)
                                                  (.asSlice arg-vals-seg (* i 8) 8)))
                                   (assoc rk
                                          :fn-handle (get fn-handles (:canonical-fn-name rk))
                                          :arg-vals-seg arg-vals-seg
                                          :kernel-params-seg kernel-params-seg)))
                               rendered-kernels)]
    {:uop-executable? true
     :name name-str
     :backend :uop
     :uop-target uop-target
     :ctx ctx
     :uop-graph uop-graph
     :schedule sched
     :module compiled-mod
     :prepared-kernels prepared-kernels
     :intermediate-buffers intermediate-buffers
     :exec-arena exec-arena
     :lock (ReentrantLock.)}))

(defn destroy-uop-executable!
  "Frees pre-allocated intermediate stage buffers and parameter pointer segments for `uop-exec`."
  [ctx uop-exec]
  (when (:uop-executable? uop-exec)
    (doseq [[_ buf] (:intermediate-buffers uop-exec)]
      (destroy-buffer! ctx buf))
    (when-let [^Arena a (:exec-arena uop-exec)]
      (when (.isAlive (.scope a))
        (try (.close a) (catch Throwable _ nil)))))
  nil)

(defn execute-uop-executable
  "Executes a compiled `UOpExecutable` given positional `input-buffers` (ordered matching `invars`).
   Returns a map `{out-kw -> UOpBuffer}`."
  [ctx uop-exec input-buffers]
  (let [^ReentrantLock lock (:lock uop-exec)]
    (.lock lock)
    (try
      (let [uop-target (:uop-target uop-exec)
            sched (:schedule uop-exec)
            buffers-meta (:buffers sched)
            output-slots (:output-slots sched)
            inter-bufs (:intermediate-buffers uop-exec)
            ;; 1. Allocate fresh output buffers (owned by caller / active DeviceArena)
            out-bufs (into {}
                           (map (fn [[_out-kw slot]]
                                  (let [b-info (get buffers-meta slot)]
                                    [slot (allocate-device-buffer ctx (:shape b-info) (:dtype b-info))]))
                                output-slots))
            ;; 2. Build slot -> MemorySegment pointer lookup table
            slot->ptr (java.util.HashMap.)]
        (doseq [[slot u-buf] (map-indexed vector input-buffers)]
          (let [dev-b (if (uop-buffer? u-buf)
                        u-buf
                        (let [b-info (get buffers-meta slot)]
                          (host->device ctx u-buf (:shape b-info) (:dtype b-info))))]
            (.put slot->ptr (long slot) (:ptr dev-b))))
        (doseq [[slot b] inter-bufs]
          (.put slot->ptr (long slot) (:ptr b)))
        (doseq [[slot b] out-bufs]
          (.put slot->ptr (long slot) (:ptr b)))

        ;; 3. Dispatch all scheduled kernels sequentially
        (if (= uop-target :rocm)
          (let [hip (or (:hip ctx) (init-hip-bindings!))
                ^MethodHandle launch-mh (:hipModuleLaunchKernel hip)
                ^MethodHandle sync-mh (:hipDeviceSynchronize hip)]
            (doseq [{:keys [fn-handle buffer-slots arg-vals-seg kernel-params-seg global-size local-size]} (:prepared-kernels uop-exec)]
              (dotimes [i (count buffer-slots)]
                (let [slot (long (nth buffer-slots i))
                      ^MemorySegment ptr (.get slot->ptr slot)]
                  (.setAtIndex ^MemorySegment arg-vals-seg ValueLayout/ADDRESS (long i) ptr)))
              (let [[gx gy gz] global-size
                    [lx ly lz] local-size
                    rc (int (.invokeWithArguments launch-mh
                                                  (into-array Object [fn-handle
                                                                      (int gx) (int gy) (int gz)
                                                                      (int lx) (int ly) (int lz)
                                                                      (int 0)
                                                                      MemorySegment/NULL
                                                                      kernel-params-seg
                                                                      MemorySegment/NULL])))]
                (check-hip! rc "hipModuleLaunchKernel")))
            (let [rc-sync (int (.invokeWithArguments sync-mh (into-array Object [])))]
              (check-hip! rc-sync "hipDeviceSynchronize")))
          ;; CPU execution
          (doseq [{:keys [fn-handle buffer-slots arg-vals-seg]} (:prepared-kernels uop-exec)]
            (dotimes [i (count buffer-slots)]
              (let [slot (long (nth buffer-slots i))
                    ^MemorySegment ptr (.get slot->ptr slot)]
                (.setAtIndex ^MemorySegment arg-vals-seg ValueLayout/ADDRESS (long i) ptr)))
            (.invokeWithArguments ^MethodHandle fn-handle (into-array Object [arg-vals-seg]))))

        ;; 4. Return `{out-kw -> UOpBuffer}` map
        (into {}
              (map (fn [[out-kw slot]]
                     [out-kw (get out-bufs slot)])
                   output-slots)))
      (finally
        (.unlock lock)))))
