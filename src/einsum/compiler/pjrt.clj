(ns einsum.compiler.pjrt
  "Project Panama FFM native bindings to OpenXLA PJRT C API (pjrt_c_api.h)."
  (:require [clojure.java.io :as io])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option MemoryLayout MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]))

;; PJRT_Api struct field offsets (64-bit architecture)
(def OFFSET_STRUCT_SIZE "Offset of struct_size in PJRT_Api." 0)
(def OFFSET_EXTENSION_START "Offset of extension_start in PJRT_Api." 8)
(def OFFSET_API_VERSION "Offset of pjrt_api_version in PJRT_Api." 16)
(def OFFSET_ERROR_DESTROY "Offset of PJRT_Error_Destroy in PJRT_Api." 40)
(def OFFSET_ERROR_MESSAGE "Offset of PJRT_Error_Message in PJRT_Api." 48)
(def OFFSET_ERROR_GET_CODE "Offset of PJRT_Error_GetCode in PJRT_Api." 56)
(def OFFSET_PLUGIN_INITIALIZE "Offset of PJRT_Plugin_Initialize in PJRT_Api." 64)
(def OFFSET_PLUGIN_ATTRIBUTES "Offset of PJRT_Plugin_Attributes in PJRT_Api." 72)
(def OFFSET_EVENT_DESTROY "Offset of PJRT_Event_Destroy in PJRT_Api." 80)
(def OFFSET_EVENT_AWAIT "Offset of PJRT_Event_Await in PJRT_Api." 104)
(def OFFSET_CLIENT_CREATE "Offset of PJRT_Client_Create in PJRT_Api." 120)
(def OFFSET_CLIENT_DESTROY "Offset of PJRT_Client_Destroy in PJRT_Api." 128)
(def OFFSET_CLIENT_PLATFORM_NAME "Offset of PJRT_Client_PlatformName in PJRT_Api." 136)
(def OFFSET_CLIENT_PLATFORM_VERSION "Offset of PJRT_Client_PlatformVersion in PJRT_Api." 152)
(def OFFSET_CLIENT_ADDRESSABLE_DEVICES "Offset of PJRT_Client_AddressableDevices in PJRT_Api." 168)
(def OFFSET_CLIENT_COMPILE "Offset of PJRT_Client_Compile in PJRT_Api." 200)
(def OFFSET_CLIENT_BUFFER_FROM_HOST_BUFFER "Offset of PJRT_Client_BufferFromHostBuffer in PJRT_Api." 216)
(def OFFSET_EXECUTABLE_DESTROY "Offset of PJRT_Executable_Destroy in PJRT_Api." 360)
(def OFFSET_EXECUTABLE_SERIALIZE "Offset of PJRT_Executable_Serialize in PJRT_Api." 432)
(def OFFSET_LOADED_EXECUTABLE_DESTROY "Offset of PJRT_LoadedExecutable_Destroy in PJRT_Api." 440)
(def OFFSET_LOADED_EXECUTABLE_GET_EXECUTABLE "Offset of PJRT_LoadedExecutable_GetExecutable in PJRT_Api." 448)
(def OFFSET_LOADED_EXECUTABLE_EXECUTE "Offset of PJRT_LoadedExecutable_Execute in PJRT_Api." 480)
(def OFFSET_EXECUTABLE_DESERIALIZE_AND_LOAD "Offset of PJRT_Executable_DeserializeAndLoad in PJRT_Api." 488)
(def OFFSET_BUFFER_DESTROY "Offset of PJRT_Buffer_Destroy in PJRT_Api." 504)
(def OFFSET_BUFFER_TO_HOST_BUFFER "Offset of PJRT_Buffer_ToHostBuffer in PJRT_Api." 600)

(def NO_OPTIONS "Empty Linker$Option array." (into-array Linker$Option []))
(def NO_LAYOUTS "Empty MemoryLayout array." (into-array MemoryLayout []))

;; Minimal binary Protobuf serialization for CompileOptionsProto:
;; CompileOptionsProto { ExecutableBuildOptionsProto executable_build_options (tag 0x1a) { num_replicas = 1 (tag 0x20), num_partitions = 1 (tag 0x28) } }
(def DEFAULT_COMPILE_OPTIONS_BYTES
  "Default serialized CompileOptionsProto bytes configuring num_replicas=1 and num_partitions=1."
  (byte-array [(byte 0x1a) (byte 4) (byte 0x20) (byte 1) (byte 0x28) (byte 1)]))

(defn- extract-ctx [ctx]
  (cond
    (nil? ctx) {}
    (:api-ptr ctx) ctx
    (:api ctx) (:api ctx)
    :else ctx))

(defn- extract-client [ctx client]
  (or client (:client ctx)))

(def ^:private downcall-cache (atom {}))

(defn- downcall-ptr
  "Creates or retrieves a cached downcall MethodHandle for function pointer at `offset` inside `api-ptr`."
  [^Linker linker ^MemorySegment api-ptr offset return-layout arg-layouts]
  (let [cache-key [api-ptr offset return-layout arg-layouts]]
    (or (get @downcall-cache cache-key)
        (let [^MemorySegment fn-ptr (.get ^MemorySegment api-ptr ValueLayout/ADDRESS (long offset))
              ^FunctionDescriptor fd (if (some? return-layout)
                                       (FunctionDescriptor/of return-layout (into-array MemoryLayout arg-layouts))
                                       (FunctionDescriptor/ofVoid (into-array MemoryLayout arg-layouts)))
              handle (.downcallHandle linker fn-ptr fd NO_OPTIONS)]
          (swap! downcall-cache assoc cache-key handle)
          handle))))

(defn check-error!
  "Checks if `err-ptr` is non-null. If so, extracts error message, frees error, and throws Exception."
  [api-ctx err-ptr]
  (when (and (some? err-ptr) (not= MemorySegment/NULL err-ptr))
    (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
          msg-handle (downcall-ptr linker api-ptr OFFSET_ERROR_MESSAGE nil [ValueLayout/ADDRESS])
          msg-args (.allocate ^Arena arena (long 40))]
      (.set ^MemorySegment msg-args ValueLayout/JAVA_LONG (long 0) (long 40))
      (.set ^MemorySegment msg-args ValueLayout/ADDRESS (long 16) err-ptr)
      (.invokeWithArguments ^MethodHandle msg-handle [msg-args])
      (let [str-ptr (.get ^MemorySegment msg-args ValueLayout/ADDRESS (long 24))
            str-len (.get ^MemorySegment msg-args ValueLayout/JAVA_LONG (long 32))
            str-bytes (.toArray (.reinterpret ^MemorySegment str-ptr str-len) ValueLayout/JAVA_BYTE)
            err-msg (String. ^bytes str-bytes "UTF-8")
            destroy-handle (downcall-ptr linker api-ptr OFFSET_ERROR_DESTROY nil [ValueLayout/ADDRESS])
            destroy-args (.allocate ^Arena arena (long 24))]
        (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
        (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) err-ptr)
        (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])
        (throw (ex-info (str "PJRT Error: " err-msg) {:error-msg err-msg}))))))

(defn- preload-libpython!
  "Pre-loads system libpython library with RTLD_GLOBAL (0x102) if present, resolving Python C API symbols for PyPI PJRT plugins."
  []
  (let [paths ["/usr/lib64/libpython3.12.so"
               "/usr/lib64/libpython3.13.so"
               "/usr/lib64/libpython3.14.so"
               "/usr/lib64/libpython3.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.12.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.11.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.10.so"
               "/usr/lib/x86_64-linux-gnu/libpython3.so"]
        existing (first (filter #(and (.exists (io/file %)) (not (.isDirectory (io/file %)))) paths))]
    (when existing
      (try
        (let [linker (Linker/nativeLinker)
              stdlib (.defaultLookup linker)
              ^MemorySegment dlopen-seg (.get (.find stdlib "dlopen"))
              ^FunctionDescriptor desc (FunctionDescriptor/of ValueLayout/ADDRESS (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
              dlopen-fn (.downcallHandle linker dlopen-seg desc (make-array Linker$Option 0))
              arena (Arena/global)
              py-path (.allocateFrom arena ^String existing)]
          (.invokeWithArguments dlopen-fn [py-path (Integer/valueOf 258)]))
        (catch Exception _ nil)))))

(defn api-version
  "Returns [major minor] version numbers from PJRT_Api struct."
  [api-ctx]
  (let [{:keys [^MemorySegment api-ptr]} (extract-ctx api-ctx)
        major (.get api-ptr ValueLayout/JAVA_INT (long OFFSET_API_VERSION))
        minor (.get api-ptr ValueLayout/JAVA_INT (long (+ OFFSET_API_VERSION 4)))]
    [major minor]))

(defn plugin-attributes
  "Queries the attributes map from the loaded PJRT plugin using PJRT_Plugin_Attributes."
  [api-ctx]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        attr-handle (downcall-ptr linker api-ptr OFFSET_PLUGIN_ATTRIBUTES ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        args (.allocate ^Arena arena (long 32))]
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 32))
    (let [err (.invokeWithArguments ^MethodHandle attr-handle [args])]
      (check-error! api-ctx err)
      (let [attrs-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 16))
            num-attrs (.get ^MemorySegment args ValueLayout/JAVA_LONG (long 24))]
        (if (or (nil? attrs-ptr) (= MemorySegment/NULL attrs-ptr) (zero? num-attrs))
          {}
          (into {}
                (map (fn [i]
                       (let [elem (.asSlice (.reinterpret ^MemorySegment attrs-ptr (* num-attrs 64)) (* i 64) 64)
                             name-ptr (.get ^MemorySegment elem ValueLayout/ADDRESS (long 16))
                             name-len (.get ^MemorySegment elem ValueLayout/JAVA_LONG (long 24))
                             attr-name (String. (.toArray (.reinterpret ^MemorySegment name-ptr name-len) ValueLayout/JAVA_BYTE) "UTF-8")
                             val-type (.get ^MemorySegment elem ValueLayout/JAVA_INT (long 32))
                             attr-val (case val-type
                                        0 (let [str-ptr (.get ^MemorySegment elem ValueLayout/ADDRESS (long 40))
                                                val-len (.get ^MemorySegment elem ValueLayout/JAVA_LONG (long 48))]
                                            (String. (.toArray (.reinterpret ^MemorySegment str-ptr val-len) ValueLayout/JAVA_BYTE) "UTF-8"))
                                        1 (.get ^MemorySegment elem ValueLayout/JAVA_LONG (long 40))
                                        2 (.get ^MemorySegment elem ValueLayout/JAVA_FLOAT (long 40))
                                        3 (.get ^MemorySegment elem ValueLayout/JAVA_DOUBLE (long 40))
                                        4 (not (zero? (.get ^MemorySegment elem ValueLayout/JAVA_BYTE (long 40))))
                                        nil)]
                         [attr-name attr-val]))
                     (range num-attrs))))))))

(defn- ensure-hsaco-cache-dir!
  "Ensures TF_XLA_HSACO_CACHE_DIR environment variable is set to enable OpenXLA HSACO on-disk binary caching."
  []
  (try
    (when (nil? (System/getenv "TF_XLA_HSACO_CACHE_DIR"))
      (let [cache-dir (io/file (System/getProperty "user.home") ".cache" "hsa_cache")
            abs-path (.getAbsolutePath cache-dir)
            linker (Linker/nativeLinker)
            stdlib (.defaultLookup linker)]
        (.mkdirs cache-dir)
        (System/setProperty "TF_XLA_HSACO_CACHE_DIR" abs-path)
        (when-let [setenv-seg (.orElse nil (.find stdlib "setenv"))]
          (let [fd (FunctionDescriptor/of ValueLayout/JAVA_INT
                                          (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
                handle (.downcallHandle linker setenv-seg fd NO_OPTIONS)]
            (with-open [arena (Arena/ofConfined)]
              (let [k-seg (.allocateFrom arena "TF_XLA_HSACO_CACHE_DIR")
                    v-seg (.allocateFrom arena ^String abs-path)]
                (.invokeWithArguments handle [k-seg v-seg (Integer/valueOf 0)])))))))
    (catch Exception _ nil)))

(defn companion-lib?
  "Returns true if file `f` (File object or filename string) is a companion library that should be preloaded,
   excluding main PJRT plugin shared objects (e.g. libpjrt_*, xla_cpu_pjrt.so, pjrt_plugin_xpu.so) and rocprofiler."
  [f]
  (let [name (if (string? f) f (.getName ^java.io.File f))]
    (and (boolean (re-find #"\.so(\.\d+)*$" name))
         (not (boolean (re-find #"rocprofiler" name)))
         (not (boolean (re-find #"^(xla_cpu|pjrt_plugin|libpjrt)" name))))))

(defn load-plugin!
  "Loads the PJRT shared object from `lib-path` and initializes the PJRT plugin."
  [lib-path]
  (preload-libpython!)
  (ensure-hsaco-cache-dir!)
  (let [arena (Arena/global)
        abs-file (io/file lib-path)
        abs-path (.toAbsolutePath (.toPath abs-file))
        ;; Pre-load companion libraries from bin/lib/ if present
        bin-lib-dir (io/file (.getParent abs-file) "lib")
        _ (when (.exists bin-lib-dir)
            (let [linker (Linker/nativeLinker)
                  stdlib (.defaultLookup linker)
                  ^MemorySegment dlopen-seg (.get (.find stdlib "dlopen"))
                  ^FunctionDescriptor desc (FunctionDescriptor/of ValueLayout/ADDRESS (into-array MemoryLayout [ValueLayout/ADDRESS ValueLayout/JAVA_INT]))
                  dlopen-fn (.downcallHandle linker dlopen-seg desc (make-array Linker$Option 0))]
              (dotimes [_ 5]
                (doseq [f (file-seq bin-lib-dir)]
                  (when (and (.isFile f) (companion-lib? f))
                    (try
                      (let [p-seg (.allocateFrom arena (.getAbsolutePath f))]
                        (.invokeWithArguments dlopen-fn [p-seg (Integer/valueOf 258)]))
                      (catch Exception _ nil)))))))
        lookup (SymbolLookup/libraryLookup abs-path arena)
        ^MemorySegment get-api (.orElseThrow (.find lookup "GetPjrtApi"))
        linker (Linker/nativeLinker)
        ^FunctionDescriptor fd (FunctionDescriptor/of ValueLayout/ADDRESS NO_LAYOUTS)
        ^MethodHandle handle (.downcallHandle linker get-api fd NO_OPTIONS)
        raw-api (.invokeWithArguments handle [])
        ^MemorySegment api-ptr (.reinterpret ^MemorySegment raw-api 984)
        api-ctx {:api-ptr api-ptr :linker linker :arena arena}]

    ;; Call PJRT_Plugin_Initialize
    (let [init-handle (downcall-ptr linker api-ptr OFFSET_PLUGIN_INITIALIZE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          init-args (.allocate arena (long 16))]
      (.set ^MemorySegment init-args ValueLayout/JAVA_LONG (long 0) (long 16))
      (let [err (.invokeWithArguments ^MethodHandle init-handle [init-args])]
        (check-error! api-ctx err)))

    (let [[major minor] (api-version api-ctx)]
      (when-not (Boolean/getBoolean "clj-xla.quiet")
        (println (format "PJRT Plugin loaded [%s] (API Version: %d.%d)" lib-path major minor))))
    api-ctx))

(defn- build-named-values [^Arena arena opts]
  (if (empty? opts)
    [MemorySegment/NULL 0]
    (let [num-opts (count opts)
          named-vals (.allocate arena (long (* 64 num-opts)) (long 8))]
      (doseq [[idx [k v]] (map-indexed vector opts)]
        (let [n-str (name k)
              n-bytes (.getBytes ^String n-str "UTF-8")
              n-seg (.allocateFrom arena (String. ^bytes n-bytes "UTF-8"))
              elem (.asSlice named-vals (* idx 64) 64)]
          (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 0) (long 64))
          (.set ^MemorySegment elem ValueLayout/ADDRESS (long 16) n-seg)
          (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 24) (long (alength n-bytes)))
          (cond
            (string? v)
            (let [v-seg (.allocateFrom arena (String. ^bytes (.getBytes ^String v "UTF-8")))]
              (.set ^MemorySegment elem ValueLayout/JAVA_INT (long 32) (int 0))
              (.set ^MemorySegment elem ValueLayout/ADDRESS (long 40) v-seg)
              (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 48) (long (count v))))

            (integer? v)
            (do
              (.set ^MemorySegment elem ValueLayout/JAVA_INT (long 32) (int 1))
              (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 40) (long v))
              (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 48) (long 1)))

            (float? v)
            (do
              (.set ^MemorySegment elem ValueLayout/JAVA_INT (long 32) (int 3))
              (.set ^MemorySegment elem ValueLayout/JAVA_FLOAT (long 40) (float v))
              (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 48) (long 1)))

            (boolean? v)
            (do
              (.set ^MemorySegment elem ValueLayout/JAVA_INT (long 32) (int 4))
              (.set ^MemorySegment elem ValueLayout/JAVA_BYTE (long 40) (byte (if v 1 0)))
              (.set ^MemorySegment elem ValueLayout/JAVA_LONG (long 48) (long 1))))))
      [named-vals num-opts])))

(defn create-client
  "Creates a PJRT_Client for the loaded plugin, optionally accepting a `create-options` map."
  ([api-ctx] (create-client api-ctx {}))
  ([api-ctx opts]
   (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
         create-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_CREATE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
         create-args (.allocate ^Arena arena (long 88))
         [opts-seg num-opts] (build-named-values arena opts)]
     (.set ^MemorySegment create-args ValueLayout/JAVA_LONG (long 0) (long 88))
     (when (and (some? opts-seg) (not= MemorySegment/NULL opts-seg))
       (.set ^MemorySegment create-args ValueLayout/ADDRESS (long 16) opts-seg)
       (.set ^MemorySegment create-args ValueLayout/JAVA_LONG (long 24) (long num-opts)))
     (let [err (.invokeWithArguments ^MethodHandle create-handle [create-args])]
       (check-error! api-ctx err)
       (.get ^MemorySegment create-args ValueLayout/ADDRESS (long 64))))))

(defn platform-name
  "Queries the platform name string from `client` (e.g. 'cpu', 'cuda')."
  [api-ctx client]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        pname-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_PLATFORM_NAME ValueLayout/ADDRESS [ValueLayout/ADDRESS])
        pname-args (.allocate ^Arena arena (long 40))]
    (.set ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 0) (long 40))
    (.set ^MemorySegment pname-args ValueLayout/ADDRESS (long 16) cli)
    (let [err (.invokeWithArguments ^MethodHandle pname-handle [pname-args])]
      (check-error! api-ctx err)
      (let [str-ptr (.get ^MemorySegment pname-args ValueLayout/ADDRESS (long 24))
            str-len (.get ^MemorySegment pname-args ValueLayout/JAVA_LONG (long 32))
            str-bytes (.toArray (.reinterpret ^MemorySegment str-ptr str-len) ValueLayout/JAVA_BYTE)]
        (String. ^bytes str-bytes "UTF-8")))))

(defn addressable-devices
  "Returns array of addressable PJRT_Device handles for `client`."
  [api-ctx client]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        args (.allocate ^Arena arena (long 40))]
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 40))
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) cli)
    (let [handle (downcall-ptr linker api-ptr OFFSET_CLIENT_ADDRESSABLE_DEVICES ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          err (.invokeWithArguments ^MethodHandle handle [args])]
      (check-error! api-ctx err)
      (let [devs-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 24))
            num-devs (.get ^MemorySegment args ValueLayout/JAVA_LONG (long 32))]
        (mapv (fn [i]
                (.getAtIndex (.reinterpret ^MemorySegment devs-ptr (* num-devs 8)) ValueLayout/ADDRESS (long i)))
              (range num-devs))))))

(defn compile-mlir
  "Compiles `mlir-text` into a PjRtLoadedExecutable native handle."
  [api-ctx client mlir-text]
  (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        code-bytes (.getBytes ^String mlir-text "UTF-8")
        code-seg (.allocateFrom ^Arena arena (String. ^bytes code-bytes "UTF-8"))
        format-seg (.allocateFrom ^Arena arena "mlir")
        opts-seg (.allocateFrom ^Arena arena ValueLayout/JAVA_BYTE DEFAULT_COMPILE_OPTIONS_BYTES)

        ;; PJRT_Program (size 48)
        program (.allocate ^Arena arena (long 48))]
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 0) (long 48))
    (.set ^MemorySegment program ValueLayout/ADDRESS (long 16) code-seg)
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 24) (long (alength code-bytes)))
    (.set ^MemorySegment program ValueLayout/ADDRESS (long 32) format-seg)
    (.set ^MemorySegment program ValueLayout/JAVA_LONG (long 40) (long 4))

    ;; PJRT_Client_Compile_Args (size 56)
    (let [compile-args (.allocate ^Arena arena (long 56))]
      (.set ^MemorySegment compile-args ValueLayout/JAVA_LONG (long 0) (long 56))
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 16) cli)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 24) program)
      (.set ^MemorySegment compile-args ValueLayout/ADDRESS (long 32) opts-seg)
      (.set ^MemorySegment compile-args ValueLayout/JAVA_LONG (long 40) (long (alength DEFAULT_COMPILE_OPTIONS_BYTES)))

      ;; PJRT_Client_Compile at offset 200
      (let [compile-handle (downcall-ptr linker api-ptr OFFSET_CLIENT_COMPILE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            err (.invokeWithArguments ^MethodHandle compile-handle [compile-args])]
        (check-error! api-ctx err)
        (.get ^MemorySegment compile-args ValueLayout/ADDRESS (long 48))))))

(defn loaded-executable->executable
  "Extracts underlying PJRT_Executable native handle from `loaded-exec-ptr`."
  [api-ctx loaded-exec-ptr]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        ^MemorySegment exec-seg (cond
                                  (instance? MemorySegment loaded-exec-ptr) loaded-exec-ptr
                                  :else loaded-exec-ptr)]
    (with-open [arena (Arena/ofConfined)]
      (let [get-exec-fn (downcall-ptr linker api-ptr OFFSET_LOADED_EXECUTABLE_GET_EXECUTABLE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            args (.allocate arena (long 32))]
        (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 32))
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) exec-seg)
        (let [err (.invokeWithArguments ^MethodHandle get-exec-fn [args])]
          (check-error! api-ctx err))
        (.get ^MemorySegment args ValueLayout/ADDRESS (long 24))))))

(defn destroy-executable!
  "Destroys un-loaded PJRT_Executable native handle `exec-ptr`."
  [api-ctx exec-ptr]
  (when (and (some? exec-ptr) (not= MemorySegment/NULL exec-ptr))
    (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)]
      (with-open [arena (Arena/ofConfined)]
        (let [dest-fn (downcall-ptr linker api-ptr OFFSET_EXECUTABLE_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
              dest-args (.allocate arena (long 24))]
          (.set ^MemorySegment dest-args ValueLayout/JAVA_LONG (long 0) (long 24))
          (.set ^MemorySegment dest-args ValueLayout/ADDRESS (long 16) exec-ptr)
          (let [err (.invokeWithArguments ^MethodHandle dest-fn [dest-args])]
            (check-error! api-ctx err)))))))

(defn destroy-loaded-executable!
  "Destroys loaded PJRT_LoadedExecutable native handle `loaded-exec-ptr`."
  [api-ctx loaded-exec-ptr]
  (when (and (some? loaded-exec-ptr) (not= MemorySegment/NULL loaded-exec-ptr))
    (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)]
      (with-open [arena (Arena/ofConfined)]
        (let [dest-fn (downcall-ptr linker api-ptr OFFSET_LOADED_EXECUTABLE_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
              dest-args (.allocate arena (long 24))]
          (.set ^MemorySegment dest-args ValueLayout/JAVA_LONG (long 0) (long 24))
          (.set ^MemorySegment dest-args ValueLayout/ADDRESS (long 16) loaded-exec-ptr)
          (let [err (.invokeWithArguments ^MethodHandle dest-fn [dest-args])]
            (check-error! api-ctx err)))))))

(defn serialize-executable
  "Serializes a compiled PJRT executable handle (`loaded-exec-ptr` or `exec-ptr`) into a byte array."
  [api-ctx exec-or-loaded-exec]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        ^MemorySegment target-seg (cond
                                    (instance? MemorySegment exec-or-loaded-exec) exec-or-loaded-exec
                                    :else exec-or-loaded-exec)
        exec-ptr (try
                   (loaded-executable->executable api-ctx target-seg)
                   (catch Exception _
                     target-seg))]
    (with-open [arena (Arena/ofConfined)]
      (let [ser-fn (downcall-ptr linker api-ptr OFFSET_EXECUTABLE_SERIALIZE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
            args (.allocate arena (long 56))]
        (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 56))
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) exec-ptr)
        (let [err (.invokeWithArguments ^MethodHandle ser-fn [args])]
          (check-error! api-ctx err))
        (let [bytes-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 24))
              bytes-size (.get ^MemorySegment args ValueLayout/JAVA_LONG (long 32))
              ser-exec-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 40))
              deleter-fn (.get ^MemorySegment args ValueLayout/ADDRESS (long 48))
              byte-arr (.toArray (.reinterpret ^MemorySegment bytes-ptr bytes-size) ValueLayout/JAVA_BYTE)]
          (when (and (some? deleter-fn) (not= MemorySegment/NULL deleter-fn))
            (let [del-desc (FunctionDescriptor/ofVoid (into-array MemoryLayout [ValueLayout/ADDRESS]))
                  del-handle (.downcallHandle linker deleter-fn del-desc NO_OPTIONS)]
              (.invokeWithArguments del-handle [ser-exec-ptr])))
          byte-arr)))))

(defn deserialize-and-load
  "Deserializes previously serialized executable `byte-arr` into a PJRT_LoadedExecutable handle."
  ([api-ctx byte-arr]
   (deserialize-and-load api-ctx nil byte-arr))
  ([api-ctx client byte-arr]
   (let [{:keys [api-ptr linker arena]} (extract-ctx api-ctx)
         cli (extract-client api-ctx client)
         ^bytes b-arr (cond
                        (bytes? byte-arr) byte-arr
                        (instance? MemorySegment byte-arr) (.toArray ^MemorySegment byte-arr ValueLayout/JAVA_BYTE)
                        :else (byte-array byte-arr))
         deser-fn (downcall-ptr linker api-ptr OFFSET_EXECUTABLE_DESERIALIZE_AND_LOAD ValueLayout/ADDRESS [ValueLayout/ADDRESS])
         deser-args (.allocate ^Arena arena (long 72))
         buf-seg (.allocateFrom ^Arena arena ValueLayout/JAVA_BYTE b-arr)]
     (.set ^MemorySegment deser-args ValueLayout/JAVA_LONG (long 0) (long 72))
     (.set ^MemorySegment deser-args ValueLayout/ADDRESS (long 16) cli)
     (.set ^MemorySegment deser-args ValueLayout/ADDRESS (long 24) buf-seg)
     (.set ^MemorySegment deser-args ValueLayout/JAVA_LONG (long 32) (long (alength b-arr)))
     (let [err (.invokeWithArguments ^MethodHandle deser-fn [deser-args])]
       (check-error! api-ctx err))
     (.get ^MemorySegment deser-args ValueLayout/ADDRESS (long 40)))))

(defn buffer-from-host-buffer
  "Transfers a host memory buffer (float-array or int-array) to a device PJRT_Buffer."
  [api-ctx client host-data shape dtype-enum]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        cli (extract-client api-ctx client)
        num-dims (count shape)
        devs (addressable-devices api-ctx cli)
        dev (first devs)]
    (with-open [arena (Arena/ofConfined)]
      (let [dims-seg (.allocate arena ValueLayout/JAVA_LONG (long num-dims))]
        (dotimes [i num-dims]
          (.setAtIndex ^MemorySegment dims-seg ValueLayout/JAVA_LONG (long i) (long (nth shape i))))
        (let [data-seg (cond
                         (instance? MemorySegment host-data)
                         (if (.isNative ^MemorySegment host-data)
                           host-data
                           (.allocateFrom arena ValueLayout/JAVA_BYTE ^MemorySegment host-data))

                         (= (int dtype-enum) 1)
                         (cond
                           (instance? (Class/forName "[Z") host-data)
                           (let [^booleans za ^booleans host-data
                                 num-elems (alength za)
                                 seg (.allocate arena (long num-elems) (long 64))]
                             (dotimes [i num-elems]
                               (.setAtIndex ^MemorySegment seg ValueLayout/JAVA_BYTE (long i) (if (aget za i) (byte 1) (byte 0))))
                             seg)
                           (instance? (Class/forName "[B") host-data)
                           (let [^bytes ba ^bytes host-data
                                 num-bytes (alength ba)
                                 seg (.allocate arena (long num-bytes) (long 64))]
                             (MemorySegment/copy ba 0 seg ValueLayout/JAVA_BYTE (long 0) num-bytes)
                             seg)
                           :else
                           (let [seg (.allocate arena 1 (long 64))]
                             (.set ^MemorySegment seg ValueLayout/JAVA_BYTE (long 0) (if host-data (byte 1) (byte 0)))
                             seg))

                         (= (int dtype-enum) 4)
                         (let [^ints ia (if (instance? (Class/forName "[I") host-data)
                                          ^ints host-data
                                          (int-array (map int (flatten host-data))))]
                           (.allocateFrom arena ValueLayout/JAVA_INT ia))

                         (instance? (Class/forName "[B") host-data)
                         (let [^bytes ba ^bytes host-data
                               num-bytes (alength ba)
                               seg (.allocate arena (long num-bytes) (long 64))]
                           (MemorySegment/copy ba 0 seg ValueLayout/JAVA_BYTE (long 0) num-bytes)
                           seg)

                         (instance? (Class/forName "[S") host-data)
                         (let [^shorts sa ^shorts host-data
                               num-shorts (alength sa)
                               byte-size (* (long num-shorts) 2)
                               seg (.allocate arena byte-size (long 64))]
                           (MemorySegment/copy sa 0 seg ValueLayout/JAVA_SHORT (long 0) num-shorts)
                           seg)

                         :else
                         (let [^floats fa (if (instance? (Class/forName "[F") host-data)
                                            ^floats host-data
                                            (float-array (map float (flatten host-data))))
                               num-floats (alength fa)
                               byte-size (* (long num-floats) 4)
                               seg (.allocate arena byte-size (long 64))]
                           (MemorySegment/copy fa 0 seg ValueLayout/JAVA_FLOAT (long 0) num-floats)
                           seg))
              args (.allocate arena (long 120))]
          (.fill args (byte 0))
          (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 120))
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) cli)
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) data-seg)
          (.set ^MemorySegment args ValueLayout/JAVA_INT (long 32) (int dtype-enum))
          (.set ^MemorySegment args ValueLayout/ADDRESS (long 40) dims-seg)
          (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 48) (long num-dims))
          ;; Compute row-major byte strides for dense layout.
          ;; ROCm PJRT requires byte_strides when num_dims > 0.
          (let [elem-bytes (case (int dtype-enum) 1 1, 2 1, 4 4, 11 4, 10 2, 13 2, 4)
                strides-seg (when (pos? num-dims)
                              (let [seg (.allocate arena ValueLayout/JAVA_LONG (long num-dims))]
                                (loop [i (dec num-dims) stride (long elem-bytes)]
                                  (when (>= i 0)
                                    (.setAtIndex ^MemorySegment seg ValueLayout/JAVA_LONG (long i) stride)
                                    (recur (dec i) (* stride (long (nth shape i))))))
                                seg))]
            (when strides-seg
              (.set ^MemorySegment args ValueLayout/ADDRESS (long 56) strides-seg)
              (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 64) (long num-dims)))
            ;; host_buffer_semantics = kImmutableUntilTransferCompletes (1) at offset 72.
            ;; Ensures GPU backends (ROCm/CUDA) complete async DMA before the host buffer is freed.
            ;; Default 0 (kImmutableOnlyDuringCall) risks SEGFAULT on async GPU transfers.
            (.set ^MemorySegment args ValueLayout/JAVA_INT (long 72) (int 1))
            (when dev
              (.set ^MemorySegment args ValueLayout/ADDRESS (long 80) dev))
            (let [handle (downcall-ptr linker api-ptr OFFSET_CLIENT_BUFFER_FROM_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                  err (.invokeWithArguments ^MethodHandle handle [args])]
              (check-error! api-ctx err)
              (let [done-event (.get ^MemorySegment args ValueLayout/ADDRESS (long 104))
                    buf (.get ^MemorySegment args ValueLayout/ADDRESS (long 112))]
                (when (and (some? done-event) (not= MemorySegment/NULL done-event))
                  (let [await-args (.allocate arena (long 24))]
                    (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
                    (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) done-event)
                    (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                          await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
                      (check-error! api-ctx await-err)))
                  (let [destroy-args (.allocate arena (long 24))]
                    (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
                    (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) done-event)
                    (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                          _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])])))
                buf))))))))

(defn destroy-buffer!
  "Frees native device PJRT_Buffer `buffer-handle`."
  [api-ctx buffer-handle]
  (when (and (some? buffer-handle) (not= MemorySegment/NULL buffer-handle))
    (try
      (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
            ^MemorySegment buf-seg (cond
                                     (instance? MemorySegment buffer-handle) buffer-handle
                                     :else buffer-handle)
            arena (Arena/ofAuto)
            args (.allocate arena (long 24))]
        (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 24))
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buf-seg)
        (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
              err (.invokeWithArguments ^MethodHandle handle [args])]
          (check-error! api-ctx err)))
      (catch Exception _ nil))))

(defn execute-executable
  "Executes compiled PjRtLoadedExecutable native handle `exec-handle` with device `PjRtBuffer` handles `input-buffers`."
  ([api-ctx exec-handle input-buffers]
   (execute-executable api-ctx exec-handle input-buffers 1))
  ([api-ctx exec-handle input-buffers num-outputs]
   (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
         num-args (count input-buffers)
         num-outs (max 1 (long (or num-outputs 1)))]
     (with-open [arena (Arena/ofConfined)]
       (let [arg-ptrs (.allocate arena ValueLayout/ADDRESS (long num-args))]
         (dotimes [i num-args]
           (let [buf (nth input-buffers i)
                 ^MemorySegment buf-seg (cond
                                          (instance? MemorySegment buf) buf
                                          :else (throw (ex-info "Invalid non-MemorySegment input buffer" {:index i :buf buf})))]
             (when (or (nil? buf-seg) (= MemorySegment/NULL buf-seg))
               (throw (ex-info "NULL MemorySegment input buffer" {:index i})))
             (.setAtIndex ^MemorySegment arg-ptrs ValueLayout/ADDRESS (long i) buf-seg)))
         (let [arg-lists (.allocate arena ValueLayout/ADDRESS (long 1))
               _ (.setAtIndex ^MemorySegment arg-lists ValueLayout/ADDRESS (long 0) arg-ptrs)
               out-ptrs (.allocate arena ValueLayout/ADDRESS (long num-outs))
               out-lists (.allocate arena ValueLayout/ADDRESS (long 1))
               _ (.setAtIndex ^MemorySegment out-lists ValueLayout/ADDRESS (long 0) out-ptrs)
               events-ptrs (.allocate arena ValueLayout/ADDRESS (long 1))
               non-donatable (.allocate arena ValueLayout/JAVA_LONG (long num-args))
               _ (dotimes [i num-args]
                   (.setAtIndex ^MemorySegment non-donatable ValueLayout/JAVA_LONG (long i) (long i)))
               opts (.allocate arena (long 144))
               _ (.fill opts (byte 0))
               ;; struct_size = 112: matches the PJRT_ExecuteOptions size expected by the ROCm
               ;; plugin (API version 0.112). Must be at least 112 or the plugin rejects it.
               ;; The original 144 was too large and caused SEGFAULT by exposing uninitialized
               ;; fields that the plugin dereferenced.
               _ (.set ^MemorySegment opts ValueLayout/JAVA_LONG (long 0) (long 112))
               _ (.set ^MemorySegment opts ValueLayout/ADDRESS (long 56) non-donatable)
               _ (.set ^MemorySegment opts ValueLayout/JAVA_LONG (long 64) (long num-args))
               args (.allocate arena (long 80))]
           (.fill args (byte 0))
           (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 80))
           (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) exec-handle)
           (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) opts)
           (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) arg-lists)
           (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long 1))
           (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 48) (long num-args))
           (.set ^MemorySegment args ValueLayout/ADDRESS (long 56) out-lists)
           ;; device_complete_events at offset 64: GPU backends populate this to signal
           ;; when execution completes. We must await it before reading output buffers.
           (.set ^MemorySegment args ValueLayout/ADDRESS (long 64) events-ptrs)
           (.flush System/out)
           (.flush System/err)
           (let [handle (downcall-ptr linker api-ptr OFFSET_LOADED_EXECUTABLE_EXECUTE ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                 err (.invokeWithArguments ^MethodHandle handle [args])]
             (check-error! api-ctx err)
             ;; Await device completion event if the runtime returned one
             (let [event-ptr (.getAtIndex ^MemorySegment events-ptrs ValueLayout/ADDRESS (long 0))]
               (when (and (some? event-ptr) (not= MemorySegment/NULL event-ptr))
                 (let [await-args (.allocate arena (long 24))]
                   (.fill await-args (byte 0))
                   (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
                   (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) event-ptr)
                   (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                         await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
                     (check-error! api-ctx await-err)))
                 (let [destroy-args (.allocate arena (long 24))]
                   (.fill destroy-args (byte 0))
                   (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
                   (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) event-ptr)
                   (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                         _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])]))))
             (if (= num-outs 1)
               (.get ^MemorySegment out-ptrs ValueLayout/ADDRESS (long 0))
               (mapv (fn [i] (.getAtIndex ^MemorySegment out-ptrs ValueLayout/ADDRESS (long i))) (range num-outs))))))))))

(defn buffer-to-host-buffer
  "Copies device PJRT_Buffer `buffer-handle` to host float array, awaiting asynchronous completion.
   Supports optional `dtype` parameter (:f32 or :bf16), converting bfloat16 to float32 on transfer."
  ([api-ctx buffer-handle num-floats]
   (buffer-to-host-buffer api-ctx buffer-handle num-floats :f32))
  ([api-ctx buffer-handle num-elements dtype]
   (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
         ^MemorySegment buf-seg (cond
                                  (instance? MemorySegment buffer-handle) buffer-handle
                                  :else buffer-handle)
         num-elements (long num-elements)
         elem-bytes (if (= dtype :bf16) 2 4)
         byte-size (* num-elements elem-bytes)]
     (with-open [arena (Arena/ofConfined)]
       (let [dst-seg (.allocate arena (long byte-size) (long 64))
             args (.allocate arena (long 56))]
         (.fill args (byte 0))
         (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 56))
         (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buf-seg)
         (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) MemorySegment/NULL)
         (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) dst-seg)
         (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long byte-size))
         (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_TO_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
               err (.invokeWithArguments ^MethodHandle handle [args])]
           (check-error! api-ctx err)
           (let [event-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 48))]
             (when (and (some? event-ptr) (not= MemorySegment/NULL event-ptr))
               (let [await-args (.allocate arena (long 24))]
                 (.fill await-args (byte 0))
                 (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
                 (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) event-ptr)
                 (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                       await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
                   (check-error! api-ctx await-err)))
               (let [destroy-args (.allocate arena (long 24))]
                 (.fill destroy-args (byte 0))
                 (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
                 (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) event-ptr)
                 (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                       _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])])))))
         (if (= dtype :bf16)
           (let [dst-floats (float-array num-elements)]
             (dotimes [i num-elements]
               (let [s (.getAtIndex dst-seg ValueLayout/JAVA_SHORT (long i))
                     raw-int (unchecked-int (bit-shift-left (bit-and (int s) 0xffff) 16))]
                 (aset dst-floats i (Float/intBitsToFloat raw-int))))
             dst-floats)
           (.toArray dst-seg ValueLayout/JAVA_FLOAT)))))))

(defn create-host-float-buffer-transfer-context
  "Pre-allocates off-heap MemorySegments for zero-allocation host float transfers."
  [num-floats]
  (let [arena (Arena/global)
        num-floats (long num-floats)
        byte-size (* num-floats 4)
        dst-seg (.allocate arena ValueLayout/JAVA_FLOAT num-floats)
        args (.allocate arena (long 56))]
    (.fill args (byte 0))
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 56))
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) MemorySegment/NULL) ;; host_layout
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) dst-seg)            ;; dst
    (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long byte-size))  ;; dst_size
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 48) MemorySegment/NULL) ;; event out
    {:dst-seg dst-seg :args args :num-floats num-floats}))

(defn copy-buffer-to-float-array!
  "Copies device PJRT_Buffer `buffer-handle` into pre-allocated `^floats dst-arr` using pre-allocated `transfer-ctx`."
  [api-ctx buffer-handle transfer-ctx ^floats dst-arr]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        {:keys [^MemorySegment dst-seg ^MemorySegment args num-floats]} transfer-ctx
        ^MemorySegment buf-seg (cond
                                 (instance? MemorySegment buffer-handle) buffer-handle
                                 :else buffer-handle)]
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buf-seg)
    (.set ^MemorySegment args ValueLayout/ADDRESS (long 48) MemorySegment/NULL)
    (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_TO_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
          err (.invokeWithArguments ^MethodHandle handle [args])]
      (check-error! api-ctx err))
    (let [arena (Arena/ofAuto)
          evt (.get ^MemorySegment args ValueLayout/ADDRESS (long 48))]
      (when (and (some? evt) (not= MemorySegment/NULL evt))
        (let [await-args (.allocate arena (long 24))]
          (.fill await-args (byte 0))
          (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
          (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) evt)
          (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
            (check-error! api-ctx await-err)))
        (let [destroy-args (.allocate arena (long 24))]
          (.fill destroy-args (byte 0))
          (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
          (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) evt)
          (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])]))))
    (MemorySegment/copy dst-seg ValueLayout/JAVA_FLOAT (long 0) dst-arr 0 (int num-floats))
    dst-arr))

(defn buffer-to-host-float-array!
  "Copies device PJRT_Buffer `buffer-handle` into pre-allocated host float array `dst-arr`."
  [api-ctx buffer-handle ^floats dst-arr]
  (let [ctx (create-host-float-buffer-transfer-context (alength dst-arr))]
    (copy-buffer-to-float-array! api-ctx buffer-handle ctx dst-arr)))

(defn buffer-to-host-int-buffer
  "Copies device PJRT_Buffer `buffer-handle` to host int array, awaiting asynchronous completion."
  [api-ctx buffer-handle num-ints]
  (let [{:keys [api-ptr linker]} (extract-ctx api-ctx)
        ^MemorySegment buf-seg (cond
                                 (instance? MemorySegment buffer-handle) buffer-handle
                                 :else buffer-handle)
        num-ints (long num-ints)
        byte-size (* num-ints 4)]
    (with-open [arena (Arena/ofConfined)]
      (let [dst-seg (.allocate arena ValueLayout/JAVA_INT num-ints)
            args (.allocate arena (long 56))]
        (.fill args (byte 0))
        (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 0) (long 56))
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 16) buf-seg)
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 24) MemorySegment/NULL)
        (.set ^MemorySegment args ValueLayout/ADDRESS (long 32) dst-seg)
        (.set ^MemorySegment args ValueLayout/JAVA_LONG (long 40) (long byte-size))
        (let [handle (downcall-ptr linker api-ptr OFFSET_BUFFER_TO_HOST_BUFFER ValueLayout/ADDRESS [ValueLayout/ADDRESS])
              err (.invokeWithArguments ^MethodHandle handle [args])]
          (check-error! api-ctx err)
          (let [event-ptr (.get ^MemorySegment args ValueLayout/ADDRESS (long 48))]
            (when (and (some? event-ptr) (not= MemorySegment/NULL event-ptr))
              (let [await-args (.allocate arena (long 24))]
                (.fill await-args (byte 0))
                (.set ^MemorySegment await-args ValueLayout/JAVA_LONG (long 0) (long 24))
                (.set ^MemorySegment await-args ValueLayout/ADDRESS (long 16) event-ptr)
                (let [await-handle (downcall-ptr linker api-ptr OFFSET_EVENT_AWAIT ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                      await-err (.invokeWithArguments ^MethodHandle await-handle [await-args])]
                  (check-error! api-ctx await-err)))
              (let [destroy-args (.allocate arena (long 24))]
                (.fill destroy-args (byte 0))
                (.set ^MemorySegment destroy-args ValueLayout/JAVA_LONG (long 0) (long 24))
                (.set ^MemorySegment destroy-args ValueLayout/ADDRESS (long 16) event-ptr)
                (let [destroy-handle (downcall-ptr linker api-ptr OFFSET_EVENT_DESTROY ValueLayout/ADDRESS [ValueLayout/ADDRESS])
                      _ (.invokeWithArguments ^MethodHandle destroy-handle [destroy-args])])))))
        (.toArray dst-seg ValueLayout/JAVA_INT)))))
