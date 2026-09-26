(ns einsum.compiler.pjrt.custom-call
  "Native registration of OpenXLA PJRT GPU Custom Call handlers via Project Panama FFM."
  (:require [clojure.java.io :as io])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandle]))

(defonce ^:private registered-targets (atom #{}))
(defonce ^:private kernel-library-arena (atom nil))

(def ^:private KERNEL_LIB_PATH "bin/libclj_einsum_kernels.so")

(defn- find-gpu-custom-call-register-fn
  "Traverses the PJRT_Extension_Base linked list from `api-ptr` at offset 8
   to locate PJRT_Extension_Type_Gpu_Custom_Call (type 0) and returns its registration fn pointer."
  [^MemorySegment api-ptr]
  (when (and api-ptr (not= MemorySegment/NULL api-ptr))
    (loop [curr (.get api-ptr ValueLayout/ADDRESS 8)]
      (when (and curr (not= MemorySegment/NULL curr))
        (let [seg (.reinterpret curr 64)
              ext-type (.get seg ValueLayout/JAVA_INT 8)
              next-ptr (.get seg ValueLayout/ADDRESS 16)]
          (if (zero? ext-type)
            (.get seg ValueLayout/ADDRESS 24)
            (recur next-ptr)))))))

(defn register-custom-call-target!
  "Registers an in-process native GPU custom call function with OpenXLA PJRT.
   `target-name`: String name referenced by StableHLO `call_target_name`
   `fn-ptr`: MemorySegment function pointer to the native GPU kernel entry point
   `api-version`: int API version (1 = API_VERSION_ORIGINAL, 0 in custom call args)
   Returns true on successful registration."
  [ctx target-name ^MemorySegment fn-ptr & [api-version]]
  (let [api-ptr (:api-ptr ctx)]
    (when-not api-ptr
      (throw (ex-info "Cannot register custom call: invalid or missing PJRT context api-ptr" {:target target-name})))
    (if (contains? @registered-targets target-name)
      true
      (if-let [reg-fn-ptr (find-gpu-custom-call-register-fn api-ptr)]
        (let [arena (Arena/ofConfined)
              linker (Linker/nativeLinker)
              desc (FunctionDescriptor/ofVoid (into-array [ValueLayout/ADDRESS]))
              ^MethodHandle mh (.downcallHandle linker reg-fn-ptr desc (into-array java.lang.foreign.Linker$Option []))
              name-bytes (.getBytes ^String target-name "UTF-8")
              name-len (long (alength name-bytes))
              fn-name-seg (.allocateFrom arena target-name)
              args-seg (.allocate arena (long 64))]
          ;; PJRT_Gpu_Register_Custom_Call_Args (64 bytes):
          ;; struct_size = 64
          (.set args-seg ValueLayout/JAVA_LONG 0 (long 64))
          ;; function_name pointer
          (.set args-seg ValueLayout/ADDRESS 8 fn-name-seg)
          ;; function_name_size
          (.set args-seg ValueLayout/JAVA_LONG 16 name-len)
          ;; api_version: 0 = legacy custom call ABI
          (.set args-seg ValueLayout/JAVA_INT 24 (int (or api-version 0)))
          ;; custom_call_function_pointer at offset 56
          (.set args-seg ValueLayout/ADDRESS 56 fn-ptr)
          (.invokeWithArguments mh (into-array Object [args-seg]))
          (swap! registered-targets conj target-name)
          true)
        (throw (ex-info "PJRT backend does not export PJRT_Extension_Type_Gpu_Custom_Call (type 0)"
                        {:backend (:backend ctx) :target target-name}))))))

(defn register-w4a16-gemv-kernel!
  "Ensures `bin/libclj_einsum_kernels.so` is loaded and registers `w4a16_gemv_rocm` with OpenXLA PJRT."
  [ctx]
  (when (= (:target ctx) :rocm)
    (let [lib-file (io/file KERNEL_LIB_PATH)]
      (when (.exists lib-file)
        (when-not @kernel-library-arena
          (reset! kernel-library-arena (Arena/ofShared)))
        (let [lookup (SymbolLookup/libraryLookup (.getAbsolutePath lib-file) @kernel-library-arena)]
          (if-let [fn-seg (try (.orElse (.find lookup "w4a16_gemv_rocm") nil) (catch Exception _ nil))]
            (register-custom-call-target! ctx "w4a16_gemv_rocm" fn-seg 0)
            (throw (ex-info "Symbol w4a16_gemv_rocm not found in kernel library" {:path KERNEL_LIB_PATH}))))))))
