# Advanced Architectural Recommendations for `clj-xla`
*Derived from OpenXLA C API (`openxla/xla`), JAX, GoMLX, ZML, and Reactant.jl*

## Executive Summary

Having established our **Phase 1-3 testing and versioning harness** (version compatibility negotiation, generative property testing via `clojure.test.check`, process-isolated worker execution, and CPU-vs-GPU numerical parity), we investigated the official OpenXLA C API codebase ([`openxla/xla/xla/pjrt/c`](https://github.com/openxla/xla/tree/main/xla/pjrt/c)) and leading peer frameworks (JAX, GoMLX, ZML, Reactant.jl).

This document outlines **5 advanced architectural recommendations** to elevate `clj-xla` beyond peer frameworks, taking full advantage of Clojure's homoiconic EDN data representation and Java 25 Project Panama FFM (`java.lang.foreign`).

---

## 1. Extension Chain Architecture & Extension Discovery (`PJRT_Extension_Base`)

### OpenXLA Design Pattern:
In `openxla/xla`, the PJRT C API handles optional and experimental features through an extensible linked-list pointer chain (`extension_start` inside `PJRT_Api`, `PJRT_Client`, and `PJRT_ExecuteOptions`):

```c
struct PJRT_Extension_Base {
  PJRT_Extension_Type type;
  struct PJRT_Extension_Base* next;
};
```

### Recommendation for `clj-xla`:
Implement an extension registry in `clj-xla.pjrt.extensions` that inspects `extension_start` off-heap structs returned by `GetPjrtApi()` and vendor shared objects (`libpjrt_rocm.so`, `libpjrt_cuda.so`).

```clojure
(ns clj-xla.pjrt.extensions
  (:require [clj-xla.pjrt :as pjrt])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(def EXTENSION_TYPE_FFI 1)
(def EXTENSION_TYPE_COLLECTIVES 2)
(def EXTENSION_TYPE_RAW_BUFFER 3)
(def EXTENSION_TYPE_PROFILER 4)

(defn discover-extensions
  "Traverses the linked list at `extension_start` pointer and returns a map of supported extension types."
  [^MemorySegment ext-start-ptr]
  (loop [curr ext-start-ptr
         acc {}]
    (if (or (nil? curr) (= MemorySegment/NULL curr))
      acc
      (let [ext-type (.get curr ValueLayout/JAVA_INT 0)
            next-ptr (.get curr ValueLayout/ADDRESS 8)]
        (recur next-ptr (assoc acc ext-type curr))))))
```

---

## 2. Java 25 Panama Upcall Handles & FFI Custom Calls (`clj-xla.ffi`)

### OpenXLA Capability:
OpenXLA supports the `PJRT_Extension_FFI` (`pjrt_c_api_ffi_extension.h`). This allows host applications to register C/C++ function pointers as custom MLIR operators (`@custom_call`) that XLA compiles directly into hardware executables.

### Clojure / Java 25 Panama Integration:
Using Java 25's `Linker.nativeLinker().upcallStub`, `clj-xla` can create JVM native upcall function handles from pure Clojure functions and register them as `@custom_call` targets:

```clojure
(ns clj-xla.ffi
  (:import [java.lang.foreign Arena FunctionDescriptor Linker MemorySegment ValueLayout]
           [java.lang.invoke MethodHandle MethodHandles MethodType]))

(defn create-panama-upcall-stub
  "Wraps a Clojure IFn into a native function pointer handle using Java 25 Panama upcallStub."
  ^MemorySegment [^Arena arena target-fn arg-layouts return-layout]
  (let [linker (Linker/nativeLinker)
        fd (if return-layout
             (FunctionDescriptor/of return-layout (into-array MemoryLayout arg-layouts))
             (FunctionDescriptor/ofVoid (into-array MemoryLayout arg-layouts)))
        mh (.bindTo (MethodHandles/lookup) target-fn)]
    (.upcallStub linker mh fd arena)))
```

#### Use Cases for `clj-xla`:
- **FlashAttention / Triton Kernels:** Register precompiled native HIP/CUDA kernels (e.g. rocWMMA / FlashAttention-2) without needing C++ wrapper libraries or JNI.
- **Dynamic Host Callbacks:** Allow XLA executables to emit telemetry or stream tokens directly into Clojure channels during GPU graph execution.

---

## 3. Data-Driven SPMD Mesh Sharding (`clj-xla.sharding`)

### Framework Insights (JAX & Reactant.jl):
Both JAX (`jax.sharding.Mesh`, `NamedSharding`) and Reactant.jl handle multi-GPU parallel training (Data Parallelism, Tensor Parallelism, Pipeline Parallelism) by annotating tensor shapes with SPMD mesh specifications.

### Clojure Data-First Sharding DSL:
Because `clj-xla` represents graphs as pure EDN data maps (`{:invars [...], :eqns [...]}`), tensor sharding can be expressed naturally as metadata annotations:

```clojure
(def mesh
  {:rows 2 :cols 1 :devices ["gpu:0" "gpu:1"]})

(defn shard-tensor
  "Annotates an EDN tensor var with SPMD mesh sharding dimensions."
  [tensor-var mesh-spec]
  (vary-meta tensor-var assoc :sharding mesh-spec))

;; Example: Tensor Parallel Attention Linear Projection across Dual AMD GPUs
(defn parallel-dense [x w mesh]
  (let [x-sharded (shard-tensor x {:mesh mesh :spec [:replicated :shard_cols]})
        w-sharded (shard-tensor w {:mesh mesh :spec [:shard_rows :replicated]})]
    [:* x-sharded w-sharded]))
```

When generating StableHLO MLIR, `clj-xla.stablehlo` automatically lowers sharding metadata into `squad` partitioning MLIR attributes or `stablehlo.custom_call @Sharding`.

---

## 4. Off-Heap Arena Tracking & BFC Allocator Telemetry (`clj-xla.pjrt.memory`)

### Problem & Hardware Reality:
OpenXLA GPU/ROCm backends use the BFC (Best-Fit with Coalescing) allocator, allocating up to 90% of available VRAM (e.g., 18 GiB out of 24 GiB on AMD Radeon RX 7900 XTX) upon client initialization.
If Clojure applications create thousands of un-destroyed `PjRtBuffer` handles or transient off-heap `MemorySegment` instances during training/inference loops, VRAM fragmentation or OOM errors occur.

### Recommendation for `clj-xla.pjrt.memory`:
Build an automatic off-heap reference tracking registry that monitors `PjRtBuffer` lifetimes:

```clojure
(ns clj-xla.pjrt.memory
  (:require [clj-xla.pjrt :as pjrt]))

(defonce ^:private active-buffers (atom #{}))

(defn track-buffer!
  "Registers `buf-handle` in active buffer tracking map."
  [api-ctx buf-handle shape dtype]
  (swap! active-buffers conj {:handle buf-handle :shape shape :dtype dtype :allocated-at (System/currentTimeMillis)})
  buf-handle)

(defn release-buffer!
  "Destroys native device buffer `buf-handle` and removes from tracking map."
  [api-ctx buf-handle]
  (pjrt/destroy-buffer! api-ctx buf-handle)
  (swap! active-buffers disj {:handle buf-handle}))

(defn active-buffer-telemetry
  "Returns memory usage telemetry for all currently allocated device buffers."
  []
  (let [bufs @active-buffers]
    {:active-count (count bufs)
     :buffers bufs}))
```

---

## 5. StableHLO Dialect Versioning & Conformance Verification (`clj-xla.conformance`)

### OpenXLA Evolution:
The `stablehlo` dialect evolves continuously with backward/forward compatibility windows (documented in `openxla/stablehlo/docs/compatibility.md`). PJRT plugins compiled against older/newer OpenXLA commits expect specific StableHLO versions (e.g. `1.0.0`, `0.19.0`).

### Recommendation for `clj-xla`:
Add a StableHLO bytecode/version target configuration in `clj-xla.stablehlo`:

1. **Version Lowering Pass (`clj-xla.stablehlo.versioning`):** Allows formatting MLIR output targeted to older OpenXLA plugin releases (e.g., converting newer StableHLO ops into compatible primitive op combinations).
2. **Interpreter Conformance Test Suite:** Run generative EDN graph test suites against OpenXLA's `stablehlo-opt` / `stablehlo-translate` tools to guarantee 100% dialect conformance before sending MLIR text to `PJRT_Client_Compile`.

---

## Summary Matrix of Proposed Capabilities

| Capability | Peer Framework Alignment | Clojure / JVM Panama Advantage | Target Component |
| :--- | :--- | :--- | :--- |
| **PJRT Extension Chain** | OpenXLA C API `pjrt_c_api.h` | Direct Panama struct offset traversal | `clj-xla.pjrt.extensions` |
| **Panama Native FFI Upcalls** | JAX `jax.ffi` / C++ custom calls | Pure Clojure `upcallStub` without C++ JNI | `clj-xla.ffi` |
| **SPMD Mesh Sharding** | JAX `NamedSharding` / Reactant | Data-first EDN graph metadata | `clj-xla.sharding` |
| **BFC Memory Telemetry** | GoMLX `arena.go` / ZML | Panama `Arena` tracking + automatic cleanup | `clj-xla.pjrt.memory` |
| **StableHLO Versioning** | OpenXLA `stablehlo-opt` | EDN graph transformations across dialect versions | `clj-xla.stablehlo.versioning` |
