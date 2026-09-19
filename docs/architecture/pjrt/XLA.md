# XLA.md: Vendor Artifacts & Runtime Dependency Specification

This document details the **Lean Vendoring & Binary Management Strategy** for `clj-xla`. It describes how the library interfaces with OpenXLA's PJRT C API, manages native precompiled binaries (`libpjrt_cpu.so`, `pjrt_cuda_plugin.so`), and provides a clean environment for both human developers and AI coding agents targeting **Java 25**.

---

## 1. Architectural Philosophy: Why Zero Submodules?

The official `openxla/xla` repository is a massive C++ monorepo. Submoduling it introduces several issues:

* **Context Bloat:** Thousands of internal C++ files clog LLM context windows and file indexers.
* **Build Complexity:** Compiling XLA from source requires complex Bazel build environments and hours of compilation time.
* **Versioning Overhead:** Tracking monorepo commits creates unstable C API boundaries.

### The Solution: Lean Vendoring

PJRT provides a standardized C API (`pjrt_c_api.h`) designed for long-term ABI stability. We vendor **only** the standard C header and an EDN specification of StableHLO operations. All hardware runtimes are loaded dynamically as precompiled shared libraries (`.so` / `.dylib` / `.dll`) via **Java 25 Project Panama** (`java.lang.foreign`).

```
clj-xla/
├── vendor/
│   ├── include/
│   │   └── pjrt_c_api.h             ;; Official C API header for Panama bindings
│   └── specs/
│       └── stablehlo_ops.edn        ;; Machine-readable spec of StableHLO operations
├── scripts/
│   └── fetch_pjrt_binaries.clj      ;; Automated fetcher for precompiled PJRT plugins
└── bin/                            ;; Local git-ignored store for native shared libs
    ├── libpjrt_cpu.so
    └── pjrt_cuda_plugin.so
```

---

## 2. Vendored Artifacts

### A. The C API Header (`vendor/include/pjrt_c_api.h`)

This header defines the C function pointers, structs, and enums exposed by OpenXLA backends:

* `PJRT_Api`: The top-level function pointer struct holding pointers to initialization, buffer allocation, compilation, and execution methods.
* `GetPjrtApi`: The standard C entrypoint symbol exported by all PJRT dynamic plugins (`const PJRT_Api* GetPjrtApi()`).
* `PJRT_Client_Create`: Initializes a runtime client for a given device (CPU, CUDA, ROCm, TPU).
* `PJRT_Buffer_FromHostBuffer`: Performs memory allocation into off-heap device memory.
* `PJRT_Executable_Compile`: Compiles a StableHLO MLIR module string into an executable handle.

> **Agent Directive:** When writing Panama bindings in `clj-xla.pjrt`, refer exclusively to `vendor/include/pjrt_c_api.h`. Do not introduce non-standard header declarations.

---

### B. StableHLO Operations Spec (`vendor/specs/stablehlo_ops.edn`)

To enable Malli schema validation and AI-agent graph generation without parsing C++ TableGen files, we maintain an EDN specification of valid StableHLO dialect operations.

```clojure
{:stablehlo/dot_general
 {:doc "General matrix multiplication and contraction across specified dimensions."
  :category :tensor-math
  :invars 2
  :outvars 1
  :required-attrs [:lhs_contracting_dimensions :rhs_contracting_dimensions]
  :attr-schema {:lhs_contracting_dimensions [:vector :int]
                :rhs_contracting_dimensions [:vector :int]
                :lhs_batch_dimensions       {:optional true} [:vector :int]
                :rhs_batch_dimensions       {:optional true} [:vector :int]}}

 :stablehlo/slice
 {:doc "Extracts a sub-array from an input tensor."
  :category :slicing
  :invars 1
  :outvars 1
  :required-attrs [:start_indices :limit_indices :strides]
  :attr-schema {:start_indices [:vector :int]
                :limit_indices [:vector :int]
                :strides       [:vector :int]}}

 :stablehlo/reduce_mean
 {:doc "Computes mean across specified reduction axes."
  :category :reduction
  :invars 1
  :outvars 1
  :required-attrs [:axes :keep_dims]
  :attr-schema {:axes [:vector :int]
                :keep_dims :boolean}}}
```

---

## 3. Precompiled Binary Management

Precompiled PJRT native binaries are distributed via official PyPI packages (such as `jaxlib` and `jax-cuda12-pjrt` / `jax-cuda13-pjrt`) or Google Cloud releases. PyPI wheels are standard Zip archives; our fetch script downloads and extracts these shared libraries directly without requiring Python or `pip` on the host system.

### Hardware Plugin Mapping

| Hardware Target | OS Platform | Shared Library Name | Source Package | Notes |
| --- | --- | --- | --- | --- |
| **CPU** (AVX2/NEON) | Linux | `libpjrt_cpu.so` | `jaxlib` PyPI wheel | Default fallback runtime for development & testing. |
| **CPU** (Apple Silicon) | macOS | `libpjrt_cpu.dylib` | `jaxlib` PyPI wheel | Native ARM64 macOS CPU plugin. |
| **NVIDIA GPU** | Linux | `pjrt_cuda_plugin.so` | `jax-cuda12-pjrt` / `jax-cuda13-pjrt` | Requires matching CUDA driver installed on host system. |
| **AMD GPU** | Linux | `pjrt_rocm_plugin.so` | `jax-rocm` PyPI wheel | Built for ROCm 6.x / 7.x runtimes. |

---

## 4. Binary Fetcher Script (`scripts/fetch_pjrt_binaries.clj`)

Execute this Clojure CLI script to fetch and extract precompiled PJRT libraries directly into your local `bin/` directory:

```clojure
;; Script to download and unpack official PJRT shared binaries from jaxlib/pypi wheels
(ns scripts.fetch-pjrt-binaries
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.util.zip ZipInputStream]))

(def PJRT-VERSION "0.11.0") ; Matched against current stable jaxlib release

(def SOURCES
  {:cpu {:url (str "https://files.pythonhosted.org/packages/jaxlib/jaxlib-" PJRT-VERSION "-cp312-cp312-manylinux_2_27_x86_64.whl")
         :so-name "libpjrt_cpu.so"
         :entry-pattern #"pjrt_c_api_cpu\.so$"}
   :cuda12 {:url (str "https://files.pythonhosted.org/packages/jax-cuda12-pjrt/jax_cuda12_pjrt-" PJRT-VERSION "-py3-none-manylinux_2_27_x86_64.whl")
            :so-name "pjrt_cuda_plugin.so"
            :entry-pattern #"pjrt_cuda_plugin\.so$"}})

(defn unpack-so-from-wheel [url entry-pattern out-file]
  (with-open [in (io/input-stream url)
              zip (ZipInputStream. in)]
    (loop []
      (when-let [entry (.getNextEntry zip)]
        (if (re-find entry-pattern (.getName entry))
          (do
            (println (str "Extracting " (.getName entry) " -> " (.getAbsolutePath out-file)))
            (io/copy zip out-file))
          (recur))))))

(defn fetch-binary [target]
  (let [{:keys [url so-name entry-pattern]} (get SOURCES target)
        out-dir (io/file "bin")
        out-file (io/file out-dir so-name)]
    (.mkdirs out-dir)
    (println (str "Fetching " target " PJRT plugin..."))
    (unpack-so-from-wheel url entry-pattern out-file)
    (println (str "Successfully installed " (.getAbsolutePath out-file)))))

(let [target (keyword (or (first *command-line-args*) "cpu"))]
  (fetch-binary target))
```

### Usage

```bash
# Fetch CPU runtime
clj scripts/fetch_pjrt_binaries.clj cpu

# Fetch CUDA 12 GPU runtime
clj scripts/fetch_pjrt_binaries.clj cuda12
```

---

## 5. Panama Native Interop Strategy (`java.lang.foreign`)

`clj-xla` uses Java 25 Project Panama (`java.lang.foreign`) to bind to PJRT. This completely eliminates C++ compilation steps and custom JNI native libraries.

### Binding Sequence

1. **Dynamic Symbol Lookup:** `SymbolLookup.libraryLookup` loads `bin/libpjrt_cpu.so` or `bin/pjrt_cuda_plugin.so` at runtime into an off-heap `Arena`.
2. **API Struct Unpacking:** Resolves the standard entrypoint symbol `GetPjrtApi()`, which returns a foreign `MemorySegment` pointing to the populated `PJRT_Api` struct.
3. **MethodHandle Downcall Creation:** `Linker.nativeLinker().downcallHandle(...)` wraps function pointers inside `MethodHandle` instances for sub-microsecond native invocation overhead.

```clojure
(ns clj-xla.pjrt.bindings
  (:import [java.lang.foreign Arena ForeignLinker SymbolLookup MemorySegment FunctionDescriptor ValueLayout]))

(defn load-pjrt-api
  "Loads the PJRT shared library from `lib-path` and returns a MemorySegment pointing to PJRT_Api struct."
  [lib-path arena]
  (let [lookup (SymbolLookup/libraryLookup (java.nio.file.Path/of lib-path (into-array String [])) arena)
        get-api-handle (.orElseThrow (.find lookup "GetPjrtApi"))
        linker (ForeignLinker/nativeLinker)
        ;; const PJRT_Api* GetPjrtApi()
        descriptor (FunctionDescriptor/of ValueLayout/ADDRESS)
        downcall (.downcallHandle linker get-api-handle descriptor)]
    (.invokeExact downcall)))
```

---

## 6. Guidelines for AI Coding Agents

When generating code or refactoring within `clj-xla`:

1. **Never generate C/C++ source code.** All interop must occur via Panama `MemorySegment` and `MethodHandle` bindings in Clojure/Java targeting Java 25.
2. **Strict Spec Compliance:** Validate all generated or transformed EDN graphs against `vendor/specs/stablehlo_ops.edn` and `clj-xla.stablehlo.schema` prior to serializing MLIR.
3. **Off-Heap Safety:** Always bind off-heap memory allocations to explicit Project Panama `Arena` blocks (`Arena/ofConfined` or `Arena/ofShared`) to guarantee deterministic cleanup without waiting for JVM Garbage Collection.
