# Debugging, Profiling, and Tracing in `clj-einsum`

This guide outlines the debugging, profiling, and telemetry tracing tools built into **`clj-einsum`** (inspired by **JAX** debugging affordances). These tools enable autonomous AI agents and software engineers to introspect execution graphs, diagnose numerical instability, profile execution latencies, and export performance spans for automated optimization feedback.

---

## 1. 🔍 Graph Introspection & Metadata Annotations

In `clj-einsum`, StableHLO equations support attached metadata and source locations directly in equation `:attrs` or via Tensor Logic AST annotations:

### A. Location Metadata
In large multi-layer architectures like Gemma 4 ($35$ layers), locating which specific layer or matrix multiplication triggered an issue can be difficult in raw MLIR text. Attaching location labels (`loc("gemma/layer_12/attn_matmul")`) to equations or AST nodes carries through into StableHLO MLIR:

```clojure
;; StableHLO SSA Equation with explicit metadata
{:op :stablehlo/dot_general
 :invars [:x :qkv_w]
 :outvars [:t_dot_12]
 :attrs {:loc "gemma/layer_12/attn"}}
```

When serialized via `einsum.compiler.stablehlo/graph->mlir-text`, instructions inherit exact source labels:
```mlir
%t_dot_12 = "stablehlo.dot_general"(%x, %qkv_w) { ... } : (tensor<1x1x768xf32>, tensor<768x2304xf32>) -> tensor<1x1x2304xf32> loc("gemma/layer_12/attn")
```

### B. Graph Validation
To catch ill-formed graphs, shape mismatches, or malformed ASTs before passing to the native compiler, Malli schema validation is run ahead-of-time via `einsum.compiler.stablehlo`:

```clojure
(require '[malli.core :as m]
         '[einsum.compiler.stablehlo :as shlo])

(m/validate shlo/GraphSchema graph)
```

---

## 2. ⏱️ High-Precision Telemetry Profiling (`einsum.runtime.profile`)

`einsum.runtime.profile` provides microsecond-resolution span profiling for tracing, compilation, memory allocation, prefill, and decode step latencies.

### A. Micro-second Profile Spans (`with-profile`)
```clojure
(require '[einsum.runtime.profile :as profile])

(let [metrics (atom {})]
  (profile/with-profile metrics "graph_compilation"
    (compile-inference-executables session prompt-len))
  (println @metrics))
```

**Output Telemetry Report (`scratch/gemma4_profile.edn`)**:
```clojure
{:weight_transfer {:duration-ms 2809.12, :duration-us 2809127}
 :graph_compilation {:duration-ms 3573.13, :duration-us 3573137}
 :autoregressive_generation {:duration-ms 13167.43, :duration-us 13167437}}
```

### B. Chrome Trace & Perfetto Visual Export (`export-chrome-trace`)
Profile spans can be exported directly into standard Chrome Tracing JSON format (`chrome://tracing` or [`ui.perfetto.dev`](https://ui.perfetto.dev)):

```clojure
(let [spans-atom (atom [])]
  (binding [profile/*active-trace-spans* spans-atom]
    (run-agent-workload ...))
  (profile/save-chrome-trace! @spans-atom "scratch/gemma4_chrome_trace.json"))
```

---

## 3. 📈 Automated Performance Observations & Recommendations for Gemma 4

Based on empirical profile metrics collected from Gemma 4 E2B inference runs:

| Phase | Observed Latency | Bottleneck Analysis | SOTA Optimization Recommendation |
| :--- | :--- | :--- | :--- |
| **Weight Transfer** | **2,809 ms** (2.81 s) | Cold host-to-device PCIe allocation for 4.6 GB parameters | **Pre-allocated Permanent VRAM Session**: Keep weights pinned in device memory across agent turns. |
| **Graph Compilation** | **3,573 ms** (3.57 s) | Runtime JIT compilation of prefill + decode graphs | **Persistent OpenXLA Cache**: Enable `--xla_gpu_kernel_cache_file` to reuse compiled GCN binaries across restarts. |
| **Prefill Latency** | **835 ms** (17 tok) | Single-pass prompt processing | **Grouped GEMM & INT8 Weights**: Quantize static weights to INT8 to double memory bandwidth throughput. |
| **Decode Step Latency** | **422 ms / tok** (CPU) / **36 ms / tok** (GPU) | Host-side token step dispatch loop | **In-VRAM Fused `stablehlo.while` Loop**: Eliminate per-token host invocations by running the entire decode loop inside GPU VRAM. |
