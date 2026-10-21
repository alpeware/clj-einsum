# clj-einsum: Pure-Clojure UOp Deep Learning Compiler & Runtime

High-performance deep learning compiler framework and hardware runtime in pure Clojure targeting **Java 25+** and **consumer AMD ROCm (`gfx1100` / RX 7900 XTX 24 GB)** via Project Panama FFM.

> **Read [`PROCESS.md`](PROCESS.md) first.** It is the single source of truth for the Staircase milestones, Ali Ghodsi's 4 RSI gates on consumer silicon, and our operating rules.

---

## 1. What It Is

### Declarative Tensor Logic to Metal
`clj-einsum` compiles Pedro Domingos' Declarative Tensor Logic (`einsum.logic.*` — homoiconic Hiccup ASTs `[:= head & body-terms]`) directly into pure-Clojure UOp schedules (`einsum.uop.*`), rendering fused HIP (`gfx1100` Wave32 BF16 WMMA tensor cores & cooperative LDS reductions) and C99 kernels. Kernels are compiled and dispatched directly to hardware via Java 25 Project Panama FFM (`java.lang.foreign` to `libamdhip64.so` / `libhiprtc.so`) with zero host-device round-trips in the hot path, alongside a pure-JVM SIMD reference interpreter (`einsum.logic.interpret`).

### Zero C++ FFI / Zero Python / Zero External ML Compilers
No PyTorch, no OpenXLA/PJRT, no C++ glue libraries. Fused in-register INT4/ternary unpacking (`:uop/unpack-int4`, `:uop/unpack-ternary`), online chunked softmax attention, and persistent ping-pong VRAM step buffers are expressed, scheduled, optimized, and rendered entirely in Clojure persistent data structures.

### Gemma 4 Family on Consumer AMD (24 GB VRAM)
Runs the entire Gemma 4 model family on a single consumer AMD Radeon RX 7900 XTX (24 GB VRAM):
- **Gemma 4 E2B** (`BF16`, 2.3B parameters)
- **Gemma 4 E4B** (`BF16`, 4.3B parameters)
- **Gemma 4 31B** (`W4A16 INT4` group-128, 16.95 GB total VRAM footprint)

```clojure
(require '[einsum.core :as ein])

;; 1. Initialize UOp ROCm backend for AMD Radeon RX 7900 XTX
(def ctx (ein/init-backend! :uop-rocm))

;; 2. Declare a fused Transformer layer in Pedro Domingos' Declarative Tensor Logic AST
(def ast
  [:block {:name :fused_layer}
   [:rms-norm [:h_norm :b :p :d] [:x :b :p :d] [:gamma :d] {:eps 1e-6}]
   [:= [:y :b :p :d] {:act :gelu} [:h_norm :b :p :din] [:w :din :d]]])

;; 3. Compile AST into native Wave32 WMMA HIP kernels scheduled in pure Clojure
(def kernel
  (ein/compile-kernel ctx "fused_block" ast
                      {:in {:x [:tensor [1 16 2048] :f32]
                            :gamma [:tensor [2048] :f32]
                            :w [:tensor [2048 2048] :f32]}
                       :out [:y]}))

;; 4. Dispatch directly to AMD GPU via Project Panama FFM (libamdhip64.so)
(def out (ein/execute kernel {:x x-buf :gamma gamma-buf :w w-buf}))
```

---

## 2. Quickstart & How to Run

### System Requirements
- **Java 25+** (with Project Panama FFM `java.lang.foreign` and `jdk.incubator.vector`)
- **Clojure 1.12+** (`clj --version`)
- **ROCm 6.0+ / 7.x** (`libamdhip64.so`, `libhiprtc.so` for `:uop-rocm`) or **GCC** (for `:uop-cpu`)

### Run Test Suites
```bash
# 1. Fast test suite (332 unit & generative tests including post-XLA golden oracle)
./tools/gemma4.sh -M:test fast

# 2. Declarative Tensor Logic & Relational Memory test suite (271 tests)
./tools/gemma4.sh -M:test logic

# 3. Model architectures & quantization test suite (88 tests)
./tools/gemma4.sh -M:test models

# 4. Hardware runtime & benchmark test suite (20 tests)
./tools/gemma4.sh -M:test hardware
```

### Run Gemma 4 Inference (`:uop-rocm` & `:uop-cpu`)
Always run inference through [`tools/gemma4.sh`](tools/gemma4.sh), which configures JVM native module exports and handles `libjsig.so` signal chaining:
```bash
# Gemma 4 E2B (BF16) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"

# Gemma 4 E4B (BF16) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E4B-it --prompt "Explain monads in Clojure"

# Gemma 4 31B (INT4 group-128, 16.95 GB VRAM) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-31b-it-int4 --prompt "Explain monads in Clojure"

# Autonomous Coding Agent Loop
./tools/gemma4.sh agent --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and list files"
```

---

## 3. Headline Benchmark Table (AMD Radeon RX 7900 XTX 24 GB, `gfx1100`)

All metrics measured on consumer silicon (**AMD Radeon RX 7900 XTX 24 GB**, **AMD Ryzen 9 9950X**, Gentoo Linux Kernel 6.x, ROCm 6.0/7.2).

### Live Gemma 4 Generation Benchmarks

| Model | Variant | Prompt Tokens ($M$) | UOp Prefill (ms) | UOp Decode (tok/s) | Archived XLA Prefill (ms) | Archived XLA Decode (tok/s) | Speedup vs XLA (Prefill / Decode / Total) |
|---|---|---|---|---|---|---|---|
| **Gemma 4 E2B** | `BF16` | $M = 14$ | **32.73 ms** | **59.12 tok/s** | 109.32 ms | 45.07 tok/s | **3.34x** / **1.31x** / **1.67x** |
| **Gemma 4 E2B** | `BF16` | $M = 17$ | **32.43 ms** | **90.06 tok/s** | 51.15 ms | 56.10 tok/s | **1.58x** / **1.61x** / **1.60x** |
| **Gemma 4 E2B** | `BF16` | $M = 32$ | **33.03 ms** | **92.03 tok/s** | 52.12 ms | 56.02 tok/s | **1.58x** / **1.64x** / **1.64x** |
| **Gemma 4 E4B** | `BF16` | $M = 14$ | **43.14 ms** | **41.37 tok/s** | 190.01 ms | 26.18 tok/s | **4.40x** / **1.58x** / **2.05x** |
| **Gemma 4 E4B** | `BF16` | $M = 17$ | **40.84 ms** | **54.45 tok/s** | 188.50 ms | 26.08 tok/s | **4.62x** / **2.09x** / **2.11x** |
| **Gemma 4 E4B** | `BF16` | $M = 32$ | **42.01 ms** | **54.26 tok/s** | 190.86 ms | 26.14 tok/s | **4.54x** / **2.08x** / **2.13x** |
| **Gemma 4 31B** | `INT4` (16.95 GB) | $M = 14$ | **128.65 ms** | **22.01 tok/s** | 145.18 ms | 19.36 tok/s | **1.13x** / **1.14x** / **1.13x** |
| **Gemma 4 31B** | `INT4` (16.95 GB) | $M = 17$ | **128.23 ms** | **20.38 tok/s** | 150.84 ms | 20.43 tok/s | **1.18x** / 1.00x / 1.01x |
| **Gemma 4 31B** | `INT4` (16.95 GB) | $M = 32$ | **131.81 ms** | **20.48 tok/s** | 190.74 ms | 20.39 tok/s | **1.45x** / 1.00x / 1.02x |

### Gemma 4 31B INT4 $M$-Sweep Prefill Latency

| Prompt Length ($M$) | Cold Prefill (ms) | Warm Prefill (ms) | Warm Latency (ms/tok) | Archived XLA Prefill (ms) | Prefill Speedup vs XLA |
|---|---|---|---|---|---|
| $M = 2$ | 88.26 ms | **79.54 ms** | 39.77 ms/tok | — | — |
| $M = 8$ | 122.49 ms | **111.50 ms** | 13.94 ms/tok | — | — |
| $M = 14$ | 129.03 ms | **120.33 ms** | 8.59 ms/tok | 145.18 ms | **1.21x** |
| $M = 15$ | 132.51 ms | **120.43 ms** | 8.03 ms/tok | — | — |
| $M = 16$ | 132.37 ms | **120.31 ms** | 7.52 ms/tok | — | — |
| $M = 32$ | 128.60 ms | **120.55 ms** | 3.77 ms/tok | 190.74 ms | **1.58x** |

### Gemma 4 E2B BF16 Token Scaling (17 $\to$ 456 Tokens)

| Tokens | UOp Prefill (ms) | UOp Latency (ms/tok) | Archived XLA Prefill (ms) | Prefill Speedup vs XLA |
|---|---|---|---|---|
| 17 | **40.31 ms** | 2.37 ms/tok | 72.77 ms | **1.81x** |
| 40 | **45.67 ms** | 1.14 ms/tok | 78.23 ms | **1.71x** |
| 72 | **47.20 ms** | 0.66 ms/tok | 79.52 ms | **1.68x** |
| 136 | **49.49 ms** | 0.36 ms/tok | 82.06 ms | **1.66x** |
| 264 | **60.45 ms** | 0.23 ms/tok | 83.24 ms | **1.38x** |
| 392 | **67.35 ms** | 0.17 ms/tok | 82.93 ms | **1.23x** |
| 456 | **71.11 ms** | 0.156 ms/tok | 82.06 ms | **1.15x** |

> **Archival & Regression Oracle Note**: OpenXLA/PJRT has been permanently excised from this repository. All historical comparisons above are recorded and frozen in [`resources/golden/xla_vs_uop_archive.edn`](resources/golden/xla_vs_uop_archive.edn). Continuous numerical parity and greedy token identity are verified by the standing post-XLA regression oracle in [`resources/golden/gemma4_uop_oracle.edn`](resources/golden/gemma4_uop_oracle.edn) run via `einsum.uop.golden-test`.

---

## 4. Current Frontier: Declarative Agent Loop & Long-Context Compression

The research roadmap is structured by the 4 RSI gates on consumer hardware (`PROCESS.md` §3):

1. **Declarative In-Tensor Agent Memory (`einsum.logic.agent.*`, `einsum.logic.memory.*`)**:
   - In-tensor relational memory cores representing knowledge bases directly as dense matrices.
   - Algebraic extract-threshold-re-embed denoising cycles executing in VRAM to eliminate context re-prompting.
   - Verified-commit memory stores ensuring durable loop state across multi-turn sessions without retraining.

2. **Long-Context INT4 KV Cache & Compression (Gate 1)**:
   - Per-token group-128 INT4 KV cache quantization (`quantize-on-write post-RoPE`, `unpack-in-graph`).
   - Targets 200,000+ token working context for Gemma 4 31B inside the 24 GB VRAM budget.

3. **Clojure Coding Intelligence & Autonomous Recursion (Gates 3 & 4)**:
   - Continuous capability benchmarking via `clojure_bench` and `multipl_e` test suites.
   - Minimizing human judgment cost per accepted, verified catalog entry.

---

## 5. Repository Layout

```
clj-einsum/
├── src/
│   ├── einsum/
│   │   ├── core.clj          # Thin high-level entrypoint & context dispatch
│   │   ├── logic/            # Declarative Tensor Logic DSL, AST expansion, interpreter
│   │   ├── uop/              # Pure-Clojure UOp IR, rangeify, scheduler, HIP/C99 codegen & Panama FFM runtime
│   │   ├── models/           # Gemma 4 (E2B, E4B, 31B INT4), Gemma 3, GPT-2, SmolLM
│   │   ├── runtime/          # Scoped DeviceArena, WeightStore, benchmarking, C2C
│   │   └── quant/            # INT4, ternary, QuIP#, EXL3, TurboQuant pack/unpack
│   ├── tools/                # Model inference CLI drivers & quantization utilities
│   └── experiments/          # Active research code sharded by gate and slug
├── test/
│   └── einsum/               # Fast, logic, models, and hardware test suites
├── resources/
│   ├── golden/               # Post-XLA regression oracle & historical XLA archive EDN
│   ├── catalog/              # Master registry.edn and silicon-verified research pods
│   ├── proposals/            # Pre-registered RFCs, falsification criteria & raw metrics
│   └── data/                 # Static test fixtures and corpora
├── tools/                    # Shell launcher scripts (gemma4.sh, benchmark.sh, quantize.sh)
└── docs/                     # Technical architecture, memory designs, and benchmarks
```

---

## License

Apache License 2.0. See [`LICENSE`](LICENSE) for details.
