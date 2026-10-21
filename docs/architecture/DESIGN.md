# DESIGN.md: Architectural Specification for `clj-einsum`

## 1. Executive Overview & Philosophy

**`clj-einsum`** is a high-performance deep learning compiler framework and hardware runtime implemented in pure Clojure targeting **Java 25+** and **consumer AMD ROCm (`gfx1100` / AMD Radeon RX 7900 XTX 24 GB)** via Project Panama Foreign Function & Memory (FFM) API (`java.lang.foreign`).

The framework compiles Pedro Domingos' Declarative Tensor Logic directly into pure-Clojure Micro-Operation (UOp) schedules, rendering native Wave32 BF16 WMMA tensor core kernels dispatched to GPU hardware with zero C++ glue libraries, zero Python, and zero external ML compilers.

### Core Guiding Principles

1. **Data over Code (Homoiconic AST & Hash-Consed UOp IR):** Both the high-level Declarative Tensor Logic (Hiccup relational vectors `[:= head & body-terms]`) and the low-level UOp IR are immutable Clojure data structures. Model architectures, schedules, and tile configurations can be inspected, serialized, transformed, and reasoned about using standard Clojure persistent functions.
2. **AI-Agent Native:** Because computation graphs are pure Clojure data, autonomous coding agents can synthesize, optimize, test, and commit model kernels deterministically without dealing with macro expansion pitfalls, Python virtual environments, or fragile C++ build systems.
3. **Sub-Millisecond REPL Feedback via Caching & Persistent VRAM:** Zero-copy off-heap memory management combined with SHA-256 disk-cached compilation (`.uop-cache/`) allows developers and agents to modify model logic at the Clojure REPL and dispatch directly to persistent VRAM buffers without reloading weights or restarting the JVM.
4. **Pure Functional Compiler Stack (Sans-IO Core):** The compiler pipeline (`lower` $\to$ `rewrite` $\to$ `schedule` $\to$ `rangeify` $\to$ `opt` $\to$ `render`) is strictly pure and side-effect free. Native device side effects (compilation via `libhiprtc.so`, buffer allocation and kernel dispatch via `libamdhip64.so`) are completely isolated to the runtime boundary shell (`einsum.uop.runtime`).
5. **RSI Measurement Apparatus:** The entire toolchain serves as the empirical apparatus for Ali Ghodsi's 4 Recursive Self-Improvement (RSI) gates on consumer silicon (AMD Radeon RX 7900 XTX 24 GB). See [`rsi_gates.md`](rsi_gates.md) and [`../../resources/catalog/README.md`](../../resources/catalog/README.md).

---

## 2. The 3-Layer System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Clojure Application Layer                   │
│  - Autoregressive Generation & Dynamic Samplers             │
│  - Autonomous Coding Agent Loop & In-Tensor Relational Core │
│  - Weight Store (.safetensors Panama mmap / hipHostRegister)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│       Layer 3: Pedro Domingos' Declarative Tensor Logic     │
│  - Hiccup Relational AST DSL (einsum.logic.ast)             │
│  - Relational Einsum Contractions & Index Unification      │
│  - Neural Primitives: RMSNorm, RoPE, Gated MLP, Attention   │
│  - In-Tensor Relational Memory & Knowledge Grounding        │
└──────────────────────────────┬──────────────────────────────┘
                               │ Lowers to UOp DAG
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            Layer 2: Pure-Clojure UOp Compiler Stack         │
│  - UOp DAG Construction & Hash-Consing (einsum.uop.lower)    │
│  - Term-Rewriting & Algebraic Canonicalization (rewrite)    │
│  - Realization Boundaries & Epilogue Fusion (schedule)      │
│  - Coordinate Rangeification & Modulo Strides (rangeify)    │
│  - RDNA3 BEAM & Tiling Optimizer (opt + beam_catalog.edn)   │
│  - C99 & AMD HIP C++ Kernel Renderer (einsum.uop.render)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits Native C99 / HIP Source
                               ▼
┌─────────────────────────────────────────────────────────────┐
│    Layer 1: Panama FFM Device Runtime & SIMD Reference      │
│  - Java 25 Panama FFM (java.lang.foreign) Native Bindings   │
│  - In-Process hiprtc Compile & SHA-256 .hsaco Disk Caching  │
│  - Zero-Alloc slot-ptrs Launch Loop & Ping-Pong Buffers     │
│  - Scoped Device Arenas (einsum.runtime.arena)              │
│  - Pure-JVM SIMD Reference Interpreter (logic.interpret)    │
└──────────────────────────────┴──────────────────────────────┘
                               │ Direct Native Dispatch
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      Target Hardware                        │
│  - AMD Radeon RX 7900 XTX (24 GB VRAM, gfx1100, Wave32 WMMA)│
│  - Host CPU (AVX-512 / AVX2 via GCC or Vector API SIMD)     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Layer Specifications

### Layer 3: Declarative Tensor Logic (`einsum.logic.*`)

Pedro Domingos' Declarative Tensor Logic represents neural network layers as relational logic assertions over multidimensional index spaces.

- **Homoiconic Hiccup ASTs**: Operations are written in standard Clojure vectors:
  ```clojure
  ;; Relational Contraction with Gated GELU Activation
  [:= [:y :b :p :d] {:act :gelu} [:x :b :p :din] [:w :din :d]]

  ;; RMS Normalization with Learnable Scale
  [:rms-norm [:out :b :p :d] [:x :b :p :d] [:gamma :d] {:eps 1e-6}]
  ```
- **Index & Shape Unification (`einsum.logic.compile`)**: Matches input and output index symbols across terms to infer shapes, transpose requirements, reduction axes, and broadcasting semantics automatically.
- **Neural Layer Primitives (`einsum.logic.nn`)**: Pre-packaged high-level modules for complete transformer blocks (Multi-Head Attention, Grouped Query Attention, SwiGLU / GeGLU MLPs, RMSNorm, Rotary Position Embeddings).
- **Relational In-Tensor Memory (`einsum.logic.memory.*`)**: Encodes facts and associative memories directly into tensor spaces, enabling algebraic extract-threshold-re-embed cycles executing inside VRAM.

---

### Layer 2: Pure-Clojure UOp Compiler (`einsum.uop.*`)

Layer 2 ingests the declarative logic AST and compiles it through a sequence of pure functional transformations:

1. **Lowering (`einsum.uop.lower`)**: Lowers relational logic ASTs into a canonical DAG of hash-consed micro-operations (`:uop/buffer`, `:uop/const`, `:uop/alu`, `:uop/reduce-axis`, `:uop/reshape`, `:uop/permute`, `:uop/unpack-int4`).
2. **Algebraic Rewrite (`einsum.uop.rewrite`)**: Executes iterative pattern-matching simplifications. Canonicalizes identities, fuses contiguous coordinate views, and enforces structural rules such as Rule 8 (preserving squeeze reshapes prior to channel-splitting projections).
3. **Schedule Partitioning (`einsum.uop.schedule`)**: Analyzes consumer dependencies to determine kernel boundaries:
   - Evaluates `effective-consumer-count` through coordinate views to materialize multi-consumer activations (e.g., shared RMSNorm outputs) exactly once in VRAM.
   - Materializes channel-splitting reshapes (`splits-channel-axis?`) for coalesced multi-head attention layouts.
   - Fuses reconvergent epilogue subgraphs (`find-reconvergent-epilogue-subchain` for `:act :gelu`, `:silu`, and `:softcap`) directly into reduction kernel registers.
   - Selects between online streaming softmax chunking ($M \le 16$) and direct 4D attention ($M > 16$).
4. **Coordinate Rangeification (`einsum.uop.rangeify`)**: Translates symbolic multidimensional tensor indexing into explicit index arithmetic, stride calculations, and modulo ring-buffer addressing.
5. **RDNA3 BEAM & Tiling Optimizer (`einsum.uop.opt` + `resources/uop/rdna3_beam_catalog.edn`)**:
   - Matches kernel shapes against empirical RDNA3 tile templates.
   - Configures cooperative Wave32 LDS reductions for decode ($M = 1$).
   - Configures Wave32 BF16 WMMA tensor core GEMM tiles with bank-conflict-free stride-18 LDS layout (`s_W[4][16][18]` + `s_X[TM][16][18]`), strip-minimizing $M$-tiling ($TM \in \{1..8\}$), and 4-way $N$-tiling ($TN = 4$) for prefill ($M \ge 2$).
   - Fuses in-register group-128 INT4 (`:uop/unpack-int4`) and 2-bit ternary (`:uop/unpack-ternary`) dequantization directly into tile loaders.
6. **Kernel Renderer (`einsum.uop.render`)**: Emits optimized, unrolled AMD HIP C++ or C99 source code.

---

### Layer 1: Native Panama FFM Runtime (`einsum.uop.runtime`)

Layer 1 bridges the pure Clojure compiler to physical silicon via Project Panama FFM (`java.lang.foreign`):

- **Direct ROCm Driver Bindings**: Dynamically resolves and invokes native functions from `libhiprtc.so` (runtime compilation) and `libamdhip64.so` (memory management and kernel dispatch).
- **SHA-256 Binary Caching**: Hashes emitted HIP C++ strings with SHA-256 and retrieves compiled `.hsaco` binaries from `.uop-cache/`, reducing repeat compilation times to under 1 ms.
- **Zero-Allocation Launch Loop**: Reuses off-heap argument pointer arrays (`slot-ptrs`), eliminating JVM object allocations in the critical dispatch path.
- **Pre-$t_0$ ROCm Warmup**: Executes a dry pass before starting benchmark timers to ensure GPU driver queues and ISA binaries are initialized.
- **Large Checkpoint Ingestion (>4 GiB)**: Pins host memory buffers via `hipHostRegister` (`hipHostRegisterMapped | hipHostRegisterPortable`), enabling high-speed direct DMA streaming of 16.95 GB Gemma 4 31B weight sets over PCIe.
- **Pure-JVM SIMD Reference Interpreter (`einsum.logic.interpret`)**: Evaluates logic ASTs on host CPU using Java's Vector API (`jdk.incubator.vector`), providing a standalone mathematical reference without GPU hardware requirements.

---

## 4. Hardware Support & Empirical Validation

The primary accelerator target is the **AMD Radeon RX 7900 XTX (24 GB VRAM, `gfx1100`)**, paired with **AMD Ryzen 9 9950X**:

| Model | Variant | Footprint | Prefill Speedup vs Archived XLA | Decode Throughput |
|---|---|---|---|---|
| **Gemma 4 E2B** | `BF16` (2.3B) | 4.6 GB | **1.58x – 3.34x** | **59.1 – 92.0 tok/s** |
| **Gemma 4 E4B** | `BF16` (4.3B) | 8.6 GB | **4.40x – 4.62x** | **41.4 – 54.5 tok/s** |
| **Gemma 4 31B** | `INT4` (31B) | **16.95 GB** | **1.13x – 1.45x** | **20.4 – 22.0 tok/s** |

All model variants achieve strict greedy token equivalence and numerical parity against the golden regression oracle (`resources/golden/gemma4_uop_oracle.edn`).
