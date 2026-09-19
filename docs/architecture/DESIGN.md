# DESIGN.md: Architectural Specification for `clj-xla`

## 1. Executive Overview & Philosophy

**`clj-xla`** is a high-performance, open-source Machine Learning compiler framework and runtime for Clojure targeting **Java 25**. It provides a pure functional DSL for defining neural network ops, compiling them to **StableHLO MLIR**, and executing them directly on accelerator hardware (CUDA GPUs, AMD ROCm, Google TPUs, and AVX/NEON CPUs) via OpenXLA's **PJRT C API** using modern **Java 25 Project Panama Foreign Function & Memory (FFM) API** (`java.lang.foreign`).

### Core Guiding Principles

1. **Data over Code (Homoiconic AST):** The intermediate representation (IR) is a pure, flat Single Static Assignment (SSA) EDN graph inspired by JAX’s `jaxpr`. It can be inspected, serialized, validated, and mutated as standard Clojure data.
2. **AI-Agent Native:** Because the AST is pure EDN governed by strict Malli schemas, AI coding agents can generate, validate, optimize, and synthesize model graphs deterministically without writing code strings or dealing with macro expansion pitfalls.
3. **Sub-Millisecond REPL Feedback via Caching:** In-process, zero-copy native execution combined with SHA-256 IR compilation caching allows developers to modify model logic at the REPL and re-execute on persistent GPU memory buffers without restarting the JVM, reloading weights, or re-triggering heavy XLA codegen for unchanged subgraphs.
4. **Clean Division of Labor:** Clojure owns high-level control loops, symbolic automatic differentiation, dynamic graph transformations, and weight management. OpenXLA handles memory layout, kernel fusion, auto-vectorization, register placement, and hardware-level codegen.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Clojure Application Layer                 │
│  - Control Loops (e.g., Autoregressive Sampling, REPL)       │
│  - Model Weight Management (.safetensors Panama mmap)       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│       Layer 3: Pedro Domingos' Declarative Tensor Logic     │
│  - Hiccup-style Relational AST DSL (clj-xla.logic.ast)      │
│  - Index & Shape Unification (clj-xla.logic.index/shape)    │
│  - Neural Primitives & Layers (clj-xla.logic.nn)            │
│  - AST Lowering directly to SSA Graph (clj-xla.logic.lower) │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits Pure EDN SSA Graph
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 2: EDN IR & Transformations           │
│  - Malli Graph Validation & Algebraic DCE/Constant Folding  │
│  - Reverse-Mode Autodiff Engine (VJP & Cotangent Accum)    │
│  - StableHLO MLIR Text Generator & Module Formatter         │
└──────────────────────────────┬──────────────────────────────┘
                               │ Emits StableHLO MLIR
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Layer 1.5: Compiler Caching & Executable Manager   │
│  - SHA-256 IR Hash Lookup & In-Memory PjRtExecutable Cache  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Layer 1: Native Runtime (PJRT)               │
│  - Java 25 Project Panama (java.lang.foreign) FFM bindings  │
│  - SymbolLookup for GetPjrtApi() dynamic plugin loading     │
│  - Off-Heap Arenas, MemorySegment & Async PjRtBuffer safety │
│  - OpenXLA Dynamic Backend (libpjrt_cuda.so / cpu / rocm)   │
└──────────────────────────────┘
```

---

## 3. Detailed Layer Specifications

### Layer 1: Native Runtime & Memory (`clj-xla.pjrt`)

Layer 1 handles foreign function invocation and off-heap memory management without C++ JNI boilerplate using Java 25's finalized `java.lang.foreign` API.

#### PJRT C API Interface & Dynamic Binding
* **Plugin Entrypoint:** Loads vendor dynamic shared objects (`libpjrt_cuda.so`, `libpjrt_cpu.so`, `libpjrt_rocm.so`) at runtime using `SymbolLookup.libraryLookup`.
* **API Function Resolution:** Resolves the C entrypoint `const PJRT_Api* GetPjrtApi()`. Using Panama `StructLayout` definitions, `clj-xla.pjrt` maps the returned `PJRT_Api` struct table to invoke API function pointers (`PJRT_Client_Create`, `PJRT_Client_Compile`, `PJRT_Buffer_FromHostBuffer`, `PJRT_LoadedExecutable_Execute`).

#### Off-Heap Memory Management (`Arena` & `MemorySegment`)
* **Zero-Copy Weight Offloading:** `.safetensors` model parameters are mapped directly into off-heap `MemorySegment` memory via `FileChannel.map` using a `Shared` or `Confined` Java 25 `Arena`.
* **Async Host-to-Device (H2D) Buffer Safety:** `PjRtBuffer` instances are populated from host memory using `PJRT_Buffer_FromHostBuffer`. To prevent premature deallocation during async DMA transfers, `Arena` lifetimes are synchronized with PJRT event completion handles or explicit safe closeable scopes.

#### Key Components:
* `PjRtClient`: Manages hardware device initialization, platform attributes, memory spaces, and compilation contexts.
* `PjRtLoadedExecutable`: Wraps compiled XLA hardware binary executables.
* `PjRtBuffer`: Opaque handle to device-allocated VRAM/SRAM memory buffers.

---

### Layer 2: Intermediate Representation (`clj-xla.stablehlo`)

Layer 2 defines the pure data specification for the computation graph. The graph represents flat Single Static Assignment (SSA) equations, avoiding nested AST trees that complicate variable reuse, constant folding, and backpropagation.

#### Jaxpr-Inspired EDN SSA Format

In `clj-xla`, all inputs, intermediate tensors, constants, and outputs are represented as explicit SSA variables. Scalar constants are auto-lifted into explicit `:stablehlo/constant` equations:

```clojure
{:name "gelu_block"
 :invars  [[:x {:type [:tensor [1 128 768] :f32]}]]
 :outvars [:y]
 :eqns    [{:op :stablehlo/constant :value 0.5          :outvars [:c0]}
           {:op :stablehlo/constant :value 1.0          :outvars [:c1]}
           {:op :stablehlo/constant :value 0.7978845608 :outvars [:c2]}
           {:op :stablehlo/constant :value 0.044715     :outvars [:c3]}
           {:op :stablehlo/constant :value 3.0          :outvars [:c_three]}
           {:op :stablehlo/power    :invars [:x :c_three] :outvars [:t0]}
           {:op :stablehlo/multiply :invars [:t0 :c3]    :outvars [:t1]}
           {:op :stablehlo/add      :invars [:x :t1]     :outvars [:t2]}
           {:op :stablehlo/multiply :invars [:t2 :c2]    :outvars [:t3]}
           {:op :stablehlo/tanh     :invars [:t3]        :outvars [:t4]}
           {:op :stablehlo/add      :invars [:t4 :c1]    :outvars [:t5]}
           {:op :stablehlo/multiply :invars [:x :t5]     :outvars [:t6]}
           {:op :stablehlo/multiply :invars [:t6 :c0]    :outvars [:y]}]}
```

#### Malli Schema Validation (`clj-xla.stablehlo.schema`)

```clojure
(def ElementTypeSchema
  [:enum :f16 :f32 :f64 :bf16 :i8 :i16 :i32 :i64 :pred])

(def TensorTypeSchema
  [:tuple [:= :tensor] [:vector :int] ElementTypeSchema])

(def VarBindingSchema
  [:tuple keyword? TensorTypeSchema])

(def EquationSchema
  [:map
   [:op keyword?]
   [:invars [:vector keyword?]]
   [:outvars [:vector keyword?]]
   [:value {:optional true} [or number? boolean? vector?]]
   [:attrs {:optional true} map?]])

(def GraphSchema
  [:map
   [:name string?]
   [:invars [:vector VarBindingSchema]]
   [:outvars [:vector keyword?]]
   [:eqns [:vector EquationSchema]]])
```

#### StableHLO MLIR Text Printing
`clj-xla.stablehlo` formats EDN SSA graphs into compliant MLIR textual module strings containing `func.func @main(...)` with dialect annotations (`stablehlo.dot_general`, `stablehlo.broadcast_in_dim`, `stablehlo.constant`, etc.), passed directly to `PJRT_Client_Compile`.

---

### Layer 3: Pedro Domingos' Declarative Tensor Logic (`clj-xla.logic.*`)

Layer 3 provides a homoiconic, declarative DSL implementing Pedro Domingos' Tensor Logic. Rather than tracing imperative operator expressions with hidden thread-local state, neural network architectures are defined as pure, relational Hiccup-style AST data vectors.

#### Core Concepts & Hiccup AST (`clj-xla.logic.ast`)
* **Relational Tensor Expressions:** Operations are expressed with explicit symbolic indices resembling Einstein notation:
  ```clojure
  ;; Matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
  [:matmul [:c :i :j] [:a :i :k] [:b :k :j]]
  ;; General relational definition
  [:= [:y :b :p :d] [:x :b :p :din] [:w :din :d]]
  ```
* **Neural Component Primitives (`clj-xla.logic.nn`):** Standard layers such as `[:rms-norm ...]`, `[:layer-norm ...]`, `[:gelu ...]`, `[:swiglu ...]`, `[:rope ...]`, `[:gqa-attention ...]`, and `[:dense-block ...]` are first-class declarative AST forms.
* **Shape & Index Unification (`clj-xla.logic.shape` & `clj-xla.logic.index`):** Symbolic indices (`:b`, `:p`, `:d`, `:kvh`, etc.) are unified across equation heads and bodies, automatically deriving broadcast dimensions, transpose permutations, and contraction axes.

#### Direct Lowering to StableHLO SSA (`clj-xla.logic.lower`)
* `clj-xla.logic.lower/ast->graph` expands compound neural operations and lowers relational equations directly into flat, optimized StableHLO EDN SSA graphs with zero intermediate Java or Python runtime overhead.

#### Pure Clojure Layer Example

```clojure
(ns example.transformer-layer
  (:require [clj-xla.logic.lower :as lower]
            [clj-xla.logic.models.smollm :as smollm]))

;; Generate a complete SmolLM transformer block AST
(def layer-ast (smollm/smollm-layer-ast 0 128))

;; Lower to StableHLO SSA graph
(def graph (lower/ast->graph "smollm_layer_0" invars layer-ast #{:h_out}))
```

---

### Layer 4: Higher-Order Transformations (`clj-xla.autodiff` & `clj-xla.opt`)

Layer 4 transforms forward graphs into backward graphs and optimizes graphs prior to MLIR serialization.

1. **Reverse-Mode Autodiff (VJPs):**
   * Traverses the forward SSA `:eqns` vector in reverse order.
   * Emits vector-Jacobian product (VJP) equations for backpropagation.
   * **Cotangent Accumulation:** When an intermediate tensor `v` is consumed by multiple downstream operations, `clj-xla.autodiff` automatically inserts a `:stablehlo/add` node to sum accumulated gradients (`dv = dv1 + dv2`) before propagating gradients backward.
   * **Shape Reduction:** Handles broadcast alignment during reverse propagation via automatic `reduce_sum` along broadcasted axes.
   * Merges forward and backward equations, appending optimizer updates (e.g., AdamW state updates).

2. **Frontend Graph Optimizations (`clj-xla.opt`):**
   * **Dead Code Elimination (DCE):** Prunes unused SSA nodes not transitively reachable from `:outvars`.
   * **Constant Folding:** Pre-computes purely scalar static subgraphs during tracing.
   * **`vmap` Vectorization:** Automatically maps batch dimensions over unbatched single-sample functions.

---

### Layer 1.5 & Execution Engine: SHA-256 Compilation Caching (`clj-xla.compile`)

To guarantee sub-millisecond REPL feedback while working with heavy XLA compiler backends:

1. **Graph Hashing:** When `trace-and-compile` is invoked, `clj-xla.compile` computes a SHA-256 hash of the normalized EDN graph (including input shapes and target hardware platform).
2. **In-Memory Executable Cache:** The compiled `PjRtLoadedExecutable` native handle is stored in an in-memory `atom` map.
3. **Execution Latency:**
   * **First Call (Cold):** Tracing (< 1ms) + StableHLO Printing (< 1ms) + XLA Codegen (50ms - 200ms) = ~50-200ms total.
   * **Subsequent Calls (Warm / REPL Re-eval):** SHA-256 Cache Hit -> Direct execution of `PjRtLoadedExecutable` on `PjRtBuffer` handles in **< 1ms**.

---

## 4. Target Use Case Architectures

### A. Non-Autoregressive Generation (DiffusionGemma)

* **Strategy:** Compile the heavy backbone denoising model into a static StableHLO executable.
* **Control Loop:** Pure Clojure code orchestrates the 256-token canvas, entropy estimation, and self-conditioning loops, invoking the compiled `PjRtLoadedExecutable` per denoising step on persistent `PjRtBuffer` handles in GPU memory.

### B. Fine-Tuning (FunctionGemma 270M / LoRA)

* **Strategy:** Trace forward pass $\to$ generate VJP backward pass via `clj-xla.autodiff` $\to$ append AdamW updates.
* **Execution:** Compile the full training step into a single XLA executable. XLA automatically manages activation checkpointing (rematerialization) to fit within GPU VRAM constraints.

### C. Distillation & Pre-Training

* **Strategy:** Execute two distinct PJRT executables concurrently:
  1. Teacher Model (Forward Pass Only, FP16/FP8).
  2. Student Model (Forward + Backward Pass).
* Loss function combines cross-entropy and KL divergence over logits directly in device memory.

---

## 5. Developer & AI Agent Workflows

### Human REPL Workflow

1. Load model weights into off-heap GPU buffers via `pjrt/to-device`.
2. Author neural layers or kernel logic as declarative Tensor Logic AST expressions (`[:rms-norm ...]`, `[:matmul ...]`).
3. Lower and compile in the REPL (`(->> ast (lower/ast->graph ...) (core/compile-and-run client ...))`).
4. Execute instantly (< 1ms warm execution latency) over existing device memory buffers without losing GPU state.

### AI Agent Protocol (RSI & Code Generation)

1. **Generation:** AI agent emits declarative Tensor Logic AST vectors or EDN graph maps (`{:invars [...], :eqns [...]}`).
2. **Validation:** Agent verifies AST shapes and graphs against Malli schemas locally in microseconds.
3. **Execution & Feedback:** Agent lowers the AST to StableHLO SSA via `clj-xla.logic.lower/ast->graph` and compiles via `clj-xla.core/compile-and-run`.
4. **Mutations:** Agent applies pure data transformations (`assoc-in`, `update`, `postwalk`) to explore novel network topologies or kernel optimizations deterministically.

---

## 6. Repository Layout & Phased Roadmap

```
clj-xla/
├── README.md
├── DESIGN.md
├── deps.edn
└── src/
    └── clj_xla/
        ├── core.clj           ;; High-level JIT execution API & REPL entrypoint
        ├── compile.clj        ;; Graph hashing & executable caching
        ├── pjrt.clj           ;; Panama Java 25 FFM bindings to libpjrt_cuda/cpu/rocm
        ├── stablehlo.clj      ;; EDN SSA schema, validation, & MLIR printer
        ├── logic/             ;; Pedro Domingos' Declarative Tensor Logic
        │   ├── ast.clj        ;; Hiccup AST syntax & validation
        │   ├── expand.clj     ;; Compound macro expansion
        │   ├── shape.clj      ;; Symbolic shape inference & propagation
        │   ├── index.clj      ;; Einstein notation index unification & permutations
        │   ├── lower.clj      ;; AST -> StableHLO SSA compilation
        │   ├── nn.clj         ;; Neural components (RMSNorm, RoPE, Attention, MLP)
        │   ├── dce.clj        ;; Dead code elimination
        │   ├── autodiff.clj   ;; Symbolic reverse-mode autodiff for AST
        │   └── models/        ;; Pre-built pure AST architectures (GPT-2, SmolLM, Gemma)
        ├── autodiff.clj       ;; SSA-level reverse-mode VJP auto-differentiation
        ├── opt.clj            ;; SSA DCE & constant folding graph passes
        ├── generation/        ;; Autoregressive & diffusion sampling loops
        ├── tokenizer/         ;; Fast BPE and SentencePiece tokenizers
        └── safetensors.clj    ;; Panama MemorySegment zero-copy off-heap weight loader
```

### Development Phases

* **Phase 1: Foundation (PJRT & Memory with Java 25 Panama)**
  * Implement Panama FFM bindings for `GetPjrtApi`, `PjRtClient`, `PjRtBuffer`, `PjRtLoadedExecutable`.
  * Build mmap `.safetensors` parser into off-heap `MemorySegment`s using Java 25 `Arena`.

* **Phase 2: StableHLO IR, Tensor Logic Engine & Caching**
  * Implement strict Malli schemas for Jaxpr-style EDN graphs.
  * Build `clj-xla.logic.*` Pedro Domingos declarative Tensor Logic AST engine.
  * Write EDN-to-StableHLO MLIR text printer.
  * Implement SHA-256 graph hash compilation cache.

* **Phase 3: Autodiff & Optimizations**
  * Build reverse-mode VJP generator with cotangent accumulation and broadcast reduction.
  * Add DCE and constant-folding passes in `clj-xla.opt` and `clj-xla.logic.dce`.

* **Phase 4: High-Level Models & Agent Tooling**
  * Implement Gemma 2, Gemma 3, Gemma 4, SmolLM, and GPT-2 Tensor Logic models.
  * Build discrete text diffusion canvas runtime.
  * Package agent schema validation tooling for autonomous graph synthesis.
