# clj-einsum: Executable Tensor Logic for Coding Agents

High-performance Machine Learning compiler framework and runtime for Clojure targeting **Java 25** and **OpenXLA PJRT C API**.

> **Mission**: Make Pedro Domingos' Tensor Logic the working language of AI development — not a formalism on paper, but a toolchain where a coding agent (or a human) writes tensor equations and gets GPU executables, gradients, and verified behavior out.
>
> The repository serves as the empirical measurement apparatus for **Ali Ghodsi's 4 Recursive Self-Improvement (RSI) gates on consumer silicon** (Resource Efficiency, Time Efficiency, Capability Improvement, and Closed-Loop Repeatability; see [`docs/architecture/rsi_gates.md`](docs/architecture/rsi_gates.md) and [`catalog/README.md`](catalog/README.md)).
>
> See [VISION.md](VISION.md) for the strategic vision, operating principles, non-goals, and research pillars.

---

## Architecture: Vertically Integrated Toolchain

One construct all the way down: `[:= head & body-terms]`. No kernels written by hand; zero host matrix math in the hot path. Equations become executables via OpenXLA PJRT:

```
tensor equations (Hiccup AST, pure data)
        │  expand · lower · autodiff — all as data transforms
        ▼
EDN SSA graph IR (Malli-schematized, verifiable)
        │
        ▼
StableHLO → PJRT executables (CPU / ROCm / CUDA)
        │
        ▼
reference interpreter ⇄ device differential testing
```

- **`einsum.logic.*` (The Language)**: AST expansion, lowering, symbolic reasoning, relational memory, in-VRAM contrastive learning, and agent loops.
- **`einsum.core` & `einsum.compiler.compile` (The Substrate)**: Zero-copy Java 25 Project Panama FFM bindings to PJRT C API (`pjrt_c_api.h`), StableHLO MLIR builder, SHA-256 compilation cache, autodiff, and device tensor management.

---

## Pillars

- **Pillar 0 — The Compiler (Built)**: Tensor Logic AST → StableHLO → PJRT, with a reference interpreter for differential verification. Real models execute end-to-end: GPT-2, SmolLM, and Gemma 2/3/4 (35-layer E2B inference on ROCm with resident weights and in-graph INT4 dequantization). Documented in Paper 1 (*Executable Tensor Logic*).
- **Pillar 1 — Trainable Tensor Logic (Active)**: Structure learning and predicate invention inside the language. Relational memory ($R_{\text{mem}}$ cores as learned soft priors), exact in-VRAM adjoints, StableHLO softmax lowering, WebNLG-scale contrastive training, and grafting onto frozen LLM backbones. Documented in Paper 2 (*Trainable Tensor Logic*).
- **Pillar 2 — The Agent Loop (Design → Prototype)**: Three-tier agent architecture (reflex contractions, device-resident deliberation, host-side synthesis) where the agent's own reasoning substrate consists of tensor equations it can read, verify, and rewrite.

---

## Features

- **Pedro Domingos' Declarative Tensor Logic:** Homoiconic Hiccup-style AST DSL (`einsum.logic.*`) unifying relational logic, tensor contraction, broadcasting, and neural network layers.
- **Pure EDN SSA Graph IR:** Flat Single Static Assignment (SSA) computation graphs governed by Malli schemas, lowered directly from Tensor Logic ASTs.
- **Java 25 Project Panama FFM:** Zero-copy native bindings to OpenXLA's PJRT C API (`pjrt_c_api.h`) via `java.lang.foreign`.
- **Sub-Millisecond REPL Feedback:** SHA-256 graph hash compilation caching (`einsum.compiler.compile`) bypassing XLA LLVM codegen on warm REPL evaluations.
- **Multi-Backend OpenXLA Execution:** Seamless hardware execution across CPU, AMD ROCm, Intel SYCL, and NVIDIA CUDA.
- **Pure Clojure LLM Implementations:** Gemma 2, Gemma 3, Gemma 4 (E2B, E4B), SmolLM, and GPT-2 running purely via XLA compilation without manual host matrix math.
- **100% In-VRAM Execution:** Forward passes, backward passes, exact gradient updates, and memory unbinding compile directly to StableHLO without host round-trips.

---

## Setup & Quickstart

### 1. Requirements
- **Java 25+** (`java --version`)
- **Clojure 1.12+** (`clj --version`)

### 2. Fetch Precompiled PJRT Plugin

Fetch the OpenXLA CPU shared binary plugin into `bin/`:

```bash
clj tools/fetch_pjrt.clj cpu
```

### 3. Start Socket REPL

Start a standard Clojure Socket REPL listening on port `5555`:

```bash
clj -M:repl
```

---

## Usage Examples

### 1. Declarative Tensor Logic (Hiccup AST)

Express neural network operations, matrix contractions, and activations in pure declarative Clojure data structures. There is no `:matmul` operator: a shared index contracted on both sides of `:=` *is* the matmul, and activations are attributes on the equation, not separate ops:

```clojure
(ns example.logic
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]))

;; Initialize runtime context
(def ctx (xla/init-backend! :cpu))

;; Define model signature (invars)
(def invars [[:x [:tensor [1 16 768] :f32]]
             [:w [:tensor [768 768] :f32]]
             [:gamma [:tensor [768] :f32]]])

;; Declare computation using Tensor Logic AST: rms-norm, then a dense
;; projection with fused GeLU -- one construct, `:=`, all the way down.
;; The shared :din index contracted on both sides is the matrix multiply.
(def ast
  [[:rms-norm [:h_norm :b :p :d] [:x :b :p :d] [:gamma :d] {:eps 1e-6}]
   [:= [:y :b :p :d] {:act :gelu} [:h_norm :b :p :din] [:w :din :d]]])

;; Lower AST to StableHLO SSA graph and compile
(def graph (lower/ast->graph "dense_block" invars ast #{:y}))
(def exec (xla/compile-graph ctx graph))
```

### 2. Direct EDN SSA Graph

Low-level homoiconic StableHLO graph construction:

```clojure
(ns example.ssa
  (:require [einsum.core :as xla]))

(def ctx (xla/init-backend! :cpu))

(def graph
  {:name "scaled_gelu"
   :invars [[:x [:tensor [1 128 768] :f32]]]
   :outvars [:y]
   :eqns [{:op :stablehlo/constant :value 0.5 :outvars [:c0]}
          {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]})

(def exec (xla/compile-graph ctx graph))
```

---

## Running Models & Relational Evaluation

CLI scripts and wrappers are included for running inference, agent loops, and relational memory evaluation:

```bash
# GPT-2 Inference
clj -M tools/gpt2_inference.clj --prompt "The capital of France is"

# SmolLM-135M Inference
clj -M tools/smollm_inference.clj --prompt "In a galaxy far away"

# Gemma 4 Autoregressive Generation (ROCm / CPU)
./tools/gemma4.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"

# Gemma 4 Autonomous Agent Loop
./tools/gemma4.sh agent --backend rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and calculate total Clojure lines"

# Gemma 4 Relational Memory Benchmark (WebNLG E15 Distractor Diagnostic)
./tools/gemma4.sh relational --checkpoint .dataset/webnlg/checkpoint_gemma4_relational.edn
```

---

## Documentation & Architecture

- [VISION.md](VISION.md): Mission, core principles, pillars, non-goals, and relation to Domingos' program.
- [DESIGN.md](docs/architecture/DESIGN.md): Detailed architectural layers, Malli schemas, and lowering pipeline.
- [AGENTS.md](AGENTS.md): Repository rules, TDD invariants, and optimal inference / agent launching guides.
- [docs/tensor_logic/empirical_journey.md](docs/tensor_logic/empirical_journey.md): Empirical log of experiments (E1–E15), telemetry, and theoretical conclusions.
- [docs/](docs/index.md): Model specifications, hardware benchmarks, PJRT versioning notes, and wiki.
