# clj-einsum: Executable Tensor Logic for Coding Agents

High-performance deep learning compiler framework and runtime in pure Clojure targeting **Java 25** and **OpenXLA PJRT C API** via Project Panama FFM.

> **Read [`PROCESS.md`](PROCESS.md) first.** It is the primary guide for our Staircase roadmap, the 4 RSI gates on consumer silicon, and our operating rules.

---

## What It Is

`clj-einsum` makes Pedro Domingos' Declarative Tensor Logic the working language of AI development:
- **One Construct All the Way Down**: Neural architectures are declared as homoiconic Hiccup ASTs (`[:= head & body-terms]`). Shared contracted indices *are* matrix multiplications; activations are equation attributes (`{:act :gelu}`).
- **Zero Java/Python Escape Hatches**: All tensor operations compile directly into StableHLO MLIR text and execute on native hardware via OpenXLA PJRT (AMD ROCm, Intel SYCL, NVIDIA CUDA, CPU).
- **Physical Consumer Hardware Apparatus**: Evaluates Ali Ghodsi's 4 Recursive Self-Improvement (RSI) gates directly on consumer GPUs (24GB VRAM).

```clojure
;; Define a dense projection block in pure Tensor Logic AST
(def ast
  [[:rms-norm [:h_norm :b :p :d] [:x :b :p :d] [:gamma :d] {:eps 1e-6}]
   [:= [:y :b :p :d] {:act :gelu} [:h_norm :b :p :din] [:w :din :d]]])

;; Compile directly to OpenXLA PJRT executable
(def exec (xla/compile-graph ctx (lower/ast->graph "dense_block" invars ast #{:y})))
```

---

## Quickstart & Verification

### Requirements
- **Java 25+** (`java --version`)
- **Clojure 1.12+** (`clj --version`)

### Run Tests
```bash
# Run fast property & unit test suite (< 3s)
clojure -M:test fast

# Run complete logic test suite
clojure -M:test logic
```

### Run Model Inference
Launch through the wrapper script (which chains `libjsig.so` to prevent signal clashes):
```bash
# Gemma 4 Autoregressive Generation
./tools/gemma4.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"

# Autonomous Agent Loop
./tools/gemma4.sh agent --backend rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and list files"
```

---

## Repository Layout

- `src/einsum/` — Core compiler, tensor logic DSL, Panama FFM bindings, model architectures.
- `src/tools/` — CLI tools, model inference entrypoints, and Hugging Face downloader.
- `src/experiments/` — Executable research code sharded by gate and slug.
- `resources/` — Non-code artifacts: `resources/proposals/`, `resources/catalog/`, `resources/data/`.
- `tools/` — Shell launcher scripts (`gemma4.sh`, `benchmark.sh`, `quantize.sh`).
- `docs/` — Permanent technical knowledge base (architecture, models, XLA substrate, benchmarks).
