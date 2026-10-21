# clj-einsum Architectural Documentation & Knowledge Base

Welcome to the **`clj-einsum` Architectural Documentation & Knowledge Base**.

`clj-einsum` is a high-performance deep learning compiler framework and hardware runtime in pure Clojure targeting **Java 25+** and **consumer AMD ROCm (`gfx1100` / AMD Radeon RX 7900 XTX 24 GB)** via Project Panama Foreign Function & Memory (FFM) API (`java.lang.foreign`). It compiles Pedro Domingos' Declarative Tensor Logic into pure-Clojure UOp schedules, rendering native Wave32 BF16 WMMA tensor core kernels dispatched directly to hardware with zero C++ glue libraries, zero Python, and zero external ML compilers.

> For the operational process, milestones (The Staircase), 4 RSI gates, and research protocol, consult [**`PROCESS.md`**](../PROCESS.md).

---

## 🏛️ System Design & Architecture ([`architecture/`](architecture/DESIGN.md))

Core compiler pipeline, memory boundaries, and deliberation loops:
- 📐 **[System Design & Architecture](architecture/DESIGN.md)**: 3-layer architecture (Declarative Logic $\to$ Pure-Clojure UOp Compiler $\to$ Panama FFM Device Runtime & SIMD Reference Interpreter).
- ⚡ **[UOp Compiler Architecture Guide](architecture/uop_compiler.md)**: End-to-end pipeline (AST $\to$ lower $\to$ rewrite $\to$ schedule $\to$ rangeify $\to$ opt $\to$ render $\to$ Panama runtime), hash-consed IR, realization boundaries, RDNA3 Wave32 BF16 WMMA tensor core optimizations, and zero-allocation runtime.
- 💾 **[Memory Layout, VRAM & LDS Limits](architecture/memory_and_vram.md)**: Mechanical sympathy, host vs. device placement, UOp VRAM arena management (`einsum.runtime.arena`), ping-pong step buffers, >4 GiB pinned host registration (`hipHostRegister`), and RDNA3 64 KB LDS budgets.
- 🔁 **[Tensor-Native Agent Deliberation Loop](architecture/agent_loop.md)**: Three-tier deliberation (reflex contractions, deliberation, discrete synthesis) and verified schema commits.
- 🔬 **[Ali Ghodsi's 4 RSI Gates on Consumer Silicon](architecture/rsi_gates.md)**: Formal apparatus for measuring Resource Efficiency, Time Efficiency, Capability Improvement, and Closed-Loop Repeatability on consumer hardware.

---

## 🏛️ Model Specifications & Execution Graphs ([`models/`](models/README.md))

Model architectures, hyperparameter specifications, and pure-Clojure UOp execution graphs:
- 🟢 **[Gemma 4 (E2B / E4B / 31B INT4)](models/gemma4.md)**: Resident VRAM inference, fused in-register INT4 group-128 dequantization (16.95 GB VRAM for 31B), reconvergent epilogue GeGLU fusion, and autonomous agent loop.
- 📖 **[Supported Models Directory](models/README.md)**: Architecture specifications for Gemma 4 (E2B, E4B, 31B INT4), Gemma 3, GPT-2, and SmolLM in `src/einsum/models/`.

---

## 🧠 Declarative Tensor Logic Foundations ([`tensor_logic/`](tensor_logic/README.md))

Theoretical foundations for Pedro Domingos' Declarative Tensor Logic:
- 📖 **[Tensor Logic Overview](tensor_logic/README.md)**: Resolving the autonomous agent triad (hallucinations, online learning, context explosion) in pure-Clojure UOp VRAM.
- 📐 **[Theoretical Foundations](tensor_logic/theory.md)**: Logic as tensor contraction (Einstein summation as conjunction/quantification), value-carrying semirings, Datalog fixpoints, and outer-product superposition memory algebra.
- 🏛️ **[Pillars & Novelty Assessment](tensor_logic/architectural_pillars_and_novelty.md)**: Structural separation of factual memory and semantic routing.
- 🌐 **[Related Work & Comparative Analysis](tensor_logic/related_work.md)**: Deep technical comparison with contemporary neuro-symbolic research and open-source implementations.

---

## 💎 Golden Baselines & Historical Benchmark Archive ([`resources/golden/`](../resources/golden/README.md))

Validation baselines, continuous regression oracles, and archived vs-XLA benchmark datasets:
- 🏆 **[Post-XLA Regression Oracle & Benchmark Archive](../resources/golden/README.md)**: Continuous numerical parity and greedy token identity oracle ([`gemma4_uop_oracle.edn`](../resources/golden/gemma4_uop_oracle.edn)), and archived post-excision historical benchmark dataset ([`xla_vs_uop_archive.edn`](../resources/golden/xla_vs_uop_archive.edn)).
