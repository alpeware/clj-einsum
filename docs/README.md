# clj-einsum Architectural Documentation & LLM Knowledge Base

Welcome to the **`clj-einsum` Architectural Documentation & Knowledge Base**.

`clj-einsum` is a high-performance numerical computing and deep learning framework for Clojure targeting **Java 25** and **OpenXLA PJRT C API** via the Foreign Function & Memory (FFM) API. It provides pure functional tensor abstractions, graph tracing, automatic differentiation, and hardware acceleration across CPU, Intel SYCL, AMD ROCm, and NVIDIA CUDA devices—with **Zero Java Escape Hatches** (pure XLA compilation).

> For the operational process, milestones (The Staircase), 4 RSI gates, and research protocol, consult [**`PROCESS.md`**](../PROCESS.md).

---

## 🏛️ System Design & Architecture ([`architecture/`](architecture/DESIGN.md))

Core compiler architecture, memory boundaries, and deliberation loops:
- 📐 **[System Design & Architecture](architecture/DESIGN.md)**: Panama FFM bindings, SSA EDN IR, StableHLO codegen, and SHA-256 caching.
- 🔁 **[Tensor-Native Agent Deliberation Loop](architecture/agent_loop.md)**: Three-tier deliberation (reflex contractions, deliberation, discrete synthesis) and verified schema commits.
- 💾 **[Memory Layout, VRAM & LDS Limits](architecture/memory_and_vram.md)**: Host vs. device placement rules, RDNA3 64KB LDS constraints, and `libjsig.so` signal chaining.
- 🔌 **[OpenXLA PJRT C API](architecture/pjrt_c_api.md)**: Panama FFM dynamic symbol resolution and off-heap arena lifecycles.
- 🔬 **[Ali Ghodsi's 4 RSI Gates on Consumer Silicon](architecture/rsi_gates.md)**: Formal apparatus for measuring Resource Efficiency, Time Efficiency, Capability Improvement, and Closed-Loop Repeatability.

---

## 🧠 Declarative Tensor Logic Foundations ([`tensor_logic/`](tensor_logic/README.md))

Theoretical foundations for Pedro Domingos' Declarative Tensor Logic:
- 📖 **[Tensor Logic Overview](tensor_logic/README.md)**: Resolving the autonomous agent triad (hallucinations, online learning, context explosion) in OpenXLA VRAM.
- 📐 **[Theoretical Foundations](tensor_logic/theory.md)**: Logic as tensor contraction (Einstein summation as conjunction/quantification), value-carrying semirings, Datalog fixpoints, and outer-product superposition memory algebra.
- 🌐 **[Related Work & Comparative Analysis](tensor_logic/related_work.md)**: Deep technical comparison with contemporary neuro-symbolic research and open-source implementations.
- 🏛️ **[Pillars & Novelty Assessment](tensor_logic/architectural_pillars_and_novelty.md)**: Structural separation of factual memory and semantic routing.

---

## 🏛️ Model Specifications & Execution Graphs ([`models/`](models/README.md))

Every supported model family includes architectural details, hyperparameter specifications, and StableHLO execution graphs:
- 🟢 **[Gemma 4 (E2B / E4B)](models/gemma4_e2b_e4b.md)** *(Resident VRAM inference & agent loop)*
- 🟢 **[Gemma 3 (1B / 4B / 12B / 27B)](models/gemma3.md)**
- 🟢 **[Gemma 2 (2B / 9B / 27B)](models/gemma2.md)**
- 🟢 **[SmolLM (135M / 360M / 1.7B)](models/smollm.md)**
- 🟢 **[GPT-2 (Small / Medium / Large / XL)](models/gpt2.md)**

---

## ⚙️ OpenXLA & PJRT Knowledge Base ([`xla/`](xla/README.md))

Hardware optimization, execution loops, and low-level PJRT compilation:
- ⚙️ **[OpenXLA & PJRT Hardware Limitations](xla/pjrt_limitations.md)**: FFM struct layout ABI (`PJRT_ExecuteOptions`), signal chaining (`libjsig.so`), and 128-byte memory alignment.
- 🔁 **[In-VRAM Autonomous Agent Loop](xla/agent_vram_loop.md)**: Single-fused `stablehlo.while` execution graph, state tuple representation, and zero-copy transfers.
- 🧩 **[Paged KV-Cache & Long-Context VRAM Allocation](xla/paged_attention_vram.md)**: VRAM math for 256K contexts, PagedAttention block tables in StableHLO, and sliding-window eviction.
- 🔍 **[Debugging, Profiling & Tracing](xla/debugging_profiling_tracing.md)**: Chrome trace generation and performance profiling.

---

## 📊 Hardware Benchmarks ([`benchmarks/`](benchmarks/README.md))

Hardware and driver baseline performance reports:
- 🖥️ **[AMD Desktop Workstation (Radeon RX 7900 XTX 24GB ROCm)](benchmarks/amd_desktop_7900_xtx_rocm.md)**
- 💻 **[Lenovo ThinkPad X1 Carbon Gen 13 (Intel Arc 140V SYCL)](benchmarks/lenovo_x1_carbon_intel_sycl.md)**
