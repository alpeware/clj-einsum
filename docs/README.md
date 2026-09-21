# clj-einsum Architectural Documentation & LLM Knowledge Base

Welcome to the **`clj-einsum` Architectural Documentation & LLM Knowledge Base**.

`clj-einsum` is a high-performance, pure Clojure numerical computing and deep learning framework built on **OpenXLA** and **StableHLO MLIR** via the Java Panama Foreign Function & Memory (FFM) API. It provides pure functional tensor abstractions, graph tracing, automatic differentiation, and hardware acceleration across CPU, Intel SYCL, AMD ROCm, and NVIDIA CUDA devices—with **Zero Java Escape Hatches** (Pure XLA compilation).

---

## 💡 Why OpenXLA & `clj-einsum`?

1. **Pedro Domingos' Declarative Tensor Logic**: Neural network architectures are defined as homoiconic, relational AST data structures using [`einsum.logic.*`](../src/einsum/logic/core.clj) and lowered directly into StableHLO SSA graphs.
2. **StableHLO MLIR Codegen**: Graphs are serialized into standard StableHLO MLIR text representation via [`einsum.compiler.stablehlo`](../src/einsum/compiler/stablehlo.clj#L20).
3. **Multi-Backend Portability**: A single Clojure model definition compiles seamlessly to native CPU binaries (`libpjrt_cpu.so`), Intel GPU Level-Zero (`libpjrt_sycl.so`), AMD ROCm (`libpjrt_rocm.so`), and NVIDIA CUDA (`libcudart.so`) via [`einsum.core/init-backend!`](../src/einsum/core.clj#L45).
4. **Kernel Fusion & Hardware Acceleration**: OpenXLA automatically fuses elementwise operations, normalizations, and GEMM matrix multiplications into hardware tensor-core kernels (Intel XMX, AMD Matrix Cores, NVIDIA Tensor Cores).

---

## 🧠 Declarative Tensor Logic & In-Tensor Relational Memory ([`tensor_logic/`](tensor_logic/README.md))

Comprehensive theoretical foundations, empirical results, related work analysis, and consumer-hardware experimental roadmap for unifying LLMs with symbolic Knowledge Bases via Pedro Domingos' Declarative Tensor Logic:

- 📖 **[Tensor Logic Overview & The Agent Triad Crisis](tensor_logic/README.md)**: Resolving hallucinations, online learning (zero backprop), and long-horizon context explosion in OpenXLA VRAM.
- 📐 **[Theoretical Foundations](tensor_logic/theory.md)**: Logic as tensor contraction (Einstein summation as conjunction/quantification), value-carrying semirings (Boolean, Continuous, Tropical, Softmax), Datalog fixpoints, and outer-product superposition memory algebra.
- 🔬 **[Empirical Experiments & Diagnostic Journey](tensor_logic/empirical_journey.md)**: Complete chronicle of experiments with Gemma 4 E2B on AMD ROCm (RX 7900 XTX), detailing the progression from de-oracled baseline to span-pooling, learned linear probes, LLM-anchored token embeddings, and QR-orthonormalization.
- 🌐 **[Related Work & Comparative Analysis](tensor_logic/related_work.md)**: Deep technical comparison with `waylandzhang/tensorlogic`, `pedronahum/tl-pjrt`, Modern Continuous Hopfield Networks, Fast Weights, and RAG pipelines.
- 🚀 **[Strategic Research Roadmap for Consumer Hardware](tensor_logic/future_experiments.md)**: Concrete proposals for Cross-Attention Memory Probes (E1), KG-Masked Attention (E2), Contrastive Subspace Pre-training (E3), In-VRAM Datalog Fixpoint State Trackers (E4), Zero-Gradient Fast Weights (E5), and the TL-Transformer architecture.

---

Empirical benchmark metrics for specific hardware and driver combinations are recorded in the dedicated **[`benchmarks/`](benchmarks/README.md)** directory:

- 💻 **[Lenovo ThinkPad X1 Carbon Gen 13 (Intel Arc 140V SYCL)](benchmarks/lenovo_x1_carbon_intel_sycl.md)**: Intel Core Ultra Series 2 Lunar Lake + Intel Arc 140V iGPU via SYCL Level-Zero V2 (`26.22.038646`). Includes Python JAX vs. `clj-einsum` performance gap analysis.
- 🖥️ **[AMD Desktop Workstation (Radeon RX 7900 XTX 24G ROCm)](benchmarks/amd_desktop_7900_xtx_rocm.md)**: AMD Ryzen CPU + AMD Radeon RX 7900 XTX 24GB VRAM RDNA3 via ROCm `7.2.0` / `6.0`.

---

## 📚 LLM Reference Wiki

Explore theoretical formulations paired with pure Clojure reference code:

- ⚡ **[Inference: Prefill vs. Decoding Phase](wiki/inference.md)**: Compute-bound context prefill vs memory-bound single token steps ($B=1$).
- 🔤 **[Tokenization & Special Tokens](wiki/tokenization.md)**: BPE vs SentencePiece algorithms and token lookups.
- 🎲 **[Autoregressive Sampling & KV Cache](wiki/sampling.md)**: Top-K, Top-P, Temperature scaling, and KV cache updates.
- 👁️ **[Attention Mechanisms](wiki/attention.md)**: MHA, GQA, RoPE, Gemma 4 Hybrid Sliding Window, DeepSeek MLA, FlashDecoding.
- ⚖️ **[Normalization Variants](wiki/normalization.md)**: LayerNorm, RMSNorm, and Gemma RMSNorm ($1.0 + w$).
- ⚡ **[Activation Functions](wiki/activations.md)**: GELU, SiLU, SwiGLU, GeGLU feed-forward blocks.
- 🔮 **[Speculative Decoding](wiki/speculative_decoding.md)**: Draft models, EAGLE-2, Medusa, Meta DFlash, and N-gram Prompt-Lookup.
- 📦 **[Low-Bit Quantization](wiki/quantization.md)**: INT8, INT4, AWQ, and StableHLO in-graph de-quantization.

---

## ⚙️ OpenXLA & PJRT Hardware Knowledge Base ([`xla/`](xla/README.md))

Technical specifications for long-running autonomous AI agent loops, OpenXLA hardware limits, and VRAM memory optimization algorithms:

- ⚙️ **[OpenXLA & PJRT Hardware Limitations](xla/pjrt_limitations.md)**: FFM struct layout ABI (`PJRT_ExecuteOptions`), signal chaining (`libjsig.so`), 128-byte hardware memory alignment, and 32-bit `dynamic_update_slice` compiler lowerings.
- 🔁 **[In-VRAM Autonomous Agent Execution Loop](xla/agent_vram_loop.md)**: Single-fused `stablehlo.while` execution graph, state tuple representation, in-graph sampling, and zero-copy direct device memory transfers.
- 🧩 **[Paged KV-Cache & Long-Context VRAM Allocation](xla/paged_attention_vram.md)**: VRAM math for 256K contexts, PagedAttention block tables in StableHLO, sliding-window eviction, and FP8/INT8 in-graph quantized KV-caches.

---

## 🏛️ Model Specifications Index & Execution Graphs

Every supported model family includes architectural details, hyperparameter specifications, and visual Mermaid diagrams of its StableHLO execution graph:

- 🟢 **[GPT-2 (Small / Medium / Large / XL)](models/gpt2.md)** *(Supported)*
- 🟢 **[SmolLM (135M / 360M / 1.7B)](models/smollm.md)** *(Supported)*
- 🟢 **[Gemma 2 (2B / 9B / 27B)](models/gemma2.md)** *(Supported)*
- 🟢 **[Gemma 3 (1B / 4B / 12B / 27B)](models/gemma3.md)** *(Supported)*
- 🟢 **[Gemma 4 (E2B / E4B)](models/gemma4_e2b_e4b.md)** *(Supported)*
- 🟡 **[Gemma 4 26B-A4B MoE](models/gemma4_26b_a4b.md)** *(Planned)*
- 🟡 **[Meta Muse-Glimmer-30B](models/muse_glimmer_30b.md)** *(Planned)*

---

## 🗺️ Backlog & Full-Stack Roadmap

- 📋 **[Model & Feature Backlog](roadmap/model_backlog.md)**: Matrix differentiating supported vs planned models and features.
- 🚀 **[Full-Stack Consumer Hardware Roadmap](roadmap/full_stack_roadmap.md)**: Vision spanning Inference $\to$ Distillation & Evals $\to$ SFT/LoRA Fine-tuning $\to$ Pre-training.
