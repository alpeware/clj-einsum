# clj-einsum Model & Feature Backlog Matrix

This backlog maintains a clear differentiation between **currently supported models/features** and **upcoming planned models/features**.

---

## 🟢 Currently Supported (Production-Ready in `clj-einsum`)

| Model Family / Feature | Target Architecture | Specification & Graph | Clojure Implementation Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **GPT-2 (Small/Med/Large/XL)** | Decoder Causal LM (MHA) | [Spec & Graph](../models/gpt2.md) | [`models.gpt2`](../../models/gpt2.clj) | **Verified** (CPU, SYCL, ROCm) |
| **SmolLM (135M/360M/1.7B)** | Lightweight Edge LM (GQA/SwiGLU) | [Spec & Graph](../models/smollm.md) | [`models.smollm`](../../models/smollm.clj) | **Verified** (CPU, SYCL) |
| **Gemma 2 (2B/9B/27B)** | GQA, Gemma RMSNorm ($1+w$) | [Spec & Graph](../models/gemma2.md) | [`models.gemma`](../../models/gemma.clj) | **Verified** (CPU, SYCL) |
| **Gemma 3 (1B/4B/12B/27B)** | Gemma 3 Architecture | [Spec & Graph](../models/gemma3.md) | [`models.gemma3`](../../models/gemma3.clj) | **Verified** (CPU, SYCL) |
| **Gemma 4 (E2B/E4B)** | Gemma 4 Per-Layer Gating | [Spec & Graph](../models/gemma4_e2b_e4b.md) | [`models.gemma`](../../models/gemma.clj) | **Verified** (CPU, Intel Arc SYCL) |
| **Safetensors INT8 Quantization** | Header JSON + Tensor Loading | [`einsum.runtime.safetensors`](../../src/einsum/runtime/safetensors.clj) | [`einsum.runtime.safetensors`](../../src/einsum/runtime/safetensors.clj) | **Verified** |
| **SOTA Multi-Backend Benchmarks** | Profiling Engine & CLI | [Overview](../../README.md) | [`tools.benchmark`](../../tools/benchmark.clj) | **Verified** (CPU, SYCL, ROCm) |

---

## 🟡 Planned Backlog (Upcoming Features)

| Target Feature | Specification & Plan | Priority | Target Namespace | Key Additions |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek-V3 / MLA** | Architecture Spec | **High** | `einsum.logic.nn` | Multi-Head Latent Attention (576-dim latent vector compression) |
| **In-Graph INT4 / FP4 Dequant** | Quantization Spec | **High** | `einsum.logic.nn` | StableHLO in-graph de-quantization fused into GEMM kernels |
| **EAGLE-2 Speculative Decoding** | Speculative Decoding Spec | **Medium** | `einsum.runtime.generation` | Tree-structured candidate draft verification pass |
