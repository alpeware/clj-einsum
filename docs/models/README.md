# Model Architecture Specifications & Execution Graphs

This directory contains individual architectural specifications and visual execution graphs for model families supported in `clj-einsum`.

All models are expressed in Pedro Domingos' Declarative Tensor Logic (`einsum.logic.*`) and compile directly into pure-Clojure UOp schedules (`einsum.uop.*`) for native execution via Project Panama FFM on consumer hardware (AMD Radeon RX 7900 XTX 24 GB) and host CPU.

---

## 🟢 Supported Models (`src/einsum/models/`)

1. **[Gemma 4 (E2B / E4B / 31B INT4)](gemma4.md)**
   - Flagship modern LLM family fully supported on consumer AMD ROCm (`gfx1100`).
   - **Gemma 4 E2B** (`BF16`, 2.3B parameters, 22 layers, 4.6 GB VRAM)
   - **Gemma 4 E4B** (`BF16`, 4.3B parameters, 34 layers, 8.6 GB VRAM)
   - **Gemma 4 31B** (`W4A16 INT4` group-128, 60 layers, **16.95 GB VRAM** footprint fitting entirely inside 24 GB VRAM)
   - Features: Gemma RMSNorm ($+1.0$ offset), GQA, RoPE, PLE per-layer scalar gating projections, fused in-register INT4 group-128 dequantization, reconvergent epilogue GeGLU fusion, and static-shape ping-pong KV cache buffers.
   - Code: [`einsum.models.gemma`](../../src/einsum/models/gemma.clj), [`einsum.models.gemma4.kernels`](../../src/einsum/models/gemma4/kernels.clj), [`einsum.models.gemma4.runtime`](../../src/einsum/models/gemma4/runtime.clj).
   - CLI: [`tools/gemma4.sh`](../../tools/gemma4.sh).

2. **Gemma 3 (1B / 4B / 12B / 27B)**
   - Compact multi-modal ready architecture with 1B lightweight edge configuration.
   - Code: [`einsum.models.gemma3`](../../src/einsum/models/gemma3.clj).

3. **GPT-2 (Small / Medium / Large / XL)**
   - Classic causal autoregressive decoder with Multi-Head Attention (MHA) and Conv1D projections.
   - Code: [`einsum.models.gpt2`](../../src/einsum/models/gpt2.clj).

4. **SmolLM (135M / 360M / 1.7B)**
   - Lightweight edge models featuring Grouped-Query Attention (GQA) and SwiGLU activations.
   - Code: [`einsum.models.smollm`](../../src/einsum/models/smollm.clj).
