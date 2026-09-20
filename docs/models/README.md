# Model Architecture Specifications & Mermaid Execution Graphs

This directory contains individual architectural specification documents and visual Mermaid graph representations for all supported and planned model families in `clj-einsum`.

---

## 🟢 Supported Models (Production-Ready Codebase & Test Verification)

1. **[GPT-2 (Small / Medium / Large / XL)](gpt2.md)**
   - Classic causal autoregressive decoder with Multi-Head Attention (MHA) and Conv1D projections.
   - Code: [`models.gpt2`](../../models/gpt2.clj).

2. **[SmolLM (135M / 360M / 1.7B)](smollm.md)**
   - Lightweight edge models featuring Grouped-Query Attention (GQA) and SwiGLU activations.
   - Code: [`models.smollm`](../../models/smollm.clj).

3. **[Gemma 2 (2B / 9B / 27B)](gemma2.md)**
   - Gemma RMSNorm ($+1.0$ weight offset), GQA, and SwiGLU MLP blocks.
   - Code: [`models.gemma`](../../models/gemma.clj).

4. **[Gemma 3 (1B / 4B / 12B / 27B)](gemma3.md)**
   - Compact multi-modal ready architectures with 1B lightweight edge configuration.
   - Code: [`models.gemma3`](../../models/gemma3.clj).

5. **[Gemma 4 (E2B / E4B)](gemma4_e2b_e4b.md)**
   - Gemma 4 architecture with per-layer scalar gating projections, Gemma RMSNorm, and GQA.
   - Code: [`models.gemma`](../../models/gemma.clj) & CLI [`tools/gemma4_inference.clj`](../../tools/gemma4_inference.clj).

---

## 🟡 Planned Backlog Models (Specs & Roadmap)

6. **[Gemma 4 26B-A4B Sparse MoE](gemma4_26b_a4b.md)**
   - Sparse Mixture-of-Experts (MoE) architecture with 26B total parameters and 4B active parameters per token.

7. **[Meta Muse-Glimmer-30B](muse_glimmer_30b.md)**
   - Multimodal agentic model featuring DFlash 16-token parallel block speculative decoding, ~1.8B ViT-G/14 vision encoder, and ATEM tool markup.
