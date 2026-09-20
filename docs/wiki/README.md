# LLM Architecture & Algorithm Reference Wiki

This Wiki serves as an open, educational reference for state-of-the-art LLM architectures, mathematical formulations, and optimization algorithms. Each topic pairs theoretical principles with the pure Clojure / StableHLO MLIR reference implementation in `clj-einsum`.

---

## 📖 Wiki Topics

### 1. ⚡ [Inference: Prefill Phase vs. Decoding Phase ($B=1$)](inference.md)
- Theoretical breakdown of the **Compute-Bound Prompt Prefill Phase** ($L$ tokens in 1 matrix pass) vs. the **Memory-Bandwidth Bound Autoregressive Decoding Phase** ($B=1$ single token steps).
- Impact on arithmetic intensity and GPU tensor core utilization.

### 2. 🔤 [Tokenization & Special Tokens](tokenization.md)
- **Byte-Pair Encoding (BPE)** (GPT-2, SmolLM) vs **SentencePiece / Unigram** (Gemma).
- Vocabulary embedding lookups and special tokens (`<bos>`, `<eos>`, `<turn|>` context control tokens).
- Reference Implementation: [`einsum.runtime.tokenizer.core`](../../src/einsum/runtime/tokenizer/core.clj).

### 3. 🎲 [Autoregressive Sampling & KV Cache Management](sampling.md)
- **Sampling Algorithms**: Temperature scaling, Top-K filtering, Top-P (Nucleus) cumulative thresholding.
- **KV Cache Updates**: Mutating and appending $K_t, V_t$ slices to historical KV cache tensors.
- Reference Implementations: [`einsum.runtime.sampling`](../../src/einsum/runtime/sampling.clj) and [`einsum.runtime.generation.autoregressive`](../../src/einsum/runtime/generation/autoregressive.clj).

### 4. 👁️ [Attention Mechanisms](attention.md)
- **Multi-Head Attention (MHA)**: Standard query-key-value self-attention.
- **Grouped-Query Attention (GQA)**: Shared KV heads for reduced memory bandwidth overhead.
- **Rotary Position Embeddings (RoPE)**: Relative position encoding via complex plane rotations.
- **Gemma 4 Hybrid Sliding Window**: Local window attention eviction combined with global layers.
- **DeepSeek Multi-Head Latent Attention (MLA)**: Low-rank KV cache vector compression.
- **FlashDecoding & Split-K**: Parallel single-query ($B=1$) KV reduction on GPU.
- Reference Implementation: [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

### 5. ⚖️ [Normalization Variants](normalization.md)
- **Standard LayerNorm**: Mean and variance scaling with gain ($\gamma$) and bias ($\beta$).
- **Root Mean Square Normalization (RMSNorm)**: Mean-square variance scaling without mean subtraction.
- **Gemma RMSNorm**: $+1.0$ weight offset variant: $y = \frac{x}{\text{RMS}(x)} \times (1.0 + w)$.
- Reference Implementation: [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

### 6. ⚡ [Activation Functions & Gated MLPs](activations.md)
- **GELU / SiLU / ReLU**: Standard non-linearities.
- **SwiGLU & GeGLU**: Gated Linear Unit feed-forward sub-blocks.
- Reference Implementation: [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

### 7. 🔮 [Speculative Decoding Algorithms](speculative_decoding.md)
- **Draft-Model Speculative Decoding**: Autoregressive candidate proposal with target verification.
- **EAGLE / EAGLE-2 & Medusa**: Multi-head / feature-level speculation without separate draft models.
- **Meta DFlash Block Speculation**: 16-token parallel block prediction.
- **Prompt-Lookup / N-gram Speculation**: Contextual string matching for zero-overhead candidate draft proposals.

### 8. 📦 [Low-Bit Quantization](quantization.md)
- **Post-Training Quantization (PTQ)**: INT8, INT4, AWQ, and GPTQ.
- **In-Graph De-quantization**: Fusing integer-to-float weight conversion with GEMM matrix multiplication in StableHLO MLIR.
- Reference Implementation: [`einsum.runtime.safetensors`](../../src/einsum/runtime/safetensors.clj).

### 9. ⚙️ [OpenXLA PJRT Hardware & In-VRAM Agent Execution](../xla/README.md)
- **Hardware Limitations**: Struct ABI alignment, signal interposition, 128-byte alignment, 32-bit compiler lowerings ([`pjrt_limitations.md`](../xla/pjrt_limitations.md)).
- **In-VRAM Agent Loop**: Single-fused `stablehlo.while` execution graph, state tuple representation ([`agent_vram_loop.md`](../xla/agent_vram_loop.md)).
- **Paged Attention & Memory Math**: StableHLO block tables, sliding-window eviction, and FP8/INT8 quantized KV-caches ([`paged_attention_vram.md`](../xla/paged_attention_vram.md)).
