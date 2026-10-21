# Gemma 4 (E2B / E4B / 31B INT4) Model Architecture Specification & UOp Graph

- **Status**: **Fully Supported on Consumer Hardware (`gfx1100` / AMD Radeon RX 7900 XTX 24 GB)**
- **Model Variants**:
  - **Gemma 4 E2B**: `BF16` (2.3B parameters, 22 layers, 4.6 GB VRAM)
  - **Gemma 4 E4B**: `BF16` (4.3B parameters, 34 layers, 8.6 GB VRAM)
  - **Gemma 4 31B**: `W4A16 INT4` group-128 (31B parameters, 60 layers, **16.95 GB VRAM**)
- **Clojure Sources**:
  - High-level Declarative Logic AST: [`src/einsum/models/gemma.clj`](../../src/einsum/models/gemma.clj)
  - Pure-Clojure UOp Kernels: [`src/einsum/models/gemma4/kernels.clj`](../../src/einsum/models/gemma4/kernels.clj)
  - Native Device Runtime: [`src/einsum/models/gemma4/runtime.clj`](../../src/einsum/models/gemma4/runtime.clj)
- **CLI Runner**: [`tools/gemma4.sh`](../../tools/gemma4.sh)
- **Test Suite**: [`test/einsum/models/gemma4_test.clj`](../../test/einsum/models/gemma4_test.clj) & [`test/einsum/uop/golden_test.clj`](../../test/einsum/uop/golden_test.clj)

---

## 1. Visual Execution Graph (Declarative Logic $\to$ Pure-Clojure UOp Schedule)

The following Mermaid diagram represents the pure-Clojure UOp execution graph compiled from `gemma4-layer-ast` in `clj-einsum`:

```mermaid
flowchart TD
    subgraph Inputs ["Layer Input Buffers"]
        X["Hidden States x [1, M, D]"]
        IN_LN["input_ln_w [D]"]
        LS["layer_scalar_w [1]"]
        QW["q_w (BF16 or INT4 group-128)"]
        KW["k_w (BF16 or INT4 group-128)"]
        VW["v_w (BF16 or INT4 group-128)"]
        OW["o_w (BF16 or INT4 group-128)"]
        QN["q_norm_w [D_k]"]
        KN["k_norm_w [D_k]"]
        POST_ATTN["post_attn_ln_w [D]"]
        PRE_MLP["pre_mlp_ln_w [D]"]
        POST_MLP["post_mlp_ln_w [D]"]
        GW["gate_w (BF16 or INT4 group-128)"]
        UW["up_w (BF16 or INT4 group-128)"]
        DW["down_w (BF16 or INT4 group-128)"]
        PLG["per_layer_gate_w [D_pl, D]"]
        PLP["per_layer_proj_w [D, D_pl]"]
        PLN["post_per_layer_norm_w [D]"]
        PLIN["per_layer_input [1, M, D_pl]"]
        POS["pos [M]"]
    end

    subgraph UOpBlock ["Pure-Clojure UOp Layer Schedule"]
        LN1["1. Gemma RMSNorm<br/>y = x / RMS(x) * (1.0 + w)"]
        QKV["2. Q, K, V Projections<br/>(Fused In-Register INT4 Unpack + Wave32 WMMA)"]
        ROPE["3. Rotary Position Embeddings (RoPE)"]
        GQA["4. Grouped-Query Attention (GQA)<br/>(Streaming Softmax M <= 16 or Direct 4D M > 16)"]
        O_PROJ["5. Output Projection MatMul"]
        POST_LN1["6. Post-Attention Gemma RMSNorm"]
        RES1["7. Residual Addition (+ x * layer_scalar)"]
        PRE_LN2["8. Pre-MLP Gemma RMSNorm"]
        SWIGLU["9. Gated MLP (gate, up, down)<br/>(Reconvergent Epilogue GeGLU Fusion)"]
        POST_LN2["10. Post-MLP Gemma RMSNorm"]
        PL_GATE["11. Per-Layer Input Residual Gating Projection (PLE)"]
        RES2["12. Final Layer Residual Addition"]
    end

    X --> LN1
    IN_LN --> LN1
    LN1 --> QKV
    QW --> QKV
    KW --> QKV
    VW --> QKV
    QN --> QKV
    KN --> QKV
    QKV --> ROPE
    POS --> ROPE
    ROPE --> GQA
    GQA --> O_PROJ
    OW --> O_PROJ
    O_PROJ --> POST_LN1
    POST_ATTN --> POST_LN1
    POST_LN1 --> RES1
    X --> RES1
    LS --> RES1
    RES1 --> PRE_LN2
    PRE_MLP --> PRE_LN2
    PRE_LN2 --> SWIGLU
    GW --> SWIGLU
    UW --> SWIGLU
    DW --> SWIGLU
    SWIGLU --> POST_LN2
    POST_MLP --> POST_LN2
    POST_LN2 --> PL_GATE
    PLG --> PL_GATE
    PLP --> PL_GATE
    PLN --> PL_GATE
    PLIN --> PL_GATE
    PL_GATE --> RES2
    RES1 --> RES2
    RES2 --> Output["Output Hidden States [1, M, D]"]
```

---

## 2. Hyperparameter Specifications

| Hyperparameter | Gemma 4 E2B | Gemma 4 E4B | Gemma 4 31B INT4 |
| :--- | :--- | :--- | :--- |
| **Precision** | `BF16` | `BF16` | **`W4A16 INT4` (group-128)** |
| **Parameters** | 2.3B | 4.3B | **30.7B** |
| **Total VRAM Footprint** | **4.6 GB** | **8.6 GB** | **16.95 GB** (Fits in 24 GB) |
| **Hidden Dimension ($d_{model}$)** | 1536 | 2560 | 5120 |
| **Layers ($N_{layers}$)** | 22 | 34 | 60 |
| **Query Heads ($H_q$)** | 8 | 10 | 32 |
| **KV Heads ($H_{kv}$)** | 1 | 2 | 8 |
| **Head Dimension ($d_k$)** | 256 | 256 | 128 |
| **Intermediate Size** | 6144 | 10240 | 16384 |
| **Per-Layer Gating Dim ($D_{pl}$)**| 256 | 256 | 512 |
| **Vocabulary Size** | 256,000 | 256,000 | 256,000 |

---

## 3. Key UOp Optimizations for Gemma 4

1. **Fused In-Register INT4 Dequantization (31B)**:
   - Quantized weights are stored in global VRAM as packed 4-bit nibbles with group-128 scales.
   - Unpacking arithmetic (nibble isolation, zero-point subtraction, scale multiplication) executes inside GPU thread registers directly before Wave32 WMMA tensor operations.
   - Eliminates 30+ GB of intermediate FP32 weight allocations in VRAM.
2. **Reconvergent Epilogue GeGLU Fusion**:
   - The gated activation $y = \text{gate}(x) \cdot \text{gelu}(\text{up}(x))$ is folded directly into the matrix multiplication reduction loop via `find-reconvergent-epilogue-subchain`.
   - Avoids writing intermediate projection tensors back to global memory.
3. **Static-Shape Ping-Pong KV Cache**:
   - Pre-allocates fixed-shape KV buffers within the session arena (`einsum.runtime.arena`).
   - KV states alternate between dual ping-pong buffers per token, completely eliminating GC allocations and graph recompilation.

---

## 4. Interactive CLI Generation Commands

All Gemma 4 inference runs via the unified [`tools/gemma4.sh`](../../tools/gemma4.sh) wrapper:

```bash
# Gemma 4 E2B (BF16, 2.3B) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"

# Gemma 4 E4B (BF16, 4.3B) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E4B-it --prompt "Explain monads in Clojure"

# Gemma 4 31B (W4A16 INT4, 16.95 GB VRAM) on ROCm
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-31b-it-int4 --prompt "Explain monads in Clojure"

# Autonomous Coding Agent Loop
./tools/gemma4.sh agent --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and list files"
```
