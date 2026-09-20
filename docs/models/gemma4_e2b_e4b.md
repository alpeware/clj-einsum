# Gemma 4 (E2B / E4B) Model Architecture Specification & StableHLO Graph

- **Status**: **Fully Supported** (Gemma 4 E2B, Gemma 4 E4B)
- **Clojure Source**: [`models/gemma.clj`](../../models/gemma.clj)
- **CLI Hardware Runner**: [`tools/gemma4_inference.clj`](../../tools/gemma4_inference.clj)
- **Test Suite**: [`test/models/gemma_test.clj`](../../test/models/gemma_test.clj)

---

## 1. Visual Execution Graph (Pure Clojure $\to$ StableHLO MLIR)

The following Mermaid diagram represents the exact StableHLO execution graph formed by [`gemma4-layer-ast`](../../models/gemma.clj) in `clj-einsum`:

```mermaid
flowchart TD
    subgraph Inputs ["StableHLO Input Tensors"]
        X["Input Tensor x [1, seq, 1536]"]
        IN_LN["input_ln_w [1536]"]
        LS["layer_scalar_w [1]"]
        QW["q_w [2048, 1536]"]
        KW["k_w [256, 1536]"]
        VW["v_w [256, 1536]"]
        OW["o_w [1536, 2048]"]
        QN["q_norm_w [256]"]
        KN["k_norm_w [256]"]
        POST_ATTN["post_attn_ln_w [1536]"]
        PRE_MLP["pre_mlp_ln_w [1536]"]
        POST_MLP["post_mlp_ln_w [1536]"]
        GW["gate_w [6144, 1536]"]
        UW["up_w [6144, 1536]"]
        DW["down_w [1536, 6144]"]
        PLG["per_layer_gate_w [256, 1536]"]
        PLP["per_layer_proj_w [1536, 256]"]
        PLN["post_per_layer_norm_w [1536]"]
        PLIN["per_layer_input [1, seq, 256]"]
        POS["pos [seq]"]
    end

    subgraph Block ["Gemma 4 Transformer Layer Block"]
        LN1["1. Gemma RMSNorm y = x / RMS(x) * (1.0 + w)"]
        QKV["2. Q, K, V Projections + Query/Key RMSNorm"]
        ROPE["3. Rotary Position Embedding (RoPE)"]
        GQA["4. Grouped-Query Attention (GQA)"]
        O_PROJ["5. Output Projection MatMul"]
        POST_LN1["6. Post-Attention Gemma RMSNorm"]
        RES1["7. Residual Addition (+ x * layer_scalar)"]
        PRE_LN2["8. Pre-MLP Gemma RMSNorm"]
        SWIGLU["9. SwiGLU Gated MLP (gate, up, down)"]
        POST_LN2["10. Post-MLP Gemma RMSNorm"]
        PL_GATE["11. Per-Layer Input Residual Gating Projection"]
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
    RES2 --> Output["Output Block Tensor [1, seq, 1536]"]
```

---

## 2. Hyperparameter Variants

| Hyperparameter | Gemma 4 E2B | Gemma 4 E4B |
| :--- | :--- | :--- |
| **Hidden Dimension ($d_{model}$)** | 1536 | 2560 |
| **Layers ($N_{layers}$)** | 22 | 34 |
| **Query Heads ($H_q$)** | 8 | 10 |
| **KV Heads ($H_{kv}$)** | 1 | 2 |
| **Head Dimension ($d_k$)** | 256 | 256 |
| **Per-Layer Gating Dim** | 256 | 256 |

---

## 3. Interactive CLI Generation Command

Run single-batch hardware text generation on Intel Arc GPU (SYCL Level-Zero):
```bash
clojure -M tools/gemma4_inference.clj --model Gemma-4-E2B-it --backend sycl --precision int8
```
