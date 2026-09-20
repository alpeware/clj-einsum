# Gemma 2 Model Architecture Specification & StableHLO Graph

- **Status**: **Fully Supported** (Gemma 2B, Gemma 2 9B, Gemma 2 27B)
- **Clojure Source**: [`models/gemma.clj`](../../models/gemma.clj)
- **Test Suite**: [`test/models/gemma_test.clj`](../../test/models/gemma_test.clj)

---

## 1. Visual Execution Graph (Pure Clojure $\to$ StableHLO MLIR)

The following Mermaid diagram represents the exact StableHLO execution graph formed by [`gemma2-layer-ast`](../../models/gemma.clj) in `clj-einsum`:

```mermaid
flowchart TD
    subgraph Inputs ["StableHLO Input Tensors"]
        X["Input Tensor x [1, seq, 2048]"]
        IN_LN["input_ln_w [2048]"]
        QW["q_w [2048, 2048]"]
        KW["k_w [256, 2048]"]
        VW["v_w [256, 2048]"]
        OW["o_w [2048, 2048]"]
        POST_ATTN["post_attn_ln_w [2048]"]
        PRE_MLP["pre_mlp_ln_w [2048]"]
        POST_MLP["post_mlp_ln_w [2048]"]
        GW["gate_w [16384, 2048]"]
        UW["up_w [16384, 2048]"]
        DW["down_w [2048, 16384]"]
        POS["pos [seq]"]
    end

    subgraph Layer ["Gemma 2 Layer Block"]
        LN1["1. Gemma RMSNorm y = x / RMS(x) * (1.0 + w)"]
        QKV["2. Q, K, V Linear Projections"]
        ROPE["3. Rotary Position Embedding (RoPE)"]
        GQA["4. Grouped-Query Attention (GQA)"]
        O_PROJ["5. Output Projection MatMul"]
        POST_LN1["6. Post-Attention Gemma RMSNorm"]
        RES1["7. Residual Addition (+ x)"]
        PRE_LN2["8. Pre-MLP Gemma RMSNorm"]
        SWIGLU["9. SwiGLU Gated MLP (gate, up, down)"]
        POST_LN2["10. Post-MLP Gemma RMSNorm"]
        RES2["11. Residual Addition (+ residual_1)"]
    end

    X --> LN1
    IN_LN --> LN1
    LN1 --> QKV
    QW --> QKV
    KW --> QKV
    VW --> QKV
    QKV --> ROPE
    POS --> ROPE
    ROPE --> GQA
    GQA --> O_PROJ
    OW --> O_PROJ
    O_PROJ --> POST_LN1
    POST_ATTN --> POST_LN1
    POST_LN1 --> RES1
    X --> RES1
    RES1 --> PRE_LN2
    PRE_MLP --> PRE_LN2
    PRE_LN2 --> SWIGLU
    GW --> SWIGLU
    UW --> SWIGLU
    DW --> SWIGLU
    SWIGLU --> POST_LN2
    POST_MLP --> POST_LN2
    POST_LN2 --> RES2
    RES1 --> RES2
    RES2 --> Output["Output Block Tensor [1, seq, 2048]"]
```

---

## 2. Hyperparameter Variants

| Hyperparameter | Gemma 2B | Gemma 2 9B | Gemma 2 27B |
| :--- | :--- | :--- | :--- |
| **Hidden Dimension ($d_{model}$)** | 2048 | 3584 | 4608 |
| **Layers ($N_{layers}$)** | 18 | 42 | 46 |
| **Query Heads ($H_q$)** | 8 | 16 | 32 |
| **KV Heads ($H_{kv}$)** | 1 | 8 | 16 |
| **Head Dimension ($d_k$)** | 256 | 256 | 128 |

---

## 3. Verification Protocol

```bash
clojure -M:test -e "(require '[models.gemma-test]) (clojure.test/run-tests 'models.gemma-test)"
```
