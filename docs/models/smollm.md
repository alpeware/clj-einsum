# SmolLM Architecture Specification & StableHLO Graph

- **Status**: **Fully Supported** (SmolLM-135M, SmolLM-360M, SmolLM-1.7B)
- **Clojure Source**: [`models/smollm.clj`](../../models/smollm.clj)

---

## 1. Visual Execution Graph (Pure Clojure $\to$ StableHLO MLIR)

The following Mermaid diagram represents the exact StableHLO execution graph formed by [`smollm-layer-ast`](../../models/smollm.clj) in `clj-einsum`:

```mermaid
flowchart TD
    subgraph Inputs ["StableHLO Input Tensors"]
        X["Input Tensor x [1, seq, dim]"]
        IN_LN["input_layernorm_w [dim]"]
        QW["q_proj_w [q_dim, dim]"]
        KW["k_proj_w [kv_dim, dim]"]
        VW["v_proj_w [kv_dim, dim]"]
        OW["o_proj_w [dim, q_dim]"]
        POST_LN["post_attention_layernorm_w [dim]"]
        GW["gate_proj_w [inter, dim]"]
        UW["up_proj_w [inter, dim]"]
        DW["down_proj_w [dim, inter]"]
        POS["pos [seq]"]
    end

    subgraph Layer ["SmolLM Layer Block"]
        LN1["1. RMSNorm (input_layernorm)"]
        QKV["2. Q, K, V Linear Projections"]
        ROPE["3. Rotary Position Embedding (RoPE)"]
        GQA["4. Grouped-Query Attention (GQA)"]
        O_PROJ["5. Output Projection MatMul"]
        RES1["6. Residual Addition (+ x)"]
        LN2["7. RMSNorm (post_attention_layernorm)"]
        SWIGLU["8. SwiGLU Gated MLP (gate, up, down)"]
        RES2["9. Residual Addition (+ residual_1)"]
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
    O_PROJ --> RES1
    X --> RES1
    RES1 --> LN2
    POST_LN --> LN2
    LN2 --> SWIGLU
    GW --> SWIGLU
    UW --> SWIGLU
    DW --> SWIGLU
    SWIGLU --> RES2
    RES1 --> RES2
    RES2 --> Output["Output Block Tensor [1, seq, dim]"]
```

---

## 2. Hyperparameter Variants

| Hyperparameter | SmolLM-135M | SmolLM-360M | SmolLM-1.7B |
| :--- | :--- | :--- | :--- |
| **Hidden Dimension ($d_{model}$)** | 576 | 960 | 2048 |
| **Layers ($N_{layers}$)** | 30 | 32 | 24 |
| **Query Heads ($H_q$)** | 9 | 15 | 32 |
| **KV Heads ($H_{kv}$)** | 3 | 5 | 32 |
| **Head Dimension ($d_k$)** | 64 | 64 | 64 |
| **Intermediate MLP Dim** | 1536 | 2560 | 5632 |

---

## 3. Verification Protocol

```bash
clojure -M:test -e "(require '[models.smollm-test]) (clojure.test/run-tests 'models.smollm-test)"
```
