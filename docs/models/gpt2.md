# GPT-2 Model Architecture Specification & StableHLO Graph

- **Status**: **Fully Supported**
- **Clojure Source**: [`models/gpt2.clj`](../../models/gpt2.clj)
- **Test Suite**: [`test/models/gpt2_test.clj`](../../test/models/gpt2_test.clj)

---

## 1. Visual Execution Graph (Pure Clojure $\to$ StableHLO MLIR)

The following Mermaid diagram represents the exact StableHLO execution graph formed by [`gpt2-layer-ast`](../../models/gpt2.clj) lowered via `einsum.logic.lower` in `clj-einsum`:

```mermaid
flowchart TD
    subgraph Inputs ["StableHLO Input Tensors"]
        X["Input Tensor x [1, seq, 768]"]
        LN1_G["ln1_g [768]"]
        LN1_B["ln1_b [768]"]
        C_ATTN_W["c_attn_w [768, 2304]"]
        C_ATTN_B["c_attn_b [2304]"]
        C_PROJ_W["c_proj_w [768, 768]"]
        C_PROJ_B["c_proj_b [768]"]
        LN2_G["ln2_g [768]"]
        LN2_B["ln2_b [768]"]
        MLP_FC_W["mlp_fc_w [768, 3072]"]
        MLP_FC_B["mlp_fc_b [3072]"]
        MLP_PROJ_W["mlp_proj_w [3072, 768]"]
        MLP_PROJ_B["mlp_proj_b [768]"]
    end

    subgraph Block ["GPT-2 Transformer Layer Block"]
        LN1["1. Pre-LayerNorm (ln1_g, ln1_b)"]
        QKV["2. QKV Projection MatMul (c_attn_w, c_attn_b)"]
        ATTN["3. Causal Multi-Head Self Attention"]
        PROJ1["4. Output Projection MatMul (c_proj_w, c_proj_b)"]
        RES1["5. Residual Addition (+ x)"]
        LN2["6. Pre-LayerNorm (ln2_g, ln2_b)"]
        FC["7. MLP FC Projection MatMul (mlp_fc_w, mlp_fc_b)"]
        GELU["8. GELU Non-linearity"]
        PROJ2["9. MLP Proj MatMul (mlp_proj_w, mlp_proj_b)"]
        RES2["10. Residual Addition (+ residual_1)"]
    end

    X --> LN1
    LN1_G --> LN1
    LN1_B --> LN1
    LN1 --> QKV
    C_ATTN_W --> QKV
    C_ATTN_B --> QKV
    QKV --> ATTN
    ATTN --> PROJ1
    C_PROJ_W --> PROJ1
    C_PROJ_B --> PROJ1
    PROJ1 --> RES1
    X --> RES1
    RES1 --> LN2
    LN2_G --> LN2
    LN2_B --> LN2
    LN2 --> FC
    MLP_FC_W --> FC
    MLP_FC_B --> FC
    FC --> GELU
    GELU --> PROJ2
    MLP_PROJ_W --> PROJ2
    MLP_PROJ_B --> PROJ2
    PROJ2 --> RES2
    RES1 --> RES2
    RES2 --> Output["Output Block Tensor [1, seq, 768]"]
```

---

## 2. Hyperparameter Variants

| Hyperparameter | GPT-2 Small | GPT-2 Medium | GPT-2 Large | GPT-2 XL |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | 124M | 355M | 774M | 1.5B |
| **Hidden Dimension ($d_{model}$)** | 768 | 1024 | 1280 | 1600 |
| **Layers ($N_{layers}$)** | 12 | 24 | 36 | 48 |
| **Attention Heads ($H$)** | 12 | 16 | 20 | 25 |
| **Vocabulary Size ($V$)** | 50,257 | 50,257 | 50,257 | 50,257 |

---

## 3. Verification & Benchmark Protocol

* **Generative & Property Test Suite**:
  ```bash
  clojure -M:test -e "(require '[models.gpt2-test]) (clojure.test/run-tests 'models.gpt2-test)"
  ```
* **Benchmark Execution**: Measured in [`tools/benchmark.clj`](../../tools/benchmark.clj) as `gpt2-block`.
