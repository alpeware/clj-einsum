# Gemma 4 26B-A4B Mixture-of-Experts (MoE) Architecture Specification

- **Status**: **Planned / Backlog**
- **Target Namespace**: `models.gemma4-moe`
- **Target Hardware**: Laptops / Workstations (Intel Arc / Apple M4 36GB+ RAM)

---

## 1. Visual MoE Routing & Expert Dispatch Graph

The following Mermaid diagram represents the planned StableHLO execution graph for Gemma 4 26B-A4B's Sparse MoE layer:

```mermaid
flowchart TD
    subgraph Inputs ["Input Activation"]
        X["Hidden State Tensor x [1, seq, 3072]"]
        ROUTER_W["router_gate_w [8, 3072]"]
    end

    subgraph Router ["Sparse Top-K Expert Router"]
        GATE_LOGITS["1. Router Softmax Logits [1, seq, 8]"]
        TOP_K["2. Top-K Index & Softmax Gate Weighting (k=2)"]
    end

    subgraph Experts ["Parallel Expert Feed-Forward Blocks"]
        EXP0["Expert 0: SwiGLU MLP (4B params)"]
        EXP1["Expert 1: SwiGLU MLP (4B params)"]
        EXP_N["Expert K: SwiGLU MLP (4B params)"]
    end

    subgraph Aggregate ["Weighted Combination"]
        COMBINE["3. Weighted Sum across Top-K Experts"]
        RES["4. Layer Residual Addition"]
    end

    X --> GATE_LOGITS
    ROUTER_W --> GATE_LOGITS
    GATE_LOGITS --> TOP_K
    TOP_K --> EXP0
    TOP_K --> EXP1
    TOP_K --> EXP_N
    EXP0 --> COMBINE
    EXP1 --> COMBINE
    EXP_N --> COMBINE
    COMBINE --> RES
    X --> RES
    RES --> Output["Output MoE Block Tensor [1, seq, 3072]"]
```

---

## 2. Hyperparameter Specifications

| Hyperparameter | Gemma 4 26B-A4B |
| :--- | :--- |
| **Total Parameters** | 25.8 Billion |
| **Active Parameters per Token** | 3.9 Billion |
| **Total Experts** | 8 Experts |
| **Active Experts ($k$)** | 2 Experts per Token |
| **Hidden Dimension ($d_{model}$)** | 3072 |
| **Layers ($N_{layers}$)** | 48 |
