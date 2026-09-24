# Declarative Tensor Logic & Relational Memory for LLMs

Welcome to the **Declarative Tensor Logic & In-Tensor Relational Memory** knowledge base of `clj-einsum`.

This documentation suite establishes the theoretical foundations, empirical results, architectural comparisons, and long-term research roadmap for unifying Large Language Models (LLMs) with symbolic Knowledge Bases (KBs) via **Pedro Domingos' Declarative Tensor Logic**.

All tensor contractions, relational memory updates, and neural layers execute via **OpenXLA** and **StableHLO MLIR** on consumer hardware (AMD ROCm RDNA3, Intel SYCL, NVIDIA CUDA, and CPU) with **Zero Java Escape Hatches**.

---

## 🎯 The Core Motivation: The LLM Triad Crisis

State-of-the-art autoregressive Transformers face three interrelated fundamental crises when deployed in autonomous, long-horizon agentic environments:

```mermaid
flowchart TD
    subgraph Triad ["The Autonomous Agent Triad Crisis"]
        H["1. Hallucinations & Factual Drift<br/><i>(Unbounded probabilistic sampling)</i>"]
        OL["2. Catastrophic Forgetting & No Online Learning<br/><i>(Static weights, costly gradient updates)</i>"]
        LH["3. Long-Horizon Context Explosion<br/><i>(O(N²) attention, state drift over 100+ turns)</i>"]
    end

    TL["Pedro Domingos' Declarative Tensor Logic<br/>& In-Tensor Relational Memory"]

    TL -->|Crisp Deductive Gating & In-Graph Verification| H
    TL -->|Zero-Gradient Fast Weights O(1) Outer Products| OL
    TL -->|Compact Resident Relational Cores O(1) Context| LH
```

1. **Hallucinations & Factual Unreliability**:
   Autoregressive generation samples tokens according to probabilistic language statistics rather than deductive truth. When queried on relational facts, models interpolate over training distributions, yielding fluent but factually incorrect assertions.
2. **Inability to Learn Online (Without Backpropagation)**:
   Once compiled and deployed, updating model knowledge requires either prompt engineering (which burns finite context tokens and degrades over time) or parameter fine-tuning (LoRA/SFT), which is slow, induces catastrophic forgetting, and requires PCIe host synchronization.
3. **Long-Horizon State Drift & Context Explosion**:
   Autonomous software agents running for dozens or hundreds of turns accumulate massive context windows. Key-Value caches grow into tens of gigabytes, attention compute scales with $O(N^2)$, and subtle state transitions (e.g. task progress, tool state, environment facts) become diluted in the long prompt, causing execution failure.

### The Tensor Logic Solution
By representing entities as dense vector embeddings ($e \in \mathbb{R}^D$) and relations as resident matrix cores ($R_r \in \mathbb{R}^{D \times D}$) directly inside OpenXLA device memory (VRAM), knowledge operations become **compiled tensor contractions**:
- **Deductive Gating**: Queries are projected into relational memory space; if retrieved confidence exceeds threshold $\tau$, the output distribution is crisp-gated to grounded entity tokens, provably eliminating hallucinations.
- **Fast-Weight Online Learning**: New facts are stored instantaneously in GPU VRAM via outer-product superposition ($R_r \leftarrow R_r + e_h \otimes e_t$) in $O(D^2)$ forward compute with **zero backpropagation**.
- **Constant-Space Agent State Tracking**: Agent environment state and multi-hop relations are tracked via compiled Datalog fixpoint iterations (`stablehlo.while`) in $O(1)$ context space.

---

## 📚 Documentation Index

The Tensor Logic documentation suite is organized into four core modules:

### 1. 📐 [Theoretical Foundations](theory.md)
Detailed mathematical formulation of Pedro Domingos' Declarative Tensor Logic:
- Dual interpretation of Einstein summation as logical conjunction and matrix contraction.
- Value-carrying semirings (Boolean, Tropical, Fuzzy/Continuous, Softmax).
- Datalog fixpoints, stratified negation, and compiled transitive closures.
- Associative relational superposition memory, unbinding contractions, and capacity bounds.

### 2. 🌐 [Related Work & Comparative Analysis](related_work.md)
Detailed technical comparison with contemporary neuro-symbolic research and open-source implementations:
- Deep-dive into `waylandzhang/tensorlogic` (`transformer_reasoning_demo.py`, `EmbeddingSpace`, attention heads as relation discovery, KG-masked attention).
- Comparison with `pedronahum/tl-pjrt` (Python/JAX surface DSL vs bare-metal Clojure/PJRT systems compiler).
- Connections to Fast Weights, Modern Hopfield Networks, and Memory-Augmented Neural Networks.
- 6-dimensional architectural comparison matrix.

### 3. 🏛️ [The Two Pillars of Neuro-Symbolic Transformers & Novelty Assessment](architectural_pillars_and_novelty.md)
Comprehensive architectural breakdown, commercial enterprise positioning, and global literature novelty assessment:
- **Pillar 1 (Ecosystem Leverage)**: Grafting In-VRAM Datalog state tracking and ephemeral memory onto frozen open-weights models (Gemma 4).
- **Pillar 2 (Groundbreaking Native Foundation Architecture — TL-Nano)**: Decoupling factual memorization from semantic routing, enabling a **$50\%$ reduction in feed-forward MLP parameters** ($D_{\text{ff}} = 2D$).
- **Industry Novelty Analysis**: Comparative positioning against Pedro Domingos (2025), Fast Weight Programmers, Knowledge-Enhanced Transformers, and Mixture of Experts.

---

## ⚡ Repository Code Map

| Subsystem | Source Path | Test Path |
| :--- | :--- | :--- |
| **Relational Memory & Superposition** | [`einsum.logic.memory.relation`](../../src/einsum/logic/memory/relation.clj) | [`einsum.logic.memory.relation-test`](../../test/einsum/logic/memory/relation_test.clj) |
| **Unified TL-Transformer Block** | [`einsum.models.tl-block`](../../src/einsum/models/tl_block.clj) | [`einsum.models.tl-block-test`](../../test/einsum/models/tl_block_test.clj) |
| **TL-Nano Foundation Model** | [`einsum.models.tl-nano`](../../src/einsum/models/tl_nano.clj) | [`einsum.models.tl-nano-test`](../../test/einsum/models/tl_nano_test.clj) |
| **Symbolic Logic & Datalog Fixpoints** | [`einsum.logic.symbolic`](../../src/einsum/logic/symbolic.clj) | [`einsum.logic.symbolic-test`](../../test/einsum/logic/symbolic_test.clj) |
| **Value-Carrying Semirings** | [`einsum.logic.semiring`](../../src/einsum/logic/semiring.clj) | [`einsum.logic.semiring-test`](../../test/einsum/logic/semiring_test.clj) |
| **Autodiff Adjoints for Probes** | [`einsum.logic.autodiff`](../../src/einsum/logic/autodiff.clj) | [`einsum.logic.autodiff-test`](../../test/einsum/logic/autodiff_test.clj) |
| **Gemma 4 Grounding AST Block** | [`einsum.models.gemma`](../../src/einsum/models/gemma.clj) | [`einsum.integration.rocm-e2e-test`](../../test/einsum/integration/rocm_e2e_test.clj) |

