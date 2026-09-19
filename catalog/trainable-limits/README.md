# Arc 2: Trainable Limits, Relational Memory & Contrastive Adjoints (E11–E20)

**Status**: `CLOSED` (Gradient predicate invention empirically refuted)  
**Minted IDs**: `e11` through `e20`  
**Primary Substrate**: AMD Radeon RX 7900 XTX (ROCm PJRT)

---

## 1. Scope & Objective
Arc 2 tested the outer bounds of Trainable Tensor Logic on standardized real-world benchmarks (WebNLG v3.0, 348 relations). It investigated whether continuous gradient descent (InfoNCE, contrastive unbinding) could discover discrete symbolic predicates and reliably select ground-truth entities from distractor pools.

## 2. Key Findings & Empirical Trajectory
- **E11 & E12 (WebNLG Benchmark Pre-training)**: Scaled relational unbinding across 348 relations with exact row-wise softmax adjoints lowered into device VRAM.
- **E14 & E15 (Distractor Diagnostic & Gemma 4 Grafting)**: Proved the Linear Unbinding Theorem. When scaled to large distractor sets or grafted onto frozen Gemma 4 E2B, linear unbinding nudges logits ($+0.38\text{ logits}$) but fails to cross the argmax threshold ($1.0\%$ Top-1 accuracy).
- **E16–E20 (The Gradient Predicate Invention Refutation)**:
  - Systematically evaluated end-to-end continuous relaxation, temperature annealing, and constrained crystallization across 2×2 factorial sweeps.
  - **Definitive Finding**: Gradients learn geometric rankings and continuous manifolds, **not discrete symbolic truth**. Unconstrained gradient optimization cannot resolve compositional unidentifiability in predicate invention.
  - **Program Decision**: The line on gradient-discovered predicate invention is permanently closed. Discrete commitment machinery must live outside the gradient loop (leading directly to Arc 3).

## 3. Paper Export
This arc constitutes the negative and boundary results documented in **Paper 2: The Limits of Trainable Tensor Logic**.
