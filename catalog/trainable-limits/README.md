# Arc 2: Trainable Limits, Relational Memory & Contrastive Adjoints (E11–E20)

**Generation**: $G_2$ (Trainable Limits & Continuous Refutation)  
**Status**: `CLOSED` (Gradient predicate invention empirically refuted)  
**Minted IDs**: `e11` through `e20`  
**Primary Substrate**: AMD Radeon RX 7900 XTX (ROCm PJRT)  
**RSI Gates Addressed**: Gate 1 (Resource), Gate 2 (Time), Gate 3 (Capability — Refuted for continuous relaxation)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective
Arc 2 tested the outer boundaries of Trainable Tensor Logic on standardized real-world benchmarks (WebNLG v3.0, 348 relations). It investigated whether continuous gradient descent (InfoNCE, contrastive unbinding) could autonomously discover discrete symbolic predicates and reliably select ground-truth entities from distractor pools.

## 2. RSI Gate Evaluations & Key Findings
- **Gate 1 (Resource Efficiency):**
  - **E11 & E12**: Lowered exact InfoNCE adjoint autodiff directly into device VRAM, scaling relational unbinding across 348 relations without host intermediate copies.
- **Gate 2 (Time Efficiency):**
  - **E12**: In-VRAM parameter pinning eliminated PCIe transfer overhead during high-frequency gradient updates.
- **Gate 3 (Capability Improvement — The Continuous Refutation):**
  - **E14 & E15 (Linear Unbinding Theorem)**: Grafting relational memory onto frozen Gemma 4 E2B proved that linear unbinding nudges logits ($+0.38\text{ logits}$) but fails to cross the argmax threshold ($1.0\%$ Top-1 accuracy) in large distractor pools.
  - **E16–E20 (Gradient Predicate Invention Refutation)**: Evaluated end-to-end continuous relaxation (E17), 2x2 factorial temperature sweeps (E18), hybrid crystallization (E19), and constrained crystallization (E20).
  - **Definitive Finding**: Gradients learn geometric rankings and continuous manifolds, **not discrete symbolic truth**. Unconstrained gradient optimization cannot resolve compositional unidentifiability.
  - **RSI Architectural Verdict**: Gate 3 cannot be cleared by pure continuous relaxation. Discrete commitment machinery must live outside the gradient loop, directly motivating the hybrid architecture of Generation $G_3$.

## 3. Contained Experiment Pods
- [`e17-predicate-invention/`](e17-predicate-invention/)
- [`e18-diagnostics/`](e18-diagnostics/)
- [`e19-hybrid-crystallization/`](e19-hybrid-crystallization/)
- [`e20-constrained-crystallization/`](e20-constrained-crystallization/)

## 4. Paper Export
This arc constitutes the negative and boundary results documented in **Paper 2: The Limits of Trainable Tensor Logic**.
