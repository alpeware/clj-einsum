# Arc 4: Differentiable Logic Compilation & Semiring Extensions (E25+)

**Generation**: $G_4$ (Self-Compiling Meta-Research Substrate)  
**Status**: `ACTIVE`  
**Minted IDs**: Pending verification (Target: `e25`)  
**Primary Substrate**: OpenXLA PJRT (CPU & AMD ROCm)  
**RSI Gates Addressed**: Gate 1 (Resource), Gate 2 (Time), Gate 3 (Capability), Gate 4 (Repeatability)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective
Arc 4 investigates the autonomous compilation of external logic programming literature and generalized semirings directly into StableHLO MLIR equations executing on accelerated silicon. It represents the realization of Generation $G_4$, where the toolchain autonomously assimilates formal symbolic literature into its own executable substrate.

## 2. Active Lines of Inquiry & RSI Gates
- **Semiring Einsum (Gates 1 & 2):** Generalizing contraction beyond standard $(+, \times)$ rings to $(\vee, \wedge)$, $(\max, +)$, and $(\text{LSE}, +)$, enabling sparse logic fixpoints to compile into single fused GPU kernels.
- **External Literature Reification (Gates 3 & 4):**
  - **First Target Candidate (`e25`)**: Koki Takemura and Katsumi Inoue (2024), *Differentiable Logic Programming for Distant Supervision* (arXiv:2408.12591).
  - Translating their $Q$, $D$, and $\min(1, \cdot)$ differentiable Datalog operators into homoiconic Clojure Hiccup ASTs and compiling to PJRT executables.
- **The Closed Meta-Research Loop:** Demonstrating autonomous end-to-end traversal from literature proposal to device execution and catalog commit.
