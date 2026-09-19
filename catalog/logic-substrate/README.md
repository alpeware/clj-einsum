# Arc 4: Differentiable Logic Compilation & Semiring Extensions (E25+)

**Status**: `ACTIVE`  
**Minted IDs**: Pending verification (Target: `e25`)  
**Primary Substrate**: OpenXLA PJRT (CPU & AMD ROCm)

---

## 1. Scope & Objective
Arc 4 investigates the formal compilation of differentiable logic programs and external literature proposals into generalized semiring einsum equations executing on accelerated hardware.

## 2. Active Lines of Inquiry
- **Semiring Einsum**: Generalizing contraction beyond $(+, \times)$ to $(\vee, \wedge)$, $(\max, +)$, and $(\text{LSE}, +)$.
- **External Literature Replication**: Testing external neural-symbolic and logic-programming literature on physical silicon.
- **First Target Candidate**: Koki Takemura and Katsumi Inoue (2024), *Differentiable Logic Programming for Distant Supervision* (arXiv:2408.12591), compiling their $Q$, $D$, and $\min(1, \cdot)$ operators into StableHLO contractions.
