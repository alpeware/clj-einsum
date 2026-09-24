# VISION.md — Executable Tensor Logic: The Consumer-Silicon Apparatus

> **For operational milestones (The Staircase), gate definitions, and operating rules, see [`PROCESS.md`](PROCESS.md).**

---

## 1. The Physics of Consumer Silicon: Deflationary Apparatus over Hype

Frontier industrial AI scaling is currently the opposite of Recursive Self-Improvement (RSI): clusters scale to billions of dollars, iteration cycles grow slower, and human intervention expands. `clj-einsum` adopts Ali Ghodsi's deflationary premise: **we do not claim the system "has achieved RSI."** Instead, we build the physical measurement apparatus on consumer silicon (single 24GB GPUs: AMD Radeon RX 7900 XTX / NVIDIA RTX 4090) to measure whether its required conditions hold.

We maintain a strict boundary between mathematical guarantees and empirical hypotheses:

### Storage & Bandwidth: Arithmetic
- Uncompressed FP16 requires $2.0\text{ bytes/param}$. A 30B model requires $\approx 60\text{ GB}$; a 70B model requires $\approx 140\text{ GB}$. Neither fits on consumer silicon.
- 1.58-bit / ternary quantization ($\{-1, 0, 1\}$) reduces weights to $\approx 0.25\text{ bytes/param}$.
- **The Arithmetic Win:** A 30B model compresses to $\approx 7.5\text{ GB}$; a 70B model compresses to $\approx 17.5\text{ GB}$. Both fit entirely within a single 24GB consumer GPU with generous headroom for KV caches. This storage reduction is mathematical certainty, not research speculation.

### Compute & Throughput: Hypothesis
- Unpacking ternary weights reduces memory bandwidth pressure during autoregressive decoding, but still executes floating-point GEMM on hardware tensor cores.
- **The Measurement Obligation:** We never assume compute speedups by default. Token throughput ($\text{tok/s}$) and achieved GFLOPs/watt must be measured on silicon against optimized FP16 and INT4 (EXL3) baselines.

---

## 2. What the Representation Covers — and What It Doesn't

### Covered — The Parts That Work:
- **Tensor Contraction (einsum)**: The shared mathematical core of deep learning: attention, MLPs, convolutions, and relational joins. One construct, `[:= head & body-terms]`; a shared contracted index *is* the matmul, the join, the attention score.
- **Function-Free Datalog Rules**: Compiled directly to contractions over embedded relations; lowered to StableHLO without runtime interpretation.
- **Elementwise Nonlinearities & Autodiff Closure**: Activations are attributes on equations (`{:act :gelu}`); the gradient of a tensor-logic program is itself a pure tensor-logic program.
- **Statically-Shaped Dataflow**: Exact compilation via OpenXLA PJRT with differential verification.

### Not Covered — Deliberately Out of Scope:
- Full first-order logic with function symbols, modal logic, or unconstrained probabilistic programming.
- Absolute soundness in continuous embedding space (PAC-style margins, not symbolic certainty).
- Continuous-only predicate invention: Gradients learn geometric rankings, not discrete selection. Predicate invention requires discrete commitment machinery living *around* the tensor core.

---

## 3. The 4 Research Pillars

- **Pillar 0 — The Compiler (Built)**: Tensor Logic AST $\to$ StableHLO $\to$ PJRT with Panama FFM bindings, validated via differential testing.
- **Pillar 1 — Trainable Limits (Established)**: Proving the boundaries of continuous structure learning; gradients optimize ranking manifolds, necessitating discrete commitment gates.
- **Pillar 2 — The Agent Deliberation Loop (Built)**: Three-tier deliberation (reflex contractions, device deliberation, host synthesis) with schema-verified memory commits.
- **Pillar 3 — The RSI Measurement Apparatus (Active)**: Automated closed loop measuring Ghodsi's 4 gates on consumer silicon across generation lineage.
