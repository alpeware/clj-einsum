# Experiment E17: Gradient-Based Predicate Invention with Held-Out Evaluation and Discrete Recovery

**Status:** COMPLETE (Definitive Negative Result)  
**Date:** 2026-09-17  
**Pod:** `catalog/trainable-limits/e17-predicate-invention/`  
**Harness:** [`run.clj`](run.clj)  
**Telemetry:** [`results.edn`](results.edn), [`summary.csv`](summary.csv), [`sample_efficiency.csv`](sample_efficiency.csv)  

---

## 1. Thesis & Theoretical Motivation

In the foundational literature on neuro-symbolic learning, two conflicting theses stand in direct tension:
- **Domingos-2020** (*"Every Model Learned by Gradient Descent is Approximately a Kernel Machine"*): Gradient descent does not invent genuinely novel internal representations or discrete symbols; it performs nearest-neighbor interpolation in data-dependent reproducing kernel Hilbert spaces.
- **Domingos-2025** (*"Declarative Tensor Logic"*): Predicate invention naturally "falls out" of gradient descent over tensor logic programs when a latent core tensor $L = \sigma(Z / \tau)$—tied to no observed relation—is trained end-to-end through relational compositions.

Experiment E17 was designed as a pre-registered, falsifiable test of this claim on a synthetic forest of family trees with known ground truth:
- **Hidden Ground-Truth Predicate:** $\text{Parent}$ (withheld from the learner, known only to the evaluator).
- **Observed Target Relation:** $\text{Grandparent}$ (two-hop composition: $G = L \cdot L$).
- **Observed Auxiliary Relation:** $\text{Sibling}$ (shared latent parent: $S = L^T \cdot L$).
- **Universe:** $N = 64$ entities partitioned into $F = 8$ discrete 3-generation family trees (2 founders $\to$ children + spouse $\to$ grandchildren, zero inbreeding, discrete generations).

---

## 2. Model Architecture & Scoring Paths

The entire training step was compiled into a single OpenXLA PJRT kernel running 100% in device memory (zero host float loops):
1. **Baseline Bilinear Path:** $s_{\text{base}}(h, t) = E_h W_{\text{GP}} E_t^T$ (dimension $d = 64$).
2. **Latent Invention Path:** $s_{\text{inv}}(h, t) = (L \cdot L)_{h, t} = \sum_m L(h, m) L(m, t)$ where $L = \sigma(Z / \tau)$.
3. **Auxiliary Sibling Path:** $s_{\text{Sib, inv}}(s_1, s_2) = (L^T \cdot L)_{s_1, s_2} = \sum_p L(p, s_1) L(p, s_2)$.
4. **Combined Scoring:** $S_{\text{GP}} = (s_{\text{base}} + s_{\text{inv}}) / \tau_{\text{ce}}$, $S_{\text{Sib}} = (s_{\text{Sib, base}} + s_{\text{Sib, inv}}) / \tau_{\text{ce}}$.
5. **Annealing & Regularization:** Geometric temperature annealing $\tau = 1.0 \to 0.05$ over 200 epochs, plus L1 sparsity penalty $\lambda \|\sigma(Z / \tau)\|_1$.
6. **In-Graph Adjoints & SGD:** Full backward pass compiled in StableHLO:
   $$\text{adj\_L}_{\text{GP}} = G_{\text{GP}} L^T + L^T G_{\text{GP}}, \quad \text{adj\_L}_{\text{Sib}} = L G_{\text{Sib}}^T + L G_{\text{Sib}}$$
   $$\text{d}Z = \frac{1}{\tau} (\text{adj\_L}_{\text{total}} + \lambda) \odot L \odot (1 - L)$$

---

## 3. Pre-Registered Ablation Matrix & Success Criteria

- **A0 (Rank-0 Baseline):** Bilinear scoring only ($s = s_{\text{base}}$), no latent core ($L = 0$).
- **A1 (Frozen Core):** $Z$ initialized randomly and frozen throughout training (extra capacity control).
- **A2 (Full E17 Invention):** Learned $Z$, temperature annealing $1.0 \to 0.05$, L1 sparsity.
- **A3 (Low-Rank Tucker):** $Z = A B^T$ with rank $r = 8$.

### Pre-Registered Success Criteria:
1. **Criterion 1 (Predictive lift):** $\Delta \text{Hits}@1$ (held-out Grandparent) of A2 over A0 $\ge +0.15$ absolute, positive on $\ge 4/5$ seeds.
2. **Criterion 2 (Discrete recovery):** $F_1(\hat{L}, \text{Parent}) \ge 0.80$ after annealing ($\hat{L} = (L > 0.5)$).
3. **Criterion 3 (Learning, not capacity):** A1 lift over A0 $< 50\%$ of A2 lift.
4. **Criterion 4 (Rank sanity):** Full-matrix A2 $\ge$ best low-rank A3 on $F_1$.

---

## 4. Empirical Verdict: Falsification of Pure GD Predicate Invention

Executed across 5 independent seeds (42, 43, 44, 45, 46) on OpenXLA PJRT (2.37 ms/epoch, 100% device VRAM):

| Ablation | Description | Held-Out Hits@1 | MRR | Discrete $F_1$ | Two-Hop Closure | Step Latency |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **A0** | Rank-0 Baseline | $0.0\% \pm 0.0\%$ | $0.1186 \pm 0.044$ | $0.0000 \pm 0.000$ | $0.0\%$ | 1.77 ms |
| **A1** | Frozen Core | $0.0\% \pm 0.0\%$ | $0.0657 \pm 0.021$ | $0.0380 \pm 0.004$ | $100.0\%$ | 1.94 ms |
| **A2** | Full E17 Invention | $0.0\% \pm 0.0\%$ | **$0.2348 \pm 0.046$** | **$0.0295 \pm 0.006$** | **$92.1\%$** | 2.37 ms |
| **A3** | Tucker Rank-8 | $0.0\% \pm 0.0\%$ | $0.2243 \pm 0.056$ | $0.0394 \pm 0.003$ | $100.0\%$ | 2.55 ms |

### Criteria Evaluation:
- **Criterion 1 (Predictive Lift):** FAILED ($\Delta \text{Hits}@1 = 0.0\%$, though MRR improved $+98\%$).
- **Criterion 2 (Discrete Recovery):** FAILED ($F_1 = 0.0295 \ll 0.80$; recovered 1,012 spurious edges, precision 1.9%).
- **Criterion 3 (Learning over Capacity):** FAILED ($F_1$ of learned A2 was lower than random frozen core A1).
- **Criterion 4 (Rank Sanity):** FAILED (A3 low-rank beat full matrix A2 on $F_1$).

**Overall Conclusion:** Gradient descent over continuous multilinear tensor relaxations learns soft associative patterns (high MRR) but cannot invent discrete predicates ($F_1 \approx 0.03$). Predicate invention requires discrete constraint satisfaction and symbolic verification, not continuous gradient descent.
