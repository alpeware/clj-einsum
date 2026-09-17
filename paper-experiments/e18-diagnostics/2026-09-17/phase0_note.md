# Experiment E18 Phase 0 Diagnostics Note: Batch Hygiene, Key Geometry, and Query Alignment

**Date:** 2026-09-17  
**Status:** COMPLETE (Gating Phase 1)  
**Artifacts:** [`phase0_diagnostics.edn`](phase0_diagnostics.edn)

---

## 1. Executive Summary & Gating Decision

The E16 write-up hypothesized that Stage-2 cross-attention failed due to "uniform in-batch negative blindness." An audit of the code revealed that batches were already relation-stratified, raising three competing hypotheses:
1. **The Sampler Hygiene Hypothesis**: `(mod i n)` cycling when $n < 16$ created self-as-negative label noise and false negatives.
2. **The Query Mismatch Hypothesis**: $W_Q$ was trained on raw entity token embeddings ($V_h$) but evaluated on contextual hidden states ($h_{\text{ctx}}$).
3. **The Representational Collapse Hypothesis**: Post-$W_K$ keys for co-typed candidates are near-parallel ($\cos \to 1.0$), making separation by any dot-product head mathematically impossible.

Phase 0 tested all three hypotheses on the resident Gemma 4 E2B checkpoint on AMD ROCm (RX 7900 XTX).

### Key Decisions:
- **Cycling bug is highly material (28.09% contamination)**: 20.00% of negative positions were copies of the positive itself, and 8.09% were valid true triples penalized as negatives.
- **Query mismatch is severe ($\cos = 0.0732$)**: Train queries ($V_h$) and eval queries ($h_{\text{ctx}}$) are practically orthogonal ($85.8^\circ$ apart).
- **The Representational Collapse Hypothesis is FALSIFIED**: Post-$W_K$ keys have a mean co-typed cosine of only $0.3971 \pm 0.3876$ (with random pairs at $0.0967 \pm 0.3502$). Keys have ample angular separation to support pointwise discrimination.
- **GATE TO PHASE 1: APPROVED [OPEN]**. Intervening on Factor A (sampler hygiene) and Factor B (contextual query alignment) is necessary and grounded in measured reality.

---

## 2. Measurement Table

| Diagnostic Metric | Measured Value | Theoretical Implication |
|:---|:---:|:---|
| **Distinct Training Triples** | 3,738 | WebNLG clean subset |
| **Total Relations** | 370 | High relational diversity |
| **Relations with $n < 16$** | **306 (82.7%)** | Vast majority of relations require batch cycling |
| **Relations with $n < 2$ (untrained)** | 120 (32.4%) | Excluded from training loop |
| **Relations with $n \ge 2$ (trained)** | 250 (67.6%) | 186 of 250 (74.4%) have $n < 16$ |
| **Total InfoNCE Off-Diagonal Positions** | 60,000 | 250 batches $\times 16 \times 15$ |
| **Duplicate-Positive Rate (Self-as-Negative)** | **20.00% (12,000 / 60,000)** | In $n < 16$, duplicate rate is **25.80%** |
| **False-Negative Rate (True Facts as Negatives)** | **8.09% (4,853 / 60,000)** | Penalized true triples in multi-tail relations |
| **Total Negative Contamination Rate** | **28.09% (16,853 / 60,000)** | Nearly 1 in 3 negatives was corrupted |
| **Query Cosine Alignment ($\cos(V_h, h_{\text{ctx}})$ Seen)** | **$0.0732 \pm 0.0596$** | Almost orthogonal ($85.8^\circ$ angle) |
| **Query Cosine Alignment ($\cos(V_h, h_{\text{ctx}})$ Unseen)** | **$0.1099 \pm 0.0477$** | Out-of-distribution evaluation confirmed |
| **Pre-$W_K$ Embedding Cosine (Co-typed)** | $0.3872 \pm 0.3581$ | Semantic cluster baseline |
| **Pre-$W_K$ Embedding Cosine (Random)** | $0.1819 \pm 0.2451$ | Background separation: $\Delta = +0.2053$ |
| **Post-$W_K$ Key Cosine (Co-typed)** | **$0.3971 \pm 0.3876$** | **NOT collapsed** (far from 1.0) |
| **Post-$W_K$ Key Cosine (Random)** | **$0.0967 \pm 0.3502$** | Background suppressed: $\Delta = +0.3005$ |
| **Stage 2 Training Loss (Epoch 1 $\to$ 5)** | **$7.8566 \to 5.0278$ (-36.0%)** | Active monotonic convergence (0.64 ms/step) |

---

## 3. Amplification Decomposition on TEST_SEEN ($N=40$)

The doubling of distractor boost in E16 ($+0.2505 \to +0.5048$) is decomposed additively:
$$\Delta_{\text{stage2}}(c) = \Delta_{\text{type}}(c) + \text{score}_{\text{resolve}}(c)$$

| Component | Target Shift | Mean Distractor Shift | Max Distractor Shift |
|:---|:---:|:---:|:---:|
| **Stage 1 ($\Delta_{\text{type}}$ alone)** | $+0.7864$ | $+0.3400$ | $+2.2176$ |
| **Resolver Alone ($\text{score}_{\text{resolve}}$)** | **$-0.0021$** | **$+0.1648$** | **$+3.4797$** |
| **Stage 2 Combined ($\Delta_{\text{stage2}}$)** | **$+0.7844$** | **$+0.5048$** | **$+4.2592$** |

### Decomposition Finding:
- The resolver head contributed **$-0.0021$** net shift to the target entity (completely flat).
- The resolver head contributed **$+0.1648$** positive shift across all 746 distractors, and pushed max distractor shifts to $+3.48$.
- **Conclusion**: The amplification is entirely driven by $\text{score}_{\text{resolve}}$ acting as an unaligned, diffuse positive booster due to the orthogonal query mismatch and self-negative label noise.

---

## 4. Plan for Phase 1 (2×2 Intervention Factorial)

Since both hypothesized mechanisms (#2 and #3) are verified and substantial, and the representational collapse hypothesis is ruled out, Phase 1 will execute the pre-registered 2×2 factorial:

1. **Factor A (Sampler Hygiene)**:
   - **A0 (Replication)**: `(mod i n)` cycling with identity Target.
   - **A1 (Hygienic)**:
     - Sample $K=16$ distinct triples per batch without replacement (or draw from all entities of relation).
     - Multi-positive Target: row $i$ distributes target probability mass uniformly across all batch positions $j$ that hold a true positive tail for $(h_i, r)$.
     - False-negative masking: set `Mask_Scale[i, j] = 0.0` wherever $(h_i, r, t_j)$ is a known true WebNLG triple.
2. **Factor B (Query Alignment)**:
   - **B0 (Replication)**: Train $W_Q$ on raw token embeddings $V_h$.
   - **B1 (Aligned)**: Train $W_Q$ on pre-cached contextual hidden states $h_{\text{ctx}}$ obtained from the frozen Gemma forward pass on the training prompt `"The <rel> of <h> is"`.
3. **Execution**: 3 seeds per cell (Seeds 42, 43, 44) evaluated on the exact 80 test queries against the 747 candidate pool.
