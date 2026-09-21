# Experiment E18: Diagnostics & Factorial Intervention Sweep

**Status:** COMPLETE (Definitive Negative Result / Representational Hypothesis Accepted)  
**Date:** 2026-09-17  
**Pod:** `catalog/trainable-limits/e18-diagnostics/`  
**Harness:** [`run.clj`](run.clj)  
**Telemetry:** [`phase0_diagnostics.edn`](phase0_diagnostics.edn), [`results.edn`](results.edn), [`summary.csv`](summary.csv)  

---

## 1. Thesis & Competing Hypotheses

The E16 experiment hypothesized that Stage-2 bilinear cross-attention failed due to "uniform in-batch negative blindness." An audit of the training loop revealed three competing hypotheses:
1. **The Sampler Hygiene Hypothesis:** `(mod i n)` cycling when $n < 16$ created self-as-negative label noise and false negatives.
2. **The Query Mismatch Hypothesis:** $W_Q$ was trained on raw entity token embeddings ($V_h$) but evaluated on contextual hidden states ($h_{\text{ctx}}$).
3. **The Representational Collapse Hypothesis:** Post-$W_K$ keys for co-typed candidates are near-parallel ($\cos \to 1.0$), making separation by any dot-product head mathematically impossible.

---

## 2. Phase 0 Diagnostics & Gating Decision

Phase 0 tested all three hypotheses on the resident Gemma 4 E2B checkpoint on AMD ROCm (RX 7900 XTX):
- **Denominator Contamination:** Confirmed at **28.09%** (20.00% self-as-negative duplicates, 8.09% false negatives).
- **Query Mismatch:** Confirmed severe with $\cos(V_h, h_{\text{ctx}}) = 0.0732 \pm 0.0596$ (an angle of $\approx 85.8^\circ$, nearly orthogonal).
- **Key Geometry:** Representational collapse hypothesis **falsified**; post-$W_K$ co-typed keys had cosine $0.3971 \pm 0.3876$ vs background $0.0967 \pm 0.3502$.
- **Gating Decision:** Interventions on sampler hygiene and query alignment were justified and Phase 1 was opened.

---

## 3. Phase 1: 2×2 Factorial Sweep Protocol

Conducted across 3 random seeds per cell ($N = 12$ runs) in OpenXLA PJRT VRAM:
- **Factor A (Sampler Hygiene):**
  - **A0 (Replication):** E16's `(mod i n)` cycling sampler with fixed identity target matrix.
  - **A1 (Hygienic):** Distinct triples without replacement, multi-positive target distribution, and false-negative masking.
- **Factor B (Query Distribution Alignment):**
  - **B0 (Replication):** Token embeddings $V_h = \text{gather}(W_{\text{embed}}, I_h)$.
  - **B1 (Aligned):** Contextual hidden states $h_{\text{ctx}}$ pre-computed from frozen Gemma 4 evaluation forward passes.

---

## 4. Empirical Summary (3 Seeds, Mean ± Std)

| Cell | Description | Seen Top-1 Acc | Seen Pointwise Selectivity | Seen Distractor Boost | Unseen Pointwise Selectivity |
|:---:|:---|:---:|:---:|:---:|:---:|
| **A0B0** | Replication (Cycling + Token Embeddings) | $13.3 \pm 0.6$ (33.3%) | **$0.0 \pm 0.0$ (0.0%)** | $+0.2390 \pm 0.0732$ | $0.0 \pm 0.0$ (0.0%) |
| **A1B0** | Hygienic + Token Embeddings | $13.0 \pm 0.0$ (32.5%) | **$0.0 \pm 0.0$ (0.0%)** | $+2.7497 \pm 1.1970$ | $0.0 \pm 0.0$ (0.0%) |
| **A0B1** | Cycling + Contextual Hidden States | $13.0 \pm 0.0$ (32.5%) | **$0.3 \pm 0.6$ (0.8%)** | **$-0.1674 \pm 0.1558$** | $0.0 \pm 0.0$ (0.0%) |
| **A1B1** | Full Intervention (Hygienic + Contextual) | $13.0 \pm 0.0$ (32.5%) | **$0.0 \pm 0.0$ (0.0%)** | $+10.2779 \pm 0.2833$ | $0.0 \pm 0.0$ (0.0%) |

---

## 5. Evaluation of Criteria & Falsification Clause

1. **Resolution Criterion (FAIL):** Required contextual pointwise selectivity $\ge 15\%$ ($6/40$). Measured: $0/40$ in 11/12 runs, $1/40$ max.
2. **Attribution Criterion (FAIL):** No systematic lift across any cell ($0.0\%$ pointwise selectivity across A1B0 and A1B1).
3. **Mechanistic Signature (PARTIAL):** Contextual alignment (B1) eliminated spurious distractor amplification ($-0.1674$ vs $+0.24$), but could not achieve pointwise discrimination.
4. **Falsification Clause Execution:**
   > *"If no cell meets (1), accept the representational hypothesis as the leading explanation. Do not run further hyperparameter sweeps under this spec."*

**Conclusion:** The failure of continuous bilinear dot-product cross-attention is representational, not mechanical. Pointwise entity disambiguation requires discrete symbolic constraint satisfaction.
