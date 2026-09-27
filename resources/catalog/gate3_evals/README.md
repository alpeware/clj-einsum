# Gate 3: Intelligence & Capability Evaluations (`catalog/gate3_evals/`)

**Metric**: `Floor retention (≥95%) relative to uncompressed base model & absolute benchmark score`  
**Target**: Zero catastrophic degradation on reasoning tasks across compression generations  
**Primary Hardware Target**: AMD Radeon RX 7900 XTX (ROCm PJRT)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective

Gate 3 evaluates whether a model generation demonstrates **increasing intelligence** ($A_{n+1} > A_n$) while strictly preserving a **$\ge 95\%$ retention floor** relative to the uncompressed base model:
- **Floor Retention**: Measuring perplexity, reasoning (GSM8K), and syntax generation (HumanEval/Clojure) against unquantized FP16 baselines.
- **The Continuous Refutation**: Empirically measuring why continuous gradient descent fails discrete predicate invention without discrete commitment priors (E14–E20).

## 2. Contained Experiment Pods (Generation $G_2$, E14–E20)

- [`e17_predicate_invention/`](e17_predicate_invention/): Evaluated end-to-end continuous relaxation for predicate discovery; proved continuous relaxation fails discrete compositional unidentifiability.
- [`e18_diagnostics/`](e18_diagnostics/): 2x2 factorial temperature sweep; definitive refutation of gradient resolver under varying temperature regimes.
- [`e19_hybrid_crystallization/`](e19_hybrid_crystallization/): Evaluated gradient search with temperature decay; proved unconstrained search finds spurious covers.
- [`e20_constrained_crystallization/`](e20_constrained_crystallization/): Falsified weak structural priors; permanently closed the line on gradient-discovered predicate invention.

## 3. Standing instruments

- [`clojure_bench/`](clojure_bench/): Clojure coding baseline for the Gemma 4 family
  (capability verdict `done`, frozen 2026-09-27). Rows in `results.edn` carry
  `:harness-sha` (repo HEAD at row-write time) and `:harness-dirty?` from the
  freeze commit onward. **Rows without `:harness-sha` are pre-versioning** —
  produced by the v1 harness before versioning existed; the SHA was never
  backfilled because the producing working trees are unknown.
