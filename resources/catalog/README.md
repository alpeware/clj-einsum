# Experiment Catalog: Hardware Measurement Apparatus for RSI

This directory contains verified, reproducible research pods and the master machine-readable registry that constitute the empirical record of `clj-einsum`.

The catalog serves as a **hardware-grounded measurement apparatus** evaluating **Ali Ghodsi's 4 Recursive Self-Improvement (RSI) conditions** on consumer silicon (AMD Radeon RX 7900 XTX 24GB, RDNA3 gfx1100, OpenXLA PJRT):
1. **Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$):** Materially fewer VRAM bytes, parameters, and compute operations.
2. **Gate 2: Time Efficiency ($T_{n+1} \ll T_n$):** Materially lower adaptation and deliberation latency.
3. **Gate 3: Capability Improvement ($A_{n+1} > A_n$):** Strictly higher task accuracy, verifiable safety, and zero invariant violations.
4. **Gate 4: Continuous Recursion ($G_n \to G_{n+1} \to \dots$):** Continuous, autonomous operation without degradation over extended horizons.

For the operational process, milestones, and gate altimeters, consult [**`PROCESS.md`**](../../PROCESS.md).

---

## 1. Functional Gate Organization

Verified pods are sharded by functional RSI gate:

```
resources/catalog/
├── registry.edn                     # Master machine-readable catalog
├── gate1_compression/<slug>/        # Resource efficiency & weight compression
├── gate2_velocity/<slug>/           # Fast calibration & sub-second adaptation
├── gate3_evals/<slug>/              # Capability retention floor (>= 95%) & task evals
└── gate4_recursion/<slug>/          # Continuous recursion & compound judgment derivative
```

---

## 2. Slug-Based Parallel Research Tracks

Proposals and catalog entries are tracked by descriptive **`<slug>`** (e.g. `cat-q-ternary`, `prefix-cache-handover`) rather than artificial sequential numbers:
- **Parallel Tracks**: Multiple agents explore independent lottery tickets simultaneously without claim collisions or centralized number coordination.
- **Verification Requirement**: A pod is promoted from `resources/proposals/<gate>/<slug>/` to `resources/catalog/<gate>/<slug>/` only when an independent agent run replicates the claim on silicon against pre-registered kill criteria.
- **Pod Artifacts**: Each verified catalog pod contains:
  - `spec.md`: The RFC specification and pre-registered falsification criteria.
  - `results.edn`: Machine-readable raw metrics collected on physical hardware.
  - `summary.csv`: Tabular benchmark results for reporting and aggregation.

---

## 3. Master Registry

The single source of machine-readable truth is [`registry.edn`](registry.edn). Each verified experiment is registered under its descriptive `:slug` alongside its target generation, gate, hardware configuration, and measured metrics.
