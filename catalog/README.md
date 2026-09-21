# Experiment Catalog: Hardware Measurement Apparatus for RSI

This directory contains the immutable, reproducible research pods that constitute the empirical record of `clj-einsum`.

The catalog serves as a **hardware-grounded measurement apparatus** evaluating **Ali Ghodsi's 4 Recursive Self-Improvement (RSI) conditions** on consumer silicon (AMD Radeon RX 7900 XTX 24GB, RDNA3 gfx1100, OpenXLA PJRT):
1. **Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$):** Materially fewer VRAM bytes, parameters, and compute operations.
2. **Gate 2: Time Efficiency ($T_{n+1} \ll T_n$):** Materially lower adaptation and deliberation latency.
3. **Gate 3: Capability Improvement ($A_{n+1} > A_n$):** Strictly higher task accuracy, verifiable safety, and zero invariant violations.
4. **Gate 4: Repeatability / Closed Loop ($G_n \to G_{n+1} \to \dots$):** Continuous, autonomous operation without degradation over extended horizons.

For the theoretical and architectural specification of the four gates and the meta-research loop, see [docs/architecture/rsi_gates.md](../docs/architecture/rsi_gates.md).

---

## 1. Functional Gate Organization

The catalog is structured into the four functional RSI gates:

```
┌───────────────────────────────┬─────────────────────────────┬────────────────────────────────────────────────────────┐
│ Gate Directory                │ Target Metric               │ Focus / Pods                                           │
├───────────────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────────┤
│ catalog/gate1-compression/    │ bytes/param & GFLOPs/watt   │ E11, E12, E24 (Prefix C2C Handover), PTQ / CAT-Q       │
│ catalog/gate2-velocity/       │ Adaptation wall-clock time  │ E01–E10 (Fast weights, unbinding, sub-2ms kernels)     │
│ catalog/gate3-evals/          │ Floor retention (≥95%)      │ E14–E20 (Limits of gradient invention, benchmark pods) │
│ catalog/gate4-recursion/      │ Compound judgment derivative│ E21–E23 (Verified commit gates, reduce loop, self-host)│
└───────────────────────────────┴─────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 2. Master Experiment Registry Table

The master machine-readable source of truth is [catalog/registry.edn](registry.edn). Every experiment is minted with an immutable `eNN` identifier, mapped to its Generation and evaluated RSI Gate:

| ID | Gate | Generation | Pod Directory | Status | Hardware Target | Empirical Measurement / Finding |
|---|---|---|---|---|---|---|
| **`e01`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | CPU / Metal | Cross-Attention Memory Probe: attention weights concentrate on head entities in LOO-CV. |
| **`e02`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | CPU / PJRT | KG-Masked Attention: in-graph distractor suppression preserving causal masks. |
| **`e03`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Contrastive Subspace Pre-training: Top-3 generalization on unseen entities in sub-second GPU training. |
| **`e04`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | In-VRAM Datalog Fixpoint State Tracker: 0 hallucinations across multi-hop deductive paths in sub-2ms PJRT. |
| **`e05`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Zero-gradient ephemeral learning: instantaneous recall with clean fact retraction. |
| **`e06`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Unified TL-Transformer Layer Block: end-to-end hybrid forward pass executing in 4.757 ms. |
| **`e07`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | SWE-Bench 100-Turn Challenge: immunity to long-horizon amnesia via $O(1)$ state tracking. |
| **`e08`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Dynamic In-VRAM Relation Induction: StableHLO tensor factorization (>85% fidelity, 1.7 ms kernel). |
| **`e09`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Native TL-Nano Open-Weights: rapid loss descent at 553 tok/s on consumer silicon. |
| **`e10`** | Gate 2 | $G_0$ | [`gate2-velocity/`](gate2-velocity/) | `VERIFIED` | RX 7900 XTX / ROCm | Real-world CEO Corpus QA: verified deductive logit boost ($+0.8900$). |
| **`e11`** | Gate 1 | $G_0$ | [`gate1-compression/`](gate1-compression/) | `VERIFIED` | RX 7900 XTX / ROCm | WebNLG v3.0 Pre-training: relational unbinding scaled across 348 relations. |
| **`e12`** | Gate 1 | $G_0$ | [`gate1-compression/`](gate1-compression/) | `VERIFIED` | RX 7900 XTX / ROCm | In-VRAM Parameter Pinning: exact InfoNCE adjoint autodiff lowered into device VRAM. |
| **`e14`** | Gate 3 | $G_0$ | [`gate3-evals/`](gate3-evals/) | `REFUTED` | RX 7900 XTX / ROCm | Distractor Diagnostic: proves linear relational unbinding cannot select discrete truth ($1.0\%$ Top-1). |
| **`e15`** | Gate 3 | $G_0$ | [`gate3-evals/`](gate3-evals/) | `VERIFIED` | RX 7900 XTX / ROCm | Gemma 4 Grafting: confirms Linear Unbinding Theorem on 2.3B parameter LLM ($+0.38$ logits). |
| **`e16`** | Gate 3 | $G_0$ | [`gate3-evals/`](gate3-evals/) | `REFUTED` | RX 7900 XTX / ROCm | Contrastive Resolver on Frozen LLM: closed line; gradients learn entity rankings, not discrete selection. |
| **`e17`** | Gate 3 | $G_0$ | [`gate3-evals/e17-predicate-invention/`](gate3-evals/e17-predicate-invention/) | `REFUTED` | RX 7900 XTX / ROCm | End-to-End Gradient Predicate Invention: continuous relaxation fails discrete compositional unidentifiability. |
| **`e18`** | Gate 3 | $G_0$ | [`gate3-evals/e18-diagnostics/`](gate3-evals/e18-diagnostics/) | `REFUTED` | RX 7900 XTX / ROCm | Phase 0/1 2x2 Factorial Sweep: definitive refutation of gradient resolver under varying temperature. |
| **`e19`** | Gate 3 | $G_0$ | [`gate3-evals/e19-hybrid-crystallization/`](gate3-evals/e19-hybrid-crystallization/) | `REFUTED` | RX 7900 XTX / ROCm | Hybrid Crystallization: gradient search with temperature decay fails discrete predicate invention. |
| **`e20`** | Gate 3 | $G_0$ | [`gate3-evals/e20-constrained-crystallization/`](gate3-evals/e20-constrained-crystallization/) | `REFUTED` | RX 7900 XTX / ROCm | Constrained Crystallization: proves discrete priors are required; permanently closes gradient invention line. |
| **`e21`** | Gate 4 | $G_0$ | [`gate4-recursion/e21-kb-write-path/`](gate4-recursion/e21-kb-write-path/) | `VERIFIED` | RX 7900 XTX / ROCm | Schema-Constrained KB Write Path: **0 illegal role assignments** commit to memory under adversarial inputs. |
| **`e22`** | Gate 4 | $G_0$ | [`gate4-recursion/e22-ingraph-dispatch/`](gate4-recursion/e22-ingraph-dispatch/) | `VERIFIED` | RX 7900 XTX / ROCm | In-Graph Query Dispatch: $100\%$ tool-token emission; $42.20\ \mu\text{s}$ host vs $691.94\ \mu\text{s}$ device lookup ($16.4\times$ host advantage). |
| **`e23`** | Gate 4 | $G_0$ | [`gate4-recursion/e23-state-reduce/`](gate4-recursion/e23-state-reduce/) | `VERIFIED` | RX 7900 XTX / ROCm | Verified State Reduce: 10-turn adversarial horizon; **0 RBAC invariant violations**; bounded memory accumulation. |
| **`e24`** | Gate 1 | $G_0$ | [`gate1-compression/e24-prefix-cache-handover/`](gate1-compression/e24-prefix-cache-handover/) | `VERIFIED` | RX 7900 XTX / ROCm | In-VRAM Prefix-Cache Handover: **74.46 ms re-prefill eliminated** (0.04 ms transition); **0 duplicate weights allocated**; **+3.3% self-correction lift**. |

---

## 3. Pod Directory Structure

Each fully verified pod in `catalog/` contains:
```
catalog/<gate>/<pod-name>/
├── spec.md              # Falsifiable pre-registered protocol (RFC headers, hypotheses, metrics)
├── run.clj              # Executable standalone harness
├── results.edn          # Machine-readable output metrics
└── summary.csv          # Tabular benchmark sweeps (where applicable)
```

---

## 4. Executing Experiment Pods

Pods can be executed directly using Clojure CLI aliases or runner paths:

```bash
# Execute Gate 4 In-Graph KB Query Dispatch (E22)
clojure -M:e22

# Execute Gate 4 Verified RBAC State Reduce (E23)
clojure -M:e23

# Execute Gate 1 In-VRAM Prefix-Cache Handover (E24)
clojure -M:e24

# Execute Gate 3 Predicate Invention Sweep (E17)
clojure -M:experiments -i catalog/gate3-evals/e17-predicate-invention/run.clj -m einsum.experiments.e17-predicate-invention.run
```

---

## 5. The Research Lifecycle (IETF RFC Protocol)

All new experiments must follow the 3-stage research protocol outlined in [CONTRIBUTING.md](../CONTRIBUTING.md):
1. **Stage 1 (Proposal):** Under `proposals/<gate>/<slug>/spec.md` with required RFC headers.
2. **Stage 2 (Implementation):** Working code and `results.edn` merged as `UNVERIFIED`.
3. **Stage 3 (Verification):** Independent physical silicon replication and permanent minting of sequential `eNN` ID in `registry.edn`.
