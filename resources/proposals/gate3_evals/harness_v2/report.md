# Verification Report: Eval Harness v2

**Capability**: `harness_v2` \
**Gate**: `gate3_evals` \
**Type**: `capability` \
**Status**: `in-progress` \
**Target Hardware**: AMD Radeon RX 7900 XTX (24GB) \
**Verification Commit**: `d1451fae34a0231f98d66e93b2f55f844570443b` \
**Date**: 2026-09-27

---

## 1. Executive Summary

RFC [`resources/proposals/gate3_evals/harness_v2/spec.md`](spec.md) proposed four runtime and harness optimizations to resolve the evaluation bottleneck (~70% runaway generation decoding waste and full prompt re-prefill tax per turn) ahead of the 1,116-cell MultiPL-E port.

Initial stage 3 verification revealed that while AC1, AC4, AC5, and AC6 passed, **AC2 failed**: 6 of 20 cells flipped from PASS to FAIL due to premature chunked semantic early stopping cutting off model reasoning and intermediate drafts. The capability is **reopened for remediation**: disabling chunked semantic early stopping (Optimization 2) to evaluate the speedup of Optimizations 1, 3, and 4 (KV cache prefix reuse, early exit, nudge short-circuit) without altering model reasoning.

---

## 2. Acceptance Criteria Verification Matrix

| Criterion | Target / Requirement | Empirical Result | Status |
|---|---|---|---|
| **AC1: Prefix KV Cache** | ≥50% per-turn prefill reduction on 3-turn workload | Turn 2: **51.0%** reduction (1531.5ms → 750.6ms)<br>Turn 3: **73.3%** reduction (2193.8ms → 584.9ms) | **PASSED** |
| **AC2: Graded Outcome Invariance** | Reproduce v1 outcomes exactly (pass/fail per cell) | **6/20 cells regressed from PASS to FAIL** due to premature semantic stopping | **FAILED** |
| **AC3: Wall-Time Reduction** | ≥40.0% reduction on 20-cell reference suite (≤ 1479.1s) | **1,434.03s** vs 2,465.20s v1 baseline (**41.83% reduction**, -1,031.17s; qualified by AC2 failures) | **QUALIFIED** |
| **AC4: Library Residence** | Pure library namespace; no `tools.*` in eval path | Core agent logic residing in `einsum.agent.core`; CLI namespaces decoupled | **PASSED** |
| **AC5: MultiPL-E Dev Subset** | 50 tasks load, grade hermetically in SCI sandbox, valid EDN ledger | 50 tasks cataloged; 50/50 solutions pass public & sealed tests; ledger appends valid EDN | **PASSED** |
| **AC6: Repo Invariants & TDD** | Generative invariant tests, 0 lint warnings, green fast suite | 8 generative property tests; `clojure -M:lint` 0 warnings; 217 tests / 3199 assertions green | **PASSED** |

---

## 3. Empirical Reference Suite Comparison (v1 vs v2)

The 20-cell reference suite was evaluated with `gemma-4-31b-it-qat-int4` at repetition penalty `1.0` and temperature `0.0` on AMD Radeon RX 7900 XTX (24GB).

### 3.1 Per-Cell Detailed Breakdown

| Task | Mode | v1 Outcome | v1 Wall (s) | v2 Outcome | v2 Wall (s) | Wall Delta (s) | Speedup (%) |
|---|---|---|---|---|---|---|---|
| `first-n` | `single-shot` | PASS | 23.6 | PASS | 22.4 | +1.3 | +5.4% |
| `my-range` | `single-shot` | PASS | 67.8 | PASS | 62.2 | +5.6 | +8.3% |
| `deep-flatten` | `single-shot` | FAIL | 140.8 | FAIL | 72.6 | +68.2 | +48.4% |
| `freqs` | `single-shot` | FAIL | 27.9 | FAIL | 72.2 | -44.3 | -158.4% |
| `partition-by-parity` | `single-shot` | PASS | 83.3 | FAIL | 73.9 | +9.4 | +11.3% |
| `my-comp` | `single-shot` | FAIL | 137.2 | FAIL | 72.6 | +64.5 | +47.1% |
| `balanced-delims?` | `single-shot` | FAIL | 104.7 | FAIL | 73.5 | +31.3 | +29.8% |
| `deep-update-vals` | `single-shot` | FAIL | 71.6 | FAIL | 73.7 | -2.0 | -2.9% |
| `lazy-interleave` | `single-shot` | PASS | 112.5 | FAIL | 74.6 | +38.0 | +33.7% |
| `my-or` | `single-shot` | FAIL | 173.9 | FAIL | 73.1 | +100.8 | +58.0% |
| `first-n` | `agentic` | PASS | 56.7 | PASS | 30.0 | +26.7 | +47.0% |
| `my-range` | `agentic` | PASS | 88.8 | PASS | 167.9 | -79.1 | -89.0% |
| `deep-flatten` | `agentic` | PASS | 313.2 | FAIL | 79.0 | +234.2 | +74.8% |
| `freqs` | `agentic` | PASS | 62.2 | FAIL | 89.4 | -27.2 | -43.7% |
| `partition-by-parity` | `agentic` | PASS | 89.5 | PASS | 56.7 | +32.8 | +36.6% |
| `my-comp` | `agentic` | FAIL | 276.7 | FAIL | 65.0 | +211.7 | +76.5% |
| `balanced-delims?` | `agentic` | FAIL | 337.7 | FAIL | 64.9 | +272.8 | +80.8% |
| `deep-update-vals` | `agentic` | PASS | 95.3 | FAIL | 64.8 | +30.5 | +32.0% |
| `lazy-interleave` | `agentic` | PASS | 115.6 | FAIL | 80.9 | +34.6 | +30.0% |
| `my-or` | `agentic` | FAIL | 86.0 | FAIL | 64.8 | +21.3 | +24.7% |

### 3.2 Aggregate Wall-Time Comparison

| Evaluation Mode | v1 Baseline Wall Time (s) | v2 Reference Wall Time (s) | Net Time Saved (s) | Relative Reduction (%) |
|---|---|---|---|---|
| **Single-Shot (10 cells)** | 943.43s | 676.81s | +266.62s | **28.26%** |
| **Agentic Loop (10 cells)** | 1,521.77s | 757.22s | +764.55s | **50.24%** |
| **Combined 20-Cell Suite** | **2,465.20s** | **1,434.03s** | **+1,031.17s** | **41.83%** |

The primary driver of the 50.24% speedup in agentic mode is the elimination of runaway generation loops (via semantic early stopping and nudge short-circuiting) combined with prefix KV-cache reuse.

---

## 4. Optimization Mechanisms & Profiling

### 4.1 Prefix KV-Cache Delta Prefill (AC1)
In OpenXLA PJRT execution, re-prefilling the static prompt on every agent turn consumed ~40ms/token. In v2, `einsum.models.gemma4.runtime/run-vram-loop-generation` detects the longest common token prefix (`p-match`) against previously committed turns in the persistent VRAM session.
- **Turn 1 (Initial Prompt)**: 145 tokens → 1,531.5ms prefill.
- **Turn 2 (Tool Output Appended)**: 247 tokens total, 145 tokens reused → 102 tokens delta prefill → 750.6ms (**51.0% reduction**).
- **Turn 3 (Subsequent Turn)**: 351 tokens total, 247 tokens reused → 104 tokens delta prefill → 584.9ms (**73.3% reduction**).

### 4.2 Chunked Semantic Early Stopping (AC2)
Generations inside `run-vram-loop-generation` decode in static-shape blocks of 32 tokens when `:stop-predicate` is supplied. The predicate parses the newly generated text:
- Checks if generation is inside an unclosed `<|channel>thought` channel.
- If outside the thought channel, extracts fenced Clojure markdown blocks or top-level forms.
- Validates structural completeness via reader balance and `einsum.agent.core/try-parse-sci-reader`.
- Once balanced code or tool call syntax is detected, decoding halts immediately without waiting for EOS or max token limits.

### 4.3 Nudge Short-Circuiting (AC2)
When Turn 1 generation already yields a valid, extractable Clojure candidate form satisfying `candidate-check-fn`, the harness bypasses the blind Turn 2 "Please test your Clojure implementation" nudge, proceeding directly to grading.

### 4.4 Early Exit on Public Pass (AC2)
In the agentic loop, whenever the model's submission passes all public test cases (`:all-passed? true`), `submission-tool-hook` immediately returns `:early-exit? true`. Because the ratchet tie-break favors earlier turns, continuing evaluation cannot improve the final score.

---

## 5. MultiPL-E Dev Subset Catalog (AC5)

To prepare for the full 1,116-cell MultiPL-E evaluation port, a 50-task dev subset was constructed and validated:
- `resources/catalog/gate3_evals/multipl_e/dev_50_public.edn`: Public prompts with docstrings and starter signatures.
- `resources/catalog/gate3_evals/multipl_e/dev_50_sealed.edn`: Sealed test suites for blind execution.
- `resources/catalog/gate3_evals/multipl_e/solutions.edn`: 50 ground-truth verified Clojure solutions.
- Property test `test-multipl-e-dev-50-fixture-and-ledger` validates that all 50 tasks execute end-to-end hermetically in the SCI sandbox and produce valid EDN ledger records stamped with `:harness-sha`.

---

## 6. Verification Checklist & Invariants (AC6)

- [x] **Generative Property Tests**: 8 `clojure.test.check` generative invariants covering reader parsing, thought channel boundaries, code block isolation, and nudge short-circuit logic.
- [x] **Formatting**: Clean code conforming to repository formatting invariants (`clojure -M:format`).
- [x] **Linter**: Zero lint errors and zero warnings (`clojure -M:lint`).
- [x] **Fast Suite**: 217 tests, 3,199 assertions passed with zero failures or errors (`clojure -M:test fast`).
- [x] **Hardware Grounding**: Executed on AMD Radeon RX 7900 XTX (24GB VRAM) via `tools/gemma4.sh` with `libjsig.so` preloading.

---

## 7. Disposition

The capability `harness_v2` is **reopened** (`status: in-progress`). Stage 3 verification identified that chunked semantic early stopping (Optimization 2) violated AC2 by cutting off reasoning drafts and causing 6 cell regressions. Remediating by evaluating with semantic stopping disabled while keeping prefix KV-cache reuse, public test early exit, and nudge short-circuit active.
