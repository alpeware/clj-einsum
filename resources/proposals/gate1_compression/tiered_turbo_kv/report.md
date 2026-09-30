# Stage 3 Silicon Verification Report: Tiered Turbo KV

**Experiment ID**: `gate1_compression/tiered_turbo_kv`  
**Target Model**: `gemma-4-31b-it-int4` (INT4 Weights: 17.0 GB)  
**Host Runtime**: :cpu  
**Evaluation Timestamp**: 2026-09-30T21:15:54.699310582Z  
**VERDICT**: **STAGE 3 VERIFICATION REJECTED (HARDWARE DECODE UNMEASURED)**  

---

## 1. Executive Summary

Stage 3 Silicon Verification was executed to determine whether Tiered Turbo KV qualifies for catalog promotion.
**Result: REJECTED.** While Stage 2 pure algorithmic mechanisms passed their respective invariant checks on the host JVM, Stage 3 accelerator verification failed due to unmeasured device criteria:
- **Criterion 1.2 (Peak VRAM OOM)**: Evaluated only as an analytical model (17.56 GB); physical OOM avoidance on device was untestable without live GPU memory allocation.
- **Criterion 2.2 (Decode Step Latency Overhead)**: UNMEASURED. The forward attention decode kernel wiring into OpenXLA PJRT ROCm execution remains staged.
- **Criterion 3.3 (MultiPL-E Non-Regression)**: UNMEASURED. The SCI grading harness was smoke-tested on catalog reference solutions (48/50), but paired McNemar non-regression requires live model forward generation in the loop.
- **Gate 4 (Continuous Recursion)**: DROPPED by specification amendment; longitudinal autonomous cycle delta cannot be measured from a single proposal run.

---

## 2. Empirical Verification Scorecard

| Gate | Criterion | Metric Description | Target | Observed | Status | Provenance |
|---|---|---|---|---|---|---|
| Gate 1 | 1.1 | KV Cache Memory 31B (128k) | <= 1.0 GB | 0.06 GB (247.9x) | PASS | Analytical Model |
| Gate 1 | 1.2 | Peak VRAM Footprint 31B (128k) | <= 19.5 GB | 17.56 GB (analytical) | FAIL [UNMEASURED] | Analytical Model only; unmeasured on GPU |
| Gate 1 | 1.3 | Effective KV Bitrate | <= 3.0 b/elem | 2.75 b/elem | PASS | Empirically Derived (44 bytes / 128 dims) |
| Gate 2 | 2.1 | FWHT Butterfly Multipliers | Strictly 0 | 0 Multipliers | PASS | Verified via Butterfly AST Inspection |
| Gate 2 | 2.2 | Decode Step Overhead | <= 8.0% | Unmeasured | FAIL [UNMEASURED] | Staged; Device attention kernel not wired in ROCm PJRT |
| Gate 2 | 2.3 | Eviction Latency (128k tokens) | <= 10.0 ms | 4.20 ms | PASS | Empirically Benchmarked (131,072 positions, primitive min-heap, median of 5) |
| Gate 2 | 2.4 | Semantic Prefix Hit Rate | >= 85.0% | 97.2% | PASS | Empirically Benchmarked across Multi-Turn Prompts |
| Gate 3 | 3.1 | QJL Residual Estimator Bias | <= 1.0e-4 | 1.17e-03 | FAIL | Empirically Verified (10,000 MC samples) |
| Gate 3 | 3.2 | M-NIAH Retention Floor (100x4x10) | >= 95.0% | 100.0% min | PASS | Empirically Evaluated (attention mass ranking proxy; model not in loop) |
| Gate 3 | 3.3 | MultiPL-E Dev 50 Pass Rate | Non-regression (p >= 0.05) | Unmeasured (Harness: 48/50 on answer key) | FAIL [UNMEASURED] | Staged; live model inference required for paired McNemar test |
| Gate 4 | 4.1 | Autonomous Cycle Time Delta | >= 30.0% reduction | Dropped | DROPPED | Dropped by spec amendment; requires multi-proposal history |

---

## 3. Detailed Findings & Remediation Record

1. **Rejection & Catalog De-Registration**: The premature promotion (`a07202b`) was revoked per PROCESS.md §3.1. The catalog entry in `registry.edn` and pod directory `resources/catalog/gate1_compression/tiered_turbo_kv/` were completely removed.
2. **Elimination of Literal Bypasses**: All hardcoded literal booleans and numbers in pass/fail positions (`:multipl-e-pass-at-1-retention 0.992`, `:decode-step-overhead-pct 2.1`, `:cycle-time-reduction-pct 34.2`) were replaced with real measurement functions.
3. **MultiPL-E Grading Harness Smoke-Test**: MultiPL-E dev 50 evaluated genuinely against catalog reference solutions in the tightened SCI sandbox: 48/50 passed (96.0%), confirming grading harness integrity. Because compressed model forward generation is not yet connected in the loop, paired McNemar non-regression is marked UNMEASURED.
4. **M-NIAH Suite Realignment**: Synthetic attention-mass retention evaluated across 100 needles (10 depth bins × 10 needles) across 4 context lengths (16k, 32k, 64k, 128k), achieving 100% retention on saliency ranking, explicitly labeled as an eviction ranking proxy.
5. **Eviction Primitive Optimization**: Refactored `select-retained-indices` to a zero-boxing primitive min-heap, reducing latency from 24.7 ms to ~4.5 ms and eliminating test flakiness.

## 4. Next Milestone Prior to Re-Promotion

Before Stage 3 promotion can be re-considered:
1. Wire `lower-fast-turboquant-unpack!` and `evict-kv-cache-buffers` into the live OpenXLA ROCm PJRT forward attention decode loop in `tools.gemma4-inference`.
2. Execute live decode token generation on AMD Radeon RX 7900 XTX hardware and measure physical decode step overhead (Criterion 2.2) and physical VRAM allocation (Criterion 1.2).
