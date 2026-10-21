# Golden Battery & Final vs-XLA Benchmark Archive (`resources/golden/`)

This directory contains the definitive golden regression oracle (`gemma4_uop_oracle.edn`) and the historical vs-OpenXLA benchmark performance archive (`xla_vs_uop_archive.edn`) captured prior to the deprecation and removal of the OpenXLA/PJRT C API backend in `clj-einsum`.

All hardware metrics and golden outputs were captured live on an **AMD Radeon RX 7900 XTX 24GB (RDNA 3 / gfx1100)** with ROCm 6.0/7.2 running against both `:uop-rocm` and `:rocm` (OpenXLA PJRT).

---

## 1. Archived Artifacts

1. **`gemma4_uop_oracle.edn`**:
   - **Gemma 4 E2B (`.models/gemma-4-E2B-it`, BF16)**:
     - Exact greedy token continuations across the Part V prompts: Prompt 1 ($M=14$), Prompt 1b ($M=17$), Prompt 2 ($M=32$).
     - Step 0 (prefill) and Step 1 (decode) raw float32 logit vectors (`:first-32-logits` and sorted `:top-16`).
   - **Gemma 4 E4B (`.models/gemma-4-E4B-it`, BF16)**:
     - Exact greedy token continuations across Prompts 1, 1b, 2.
     - Step 0 and Step 1 raw float32 logits.
   - **Gemma 4 31B INT4 (`.models/gemma-4-31b-it-int4`, GPTQ)**:
     - Prompts 1, 1b, 2 continuations and step logits.
     - Prefill sweep across $M \in \{2, 8, 14, 15, 16, 32\}$ prompt lengths, including next-token IDs, raw logit snapshots ($M=14, 32$), and multi-token greedy continuation ($M=16$).
   - **Micro-Gemma-4 Deterministic Regression Oracle**:
     - 1-layer BF16 and 1-layer INT4 GPTQ micro models with `hidden_size=32, intermediate_size=64, num_attention_heads=2, num_kv_heads=1, head_dim=16, vocab_size=64, max_seq_len=16`.
     - Seeded PRNG generated weights.
     - Compiled and executed on `:uop-cpu` via `compile-gemma4-chunked-prefill-executable` and `compile-gemma4-kv-executable`.
     - Full 4-token greedy sequence `[30 30 30 30]` and raw float32 logits for Step 0 and Step 1.

2. **`xla_vs_uop_archive.edn`**:
   - Final live benchmark comparison table for E2B, E4B, and 31B INT4 on AMD Radeon RX 7900 XTX.
   - Historical prompt-scaling archive from Phase 6D (17 to 456 tokens prefill scaling and 99-token decode throughput).
   - Phase 7 456-token E4B comparison and 31B INT4 M-sweep scaling data.

---

## 2. Performance Comparison Summary (RX 7900 XTX `gfx1100`)

### Final Live Benchmarks (`:uop-rocm` vs `:rocm` OpenXLA PJRT)

| Model | Prompt | Length ($M$) | UOp Prefill | XLA Prefill | Prefill Speedup | UOp Decode | XLA Decode | Decode Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4 E2B (BF16)** | Prompt 1 ("The capital...") | 14 | **32.73 ms** | 109.32 ms | **3.34x** | **59.12 tok/s** | 45.07 tok/s | **1.31x** |
| | Prompt 1b ("Explain monads...") | 17 | **32.43 ms** | 51.15 ms | **1.58x** | **90.06 tok/s** | 56.10 tok/s | **1.61x** |
| | Prompt 2 ("Explain difference...") | 32 | **33.03 ms** | 52.12 ms | **1.58x** | **92.03 tok/s** | 56.02 tok/s | **1.64x** |
| **Gemma 4 E4B (BF16)** | Prompt 1 | 14 | **43.14 ms** | 190.01 ms | **4.40x** | **41.37 tok/s** | 26.18 tok/s | **1.58x** |
| | Prompt 1b | 17 | **40.84 ms** | 188.50 ms | **4.62x** | **54.45 tok/s** | 26.08 tok/s | **2.09x** |
| | Prompt 2 | 32 | **42.01 ms** | 190.86 ms | **4.54x** | **54.26 tok/s** | 26.14 tok/s | **2.08x** |
| **Gemma 4 31B (INT4 GPTQ)** | Prompt 1 (6 decode tok) | 14 | **128.65 ms** | 145.18 ms | **1.13x** | **22.01 tok/s** | 19.36 tok/s | **1.14x** |
| | Prompt 1b (60 decode tok) | 17 | **128.23 ms** | 150.84 ms | **1.18x** | **20.38 tok/s** | 20.43 tok/s | **1.00x (Parity)** |
| | Prompt 2 (60 decode tok) | 32 | **131.81 ms** | 190.74 ms | **1.45x** | **20.48 tok/s** | 20.39 tok/s | **1.00x (Parity)** |

### Historical Scaling (Phase 6D Gemma 4 E2B BF16)

| Prompt Tokens | UOp Prefill Latency | OpenXLA PJRT Latency | Speedup | UOp Prefill Rate |
| :---: | :---: | :---: | :---: | :---: |
| 17 | **40.31 ms** | 72.77 ms | **1.81x** | 2.37 ms/tok |
| 40 | **45.67 ms** | 78.23 ms | **1.71x** | 1.14 ms/tok |
| 72 | **47.20 ms** | 79.52 ms | **1.68x** | 0.66 ms/tok |
| 136 | **49.49 ms** | 82.06 ms | **1.66x** | 0.36 ms/tok |
| 264 | **60.45 ms** | 83.24 ms | **1.38x** | 0.23 ms/tok |
| 392 | **67.35 ms** | 82.93 ms | **1.23x** | 0.17 ms/tok |
| 456 | **71.11 ms** | 82.06 ms | **1.15x** | 0.156 ms/tok |
| **Decode (99 tok)** | **90.03 tok/s** | 53.50 tok/s | **1.68x** | — |

---

## 3. Fast Regression Testing (`test/einsum/uop/golden_test.clj`)

The fast test suite (`clojure -M:test fast`) runs `einsum.uop.golden-test` on any CPU host (no GPU or model weights required) in < 5 seconds:

1. **Oracle Schema & Invariant Integrity (`test-golden-oracle-completeness-and-parity`)**:
   - Asserts that all models (`:gemma-4-e2b-it`, `:gemma-4-e4b-it`, `:gemma-4-31b-it-int4`, and `:micro-gemma4-oracle`) are present and complete.
   - Asserts 100% token sequence match between `:uop-rocm` and the reference XLA runs.
   - Asserts strict descending order of `:top-16` / `:top-8` logits and verifies that the top-1 predicted token strictly equals the greedy generated next-token ID across all prompts and sweeps.

2. **Micro-Model End-to-End Regression (`test-micro-gemma4-bf16-and-int4-uop-golden-regression`)**:
   - Reconstructs the 1-layer BF16 and 1-layer INT4 GPTQ micro models with deterministic seeded PRNG weights.
   - Compiles and executes chunked prefill and 3 decode steps via UOp on `:uop-cpu`.
   - Asserts bit-level or tight numerical tolerance ($< 10^{-4}$ max absolute error) on raw float32 logits and exact token match (`[30 30 30 30]`).

3. **Top-K Generative Invariant (`prop-golden-logit-top-k-invariants`)**:
   - `clojure.test.check` property verifying that for any randomly sampled prompt step in the oracle, the top-k logit sequence is strictly monotonically decreasing and the top-1 logit matches the greedy argmax.

### Full Hardware Verification on ROCm

To re-verify golden tokens live against the full weight models on the RX 7900 XTX:
```bash
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "The capital of France is" --max-new-tokens 20 --temperature 0.0
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E4B-it --prompt "The capital of France is" --max-new-tokens 20 --temperature 0.0
./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-31b-it-int4 --prompt "The capital of France is" --max-new-tokens 6 --temperature 0.0
```
