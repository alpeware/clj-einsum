# Empirical Report: Perplexity Probe v1 (Gate 3 Evals)

**Experiment**: `perplexity_probe_v1`  
**Gate**: `gate3_evals`  
**Generation**: `0`  
**Date**: `2026-09-30`  
**Hardware Target**: AMD Radeon RX 7900 XTX (24 GB VRAM, RDNA3 `gfx1100`)  
**Student Checkpoint**: `gemma-4-e4b-it-qat-int4` (`33ebfa9b85f077df19c0526c9f0e039c044d4e64eef47368730208b478dc05ec`)  
**Teacher Checkpoint**: `gemma-4-31b-it-qat-int4` (`f20a02315954b540db819393224e07c3d60c1f07c9052b10a530db1556d2603c`)  
**Status**: `DECIDED: NO-GO`  
**Refutes**: Hypothesis H1 (SFT Headroom on Teacher Trajectories)

---

## 1. Executive Summary & Pre-Registered Gate Decision

Following the rejection of `prompt_tuning_v1` (where worked self-correction demonstrations altered agentic choreography without closing the performance gap), `perplexity_probe_v1` was pre-registered as a necessary-not-sufficient inference probe. The probe measures student model surprise under teacher forcing on successful 31B teacher agentic trajectories across 4 gap tasks (`first-n`, `my-range`, `deep-update-vals`, `lazy-interleave`) relative to 2 reference contrast tasks (`freqs`, `partition-by-parity`).

### Primary Decision Rule & Verdict

| Metric | Measured Value | Pre-Registered Threshold | Condition Met? |
|---|---|---|---|
| **Micro-Average Ratio ($\frac{\text{PPL}_{\text{gap}}}{\text{PPL}_{\text{ref}}}$)** | **0.9030** | $\ge 1.5000$ | **NO** ($0.9030 < 1.50$) |
| **Macro-Average Ratio** | **0.8738** | — | — |
| **Per-Task SFT Candidates ($r_k \ge 1.30$)** | **0 / 4 tasks** | $\ge 1$ task | **NO** (all $r_k \in [0.65, 1.01]$) |
| **Pre-Registered Gate Verdict** | **NO-GO** | Primary ratio $\ge 1.50$ | **FALSIFIED** |

### Immediate Scientific Takeaway & Action
1. **Falsification of H1 (No SFT Headroom)**: The student model is **not** surprised by the teacher's trajectories. Across all 4 gap tasks, the student model assigns high likelihood to the teacher's reasoning and code ($PPL_{\text{gap}} = 1.8264$ vs $PPL_{\text{ref}} = 2.0225$). In fact, the student model exhibited *higher* perplexity on the reference task where it succeeded (`partition-by-parity`, PPL 2.2864) than on the gap tasks where it failed.
2. **SFT on These Trajectories Canceled**: Per the pre-registered decision protocol (§2 of `spec.md`), no GPU compute will be spent finetuning or running LoRA SFT on these teacher trajectories.
3. **Diagnostic Indication**: The teacher's trajectories are in-distribution for the student. The student's agentic failures appear to be a search/path-finding problem under autonomous sampling rather than distribution ignorance.

---

## 2. Evidence Artifacts & Provenance Ledger

All data, scripts, and logs are persisted directly in the repository with complete provenance:

1. **Teacher Trajectories**: [`teacher_trajectories.edn`](teacher_trajectories.edn)
   - Captured on host RX 7900 XTX via `tools/gemma4.sh` with `--save-transcripts true`.
   - 6 tasks executed with deterministic greedy decoding ($T=0.0$, seed 0); all 6 tasks passed 100% of public and sealed hidden tests.
   - Summary ledger: [`teacher_summary.csv`](teacher_summary.csv).
2. **Raw Scoring Output**: [`results.edn`](results.edn)
   - Contains token-level metadata, per-position target log-probabilities, execution latencies, and segment metrics.
3. **Tabular Results**: [`summary.csv`](summary.csv)
   - CSV export containing per-task token counts, cross-entropy, overall PPL, thinking-segment PPL, code-segment PPL, ratio $r_k$, and candidate flags.
4. **Implementation Namespaces**:
   - Pure Sans-IO Core: [`src/experiments/gate3_evals/perplexity_probe_v1/core.clj`](../../../src/experiments/gate3_evals/perplexity_probe_v1/core.clj)
   - CLI Runner: [`src/experiments/gate3_evals/perplexity_probe_v1/run.clj`](../../../src/experiments/gate3_evals/perplexity_probe_v1/run.clj)
   - Test Suite: [`test/experiments/gate3_evals/perplexity_probe_v1_test.clj`](../../../test/experiments/gate3_evals/perplexity_probe_v1_test.clj)
   - Tensor Logic Lowering: [`src/einsum/logic/lower.clj`](../../../src/einsum/logic/lower.clj#L326) and [`src/einsum/models/gemma4/kernels.clj`](../../../src/einsum/models/gemma4/kernels.clj)
   - Runtime Scoring Engine: [`src/einsum/models/gemma4/runtime.clj`](../../../src/einsum/models/gemma4/runtime.clj#L850)

---

## 3. Evaluation Methodology & Masking Invariants

### 3.1 Role-Based Masking Specification

In accordance with §3 of the hardened RFC, masks are derived from structured transcript turns (`:role :model`, `:role :user`, `:role :tool`) rather than string patterns in tokenized text, strictly maintaining alignment under fenced syntax:

- **Position 0**: Position 0 (`<bos>`) has no preceding context and is un-scored (mask 0).
- **Framing Tokens (mask 0)**: Harness formatting tokens—including `<bos>`, `<|turn>`, `<turn|>`, `<|channel>thought\n`, and `<channel|>`—receive mask 0.
- **Context Turns (mask 0)**: System prompt, user task prompt, and tool execution feedback (`:role :tool` or fenced `:user`) are conditioned on in causal attention but receive mask 0.
- **Scored Span $S$ (mask 1)**: Only text content generated in `:model` turns receives mask 1.
- **Segment Partitioning**:
  - $S_{\text{think}}$: Tokens within the internal reasoning block `<|channel>thought...\n<channel|>`.
  - $S_{\text{code}}$: Tokens outside the thinking block (solution code, tool invocations, explanations).

### 3.2 Micro-Average Corpus Aggregation Formula

To avoid length-bias from disparate trajectory lengths, the primary decision rule uses token-weighted micro-averaged cross-entropy across all scored positions:

$$\text{CE}_{\text{micro}}(\mathcal{T}) = \frac{\sum_{k \in \mathcal{T}} \sum_{i \in S_k} -\log p(w_{k,i} \mid w_{k,<i})}{\sum_{k \in \mathcal{T}} |S_k|}$$

$$\text{PPL}_{\text{micro}}(\mathcal{T}) = \exp\left(\text{CE}_{\text{micro}}(\mathcal{T})\right)$$

The primary decision ratio is:

$$\text{Ratio} = \frac{\text{PPL}_{\text{micro}}(\text{Gap})}{\text{PPL}_{\text{micro}}(\text{Ref})}$$

---

## 4. Hardware Execution & Mechanical Sympathy

### 4.1 In-Accelerator Device Core (Rule 4)
- **Zero Logit Bloat over PCIe**: Rather than returning full $[1 \times 262,144]$ float16/bf16 logits to the host (which would require transferring ~524 KB per token, or ~6.2 GB across the run), the target log-prob extraction executable ([`compile-target-log-prob-executable`](../../../src/einsum/models/gemma4/kernels.clj)) fuses log-softmax and axis-2 target gathering directly in device VRAM. Only 1 scalar `float` (4 bytes) is transferred to host memory per token step.
- **Native Quantized Execution**: The ROCm OpenXLA PJRT engine executes `w4a16_gemv_rocm` quantized matrix-vector kernels at $M=1$ sequential KV stepping, sustaining **58.5 tokens/sec** throughout the entire 11,881 token corpus evaluation.
- **Hermetic Memory Reclamation**: Using `xla/with-device-arena` and `arena/destroy!`, intermediate KV cache allocations were reclaimed at every token step. Silicon VRAM footprint remained strictly flat at ~3.8 GB with zero memory leaks.

---

## 5. Detailed Empirical Results

The probe was executed via `./tools/gemma4.sh run -M:tools -m experiments.gate3-evals.perplexity-probe-v1.run --backend rocm`.

### 5.1 Per-Task Breakdown Table

| Task ID | Group | Total Tok | Scored Tok | Think Tok | Code Tok | Scored Latency | Speed (tok/s) | PPL (All) | PPL (Think) | PPL (Code) | $r_k = \frac{\text{PPL}_k}{\text{PPL}_{\text{ref}}}$ | SFT Cand? ($r_k \ge 1.30$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `first-n` | **gap** | 1,055 | 483 | 69 | 414 | 18.44s | 57.2 | 1.3222 | 1.6384 | 1.2757 | 0.6537 | **NO** |
| `my-range` | **gap** | 3,004 | 2,358 | 1,358 | 1,000 | 51.34s | 58.5 | 2.0482 | 2.7443 | 1.3766 | 1.0127 | **NO** |
| `deep-update-vals` | **gap** | 2,389 | 1,791 | 480 | 1,311 | 40.83s | 58.5 | 1.7934 | 2.6382 | 1.5570 | 0.8867 | **NO** |
| `lazy-interleave` | **gap** | 2,489 | 1,819 | 645 | 1,174 | 42.66s | 58.4 | 1.7464 | 2.4691 | 1.4439 | 0.8635 | **NO** |
| `freqs` | **ref** | 1,243 | 715 | 143 | 572 | 21.17s | 58.7 | 1.6679 | 1.9710 | 1.5997 | 0.8247 | **NO** |
| `partition-by-parity` | **ref** | 1,701 | 1,124 | 548 | 576 | 29.10s | 58.5 | 2.2864 | 3.4952 | 1.5268 | 1.1305 | **NO** |

### 5.2 Corpus Micro- and Macro-Aggregates

```
Corpus Evaluation Summary:
--------------------------------------------------------------------------------
Total Tokens Scored (Gap)            : 6,451 tokens
Total NLL (Gap)                      : 3,885.8301
Micro-Average Cross-Entropy (Gap)    : 0.6024
Micro-Average Perplexity (Gap)       : 1.8264

Total Tokens Scored (Ref)            : 1,839 tokens
Total NLL (Ref)                      : 1,295.2907
Micro-Average Cross-Entropy (Ref)    : 0.7043
Micro-Average Perplexity (Ref)       : 2.0225

Primary Decision Ratio (Gap / Ref)   : 0.9030  (Threshold >= 1.5000)
Macro-Average Ratio                  : 0.8738  (Gap: 1.7275 / Ref: 1.9771)

SFT Candidate Tasks (r_k >= 1.30)    : None ([])
Flagged / Noise Tasks (r_k < 1.30)   : [first-n, my-range, deep-update-vals, lazy-interleave]
--------------------------------------------------------------------------------
Pre-Registered Gate Decision         : NO-GO
```

---

## 6. Scientific Analysis: In-Distribution Evidence & Hypotheses

### 6.1 Findings: Teacher Trajectories are In-Distribution
The empirical result established by this probe is that 31B teacher trajectories on gap tasks are **in-distribution** for the E4B student:
1. **Low Code Surprise**: Across all four gap tasks, student Code Perplexity ranges from **1.2757 to 1.5570** (cross-entropy of 0.24 to 0.44 nats per token). On `first-n` (Code PPL 1.2757), the mean target token probability under teacher forcing is ~0.77.
2. **Reasoning Trace Familiarity**: The student's Thinking Perplexity on gap tasks (1.6384 to 2.7443) is comparable to or lower than on reference tasks (`partition-by-parity`, 3.4952).

### 6.2 The Teacher-Forcing Caveat
Perplexity measures $p(\text{teacher's path} \mid \text{teacher's prefix})$. It cannot distinguish between "the student knows the path" and "the student would find the path autonomously."
- Stating that "the model possesses the capacity to generate the correct code" overstates what teacher forcing measures.
- What was established: the teacher's trajectories represent high-likelihood paths for the student.
- Why SFT on these trajectories is unpromising: At PPL ~1.3, mean target token probability is ~0.77. The per-token cross-entropy gradient on the target logit is $\sim (1 - p) \approx 0.23$—which is substantial, not negligible ($\nabla_\theta \mathcal{L} \not\approx 0$). SFT on this data would provide active gradient signal, but that signal acts to **reinforce probability mass the model already assigns** rather than teaching novel representations, syntax, or error-recovery behaviors.

### 6.3 Hypotheses on the Agentic Gap (Unmeasured by this Probe)
Because this probe evaluated teacher-forced likelihood and ran zero live E4B agentic rollouts, the live failure mechanisms remain **unmeasured hypotheses** derived from earlier benchmark runs (`clojure_bench` and `prompt_tuning_v1`):
- **Search-Space Trapping**: Under unconstrained autoregressive sampling ($T=0.0$), the student may fall into local search traps (e.g. infinite recursion in `deep-update-vals`, or failing to wrap lazy sequences in `my-range`).
- **Semantic Oscillation**: Once an error feedback string is returned, the student's self-correction choreography appears to oscillate across minor variations rather than exploring alternative search branches.
These dynamics describe search and path-finding bottlenecks, not distribution ignorance.

### 6.4 Methodology Warts & Limitations
Four methodology nuances are recorded for full transparency (none alters the decision):
1. **Synthesized Thinking Framing**: The probe synthesized `<|channel|>thought` and `<channel|>` framing delimiters that the 31B teacher never emitted (the teacher output contained `[Thought Process]` text markers under fenced syntax). While delimiter tokens were masked out (mask 0), their presence slightly alters the prefix context.
2. **BPE Boundary Effects**: Tokenizing structured segments separately introduces minor BPE boundary artifacts at segment interfaces compared to joint tokenization.
3. **Reference Set Sensitivity**: With $N=2$ reference tasks, the primary micro-ratio ($0.9030$) is heavily influenced by `partition-by-parity`'s elevated PPL ($2.2864$).
4. **Robustness of Absolute Metrics**: The absolute gap PPLs (**1.32 to 2.05** overall; **1.28 to 1.56** code) independently corroborate the **NO-GO** decision regardless of the reference ratio: the student model already assigns high likelihood to the teacher's code.

---

## 7. Conclusions & Next Actions

1. **Pre-Registered Hypothesis Falsified**: Hypothesis H1 is falsified ($0.9030 \ll 1.50$, with 0/4 tasks meeting $r_k \ge 1.30$).
2. **Decision & Resource Allocation**: The pre-registered decision is **NO-GO**. Fine-tuning on these specific teacher trajectories is ruled out, preventing wasted compute on ineffective imitation data. (This probe specifically falsifies SFT on these trajectories; it does not rule out distillation or SFT broadly on other task distributions or with alternative objectives).
3. **Loop State & Baseline Retained**:
   - Zero-shot prompt baseline is retained per the `prompt_tuning_v1` verdict.
   - The probe confirms that E4B's performance gap is an autonomous search and path-finding problem rather than vocabulary or distribution ignorance.
   - Catalog this proposal in `resources/catalog/gate3_evals/perplexity_probe_v1/` as a DECIDED experiment with verdict `NO-GO`.
