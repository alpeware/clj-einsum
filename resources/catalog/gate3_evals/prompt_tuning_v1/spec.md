# RFC: Prompt Tuning v1: Worked Agentic Trajectories

**Experiment**: `prompt_tuning_v1` \
**Gate**: `gate3_evals` \
**Generation**: `0` \
**Type**: `experiment` \
**Status**: `rejected` \
**Literature**: `["Wei et al. (2022) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arXiv:2201.11903)", "Yao et al. (2022) ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629)", "Gemma Team (2025/2026) Gemma 4: Open Models for Advanced Reasoning and Agentic Workflows", "Cassano et al. (2023) MultiPL-E: A Scalable and Polyglot Benchmark for Evaluating Large Language Models (arXiv:2208.08227)"]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape ">= +5.0% solve rate on MultiPL-E dev (447 tasks) for E4B, zero regressions on clojure_bench sealed-10, prompt prefix <= 600 tokens"}` \
**Extends**: `"harness_v2"` \
**Refutes**: `nil` \
**Supersedes**: `nil` \
**Reopens**: `nil`

---

## 1. Abstract & Motivation

The Eval Harness v2 capability established that syntactic alignment and prompt framing dominate agentic performance in Clojure. Transitioning from JSON-escaped native tool-calling schemas to markdown fenced code blocks (`--tool-syntax :fenced`) lifted the student substrate (`gemma-4-E4B-it-qat-int4`) from 20% native agentic (30% native cross-mode union) to **50% agentic pass rate (5/10)**, while unlocking `my-comp` on the teacher substrate (`gemma-4-31b-it-qat-int4`, achieving **70% agentic pass rate**).

However, our current agentic system prompt (`AGENT-SYSTEM-PROMPT-FENCED`) remains purely prescriptive:
```text
You are an expert Clojure engineer with access to an interactive Clojure REPL sandbox.
Keep internal reasoning very concise (1-2 paragraphs max): plan your logic briefly, do not write full code drafts in thought, and wrap your Clojure code in ```clojure ... ``` tags to evaluate and test your code against public examples.
If any tests fail, inspect the failure errors, revise your implementation, and output revised code in ```clojure ... ``` to re-test.
Only provide your final response once your definition passes all public tests.
```

Error analysis of failed agent runs reveals that models fail not necessarily from mathematical incapacity, but from distinct agentic choreography failure modes:
1. **Thinking Paralysis & Draft Traps**: Small or heavily quantized models (e.g. E4B-QAT) often exhaust turn budgets inside `<|channel>thought` by drafting and redrafting broken code instead of emitting an executable tool call.
2. **Single-Turn Surrender**: Upon receiving failing assertions in ````clojure_result ... ````, models frequently surrender: outputting plain text explanations without revised code, repeating identical failing code verbatim, or losing track of the required signature.
3. **Execution Gap in Error Diagnosis**: When an exception occurs (e.g. `ClassCastException`, `NullPointerException`), models often do not understand how to interpret the error message to localize the bug to paren placement or edge-case handling.

### The Objective & Substrate Hierarchy
This experiment tests whether **few-shot worked agentic trajectories**—concrete, end-to-end demonstrations of the self-correction choreography embedded in the system prompt—measurably improve the agentic solve rate on Clojure coding tasks.

- **Primary Target Substrate**: `gemma-4-E4B-it-qat-int4` (the 4B student model). Prompt scaffolding is primarily evaluated here, where agentic scaffolding is load-bearing.
- **Secondary / Teacher Substrate**: `gemma-4-31b-it-qat-int4`. Monitored for non-regression and clean teacher trajectory harvesting for downstream distillation.

A worked agentic trajectory demonstrates:
1. A concise planning turn (1–2 sentences).
2. An initial implementation tested in ```clojure ... ``` against public examples.
3. An execution response in ```clojure_result ... ``` containing a test failure.
4. A concise diagnostic thought (1 sentence identifying the bug).
5. A corrected implementation in ```clojure ... ``` that passes.
6. A final plain-text confirmation.

### Statistical Scale via the MultiPL-E Dev Corpus
Previously, prompt tuning on the 10-task `clojure_bench` suite was intractable due to high binomial variance ($\pm 10\%$ per flip at $N=10$) and the critical imperative to avoid leaking prompt tuning onto the sealed evaluation fixture.

With the delivery of the **MultiPL-E Clojure suite (Gate 3 Evals)**, we now possess an open, machine-verified development corpus of **447 tasks (`tasks_dev.edn`)**. At $N=447$, statistical noise drops to $\approx \pm 2.3\%$ (95% CI), enabling paired hypothesis testing via McNemar's test.

---

## 2. Disjointness Invariant & Mechanical Echoing Detection

A fundamental failure mode in few-shot prompting is unintentional task contamination or domain overlap.

**Invariant 1 (Strict Disjointness)**:
The worked example tasks in the prompt must be strictly disjoint from:
- All 447 tasks in `resources/catalog/gate3_evals/multipl_e/tasks_dev.edn`
- All 111 tasks in `resources/catalog/gate3_evals/multipl_e/tasks_sealed.edn`
- All 10 tasks in `resources/catalog/gate3_evals/clojure_bench/tasks_sealed.edn`
- All 50 legacy tasks in `resources/catalog/gate3_evals/multipl_e/dev_50_public.edn`

### Pre-Registered Synthetic Worked Examples:
1. **Example 1: Arithmetic & Predicate Bug (`sum-even-squares`)**
   - *Problem*: Given a vector of numbers, return the sum of the squares of only the even numbers.
   - *Error demonstrated*: Model initially checks `odd?` instead of `even?` (logic flaw).
   - *Self-correction*: Model reads test failure `(sum-even-squares [1 2 3 4]) -> Expected 20, got 10`, diagnoses the predicate inversion, patches `odd?` to `even?`, and passes.
2. **Example 2: Boundary & Arity Handling (`word-lengths`)**
   - *Problem*: Given a sentence string, return a vector of lengths of words separated by whitespace. Return `[]` for empty or nil strings.
   - *Error demonstrated*: Model calls `(str/split s #" ")` without handling multiple consecutive spaces or nil input, triggering `NullPointerException`.
   - *Self-correction*: Model inspects exception in `clojure_result`, adds nil guard `(if (str/blank? s) [] ...)`, uses `(re-seq #"\S+" s)`, and passes.

### Mechanical Canary Echoing Detector
To make example echoing falsification strictly mechanical, we pre-register synthetic canary identifiers unique to the worked examples:
```clojure
(def ECHO-CANARY-SYMBOLS
  #{"sum-even-squares" "word-lengths"})

(def ECHO-CANARY-LITERALS
  #{"Expected 20, got 10"})
```
**Detector Rule**: Across all benchmark task submissions (where `:fn-name` is distinct from these canaries):
1. Extract all symbol names from the parsed AST of candidate code using `clojure.walk`. If any symbol matches `ECHO-CANARY-SYMBOLS`, flag `CANARY-SYMBOL-LEAK`.
2. Grep raw submission text for any substring in `ECHO-CANARY-LITERALS`. If found, flag `CANARY-LITERAL-LEAK`.
Any leak triggered on any task constitutes a fatal falsification event (`KILLED`).

---

## 3. Pre-Registered Hypotheses & Gate Criteria

### Gate 1: Resource & Context Budget ($C_{n+1} \ll C_n$)
- **Prompt Size Constraint**: The combined few-shot prompt prefix (instructions + worked examples) must not exceed **600 tokens** total when tokenized with the Gemma 4 BPE tokenizer.
- **Context Headroom**: At `:max-seq-len 2048`, a $\le 600$ token prefix consumes $\le 29.3\%$ of the budget, leaving $\ge 1,448$ tokens for multi-turn model rollouts and tool interaction history.
- **Falsification Condition ($H_{1,\text{kill}}$)**:
  - If the prompt prefix exceeds 600 tokens (strict cap).
  - If the prompt causes host/device VRAM allocation failure under 2048-token context.

### Gate 2: Time Efficiency & Prefix Cache Amortization ($T_{n+1} \ll T_n$)
- **Prefix KV-Cache Amortization**: Under `harness_v2` prefix KV-cache reuse, the few-shot prompt prefix is computed on Turn 1 and reused across all subsequent agentic turns ($P_{\text{prefix}}$ tokens skipped).
- **Wall-Clock Latency Target**: Turn 2+ prefill latency must remain within $\le 10\%$ of zero-shot prefill latency.
- **Falsification Condition ($H_{2,\text{kill}}$)**:
  - If average per-task wall-clock time increases by $> 25\%$ relative to zero-shot `:fenced` baseline due to model verbosity inflation or self-correction loops stalling.

### Gate 3: Capability Ceiling & Floor Retention ($A_{n+1} > A_n$)
- **Primary Hypothesis $H_3$ (MultiPL-E Dev Gain on Student Substrate)**:
  - Evaluated on `gemma-4-E4B-it-qat-int4`: the selected candidate prompt $P^*$ achieves a net paired solve rate gain:
    $$\Delta_{\text{E4B}} = \text{PassRate}(P^*) - \text{PassRate}(P_0) \ge +5.0\%$$
    on the 447-task MultiPL-E dev corpus ($N=447$, $+23$ net tasks), with statistical significance by McNemar's test ($p < 0.05$).
- **Floor Retention Invariant $H_4$ (Zero Adverse Flips on Sealed-10)**:
  - On the frozen 10-task `clojure_bench` sealed suite, the prompt must produce **zero regressions** (no adverse flips on previously passing tasks) on both E4B-QAT and 31B-QAT.
- **Teacher Non-Regression**:
  - `gemma-4-31b-it-qat-int4` dev solve rate must not regress ($\Delta_{\text{31B}} \ge 0.0\%$).
- **Paired Statistical Test**:
  - Discordant pairs $(b, c)$ across the identical 447 dev tasks ($b = P^* \text{ pass / } P_0 \text{ fail}$, $c = P^* \text{ fail / } P_0 \text{ pass}$) are evaluated via McNemar's test with continuity correction:
    $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{one-tailed } p < 0.05 \quad (\chi^2 \ge 2.71)$$
    (Exact binomial test on $B(b + c, 0.5)$ if $b + c < 25$).

### Gate 4: Continuous Recursion & Distillation Enablement
- **Trajectory Capture**: All successful multi-turn agentic self-corrections from 31B and E4B are recorded with full telemetry (`:prompt-sha`, `:transcript`, `:tokens-in`, `:tokens-out`, `:turns`, `:wall-ms`).
- **Unlocks**:
  - `Student perplexity on teacher trajectories`: Clean 31B teacher trajectories for cross-entropy measurement.
  - `LoRA pure-stack trainer`: High-quality SFT distillation dataset of worked Clojure agentic self-corrections.

---

## 4. Candidate Prompt Designs

To prevent DOA proposals, all evaluated candidates are strictly budgeted $\le 600$ tokens:

| Variant | Name | Description | Target Token Budget |
|---|---|---|---|
| **$P_0$ (Control)** | `fenced-zero-shot` | Current default `AGENT-SYSTEM-PROMPT-FENCED` (prescriptive rules only). | ~160 tokens |
| **$P_1$ (Single)** | `fenced-worked-single` | Rules + 1 worked trajectory (`sum-even-squares`, logic error correction). | ~380 tokens |
| **$P_2$ (Dual)** | `fenced-worked-dual` | Rules + 2 worked trajectories (`sum-even-squares` + `word-lengths`, nil/exception handling). | ~560 tokens |

*(Negative constraint ablation $P_3$ is omitted from candidate evaluation to keep prompt size strictly within the 600-token cap).*

---

## 5. Execution & Multiplicity Protocol

To prevent multi-hypothesis shopping across variants and models, execution follows a strict gated funnel:

### Phase 1: TDD & Static Verification
1. Implement invariant tests in `test/experiments/gate3_evals/prompt_tuning_v1_test.clj`:
   - Exact token count $\le 600$ tokens with Gemma 4 tokenizer for all variants.
   - Mechanical canary echoing detector verification (clean on disjoint catalog tasks; flags canary injections).
   - Zero symbol or task overlap against all catalog tasks (`tasks_dev.edn`, `tasks_sealed.edn`, `clojure_bench/tasks_sealed.edn`).
   - Deterministic `:prompt-sha` rendering.

### Phase 2: Stratified Pilot & Single-Candidate Downselection ($N=50$)
1. Evaluate $P_1$ and $P_2$ against control $P_0$ on the 50-task stratified dev pilot using `gemma-4-E4B-it-qat-int4`:
   ```bash
   ./tools/gemma4.sh bench --tasks multipl-e-dev --limit 50 --prompt-template <variant> --model .models/gemma-4-E4B-it-qat-int4
   ```
2. **Downselection Rule**: Select **at most one** winning candidate $P^*$ (highest solve rate on the 50-task pilot; ties broken by smaller token count).
   - *Early Stop*: If neither $P_1$ nor $P_2$ beats $P_0$ by $\ge 2$ tasks on the pilot, the experiment terminates directly as `REJECT` without spending compute on the full 447-task dev corpus or evaluating sealed-10 (no candidate selected).
3. Multiplicity is resolved: exactly **one** candidate $P^*$ advances to Phase 3 ($m=1$).

### Phase 3: Full MultiPL-E Dev Benchmark ($N=447$)
1. Execute full dev evaluation of $P^*$ vs $P_0$ on `gemma-4-E4B-it-qat-int4` (and `gemma-4-31b-it-qat-int4`).
2. Run mechanical canary echoing detector on all submissions.
3. Compute McNemar's test on paired outcomes.

### Phase 4: Sealed Verification & Freeze (Gate Check vs Reporting-Only)
1. **`clojure_bench` sealed-10 (Gate Check)**:
   - Evaluate $P^*$ on `tasks_sealed.edn` ($N=10$) for both E4B and 31B.
   - Must achieve **zero adverse flips** vs baseline.
2. **MultiPL-E sealed-111 (Reporting-Only, Touch-Once)**:
   - Evaluated **strictly once** at the conclusion of the experiment to establish official catalog pod numbers.
   - **Zero iteration or re-tuning permitted**: if sealed-111 numbers are disappointing, the prompt is NOT modified; results are reported verbatim.

---

## 6. Complete Decision Rules (Zero Dead Zones)

To eliminate all classification ambiguity and dead zones, evaluation follows a strict **precedence hierarchy**:

1. **`KILLED` Triggers Dominate**: If any fatal condition occurs, the experiment terminates immediately as `KILLED`:
   - $\Delta_{\text{E4B}} < 0.0\%$ (net negative flips on the dev corpus, $b < c$)
   - OR any adverse flip on previously passing tasks in `clojure_bench` sealed-10
   - OR any canary leak detected by the mechanical echoing detector
   - OR candidate prompt length $> 600$ tokens
   - OR average per-task wall-clock time inflation $> 25\%$ vs zero-shot `:fenced` baseline.
2. **`ADOPT` Requires All Conjoined Criteria**: If no `KILLED` trigger occurs, the candidate is `ADOPT` if and only if **all** of the following hold:
   - $\Delta_{\text{E4B}} \ge +5.0\%$ on the 447-task MultiPL-E dev corpus
   - AND statistical significance holds via McNemar's test ($p < 0.05$)
   - AND $\Delta_{\text{31B}} \ge 0.0\%$ on the dev corpus (teacher non-regression)
   - AND zero adverse flips on `clojure_bench` sealed-10 (both E4B and 31B)
   - AND zero canary leaks detected
   - AND prompt length $\le 600$ tokens.
3. **`REJECT` is the Exhaustive Fallback**: If no `KILLED` trigger occurs and the candidate fails to meet all conjoined `ADOPT` criteria (e.g. $+0.0\% \le \Delta_{\text{E4B}} < +5.0\%$, or $\Delta_{\text{E4B}} \ge +5.0\%$ but failing McNemar's $p < 0.05$, or $\Delta_{\text{31B}} < 0.0\%$, or Phase 2 early-stop triggered), the outcome is deterministically `REJECT`.
   - *Phase 2 Early-Stop Note*: If neither $P_1$ nor $P_2$ beats $P_0$ by $\ge 2$ tasks on the pilot, the experiment terminates directly as `REJECT` without evaluating sealed-10 (vacuous with no candidate selected).

| Outcome | Summary Criteria | Action |
|---|---|---|
| **ADOPT** | All conjoined ADOPT criteria satisfied ($\Delta_{\text{E4B}} \ge +5.0\%$, McNemar $p < 0.05$, $\Delta_{\text{31B}} \ge 0.0\%$, zero sealed-10 regressions, zero canary leaks, prompt $\le 600$ tok) | Promote $P^*$ as default agentic prompt in `einsum.agent.core`; export 31B teacher self-correction trajectories for distillation |
| **REJECT** | No KILLED trigger, but fails to satisfy all conjoined ADOPT criteria (e.g. $+0.0\% \le \Delta_{\text{E4B}} < +5.0\%$, or non-significant gain $p \ge 0.05$, or $\Delta_{\text{31B}} < 0.0\%$, or Phase 2 pilot early-stop) | Reject few-shot worked examples; document null result; proceed directly to LoRA SFT distillation with zero-shot prompt |
| **KILLED** | Any fatal condition triggered ($\Delta_{\text{E4B}} < 0.0\%$, or any adverse flip on sealed-10, or any canary leak, or prompt $> 600$ tok, or wall-clock inflation $> 25\%$) | Immediately terminate prompt tuning track; record fatal failure mode in decision record |

*Note on Complete Partition*: The precedence order guarantees every possible outcome maps deterministically to exactly one verdict. Edge cases such as $\Delta_{\text{E4B}} \ge +5.0\%$ with $p \ge 0.05$ or $\Delta_{\text{31B}} < 0.0\%$ without a KILLED trigger route cleanly to `REJECT`. Zero unclassified states exist.

---

---

## 7. Empirical Results (Phase 2 Stratified Pilot)

Evaluation of all three candidate prompt variants was executed on the reference AMD Radeon RX 7900 XTX accelerator (24GB VRAM) running OpenXLA PJRT in-VRAM agentic sessions on `gemma-4-e4b-it-qat-int4` across the 50-task stratified dev pilot ($N=50$, 15 HumanEval + 35 MBPP, quarantined IDs excluded). All 150 candidate submissions were mechanically audited by `detect-canary-echo`.

### Empirical Performance Summary

| Variant | System Prompt Template | Tokens (Prefix) | Passed ($N=50$) | Pass Rate | $\Delta$ vs $P_0$ | Wall Time (s) | Mean s/Task | Canary Leaks |
|---|---|---|---|---|---|---|---|---|
| **$P_0$ (Control)** | `fenced-zero-shot` | 124 tok | **23 / 50** | **46.0%** | Baseline | 2,634.0s | 52.7s | 0 / 50 |
| **$P_1$ (Single)** | `fenced-worked-single` | 348 tok | **23 / 50** | **46.0%** | **+0 tasks** | 3,621.7s | 72.4s (+37.4%) | 0 / 50 |
| **$P_2$ (Dual)** | `fenced-worked-dual` | 568 tok | **19 / 50** | **38.0%** | **−4 tasks** | 3,658.8s | 73.2s (+38.9%) | 0 / 50 |

### Key Empirical Findings

1. **Self-Correction Demonstration Does Not Generalize**:
   Embedding worked self-correction demonstrations into the system prompt did not improve the student model's overall solve rate on Clojure coding tasks. While $P_1$ and $P_2$ succeeded on several tasks that $P_0$ failed (e.g., $P_1$ solved `humaneval-clj-008`, `mbpp-clj-014`, `mbpp-clj-017`; $P_2$ solved `humaneval-clj-005`, `006`, `017`, `mbpp-clj-026`), these localized wins were entirely offset by regressions on tasks $P_0$ solved cleanly.
2. **Context Latency Inflation Without Accuracy Gain**:
   The few-shot prompt prefixes inflated per-task inference latency by $+37.4\%$ ($P_1$) and $+38.9\%$ ($P_2$) due to increased prompt context and larger turn histories.
3. **Zero Canary Contamination**:
   Across all 150 candidate code submissions from $P_0$, $P_1$, and $P_2$, `detect-canary-echo` detected 0 leaks of canary symbols (`sum-even-squares`, `word-lengths`) or literal strings (`Expected 20, got 10`).

### Downselection & Termination Rationale

Under the pre-registered Phase 2 execution protocol:
- Downselection required at least one candidate to beat control $P_0$ by $\ge 2$ tasks ($\max(\Delta_1, \Delta_2) \ge 2$) to select $P^*$ and proceed to Phase 3 full dev benchmarking.
- Here, $\Delta_1 = 0$ and $\Delta_2 = -4$. Neither candidate beat $P_0$ by $\ge 2$ tasks.
- **Phase 2 Early-Stop Condition Triggered**: Under Section 5 (Phase 2, Step 2) and Section 6 (Rule 3), the experiment terminates directly as **`REJECT`** without spending compute on the full 447-task dev corpus or evaluating sealed-10 (which is vacuous with no candidate selected).

---

## 8. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-29 | `proposed` | Stage 1 RFC drafted; unlocked by cataloged MultiPL-E 447-task dev corpus; targets agentic self-correction failure modes observed in `harness_v2`. |
| 2026-09-29 | `amended` | Hardened spec per review: partitioned decision space with zero dead zones; harmonized REJECT vs KILLED; pre-registered McNemar's paired test; established single-candidate downselection ($m=1$); pre-registered mechanical canary echoing detector; dropped DOA $P_3$; designated MultiPL-E sealed-111 as reporting-only / touch-once; set Generation 0. |
| 2026-09-29 | `amended` | Refined precedence hierarchy (KILLED dominates -> ADOPT all conjoined -> REJECT exhaustive fallback); eliminated Case A/B fallthroughs; dropped non-mechanical parenthetical on Δ < 0; noted Phase 2 early-stop skips sealed-10. |
| 2026-09-29 | `rejected` | Phase 2 stratified pilot (N=50) completed on AMD RX 7900 XTX: P0=23/50 (46.0%), P1=23/50 (46.0%, Δ+0), P2=19/50 (38.0%, Δ-4). Zero canary leaks detected. Phase 2 early-stop triggered (neither candidate beat P0 by >= 2 tasks). Per pre-registered protocol, experiment terminates cleanly as REJECT without evaluating full dev or sealed-10. Prompt tuning track concluded; proceeding directly to LoRA SFT distillation with zero-shot prompt. |

---

## 9. Catalog Record

**Verdict: rejected** (2026-09-30).

Pilot disposition at cataloging (independently verified by review):

- **Phase 2 stratified pilot (N=50)** — 15 HumanEval / 35 MBPP, run 2026-09-29 on AMD RX 7900 XTX (OpenXLA PJRT/ROCm), `.models/gemma-4-e4b-it-qat-int4`, record timestamp `2026-09-29T22:33:34Z`.
- **P0** (zero-shot fenced, 124 tok): 23/50 = 46.0%, 52.7 s/task — baseline.
- **P1** (single worked example, 348 tok): 23/50 = 46.0%, Δ+0, 72.4 s/task (+37.4%).
- **P2** (dual worked example, 568 tok): 19/50 = 38.0%, Δ−4, 73.2 s/task (+38.9%).
- **Early-stop**: neither candidate beat P0 by ≥2 tasks → pre-registered Phase-2 early-stop fired; verdict REJECT per §6 Rule 3 (not KILLED — the wall-clock KILLED trigger scopes to an advancing candidate; none was selected).
- **Canaries**: 0 leaks across all 150 submissions (`sum-even-squares`, `word-lengths`, `"Expected 20, got 10"`).
- **Token cap**: all measured prompt prefixes under the frozen 600-token cap (P0 124 / P1 348 / P2 568).

**Key finding**: worked self-correction demonstrations changed behavior without converting to net hidden-test passes. The model produced genuine error→revise→retest traces (e.g. an `every?` arity fix across turns) and localized flips (P1: `humaneval-clj-008`, `mbpp-clj-014`, `mbpp-clj-017`), fully offset by regressions on tasks P0 solved cleanly. +37–39% wall-clock inflation with zero accuracy gain.

**Downstream**: prompt-tuning track concluded per protocol. The zero-shot fenced prompt (P0) carries forward into LoRA SFT trajectory distillation. Phase 3 (447-task dev benchmarking) and sealed-10 were not run.

Evidence: `pilot_results.edn` (variant-level aggregates with per-variant prompt SHAs, 50 task IDs, 15/35 stratification), `summary.csv`.
