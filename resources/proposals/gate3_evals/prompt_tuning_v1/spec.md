# RFC: Prompt Tuning v1: Worked Agentic Trajectories

**Experiment**: `prompt_tuning_v1` \
**Gate**: `gate3_evals` \
**Generation**: `1` \
**Type**: `experiment` \
**Status**: `proposed` \
**Literature**: `["Wei et al. (2022) Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arXiv:2201.11903)", "Yao et al. (2022) ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629)", "Gemma Team (2025/2026) Gemma 4: Open Models for Advanced Reasoning and Agentic Workflows", "Cassano et al. (2023) MultiPL-E: A Scalable and Polyglot Benchmark for Evaluating Large Language Models"]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape ">= +5.0% solve rate on MultiPL-E dev (447 tasks), 0 regressions on sealed-10, prompt prefix <= 600 tokens"}` \
**Extends**: `"harness_v2"` \
**Refutes**: `nil` \
**Supersedes**: `nil` \
**Reopens**: `nil`

---

## 1. Abstract & Motivation

The Eval Harness v2 capability demonstrated that syntactic alignment and prompt framing dominate agentic performance in Clojure. Transitioning from JSON-escaped native tool-calling schemas to markdown fenced code blocks (`--tool-syntax :fenced`) lifted the student substrate (`gemma-4-E4B-it-qat-int4`) from 20% native agentic (30% native cross-mode union) to **50% agentic pass rate (5/10)**, while unlocking `my-comp` on the teacher substrate (`gemma-4-31b-it-qat-int4`, achieving **70% agentic pass rate**).

However, our current agentic system prompt (`AGENT-SYSTEM-PROMPT-FENCED`) remains purely prescriptive:
```text
You are an expert Clojure engineer with access to an interactive Clojure REPL sandbox.
Keep internal reasoning very concise (1-2 paragraphs max): plan your logic briefly, do not write full code drafts in thought, and wrap your Clojure code in ```clojure ... ``` tags to evaluate and test your code against public examples.
If any tests fail, inspect the failure errors, revise your implementation, and output revised code in ```clojure ... ``` to re-test.
Only provide your final response once your definition passes all public tests.
```

Error analysis of failed agent runs reveals that models fail not necessarily from mathematical incapacity, but from distinct agentic choreography failure modes:
1. **Thinking Paralysis & Draft Traps**: Small or heavily quantized models (e.g. E4B-QAT, 12B) often exhaust turn budgets inside `<|channel>thought` by drafting and redrafting broken code instead of emitting an executable tool call.
2. **Single-Turn Surrender**: Upon receiving failing assertions in ````clojure_result ... ````, models frequently surrender: outputting plain text explanations without revised code, repeating identical failing code verbatim, or losing track of the required signature.
3. **Execution Gap in Error Diagnosis**: When an exception occurs (e.g. `ClassCastException: java.lang.Long cannot be cast to clojure.lang.IFn`), models often do not understand how to diagnose the error message and localize the bug to paren placement.

### The Objective
This experiment tests whether **few-shot worked agentic trajectories**—concrete, end-to-end demonstrations of the self-correction choreography embedded in the system prompt—measurably improve the agentic solve rate on Clojure coding tasks.

A worked agentic trajectory demonstrates:
1. A concise planning turn (1–2 sentences).
2. An initial implementation tested in ```clojure ... ``` against public examples.
3. An execution response in ```clojure_result ... ``` containing a test failure.
4. A concise diagnostic thought (1 sentence identifying the bug).
5. A corrected implementation in ```clojure ... ``` that passes.
6. A final plain-text confirmation.

### Statistical Scale via the MultiPL-E Dev Corpus
Previously, prompt tuning on the 10-task `clojure_bench` suite was intractable due to high binomial variance ($\pm 10\%$ per flip at $N=10$) and the critical imperative to avoid leaking prompt tuning onto the sealed evaluation fixture.

With the delivery and verification of the **MultiPL-E Clojure suite (Gate 3 Evals)**, we now possess an open, machine-verified development corpus of **447 tasks (`tasks_dev.edn`)**. At $N=447$, statistical noise drops to $\approx \pm 2.3\%$ (95% CI), enabling rigorous detection of $\ge +5.0\%$ shifts.

---

## 2. The Disjointness Invariant (Zero Contamination)

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

---

## 3. Pre-Registered Hypotheses & Gate Criteria

### Gate 1: Resource & Context Budget ($C_{n+1} \ll C_n$)
- **Prompt Size Constraint**: The combined few-shot prompt prefix (instructions + 2 worked examples) must not exceed **600 tokens** total when tokenized with the Gemma 4 BPE tokenizer.
- **Context Headroom**: At `:max-seq-len 2048`, the prompt prefix consumes $\le 30\%$ of the budget, leaving $\ge 1,448$ tokens for multi-turn model rollouts and tool interaction history.
- **Falsification Condition ($H_{1,\text{kill}}$)**:
  - If the prompt prefix exceeds 750 tokens.
  - If the extended prompt causes VRAM allocation failure or ROCm RDNA3 LDS compilation failure (exceeding 64 KB workgroup LDS limits on RX 7900 XTX).

### Gate 2: Time Efficiency & Prefix Cache Amortization ($T_{n+1} \ll T_n$)
- **Prefix KV-Cache Amortization**: Under `harness_v2` prefix KV-cache reuse, the few-shot prompt prefix is computed on Turn 1 and reused across all subsequent agentic turns ($P_{\text{prefix}}$ tokens skipped).
- **Wall-Clock Latency Target**: Turn 2+ prefill latency must remain within $\le 10\%$ of zero-shot prefill latency.
- **Falsification Condition ($H_{2,\text{kill}}$)**:
  - If average per-task wall-clock time increases by $> 25\%$ relative to zero-shot `:fenced` baseline due to model verbosity inflation or self-correction loops stalling.

### Gate 3: Capability Ceiling & Floor Retention ($A_{n+1} > A_n$)
- **Primary Hypothesis $H_3$ (MultiPL-E Dev Gain)**:
  - Worked agentic trajectories increase the agentic pass rate on the 447-task MultiPL-E dev corpus by **$\ge +5.0\%$ absolute** for `gemma-4-E4B-it-qat-int4` (e.g. from baseline to baseline $+ 23$ solved tasks).
  - For the teacher model `gemma-4-31b-it-qat-int4`, worked agentic trajectories increase dev solve rate by **$\ge +3.0\%$ absolute**.
- **Floor Retention Invariant $H_4$ (Zero Adverse Flips on Sealed-10)**:
  - When evaluated frozen against the 10-task `clojure_bench` sealed suite, the prompt must produce **zero regressions** on previously passing tasks:
    - 31B-QAT: $\ge 7/10$ pass rate maintained.
    - E4B-QAT: $\ge 5/10$ pass rate maintained.
- **Falsification Condition ($H_{3,\text{kill}}$)**:
  - If dev pass rate improves by $< +2.0\%$ (insufficient capability signal).
  - If any adverse flip occurs on `clojure_bench` sealed-10 (capability degradation).
  - If the model exhibits "example echoing" (copying function names or logic from the worked examples into unrelated task solutions).

### Gate 4: Continuous Recursion & Distillation Enablement
- **Trajectory Capture**: All successful multi-turn agentic trajectories from 31B and E4B are recorded with full telemetry (`:prompt-sha`, `:transcript`, `:tokens-in`, `:tokens-out`, `:turns`, `:wall-ms`).
- **Unlocks**:
  - `Student perplexity on teacher trajectories`: Clean 31B teacher trajectories for cross-entropy measurement.
  - `LoRA pure-stack trainer`: High-quality SFT distillation dataset of worked Clojure agentic self-corrections.

---

## 4. Prompt Variants Under Investigation

| Variant | Name | Description | Target Token Budget |
|---|---|---|---|
| **$P_0$ (Control)** | `fenced-zero-shot` | Current default `AGENT-SYSTEM-PROMPT-FENCED` (rules only). | ~160 tokens |
| **$P_1$ (Single)** | `fenced-worked-single` | Rules + 1 worked trajectory (`sum-even-squares`, logic patch). | ~380 tokens |
| **$P_2$ (Dual)** | `fenced-worked-dual` | Rules + 2 worked trajectories (`sum-even-squares` + `word-lengths`, nil/exception handling). | ~580 tokens |
| **$P_3$ (Negative)** | `fenced-worked-anti-trap` | $P_2$ + explicit negative constraint against writing code in `<|channel>thought`. | ~620 tokens |

---

## 5. Execution & Verification Protocol

### Phase 1: Context Budget & Disjointness Verification (TDD)
1. Write unit and generative invariant tests in `test/experiments/gate3_evals/prompt_tuning_v1_test.clj`:
   - Invariant: Exact token count $\le 600$ tokens with Gemma 4 tokenizer.
   - Invariant: Zero symbol or task overlap between worked examples and all catalog tasks (`tasks_dev.edn`, `tasks_sealed.edn`, `clojure_bench/tasks_sealed.edn`).
   - Invariant: Prompt synthesis renders identically across runs and generates deterministic `:prompt-sha`.

### Phase 2: Stratified Fast Pilot on Dev Corpus ($N=50$)
1. Run evaluation on the 50-task stratified dev subset using `clojure_bench.run`:
   ```bash
   ./tools/gemma4.sh bench --tasks multipl-e-dev --limit 50 --prompt-template :prompt-tuning-v1 --model .models/gemma-4-E4B-it-qat-int4
   ```
2. Measure pass rate, turn distribution, and failure modes across $P_0, P_1, P_2, P_3$.
3. Falsify any variant exhibiting example echoing or context bloat.

### Phase 3: Full MultiPL-E Dev Benchmark ($N=447$)
1. Execute full dev evaluation of the winning candidate on `gemma-4-E4B-it-qat-int4` and `gemma-4-31b-it-qat-int4`.
2. Compute binomial confidence intervals and net solve rate delta ($\Delta \text{Pass Rate} \ge +5.0\%$).

### Phase 4: Sealed Verification & Freeze ($N=10$ and $N=111$)
1. Execute single frozen evaluation on `clojure_bench` sealed-10 and `multipl_e` sealed-111.
2. Confirm zero regressions on baseline passing tasks.
3. Record ledger rows with versioned `:prompt-sha`.

---

## 6. Decision Rules

| Outcome | Criteria | Action |
|---|---|---|
| **ADOPT** | $\Delta \text{Pass Rate}_{\text{dev}} \ge +5.0\%$ on E4B OR $\ge +3.0\%$ on 31B; 0 regressions on sealed-10; prompt $\le 600$ tok | Promote prompt as default agentic prompt in `einsum.agent.core`; export teacher trajectories for distillation |
| **REJECT** | $+0.0\% \le \Delta \text{Pass Rate}_{\text{dev}} < +2.0\%$ | Reject few-shot prompting; document null result; proceed directly to LoRA SFT distillation |
| **KILLED** | $\Delta \text{Pass Rate}_{\text{dev}} < 0.0\%$ (negative transfer) OR prompt $> 750$ tok OR example echoing observed | Immediately kill line; archive findings in decision record |

---

## 7. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-29 | `proposed` | Stage 1 RFC drafted; unlocked by cataloged MultiPL-E 447-task dev corpus; targets agentic self-correction failure modes observed in `harness_v2`. |
