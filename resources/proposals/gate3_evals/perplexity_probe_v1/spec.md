# RFC: Perplexity Probe v1 — E4B-QAT Surprise on 31B-QAT Teacher Trajectories

**Experiment**: `perplexity_probe_v1`
**Gate**: `gate3_evals`
**Generation**: `0`
**Literature**: `[]`
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "inference-only; no training"}`
**Extends**: `"prompt_tuning_v1"`
**Refutes**: `nil`
**Supersedes**: `nil`
**Reopens**: `nil`

---

## 1. Abstract & Motivation

`prompt_tuning_v1` (cataloged 2026-09-30, verdict REJECT) showed that worked self-correction
demonstrations change the E4B student's behavior without converting to net hidden-test passes:
the model performs the self-correction choreography but corrects poorly. That dissociation —
elicitation works, conversion doesn't — points at a *capacity* gap, not an *elicitation* gap,
and motivates weight updates (LoRA SFT on teacher trajectories) over further prompt work.

Before the LoRA training ticket lands, run a cheap inference-only probe: measure the student's
perplexity on the teacher's successful agentic trajectories. If the student finds the teacher's
reasoning on the four gap tasks substantially more surprising than reasoning it can already
reproduce, SFT has headroom (GO). If perplexity is comparable, the student's failures are
behavioral/sampling rather than distributional, and the SFT-first plan should be rethought (NO-GO).

This probe is *necessary-not-sufficient* evidence by design: high perplexity does not guarantee
SFT will close the gap, and low perplexity does not prove it cannot. It is a directional,
inference-only gate before training compute is spent.

Teacher/student substrates (pinned):

- Teacher: `gemma-4-31b-it-qat-int4`, frozen union 7/10 (8/10 cross-syntax).
- Student: `gemma-4-e4b-it-qat-int4`, frozen union 3/10.
- Gap tasks (teacher solves, student fails): `first-n`, `my-range`, `deep-update-vals`,
  `lazy-interleave`.
- Contrast tasks (student solves; teacher trajectories as reference): `freqs`,
  `partition-by-parity`.

## 2. Pre-Registered Hypotheses & Gate Criteria

**H1 (headroom)**: Mean per-token perplexity of E4B-QAT-INT4 on 31B-QAT's successful
trajectories is substantially higher on the four gap tasks than on the two contrast tasks
the student already solves.

**Decision rule (pre-registered)**: Let `PPL_gap` be the mean perplexity over the four
gap-task trajectories (teacher-generated tokens only, §4) and `PPL_ref` the mean over the
two contrast-task trajectories, both scored by the student.

- **GO** iff `PPL_gap / PPL_ref >= 1.5`.
- **NO-GO** otherwise.

**Verdict mapping**: GO → adopt the LoRA SFT ticket with the four gap tasks as first
targets. NO-GO → do not proceed to LoRA on these trajectories; rethink (sampling,
harness, or target selection) and record the rationale in the decision log.

**Kill criteria** (abort the probe, verdict `killed`):

- Teacher trajectories cannot be captured (31B fails to solve a gap task within the
  standard agentic budget after re-run).
- Tokenizer mismatch: E4B and 31B tokenizers are not the identical Gemma 4 BPE vocab
  (262144). Abort rather than score across vocabularies.
- Checkpoint SHA mismatch on either substrate vs the pinned values in §4.

### Gate 1: Resource Efficiency

- Inference-only: zero backpropagation. Scoring is one prefill-variant forward pass per
  trajectory plus the teacher capture runs.
- Compute budget: teacher capture ≈ 6 agentic tasks on 31B (~15–25 min host GPU);
  student scoring is minutes. Falsification: scoring op requires materializing full
  `[seq_len × vocab]` logits (see §3 — forbidden by construction).

### Gate 2: Time Efficiency

- Target: probe complete in < 2 hours wall-clock on the reference host, excluding the
  one-time scoring-op implementation.

### Gate 3: Intelligence & Capability Floor

- Not a capability eval; the probe produces no model artifact. Non-regression: the
  scoring op must not alter any existing generation path (pure addition).

### Gate 4: Continuous Recursion

- The scoring op (`score-sequence-log-probs`) is reusable infrastructure: it is the
  same per-position target log-prob operator SFT training will need for its loss.

## 3. Mathematical Specification

Per-position log-probability of target tokens under teacher forcing. Given token ids
`x[0..n-1]` (prompt ++ teacher-generated tokens) and the same sequence as targets
shifted by one, the model defines

```
log p(x[i] | x[<i]) = log_softmax(logits[i-1])[x[i]]
```

where `logits[i-1]` are the model's output logits at position `i-1`. Perplexity over a
scored span `S` of teacher-generated tokens:

```
PPL(S) = exp( - (1/|S|) * Σ_{i ∈ S} log p(x[i] | x[<i]) )
```

Implementation constraint: the prefill executable is compiled with `:last-token-only?`
today, so only the final position's logits materialize. The scoring op is a prefill
*variant* that, instead of emitting `[seq_len × vocab]` logits (2048 × 262144 bf16 ≈
1 GB per forward — forbidden), takes target token ids as an additional input and
emits a `[seq_len]` vector of `log_softmax(logits)[pos, target[pos]]` via a fused
gather. The per-position `[:logits :b :p :v]` tensor already exists in the model AST
(`gemma4.clj`); it is currently just not emitted. This is one additional output
variable plus a gather — declarative Tensor Logic, no new math.

## 4. Execution Harness & Silicon Verification Plan

- **Harness Namespace**: `experiments.gate3-evals.perplexity-probe-v1.run`
- **Output Artifacts**: `results.edn`, `summary.csv` (per-task PPL overall, thinking-segment
  PPL, code/tool-call-segment PPL, the GO/NO-GO verdict)

### Step 1 — Trajectory capture (host, 31B-QAT)

Scoping finding (2026-09-30): the bench runner captures `:transcript` on every agentic
run, but `format-results-row` persists it **only on failure**. No successful teacher
trajectories are on disk. Required harness change: a `--save-transcripts` flag (or
equivalent opt) that bypasses the failure-only filter. Keep the ledger append-only;
do not alter existing rows.

Then run 31B-QAT agentic (fenced, rep-pen 1.0, current defaults) on the four gap tasks
plus the two contrast tasks. Keep only successful trajectories (hidden tests all-pass).
Save full transcripts: every `:model` turn's response text in order, with turn boundaries
and tool-result interleaving preserved.

Pinned checkpoint SHAs (verify before running; fail loudly on mismatch):

- Teacher 31B-QAT-INT4: `f20a02315954b540db819393224e07c3d60c1f07c9052b10a530db1556d2603c`
- Student E4B-QAT-INT4: source `google/gemma-4-E4B-it-qat-q4_0-unquantized`, project SHA
  `33ebfa9b85f077df19c0526c9f0e039c044d4e64eef47368730208b478dc05ec`

### Step 2 — Scoring op

New runtime fn, e.g. `einsum.models.gemma4.runtime/score-sequence-log-probs`: inputs are
the token-id vector and target ids; runs the prefill-variant executable with the fused
target gather; returns the `[seq_len]` log-prob vector.

Strict TDD per repo rules, generative tests first. Test on the pure-JVM StableHLO
interpreter with a tiny model config (first-class backend for this):

1. Scored log-probs are finite, ≤ 0, and length matches input.
2. Consistency (property test, small n): scoring a sequence prefix-by-prefix equals the
   per-step logits from the existing stepwise generation path.
3. `perplexity = exp(-mean(log-probs))` sanity on a known sequence.

Host-verify on the real E4B-QAT checkpoint before trusting numbers.

### Step 3 — Probe

For each trajectory: tokenize the teacher's model-turn texts with the verified tokenizer;
teacher-force through E4B-QAT-INT4. **Score only teacher-generated tokens** (thinking,
code, tool-call syntax); condition on the full context (prompt + prior turns + tool
results) but do not score tool outputs — the student didn't generate them and scoring
them would muddy the signal.

Report per task: overall PPL, thinking-segment PPL, code/tool-call-segment PPL. Apply
the §2 decision rule and record GO or NO-GO with the ratio. The report states the
numbers, the ratio, and the verdict — no stronger claim (cf. §1 necessity caveat).

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-30 | `proposed` | Stage 1 RFC drafted. Motivated by the `prompt_tuning_v1` REJECT dissociation (behavior change without conversion → capacity gap, not elicitation gap). Scoping found two blockers: no successful teacher trajectories on disk (ledger keeps transcripts on failure only) and no scoring path in the generation-only runtime. Pre-registered GO/NO-GO rule: `PPL_gap / PPL_ref >= 1.5`. |
