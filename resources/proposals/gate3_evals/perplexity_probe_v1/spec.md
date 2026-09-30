# RFC: Perplexity Probe v1 — E4B-QAT Surprise on 31B-QAT Teacher Trajectories

**Experiment**: `perplexity_probe_v1`
**Gate**: `gate3_evals`
**Generation**: `0`
**Literature**: `[]`
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "inference-only; no training"}`
**Extends**: `"prompt_tuning_v1"`
**Refutes**: "H1 (SFT Headroom): E4B-QAT student exhibits elevated perplexity on 31B teacher trajectories (PPL_gap / PPL_ref >= 1.50)"
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

**Primary Metric & Aggregation Formula**:
Perplexity across sequences is computed via **micro-average (token-weighted corpus cross-entropy)**
over scored teacher tokens, preventing outlier sequences from skewing the aggregate:

```
PPL_gap = exp( - (Σ_{k ∈ gap} Σ_{i ∈ S_k} log p(x_{k,i} | x_{k,<i})) / (Σ_{k ∈ gap} |S_k|) )
PPL_ref = exp( - (Σ_{k ∈ ref} Σ_{i ∈ S_k} log p(x_{k,i} | x_{k,<i})) / (Σ_{k ∈ ref} |S_k|) )
```

For diagnostic transparency, macro-average perplexities (`(1/K) * Σ PPL_k`) and per-task
perplexities (`PPL_k`) are also computed and persisted in `results.edn` and `summary.csv`.

**Decision rule (pre-registered)**:
- **GO** iff `PPL_gap / PPL_ref >= 1.50`.
- **NO-GO** otherwise.

**Verdict mapping & downstream SFT filtering**:
- **GO** → Adopt the LoRA SFT ticket. For initial dataset composition, include gap tasks whose individual
  headroom ratio `r_k = PPL_k / PPL_ref >= 1.30`. If any gap task exhibits `r_k < 1.30` despite
  aggregate GO, flag it for sampling/prompting investigation rather than unconditional inclusion in the initial SFT mix.
- **NO-GO** → Do not proceed to LoRA on these trajectories; rethink (sampling, harness, or target selection)
  and record the rationale in the decision log.

**Kill criteria** (abort the probe, verdict `killed`):

- Teacher trajectories cannot be captured: 31B fails to solve a gap task under deterministic greedy
  decoding (`T=0.0`, seed 0) and fails after up to 2 seed-perturbed retries (`T=0.2`, seeds 1 and 2).
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
`x[0..n-1]` (prompt ++ teacher-generated tokens), the model defines:

```
log p(x[i] | x[<i]) = log_softmax(logits[i-1])[x[i]]  for i ∈ [1, n-1]
```

where `logits[i-1]` are the model's output logits at position `i-1`. Position 0 has no
preceding context and is un-scored.

Perplexity over a scored token span `S ⊆ {1, ..., n-1}` of teacher-generated tokens:

```
PPL(S) = exp( - (1/|S|) * Σ_{i ∈ S} log p(x[i] | x[<i]) )
```

**Masking & Segment Delineation**:
Evaluation masks are constructed directly from the transcript's structured turn sequence
(`:role` annotations `:model`, `:user`, `:tool`) rather than string pattern matching in tokenized text,
because under fenced agentic syntax tool results are formatted as `<|turn>user` turns (`agent/core.clj:579-584`):

- **Framing-tag mask policy (mask 0)**: Delimiter framing tokens—including `<bos>`, `<|turn>`,
  `<turn|>`, and turn prefixes (`<|turn>model\n`, `<|turn>user\n`)—receive mask 0. They are harness
  artifacts, not teacher-generated tokens.
- **Scored span `S` (mask 1)**: Only the actual text *content* generated within `:model` turns.
  System prompts, user prompt turns, and tool-result turns (`:role :tool` or fenced `:user`) are conditioned
  on in causal attention but strictly receive mask 0.
- **Thinking segment `S_think ⊆ S`**: Tokens of `:model` turn content within thinking blocks
  (`<|channel>thought...\n<channel|>`, matching `agent/core.clj:207`). Thinking channel boundary tokens
  receive mask 0. Delimiter token IDs are resolved from the tokenizer before scoring.
- **Code & action segment `S_code ⊆ S`**: Scored model content tokens outside the thinking channel
  (i.e., solution code, tool invocations, and agent response text).

**Implementation constraint & Tensor Logic Lowering**:
The existing prefill executable is compiled with `:last-token-only?` to emit only final-position logits.
The scoring op is a prefill *variant* that takes target token ids as an input and emits a `[seq_len]`
vector of target log-probabilities `log_softmax(logits)[pos, target[pos]]`. Returning only the
`[seq_len]` vector (`2048 × 4` bytes = 8 KB) to the host adheres to Rule 4 (zero off-heap
bandwidth waste), strictly avoiding materializing full `[seq_len × vocab]` logits (`2048 × 262144`
bf16 ≈ 1 GB per forward).

*Lowering Scope*: `lower-gather!` in `einsum.logic.lower` is currently specialized for rank-2 table
embedding lookup (`:collapsed_slice_dims [0]`, `:start_index_map [0]`). Step 2 will generalize
Tensor Logic lowering to support gathering target logits along axis `:v` (dimension 2 of rank-3
logits `[:log_probs :b :p :v]`) across both the PJRT MLIR lowering compiler and the pure-JVM
StableHLO interpreter.

## 4. Execution Harness & Silicon Verification Plan

- **Harness Namespace**: `experiments.gate3-evals.perplexity-probe-v1.run`
- **Output Artifacts**: `results.edn`, `summary.csv` (per-task PPL overall, thinking-segment
  PPL, code/tool-call-segment PPL, ratio to reference, and the GO/NO-GO verdict)

### Step 1 — Trajectory capture (host, 31B-QAT)

Scoping finding (2026-09-30): the bench runner captures `:transcript` on every agentic
run, but `format-results-row` persists it **only on failure**. No successful teacher
trajectories are on disk. Required harness change: a `--save-transcripts` flag (or
equivalent opt) that bypasses the failure-only filter. Keep the ledger append-only;
do not alter existing rows.

Protocol:
1. Run 31B-QAT agentic (fenced, rep-pen 1.0, current defaults) with deterministic greedy decoding
   (`T=0.0`, seed 0) on the four gap tasks (`first-n`, `my-range`, `deep-update-vals`, `lazy-interleave`)
   plus the two contrast tasks (`freqs`, `partition-by-parity`).
2. If any task fails under greedy decoding for any reason, retry up to 2 times
   with fixed seeds (`T=0.2`, seeds 1 and 2). If any task fails all 3 attempts, abort with verdict `killed`.
3. Keep only successful trajectories (hidden tests all-pass). Save full transcripts: every `:model`
   turn's response text in order, with turn boundaries and tool-result interleaving preserved.

Pinned checkpoint SHAs (verify before running; fail loudly on mismatch):

- Teacher 31B-QAT-INT4: `f20a02315954b540db819393224e07c3d60c1f07c9052b10a530db1556d2603c`
- Student E4B-QAT-INT4: source `google/gemma-4-E4B-it-qat-q4_0-unquantized`, project SHA
  `33ebfa9b85f077df19c0526c9f0e039c044d4e64eef47368730208b478dc05ec`

### Step 2 — Scoring op & Tensor Logic generalization

Strict TDD per repo rules (generative tests first on the pure-JVM StableHLO interpreter):

1. **Tensor Logic Gather Generalization**:
   - Extend `einsum.logic.lower` and `einsum.logic.interpret` to support gathering along axis 2
     (e.g., extracting `target[p]` from `[1, seq_len, vocab_size]` log-softmax distribution).
   - Generative property test: verify axis-gather matches slice-and-index references for arbitrary shapes.
2. **Runtime Scoring Op** (`einsum.models.gemma4.runtime/score-sequence-log-probs`):
   - Compiles prefill scoring executable with target gather, returning `[seq_len]` log-prob vector.
   - Property tests:
     - Scored log-probs are finite, $\le 0.0$, and length matches sequence input.
     - Consistency test: scoring a sequence matches the per-step logits from stepwise generation.
     - Known-sequence test: verifies `PPL = exp(-mean(log-probs))` against analytical ground truth.
   - Silicon verification: test forward pass on host RX 7900 XTX with `gemma-4-e4b-it-qat-int4`.

### Step 3 — Probe

For each trajectory:
1. Tokenize the teacher's transcript with the pinned BPE tokenizer.
2. Construct the boolean evaluation mask vector directly from structured transcript turns:
   - Mark position 0 as un-scored (mask 0).
   - For positions $1 \dots n-1$, assign mask 1 iff the target token is within `:model` turn content.
     All harness framing tokens (`<bos>`, `<|turn>`, `<turn|>`, and role headers) receive mask 0.
   - Partition scored model tokens into `S_think` (thinking channel content) and `S_code` (outside thinking).
3. Teacher-force through E4B-QAT-INT4 using `score-sequence-log-probs`.
4. Report per task: overall PPL, thinking-segment PPL, code/syntax-segment PPL, scored token counts,
   and individual ratio `r_k = PPL_k / PPL_ref`.
5. Compute corpus micro-average `PPL_gap` and `PPL_ref`, apply the §2 decision rule,
   and record GO or NO-GO with the aggregate ratio.

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-30 | `proposed` | Stage 1 RFC drafted. Motivated by the `prompt_tuning_v1` REJECT dissociation (behavior change without conversion → capacity gap, not elicitation gap). Scoping found two blockers: no successful teacher trajectories on disk (ledger keeps transcripts on failure only) and no scoring path in the generation-only runtime. Pre-registered GO/NO-GO rule: `PPL_gap / PPL_ref >= 1.5`. |
| 2026-09-30 | `hardened` | Hardened Stage 1 spec following review: formalized micro-average (token-weighted cross-entropy) primary decision rule, disaggregated per-task SFT filtering criterion (`r_k >= 1.30`), pre-registered greedy capture with bounded retry for any failure, pinned role-based `:model` content masking with framing-tag mask 0 policy, and scheduled Tensor Logic gather generalization in Step 2. |
| 2026-09-30 | `decided` | Stage 2 silicon evaluation completed on AMD Radeon RX 7900 XTX. Micro-average gap PPL = 1.8264 (N=6451), ref PPL = 2.0225 (N=1839), primary ratio PPL_gap / PPL_ref = 0.9030 < 1.50 pre-registered threshold. Macro ratio = 0.8738. All per-task ratios r_k in [0.65, 1.01] < 1.30 SFT candidate cutoff. Verdict: NO-GO. E4B student assigns high likelihood to teacher code (code PPL 1.28-1.56); failure is search/sampling dynamics rather than distribution ignorance. SFT compute canceled; proceed directly to Kat-Q ternary / architecture scaling. |

