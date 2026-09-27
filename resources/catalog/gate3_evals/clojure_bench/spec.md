# clojure_bench: Clojure coding baseline for the Gemma 4 family

**Capability**: `clojure_bench` \
**Type**: `capability` \
**Gate**: `gate3_evals` \
**Generation**: `1` \
**Literature**: `[]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "10-task suite runs end-to-end in-VRAM, both modes, append-only ledger"}` \
**Extends**: `"quant_baseline_eval"` \
**Refutes**: `nil` \
**Supersedes**: `nil` \
**Reopens**: `nil` \
**Unlocks**: `["substrate comparisons (QAT vs PTQ)", "prompt tuning", "distillation SFT eval", "rep-pen operating config"]`

> **Track note.** This is a *capability*, not an experiment. It was built
> on a human judgment call — the program needed a measurement instrument
> before it could run any eval-gated experiment — not on a falsifiable
> hypothesis. The rigor lives in the acceptance criteria below.

---

## 1. Abstract & Motivation

The program had no way to measure what its loop could actually do. Every
substrate question (QAT vs PTQ), every decoding question (rep-pen), every
future prompt-tuning or distillation claim would have been argued on
vibes. So we built the instrument first: a 10-task Clojure coding
benchmark with single-shot and agentic modes, a sealed fixture, and an
append-only ledger. Concrete `clojure-gen` task of the G3 eval
instrument (`../quant_baseline_eval/spec.md`); inherits its discipline
(dev/sealed split, cost ledger, pre-registered verdicts, comparison not
leaderboard).

Two numbers per model: raw s-expression ability (single-shot) and loop
ability (agentic with the `eval_clojure` tool). Their difference is the
story: it measures what the first loop adds.

## 2. Acceptance Criteria & Non-Goals

- **AC1** — 10 laddered tasks with machine-verified expected values:
  **done** 2026-09-25 (10/10 tasks, 68/68 tests verified against
  reference solutions in SCI 0.9.44).
- **AC2** — Sealed fixture never entering model context, sha256 recorded
  per row: **done** (`tasks_sealed.edn`
  `656f97b09127561b857f10eba46262750eb358279705d252cfea247d0300cda0`).
- **AC3** — Both modes run end-to-end in-VRAM on the 7900 XTX:
  **done** (200 ledger rows across 7 model configs as of 2026-09-27).
- **AC4** — Append-only ledger; no row ever mutated: **done**
  (`results.edn`, sequential EDN, every row carries `:model :task :mode
  :passed? :tokens-in :tokens-out :max-new-tokens :wall-ms :sealed-sha
  :checkpoint-sha :prompt-sha :repetition-penalty`).
- **AC5** — Best-submission ratchet (highest public-test pass wins ties
  by earliest turn): **done**.

**Non-goals:** no partial credit in v1; no multi-file or repo-level
tasks; no contamination rotation yet (the sealed fixture needs a
rotation story before it becomes the distillation gate).

**Verdict: done** (2026-09-27).

## 3. Interface & Integration

- **Code location**: `src/experiments/gate3_evals/clojure_bench/`
  (`core.clj`, `run.clj`); tasks in `resources/catalog/gate3_evals/clojure_bench/`.
- **Consumed by**: every substrate/decoding/prompt experiment via
  `run.clj`; future distillation eval via the same sealed fixture.
- **Artifacts produced**: `results.edn` (append-only ledger),
  `summary.csv`.
- **Harness changes**: ledger schema extensions are additive only
  (`:prompt-sha`, `:max-new-tokens`, `:repetition-penalty` all landed
  this way).

## Why Clojure first

Niche language, training-data thin, s-expr discipline unforgiving. Idiom here
is genuinely discriminative: persistent collections, the seq abstraction,
`loop`/`recur`, destructuring, and macros. A model that writes a correct
`my-or` macro understands something a Python-HumanEval passer may not.

## Task format

- `tasks_public.edn`: vector of `{:id :title :fn-name :prompt :public-tests}`.
  Shown to the model (prompt + public tests as examples).
- `tasks_sealed.edn`: vector of `{:id :hidden-tests}`. Never enters model
  context. sha256 of the file is recorded on every grading run.
- A test is `{:code "<clojure expr>" :expected "<edn>"}`. Grading evaluates
  `:code` in the tightened SCI context and compares with `=` against
  `(clojure.edn/read-string :expected)`.
- Pass = **all** hidden tests pass. No partial credit in v1 (pre-registered).

## The 10 tasks

Laddered on s-expr/idiom difficulty, not algorithmic cleverness. Each targets
one Clojure-specific failure mode observed in small models.

| # | id | discriminates |
|---|----|---------------|
| 1 | `first-n` | seq→vector, `take` on infinite/short colls |
| 2 | `my-range` | `loop`/`recur` binding forms, no `range` crutch |
| 3 | `deep-flatten` | recursion, `sequential?`, strings/maps as leaves |
| 4 | `freqs` | `reduce` + `(fnil inc 0)` idiom, no `frequencies` |
| 5 | `partition-by-parity` | `reduce` into map of vectors, order preservation |
| 6 | `my-comp` | variadic HOF, right-to-left, multi-arg rightmost fn |
| 7 | `balanced-delims?` | explicit stack, proper nesting, ignore-noise |
| 8 | `deep-update-vals` | recursion over maps+vectors, keys untouched |
| 9 | `lazy-interleave` | real `lazy-seq` (infinite-input tests fail if eager) |
| 10 | `my-or` | macro: syntax-quote, short-circuit, `~'` hygiene |

Task 10 is the sharpest discriminator and the most Clojure-specific. SCI
0.9.44 supports `defmacro` in the grading context (verified 2026-09-25),
so the macro is graded by behavior, including a side-effect
short-circuit test — not by expanding and eyeballing.

## Modes

1. **single-shot**: one generation → extract code with the fixed balanced
   reader → eval → grade. Raw s-expression ability.
2. **agentic**: `gemma4_agent` loop with the `eval_clojure` tool, public tests
   visible, ≤5 turns, ≤3 consecutive tool errors then forced final answer.
   After each submission-shaped tool call, public tests are evaluated and the
   score fed back. Best-public submission is graded hidden exactly once.
   This is the first loop, measured.

Both modes share prompts, extraction, grading context, and seeds.

## Models & decoding (operating config as of 2026-09-27)

INT4 checkpoints, thinking on, temperature 0.0, top-k 10,
**repetition-penalty 1.0** (1.15 deprecated: hurts agentic on both 31B
and 12B substrates). Same harness, same prompts, same seeds. 31B INT4 is
its own reference (BF16 31B cannot load on the 7900 XTX); cross-model cells
are absolute comparisons, never retention claims.

## Metrics

- Primary: mean hidden pass rate per model (10 tasks).
- Per-task pass table — shows where the ladder breaks per model.
- pass@1 (first submission passes hidden), pass@k (any of k passes).
- Agentic efficiency: median submissions to first public pass.
- Cost ledger per (model, task, mode): tokens in/out per turn, wall ms,
  VRAM residency, checkpoint sha, sealed sha256, prompt sha,
  repetition penalty, stop reason.
- Trace metrics: pre-tool thinking tokens, budget-saturation rate.

## Grading sandbox

Tightened SCI context, separate from the agent's: no `slurp`/`spit`/
`list-files`, no `System/`, deterministic. The agent keeps its tools; the
grader keeps its honesty. Reference solutions live only in the verification
script used to validate the tasks — they are never shipped with the
benchmark and never enter model context.

## Runner (`src/experiments/gate3_evals/clojure_bench/run.clj`)

- Load `tasks_public.edn`; render prompt = task prompt + fn name + public
  tests as examples.
- single-shot: one generation, extract, grade, record row.
- agentic: reuse `run-agent-loop` with a per-task prompt and a submission
  hook (defn/defmacro-shaped tool call → eval public tests → feed back score).
- Append one EDN row per (model, task, mode) to `results.edn`.

## Resolved questions

1. Keep the macro task in v1? **Yes** — best discriminator, confirmed
   across 200 rows (my-or remains in the stable hard core).
2. Single-shot first, then agentic? **Single-shot first** — gives the
   agentic mode something to beat; the mode delta is the loop's story.

## 4. Cost Estimate (actuals)

- Engineering: agent-built across 2026-09-25 → 2026-09-27 (spec,
  ratchet, ledger-schema extensions, agent-issue fixes).
- Compute: full 20-cell eval ≈ 44 min end-to-end on the 7900 XTX
  (31B-QAT-INT4 baseline, 2026-09-27).

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-25 | `proposed` | Program needed a measurement instrument before any eval-gated experiment; human judgment call |
| 2026-09-27 | `done` | All acceptance criteria hold; 200 ledger rows; unlocking substrate, decoding, and prompt experiments |
