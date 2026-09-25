# clojure_bench: Clojure coding baseline for the Gemma 4 family

Proposal — not yet built. Concrete `clojure-gen` task of the G3 eval instrument
(`../quant_baseline_eval/spec.md`); inherits its discipline (dev/sealed split,
cost ledger, pre-registered verdicts, comparison not leaderboard).

## Goal

Establish a measured baseline of how well Gemma 4 E2B / E4B / 31B (INT4,
thinking + tool calling enabled) write correct, idiomatic Clojure — before we
try to improve the loop. Two numbers per model: raw s-expression ability
(single-shot) and loop ability (agentic with the `eval_clojure` tool). Their
difference is the story: it measures what the first loop adds.

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
- All expected values in this proposal were machine-verified against reference
  solutions evaluated in SCI 0.9.44 on 2026-09-25 (10/10 tasks, 68/68 tests).

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

Task 10 is the sharpest discriminator and the most Clojure-specific; keep it
in v1. SCI 0.9.44 supports `defmacro` in the grading context (verified
2026-09-25), so the macro is graded by behavior, including a side-effect
short-circuit test — not by expanding and eyeballing.

## Modes

1. **single-shot**: one generation → extract code with the fixed balanced
   reader → eval → grade. Raw s-expression ability.
2. **agentic**: `gemma4_agent` loop with the `eval_clojure` tool, public tests
   visible, ≤5 turns, ≤3 consecutive tool errors then forced final answer.
   After each submission-shaped tool call, public tests are evaluated and the
   score fed back. Best-public submission is graded hidden exactly once.
   This is the first loop, measured.

Both modes share prompts, extraction, grading context, and seeds. Depends on
the agent-issue fixes (balanced-reader extraction, greedy arg parsing, token
telemetry, error budget, tightened grading sandbox) landing first.

## Models & decoding

E2B, E4B, 31B — INT4 checkpoints, thinking on, temperature 0.0, top-k 10,
repetition-penalty 1.15. Same harness, same prompts, same seeds. 31B INT4 is
its own reference (BF16 31B cannot load on the 7900 XTX); cross-model cells
are absolute comparisons, never retention claims.

## Metrics

- Primary: mean hidden pass rate per model (10 tasks).
- Per-task pass table — shows where the ladder breaks per model.
- pass@1 (first submission passes hidden), pass@k (any of k passes).
- Agentic efficiency: median submissions to first public pass.
- Cost ledger per (model, task, mode): tokens in/out per turn, wall ms, VRAM
  residency, checkpoint sha, sealed sha256, stop reason.

## Grading sandbox

Tightened SCI context, separate from the agent's: no `slurp`/`spit`/
`list-files`, no `System/`, deterministic. The agent keeps its tools; the
grader keeps its honesty. Reference solutions live only in the verification
script used to validate this proposal — they are never shipped with the
benchmark and never enter model context.

## Runner sketch (`tools/clj_bench.clj`, to build)

- Load `tasks_public.edn`; render prompt = task prompt + fn name + public
  tests as examples.
- single-shot: one `generate-new-text-string`, extract, grade, record row.
- agentic: reuse `run-agent-loop` with a per-task prompt and a submission
  hook (defn/defmacro-shaped tool call → eval public tests → feed back score).
- Append one EDN row per (model, task, mode) to `results.edn`:
  `{:model :task :mode :passed? :n-submissions :tokens-in :tokens-out
    :wall-ms :sealed-sha :checkpoint-sha :stop-reason}`.

## Open questions (Simon's call)

1. Keep the macro task in v1? Recommended: yes — best discriminator.
2. Run single-shot first, then agentic — or straight to agentic?
   Recommended: single-shot first; gives the agentic mode something to beat.
