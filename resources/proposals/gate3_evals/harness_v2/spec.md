# RFC: Eval Harness v2 — versioned, fast, library-resident

**Capability**: `harness_v2` \
**Type**: `capability` \
**Gate**: `gate3_evals` \
**Generation**: `1` \
**Literature**: `[]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "1116-cell MultiPL-E suite completes overnight on E4B-INT4"}` \
**Extends**: `clojure_bench` \
**Refutes**: `nil` \
**Supersedes**: `nil` \
**Reopens**: `nil` \
**Unlocks**: `["multipl_e_port", "prompt_tuning_v1", "distill_sft"]`

> **Track note.** This is a *capability* proposal, not an experiment
> proposal. The decision to build it is a human judgment call about
> program needs, stated plainly in Section 1. The rigor lives in
> pre-registered acceptance criteria (Section 2), not in falsification
> conditions.

---

## 1. Abstract & Motivation

The `clojure_bench` instrument works — 240 ledger rows, a sealed
fixture, prompt/checkpoint provenance — but it is slow and unversioned.
Measured 2026-09-27: **~70% of all benchmark wall time is spent
decoding runaway generations to the token cap**, and the harness
re-prefills the full prompt from token 0 on every agentic turn
(~40ms/token sequential prefill × 500–2000 tokens of prefix = 20–80s
of pure waste per turn). The harness is also split across two
namespaces (`experiments.gate3-evals.clojure-bench.run` and
`tools.gemma4-agent`) with no record of which code produced which row.

The judgment call: the MultiPL-E port (161 humaneval-clj + 397 mbpp-clj
= 558 tasks × 2 modes = **1116 cells**) is intractable at current
costs — ~13 hours on E4B-INT4 at ~42s/cell, far worse on 31B — and
every prompt-tuning iteration and distillation SFT eval loop pays the
same tax. We need a fast, versioned, library-resident eval harness
*before* the port, not after.

This proposal assumes the v1 freeze lands first: `clojure_bench`
promoted to `resources/catalog/gate3_evals/clojure_bench/`, every row
carrying `:harness-sha` (repo HEAD at row-write time) and
`:harness-dirty?`, sealed record runs requiring a clean tree. v2 is
then a *versioned* change to a frozen instrument, and AC2 below is
checkable as "same graded outcomes, new SHA."

The four optimizations (reviewed 2026-09-27):

1. **Prefix KV-cache reuse across agentic turns** — pure speedup, no
   measurement change. Port the p-match mechanism into
   `run-vram-loop-generation`.
2. **Semantic early stopping** — halt generation on a complete fenced
   code block or balanced top-level form outside the thought channel.
3. **Nudge short-circuit** — skip the blind second turn when Turn 1
   already yields extractable candidate code.
4. **Early exit on public pass** — terminate the loop when public tests
   are 2/2 (the ratchet's own tie-break logic makes this
   outcome-preserving: max public score cannot be beaten, ties go to
   the earliest turn).

Explicitly rejected in review and **out of scope**: cutting token
budgets to 512/768 (contradicted by the ledger — successful 31B runs
used 2,375–3,572 thinking tokens; a budget cut redefines the measured
capability), defaulting repetition-penalty to 1.15 (deprecated by the
2×2 verdict the same morning), and auto-skipping agentic mode on
single-shot pass as a default (destroys the matched-pair design; ships
only as an opt-in `--mode :auto` dev flag).

---

## 2. Acceptance Criteria & Non-Goals

- **AC1**: Prefix KV-cache reuse implemented for the in-VRAM
  generation path. On a fixed 3-turn reference workload, per-turn
  prefill cost reduced ≥50% (measured on host, not projected).
- **AC2**: Optimizations 2–4 implemented. The 20-cell reference suite
  (31B-QAT-INT4 @ rep-pen 1.0, frozen v1 config) run under v2
  reproduces v1 **graded outcomes exactly** — pass/fail per cell —
  with rows carrying the new `:harness-sha`. Any cell that changes
  outcome fails this criterion (self-correction-inside-generation
  risk is the known threat; the re-run is the test).
- **AC3**: Wall-time reduction ≥40% on the 20-cell reference suite vs
  v1, same host, same checkpoint, same decoding params.
- **AC4**: Agent loop ported out of `tools.gemma4-agent` into the
  library (proposed: `einsum.agent.*`; implementer refines per
  `AGENTS.md` ownership). The bench runner consumes the library
  namespace; no `tools.*` dependency remains in the eval path.
- **AC5**: A 50-task MultiPL-E dev subset runs end-to-end through the
  v2 harness: tasks load, grade in the SCI sandbox, ledger appends
  parse as valid EDN with `:harness-sha` present on every row.
- **AC6**: `format`, `lint` (0 errors, 0 warnings), and the fast test
  suite green on the refactor commit, per repo invariants. New
  stopping/short-circuit logic ships with generative tests written
  first (Strict TDD).

**Non-goals**:

- No task-set changes (the 10 sealed tasks are frozen; MultiPL-E
  tasks arrive via the separate port proposal).
- No grading-semantic changes (SCI sandbox, public/hidden split,
  best-public ratchet all unchanged).
- No prompt tuning and no decoding-param changes (separate
  experiment track).
- No full program-signature hashing (separate ticket; `:harness-sha`
  + existing `:prompt-sha`/`:checkpoint-sha`/decoding params is the
  v1 approximation).
- No multi-GPU; single 7900 XTX only.

**Verdict.** When the criteria hold: `done`. If the program decides
not to build it: `killed-by-judgment`, reason recorded here.

---

## 3. Interface & Integration

- **Code location**: agent loop moves `tools.gemma4-agent` →
  `src/einsum/agent/` (library); bench runner stays the Gate 3
  instrument, consuming it.
- **Consumed by**: `resources/catalog/gate3_evals/clojure_bench/run.clj`
  (post-freeze location), then `multipl_e_port` and `prompt_tuning_v1`.
- **Artifacts produced**: v2 ledger rows with `:harness-sha` ≠ v1 SHA;
  reference re-run report (v1 vs v2 per-cell outcomes + timings).
- **Harness changes**: `:harness-sha` / `:harness-dirty?` on rows
  (landed in the v1 freeze); v2 rows are comparable to v1 rows by AC2
  only — anything else that changes outcomes is a v3.

---

## 4. Cost Estimate

- Engineering: ~1 agent-week (mechanical: cache plumbing, stopping
  hooks, namespace move, tests).
- Compute: ~10 host-GPU hours (20-cell reference suite × v1
  re-baseline + v2 runs + 50-task MultiPL-E subset).
- Sequencing: depends on the v1 freeze (`:harness-sha` recording);
  independent of prompt tuning; blocks `multipl_e_port` at full scale.

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-27 | `proposed` | Harness review measured ~70% waste on runaway generation; MultiPL-E port (1116 cells) intractable without speedups; v1 freeze gives a versioned baseline to compare against |
