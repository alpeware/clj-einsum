# RFC: MultiPL-E humaneval-clj & mbpp-clj Port

**Capability**: `multipl_e_port` \
**Type**: `capability` \
**Gate**: `gate3_evals` \
**Generation**: `1` \
**Literature**: `["Cassano et al. (2023) MultiPL-E: A Scalable and Extensible Polyglot Benchmark for Large Language Models (arXiv:2208.08227)", "nuprl/MultiPL-E Pull Request #136: Add Clojure programming language (2023)", "nibzard (2026) clojure-llm: Benchmark-first autoresearch for Clojure code models (https://github.com/nibzard/clojure-llm)"]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "558-cell single-shot evaluation completes in ~2.3h; full 1116-cell two-mode suite completes overnight in ~7.4h on E4B-QAT-INT4 with prefix KV reuse"}` \
**Extends**: `harness_v2` \
**Refutes**: `nil` \
**Supersedes**: `nil` \
**Reopens**: `nil` \
**Unlocks**: `["prompt_tuning_v1", "student_perplexity_teacher", "distill_sft"]`

> **Track note.** This is a *capability* proposal, not an experiment
> proposal. Capabilities are judgment-called infrastructure the program
> needs: harnesses, trainers, ports, runtime machinery. They are not
> falsifiable claims about the world — they are *completable* work. The
> decision to build one is a human judgment call about program needs,
> stated plainly in Section 1. Never a fake hypothesis. The rigor lives
> in pre-registered acceptance criteria (Section 2), not in
> falsification conditions.

---

## 1. Abstract & Motivation

The `clojure_bench` baseline instrument was frozen with 10 sealed tasks
(`tasks_public.edn` and `tasks_sealed.edn`). While sufficient for sanity-checking
the evaluation harness and initial substrate comparisons, an $N=10$ evaluation
suffers from severe sample variance: every single task pass/fail flip represents
a $\pm 10\%$ shift in the capability score. Furthermore, developing prompt
improvements (such as worked agentic trajectories) directly on the 10 sealed tasks
violates experimental hygiene by overfitting to the test distribution.

To build an honest, robust Gate 3 capability floor and unlock downstream
distillation, we need a standard, large-scale Clojure coding distribution:
the **MultiPL-E Clojure suite** (161 `humaneval-clj` + 397 `mbpp-clj` = **558 tasks**).

### 1.1 Review of Prior Art & Ecosystem Context

1. **MultiPL-E (Cassano et al., 2023)**:
   The canonical polyglot translation framework translating HumanEval and MBPP
   into multiple programming languages via AST transformations and unit test synthesis.
2. **MultiPL-E Pull Request #136 (`nuprl/MultiPL-E#136`)**:
   Submitted by Clojure core contributors to officially bring Clojure into the
   MultiPL-E benchmark suite. It established canonical translations of docstrings,
   function signatures, and test cases using `clojure.test`.
3. **`clojure-llm` (nibzard, 2026)**:
   An autoresearch program exploring whether smaller open models
   (Qwen3-8B/30B) with fast verification loops (REPL, `clj-kondo`, `clojure.test`)
   can beat frontier models (GPT-5.4, Opus 4.7) on Clojure generation.
   Crucial insights from `clojure-llm`:
   - Evaluated the full 558-task MultiPL-E suite using a standard **447 training/dev**
     and **111 held-out test** split (20% held-out test set).
   - Demonstrated that a verifier loop lets an 8B model (best-of-8: 67.6%) exceed
     single-pass GPT-5.4 (64.0%).
   - Identified execution overhead in fresh JVM/clojure subprocess evaluation
     per task. In our architecture, we substitute subprocess spawning with
     our in-process, hermetic Small Clojure Interpreter (`create-benchmark-sci-ctx`),
     yielding sub-millisecond per-task evaluation with strict isolation.

### 1.2 Evaluation & Disposition of Existing `resources/catalog/gate3_evals/multipl_e/`

During the development of `harness_v2` (commit `d1451fa`), a 50-task subset
(`dev_50_public.edn`, `dev_50_sealed.edn`, `solutions.edn`) was temporarily staged
in `resources/catalog/gate3_evals/multipl_e/` to test multi-task harness plumbing.

**Audit & Assessment**:
- The current `dev_50` files represent tasks 0–49 of `humaneval-clj`.
- Unit tests in `test/einsum/agent_test.clj` (`test-multipl-e-dev-50-fixture-and-ledger`)
  and dry-run options in `clojure_bench/run.clj` depend on these paths.
- However, having only a 50-task partial subset in the catalog is incomplete.
- **Disposition Plan**:
  - We will **retain and canonicalize** `dev_50` as a fast smoke test fixture (`smoke_50_*.edn` or preserved `dev_50_*.edn`).
  - We will **expand the catalog pod** to house the complete, verified 558-task suite partitioned into `tasks_dev.edn` (447 tasks) and `tasks_sealed.edn` (111 tasks), accompanied by complete ground-truth solutions in `solutions.edn` and a quarantined task record `quarantine.edn`.

### 1.3 Dataset Provenance & Pinned SHAs

To eliminate ambiguity and ensure reproducibility, data ingestion is strictly pinned:
- **Upstream Hugging Face Repository**:
  - URL: `https://huggingface.co/datasets/nuprl/MultiPL-E`
  - Commit SHA: `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`
- **Upstream GitHub Repository**:
  - URL: `https://github.com/nuprl/MultiPL-E`
  - Commit SHA: `3025a531af7450e7df8b96fe0440e9804480bbad`
- **Independent Cross-Check Reference**:
  - `nibzard/clojure-llm` commit SHA: `8ed80ac59ad26413ed8f9e7f167860b384478e41` (`benchmark/tasks-v0.edn`).
  - Cross-check against local copies when available to verify task inventory bit-for-bit.

### 1.4 Runtime & Cell Timing Arithmetic

Evaluation costs are partitioned cleanly across modes on AMD Radeon RX 7900 XTX (E4B-QAT-INT4):
- **Single-Shot (1 turn, no tools)**:
  - Average wall-clock latency: ~15.0s per cell (greedy decoding, 200–500 tokens).
  - 558 tasks × 1 mode = **558 cells** → ~8,370s (**~2.33 hours**, < 2.5h).
- **Agentic Mode (multi-turn, up to 5 turns, prefix KV cache reuse active)**:
  - Average wall-clock latency: ~32.5s per cell (empirically 21.3%–26.7% faster than v1 42s baseline due to prefix KV reuse, early exit on public pass, and nudge short-circuit).
  - 558 tasks × 1 mode = **558 cells** → ~18,135s (**~5.04 hours**).
- **Full Two-Mode Suite (1,116 cells)**:
  - 2.33h + 5.04h = **~7.37 hours** total wall-time, executing comfortably overnight unattended in a single session.

---

## 2. Acceptance Criteria & Non-Goals

Every capability must state measurable done-conditions *before* work starts.
When all criteria hold, the capability is done.

- **AC1: Pinned 558-Task Ingestion**:
  Ingest all 161 `humaneval-clj` tasks and 397 `mbpp-clj` tasks from `nuprl/MultiPL-E` pinned at commit `28441b6`. Each task is normalized to the canonical harness schema:
  `{:id ... :source ... :fn-name ... :prompt ... :public-tests [...] :hidden-tests [...]}`.
- **AC2: SCI Sandbox Verification & Quarantine Escape Hatch**:
  Every ground-truth reference solution in `solutions.edn` must execute in `create-benchmark-sci-ctx`.
  - **Verification Floor**: $\ge 95\%$ of tasks (at least 530 out of 558) must pass 100% of their test assertions in pure SCI.
  - **Quarantine Escape Hatch**: Any task requiring unsupported Java reflection, unseeded randomness, or un-emulatable host I/O is isolated in a committed artifact `quarantine.edn` with a recorded per-task diagnostic rationale. No task is silently waived or deleted.
- **AC3: Exact 111 Held-Out Task ID Alignment**:
  Adopt verbatim the 111 held-out task IDs established by `nibzard/clojure-llm` (defined in run manifest `benchmark/runs/2026-04-20-rlvr-qwen3-30b-heldout.edn` and evaluated in directory `benchmark/results/2026-04-20-rlvr-qwen3-30b-heldout/`), stratified 20% across sources:
  - **32 HumanEval tasks**: `humaneval-clj-001`, `003`, `012`, `018`, `019`, `020`, `031`, `039`, `043`, `053`, `062`, `067`, `068`, `073`, `077`, `086`, `090`, `095`, `097`, `105`, `121`, `123`, `124`, `132`, `134`, `137`, `141`, `143`, `144`, `153`, `157`, `158`.
  - **79 MBPP tasks**: `mbpp-clj-006`, `007`, `009`, `022`, `047`, `057`, `060`, `069`, `074`, `075`, `076`, `082`, `084`, `087`, `091`, `093`, `097`, `117`, `119`, `120`, `132`, `133`, `137`, `144`, `148`, `154`, `156`, `162`, `163`, `176`, `179`, `180`, `181`, `185`, `191`, `196`, `197`, `201`, `204`, `214`, `216`, `220`, `226`, `245`, `248`, `252`, `258`, `259`, `266`, `267`, `268`, `269`, `279`, `283`, `286`, `287`, `295`, `297`, `299`, `303`, `306`, `308`, `313`, `316`, `326`, `330`, `336`, `342`, `346`, `350`, `361`, `363`, `366`, `368`, `371`, `375`, `376`, `384`, `395`.
  - Partition files: `tasks_dev.edn` (447 tasks, open) and `tasks_sealed.edn` (111 tasks, held-out). Matching these exact IDs ensures direct comparability with published frontier baselines (GPT-5.4 64.0%, Gemini 3.1 Pro 72.8%).
- **AC4: Public / Hidden Test Split Rule**:
  Define a deterministic split rule across the parsed `(is (= ...))` assertion forms $[a_1, a_2, \dots, a_K]$:
  - If $K \ge 3$: the first 2 assertions become `:public-tests`; the remaining $K-2$ assertions become `:hidden-tests`.
  - If $K = 2$: assertion $a_1$ is public; assertion $a_2$ is hidden.
  - If $K = 1$: assertion $a_1$ is present in both public and hidden.
  This rule matches the existing `dev_50` convention identically and guarantees that models receive concrete examples for agentic self-correction while reserving the majority for held-out grading.
- **AC5: Named Default Prompt Template (`:multipl-e-v0`)**:
  Establish versioned prompt rendering for the port:
  - Template name: `:multipl-e-v0`.
  - Single-shot: System prompt `SINGLE-SHOT-SYSTEM-PROMPT`, user turn contains verbatim docstring and function signature.
  - Agentic: System prompt `AGENT-SYSTEM-PROMPT`, `--tool-syntax :fenced`, user turn contains docstring plus public examples formatted for interactive testing.
- **AC6: Existing Test Suite Non-Regression**:
  Preserve backwards compatibility for `test/einsum/agent_test.clj`
  (`test-multipl-e-dev-50-fixture-and-ledger`) so existing regression tests pass with
  zero errors.
- **AC7: Harness Integration & Dry-Run Ledgering**:
  `experiments.gate3-evals.clojure-bench.run/run-benchmark` must execute successfully
  in `--dry-run` mode against both `tasks_dev.edn` and `tasks_sealed.edn`, recording valid
  rows with `:harness-sha`, `:tool-syntax`, `:temperature`, and `:sealed-sha256`.

**Non-goals**:
- **No full model inference run during capability stage**: This proposal delivers the
  verified dataset, ingestion pipeline, SCI conformance, and test harness integration.
  Running model generations (e.g. evaluating 31B or E4B across all 558 tasks) belongs
  to the subsequent evaluation and distillation proposals that this capability unlocks.
- **No external OS subprocess evaluation**: We intentionally do not use the heavy
  subprocess-per-task architecture of `clojure-llm`; all grading remains in-process
  via hermetic SCI sandboxes.

---

## 3. Interface & Integration

- **Ingestion Pipeline**:
  `src/experiments/gate3_evals/multipl_e/ingest.clj`
  CLI tool to fetch, parse, and validate the dataset directly from HuggingFace / upstream JSONL.
- **Catalog Artifacts** (`resources/catalog/gate3_evals/multipl_e/`):
  - `tasks_dev.edn`: 447 open dev tasks.
  - `tasks_sealed.edn`: 111 held-out eval tasks.
  - `solutions.edn`: 558 reference solutions.
  - `quarantine.edn`: Quarantined tasks failing pure SCI sandbox criteria with documented rationales.
  - `dev_50_public.edn` / `dev_50_sealed.edn`: preserved for fast CI smoke testing.
- **Harness Flags**:
  `--tasks multipl-e-dev` and `--tasks multipl-e-sealed` added to `clojure_bench/run.clj`
  and `tools/gemma4-agent`.

---

## 4. Cost Estimate

- Engineering: ~1–2 agent-days (ingestion script, SCI test verification, test harness plumbing).
- Compute: Minimal during capability stage (< 0.1 GPU-hours; SCI ground-truth verification runs entirely host-side on CPU).
- Sequencing: Unlocks `prompt_tuning_v1` and `distill_sft`.

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-29 | `proposed` | `harness_v2` promoted to catalog. Unlocks MultiPL-E port to address $N=10$ sample variance, unblock prompt tuning, and establish distillation corpus. Spec pre-registers 447/111 partition, SCI sandboxing, and existing `dev_50` compatibility. |
| 2026-09-29 | `amended` | Hardened spec per review recommendations (R1–R6): adopted exact 111 held-out IDs from `clojure-llm`, added `quarantine.edn` escape hatch ($\ge 95\%$ verification floor), specified deterministic public/hidden test split rule, pinned upstream SHAs (HF `28441b6`, GitHub `3025a53`), named `:multipl-e-v0` prompt template, and reconciled cell arithmetic (single-shot ~2.3h, agentic ~5.0h, 1116-cell suite ~7.4h). |
| 2026-09-29 | `implemented` | Stage 2 implementation complete. Ingestion CLI (`ingest.clj`) normalized 558 tasks, partitioned 447 dev / 111 sealed tasks (verbatim `clojure-llm` IDs), applied deterministic AC4 test split, and verified 543/558 tasks (97.31%) in hermetic SCI sandbox. 15 quarantined tasks cataloged with diagnostic rationales. Backwards compatibility for `dev_50` preserved. `--tasks multipl-e-dev` and `--tasks multipl-e-sealed` wired into benchmark runner with `:multipl-e-v0` prompt template. |
| 2026-09-29 | `reviewed` | Independent adversarial review (Axiom): corpus, split, and verification floor confirmed — 447 dev / 111 sealed, disjoint; sealed IDs zero-diff vs `clojure-llm` 111-ID held-out manifest (32 HE + 79 MBPP); SCI floor independently reproduced at 543/558 = 97.31% with the quarantine set exactly the complement of the passing set (no silent waivers); `dev_50` backwards compatibility preserved (604 solution keys: 554 ID + 50 legacy). Six follow-ups raised: quarantine enforcement, sealed adjudications, truthiness coercion, fenced preset wiring, solution provenance, BACKLOG attribution. |
| 2026-09-29 | `remediated` | All review follow-ups landed: zero-test grading rejects vacuous passes; quarantine enforced at run time (`:include-quarantine`, default exclude); all 15 quarantine entries carry `:split` and technical adjudications; `multipl-e-dev`/`sealed` presets default `:tool-syntax :fenced`; `provenance.edn` (604 records: 476 exact upstream candidate matches, 50 dev-50-legacy, 69 synthetic-reference, 9 patch-fix) added; task files regenerated with `(boolean ...)` truthiness coercion on all 232 true-expecting assertions (verified: only wrappers changed). |
| 2026-09-29 | `cataloged` | Capability verdict `done` recorded. Frozen spec copy promoted to `resources/catalog/gate3_evals/multipl_e/spec.md`; registry entry `:multipl-e` added to `resources/catalog/registry.edn`. |

---

## 6. Catalog Record

**Verdict: done** (2026-09-29).

AC disposition at cataloging (independently verified by review):

- **AC1** — Pinned 558-task ingestion: **done**. 161 `humaneval-clj` + 397 `mbpp-clj` ingested from pinned upstream SHAs (HF `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`, GitHub `3025a531af7450e7df8b96fe0440e9804480bbad`, cross-checked against `nibzard/clojure-llm` `8ed80ac5`).
- **AC2** — SCI verification floor: **done**. 543/558 = 97.31% reference solutions pass 100% of assertions in hermetic SCI sandbox (floor: ≥530). 15 quarantined tasks isolated in committed `quarantine.edn` with per-task technical adjudications; quarantine set is exactly the complement of the passing set — no silent waivers. Runner enforces quarantine by default (`:include-quarantine false`); `grade-submission` rejects zero-test cases as vacuous passes.
- **AC3** — Exact 111 held-out IDs: **done**. Sealed partition zero-diff vs `clojure-llm` held-out manifest (32 HE + 79 MBPP), disjoint from dev (447/111).
- **AC4** — Public/hidden split rule: **done**. Deterministic rule implemented in `ingest.clj` and covered by generative property tests; bare `(is (f ...))` assertions normalized with `(boolean ...)` coercion to match `clojure.test` truthiness semantics (232 cases).
- **AC5** — `:multipl-e-v0` prompt template: **done**. Default for `--tasks multipl-e-dev`/`multipl-e-sealed`; agentic preset defaults `:tool-syntax :fenced`, explicit override respected.
- **AC6** — Test suite non-regression: **done**. `multipl-e-test` 10/4030, `clojure-bench-test` 19/113, `agent-test` 43/209 green, including the `dev_50` backwards-compatibility fixture.
- **AC7** — Harness integration & dry-run ledgering: **done**. `--dry-run` executes against both partitions with valid ledger rows (`:harness-sha`, `:sealed-sha`, `:prompt-sha`, `:temperature`, `:dry-run?`); sealed dry-run yields 106 rows default (5 quarantined excluded), 111 with `--include-quarantine true`.

**Non-goal boundary honored**: no model inference during the capability stage; all grading remains in-process via SCI. Model runs (prompt tuning, distillation evals) belong to the unlocked downstream proposals.

Sealed-corpus integrity anchor: `tasks_sealed.edn` SHA-256
`c5bce6c97e320f1e491e67fc9f0c0d8037ef2889ecd3e130ac1ca75a614d309f`.
