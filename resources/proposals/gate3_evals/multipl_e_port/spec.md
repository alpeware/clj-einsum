# RFC: MultiPL-E humaneval-clj & mbpp-clj Port

**Capability**: `multipl_e_port` \
**Type**: `capability` \
**Gate**: `gate3_evals` \
**Generation**: `1` \
**Literature**: `["Cassano et al. (2023) MultiPL-E: A Scalable and Extensible Polyglot Benchmark for Large Language Models (arXiv:2208.08227)", "nuprl/MultiPL-E Pull Request #136: Add Clojure programming language (2023)", "nibzard (2026) clojure-llm: Benchmark-first autoresearch for Clojure code models (https://github.com/nibzard/clojure-llm)"]` \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "558-cell MultiPL-E evaluation completes in < 4 hours on E4B-QAT-INT4 with prefix KV reuse"}` \
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
   A pioneering autoresearch program exploring whether smaller open models
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
  - We will **expand the catalog pod** to house the complete, verified 558-task suite partitioned into `tasks_dev.edn` (447 tasks) and `tasks_sealed.edn` (111 tasks), accompanied by complete ground-truth solutions in `solutions.edn`.

---

## 2. Acceptance Criteria & Non-Goals

Every capability must state measurable done-conditions *before* work starts.
When all criteria hold, the capability is done.

- **AC1: Full 558-Task Ingestion**:
  Ingest all 161 `humaneval-clj` tasks and 397 `mbpp-clj` tasks from the official
  `nuprl/MultiPL-E` dataset. Each task is normalized to the canonical harness schema:
  `{:id ... :source ... :fn-name ... :prompt ... :public-tests [...] :hidden-tests [...]}`.
- **AC2: 100% SCI Sandbox Verification**:
  Every ground-truth reference solution in `solutions.edn` must execute cleanly and
  pass all associated `:public-tests` and `:hidden-tests` within `create-benchmark-sci-ctx`.
  Zero uncaught reflection exceptions, zero missing standard library functions,
  and zero external host I/O dependencies.
- **AC3: Deterministic Partitioning (447 Dev / 111 Sealed)**:
  Partition the 558 tasks into:
  - `tasks_dev.edn` (447 tasks): An open development partition available for prompt
    tuning (`prompt_tuning_v1`), few-shot trajectory mining, and teacher rollout generation.
  - `tasks_sealed.edn` (111 tasks): A strictly held-out evaluation partition (matching the
    `clojure-llm` 20% test split convention) reserved for un-snooped model capability reporting.
- **AC4: Existing Test Suite Non-Regression**:
  Preserve backwards compatibility for `test/einsum/agent_test.clj`
  (`test-multipl-e-dev-50-fixture-and-ledger`) so existing regression tests pass with
  zero errors.
- **AC5: Harness Integration & Dry-Run Ledgering**:
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
  - `solutions.edn`: 558 verified reference solutions.
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
