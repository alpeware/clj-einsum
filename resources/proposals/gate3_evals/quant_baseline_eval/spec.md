# RFC: Baseline-Anchored Comparative Eval for the Loop's Task Distribution

**Experiment**: `quant_baseline_eval`
**Gate**: `gate3_evals`
**Generation**: method v1 (this spec); baseline run = generation 1 numbers
**Literature**: `["Bespoke Labs (2026) AutoResearchExam — baseline-anchored reward maps, validation/hidden-test separation, HPO-only classification, adversarial reward-hacking review", "TWLA (ICML'26, arXiv:2606.13054) — the quant line this method will judge", "Evidence-annotated RAG protocol (arXiv:2609.28007) — failure-localization discipline, future extension"]`
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "retention CIs resolve ±5% on the sealed set; harness overhead <5% of inference wall time"}`
**Extends**: `nil`
**Refutes**: `"ad-hoc Gate 3 smoke testing (accepted broken Clojure with unbound vars; no uncertainty, no cost ledger, no sealed split)"`
**Supersedes**: `nil` (does not delete existing tools; replaces them as the gate instrument)
**Reopens**: `nil`

---

## 1. Abstract & Motivation

Every hard decision in this program is a *comparison*: INT4 vs BF16, ternary
vs INT4, E2B vs 31B, harness A vs harness B. Raw benchmark scores do not
answer comparisons — they answer "how smart," and they do it noisily, on
someone else's distribution, with no record of what the numbers cost.

This spec defines the lab's standing comparison instrument: a
**baseline-anchored retention eval**. Given a reference config and a
candidate config, it produces a retention report with uncertainty, a cost
ledger, and a mechanical verdict — on *our* task distribution (reasoning
and code the loop actually runs). It is generally useful: any future
quant ticket, model swap, or harness change is judged through it. The
first instantiation is the concrete use case that motivated it: the
Gemma 4 family (E2B, E4B, 12B, 31B) across quant levels, with the
INT4-31B reasoning/coding baseline as the ceiling artifact, closing the
loop once to establish committed numbers that E2M-ATQ, KOTMS, and every
later ticket are scored against.

Design theft, stated plainly: the baseline-anchored reward maps,
validation/hidden-test separation, and adversarial reward-hacking review
are ported from AutoResearchExam (Bespoke Labs, Sept 2026); the
predeclared kill criteria, matched-cost rule, and null-results discipline
are our own PROCESS.md. The benchmark fixes the harness and varies the
model; we invert it — we fix the model and vary the loop — so the
instrument is built for paired candidate-vs-reference runs, not
leaderboards.

---

## 2. Pre-Registered Hypotheses & Gate Criteria

### Gate 1: Resource Efficiency

- **Storage / Memory (Arithmetic):**
  - Harness-side state per run: `<= 50 MB` (prompt sets, results.edn,
    sealed hashes). The instrument must not become a storage line item.
  - Results artifact: single `results.edn` + `summary.csv`, content-hashed.
- **Compute / Throughput (Hypothesis):**
  - Scoring + statistics overhead `< 5%` of total inference wall time.
  - Falsification condition: the eval harness costs more than the
    inference it judges, or requires GPU time for pure scoring logic.

### Gate 2: Time Efficiency

- **Wall-Clock Latency:**
  - A full baseline matrix run completes unattended on the RX 7900 XTX in
    `< 12 hours` (exact budget fixed when per-task `n` is predeclared;
    the bound, not the number, is the criterion).
  - Zero retraining, zero gradient steps: forward inference + host-side
    scoring only.
  - Falsification condition: the run needs human babysitting mid-flight,
    or exceeds the predeclared wall budget by `> 25%`.

### Gate 3: Intelligence & Capability Floor

- **Floor Retention (`>= 95%`):**
  - For quant tickets judged against these baseline numbers: geometric-mean
    retention `>= 0.95` vs the reference config, with bootstrap 95% CI
    lower bound `>= 0.93`, and no single discriminative task below `0.80`
    retention (a collapse on one capability must not average out).
- **Absolute Capability Score:**
  - This baseline run *establishes* the absolute numbers; no kill fires
    on it. The numbers it commits become the floor future tickets defend.
- **Falsification conditions:**
  - The scorer accepts the known-bad completion the old Gate 3 smoke test
    accepted (Clojure with unbound `trimmed`). This is a **regression
    test on the scorer itself** — the instrument must demonstrably catch
    the failure that motivated it.
  - Retention CIs on the sealed set cannot resolve a 5-point difference
    (the set is underpowered; the report must say so, not claim "no
    difference").

### Gate 4: Continuous Recursion & The Compound Derivative

- **The Derivative of Judgment:**
  - Human judgment per baseline run: `< 30 minutes`, spent on the
    adversarial-review checklist (§7), not on eyeballing completions.
  - The loop itself can execute the full run unattended (G2 criterion);
    the human only reviews the verdict and the red-team checklist.
  - Deployment target: this becomes the standing G3 instrument — every
    proposal's Gate 3 section cites a run of it, not a bespoke script.
  - Falsification condition: running the eval requires more judgment
    than running the experiment it judges.

---

## 3. Design

### 3.1 The comparison, formalized

For config `c` and task `t`, let `s(c,t) ∈ [0,1]` be the task score.
Let `r` be the reference config. Retention on a task:

```
ρ(c,t) = s(c,t) / s(r,t)     if s(r,t) >= ε
         non-discriminative   otherwise
```

`ε = 0.2`: if the reference itself cannot do the task, the task carries
no retention signal — it is dropped from the retention aggregate and kept
in the absolute table. This blocks "retention laundering," where a high
retention number is manufactured from tasks nobody passes.

Aggregate retention (paired: identical task set for both configs):

- **Geometric mean** of `ρ` over discriminative tasks — the headline. It
  punishes single-task collapse, which is exactly the failure mode
  quantization exhibits (cf. CAT-Q layers 10–14).
- **Minimum** `ρ` — the honest number, reported alongside, never buried.
- **Arithmetic mean** — reported for comparability with literature.

Absolute scores `s(c,t)` are always published next to retention, so a
rising floor cannot hide a sinking ceiling (PROCESS.md §3, G3).

### 3.2 Tasks: our distribution, execution-checked

Every task ships as data: `{id, type, dev-prompt-set, test-prompt-set-hash,
scorer, n, budget}`. The scorer is a pure function of
`(prompt, completion, reference) → [0,1]`. Four task types for the
baseline instantiation:

1. **`clojure-gen`** — prompt → model writes Clojure. Scorer: (a) the
   completion *reads* without exception, (b) *compiles* with no unbound
   vars (macroexpand + resolve check — the exact hole in the old smoke
   test), (c) passes hidden unit tests executed in a sandbox. Score =
   weighted pass rate. This is "coding as the arena" made mechanical.
2. **`einsum-spec`** — prompt → model writes an einsum/TL expression.
   Scorer: parses as a valid declarative TL AST *and* contracts to the
   reference value within `ε ≤ 1e-3` on the CPU interpreter (PJRT where
   available). Domain-native: the loop's own action language.
3. **`reasoning`** — GSM8K-style numeric exact-match slice. Deterministic,
   no judge.
4. **`generation-quality`** — the upgraded smoke set: small, diagnostic,
   execution-checked where possible. Kept deliberately small; it is a
   tripwire, not a benchmark.

**LLM-as-judge is not a primary scorer.** The judge would be the system
under test (circular), and it costs frontier inference per completion.
A pinned frontier judge may appear as a *secondary diagnostic column*,
never in the retention aggregate.

**Long-context probe (extension, not core):** NIAH-style retrieval at
each config's `S_max`, reported as a separate table. Never folded into
the retention number — context failure is a different failure mode from
quant degradation, and mixing them produces uninterpretable aggregates.

### 3.3 Dev/test split: tune on visible, report on sealed

- Each task's prompt set is split `:dev` (visible, for prompt/format
  debugging) and `:test` (sealed: sha256-committed before any candidate
  runs, scored exactly once per config).
- Prompt engineering, format fixes, and harness debugging happen on
  `:dev`. Touching `:test` for anything but the single scoring run is a
  protocol violation and voids the verdict.
- This is the AutoResearchExam validation/hidden-test separation, applied
  to our scale: we cannot afford their 24-hour frontier runs, but we can
  afford the discipline.

### 3.4 Statistics: uncertainty is part of the number

All in `einsum.eval.core`, pure, deterministic under a fixed seed:

- `pass@k` with the unbiased estimator for generation tasks.
- Bootstrap (seeded, `B = 10000`) 95% CI on the geometric-mean retention,
  resampling tasks (paired resampling: same task indices for both configs —
  the matched comparison).
- Paired permutation test for H0: candidate ≡ reference on the task set.
- **Power pre-registration:** the proposal fixes the minimum detectable
  effect (5 points). If the sealed set is too small to resolve it, the
  report states "underpowered for effects < X" instead of "no significant
  difference." Absence of evidence is not evidence of absence, and the
  instrument says which one it has.

### 3.5 Cost ledger: a score without a cost row is not a result

Every result row carries (schema in `einsum.eval.ledger`):

```clojure
{:config-id        "gemma4-31B-int4"
 :checkpoint-sha   "sha256:…"      ; of the weight artifact actually loaded
 :harness-git-sha  "…"             ; this repo, the run namespace
 :prompt-set-hash  "sha256:…"      ; dev/test split commitment
 :hardware        "AMD Radeon RX 7900 XTX (24GB)"
 :weights-gb       17.6            ; measured, not projected
 :kv-bytes-at-smax …
 :s-max-tokens     …               ; KV-supported max context for decode
 :decode-tok-s-at-smax …
 :wall-minutes     …
 :judgment-minutes …               ; human time spent (G4)
 :scores           {task-id s}
 :retention        {:geomean … :min … :arith … :ci95 [… …]}}
```

Matched-cost rule (§3.2 of PROCESS.md) is thus mechanized: two configs
are comparable only with their cost rows attached.

### 3.6 Verdict: criteria as data

```clojure
{:min-geomean-retention 0.95
 :min-ci-lower          0.93
 :min-task-retention    0.80
 :max-judgment-min      30}
;; → :pass | :fail | :inconclusive   (CI straddles the bar)
```

The verdict function is pure and tested. `:inconclusive` is a first-class
outcome — it means "run more tasks," not "ship it."

### 3.7 Namespace layout (Rule 4: host-side is correct here)

Scoring, statistics, and verdicts are discrete symbolic logic — they
belong host-side in pure Clojure, not on device. No StableHLO involved,
and that is the mechanically sympathetic choice, stated so nobody
"fixes" it later.

- `einsum.eval.core` — pure: `task-score`, `retention`, `geomean/min/arith`,
  `bootstrap-ci`, `permutation-pvalue`, `verdict`.
- `einsum.eval.tasks` — task registry (data), dev/test split enforcement,
  sealed-hash commitment and verification.
- `einsum.eval.ledger` — cost-row construction and schema validation.
- `einsum.eval.report` — `results.edn` / `summary.csv` writers.
- `experiments.gate3-evals.quant-baseline-eval.run` — the impure boundary:
  drives the model (via `tools/gemma4.sh` on Simon's host), collects
  completions, calls the pure core. GPU work happens on the 7900 XTX;
  the sandbox runs only the pure statistical layer (CPU-sufficient).
- TDD: generative tests for statistical invariants *before* implementation
  (bootstrap CI monotonic in effect size, retention symmetric under
  config swap, verdict totality over the criteria lattice).

---

## 4. Execution Harness & Silicon Verification Plan

- **Harness Namespace**: `experiments.gate3-evals.quant-baseline-eval.run`
- **Output Artifacts**: `resources/proposals/gate3_evals/quant_baseline_eval/results.edn`,
  `summary.csv`, plus the sealed test-set hashes committed *before* the run.
- **Differential Parity**: n/a for the pure layer (no tensor math); the
  `einsum-spec` task type reuses the interpreter parity bound (`ε ≤ 1e-3`).

### 4.1 The concrete baseline instantiation (the use case)

Config matrix: models `{E2B, E4B, 12B, 31B}` × quant levels
`{bf16 (where it fits in 24GB), int4, catq-ternary (when it lands)}`.

**The honest-reference problem, stated in advance:** 31B BF16 (~62GB+)
cannot load on the 7900 XTX. Therefore the 31B column's reference is the
**INT4-31B ceiling artifact itself**, labeled `"deployed-ceiling
reference, not BF16"` in the ledger. Cross-model cells (E2B BF16 vs 31B
INT4) are *absolute-score comparisons*, never retention claims. The spec
exists to catch exactly this kind of quiet baseline substitution — so it
is written here, before any number exists.

Each config runs at its own `S_max` (KV-supported max decode context);
`S_max` is a reported cost-axis value, and task prompts are fixed length
so context is not a confound. The long-context probe is separate (§3.2).

**Predeclared kill criteria for the tickets this baseline will judge**
(E2M-ATQ, KOTMS — the baseline run itself establishes numbers, so no
kill fires on it):

- A quant ticket passes iff `verdict` → `:pass` against these baseline
  numbers on the sealed set.
- `:inconclusive` → the ticket funds a larger sealed set or dies; it does
  not ship on vibes.
- Any ticket whose proponent tunes on the sealed set is void, and the
  voiding is recorded as a contribution (null results are contributions;
  protocol violations are data too).

---

## 5. What this method explicitly does not do

- No perplexity-only reporting (perplexity is a diagnostic column, not a verdict).
- No LLM-judge primary scores.
- No test-set tuning; no leaderboard without cost rows.
- No pooling of short-context retention with long-context probe results.
- No claiming reference numbers for configs never run (the 31B-BF16 rule).

---

## 6. Adversarial review (standing step)

Before any baseline `results.edn` is accepted into the catalog, a second
pass — agent or Simon — red-teams the harness against a committed
checklist kept in this proposal directory:

1. Could a completion score without being correct? (scorer loopholes —
   the unbound-var class of bug)
2. Did any `:test` material influence prompts, formats, or thresholds?
   (leakage — check git timestamps vs the seal commitment)
3. Is every number in the ledger measured, with projected values labeled?
4. Does the verdict reproduce from `results.edn` alone? (replay the pure
   core on the committed artifact)

The checklist sign-off is part of the committed record. This is the
ported AutoResearchExam adversarial review, and it is where the old
Gate 3 would have died before publishing its 3/3.

---

## 7. Follow-up tickets (parked, not part of this spec)

- **AUARC-style time-to-verified** for agentic coding runs: area under the
  verified-progress curve, rewarding reaching working code earlier. G4-facing;
  needs the loop instrumented first. Parked until the baseline lands.
- **Evidence-annotated failure localization** (arXiv:2609.28007 protocol)
  for the retrieval/memory tasks: separate retrieval failure from
  reasoning failure. Relevant when the vector-memory line starts producing
  evals.
- **HPO-only classification analog**: detect runs that spend turns tuning
  prompts/hyperparameters without ever committing a measurement — the
  harness-tuner version of AutoResearchExam's tune-forever agents.
