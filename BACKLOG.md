# BACKLOG — the flight plan

The gates are the altimeter; this is the flight plan. Every line of
active work is visible here, triaged by track, with status and
dependencies. See `PROCESS.md` §6 for the two-track system and
`resources/proposals/README.md` for proposal rules.

**Status values:** `proposed` (designed, not started) · `active`
(underway) · `done` (verdict recorded) · `parked` (explicitly deferred,
with reason) · `killed` (stopped, reason recorded — a contribution).

**Tracks:** `experiment` (falsifiable, verdict) · `capability`
(judgment-called, acceptance criteria) · `measurement` (standing data
program).

---

## Active queue

| Item | Track | Status | Owner | Blocked by / Unlocks | Notes |
|---|---|---|---|---|---|
| Prompt tuning v1 (worked agentic trajectories) | experiment | active | — | Unlocked by MultiPL-E dev corpus | Stage 1 spec proposed; freeze reference prompt; v1 with 2–3 disjoint worked trajectories; iterate on dev, report on sealed-10 only |
| LoRA pure-stack trainer | capability | proposed | — | Unlocks distillation SFT | Large ticket. Pure Clojure/XLA stack, no PyTorch (program constraint) |
| Student perplexity on teacher trajectories | measurement | proposed | — | Student = E4B-QAT-INT4 (3/10); teacher = 31B-QAT-INT4 (7/10); gap is 4 tasks | Inference-only distillation de-risk; runs before the trainer exists |
| Program-signature versioning | capability | proposed | — | Superseded in part by :harness-sha (freeze) | Extend prompt-sha: hash(prompt, tool defs, turn structure, decoding params, harness version) on every ledger row |
| Grading-sandbox hardening (slurp/spit) | capability | proposed | — | — | From adversarial review; load-bearing before the loop becomes training infra |
| Paren-repair as agent-facing tool | experiment | proposed | — | — | Prompt-tuning sub-avenue; explicit tool, never silent harness fixup |
| QAT adoption decision record | capability | proposed | — | — | DECISIONS.md: judgment call → rule confirmation (7/10 > 6/10) |

## Recently done

| Item | Track | Verdict |
|---|---|---|
| MultiPL-E humaneval-clj/mbpp-clj port | capability | done — 558 tasks ingested (161 HE + 397 MBPP); 447 dev / 111 sealed partition; 543/558 (97.31%) verified in pure SCI; 15 quarantined with technical adjudications |
| Eval harness v2 (KV cache, library port, --tool-syntax) | capability | done — 21.3% pre-conformance wall-time reduction on 31B (isolated KV reuse); AC2 verified (no adverse flips vs v1); library in einsum.agent.core; --tool-syntax :fenced lifts E4B agentic from 20% native (30% union) to 50% & unlocks my-comp on 31B (70% agentic suite) |
| clojure_bench freeze (v1 → catalog) | capability | done — moved to `resources/catalog/gate3_evals/clojure_bench/`; rows carry `:harness-sha`/`:harness-dirty?` from 2026-09-27 |
| E4B-QAT-INT4 eval | experiment | 3/10 union — beats PTQ-INT4 (1/10) and BF16 (~2/10); **student substrate decided** |
| clojure_bench v1 instrument | capability | done — 200 ledger rows, sealed fixture, append-only |
| Best-submission ratchet | capability | done |
| Prompt-sha ledger tracking | capability | done |
| E4B-INT4 baseline | experiment | 1/10 union — PTQ-INT4 roughly halves E4B BF16 (~2–3/10); single-shot 0/10, agentic recovers first-n via 3 submissions |
| QAT-31B substrate evaluation | experiment | **ADOPT** (judgment call, then rule-confirmed 7/10 > 6/10) |
| 12B-QAT 2×2 matrix (rep-pen × mode) | experiment | Finding: below thinking-convergence threshold; no decoding rescue |
| Repetition-penalty operating config | experiment | **1.0** adopted; 1.15 deprecated (hurts agentic on both substrates) |

## Parked

| Item | Track | Reason |
|---|---|---|
| 12B diffusion variant | capability | Different inference paradigm (iterative denoising); separate project, not an eval arm |
| 26B-A4B MoE | capability | Needs MoE infra (routing, expert dispatch); revisit if speed-at-capability is needed |
| Ternary / CAT-Q / E2M-ATQ / KOTMS | experiment | Quant track paused behind the baseline/eval program |
| Chunked prefill, MTP speculative decoding, INT4 KV, W4A16 profiling artifacts | capability | Runtime work; sequenced after the eval program |
| RLVR stage | experiment | After SFT; needs rollout+update loop |
| E2B-QAT eval | experiment | Only if E2B becomes relevant (currently out as student) |
| EmbeddingGemma port | capability | Separate bidirectional-encoder question; tokenizer compat unestablished |
| Alternative-model hedge (Muse-Glimmer-30B, MiMo-9B) | experiment | Parked; Gemma 4 31B retained as the vehicle |
| Task D | experiment | Parked by Simon |

---

## How to work this list

1. Pick from the top of Active queue; respect `Blocked by`.
2. Experiments enter via `resources/proposals/TEMPLATE.md`; capabilities
   via `CAPABILITY-TEMPLATE.md`. No template, no work.
3. Move the row when status changes. Done rows carry their verdict;
   killed rows carry their reason. Nothing disappears silently.
4. Parked rows are revisited deliberately, never by drift. To unpark:
   state what changed.
