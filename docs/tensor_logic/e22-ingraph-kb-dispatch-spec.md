# Experiment E22 — In-Graph KB Query Dispatch

**Status:** SPEC (unimplemented)
**Date:** 2026-09-17
**Thread:** Agent loop (builds on E21; implements the Tier 2 KB-query arm
from AGENT-LOOP.md §9)

---

## 1. Thesis

E21 built the write path. E22 builds the read path the agent loop
actually needs: the LLM emits a KB query **as tokens**, the query is
**recognized in-graph by token-ID compare**, **executed against
VRAM-resident relation tensors with no host-side parsing**, and the
answer **feeds back into the decode loop as tokens**. This is the
"recognize the datalog tool call on the GPU" step from the original
sketch, now as a falsifiable experiment.

The demo: the E21 family-graph KB (64 entities, 8 trees) resident in
VRAM; Gemma 4 E2B prompted with questions like "give me all ancestors
of person 3"; the model must call the KB tool to answer. These are
synthetic entities the model never saw — without the tool it can only
hallucinate, which is precisely the causal test.

## 2. Hardware / reuse

- **GPU required** (ROCm, PJRT). The E-series convention; the coding
  agent's machine.
- Reuse: `scripts/gemma4_agent.clj` VRAM-session pattern
  (`init-agent-vram-session`, weights pinned, static-shape decode with
  `:max-seq-len` padding, dynamic slice/update); `clj-xla.logic.kb`
  (E21) as the oracle and the source of materialized relations.

## 3. Design

### 3.1 The KB in VRAM

Three 64×64 boolean tensors, allocated once in the same PJRT
client/context as the model weights:

- `PARENT` — from the E21 KB's committed facts.
- `SIBLING` — from the E21 KB (pair-identity constraint).
- `ANCESTOR` — transitive closure of parent/2, **materialized
  host-side** from the E21 KB and shipped over. Per E21's rule,
  fixpoint stays host-side; E22 tests *dispatch*, not derivation.
  Deriving closure in-graph would confound the measurement.

Query = row gather. `ancestors(3)` is one gather of row 3 of
`ANCESTOR`. Microseconds; the decode dominates end-to-end latency.

### 3.2 The protocol (FSM-trivial by design)

In-graph parsing of free text is a non-starter, so the query is a
fixed-format token window:

```
<kbq> REL ARG </kbq>
```

- `REL ∈ {ancestors, siblings, parent}` — one token each.
- `ARG` — entity index as digit tokens (`3`, or multi-digit via a
  tiny digit-accumulation FSM; v1 demo uses entities 0–7, single
  digit — asserted single-token in Phase 0, §5).
- The detector is token-ID compare on `<kbq>`, exactly the interrupt
  mechanism in AGENT-LOOP.md §9. On fire: the chained KB executable
  runs (device→device; inputs already resident), writing up to 16
  entity indices + a count into a fixed-size answer buffer.
- Answer feedback: indices → digit tokens via the device-resident
  index→token-ID table (built host-side once, the verbalization
  bridge in its minimal form), injected into the decode stream as an
  observation, e.g. `<kbans> 1 0 </kbans>`. The model then verbalizes.

### 3.3 Honest scoping of "no host round-trip"

Stage A semantics (AGENT-LOOP.md §9): the host still pumps decode
steps — full Stage-B in-graph `while` is a separate research build.
What never touches the host: **recognition** (token-ID compare),
**parsing** (fixed-format window, no JSON/SCI), **execution** (chained
device executable), and the **answer path** (stays in VRAM until it's
tokens again). The host pumps; the host doesn't think. The win is
architectural (no parse/dispatch logic, no data-dependent host
branching) plus latency (no transfers beyond the already-pumped
loop). Do not claim "zero host involvement."

### 3.4 Prompting

Few-shot: 3–4 worked examples showing the protocol
(`Q: Who are the ancestors of person 3?` → `<kbq> ancestors 3 </kbq>`
→ `<kbans> 1 0 </kbans>` → `A: The ancestors of person 3 are persons
1 and 0.`). Temperature 0 for eval. Protocol-emission rate is itself
the primary behavioral metric (§5, P1).

## 4. Cells

- **H (in-graph dispatch):** full path — prompt → decode → `<kbq>`
  detected → device KB query → `<kbans>` fed back → final answer.
- **B0 (no tool):** same prompt minus the tool documentation and
  examples. The model must answer from weights/context. Expected:
  hallucination or refusal — the causal demonstration.
- **B1 (host-side lookup):** the `<kbq>` is parsed host-side and
  answered from the E21 KB (data crosses to host and back). Accuracy
  should tie H; the comparison is latency + architecture, isolating
  what in-graph dispatch buys.

## 5. Phases and gates

### Phase 0 (gates Phase 1 — all must pass)

- **P0a — tokenizer check:** `<kbq>`, `</kbq>`, `<kbans>`,
  `</kbans>`, `ancestors`, `siblings`, `parent`, digits `0-9` are
  single tokens in Gemma 4's vocab. If any isn't: pick alternates
  that are, or stop and report (do not silently multi-token).
- **P0b — query parity:** for all 64 entities × 3 relations, the
  VRAM gather results ≡ E21 KB answers exactly. Mismatch → stop;
  the port is wrong.
- **P0c — emission smoke test:** 5 dry-run prompts; does the model
  emit well-formed `<kbq>` at T=0 prompting? < 3/5 → flag the
  verbalization risk before burning the full eval.

### Phase 1 — eval

Scripted question set: entities 0–7 (tree 0) × {ancestors, siblings}
× 3 phrasings = 48 questions. Ground truth from the E21 KB.
Metrics per cell:

1. **Protocol emission rate** — fraction of questions where a
   well-formed `<kbq>` was emitted (H only; the behavioral metric).
2. **End-to-end exact-match accuracy** — the final answer's entity
   set ≡ ground truth (all cells).
3. **Per-query latency** — device time of the KB executable (H) vs
   host round-trip (B1); end-to-end per-question time (all cells).
4. **Failure-mode log** — for every miss: no-emission /
   malformed-query / misread-answer / wrong-entities. Quoted
   examples, not just counts.

### Pre-registered criteria

1. **P0b parity exact** (gate, §5).
2. **Emission:** H protocol-emission rate ≥ 80%. Below → the
   LLM-behavior hypothesis fails; report as a verbalization-bridge
   finding. Do not hand-hold per question to inflate the number.
3. **Accuracy:** H exact-match ≥ 90% *on questions with well-formed
   queries*, and H − B0 ≥ +0.50 (the causal gap). B0 is expected
   near zero — if B0 scores high, the entities leaked into the
   prompt or the model's context and the experiment is void.
4. **Latency:** per-query device time reported; H vs B1 compared
   honestly (expect the query itself in microseconds; end-to-end
   dominated by decode either way).

### Falsification clause

If criterion 2 fails, E22 fails at the LLM-behavior level: the
tensor path works (P0b) but the model won't speak the protocol
under few-shot prompting. That is a finding, not a bug — the
remedies (constrained decoding, fine-tuning on the protocol) are a
different experiment. If criterion 3 fails with emission passing,
the feedback path is lossy — debug before any follow-up.

## 6. Non-goals (explicitly deferred)

- Names instead of indices (verbalization-bridge upgrade;
  AGENT-LOOP.md open question 8).
- Multi-hop *compositional* queries in-graph (`grandparent` via two
  gathers — trivial extension once E22 passes; note it, don't build
  it here).
- Stage-B fully in-graph `while` (research build).
- KB writes from the device (writes stay host-verified per E21).

## 7. Deliverables

- `src/clj_xla/logic/kb_device.clj` — VRAM-resident relation tensors,
  the query executable (gather + answer buffer), the token-ID
  detector wiring. Sans-IO where the boundary allows; device code
  through the existing tensor-logic → StableHLO → PJRT path (repo
  Rule 4 — no host float loops, no Java escape hatches).
- `scripts/e22_kb_dispatch.clj` — Phase 0 checks + Phase 1 eval
  (H / B0 / B1), printing the criteria report.
- `paper-experiments/e22-dispatch/2026-09-17/` — `phase0.edn`,
  `results.edn` (per-question log: prompt, emitted tokens,
  `<kbans>` contents, final answer, timings).
- `docs/tensor_logic/empirical_journey.md` — new **Section 23**.

## 8. Delivery instruction

**On completion: `git add` the spec's artifacts, commit
(`feat(e22): in-graph KB query dispatch with Gemma 4 E2B`), and
`git push origin main`. Do not wait for a separate review — the
criteria above are the review.**
