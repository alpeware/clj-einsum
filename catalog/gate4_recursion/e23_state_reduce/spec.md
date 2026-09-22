# E23 — The Reduce: Verified KB Accumulation Over a Horizon (RBAC)

Date: 2026-09-17
Status: SPEC (not implemented)
Depends on: E21 (verified write path), E22 (KB query dispatch)
Extends: E22's read arm with the write arm; implements the reduce from
  AGENT-LOOP.md §9: `state' = verified_commit(state, llm_proposals)`.

## 1. Background

E21 built the verified write path (typecheck → constraint → ambiguity →
provenance → commit/reject) and proved it on a static family graph.
E22 proved the read arm: Gemma 4 E2B emits native tool calls, queries the
KB at 100% emission / 100% exact-match, and the host-side path won on
latency (42µs vs 692µs device).

What neither experiment tested: the KB *changing* across steps under
model direction — the accumulation half of the original sketch ("update
the KB from the host side after tool calls"). E23 closes that gap. The
agent runs a multi-step reduce: each step it observes an event, queries
the KB (E22 read arm), *proposes fact edits* (new arm), and the E21 gate
commits or rejects them. The next step's snapshot is the new KB state.

The family tree is the wrong domain for this: it is a static generator
and new facts would require contrived births/deaths. The domain is a
**synthetic RBAC system**: joins, grants, revokes, and role changes are
natural fact arrivals; separation-of-duty violations are *meaningful*
(a bad grant is a security hole), so the verified write path is
load-bearing in the story, not just in the test; and "who has access to
what" is canonical LLM confabulation territory. A synthetic software
graph (services, deploys, dependency acyclicity) is the natural second
domain for the same harness, closer to the coding-agent vision — noted
as follow-up, not this experiment.

## 2. Goal

Demonstrate the reduce end-to-end: over a scripted event stream of
length T, the agent maintains a KB snapshot via proposed-then-verified
writes; the snapshot (a) contains zero constraint-violating facts, (b)
matches the oracle state at high F1, (c) does not degrade with horizon
length — while an in-context baseline (B0) does.

## 3. Non-goals

- Device-resident writes. Writes stay host-side and verified, per the
  architecture (AGENT-LOOP.md §9: the KB lives host-side; the device
  holds the hot fact set for reads). E23 does not move the commit path
  into VRAM.
- Multi-agent or concurrent writers. Single agent, single stream.
- Natural (non-synthetic) event streams. The stream is scripted with a
  seeded generator so the oracle is exact.
- Role hierarchy / inheritance. The schema is minimal on purpose.
- The software-graph domain. Same harness, later experiment.

## 4. Domain and schema

Synthetic org, defaults (§11): 12 users, 6 roles, 8 permissions.

Relations (all discrete, E21-style):

- `user(u)`, `role(r)`, `perm(p)` — declared entities (static).
- `role_perm(r, p)` — static role→permission mapping.
- `has_role(u, r)` — **the dynamic relation.** The only relation the
  agent may assert/retract.
- `sod(r1, r2)` — static separation-of-duty pairs (3 pairs default).

Integrity constraints (checked on every commit, E21 pattern
generalized to a schema-driven commit — see §6):

1. **Domain:** every arg of every proposed fact must be a declared
   entity of the right sort. `has_role(mallory, developer)` with
   Mallory undeclared → `:domain`.
2. **Denial (SoD):** no user may hold both roles of a declared
   `sod` pair. Attempt → `:separation-of-duty`, naming the pair.
3. **Cardinality:** `admin` role ≤ 3 holders (default). Attempt beyond
   cap → `:cardinality`, naming the cap.
4. **No-op:** asserting an already-held role, or retracting a
   non-held role → `:no-op` (rejected, not an error; keeps the
   snapshot clean and gives the model a learnable signal).
5. **Ambiguity:** an underdetermined event must be stored as an
   explicit disjunction (E21 c4), never prematurely committed to one
   branch. A later disambiguating event collapses it via supersede.

## 5. Event stream and oracle

A seeded generator produces a scripted stream of T natural-language
observations (tickets, messages, HR notes). Event kinds:

- `grant(u, r)` — "Alice joined the backend team; grant her the
  developer role." (valid)
- `revoke(u, r)` — "Bob moved to frontend; revoke his deployer role."
  (valid)
- `trap-sod(u, r)` — grant that would violate a declared SoD pair.
- `trap-cardinality(u, r)` — grant beyond the admin cap.
- `trap-domain(u, r)` — grant to an undeclared user / undeclared role.
- `trap-noop(u, r)` — duplicate grant / vacuous revoke.
- `ambiguous` — "The ticket scan is blurry: the oncall role went to
  either Erin or Frank." → must become a disjunction.
- `disambiguate` — "Confirmed: it was Erin." → collapses the
  disjunction, superseding it.

Trap rate default 25%. The T=60 stream contains the T=20 stream as its
first 20 events, so horizon degradation is measured on a shared prefix.

**Oracle:** a deterministic applier folds the event stream through the
*same* constraint set, applying only valid events. Its final KB (and its
checkpoints at t=20) is the ground truth. The oracle never sees the
model; the model never sees the oracle.

## 6. The reduce (H cell)

E21's write path was family-graph-specific. E23 generalizes it to a
**schema-driven commit** — `commit-with-schema(schema, kb, proposal)`
returning `{:committed new-kb}` or `{:rejected violation}` — with the
RBAC schema (§4) as its first schema. The family schema must keep
passing its E21 tests through the generalized path (no regression).

Per step t (observation o_t, snapshot S_t):

1. Prompt = system (schema, tool docs, 2 few-shot propose examples)
   + rendered snapshot S_t (text view, §7) + o_t.
2. Model may emit `kb_query(...)` calls (E22 read arm, unchanged) and
   `kb_propose` calls:
   - `kb_propose(assert(has_role alice developer))`
   - `kb_propose(retract(has_role bob deployer))`
   (Native `<|tool_call|>` tokens, per E22's P0a finding — no invented
   protocol.)
3. Host applies each proposal through `commit-with-schema`. Rejections
   return the named violation and are fed back *within the step* as an
   observation ("rejected: separation-of-duty(deployer, auditor)"),
   so the model can revise rather than silently diverge. Max 3 propose
   rounds per step (default), then the step closes.
4. S_{t+1} = the KB after all commits. Next step.

The update function is the whole experiment:
`S_{t+1} = verified_commit(S_t, proposals_t)`.
The model proposes; the gate disposes. **We score the snapshot, not
the model.** The model *attempting* a trap grant is expected (it is
naive); the trap *present in the snapshot* is the failure.

## 7. Snapshot rendering

The harness owns the real KB (E21 store). The model sees a text view:

```
;; KB snapshot (t=17)
(has_role alice developer)
(has_role alice oncall)
...
;; ambiguous
(one-of (has_role erin oncall) (has_role frank oncall))
```

Bounded by the fact count (~20 held roles at default scale), never by
history. Per-step prompt cost is O(snapshot + step).

## 8. Cells

- **H (the reduce):** KB + verified writes, §6. The architecture's
  claim.
- **B0 (in-context baseline):** no KB. The model gets the same event
  stream as an accumulating transcript and is told to track access
  state; interrogated the same way. Current-practice baseline.
- **B1 (unverified KB):** KB present, proposals applied raw without
  the gate. Isolates what the gate buys; expect drift and accumulated
  hallucinated grants.

## 9. Interrogation and metrics

After the stream (and at the t=20 checkpoint for the long stream),
each cell is interrogated with scripted questions ("Does Carol hold
the auditor role?", "Who are the admins?", "List Alice's roles."),
scored exact-match against the oracle — plus a direct fact-level diff
of H/B1 snapshots vs the oracle (precision/recall/F1 on `has_role`
facts, disjunctions compared as sets).

- **Gate-hold (H, hard):** violating facts in final snapshot = 0.
  Any trap committed → the write path has a hole; stop.
- **Snapshot F1 (H):** ≥ 0.90 (default threshold; report actual).
- **Horizon flatness:** |F1_H(T=60) − F1_H(T=20)| ≤ 0.05. The
  architecture promises no degradation; measure it.
- **B0 degradation (informational):** report F1_B0(T=20) vs
  F1_B0(T=60). Expected to fall (context pressure). No threshold —
  the curve is the data.
- **Ambiguity lifecycle (binary):** disjunction stored at the
  ambiguous event (no premature branch commit); collapsed to the
  confirmed branch at disambiguation, with the supersede marker.
- **Rejection recovery (informational):** after a rejection, does the
  model stop re-proposing the same violating fact within the step?
- **Latency:** per-step prompt/decode time and commit-path time,
  reported honestly (E22's lesson: decode dominates; the commit path
  is microseconds).

## 10. Phase 0 (no LLM; all must pass before Phase 1)

- **P0a — oracle determinism:** run the generator + oracle applier on
  the scripted streams; assert the final KB equals a hand-computed
  expected set. Catches generator/oracle bugs before the model is
  involved.
- **P0b — trap rejection:** feed every trap event directly to
  `commit-with-schema`; assert `{:rejected <named-violation>}` for
  each (domain / separation-of-duty / cardinality / no-op). The gate
  is tested without the model in the loop.
- **P0c — write-arm emission smoke:** 5 scripted observations (2
  grants, 1 revoke, 1 trap, 1 ambiguous); assert the model emits
  well-formed `kb_propose` calls (parse check, not semantic).

## 11. Defaults (defensible; keep unless a better value is argued)

| Decision | Default | Rationale |
|---|---|---|
| Users / roles / perms | 12 / 6 / 8 | Small enough to read, large enough that tracking in-context is non-trivial |
| SoD pairs | 3 | Enough to trap, few enough to keep the schema legible |
| Admin cap | 3 | Exercises cardinality without dominating the stream |
| Horizons | T=20, T=60 (shared prefix) | Short enough to run twice, long enough to pressure B0's context |
| Trap rate | 25% | Traps are the measurement; valid events are the cover |
| Ambiguity events | 1 + 1 disambiguation per stream | The lifecycle, once, cleanly |
| Propose rounds / step | 3 | Bounds the rejection-revision loop |
| Snapshot F1 threshold (H) | 0.90 | High but not 1.0 — the model must actually drive |
| Horizon flatness (H) | ≤ 0.05 F1 drift | The no-degradation claim, quantified |
| Temperature | 0 | Deterministic; matches E22 |

## 12. Falsification criteria

1. A trap fact present in H's final snapshot → E21's write path has a
   hole. **Stop.** Fix the gate before any further reduce work.
2. H snapshot F1 < 0.90 → the model does not use the write arm
   reliably (behavioral finding, in the style of E22's emission
   result). The reduce works; the driver doesn't.
3. B0 flat across horizons → the long-horizon premise needs
   re-examination: either T=60 doesn't pressure the context (extend
   T) or in-context tracking is sturdier than assumed (revise the
   claim).
4. B1 ≈ H on all metrics → the gate buys nothing in this domain;
   the schema or traps are too weak. Strengthen, don't celebrate.

## 13. Telemetry

`catalog/agent-deliberation/e23-state-reduce/`: `phase0.edn`,
`stream.edn` (seeded events + oracle checkpoints), `results.edn`
(per-step: observation, proposals, commits/rejections with named
violations, snapshot hash; per-cell: interrogation log, fact-level
diff vs oracle). Journey section 24 on completion.

## 14. AGENT-LOOP.md linkage

On success, E23 becomes the empirical anchor for §9's write arm
("KB write arm (propose, never direct-write)") and the reduce
equation. On gate failure (criterion 1), it is an E21 regression
report instead.
