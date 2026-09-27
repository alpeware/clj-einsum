# RFC: <Capability Title>

**Capability**: `<slug>` \
**Type**: `capability` \
**Gate**: `<gate1_compression | gate2_velocity | gate3_evals | gate4_recursion>` \
**Generation**: `<target-generation-number, e.g. 1>` \
**Literature**: `["<Author et al. (Year) arXiv:XXXX.XXXXX>"]` *(optional — capabilities often have no paper; cite the nearest prior art or write `[]`)* \
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "<e.g. LoRA rank-16 on E4B-INT4 trains within 24GB>"}` \
**Extends**: `"<prior-slug-or-nil>"` \
**Refutes**: `"<refuted-claim-or-nil>"` \
**Supersedes**: `"<superseded-slug-or-nil>"` \
**Reopens**: `"<reopened-slug-or-nil>"` \
**Unlocks**: `["<experiment-slug-this-unblocks>", ...]`

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

*What program goal needs this? State the judgment call plainly: what
scaffolding is missing, why we believe it is missing, and what changes
once it exists. Name the lottery tickets this unlocks — a capability
that unlocks no experiment is a hobby.*

---

## 2. Acceptance Criteria & Non-Goals

Every capability must state measurable done-conditions *before* work
starts. Each criterion must be checkable by a third party without
taste. When all criteria hold, the capability is done — no further
argument required.

- **AC1**: `<e.g. Rank-16 LoRA adapters train on E4B-INT4 within 24GB VRAM, end to end, via the pure Clojure/XLA stack>`
- **AC2**: `<e.g. Training loss decreases monotonically on a 100-example toy task over 3 epochs>`
- **AC3**: `<e.g. Adapters save/load round-trip bit-identically and apply to the base checkpoint without recompilation>`

**Non-goals** (what this explicitly does *not* do — keeps scope tight):

- `<e.g. No multi-GPU distribution; single 7900 XTX only>`
- `<e.g. No full-finetune support; LoRA only in v1>`

**Verdict.** When the criteria hold: `done`. If the program decides not
to build it after all: `killed-by-judgment`, with the reason recorded
here. A killed capability is a contribution, like a null result —
never silently dropped.

---

## 3. Interface & Integration

*Where does this live, and how do experiments consume it? Follow the
directory ownership in `AGENTS.md`.*

- **Code location**: `<e.g. src/einsum/train/lora.clj>`
- **Consumed by**: `<e.g. experiments/gate3_evals/distill_sft/run.clj via train!>`
- **Artifacts produced**: `<e.g. .models/<base>-lora-r16/ adapter checkpoints>`
- **Harness changes**: `<e.g. ledger gains :lora-sha; nil if none>`

---

## 4. Cost Estimate

*Rough engineering/compute budget, as judgment input — not a promise.*

- Engineering: `<e.g. ~2 agent-weeks>`
- Compute: `<e.g. ~20 host-GPU hours for acceptance runs>`
- Sequencing: `<e.g. Blocks distill_sft; independent of prompt tuning>`

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| `<YYYY-MM-DD>` | `proposed` | `<why now>` |
| `<YYYY-MM-DD>` | `done` / `killed-by-judgment` | `<acceptance evidence or kill reason>` |
