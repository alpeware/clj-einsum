# Proposals Directory (Stage 1 & 2 Drafts)

This directory houses ephemeral proposal drafts (Stage 1) and active unverified implementation scaffolds (Stage 2).

---

## Two tracks

Work here comes in two tracks. Both are predeclared; neither is optional.

- **Experiments** (`TEMPLATE.md`) — falsifiable claims about the world
  (substrates, decoding settings, prompt variants). Pre-registered
  hypotheses, decision rules, kill criteria. Verdicts: `adopt` /
  `reject` / `killed-by-evidence`.
- **Capabilities** (`CAPABILITY-TEMPLATE.md`) — judgment-called
  infrastructure the program needs (harnesses, trainers, ports, runtime
  machinery). Not falsifiable; *completable*. Pre-registered acceptance
  criteria and non-goals instead of hypotheses. Verdicts: `done` /
  `killed-by-judgment` (reason recorded — a killed capability is a
  contribution, like a null result).
- **Measurements** — standing programs (the ledger, baselines) that live
  under a capability's umbrella. Data, not proposals.

The decision to build a capability is a human judgment call about
program needs, stated plainly in the proposal's motivation section —
never a fake hypothesis. The rigor lives in the acceptance criteria.

---

## The IETF RFC-Style Proposal Rules

1. **Slugs at Proposal, Numbers at Verification**:
   - Every proposal lives in a descriptive slug folder: `proposals/<arc>/<slug>/` (e.g. `proposals/logic-substrate/takemura-min1/`).
   - **No `eNN` identifier is minted at proposal time.** Numbers are reserved strictly for verified, peer-reviewed findings merged into `catalog/`.
2. **RFC Headers Required**:
   Every proposal `spec.md` must declare:
   - `Experiment: <slug>` or `Capability: <slug>`
   - `Type: <experiment | capability>`
   - `Arc: <arc-name>`
   - `Literature: [<citations>]`
   - `Hardware-Target: {Reference: "...", Claim-Shape: "..."}`
   - `Extends:`, `Refutes:`, `Supersedes:`, `Reopens:`
   - Capabilities additionally declare `Unlocks: [<experiment-slugs>]`
3. **Closed-Arc Policy**:
   - Arcs marked `CLOSED` in `catalog/registry.edn` require an explicit `Reopens:` header providing a novel mathematical mechanism. CI automatically rejects proposals targeting closed arcs without this header.
4. **Draft Expiry**:
   - Proposals inactive for 90 days are marked `STALE` and archived without consuming an `eNN` number.
5. **Predeclared rigor, right shape**:
   - An experiment proposal without pre-registered kill criteria is returned unread.
   - A capability proposal without checkable acceptance criteria and explicit non-goals is returned unread.
