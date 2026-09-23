# Proposals Directory (Stage 1 & 2 Drafts)

This directory houses ephemeral proposal drafts (Stage 1) and active unverified implementation scaffolds (Stage 2).

---

## The IETF RFC-Style Proposal Rules

1. **Slugs at Proposal, Numbers at Verification**:
   - Every proposal lives in a descriptive slug folder: `proposals/<arc>/<slug>/` (e.g. `proposals/logic-substrate/takemura-min1/`).
   - **No `eNN` identifier is minted at proposal time.** Numbers are reserved strictly for verified, peer-reviewed findings merged into `catalog/`.
2. **RFC Headers Required**:
   Every proposal `spec.md` must declare:
   - `Experiment: <slug>`
   - `Arc: <arc-name>`
   - `Literature: [<citations>]`
   - `Hardware-Target: {Reference: "...", Claim-Shape: "..."}`
   - `Extends:`, `Refutes:`, `Supersedes:`, `Reopens:`
3. **Closed-Arc Policy**:
   - Arcs marked `CLOSED` in `catalog/registry.edn` require an explicit `Reopens:` header providing a novel mathematical mechanism. CI automatically rejects proposals targeting closed arcs without this header.
4. **Draft Expiry**:
   - Proposals inactive for 90 days are marked `STALE` and archived without consuming an `eNN` number.
