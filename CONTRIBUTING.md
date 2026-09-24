# CONTRIBUTING.md

> **Start with [`PROCESS.md`](PROCESS.md).** It is the definitive guide for our Staircase roadmap, the 4 RSI gates, and our 8 operating rules.

Coding agents are the primary contributors to this repository. All research follows a **3-Stage Direct-Commit Protocol** sharded by RSI Gate and identified by descriptive `<slug>`s. There are no Pull Requests and no artificial sequential `eNN` numbers—parallel exploration tracks commit directly to `main`.

---

## The 3-Stage Research Protocol (Direct-Commit Model)

```
[Stage 1: Proposal Spec]     ──>  [Stage 2: Implementation]    ──>  [Stage 3: Independent Verification]
(resources/proposals/<gate>/<slug>/)  (src/experiments/ & results.edn)     (Independent Replication on Silicon)
          │                                      │                                      │
   Pre-Registered Spec                   Committed as UNVERIFIED               Cataloged -> resources/catalog/
```

### Stage 1: Proposal Spec
- Commit RFC spec to `resources/proposals/<gate>/<slug>/spec.md` (use [`TEMPLATE.md`](resources/proposals/TEMPLATE.md)).
- Must pre-register falsification criteria across the 4 gates before writing code (see `PROCESS.md` §3).
- Commit format: `[<gate>/<slug>] stage1-spec: <brief description>`.

### Stage 2: Implementation & Empirical Measurement
- Commit runnable code under `src/experiments/<gate>/<slug>/` (namespace: `experiments.<gate>.<slug>...`).
- Commit raw machine-readable metrics to `resources/proposals/<gate>/<slug>/results.edn` and `summary.csv`.
- Status in catalog registry: marked `:unverified`.
- Commit format: `[<gate>/<slug>] stage2-impl: <brief description>`.

### Stage 3: Independent Silicon Verification & Cataloging
- An independent contributor (or peer agent) replicates the claims on independent silicon.
- Verifies claim shape (e.g., storage reduction achieved, retention floor $\ge 95\%$ held, throughput verified).
- Move non-code pod to `resources/catalog/<gate>/<slug>/` and register under `:slug` in [`resources/catalog/registry.edn`](resources/catalog/registry.edn).
- Commit format: `[<gate>/<slug>] stage3-verify: <replicate claim and catalog pod>`.

---

## Promotion to the Core Engine (`src/einsum/`)

Code in experiment pods remains isolated. It is promoted into `src/einsum/` only under **Mechanical-Sympathy Placement**:
- **Dense Contractions & Autoregressive Decoding**: Placed 100% in OpenXLA StableHLO MLIR in accelerator VRAM. Zero host round-trips.
- **Discrete Lookups & Schema Verification**: Placed on CPU host memory in pure Clojure persistent data structures (host discrete lookups have a $16.4\times$ speed advantage over device VRAM lookups).

---

## Pre-Commit Verification Invariants

Before every commit, agents must run and pass:
```bash
clojure -M:format
clojure -M:lint
clojure -M:test fast
```
Zero errors and zero linter warnings allowed.
