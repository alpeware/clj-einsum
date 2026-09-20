# CONTRIBUTING.md

**If you are a coding agent, start here. You are the primary contributor this
project is built for.**

Humans are welcome too — the same bar applies to everyone, and human judgment
is still the backstop where the loop can't reach yet (design review, the
open composable-evaluation problem). But if a task can be specified precisely
enough for an agent, it should be done by an agent. That is the point of this
repo: programs are data, verification is automatic, and the E-series
experiments that define this project were designed, implemented, measured, and
written up by coding agents.

## Before you write code

Read these first — they are short, and they will save you from re-running
someone else's failed experiment:

1. **[VISION.md](VISION.md)** — what the representation covers and what it
   doesn't. The covers/doesn't-cover list is a living contract. If your change
   touches a boundary, update the boundary in writing.
2. **[AGENTS.md](AGENTS.md)** — the repo's coding rules (TDD, pure functions,
   clean CI, pure-XLA execution). Non-negotiable.
3. **[docs/tensor_logic/empirical_journey.md](docs/tensor_logic/empirical_journey.md)** —
   the catalog of everything tried, including the failures. Check whether your
   idea is a closed line before you spend GPU time on it. E16–E18 (contrastive
   resolver) and E19–E20 (gradient predicate invention) are closed: gradients
   learn rankings, not discrete truth. Do not re-litigate them without a new
   mechanism and a written reason the old verdict doesn't apply.
4. **[docs/architecture/agent_loop.md](docs/architecture/agent_loop.md)** — if your change touches the agent loop,
   the KB write path, or the reduce.

## The 3-Stage Research Protocol (IETF RFC Model)

To scale decentralized research across coding agents and human contributors without claim collisions or abandoned gaps, contributions follow the 3-stage lifecycle:

```
[Stage 1: Proposal PR]  ──>  [Stage 2: Implementation PR]  ──>  [Stage 3: Verification PR]
 (proposals/<arc>/<slug>/)        (Scaffold & results.edn)           (Independent Replication)
          │                                  │                                   │
   Approved Spec                      Merged as UNVERIFIED               eNN Minted -> catalog/
```

### Stage 1: Proposal PR (Descriptive Slugs, No Numbers)
- Propose new experiments under `proposals/<arc>/<slug>/spec.md` (e.g. `proposals/logic-substrate/takemura-min1/spec.md`).
- **No `eNN` numbers are assigned at proposal time.** Two agents proposing ideas concurrently cannot collide on a number. Slug collisions surface loudly as git merge conflicts.
- **RFC Metadata Headers Required**:
  - `Experiment: <slug>`
  - `Arc: <arc-name>` (must match an active arc in `catalog/registry.edn`)
  - `Literature: [<formal paper citations in the DAG>]`
  - `Hardware-Target: {Reference: "...", Claim-Shape: "..."}`
  - `Extends:`, `Refutes:`, `Supersedes:`, `Reopens:`
- **The Split Novelty Gate on Closed Arcs**:
  - *Mechanical CI Gate*: If targeting an arc marked `CLOSED` in `catalog/registry.edn`, CI automatically fails if the `Reopens:` header is missing.
  - *Human Maintainer Gate*: The reviewer evaluates whether the rationale presents a genuinely novel mathematical mechanism (rather than re-tuning hyperparameters on a closed line).

### Stage 2: Implementation & Claim PR
- Contributor merges scaffolding code, tests, and raw output metrics (`results.edn`) into `proposals/<arc>/<slug>/`.
- **Status in Catalog**: Marked **`ACTIVE (UNVERIFIED)`**.
- **Taint Propagation**: Any downstream experiment declaring this unverified pod as a dependency inherits the `UNVERIFIED` status until the parent is verified.
- **Draft Expiry**: Proposals inactive for 90 days are marked `STALE` and archived without consuming an `eNN` number.

### Stage 3: Independent Verification PR (Minting the `eNN` ID)
- An independent contributor (human or peer agent) replicates the claim on independent silicon.
- **Hardware-Relative Claim-Shape**: Verification tests do not need identical milliseconds on different GPUs. They verify **claim shape**: e.g., was prefill eliminated ($\ge 50\text{ ms}$ saved)? Was handover negligible ($< 2\text{ ms}$)? Did semantic parity hold ($\ge 90\%$)?
- **Number Minting**: **Only upon merging Stage 3 does the repository merger permanently mint the next sequential `eNN` ID.** The pod moves to `catalog/<arc>/eNN-<slug>/` and is recorded in `catalog/registry.edn`.
- *Verification is an independent, credited contribution.* Both the proposer and the verifier receive provenance in the catalog.

## Promotion to the Core Library (`src/`)

Code in experiment pods remains isolated. It is promoted into the core library only under **Mechanical-Sympathy Placement**:
- **Dense Contractions & Autoregressive Decoding**: Placed 100% in OpenXLA StableHLO MLIR on accelerator device memory (GPU VRAM). Zero host round-trips.
- **Discrete Index Lookups, Tries & Schema Validation**: Placed on CPU host memory in pure Clojure persistent data structures. The canonical justification is E22: host discrete lookups achieved $42.20\ \mu\text{s}$ vs $691.94\ \mu\text{s}$ on device ($16.4\times$ host advantage).
- **Core Stays Lean**: The canonical Vaswani (2017) transformer baseline lives in `src/`. Large vehicle architectures (Gemma 4, SmolLM, GPT-2) reside in top-level `models/` on the `:experiments` classpath.

## Testing & CI Invariants

- `clojure -M:format`, `clojure -M:lint`, and `clojure -M:test -m einsum.test-runner` must all pass with zero warnings or errors.
- New behaviors require generative property tests (`clojure.test.check`).
- Staged diffs must be narrow and single-purpose.
- All numbers in reports must be grounded to physical hardware. Modest numbers reported as modest. No ungrounded claims.

## Ground rules

- **Small, single-purpose commits** with conventional-commit messages:
  `feat(logic): …`, `fix(compiler): …`, `docs(e24): …`, `test(kb): …`.
  One commit does one thing.
- **Never commit binaries, model weights, or secrets.** Native libraries
  (e.g. the 259MB `libpjrt_cpu.so`), `.models/`, credentials, and API keys
  stay out of the tree. If a result depends on a local artifact, document how
  to reproduce it instead.
- **Don't move the goalposts in the writeup.** If the measurement disagrees
  with the spec's prediction, the spec was wrong — say so, in the journey.
  Doc-vs-data contradictions are the fastest way to lose trust here.
- **Respect closed lines, or reopen them properly.** A closed line needs a new
  mechanism and a written argument, not a re-run with different
  hyperparameters.
- **Keep the hacks explicit.** The 80/20 rule is fine — stride fixes, padding,
  host orchestration — but hacks get measured and labeled, not hidden.

## For human contributors

Welcome. The same loop and the same bar apply. A few notes:

- Review here is often done by agents, including the adversarial kind. A
  review that asks for a fairer baseline or a decomposed metric is not
  hostility — it's the project's immune system. E24 survived it; your change
  will be better for it.
- If you're unsure whether an idea is a closed line, ask before building.
  Open an issue with the mechanism and why the old verdict doesn't apply.
- Human time is most valuable at the judgments the loop can't make yet:
  experimental design, what counts as a fair comparison, and whether a result
  is actually interesting. Spend it there.

## License

No contributor license agreement. Inbound = outbound: your contribution is
accepted under the same [Apache License 2.0](LICENSE.md) as the rest of the
project. Don't submit anything you don't have the right to license that way.

## Getting help

Open an issue. Describe the mechanism, the expected measurement, and what
would falsify it. If it's a closed line, say why this time is different.
