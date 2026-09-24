# PROCESS.md — How This Lab Works

**Thesis.** Liang Wenfeng's DeepSeek playbook gives us the *process
architecture*: a staircase roadmap, lottery-ticket exploration, and
internal-first recursion. Our gates give us the *measurement architecture*:
falsifiable, hardware-grounded, scored by machines. The staircase is the
roadmap; the gates are the altimeter. Process without measures is faith;
measures without process is bureaucracy. This document defines both, and
how they connect.

> **Attribution.** Section 1 and parts of Section 2 adapt remarks
> *attributed to* Liang Wenfeng in a transcript of a May 2026 investor
> conversation (via Fred Gao, July 2026). The transcript's authenticity
> could not be verified, so treat these as attributed positions, not
> authenticated DeepSeek policy. The gates, the measurement layer, and the
> synthesis of the two are our own.

---

## 1. The staircase

Each stage builds on the previous one. We did not invent this roadmap; we
are placing ourselves on it precisely.

| Stage | Status for us | Gate signature |
|---|---|---|
| 1. Base language models | We ride open weights. We do not train foundation models. | — (supplier market) |
| 2. Chain of thought | Raises the intelligence ceiling. We use it; we don't research it. | G3: reasoning-task retention |
| 3. Agents (current stage) | Our operating point. The loop runs here. | G1–G4 all active |
| 4. Continuous learning | Unsolved globally; the field is groping. Our memory + verified-commit work is our lottery ticket for this step. | **G2 and G4 must move** on our own task distribution |
| 5. Self-iteration | The loop improving the loop. Built toward, not scheduled. | G4 derivative: verified output per unit judgment, compounding across generations |
| 6. Embodied intelligence | Explicitly out of scope. | — |

**Deliberately skipped.** Video generation, world models as the
intelligence main line, domain-specific vertical agents. These are
commercial opportunities or secondary components, not drivers of the core
ceiling. Restraint is strategy: every skipped line is budget returned to
the main line.

**The operational rule the staircase implies.** You have not "done"
continuous learning because you added a memory module. You have done it
when G2 (adaptation velocity) and G4 (recursion/judgment cost) move on
*your own task distribution*, measured the way Section 3 describes. The
staircase tells you what to attempt; the gates tell you whether the step
landed.

---

## 2. The process

### 2.1 Internal-first: "useful for ourselves"

The first customer of everything we build is our own loop. New machinery
deploys to our own R&D pipeline before it is documented, exported, or
announced. This doubles as our evaluation criterion: the question is never
"does it score well," it is "did Tuesday's experiments run cleaner." If a
frontier lab released a model tomorrow that doubled our benchmark scores
but couldn't run our loop, we'd shrug.

### 2.2 Lottery tickets

Breakthroughs in the unsolved steps come from algorithmic ideas, not from
brute-force compute. So exploration is budgeted structurally, not
begrudgingly: a standing fraction of capacity is reserved for bottom-up,
self-directed tickets with no fixed KPI beyond the gates themselves.

At our scale this inverts. DeepSeek reserves unassigned *researcher* time;
we reserve unassigned *agent* time — background exploration threads that
cost inference, not salary. Section 3.1 explains why the gates make this
affordable.

### 2.3 Coding as the arena

The concrete environment for progress is the coding agent: multi-step
reasoning, tool use, verification against a compiler and a test suite.
Domain verticals (finance, healthcare, etc.) are skipped. Our action
language is Clojure; our proving ground is this repository.

### 2.4 One piece, done well

We do not vertically integrate. We don't train base models, don't build
chips, don't ship end-user applications. Our one piece is **the loop**:
memory, gates, harness. Base models are converging to a handful of
suppliers; we ride their wave instead of racing it. When a better open
model drops, we swap the engine and keep the vehicle.

### 2.5 Continuity as the #1 asset

DeepSeek's stated non-negotiable is team stability — the compounding asset
is the people who stay. Ours is **loop-state durability**: the verified
store, the catalog, the gates, the lineage. A failed experiment costs an
afternoon; a lost memory store resets the program. Durability of loop
state gets investment priority over any single experiment.

---

## 3. The measures: gates as the altimeter

The four gates live in `resources/catalog/` (`gate1_compression`, `gate2_velocity`,
`gate3_evals`, `gate4_recursion`, with per-entry lineage in
`registry.edn`):

- **G1 — Compression.** Real memory savings in bytes per parameter, on
  real hardware. Split into storage (bytes/param) and throughput; a
  compression win that tanks matmul throughput is two facts, not one.
- **G2 — Adaptation velocity.** How fast knowledge updates *without
  retraining*: calibration minutes, memory-write latency, time from
  observation to committed, queryable knowledge.
- **G3 — Evals.** Capability retention on our own task distribution, with
  a floor we refuse to go under — plus the absolute capability trend, so
  a rising floor can't hide a sinking ceiling.
- **G4 — Recursion.** The loop improving the loop. Derivative metric:
  **verified output per unit of human judgment** (judgment cost per
  accepted catalog entry, repeated-error rate, evidence-use accuracy).

### 3.1 Who scores the lottery tickets

DeepSeek can run exploration "without rigid KPIs" because hundreds of
researchers supply taste as the scorer. Taste does not scale down to
autonomous agents — and that is exactly the gap the gates fill. **The
gates are the KPIs that let agents draw *and score* lottery tickets
without a human tasting each one.**

This inverts exploration economics at small scale: a ticket costs
inference plus a gate run, not a researcher's week. We can draw a hundred
tickets where a human lab draws ten — *if* the scoring is honest. The
entire program therefore rests on gate integrity: matched baselines,
grounded numbers, pre-registered criteria, null results recorded as
contributions. A corrupt altimeter is worse than none.

### 3.2 Compare at equal cost

Compare models at the same cost, the way you'd compare cars in the same
price range. Our standing rule: matched baselines, honest accounting,
end-to-end task metrics — never per-op vanity numbers, never a baseline
you wouldn't defend. (This is the E24 baseline-fairness rule, kept.)

---

## 4. Memory as mechanical onboarding

Liang's image for continuous learning is a human employee absorbing company
context over two months instead of needing every background detail
re-prompted. That is our continuity problem stated from the other side:
**the agent that needs everything re-prompted every session is the
pre-continuous-learning agent.**

Our mechanical version of onboarding:

- **Facts in the database, skills in the weights.** The model reasons;
  the store remembers. When the store is wrong, fix the store — don't
  retrain the model.
- **Write discipline.** Nothing enters the permanent store without a
  verified commit: schema check, provenance, contradiction quarantine,
  reviewer disposition. Unverified output is exhaust; verified output is
  memory.
- **G2/G4 verify the onboarding took.** Onboarding isn't "we added a
  memory module" — it's adaptation velocity and judgment cost moving on
  our distribution.

**The write path is the battleground.** Poisoned memory is the attack that
matters, not prompt injection. If anything can write to your store,
anything can make you stupid over time. Guard writes like your
intelligence depends on it, because it does.

---

## 5. Operating rules

1. **Judgment is the scarcest input.** Compute gets cheaper, context gets
   longer, inference gets faster — deciding what's true doesn't. Meter
   judgment cost per verified entry. The loop's job is to drive verified
   output per unit of judgment as high as it will go.
2. **Evidence discipline.** Ground every number. Mark interpretation as
   interpretation. Distinguish measured from projected, on-device from
   reference-interpreter, committed from claimed. Modest numbers reported
   as modest.
3. **Kill criteria in advance.** Every exploratory line gets its
   falsification condition *before* it starts. A line that can't state
   what would kill it doesn't start.
4. **Null results are contributions.** A closed line with committed
   evidence is worth more than an open line with vibes.
5. **Open by default.** Apache 2.0, arXiv-first, public catalog. You
   can't own the tide, and openness doesn't commoditize operational
   excellence — the edge is the loop, not any single weight file.
6. **Byproducts, not goals.** Aim at the loop; papers, checkpoints, and
   tools arrive as byproducts. Don't optimize for the byproducts.
7. **Maximize mechanical sympathy.** Run where it runs best, even
   host-side. The 24GB constraint is the strategy, not just a limit —
   being compute-poor forces the clever methods that become the advantage.
8. **Sustainable pace.** The loop compounds without fatigue; humans don't
   have to. Unattended execution is normal; heroics are a process smell.

---

## 6. How work flows here

This document states the *why*. The *how* lives where it's enforced:

- `resources/proposals/` — RFC lifecycle. Tracked by descriptive `<slug>`
  across parallel exploration tracks (`TEMPLATE.md`). A proposal without
  pre-registered kill criteria is returned unread.
- `resources/catalog/` — the four gate directories (`<gate>/<slug>/`) plus
  `registry.edn`: lineage per entry (bytes/param, calibration minutes, eval
  scores, hardware, judgment cost). Score the snapshot, not the model.
- `docs/architecture/agent_loop.md` — the loop design: tiers, the
  committed store, the contraction reflex.
- `CONTRIBUTING.md` — the contributor contract (coding agents as primary
  contributors; spec → implement → test → measure → catalog).

New work enters through a proposal, exits through a gate, and compounds
through the catalog. That is the whole machine.
