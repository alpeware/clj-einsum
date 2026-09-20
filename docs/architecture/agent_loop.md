# AGENT-LOOP.md — Tensor-native agent loop: compiling mealy cells to contraction graphs

**Status:** design doc for discussion, 2026-09-15. Not implemented.
**Revision 2026-09-15:** §9 rewritten as the three-tier design (reflex /
deliberation / synthesis) — Tier 2 device-resident LLM deliberation with
uncertainty routing, staged A (two executables, scalar dispatch) then B
(in-graph while); phased plan extended to 0–8.
**Revision 2026-09-15 (2):** folded in the in-graph lexical interrupt spec
(token-ID compare → stop flag, replacing host-side code-block regex
extraction) as the concrete Stage B mechanism, and the T=0 precondition gate
as the embedding-space formulation of the frozen mask — with the soundness
caveat (margins measured on the symbolic construction, not on LLM-generated
candidates) and the `{:act :step}` staleness note.

- *Implementation status (2026-09-15).* The gate exists in code:
  `einsum.logic.memory.relation/relational-grounding-ast` (commits 26982ae,
  fc10dfd) implements probe → project → contract-against-relation-core →
  T=0 compare → mask → vocab-bias-fused-into-logits, and
  `tools/poc_fact_grounding.clj` wires it into a full 35-layer Gemma 4 E2B
  forward pass on ROCm and CPU. Two distinctions from the design above:
  (1) it is implemented as logit *biasing* (influence), not post-LLM action
  *masking* (enforcement) — the safety guarantee is weaker than §9's design,
  and hard masking remains future work; (2) fc10dfd fixed the gate being
  computed but never consumed (silently pruned by DCE) — the cautionary tale
  for testing gate *effects*, not gate *presence*. Open: projection
  provenance (`w_mem_proj`, `w_entity_to_vocab`), margin measurement through
  the full grounding chain, per-token host-side buffer rebuild cost.
**Companion:** `SYMBOLIC.md` (the `einsum.logic.symbolic` design this builds on).
**Audience:** whoever implements the mealy × clj-einsum integration — the "tensor-logic
reflex vs LLM path" demo from the stack thesis.

## 1. Goal

Compile a mealy cell's event loop into a single tensor-logic contraction graph that
executes on the accelerator (StableHLO → PJRT via clj-einsum), with the host doing
nothing per step but feeding in event indices and reading out action indices.

Non-goals:

- Replacing mealy's Clojure reducer. It becomes the **oracle** for property tests
  and the fallback backend. Same surface semantics, new execution backend.
- Tensorizing the LLM, I/O, the bus, SCI bootstrap, or cryptography. Those stay
  host-side (§11).
- A neural policy network. The policy is a *relation tensor* with tensor-logic
  semantics, not an MLP. Decisions are contractions with proven soundness margins,
  not black-box activations.

## 2. Background: what the loop actually is

Grounded in the mealy tree as reviewed 2026-09-15 (`src/mealy/cell/`). The doc
designs for this architecture, not a textbook Mealy machine.

- **Event-sourced reducer.** `handle-event` is a multimethod: `state × event →
  {:state :actions}`, dispatching on the event-type keyword. Cell state is a map
  with `:aim :memory :observations :policies :phase :bus-topics :parent`
  (plus a transient `:sci-ctx`).
- **Frozen kernel vs extensible handlers.** `frozen-dispatch-values` =
  `#{:default :observation :evaluation-error :tick}`. Bootstrap/SCI code may add
  handlers but can never override these. This is a safety invariant, and the
  tensor design preserves it structurally (§5).
- **OODA pipeline.** The bootstrap script implements observe → orient → propose →
  policy-change → tap → act stages (per the reducer docstring). The decide path
  already routes to an LLM (`llm/route-llm-request`).
- **Mitosis.** `threshold-reached?` watches `:max-memory-keys` / `:max-observations`;
  `divide` partitions a parent's memory into a child genesis state; the child
  inherits the parent's forked SCI context and policies.
- **LLM slow path already exists.** Code generation, repair loops, prompt
  compilation — all host-side today.

## 3. Core thesis

`(s', actions) = step(s, event)` is expressible as a staged contraction program:

1. **Observe** — bind the event to embeddings; accumulate into an observation
   superposition tensor.
2. **Orient** — contract memory tensors against the observation; produce context.
3. **Decide** — contract the policy/transition tensors against
   (phase, event-type, context); produce action scores.
4. **Act** — argmax / tempered sample; emit action indices.

Every stage is a contraction over near-orthonormal random embeddings — the exact
machinery `einsum.logic.symbolic` already proves sound (fact margin 0.88, rule
margin 0.89 at D=512, PJRT CPU). The whole step lowers to **one StableHLO graph**
per cell type. Populations of cells become a **batch dimension**: advancing 1024
agents is one batched execution, not 1024 loop iterations.

The LLM is not exiled to the host — it occupies the middle tier (§9): Tier 1
reflex (contractions, verified), Tier 2 deliberation (device-resident LLM
inference, uncertainty-routed, masked), Tier 3 synthesis (host-side rule
invention, parity-gated). The honest axis is latency tier plus verifiability,
not location.

## 4. The Cell IR

New namespace `einsum.agent.ir`. The IR is EDN, serializable, and deliberately
boring — it describes *which* contractions run, not how.

```clojure
{:cell/id :courier
 :symbols
 {:phase  [:idle :orienting :deciding :acting :generating-code]
  :event  [:observation :tick :heartbeat :proposal :policy-change :tap :default]
  :action [:inject-event :emit-signal :route-llm :divide :noop]}
 :entity-schema
 {;; Per event-type: which payload fields entity-ize into the tensor path.
  ;; Rich EDN payloads stay host-side; only their type signature crosses.
  :observation {:payload [:topic :summary-entity]}
  :tick        {:payload []}}
 :relations
 {;; Transition and output: third-order, usually sparse.
  :trans {:arity [:phase :event :phase]  :storage :sparse-coords}
  :emit  {:arity [:phase :event :action] :storage :sparse-coords}
  ;; Plastic memory: superposition tensors, Hebbian writes, decay.
  :episodic {:arity [:entity :rel :entity] :plastic true :decay 0.99}
  :semantic {:arity [:entity :rel :entity] :plastic true :decay 0.999}
  :policy   {:arity [:phase :event :action] :plastic true :decay 1.0}}
 :frozen #{:observation :tick :evaluation-error :default}
 :masks
 {;; Structural constraints, not weights: compile-time fixed.
  :action-allow {:from :phase :to :action}}}
```

Semantics:

- `:trans` / `:emit` encode the handler table. Today's multimethod dispatch
  (`handle-event` on event-type) becomes: score handlers by contraction, take
  argmax. For the frozen set this is exact by construction.
- `:phase` is today's `:phase` keyword, as a state symbol. `:observations`
  becomes accumulation into `:episodic` with per-event decay — an append-only
  superposition, not a Clojure vector.
- `:policies` (mealy's boundary strings) become rows in `:policy` / `:semantic`:
  policy statements entity-ized once, then matched by contraction at decide time.
- `:aim` becomes a fixed context embedding concatenated into every orient query
  (a constant row — zero per-step cost).
- **Frozen handlers are structural.** `:observation` compiles to a fixed
  accumulate op; `:tick` to fixed time-slot writes. There is no weight tensor for
  bootstrap code to perturb, so the frozen-kernel safety property becomes a
  *compile-time invariant*: the graph has no plastic parameters on those paths.
  This is strictly stronger than today's multimethod guard.

The Clojure DSL for defining cells keeps mealy's surface semantics (a cell author
writes handlers); a new `einsum.agent.frontend` macro lowers handler definitions
to this IR. Authors don't write contractions by hand.

## 5. Compilation pipeline

```
cell-def (Clojure DSL)
  → Cell IR (EDN, §4)
  → tensor-logic AST  (einsum.logic.symbolic programs; reuse [: = ] equations,
                       superposition-relation-matrix, correlate-embeddings)
  → EDN SSA IR → StableHLO → PJRT executable (existing pipeline, untouched)
```

**Dense vs sparse backend selection** (novel, §8.2). The compiler runs a
cardinality analysis on each relation:

- *Dense path*: small closed relations (phases × events × actions is tiny).
  Materialize the embedded tensor; the step is pure matmuls — XLA's favorite
  shape. A 512³ contraction ≈ 0.27 GFLOPs, microseconds on a GPU.
- *Sparse path*: open relations (episodic memory over thousands of entities).
  Keep coordinate lists; the contraction is a gather over matching coordinates,
  O(k·D) for k matches. This is §3.2's sparse-coordinate accumulation run in
  reverse (gather instead of scatter).

One semantics, two backends, chosen per relation at compile time. The existing
`forward-chain` machinery already evaluates rule-shaped contractions; the agent
compiler reuses it for the dense path and adds the gather evaluator for the
sparse path.

**House-rule compliance** (repo `AGENTS.md`, read 2026-09-15). Rule 4 (pure XLA
execution, no Java escape hatches): both backends are compiled StableHLO. The
sparse path's "gather" is an in-graph `stablehlo.gather` over device-resident
coordinate tensors, and coordinate appends are in-graph scatters — never
host-side float loops. The coordinate list is a device tensor with a static
`max-coords` bound (the §3.2 precedent). Rule 2 (Sans-IO): the executable is a
pure function of `(indices-in, plastic-state)` → `(indices-out,
plastic-state')`; all effects live in the host shell (§11).

**Compile once, step many times** (the repo's agent-VRAM-session pattern): the
cell executable is compiled once with static shapes — batch padded to `max-B`,
coordinate lists padded to `max-coords` — and plastic tensors stay resident in
device memory across steps. Per step the host transfers only the input index
vectors and reads back the output index vectors, never recompiling, never
reloading weights.

**One graph per cell type; batch dim per population.** The compiled executable
takes index vectors `phase[B]`, `event[B]` and returns `phase'[B]`,
`action[B]`. B=1 is a single cell; B=1024 is a population. Per-cell plastic
tensors gain the leading batch dim (`episodic[B, ...]`); shared tensors
(`:trans`, `:emit`, `:semantic`) don't.

## 6. The OODA loop as a staged contraction graph

Per step, with `e(x)` = embedding row gather, all inside the single graph:

1. **Observe.** `o = e(event-type) ⊗ e(payload-entities...)`; 
   `episodic[B] += o` (sparse accumulate, frozen structural op for
   `:observation`; `:tick` writes the time slot instead).
2. **Orient.** `ctx = contract(semantic, episodic[B], aim)` — retrieve memory
   relevant to the current observation. Dense or sparse per §5.
3. **Decide.** `scores[a] = Σ contract(policy[B], e(phase), e(event), ctx)[a]`,
   then apply the `:action-allow` structural mask (frozen — disallowed actions
   score −∞ regardless of weights).
4. **Act.** `action = argmax(scores)` or tempered sample (XLA RNG) for
   exploration. Emit index; host maps index → action payload.

Phase transition runs alongside: `phase' = argmax(contract(trans, e(phase),
e(event)))`. The whole step is ~4 contractions; at D=512 sparse-dominated,
per-agent cost is a few × k·D gathers plus small dense matmuls.

## 7. Hierarchy and mitosis

**Dispatch tensor** (novel, §8.3). A parent cell's `:emit` includes child-select
actions. Two compiled modes from the same IR:

- *Discrete dispatch* (deploy): parent emits child index `c`; the host routes the
  next event to child cell `c`'s executable. One index crosses the boundary —
  negligible cost, branch stays host-side.
- *Soft dispatch* (train): `π_total = Σ_c w_c · π_c`, weights from the parent's
  mode scores. Fully tensor-native, differentiable through `w`, pays for all
  children per step.

**Mitosis** becomes batch-dim management. `threshold-reached?` compiles to a
cheap on-device predicate (memory-tensor norm / observation count vs
thresholds). `divide` compiles to: allocate new batch index `B+1`; copy the
parent's `:policy` rows and entity bindings (inheritance = row copy, mirroring
today's SCI-context fork + policy inheritance); partition `:episodic[B]` rows by
the partition-key mask. No host-side state surgery.

## 8. Plasticity: learning without leaving the accelerator

All as compiled contractions over the plastic relations (§4):

- **Hebbian write** (reward-modulated): `R += η · r · e(x)⊗e(rel)⊗e(y)`.
  Sparse coordinate append — O(D) per write, no dense materialization.
- **Decay**: `R *= λ` per epoch (fused scalar multiply; episodic 0.99,
  semantic 0.999, policy 1.0).
- **Eligibility traces** (research, Phase 6): `Z += e(s)⊗e(a); Z *= γλ` per step;
  on reward, `policy += η · r · Z`. A trace tensor is just another decaying
  superposition — multi-step credit without leaving tensor-land.

Capacity is the open empirical question: the margin-vs-D sweep measured *static*
KBs (margins 0.75→0.94, variance collapsing with D). Online Hebbian writes need
their own margin experiment (§10, Phase 2) before being trusted.

*Honest scope note (E17–E20).* Superpositional Hebbian writes are working
memory, not long-horizon memory. E17–E20 showed continuous/soft
representations cannot maintain discrete relational truth — the continuous
optimum is adversarial to the discrete one (E19), and soft structure does
not crystallize under any tested optimizer (E17/E20). So §8's plasticity
covers traces, working context, and reward modulation — the things that
are *allowed* to be lossy. Anything the agent must still believe in
an hour goes in the Tier 1a committed KB (§9), through the verified
write path, or it isn't believed at all.

## 9. Three tiers: reflex, deliberation, synthesis

**The loop is a reduce.** Not a loop over an ever-growing history —
a reduce over a bounded snapshot:

`state' = verified_commit(state, llm_proposals(state, observation))`

The accumulator is the Tier 1a KB snapshot, never the transcript.
Each step *replaces* state; nothing appends except through the
verified commit. Compaction isn't a periodic emergency — it is the
step itself. This is the old REST-harness pattern (the accumulator
as a namespace the LLM rewrites by redefinition), hardened: the
"redefinition" goes through E21's write path (typecheck → constraint
→ ambiguity → provenance), so the snapshot is the compressed history,
*verifiably*. The append-only provenance log is the audit trail; the
active KB is the snapshot the loop reduces over. Both exist; they are
different things.

Consequences, stated once so the tiers below don't re-litigate them:

- *Bounded compute per step, forever.* Each step's prompt is the
  snapshot (bounded) plus the current observation — never the
  history. Per-step cost is O(snapshot + step), with the static
  prefix (system, schema) warm across steps. The KV cache has a
  ceiling. Infinite-horizon operation is well-defined; there is
  nothing to "summarize" because nothing accumulates outside the
  commit path. (E17–E20 are why the snapshot must be discrete: a
  soft accumulator drifts, and drift in discrete truth is lying.)
- *One pattern at two latencies.* This reduce is a mealy cell:
  `handle-event(kb_snapshot, observation) → kb_snapshot'`. Tier 1
  runs it in microseconds over contraction state; Tier 2 runs it in
  seconds over KB state. There are not two architectures here.
- *Stage B shape.* The reduce body (decode step → optional KB query
  → verified commit) is a StableHLO while body, with the KB snapshot
  and KV cache as loop-carried state — buildable from the while/cond
  primitives the repo already has. The per-query surface wants to
  be small: named queries (invars + AST, ~6 lines) compiled to
  executables, not 40-line compile ceremonies.

The original draft of this section put the whole LLM host-side. That conflated
two claims: "LLM outputs are slow and unverified" (true) with "the LLM must
live off-device" (false in this stack — clj-einsum compiles full LLM forward
passes to StableHLO/PJRT, with weights pinned in VRAM and static-shape graphs).
The honest axis is latency tier plus verifiability, not location. Intelligence
sits in the middle tier of the execution architecture.

**Tier 1 — reflex (device, verified).** Two sub-tiers, split by a lesson
the E17–E20 experiment series forced:

- *Tier 1a — the committed KB.* A schema-constrained Datalog store,
  discrete and committed: relations with declared types, cardinality
  constraints (e.g. every child has exactly 2 parents), identity
  constraints (e.g. siblings share *identical* parent-pairs, not merely
  overlapping ones), and denial constraints (e.g. acyclicity). Derived
  relations materialize host-side; the hot fact set lives in VRAM for
  microsecond query answering (ground queries, bounded joins — boolean
  matmuls and gathers, the same machinery as the E19 diagnostics).
  The LLM *consults* the KB but never *owns* it.
- *Tier 1b — the contraction step* from §§4–6 (unchanged): microseconds
  to milliseconds, fully tested semantics. The default action path every
  step.

Why the KB must be discrete and committed rather than emergent: E17
showed gradient descent learns soft associations but cannot crystallize
discrete predicates (Parent F1 0.03 vs 0.80 target); E19 showed the
continuous optimum is *adversarial* to the discrete truth (true edges
ranked 2600/4032 by GD logits — worse than random) and that
unconstrained search finds spurious covers; E20 showed even the true
structural prior fails when the prior is weak (overlap slack: S-pairs
need only intersecting, not identical, parent-pairs) and the search is
local. Jointly: long-horizon truth cannot live in weights, activations,
or learned proposals. It lives in a schema-checked store, written only
through the verified commit path (§11). The schema *is* the prior that
E20 was missing.

*Ambiguity is first-class.* E20's P0b proved some truths are unknowable
from the agent's observations (the mating ambiguity: two completions
explain every observed pair; raw-F1 ceiling 0.70, derived and confirmed).
The KB stores such disjunctions explicitly and answers `:ambiguous`
rather than committing to a guess. A KB that can't say "I don't know
which" launders guesses into facts — hallucination with a commit bit.

**Tier 2 — deliberation (device, masked).** Uncertainty-routed LLM inference,
sharing device memory with the reflex graph:

- *Router.* The decide step already produces action scores. The uncertainty
  signal is tensor-native: `margin = top1_score − top2_score`. If `margin < τ`,
  deliberate. τ starts as a constant, graduates to a learned scalar.
- *Verbalization bridge.* The LLM needs tokens, not entity embeddings. Phase 1
  path: a device-resident int table mapping entities → token ids, built once
  host-side. The prompt is a stable prefix (aim + policy rows + system) with
  the verbalized situation as suffix — so the KV cache for the prefix stays
  warm across steps (the repo's dynamic-slicing pattern). Upgrade path: a
  soft-prompt projection from entity space into token-embedding space
  (research; the table works now).
- *Bounded loop.* `while (margin < τ ∧ iters < K)`: run the decode subgraph,
  max T tokens per iteration. Static shapes throughout, so worst-case step
  latency is known at compile time. K and T stay small — deliberation, not
  essay writing.
- *In-graph lexical interrupt (the concrete Stage B mechanism).* Instead of
  generating text and regex-extracting code blocks host-side (the current
  `tools/gemma4_agent.clj` pattern), the while condition carries a stop
  flag flipped by in-graph token-ID comparisons:
  `Cond(step, tok, stopped) = (step < max-steps) ∧ ¬stopped`, with
  `stopped ← (tok = tool-call-token-id) ∨ (tok = turn-end-token-id)`.
  The KV cache and token history ride as loop-carried state (static shapes,
  dynamic slice/update — the repo's agent-VRAM-session pattern, moved inside
  the while). Note the honest split: trigger *detection* moves in-graph, but
  argument *parsing* (token slice → s-expr → SCI eval) stays host-side by
  necessity — SCI execution is an effect, and effects live at the boundary.
- *KB query arm (in-graph) vs yield-to-host.* One interrupt arm is
  reserved for the Tier 1a KB: when the decoded token sequence matches
  the KB-query protocol, the query executes against the VRAM-resident
  fact set *inside the graph* — gather + bounded join, no host
  round-trip — and the answer tokens feed back into the decode loop as
  an observation. Every other tool call yields to the host: the graph
  pauses, DMAs the request out, and resumes when the host DMAs the
  result back. The KV cache stays resident across the yield — without
  that, every tool call pays a re-prefill and the round-trip saved on
  KB queries is lost tenfold elsewhere. So the build order is:
  pause/resume with hot KV cache *first*, KB query arm second.
  A KB *miss* (no matching facts in the VRAM hot set) is itself a
  yield-to-host: the host consults the cold store and optionally
  promotes facts to VRAM. The miss path doubles as the paging
  mechanism — hot working set dense in VRAM, full KB on host.
  (E18/E19 say why this interface is shaped this way: the LLM is a
  proposer/ranker over candidates — E17 MRR +98% is real ranking
  ability — and the KB is the disposer. The tool call is the
  propose→dispose boundary made concrete.)
- *KB write arm (propose, never direct-write).* The reduce needs the
  other direction too: the LLM proposes *new facts* (tool calls
  emitting candidate assertions), and the E21 verified commit path
  applies or rejects them — typecheck → constraint → ambiguity →
  provenance — before the snapshot updates. The LLM never holds a
  raw write handle; `state' = verified_commit(state, proposals)` is
  the whole update function. Rejected proposals feed back as
  observations (with the named violation), so the model can revise
  rather than silently diverge. This is the online-learning half of
  the original sketch ("update the KB from the host side after tool
  calls"): the KB accumulates verified facts across steps, and the
  next step's snapshot is richer. Nothing the model "learned" ever
  bypasses the gate — E17–E20 proved the gate is the memory.
- *T=0 precondition gate (embedding-space formulation of the frozen mask).*
  LLM output token ids map through a device-resident action vocabulary, and
  permissibility is a crisp contraction before host dispatch:

  `Permitted[a] = step(Σᵢⱼ A[a,i] · M_pre[i,j] · S[j] − θ)`

  where A holds candidate-action embeddings, S the superposed environmental
  state, and M_pre the precondition relation. Invalid actions are masked to
  −∞ before the interrupt can fire, forcing regeneration or an alternate
  trajectory. Soundness caveat, stated plainly: the measured 0.88 margins are
  for the paper's symbolic construction with near-orthonormal embeddings —
  not for LLM-generated candidates over superposed tool observations. Until
  the margin-vs-D work is repeated on *that* distribution, this gate is a
  heuristic filter, not a verified invariant. (Implementation note: `{:act
  :step}` no longer lowers on the current tree — 39ab527 removed the alias
  per review feedback; use explicit compare+select.)
- *Staging.* Stage A (pragmatic): two PJRT executables — reflex graph and LLM
  graph — sharing device memory; the host reads back one scalar (the margin)
  and dispatches. State never leaves VRAM; the host round-trip is a float, not
  the world. Works with today's repo. Stage B (full): the while loop compiled
  in-graph via StableHLO cond/while. Attempt only after Stage A is measured —
  co-compiling a multi-billion-parameter LLM with the agent graph is a
  compile-time monster, and the repo's own notes document XLA's sensitivity on
  ROCm (LDS limits, compile ordering).

**Tier 3 — synthesis (host, parity-gated).** The open-ended path: the LLM
synthesizes new rules (new `:trans`/`:emit` rows, policy statements, embedding
adjustments), constrained to the sound fragment. The synthesis compiles through
the §5 pipeline to a candidate graph, which must pass compiled-vs-oracle
property tests against mealy's reducer on randomized event sequences before
hot-swap (fail → back to the LLM with the counterexample). Frozen paths are
unaffected by construction — the IR rejects structural changes to frozen
handlers at compile time.

The division of labor: Tier 2 proposes *actions* (masked, bounded, in-graph);
only Tier 3 promotes new *rules* (gated, host-side). The parity gate needs the
mealy oracle, which is host-side — that part of the boundary is load-bearing.

## 10. Reuse vs novel

**Reuse — proven, measured, do not rebuild:**

- clj-einsum pipeline: tensor-logic AST → EDN SSA → StableHLO → PJRT (untouched).
- Embedding construction + margin guarantees (D=512: 0.88 fact / 0.89 rule).
- Rule-as-contraction lowering, `forward-chain`, sparse-coordinate accumulation.
- Property-test harness: compiled-vs-oracle parity, seeded.
- mealy: cell semantics, handler contract, OODA stage names, mitosis thresholds —
  and its reducer as the **oracle** for every parity test below.

**Novel — the actual research:**

1. Cell IR (§4): Mealy-cell → tensor-logic rules with formal semantics.
2. Cardinality-driven dense/sparse backend selection in the compiler (§5).
3. Dispatch tensor for hierarchy; discrete + soft modes (§7).
4. Plasticity ops as compiled contractions: Hebbian writes, decay, traces (§8).
5. Population batching: batch-dim execution, per-agent state indices (§5).
6. Tier 2 deliberation: uncertainty-routed in-graph LLM (§9) — verbalization
   bridge, warm KV cache, in-graph lexical interrupt, T=0 precondition gate
   (embedding-space frozen mask); Stage A / Stage B.
7. Tier 3 synthesis: LLM rule synthesis with parity-gated hot-swap (§9).
8. Frozen-as-structural: safety invariants as compile-time graph properties (§4).
9. Tier 1a committed KB (§9): schema-constrained Datalog store —
   typed relations, cardinality/identity/denial constraints as the
   hypothesis-class prior E20 showed was missing; verified commit path
   (typecheck → constraint check → ambiguity check → provenance stamp);
   disjunctions stored explicitly where observations underdetermine the
   truth (E20 P0b); fixpoint host-side, materialized facts shipped to
   the VRAM hot set.
10. In-graph KB query arm vs yield-to-host (§9): token-ID-triggered KB
    protocol answered inside the decode graph without host round-trip;
    all other tool calls pause-DMA-resume with the KV cache resident;
    KB miss doubles as the host-paging mechanism.
11. Pause/resume with hot KV cache as the first mechanism — everything
    in item 10 hangs off it.

## 11. What stays host-side

I/O and the bus, SCI bootstrap evaluation, Tier 3 synthesis LLM calls,
persistence snapshots (`:sci-ctx` exclusion already marks the boundary),
cryptography, cross-cell message transport, the discrete-dispatch router,
mitosis orchestration above the on-device predicate. (Tier 2 deliberation LLM
execution is device-resident per §9 — "host-side" was never the right axis;
latency tier and verifiability are.) Discipline: effects at the boundary,
deterministic core — one level up from mealy's current Sans-IO posture.

Also host-side, load-bearing rather than incidental: the Tier 1a KB
*write path* — schema-checked commit (typecheck → constraint check →
ambiguity check → provenance stamp), Datalog fixpoint/materialization,
and the cold fact store behind the VRAM hot set. Reads are device-hot;
writes are host-verified. That asymmetry is the epistemic boundary the
E17–E20 series earned: the LLM proposes (Tier 2), the discrete gate
disposes (host), the KB remembers (Tier 1a).

## 12. Grounded numbers and budgets

- D=512, seed 42 (paper-experiments): fact margin 0.8843, rule margin 0.8866 —
  argmax decisions are clean; this is what makes discrete dispatch and phase
  transitions trustworthy.
- Margin-vs-D sweep (5 seeds × D∈{64..2048}): all 30+30 runs clean at threshold
  0.5; mean margins 0.75→0.94, variance collapses with D. Pick D per capacity
  needs; 512 is the validated default.
- Per-step cost (sparse-dominated, k matches): ~4 × k·D gathers + small dense
  matmuls. Dense fallback: 512³ ≈ 0.27 GFLOPs per contraction.
- Population: B agents = batch dim; 1024 × 0.27 GFLOPs ≈ 0.27 TFLOPs of dense
  compute — trivially absorbed by one GPU; expect launch overhead, not FLOPs,
  to dominate. (All PJRT-CPU measured for primitives; GPU figures projected.)
- Honest limits: soft-dispatch-at-scale untested; online-write capacity curve
  unmeasured (Phase 2 experiment); tempered sampling needs XLA RNG in-graph
  (supported, unverified in this pipeline).
- E17–E20 (predicate invention / retrieval boundary — the empirical case
  for the Tier 1a KB): E17 A2 MRR +98% (5/5 seeds) with discrete Parent
  F1 0.03 vs 0.80 target — soft ranking without crystallization; E18
  pointwise selectivity 0/40 in 11/12 runs on frozen Gemma 4 E2B —
  boundary replicates at 2.3B scale; E19 GD logit ranking of true edges
  2600/4032 (worse than random), unconstrained discrete search F1 0.01 —
  the continuous optimum is adversarial to the discrete truth; E20
  constrained SA F1* 0.03 with the true structural prior (search
  confound: 1.42/2.00 vs truth 2.00; objective confound: overlapping
  parent-pairs tie truth) — the prior must be *schema*-strength, and the
  optimizer exact. Mating ambiguity: raw-F1 ceiling 0.70, derived and
  confirmed (P0b) — the KB must store disjunctions, not guesses.
- Tier 2 budgets (projected, not measured): worst-case deliberation latency =
  K iters × T tokens × per-token decode — bounded and known at compile time
  by construction; KV-cache prefix (aim + policies + system) resident across
  steps; Stage A host round-trip is one f32 margin scalar per step. Stage B
  co-compilation cost unknown — measure before committing.

## 13. Phased plan

- **Phase 0 — spike.** Express `:observation` (frozen) as a tensor-logic
  program; parity-test one handler against `mealy.cell.reducer/handle-event`.
  Kills the biggest unknown first: does the IR capture real handler semantics?
- **Phase 1 — closed cell.** Cell IR + compiler for a fixed transition table
  (no plasticity, no hierarchy). Test: `agent-parity-test` — randomized event
  sequences, seeded, compiled output ≡ reducer output, 200+ cases.
- **Phase 2 — memory.** Episodic/semantic tensors with Hebbian writes.
  Tests: write→read round-trip parity; **the capacity experiment** — margin vs
  number of superimposed facts under online writes, D∈{512,1024,2048}.
  This experiment gates Phase 8.
- **Phase 2B — committed KB (write path).** The Tier 1a schema-constrained
  Datalog store, host-side: schema declaration (types, cardinality,
  identity, denial constraints), verified commit
  (typecheck → constraint → ambiguity → provenance), disjunction store,
  host-side fixpoint with materialized facts. Demo schema: the E20
  family generator (parent/2 indegree-{0,2} + acyclicity, sibling/2
  pair-identity, grandparent/2 derived). Tests, in the E-spec style:
  true facts commit / violations rejected by name / mating-ambiguity
  stored as disjunction (`:ambiguous`, never a guess) / provenance chain
  queryable / read-latency baseline measured. This phase gates the
  in-graph KB query arm — no dispatch without a store worth dispatching to.
- **Phase 3 — hierarchy.** Two-level dispatch, discrete mode. Test: parity vs
  nested mealy cells over scripted parent/child event flows.
- **Phase 4 — populations.** Batch-dim execution. Test: B=128 deterministic
  equivalence vs sequential single-cell execution; measure per-step latency vs
  host loop (the optimality claim, with numbers).
- **Phase 5 — Tier 2 deliberation, Stage A.** Two PJRT executables (reflex +
  LLM) sharing device memory; host dispatches on the margin scalar. Device-
  resident verbalization table; warm KV-cache prefix. Test: scripted
  low-margin situations trigger deliberation; high-margin situations don't
  (parity on the router); deliberated actions stay within the frozen mask;
  per-step latency measured vs Tier 1-only.
- **Phase 6 — Tier 2 deliberation, Stage B.** The while loop compiled in-graph
  via StableHLO cond/while. Research-grade; gated on Stage A measurements and
  on XLA co-compilation tractability. Test: behavioral equivalence with
  Stage A on scripted flows; compile-time and memory recorded.
- **Phase 7 — Tier 3 synthesis.** Rule synthesis → compile → parity gate →
  hot-swap on a toy domain (e.g., synthesize a new `:tap` handler). Test: gate
  rejects a deliberately unsound rule; frozen paths immutable across swaps.
- **Phase 8 — plasticity.** Eligibility traces; reward-modulated policy updates
  on a bandit-style task. Research-grade; gated on Phase 2 capacity results.

Each phase: generative (`clojure.test.check`) property tests written before
implementation (repo Rule 1, strict TDD), `clojure -M:format` / `-M:lint` /
full test suite clean before merge (repo Rule 3); tests run on PJRT CPU (the
verified path); GPU numbers recorded as projections until measured.

## 14. Open questions

1. Online-write capacity curve (§8) — the one experiment everything depends on.
2. Does discrete dispatch's host round-trip matter? (Probably not: one index
   each way. Measure in Phase 3.)
3. Constrained rule synthesis: how does the LLM stay inside the sound fragment?
   (Grammar-constrained decoding vs post-hoc parity rejection — try rejection
   first, it's already the gate.)
4. Entity-ization schema for rich payloads: how much of an event's EDN crosses
   into tensor-land before returns diminish? (Start: keywords + small enums only.)
5. Soft dispatch cost at depth: superposed child execution per step per level —
   where's the hierarchy depth limit? (Measure in Phase 3.)
6. D selection per relation: one global D or per-tensor D? (Per-tensor complicates
   cross-tensor contractions; start global, revisit with data.)
7. Tier 2 router: constant τ vs learned τ — what trains τ, and against what
   objective? (Start constant; learned τ needs a meta-objective, e.g.
   deliberation budget vs decision quality.)
8. Verbalization coverage: which entities get token-id rows, and what happens
   when deliberation needs one that isn't in the table? (Fallback: Tier 3
   synthesis to extend the table — a clean Tier 2 → Tier 3 escalation.)
9. Stage B compile tractability: at what LLM size does co-compilation with the
   agent graph become infeasible on the available hardware? (Measure in
   Phase 6; small models first.)
10. T=0 gate soundness: repeat the margin-vs-D measurement on LLM-generated
    action candidates over superposed tool-observation state (§9). Until then
    the precondition gate is a heuristic filter, not a verified invariant.
11. Disjunction at scale: the mating ambiguity was a 2-way disjunction over
    8 trees. What is the representation when ambiguity compounds (nested
    disjunctions, hundreds of completions)? When does the KB collapse a
    disjunction by seeking information vs carrying it? (Carrying is the
    default; collapsing needs a decision rule, which is Tier 2's job.)
12. Schema evolution: who updates the schema when the world changes — new
    relations, revised cardinalities? (Tier 3 synthesis is the natural
    owner, parity-gated like rule synthesis. Unowned schema drift is how
    KBs rot.)
13. KB paging policy: what promotes/demotes facts between the VRAM hot set
    and the host cold store? (Start: recency + query frequency; the miss
    path already exists, so the policy can start dumb and graduate.)
