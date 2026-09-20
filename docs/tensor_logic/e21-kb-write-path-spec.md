# Experiment E21 — Schema-Constrained KB Write Path

**Status:** SPEC (unimplemented)
**Date:** 2026-09-17
**Thread:** Agent loop (builds on E17–E20; first construction spec after four
diagnostic experiments)

---

## 1. Thesis

E17–E20 were a requirements document in disguise. Jointly they specify the
agent's long-horizon memory:

- **E17:** GD learns soft associations, cannot crystallize discrete
  predicates → long-horizon memory must be an external *discrete* store,
  not weights or activations.
- **E18:** soft proposes, T=0 disposes → the store's *write path* is a
  discrete verification gate. Nothing enters on confidence alone.
- **E19:** GD's structural rankings are anti-informative; unconstrained
  search finds spurious covers → the LLM is a proposer/ranker over
  candidates, never the decider.
- **E20:** indegree + acyclicity were too weak; the missing prior was
  *schema* — integrity constraints with teeth (siblings ⇒ identical
  parent-pairs). And P0b's mating ambiguity: some truths are unknowable
  from the agent's observations, so the store must represent *disjunction*,
  not arbitrary commitment.

E21 builds the artifact these four experiments specify: a
**schema-constrained Datalog KB with a verified write path**. This is the
Tier 1 committed store from the agent-loop sketch (Datalog KB in VRAM,
queried in-graph; writes committed host-side). E21 builds the **write
path and the constraint machinery**; in-graph (VRAM) query dispatch is
the follow-up build, not this one.

## 2. What it is

`einsum.kb.store` — a Datalog-with-integrity-constraints store.
Sans-IO core (pure functions over an immutable KB value); effects at the
boundary shell. In that order:

1. **Schema declaration.** Relations declare: name, arity, per-argument
   types (entity sorts), and integrity constraints:
   - *cardinality:* e.g. `parent/2`: every child has exactly 2 parents
     (the E20 indegree ∈ {0,2} prior, now a schema invariant);
   - *identity:* e.g. `sibling/2 ⇒ identical parent-pairs` (the E20
     overlap-slack lesson — S-pairs sharing *a* parent is not enough);
   - *denial:* e.g. acyclicity of `parent/2` (no directed cycles);
   - *domain:* argument sorts (a `parent` edge connects persons, not
     toasters — cheap, kills whole classes of LLM confabulation).
2. **Derived relations.** Rules in the Datalog fragment, e.g.
   `grandparent(X,Z) :- parent(X,Y), parent(Y,Z)`. Fixpoint computed
   **host-side** (semi-naive); materialized facts shipped to the hot
   store. No in-graph fixpoint in E21 (XLA while-loop pain is documented;
   the hot path needs query answering, not derivation).
3. **Verified commit.** `assert-fact` / `retract-fact` pass, in order:
   typecheck → constraint check → ambiguity check → provenance stamp.
   A fact that fails any check is *rejected with a named violation*,
   never silently stored. Retraction is a supersede marker, not a
   delete — the log is append-only and history is queryable.
4. **Ambiguity as a first-class citizen.** If a candidate fact is
   *underdetermined* — two or more completions satisfy schema + all
   observations (E20's mating ambiguity: `{m,s}` vs `{o,s}`) — the KB
   stores the **disjunction explicitly** and answers queries over it
   with `:ambiguous`, never a coin flip. Guessing is hallucination with
   a commit bit; the schema forbids it.
5. **Read path (host-side in E21).** Ground queries, selections, and
   bounded joins over materialized facts — boolean matmuls and gathers,
   i.e. E19's machinery repurposed from diagnosis to infrastructure.
   Latency measured and reported (target: microseconds per query
   host-side). In-graph VRAM dispatch is the *next* spec.

## 3. Demo domain (first schema)

The E20 family generator — we know its ground truth, its schema, and
exactly where naive approaches fail:

- `parent/2`: indegree ∈ {0,2}, acyclic, args :person/:person.
- `sibling/2`: pair-identity constraint (shared *identical* parent-pair).
- `grandparent/2`: derived by rule.
- The E20 observation setup (G/S pairs only) as the ambiguity demo.

## 4. Acceptance criteria (falsifiable)

1. **Schema-checked commit:** all 80 true parent edges commit cleanly;
   an 81st edge giving a child a *third* parent is rejected naming the
   cardinality violation.
2. **Cycle rejection:** asserting a backward edge (child → founder) is
   rejected naming the acyclicity violation.
3. **Type rejection:** `parent(alice, 42)` (or any sort violation) is
   rejected naming the domain violation.
4. **Ambiguity, not guessing:** fed only the E20 G/S observations, the KB
   stores the mating disjunction explicitly; query `parent(m, g1)?`
   returns `:ambiguous` with both completions enumerated. Committing
   either completion as fact is a *test failure*.
5. **Derived relations:** `grandparent/2` materializes exactly the
   generator's grandparent pairs (F1 = 1.0 vs ground truth).
6. **Provenance:** retract a committed edge → supersede marker present,
   history query returns the full assertion/retraction chain.
7. **Read latency:** p50/p99 ground-query latency measured and reported
   (no target gate — this is a baseline for the in-graph dispatch spec).

### Falsification / stop clause

If the constraint machinery cannot be made total (some schema-valid
state is unreachable, or some schema-violating state is committable) the
design is wrong, not the tests — stop and report. A KB that *sometimes*
enforces its schema is worse than no KB: it launders guesses into facts.

## 5. Non-goals (explicitly deferred)

- In-graph / VRAM query dispatch (next spec: the "recognize the tool
  call on GPU" build).
- Full recursive fixpoint in-graph.
- Truth maintenance beyond supersede markers (v2).
- Schema evolution / learning new relations (Tier 3 synthesis problem).

## 6. Deliverables

- `src/einsum/kb/store.clj` — schema, constraints, verified commit,
  disjunction store, materialization (Sans-IO core).
- `test/einsum/kb/store_test.clj` — generative tests FIRST (repo Rule 1):
  constraint totality (random schemas × random fact streams: committable
  ⟺ schema-valid), disjunction soundness (ambiguous queries never return
  a single completion), supersede-chain integrity.
- `tools/e21_kb_write_path.clj` — demo on the family-generator schema,
  printing the acceptance-criteria report.
- `paper-experiments/e21-kb/2026-09-17/` — `results.edn` with the
  acceptance report + latency measurements.
- `docs/tensor_logic/empirical_journey.md` — new **Section 22**.

## 7. Delivery instruction

**On completion: `git add` the spec's artifacts, commit
(`feat(e21): schema-constrained KB write path with verified commit`),
and `git push origin main`. Do not wait for a separate review — the
acceptance criteria above are the review.**
