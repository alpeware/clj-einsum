# Experiment E20 — Constrained Crystallization: Symbolic Priors Make Factorization Identifiable

**Status:** SPEC (unimplemented)
**Date:** 2026-09-17
**Thread:** Predicate invention (follows E19; informed by E17/E18)

---

## 1. Thesis

E19 proved two things: (a) GD's soft knowledge lives at the composition
level and its edge ranking is *anti*-informative (median rank 2600/4032),
so "GD proposes edges" is dead; (b) unconstrained discrete search finds
spurious 40-edge shortcut covers scoring as well as truth — **(G,S)
underdetermines L**.

E20 tests the remaining hypothesis:

> **The missing ingredient is not better search but symbolic domain
> priors.** The generator is a forest with indegree ∈ {0,2} (every
> non-founder has exactly a mother and a father) and acyclic generations.
> Constrain discrete search to exactly that hypothesis class — no GD
> anywhere except a pre-registered tiebreaker cell — and test whether the
> factorization becomes identifiable.

## 2. The generator's true structure (read from `family_trees.clj`)

Per tree (8 entities): 2 founders → 2 gen1 children (4 edges);
gen1 mating-child + gen1 spouse → 3 gen2 children (6 edges).
**Indegree ∈ {0,2} exactly** — 3 roots (founders + spouse), 5 non-roots.
(Correction: an earlier note said "indegree ≤ 1"; the code shows every
child has two parents. The spec follows the code.)

### 2a. Pre-registered identifiability analysis: the mating ambiguity

Claim: **no method observing only (G,S) can exceed raw Parent F1 = 0.70
on this generator**, because of a symmetry the observations cannot break.

*Derivation.* Fix a tree with gen1 children {m (mating-child), o (other)}
and spouse s, gen2 children {g1,g2,g3}. Truth: parents(g_i) = {m,s}.
Alternative: parents(g_i) = {o,s}. Both assignments generate:
- identical G pairs: (f1,g_i),(f2,g_i) via m resp. o, where (f1,m),(f2,m)
  and (f1,o),(f2,o) are all true parent edges;
- identical S pairs: gen2 children share {m,s} resp. {o,s}; (m,o) share
  f1,f2 in both worlds.
Hence the alternative explains *every* observed pair exactly as well as
truth. Per tree the alternative recovers 7/10 edges (4 founder + 3 spouse
edges) with 3 FP → F1 = 0.70; across 8 trees the ceiling is **0.70**.

*Consequence.* The primary metric must be ambiguity-aware:
**F1\* = mean over trees of max(F1 vs truth, F1 vs mating-swapped truth)**,
ceiling 1.0. Raw F1 (ceiling 0.70) is reported secondary — if constrained
search clusters near 0.70 raw, the ambiguity theory is confirmed live.

Gate P0b (Phase 0) verifies this claim as a pure function before any
search runs (see §4). Note: the evaluator needs each tree's mating-child,
which `generate-family-forest` does not currently expose — thread it
through (extend the forest map with per-tree `:mating-child`).

## 3. Search design: pair-assignment simulated annealing

E19's edge-greedy was structurally blind: a single parent edge covers
zero new pairs until its partner arrives (complementarity). The natural
granularity is the **parent-pair assignment**: state = for each node,
`root` or an unordered parent pair `{p1, p2}` (p1 ≠ p2, both ≠ node).

- **Hard constraints:** indegree ∈ {0,2} by construction; proposed moves
  that create a directed cycle are rejected (incremental DFS on 64 nodes).
- **Moves (single-node, SA-friendly):** pick a random node; reassign to a
  random pair (1953 candidates), make it a root, or swap pair-assignments
  of two nodes. No complementarity blindness: one move assigns a full
  parental unit.
- **Objective:** `F1(Bool(L̂∘L̂), G_train) + F1(Bool(L̂ᵀ∘L̂), S_train)`.
  **No sparsity penalty λ.** E19's lesson: λ=0.01 made the spurious
  40-edge cover tie truth (1.20 vs 1.20). The indegree constraint *is*
  the sparsity prior; λ would reintroduce the misalignment.
- **Schedule (defaults, defensible not tuned):** 200k iterations,
  T 0.1 → 1e-4 geometric, 3 restarts/seed, keep best. Seeds 42–46.
  All host-side boolean ops; expect seconds per restart.
- **Implementation:** new `einsum.logic.constrained-search` namespace
  (Sans-IO core: pair-state, boolean composition, SA loop as pure fns);
  do not disturb E19's tested `predicate-crystallization` ns. Generative
  tests first: boolean composition vs naive oracle; indegree invariant
  holds across random move sequences; acyclicity rejection fires on a
  hand-built cycle; P0a doubles as the oracle test.

## 4. Phase 0 — Gates (cheap; run first)

- **P0a (oracle ceiling):** constrained SA restricted to the true
  parent-pairs as the only assignable pairs. **Gate: F1\* = 1.0.**
  Failure means the machinery (not the thesis) is broken.
- **P0b (ambiguity verification):** pure function — for each tree/seed,
  build the mating-swapped edge set and confirm it reproduces *all*
  train G/S pairs. **Gate: passes on all 8 trees × 5 seeds.**
  Failure means the §2a derivation is wrong; stop and revise the metric
  before burning SA cycles.

## 5. Phase 1 — Cells (per seed)

| Cell | Search | Tests |
|---|---|---|
| **H (constrained)** | Pair-assignment SA, indegree ∈ {0,2} + acyclicity | The thesis: symbolic priors make it identifiable |
| **B0 (unconstrained)** | E19's edge-greedy, re-run under identical conditions | Control: do the constraints do the work? (E19: F1 ≈ 0.01) |
| **T (tiebreaker)** | H, but SA move ties broken by mean GD logit Z of the pair | Pre-registered test of "GD only as tiebreaker": expect T ≈ H (E19: Z anti-informative); T < H would show Z is actively harmful |

### Pre-registered success criteria

1. **Identifiability (primary):** H reaches **F1\* ≥ 0.80** (mean over
   seeds) — the bar E17/E19 missed, now achievable (ceiling 1.0).
2. **Attribution:** H beats B0 by ≥ 0.50 F1\* — the prior, not the
   search budget, explains the gap.
3. **Ambiguity confirmation:** H raw F1 ∈ [0.60, 0.75] — clustering at
   the derived 0.70 ceiling validates §2a live. (Raw F1 > 0.80 would
   *falsify* the ambiguity derivation — report it as such.)
4. **No collapse:** recovered non-roots within 2× of truth; < 10 min/seed.

### Falsification clause

- If H mean F1\* < 0.80 → **even the true symbolic prior does not make
  the factorization identifiable** from compositions alone. Close the
  predicate-invention-from-compositions line; the problem needs richer
  observations (typed variables, arity signatures, partial labels), not
  better optimizers.
- If P0b fails → the metric theory is wrong; do not run Phase 1 blind.

## 6. Deliverables

- `src/einsum/logic/constrained_search.clj` + generative tests.
- `tools/e20_constrained_crystallization.clj`.
- `paper-experiments/e20-constrained/2026-09-17/` — `phase0.edn`,
  `results.edn`, `summary.csv` (per-seed: F1\*, raw F1, non-roots,
  objective, restarts, wall time).
- `docs/tensor_logic/empirical_journey.md` — new **Section 21**.

## 7. Delivery instruction

**On completion: `git add` the spec's artifacts, commit
(`feat(e20): constrained crystallization — symbolic priors for predicate
identifiability`), and `git push origin main`. Do not wait for a separate
review — the pre-registered criteria above are the review.**

## 8. Extensions (not required)

- Ablate the prior: indegree ≤ 2 (allow single parents) vs ∈ {0,2}; drop
  acyclicity — which constraint carries the identifiability?
- If H succeeds: the retrieval-side hybrid (E18 follow-up) is the next
  frontier — soft shortlist + T=0 symbolic verification over candidates.
