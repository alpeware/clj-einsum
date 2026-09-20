# Experiment E19 — Hybrid Crystallization: GD Proposes, Discrete Search Disposes

**Status:** SPEC (unimplemented)
**Date:** 2026-09-17
**Thread:** Predicate invention (follows E17; informed by E18)

---

## 1. Thesis

E17 established a clean boundary: gradient descent over continuous tensor
compositions learns **soft associative structure** (A2 MRR 0.2348 vs 0.1186
baseline, +98%, 5/5 seeds) but does **not** crystallize discrete predicates
(Parent F1 = 0.0295 vs the 0.80 target; 1012 false-positive edges after
annealing). E18 replicated the same boundary in retrieval with a real LLM:
soft metrics movable (neighborhood selectivity, loss, boost sign), discrete
metric stuck (pointwise selectivity 0/40 in 11/12 runs).

E19 tests the hybrid thesis that both negatives point to:

> **GD is a proposer, not a crystallizer.** Let gradient descent learn a soft
> ranking over relational structure (it is good at this), then run exact
> combinatorial search over the GD-proposed candidates to commit to discrete
> structure (the step pure GD cannot take).

Concretely: E17's A2 already ranks edges well (MRR +98%). E19 takes the
top-K edges by A2's soft score as a *candidate set* and runs greedy discrete
search for the sparsest boolean matrix whose Boolean square/cross-product
covers the observed Grandparent/Sibling compositions. Success is Parent
F1 ≥ 0.80 — the exact bar pure GD missed by two orders of magnitude.

This is a known-good shape (structural EM's discrete E-step, AlphaGo's
MCTS over a neural prior, DreamCoder's neural-guided program search). E19
is its first rigorous test on the predicate-invention problem.

---

## 2. Setup (mirrors E17)

- Synthetic forest: 64 entities, 8 family trees, hidden evaluator-only
  predicate `Parent` (56 true edges: 8 trees × 7).
- Learner/searcher observes `Grandparent` and `Sibling` compositions only,
  on the **same train split** E17 used (`poc_predicate_invention.clj`).
- Evaluation: discrete F1 of the recovered boolean matrix against hidden
  `Parent` over **all** pairs (the metric E17 failed).
- Seeds: 42–46 (same five as E17).

### 2a. The soft proposer (retrain, don't reuse)

Retrain E17's A2 variant (full latent matrix `L = sigmoid(Z / τ)`,
τ annealed 1.0 → 0.05, 200 epochs, CPU PJRT — ~0.5 s/seed) per seed.
**Rank candidate edges by raw logits `Z`, not by annealed sigmoids.**
Rationale: τ → 0.05 snaps positives to 1.000 and destroys magnitude
information; the logits preserve GD's learned ordering. This is a
deliberate choice — the proposer wants GD's *ranking*, not its
(pseudo-)discretization.

- Candidate set `C` = top-K off-diagonal edges by `Z`, **K = 256**
  (≈4.6× the 56 true edges; large enough for high recall, small enough
  for greedy search). Exclude self-loops (Parent is irreflexive).

---

## 3. Phase 0 — Gates (cheap; run first)

### P0a. Oracle ceiling (validates the search machinery, not the learning)

Run the Phase-1 discrete search with `C` = the 56 true Parent edges.
**Gate: must reach F1 = 1.0.** If it does not, the objective/search is
mis-specified — fix the machinery before testing the thesis.

### P0b. Proposer quality (validates GD as a proposer)

Measure candidate recall: fraction of the 56 true Parent edges present in
the top-256 by `Z`, per seed.
**Gate: mean recall ≥ 0.90.** If GD does not even rank the true edges
highly, the hybrid thesis fails in this form — stop and report (do not
run Phase 1 blind).

---

## 4. Phase 1 — Discrete search + controls

### Objective (Boolean semiring — this *is* the crystallization step)

For boolean matrix `L̂` with support ⊆ C:

```
score(L̂) = F1(Bool(L̂ ∘ L̂), G_train) + F1(Bool(L̂ᵀ ∘ L̂), S_train) − λ·‖L̂‖₀
```

- `∘` is Boolean (OR–AND) matrix product; `Bool(M) = (M > 0)`.
- `G_train`, `S_train`: observed Grandparent/Sibling pairs on the train split.
- **λ = 0.01** (default): each edge costs 0.01 against ≈2.0 of achievable
  coverage F1 — sparsity matters, coverage dominates. Defensible, not tuned.

### Algorithm (deterministic greedy + backward elimination)

1. Forward: start from ∅; repeatedly add the edge in C \ E maximizing
   `score`; stop when no addition improves. Tie-break by lower edge index.
2. Backward: repeatedly remove the edge whose removal most improves
   `score`; stop when no removal improves.
3. Report `L̂`, its F1 vs hidden Parent, |L̂|, and search cost
   (objective evals, wall time). Boolean 64×64 matmuls are microseconds;
   the whole search should run in seconds on host.

### Cells (per seed)

| Cell | Candidate set | Tests |
|---|---|---|
| **H (hybrid)** | Top-256 by `Z` | The thesis |
| **B0 (no proposal)** | All 4032 off-diagonal edges | Is GD's proposal *necessary*? Greedy over all edges is still tractable — the question is whether it finds spurious solutions or matches H. |
| **B1 (random proposal)** | 256 random edges (3 draws/seed) | Does GD's *ranking* carry the information, or would any 256 edges do? |

### Pre-registered success criteria

1. **Crystallization (primary):** H reaches Parent F1 ≥ 0.80 (mean over
   seeds) — the bar E17 set and missed at 0.03.
2. **Attribution:** H beats B1 by ≥ 0.30 F1 (mean) — GD's ranking carries
   the information, not just the candidate budget.
3. **No collapse:** |L̂| within 2× of truth (28–112 edges); search
   completes in < 5 min/seed on host.

### Falsification clause

- If H mean F1 < 0.80 → the hybrid thesis **fails in this form**; report
  as a second clean negative (the boundary holds against hybrids too,
  which would point toward stronger symbolic priors, not just search).
- If P0b recall < 0.90 → GD is not even a sufficient proposer; stop.
- Pre-registered interpretation of B0: if B0 ≥ 0.80 too, the win belongs
  to "discrete search is the missing piece" (proposal = tractability
  convenience, not necessity). If B0 < H, the proposal step is load-bearing.
  Either way the discrete-search conclusion stands; only the *role* of GD
  is at stake.

---

## 5. Deliverables

- `tools/e19_hybrid_crystallization.clj` — proposer retrain + P0 gates +
  Phase-1 search (Sans-IO core: boolean matmul, greedy search as pure fns).
- `paper-experiments/e19-hybrid/2026-09-17/` — `phase0.edn`,
  `results.edn`, `summary.csv` (per-seed: recall@256, H/B0/B1 F1,
  |L̂|, evals, wall time).
- `docs/tensor_logic/empirical_journey.md` — new **Section 20** with the
  result, the gate decisions, and the updated program verdict.
- Generative tests first (repo Rule 1): boolean-matmul correctness vs a
  naive oracle, greedy monotonicity (score never decreases across a
  forward/backward pass), oracle-ceiling as a test (P0a doubles as one).

## 6. Delivery instruction

**On completion: `git add` the spec's artifacts, commit
(`feat(e19): hybrid crystallization — GD-proposed discrete predicate
search`), and `git push origin main`. Do not wait for a separate review —
the pre-registered criteria above are the review.**
