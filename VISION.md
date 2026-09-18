# VISION.md — Executable Tensor Logic for Coding Agents

## Mission

**Make the tensor equation the working representation for the parts of AI that
are tensor computation — not a formalism on paper, but a toolchain where a
coding agent (or a human) writes equations and gets CPU/GPU executables,
gradients, and verified behavior out.**

We are deliberately narrower than Domingos' program. We do not claim the
tensor equation unifies all of AI, and we do not need it to. What we claim is
pragmatic: a large, commercially central slice of AI *is* tensor computation —
attention, dense projections, convolutions, embeddings, relational joins over
embedded facts — and that slice deserves a representation that is readable,
executable, and agent-manipulable. That representation is the Hiccup AST in
this repo. Where the grand claims break (absolute symbolic truth in embedding
space, gradient-discovered predicates), we measure the break and build the
discrete machinery around it instead of pretending the equation covers it.

## What the representation covers — and what it doesn't

Explicit, because coding agents (and reviewers) should know the boundary
before they trust it.

**Covered — the parts that work:**

- **Tensor contraction (einsum).** The shared computational core of modern deep
  learning: attention, MLPs, convolutions, embedding lookups, kernel
  operations. One construct, `[:= head & body-terms]`; a shared index
  contracted on both sides *is* the matmul, the join, the attention score.
  No `:matmul` op is needed because contraction is the operation.
- **Function-free Datalog-style rules**, compiled to contractions over embedded
  relations. Paper 1 measures this: rule-as-contraction, separation margins,
  extract–threshold–re-embed denoising, the deductive/analogical temperature
  tradeoff.
- **Elementwise nonlinearities and the autodiff closure.** Activations are
  attributes on equations (`{:act :gelu}`), not separate ops, and the gradient
  of a tensor-logic program is itself a tensor-logic program.
- **Bounded, statically-shaped dataflow** — exactly what XLA/StableHLO executes
  well — with a reference interpreter for differential verification of every
  compiled result.

**Not covered — and not claimed:**

- Full first-order logic (function symbols), modal or temporal logics,
  non-monotonic reasoning, exact probabilistic inference. Where this language
  is logical at all, it is Datalog-shaped.
- *Absolute* soundness in embedding space. Random embeddings give PAC-style,
  ε-bounded separation — measured margins, not metaphysical guarantees.
- Gradient-discovered exact predicates. The E14–E20 arc says gradients learn
  rankings, not discrete truth. Predicate invention needs exact optimization
  plus strong schema priors, and that machinery lives *around* the tensor core
  as host-side discipline (verified KB writes, the T=0 commitment gate) — not
  inside the equation language.

**The durable thesis:** one readable language for the tensor-computable slice,
multiple engines underneath (dense XLA today; sparse/relational lowering where
the equations don't reach), and a discrete commitment mechanism wherever
exactness is required. The experiments are the contribution; the grand
unification is not.

## Why this, why now

Three observations, all from the last year of work:

1. **The notation is the leverage.** Domingos: *"a good notation is half the
   battle... you can just think better and faster once you have this notation."*
   Our Hiccup AST has one construct — `[:= head & body-terms]` — and everything
   in the covered slice (attention, MLPs, relational memory, symbolic rules) is
   expressed in it. It is also *data*, not syntax: homoiconic, diffable,
   programmatically rewritable. Dead-code elimination, index allocation, and
   shape inference are data transforms, which is why agents manipulate it
   fluently and why no competing representation (einsum strings, einops,
   hand-written PyTorch) is as workable. The E1→E12 arc — a relational
   transformer designed, trained, debugged, and re-engineered in weeks — is the
   empirical test of that claim. It passed.

2. **Coding agents are the native users of this notation.** When MLST's Tim Scarfe
   fed the Tensor Logic paper to Claude Code, it *"implemented the whole lot this
   afternoon."* That is not a party trick; it is evidence about the shape of the
   formalism. A tiny grammar, explicit dataflow, pure-data programs, and a
   reference interpreter that checks device output give agents a tight
   write→compile→verify loop. Frontier agents don't just tolerate this notation —
   they are fluent in it. We build for them first, humans second.

3. **The field is re-learning reasoning at trillion-dollar cost.** Domingos'
   challenge stands: read the textbook before burning the compute. But the
   textbook's grandest claim — one language for all of AI, everything learned
   jointly — is not our bet. Our bet is smaller and checkable: the tensor
   equation is the right *representation* for the tensor-computable slice, the
   compiler makes it executable, and discrete commitment machinery handles the
   exactness the equations cannot supply. E23 is the existence proof of that
   split: proposals from the continuous side, commitment from the discrete
   side, measured over an adversarial horizon.

## What this repo is

A vertically integrated Tensor Logic toolchain in Clojure:

```
tensor equations (Hiccup AST, pure data)
        │  expand · lower · autodiff — all as data transforms
        ▼
EDN SSA graph IR (Malli-schematized, verifiable)
        │
        ▼
StableHLO → PJRT executables (CPU / ROCm / CUDA)
        │
        ▼
reference interpreter ⇄ device differential testing
```

One construct for the covered slice, all the way down. No kernels written by
hand; no host matrix math in the hot path. Equations become executables —
Domingos' *"map onto a GPU with almost no change,"* realized as
`clj-xla.logic.*` → `clj-xla.compile`.

Two halves, one repo: **`logic/` is the language** (AST, lowering, symbolic
reasoning, relational memory, the agent loop); **the rest is the substrate**
(PJRT bindings, StableHLO builder, compiler cache, autodiff, model loaders).
The split is at the namespace level, deliberately — the integration *is* the
product.

## Pillars

**Pillar 0 — The compiler. (Built.)** Tensor-logic AST → StableHLO → PJRT, with a
reference interpreter for differential verification. Real models run on it:
GPT-2, SmolLM, Gemma 2/3/4 (35-layer E2B inference on ROCm, in-graph dequantized
INT4). Paper 1 ("Executable Tensor Logic") documents this half.

**Pillar 1 — Trainable Tensor Logic. (Now.)** Structure learning and predicate
invention *inside the language* — but honest about the E14–E20 results:
gradients reliably learn rankings, not exact predicates. Current work:
relational memory (per-relation cores as learned soft priors), exact in-VRAM
adjoints, per-rule temperatures, WebNLG-scale training. The live hypothesis is
exact optimization plus strong schema priors, with the tensor core supplying
fast proposals and the discrete machinery supplying commitment. The answer will
come from the equation→compile→measure loop, one equation at a time. Paper 2
("Trainable Tensor Logic") is gated on this half working end-to-end.

**Pillar 2 — The agent loop. (Design → prototype.)** A three-tier agent
architecture — reflex contractions, device-resident deliberation, host-side
synthesis — where the agent's *own* reasoning substrate is tensor equations it
can read, verify, and rewrite, and where every state mutation passes through a
verified discrete commit. `AGENT-LOOP.md` holds the design. The endgame:
agents that do AI research in the same write→compile→verify loop we used to
build this.

## Operating principles

- **Ground every number.** Reference-interpreter results, simulation, and
  compiled on-device measurements are labeled as what they are. No SOTA claims
  without evidence; modest numbers reported as modest.
- **Verify on the device.** The reference interpreter exists so agents can check
  compiled output differentially. A claim about the compiler is not done until a
  test runs it on PJRT.
- **Claim only what the representation covers.** The "covers / doesn't cover"
  list above is a living contract. When an experiment finds a boundary, the
  boundary gets written down — it doesn't get hand-waved.
- **Soundness is a property of the gate, not a slogan.** T=0 deduction guarantees
  conclusions follow from premises — it says nothing about whether the premises
  are true. The gate is discrete machinery *around* the tensor core, not a
  property of embedding space. Measure the gate's *effect*, not its presence.
- **Agents in the loop, by construction.** Programs are data; diffs are semantic;
  verification is automatic. Every new capability must be usable — and checkable
  — by a coding agent, not just a human at a REPL.
- **The 80/20 rule is fine.** The clean formalism covers most of it; the rest is
  hacks (stride fixes, padding, host orchestration). Domingos: Tensor Logic
  *"makes it much easier and faster to do those hacks."* We keep the hacks
  explicit and measured, not hidden.

## Non-goals

- **Not a unification theory.** We do not claim the tensor equation is the
  language of all AI, and we don't need that claim to be useful. The
  representation earns its keep on the slice it covers.
- **Not a PyTorch competitor.** We are not chasing FLOPS parity or framework
  adoption on their terms. The contest is expressiveness-per-equation and
  agent-velocity, not benchmark throughput.
- **Not a chatbot product.** No assistant, no demo app. The repo is infrastructure
  for people and agents building AI systems.
- **Not formalism for its own sake.** Every language feature must compile, run,
  and be measured — or be cut.

## What success looks like

- A coding agent, given a research idea expressible as tensor equations, produces
  a compiled, device-verified, trained result in the same session — the E1→E12
  loop, fully agent-driven.
- Trainable Tensor Logic retrieves reliably: relational memory that *selects* the
  right fact, not just steers toward it; per-rule temperatures that are actually
  learned rather than configured; predicate invention demonstrated on a real
  benchmark — or a measured, written-down account of where it stops working.
- Paper 1 published (the compiler exists, measured). Paper 2 submitted (the
  learning works, measured — with the E14–E20 negative results given full weight).
- The repo becomes the place agent builders reach for when they want verified,
  compiled neural-symbolic programs instead of Python string-soup.

## Relation to Domingos' program

Domingos gave a notation and a destination — *"the Turing machine equivalent
for induction."* We take the executable-representation half of that vision
seriously and hold the unification metaphysics at arm's length. Where his
claims measure out — contraction *is* the join, ε-bounded soundness at T=0,
the uniformity of the lowering — we build on them. Where they don't —
absolute soundness in embedding space, gradient predicate invention — we say
so with numbers (E14–E20) and build the discrete machinery the framework
lacks (E21–E23). This repo is the pragmatic core of Tensor Logic: the
representation that works, compiled, measured, and placed exactly where its
boundaries are.
