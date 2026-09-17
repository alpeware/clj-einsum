# VISION.md — Executable Tensor Logic for Coding Agents

## Mission

**Make Tensor Logic the working language of AI development — not a formalism on
paper, but a toolchain where a coding agent (or a human) writes tensor equations
and gets GPU executables, gradients, and verified behavior out.**

Pedro Domingos' program is our overarching goal: a single language — the tensor
equation — that unifies symbolic and neural computation, where *the gradient of a
program is just another program*, where per-rule temperature dials continuously
between analogy and sound deduction, and where learning does structure discovery
instead of brute force. What the paper sketches, this repo builds: the compiler,
the runtime, and the agent loop that make it executable.

## Why this, why now

Three observations, all from the last year of work:

1. **The notation is the leverage.** Domingos: *"a good notation is half the
   battle... you can just think better and faster once you have this notation."*
   Our Hiccup AST has one construct — `[:= head & body-terms]` — and everything
   (attention, MLPs, relational memory, symbolic rules) is expressed in it. The
   E1→E12 arc — a relational transformer designed, trained, debugged, and
   re-engineered in weeks — is the empirical test of that claim. It passed.

2. **Coding agents are the native users of this notation.** When MLST's Tim Scarfe
   fed the Tensor Logic paper to Claude Code, it *"implemented the whole lot this
   afternoon."* That is not a party trick; it is evidence about the shape of the
   formalism. A tiny grammar, explicit dataflow, pure-data programs, and a
   reference interpreter that checks device output give agents a tight
   write→compile→verify loop. Frontier agents don't just tolerate this notation —
   they are fluent in it. We build for them first, humans second.

3. **The field is re-learning reasoning at trillion-dollar cost.** Domingos'
   challenge stands: read the textbook before burning the compute. Tensor Logic
   puts deduction, analogy, and learning in one language with a temperature knob
   per rule — mathematical truths at T=0, mined heuristics warmer, all learned
   jointly. Our T=0 deductive gate and per-rule temperatures are this idea made
   concrete.

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

One construct all the way down. No kernels written by hand; no host matrix math
in the hot path. Equations become executables — Domingos' *"map onto a GPU with
almost no change,"* realized as `clj-xla.logic.*` → `clj-xla.compile`.

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

**Pillar 1 — Trainable Tensor Logic. (Now.)** Domingos' real bet: structure
learning and predicate invention *by gradient descent, inside the language*.
Current work: relational memory (per-relation cores as learned soft priors),
exact in-VRAM adjoints, per-rule temperatures, WebNLG-scale training. The open
scientific question is retrieval reliability — steering vs. selection — and the
answer will come from the equation→compile→measure loop, one equation at a time.
Paper 2 ("Trainable Tensor Logic") is gated on this half working end-to-end.

**Pillar 2 — The agent loop. (Design → prototype.)** A three-tier agent
architecture — reflex contractions, device-resident deliberation, host-side
synthesis — where the agent's *own* reasoning substrate is tensor equations it
can read, verify, and rewrite. `AGENT-LOOP.md` holds the design. The endgame:
agents that do AI research in the same write→compile→verify loop we used to
build this.

## Operating principles

- **Ground every number.** Reference-interpreter results, simulation, and
  compiled on-device measurements are labeled as what they are. No SOTA claims
  without evidence; modest numbers reported as modest.
- **Verify on the device.** The reference interpreter exists so agents can check
  compiled output differentially. A claim about the compiler is not done until a
  test runs it on PJRT.
- **Soundness is a property of the gate, not a slogan.** T=0 deduction guarantees
  conclusions follow from premises — it says nothing about whether the premises
  are true. Measure the gate's *effect*, not its presence.
- **Agents in the loop, by construction.** Programs are data; diffs are semantic;
  verification is automatic. Every new capability must be usable — and checkable
  — by a coding agent, not just a human at a REPL.
- **The 80/20 rule is fine.** The clean formalism covers most of it; the rest is
  hacks (stride fixes, padding, host orchestration). Domingos: Tensor Logic
  *"makes it much easier and faster to do those hacks."* We keep the hacks
  explicit and measured, not hidden.

## Non-goals

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
  right fact, not just steers toward it; learned per-rule temperatures; predicate
  invention demonstrated on a real benchmark.
- Paper 1 published (the compiler exists, measured). Paper 2 submitted (the
  learning works, measured).
- The repo becomes the place agent builders reach for when they want verified,
  compiled neural-symbolic programs instead of Python string-soup.

## Relation to Domingos' program

Domingos gave the language and the destination — *"the Turing machine equivalent
for induction."* Deduction's universal machine exists; induction's is missing.
This repo builds the executable half of that vision: first the compiler (Paper 1),
then the learning machine on top of it (Paper 2), with coding agents as both the
builders and the first native speakers.
