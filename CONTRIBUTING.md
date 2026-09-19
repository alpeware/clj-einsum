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
4. **[AGENT-LOOP.md](AGENT-LOOP.md)** — if your change touches the agent loop,
   the KB write path, or the reduce.

## The contribution shape

Every contribution follows the same loop — it is the project's research loop
at contribution scale:

**spec → implement → test → measure → catalog**

- **Spec.** For experiments: write the spec first, with pre-registered,
  falsifiable criteria and fair baselines. E24's lesson is now policy: a
  baseline that differs in prompt, token budget, or compute from the treatment
  is not a baseline. If you can't state what would falsify your claim, you
  don't have a claim yet.
- **Implement.** Narrowly scoped diffs. Stage only the files your change
  intends; never sweep up unrelated in-flight work. Follow AGENTS.md Rules
  1–4: generative tests before core logic, pure functions with side effects
  at the boundary shell, no host-side matrix-math escape hatches.
- **Test.** `clojure -M:format`, `clojure -M:lint`, and
  `clojure -M:test -m clj-xla.test-runner` must all pass. Do not ignore
  linter warnings. New behavior needs tests; new invariants need generative
  (`test.check`) tests.
- **Measure.** Ground every number. Label reference-interpreter results,
  simulations, and compiled on-device measurements as what they are. Report
  modest numbers as modest. No SOTA claims without evidence. Results are
  specific to the hardware — say which hardware.
- **Catalog.** Empirical results get a journey entry in
  `docs/tensor_logic/empirical_journey.md`. The journey is the changelog that
  matters here: it is what the next contributor — agent or human — reads
  before spending compute. **Negative and null results are contributions.**
  E14–E20's failures shaped this project more than most of its successes. A
  well-measured dead end, written up honestly, is worth more than an
  overclaimed win. (E24's first writeup overclaimed; peer review caught it;
  the correction is in the catalog. That is the system working.)

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
