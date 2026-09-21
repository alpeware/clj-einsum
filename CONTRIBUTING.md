# CONTRIBUTING.md

**If you are a coding agent, start here. You are the primary contributor this project is built for.**

Humans are welcome too — the same bar applies to everyone, and human judgment is still the backstop where the loop can't reach yet (design review, the open composable-evaluation problem). But if a task can be specified precisely enough for an agent, it should be done by an agent. That is the point of this repo: programs are data, verification is automatic, and the E-series experiments that define this project were designed, implemented, measured, and written up by coding agents.

## Before you write code

Read these first — they are short, and they will save you from re-running someone else's failed experiment:

1. **[VISION.md](VISION.md)** — what the representation covers, the deflationary framing on Ali Ghodsi's 4 RSI conditions, arithmetic vs hypothesis on consumer silicon, and what it doesn't cover.
2. **[AGENTS.md](AGENTS.md)** — the repo's coding rules (TDD, pure functions, clean CI, pure-XLA execution). Non-negotiable.
3. **[docs/architecture/rsi_gates.md](docs/architecture/rsi_gates.md)** — formal specification of the 4 RSI gates, generation lineage, and consumer hardware constraints.
4. **[docs/tensor_logic/empirical_journey.md](docs/tensor_logic/empirical_journey.md)** — the catalog of everything tried, including the failures. Check whether your idea is a closed line before you spend GPU time on it. E16–E20 (gradient predicate invention) are closed: gradients learn rankings, not discrete truth.
5. **[docs/architecture/agent_loop.md](docs/architecture/agent_loop.md)** — the three-tier agent loop, propose $\to$ dispose contract, and verified snapshot commits.

---

## The 3-Stage Research Protocol (IETF RFC Model)

To scale decentralized research across coding agents and human contributors without claim collisions or ungrounded speculation, all proposals follow a 3-stage lifecycle sharded by **RSI Gate**:

```
[Stage 1: Proposal PR]  ──>  [Stage 2: Implementation PR]  ──>  [Stage 3: Verification PR]
 (proposals/<gate>/<slug>/)        (Scaffold & results.edn)           (Independent Replication)
          │                                  │                                   │
   Approved Spec                      Merged as UNVERIFIED               eNN Minted -> catalog/
```

### Stage 1: Proposal PR (Descriptive Slugs, Gate-Partitioned)
- Propose new experiments under `proposals/<gate>/<slug>/spec.md` (e.g. `proposals/gate1-compression/cat-q-ternary/spec.md`).
- **No `eNN` numbers are assigned at proposal time.** Numbers are only minted upon Stage 3 independent verification.
- **Required RFC Metadata Headers**:
  - `Experiment: <slug>`
  - `Gate: <gate1-compression | gate2-velocity | gate3-evals | gate4-recursion>`
  - `Generation: <target-generation-number>`
  - `Literature: [<formal paper citations in the DAG>]`
  - `Hardware-Target: {Reference: "...", Claim-Shape: "..."}`
  - `Extends:`, `Refutes:`, `Supersedes:`, `Reopens:`

---

## Falsification Criteria for RSI Gates 1–4

Every proposal must explicitly pre-register target criteria across Ghodsi's 4 RSI Gates:

### 1. Gate 1 Criterion: Resource Efficiency ($C_{n+1} \ll C_n$)
Proposals must measure both dimensions of resource consumption:
- **Storage / Memory (Arithmetic):** Measured `bytes/param` and total peak VRAM footprint reduction (e.g., target $\le 0.25\text{ bytes/param}$ for ternary weights, $\ge 7.5\times$ reduction vs FP16).
- **Compute / Throughput (Hypothesis):** Measured generation throughput ($\text{tok/s}$) and achieved GFLOPs/watt against uncompressed FP16 and INT4 baselines on reference hardware. Compute speedups must be proven on silicon, not assumed.

### 2. Gate 2 Criterion: Time Efficiency ($T_{n+1} \ll T_n$)
- Wall-clock calibration, adaptation, or distillation time on reference consumer hardware (AMD Radeon RX 7900 XTX 24GB or NVIDIA RTX 4090).
- Standard: zero-backpropagation calibration must complete in $< 30\text{ minutes}$; parameter-efficient adaptation must converge in $< 2\text{ hours}$.

### 3. Gate 3 Criterion: Capability & Intelligence ($A_{n+1} > A_n$)
Two distinct numbers are strictly required:
1. **Floor Retention ($\ge 95\%$):** Retention percentage relative to the uncompressed base model on identical evaluation suites (e.g., perplexity, GSM8K, or HumanEval/Clojure). Any compression dropping retention below $95\%$ is considered degraded.
2. **Absolute Score:** Absolute capability score on standardized benchmark suites across successive generations ($G_0 \to G_1 \dots$).

### 4. Gate 4 Criterion: Continuous Recursion & The Derivative of Judgment
Verification must track the **derivative of judgment**:
- **Wall-clock cycle latency:** Total elapsed time from initial RFC proposal to verified catalog commit.
- **Autonomous Execution Ratio:** Ratio of machine-executed generation/verification time versus human intervention hours required for peer review and bug remediation.
- Success is measured by declining human intervention hours per verified catalog entry across generations.

---

### Stage 2: Implementation & Claim PR
- Contributor merges scaffolding code, tests, and raw output metrics (`results.edn`) into `proposals/<gate>/<slug>/`.
- **Status in Catalog**: Marked **`ACTIVE (UNVERIFIED)`**.
- **Taint Propagation**: Any downstream experiment declaring this unverified pod as a dependency inherits the `UNVERIFIED` status until the parent is verified.
- **Draft Expiry**: Proposals inactive for 90 days are marked `STALE` and archived without consuming an `eNN` number.

### Stage 3: Independent Verification PR (Minting the `eNN` ID)
- An independent contributor (human or peer agent) replicates the claim on independent silicon.
- **Hardware-Relative Claim-Shape**: Verification tests do not need identical milliseconds on different GPUs. They verify **claim shape**: e.g., was prefill eliminated ($\ge 50\text{ ms}$ saved)? Was handover negligible ($< 2\text{ ms}$)? Did retention floor hold ($\ge 95\%$)?
- **Number Minting**: Only upon merging Stage 3 does the repository merger permanently mint the next sequential `eNN` ID. The pod moves to `catalog/<gate>/eNN-<slug>/` and is recorded in `catalog/registry.edn`.
- *Verification is an independent, credited contribution.* Both the proposer and the verifier receive provenance in the catalog.

---

## Promotion to the Core Library (`src/`)

Code in experiment pods remains isolated. It is promoted into the core library only under **Mechanical-Sympathy Placement**:
- **Dense Contractions & Autoregressive Decoding**: Placed 100% in OpenXLA StableHLO MLIR on accelerator device memory (GPU VRAM). Zero host round-trips.
- **Discrete Index Lookups, Tries & Schema Validation**: Placed on CPU host memory in pure Clojure persistent data structures. The canonical justification is E22: host discrete lookups achieved $42.20\ \mu\text{s}$ vs $691.94\ \mu\text{s}$ on device ($16.4\times$ host advantage).
- **Core Library Layout**:
  - `src/einsum/logic/`: Tensor logic DSL, relational expansions, lowering, autodiff.
  - `src/einsum/compiler/`: Project Panama PJRT C API bindings, StableHLO text generation, caching.
  - `src/einsum/quant/`: EXL3, QuIP, and ternary (CAT-Q) unpacking routines.
  - `src/einsum/models/`: Canonical vehicle model architectures (Gemma 4, Gemma 3, SmolLM, GPT-2).

---

## Testing & CI Invariants

- `clojure -M:format`, `clojure -M:lint`, and `clojure -M:test -m einsum.test-runner` must all pass with zero warnings or errors.
- New behaviors require generative property tests (`clojure.test.check`).
- Staged diffs must be narrow and single-purpose.
