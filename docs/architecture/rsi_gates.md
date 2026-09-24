# Ali Ghodsi's 4 Recursive Self-Improvement (RSI) Gates on Consumer Silicon

## 1. Executive Summary & Foundational Framing

In September 2026, Databricks CEO Ali Ghodsi proposed a rigorous four-condition test to evaluate whether **Recursive Self-Improvement (RSI)** in artificial intelligence is an empirical reality or an ungrounded speculative claim. For a system to exhibit genuine recursive self-improvement—where each generation $G_{n+1}$ autonomously refines itself beyond $G_n$—the following four gates must hold **simultaneously**:

1. **Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$):** The next generation must require materially (super-linearly) fewer computational resources, VRAM bytes, or energy than its predecessor.
2. **Gate 2: Time Efficiency ($T_{n+1} \ll T_n$):** The next generation must train, adapt, or deliberate in materially less latency.
3. **Gate 3: Capability Improvement ($A_{n+1} > A_n$):** The next generation must demonstrate strictly higher intelligence, accuracy, or verified invariant preservation.
4. **Gate 4: Repeatability / Closed Loop ($G_n \to G_{n+1} \to G_{n+2}$):** The process must operate continuously in an autonomous, self-reinforcing loop without catastrophic degradation, human intervention, or amnesia over extended horizons.

Ghodsi observed that frontier industrial AI development exhibits the exact opposite: costs scale to billions of dollars ($C_{n+1} \gg C_n$), training runs grow longer and more brittle ($T_{n+1} \gg T_n$), capability gains require exponentially more compute, and closed-loop recursion collapses under model collapse or ungrounded hallucinations.

`clj-einsum` exists as a **hardware-grounded measurement apparatus** to determine whether these four gates can be cleared on **consumer silicon** (AMD Radeon RX 7900 XTX 24GB, RDNA3 gfx1100, OpenXLA PJRT).

---

## 2. The Consumer Silicon Crucible

The true test of recursive self-improvement is whether it can operate within hard physical constraints. Achieving marginal accuracy gains by throwing 100,000 datacenter GPUs at a problem is brute-force compute scaling, not algorithmic recursion.

Consumer silicon imposes non-negotiable physical invariants:
- **VRAM Hard Ceiling:** 24 GB total unified memory on AMD Radeon RX 7900 XTX.
- **RDNA3 Local Data Share (LDS) Limit:** Exactly 64 KB (65,536 bytes) shared memory per workgroup. Kernel compilation fails (`requested 98304, available 65536`) if attention state structures are not statically planned.
- **Host-to-Device (H2D) Bandwidth:** PCIe 4.0 x16 (~31.5 GB/s theoretical, ~25 GB/s practical). Any design that shuttles large intermediate activation tensors between host and device incurs lethal latency penalties.
- **Pure StableHLO Compilation:** All mathematical transformations must compile through OpenXLA PJRT into fused machine code. No Java primitive loops or host Python runtimes in the hot path.

Every claim in `clj-einsum` is measured directly against this hardware apparatus.

---

## 3. The 4 Gates Formalized

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ALi GHODSI'S 4 RSI VERIFICATION GATES                 │
├────────────────────────────────┬────────────────────────────────────────────┤
│ Gate 1: Resource Efficiency    │ Gate 2: Time Efficiency                    │
│   C_{n+1} ≪ C_n                │   T_{n+1} ≪ T_n                            │
│   - Zero duplicate weights     │   - In-graph dispatch (42µs host / 692µs)  │
│   - Prefix cache handover      │   - Prefill elimination (74.46ms saved)    │
│   - O(1) state memory          │   - 0.04ms replacement handover            │
├────────────────────────────────┼────────────────────────────────────────────┤
│ Gate 3: Capability Improvement │ Gate 4: Repeatability / Closed Loop        │
│   A_{n+1} > A_n                │   G_n → G_{n+1} → G_{n+2}                  │
│   - Zero RBAC violations       │   - Adversarial horizon reduction          │
│   - Verified self-correction   │   - Immutable machine-readable catalog     │
│   - +3.3% reasoning lift       │   - Homoiconic AST meta-synthesis          │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$)
- **Formal Metric:** Memory footprint $\mathcal{M}(G_{n+1}) \le \alpha \mathcal{M}(G_n)$ where $\alpha < 1$, and compute operations per inference turn $\mathcal{F}(G_{n+1}) \ll \mathcal{F}(G_n)$.
- **Empirical Apparatus:**
  - **E24 (In-VRAM Prefix-Cache Handover):** Realizes KV-aligned homogeneous instance handover in device VRAM. Allocates **0 duplicate model weights** across 493 PJRT device buffers, maintaining a persistent 2.3B parameter footprint resident in VRAM.
  - **E04 / E07 (In-VRAM Fixpoint State Tracker):** $O(1)$ constant memory state tracking replaces quadratic KV context accumulation during multi-turn agent deliberations.

### Gate 2: Time Efficiency ($T_{n+1} \ll T_n$)
- **Formal Metric:** Latency per reasoning step $\mathcal{T}(G_{n+1}) \ll \mathcal{T}(G_n)$.
- **Empirical Apparatus:**
  - **E24 (Prefill Elimination):** Eliminates 74.46 ms of re-prefill computation by directly handing over resident KV cache slices in VRAM, replacing it with a 0.04 ms pointer transition ($1861\times$ faster).
  - **E22 (In-Graph KB Query Dispatch):** Measures discrete table lookup vs device dispatch. Demonstrates $42.20\ \mu\text{s}$ host latency vs $691.94\ \mu\text{s}$ device dispatch ($16.4\times$ host advantage for sparse discrete lookups), establishing the mechanical-sympathy placement rule.

### Gate 3: Capability Improvement ($A_{n+1} > A_n$)
- **Formal Metric:** Accuracy on target benchmark $\mathcal{A}(G_{n+1}) > \mathcal{A}(G_n)$, and safety violation rate $\mathcal{V}(G_{n+1}) = 0$.
- **Empirical Apparatus:**
  - **E21 & E23 (Schema-Constrained KB & Adversarial Reduce):** Evaluated over a 10-turn adversarial degradation horizon with conflicting role assignments. Achieves **0 RBAC invariant violations** committed to memory.
  - **E24 (Emergent Self-Correction):** Evaluated on a 30-problem reasoning benchmark (`arith-07`). Verification handover unlocks an emergent $+3.3\%$ accuracy lift via zero-overhead second-opinion deliberation.
  - **E14–E20 (Continuous Relaxation Refutation):** Systematically tested continuous gradient descent for predicate discovery. Proved that gradients learn geometric rankings, not discrete selection ($1.0\%$ Top-1 accuracy across distractor pools). Proves that Gate 3 cannot be cleared via continuous relaxation alone; strong discrete commitment priors are required.

### Gate 4: Repeatability / Closed Loop ($G_n \to G_{n+1} \to G_{n+2}$)
- **Formal Metric:** Stability across recursive generations $\lim_{n \to \infty} \text{Degradation}(G_n) = 0$.
- **Empirical Apparatus:**
  - **E23 (State Reduction):** Formally proved that state superposition does not saturate or diverge across multi-hop reasoning chains.
  - **The Meta-Research Loop:** The repo itself functions as an automated loop where paper hypotheses are reified into homoiconic einsum equations, executed on physical silicon, checked by adversarial critics, and registered into an immutable catalog.

---

## 4. Generation Lineage: The Staircase Roadmap

The research trajectory of `clj-einsum` maps directly to the Staircase roadmap defined in [`PROCESS.md`](../../PROCESS.md):

```
[1. Base Language Models] ──> [2. Chain of Thought] ──> [3. Coding Agents] ──> [4. Continuous Learning] ──> [5. Self-Iteration]
    (Open Weights Vehicle)        (Reasoning Ceiling)       (Current Arena)         (Memory + Commit Gates)     (Compounding Loop)
```

1. **Stage 1 (Base Models)**: We ride open weights (Gemma 4, SmolLM, GPT-2). We do not train foundation models.
2. **Stage 2 (Chain of Thought)**: Raises the reasoning ceiling. Evaluated on Gate 3 retention.
3. **Stage 3 (Agents — Current Arena)**: The proving ground. Multi-step reasoning and verification against compilers and test suites. All gates (G1–G4) active.
4. **Stage 4 (Continuous Learning — The Lottery Ticket)**: Knowledge updates without retraining. Memory module + schema-constrained commit path. Evaluated on G2 (adaptation velocity) and G4 (declining judgment cost).
5. **Stage 5 (Self-Iteration)**: The loop improving the loop. Measured by the compound derivative of verified catalog output per unit human judgment.

---

## 5. The Closed Meta-Research Loop

The meta-research loop elevates the agent loop from executing tasks to executing scientific inquiry:

```
                  ┌───────────────────────────────┐
                  │    1. Literature Proposal     │
                  │   (arXiv / External Thesis)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 2. Homoiconic Reification     │
                  │    ([:= head & body-terms])   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 3. Hardware Compilation       │
                  │  (StableHLO MLIR -> PJRT ROCm)│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 4. Adversarial Review Gate    │
                  │ (Differential Parity / Tests) │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 5. Immutable Catalog Commit   │
                  │ (registry.edn / results.edn)  │
                  └───────────────┬───────────────┘
                                  │
                                  └────────► (Advances Generation G_n -> G_{n+1})
```

1. **Literature as Proposal Distribution:** Papers in the literature are treated as noisy proposals over model architectures.
2. **Einsum as Lingua Franca:** The proposal is rewritten into pure Clojure Hiccup AST. Because every component speaks the exact same homoiconic language, architectures from different papers (e.g. E22 dispatch + E23 reduce + E24 cache handover) compose without impedance mismatches.
3. **Hardware as the Incorruptible Verifier:** Executables compile to StableHLO and run on discrete accelerator hardware. Claims must survive physical execution on the device.
4. **Adversarial Critique:** The reference interpreter verifies mathematical parity; rigorous generative tests (`clojure.test.check`) test corner cases; automated review checks for confounded baselines or overclaimed ratios.
5. **Catalog as Cumulative Memory:** Validated outputs commit to `resources/catalog/registry.edn` and `resources/catalog/<gate>/<slug>/`. Subsequent iterations read the catalog to avoid closed lines and remix verified components.

Through this discipline, `clj-einsum` converts speculative claims of recursive self-improvement into an empirical science grounded in hardware measurement.
