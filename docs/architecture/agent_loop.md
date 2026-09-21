# AGENT-LOOP.md — Tensor-Native Deliberation & The Meta-Research Cycle

## 1. Goal & Architectural Purpose

This document specifies the **tensor-native deliberation loop** and **autonomous meta-research cycle** in `clj-einsum`.

Rather than treating the agent loop as an unconstrained chat assistant or relying on toy finite-state machines, `clj-einsum` architectures the agent loop as a **rigorous measurement apparatus for Ali Ghodsi's 4 Recursive Self-Improvement (RSI) gates** on consumer silicon (AMD Radeon RX 7900 XTX 24GB / NVIDIA RTX 4090).

The loop is defined by two fundamental engineering principles:
1. **The Propose $\to$ Dispose Contract:** Continuous generative models only propose; discrete host-side verification gates dispose and commit.
2. **Score the Snapshot, Not the Model:** Agent state and long-term memory reside in immutable, verified discrete snapshots verified by physical silicon execution, never in ungrounded token strings or continuous weight drift.

---

## 2. The Propose $\to$ Dispose Contract

The empirical trajectory of Generation $G_2$ (E14–E20) definitively proved that continuous gradient descent and unconstrained autoregressive generation learn geometric ranking manifolds, **not discrete symbolic truth**. When an LLM directly mutates its own memory or state without an external verifier, errors compound exponentially, leading to state amnesia, hallucination laundering, and loop divergence.

To achieve stability across extended multi-turn horizons (Gate 4), the architecture strictly decouples proposal generation from state mutation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            THE PROPOSE ──► DISPOSE CONTRACT                 │
├─────────────────────────────────────────┬───────────────────────────────────┤
│ The Continuous Proposer (Device VRAM)   │ The Discrete Disposer (Host CPU)  │
├─────────────────────────────────────────┼───────────────────────────────────┤
│ - Gemma 4 / Vehicle LLM                 │ - Pure Clojure Validation Core    │
│ - Resident weights, in-graph INT4       │ - Strict Schema & Type Checks     │
│ - Ingests prompt & situation            │ - Generative Invariant Testing    │
│ - Emits candidate AST / code / hypothesis│ - Physical Silicon Execution Parity│
│ - Zero direct state mutation handles    │ - Commits verified snapshot ONLY  │
└─────────────────────────────────────────┴───────────────────────────────────┘
```

### The Invariant Rule
The generative model **never holds a raw write handle** to memory, system state, or the experiment catalog:
$$\text{Snapshot}' = \text{VerifiedCommit}(\text{Snapshot}, \text{Proposals})$$
- If a proposal satisfies all schema rules, semantic constraints, and differential execution tests, it commits to the immutable snapshot.
- If a proposal fails any check, it is rejected back to the model as a structured observation with exact failure diagnostics, enabling self-correction without corrupting memory.

---

## 3. Score the Snapshot, Not the Model

Traditional agent frameworks evaluate "model weights" or "prompt templates," treating agent memory as a lossy text transcript stuffed into quadratic KV context windows. This approach fails Gate 1 (memory explosion) and Gate 4 (drift and catastrophic forgetting).

In `clj-einsum`:
- **Memory is an Immutable Discrete Snapshot:** Memory consists of verified relational assertions, compiled graph executables, and registered metrics.
- **Physical Verification:** A memory entry is not valid because an LLM generated it; it is valid because an executable was compiled to StableHLO MLIR, executed on device silicon, and matched its reference specification.
- **Constant Memory Horizon:** In-VRAM state reduction (E04, E07, E23) maintains an $O(1)$ memory footprint across multi-turn reasoning chains, replacing unbounded quadratic context accumulation.

---

## 4. The Three-Tier Execution Architecture

Deliberation is partitioned into three latency and verification tiers, respecting **mechanical sympathy** between host CPU and accelerator GPU:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Reflex Contractions (OpenXLA StableHLO on Device GPU)               │
│ - Sub-2ms PJRT execution                                                    │
│ - Relational table joins, attention masking, state tracking                 │
│ - 100% deterministic, mathematically sound                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Uncertainty / Routing (Margin < τ)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 2: Resident Deliberation (Gemma 4 in Device VRAM)                      │
│ - Zero weight reload overhead (persistent resident weights)                 │
│ - Prefix-Cache Handover (E24: 0.04ms transition, 74.46ms prefill saved)     │
│ - In-graph token comparison & bounded while loop execution                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Candidate Proposal
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 3: Discrete Synthesis & Commit Gate (Host CPU in Clojure)              │
│ - Mechanical Sympathy: 42.20µs host discrete lookup vs 691.94µs device (E22) │
│ - Pure schema validation, Malli AST type checking, test-runner execution     │
│ - Immutable snapshot update & catalog commit                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Reflex Contractions (Device GPU)
Deterministic tensor contractions compiled via StableHLO MLIR to PJRT executables. Executes state lookups, relational joins, and attention masks in $< 2\text{ ms}$ on consumer GPUs with zero host communication.

### Tier 2: Resident Deliberation (Device VRAM)
When Tier 1 reflex scores fall below a decision margin ($\text{Top1} - \text{Top2} < \tau$), deliberation triggers. Instead of launching a cold inference job:
- The resident 2.3B/4B LLM deliberates directly within the same device VRAM.
- Homogeneous instance-to-instance prefix-cache handover (E24) eliminates 74.46 ms of re-prefill latency, replacing it with a 0.04 ms in-VRAM slice swap ($1861\times$ faster).
- In-graph lexical interrupts detect tool triggers and end-of-turn tokens natively without host regex parsing.

### Tier 3: Discrete Synthesis & Commit Gate (Host CPU)
Candidate proposals emitted by Tier 2 yield to the host CPU. By mechanical sympathy (proven in E22), sparse discrete symbol tables and validation logic execute $16.4\times$ faster on host CPU ($42.20\ \mu\text{s}$) than on device ($691.94\ \mu\text{s}$). The host checks all constraints and commits updates to the snapshot.

---

## 5. The Closed Meta-Research Cycle

The ultimate expression of the agent deliberation loop is the **closed meta-research cycle**, which tests Ghodsi's Gate 4 (Continuous Recursion) on consumer silicon:

```
                  ┌───────────────────────────────┐
                  │          1. PROPOSE           │
                  │ Ingest literature / hypothesis│
                  │   Emit declarative Hiccup AST │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │      2. LOWER & COMPILE       │
                  │ Lower AST to StableHLO SSA IR │
                  │ Compile via Project Panama FFM│
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │       3. VERIFY SILICO        │
                  │ Execute on consumer GPU (ROCm)│
                  │ Evaluate Gates 1, 2, and 3    │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │         4. COMPOUND           │
                  │ Mint eNN ID in registry.edn   │
                  │ Deploy result to Generation   │
                  └───────────────┬───────────────┘
                                  │
                                  └────────► Re-enters Loop at G_{n+1}
```

### Step 1: Propose
The coding agent ingests an external research proposal (e.g. arXiv paper or prior generation anomaly) and reifies it purely into Pedro Domingos' Declarative Tensor Logic:
```clojure
[:= [:out :b :dim] [:x :b :d] [:w :d :dim] {:act :gelu}]
```
Because the AST is homoiconic pure EDN, it can be checked, analyzed, transformed, and validated by the compiler before any execution occurs.

### Step 2: Lower & Compile
The AST lowers directly into a flat SSA EDN graph governed by Malli schemas (`einsum.compiler.stablehlo`). The compiler emits StableHLO MLIR text and compiles it directly into hardware executables via OpenXLA PJRT dynamic plugins using Java 25 Project Panama FFM (`java.lang.foreign`). SHA-256 caching ensures sub-millisecond compilation turnaround on warm graphs.

### Step 3: Verify Silico
The compiled artifact executes on physical consumer silicon (AMD Radeon RX 7900 XTX 24GB or NVIDIA RTX 4090):
- **Gate 1 Check:** Measure peak VRAM bytes allocated, parameter compression (`bytes/param`), and sustained generation throughput ($\text{tok/s}$).
- **Gate 2 Check:** Measure wall-clock calibration and adaptation latency.
- **Gate 3 Check:** Measure capability retention ($\ge 95\%$ floor retention against uncompressed baseline) and benchmark score.
- **Differential Verification:** Verify numerical outputs against the reference Clojure CPU interpreter to catch compiler miscompilations or NaN regressions.

### Step 4: Compound
If and only if all physical verification gates pass:
- An immutable sequential `eNN` identifier is minted.
- The experiment pod is permanently recorded in `catalog/registry.edn` and `catalog/<gate>/<pod>/`.
- The verified capability (e.g., ternary unpack kernel, resident KV cache handover) is deployed into `src/einsum/` to empower the next local agent generation ($G_n \to G_{n+1}$).
- Total human intervention hours are recorded to measure the derivative of judgment.

Through this closed cycle, `clj-einsum` realizes an empirical measurement apparatus that replaces speculation with reproducible, hardware-grounded verification.
