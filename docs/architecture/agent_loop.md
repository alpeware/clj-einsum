# The Tensor-Native Autonomous Coding Agent Loop

> **Operational Status:** Production-Ready & Verified in Silicon (`einsum.logic.agent.*`, `einsum.logic.memory.*`).
> **Core Invariant:** Relational memory lives directly inside device VRAM. Discrete synthesis and schema gates live host-side in pure Clojure.

---

## 1. Executive Summary & Problem Formulation

Standard autonomous coding agents (AutoGPT, SWE-agent, Devin-style architectures) suffer from three fatal scaling limits when executing continuous multi-turn development loops:

1. **Context Window Saturation ($O(N^2)$ Memory Explosion):** Every file inspection, test output, and intermediate thought is concatenated into the prompt. Context length explodes quadratically, driving latency to tens of seconds per step and exceeding VRAM limits.
2. **Context Fragmentation & Hallucinatory Drift:** Because text context is unstructured, intermediate hypotheses conflict with ground truth. The model forgets constraints established 10 turns prior.
3. **No Direct Grounding:** The model cannot directly "query" its own internal state or intermediate representations without paying full autoregressive decode costs on every query.

`clj-einsum` resolves this by unifying Pedro Domingos' Declarative Tensor Logic with an in-tensor relational memory core executing inside device VRAM.

---

## 2. Theoretical Grounding: In-Tensor Relational Memory

In `clj-einsum`, knowledge is not maintained as a text transcript. Instead, knowledge is encoded as a set of continuous relation matrices resident in device VRAM:

$$M_r \in \mathbb{R}^{D \times D}$$

Where each relation $r$ (e.g. `:calls`, `:imports`, `:fails-test`, `:implements`) is a superposition of outer products between head entity embeddings $e_h$ and tail entity embeddings $e_t$:

$$M_r = \sum_{(h, t) \in \mathcal{E}_r} e_h \otimes e_t$$

Querying whether entity $h$ has relation $r$ to entity $t$ is a single bilinear tensor contraction:

$$\text{score}(h, r, t) = e_h^T M_r e_t$$

Because $M_r$ is a fixed $D \times D$ matrix, **associative retrieval latency is $O(1)$ with respect to the number of stored facts**, executing in sub-millisecond tensor contractions without autoregressive generation.

---

## 3. Score the Snapshot, Not the Model

Traditional agent frameworks evaluate "model weights" or "prompt templates," treating agent memory as a lossy text transcript stuffed into quadratic KV context windows. This approach fails Gate 1 (memory explosion) and Gate 4 (drift and catastrophic forgetting).

In `clj-einsum`:
- **Memory is an Immutable Discrete Snapshot:** Memory consists of verified relational assertions, compiled graph executables, and registered metrics.
- **Physical Verification:** A memory entry is not valid because an LLM generated it; it is valid because an executable was compiled to a pure-Clojure UOp schedule, executed on device silicon, and matched its reference specification.
- **Constant Memory Horizon:** In-VRAM state reduction maintains an $O(1)$ memory footprint across multi-turn reasoning chains, replacing unbounded quadratic context accumulation.

---

## 4. The Three-Tier Execution Architecture

Deliberation is partitioned into three latency and verification tiers, respecting **mechanical sympathy** between host CPU and accelerator GPU:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Reflex Contractions (Pure-Clojure UOp Schedule on Device GPU)       │
│ - Sub-2ms Panama FFM native execution                                       │
│ - Relational table joins, attention masking, state tracking                 │
│ - 100% deterministic, mathematically sound                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Uncertainty / Routing (Margin < τ)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier 2: Resident Deliberation (Gemma 4 in Device VRAM)                      │
│ - Zero weight reload overhead (persistent resident weights)                 │
│ - Ping-Pong KV cache buffer reuse (sub-millisecond step transitions)        │
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
Deterministic tensor contractions compiled via pure-Clojure UOp schedules to native HIP kernels. Executes state lookups, relational joins, and attention masks in $< 2\text{ ms}$ on consumer GPUs with zero host communication.

### Tier 2: Resident Deliberation (Device VRAM)
When Tier 1 reflex scores fall below a decision margin ($\text{Top1} - \text{Top2} < \tau$), deliberation triggers. Instead of launching a cold inference job:
- The resident LLM deliberates directly within the same device VRAM.
- Static-shape ping-pong KV cache buffers eliminate re-allocation latency and avoid GC pressure.
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
                  │ Lower AST to Pure UOp DAG     │
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
The AST lowers directly into a canonical UOp DAG governed by immutable Clojure data structures (`einsum.uop.lower`). The compiler partitions schedules, rangeifies coordinates, optimizes Wave32 LDS/WMMA tiles, renders native HIP C++ code, and executes directly on accelerator hardware via Project Panama FFM (`java.lang.foreign` to `libamdhip64.so` / `libhiprtc.so`). SHA-256 binary caching in `.uop-cache/` ensures sub-millisecond compilation turnaround on warm graphs.

### Step 3: Verify Silico
The compiled artifact executes on physical consumer silicon (AMD Radeon RX 7900 XTX 24GB):
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
