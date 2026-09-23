# VISION.md — Executable Tensor Logic: The Consumer-Hardware Measurement Apparatus for RSI

## 1. Mission

**Build the open-source measurement apparatus and execution engine to test Ali Ghodsi's 4 conditions for Recursive Self-Improvement (RSI) on consumer hardware (single/dual 24GB GPUs):**

1. **Fewer Resources ($C_{n+1} \ll C_n$):** Super-linear reduction in VRAM footprint, model weight storage, and memory bandwidth.
2. **Reduced Training Time ($T_{n+1} \ll T_n$):** Calibration, parameter-efficient adaptation, and distillation converging in hours on reference consumer hardware, rather than months across datacenters.
3. **Increasing Intelligence ($A_{n+1} > A_n$):** Verifiable reasoning and capability gains evaluated against a strict $\ge 95\%$ uncompressed base model retention floor.
4. **Continuous Recursion ($G_n \to G_{n+1} \to \dots$):** Compounding loop efficiency across successive generations, measured by declining judgment-cost (wall-clock hours and human interventions required per verified catalog entry).

We do not claim that recursive self-improvement is achieved or inevitable. We build the physical apparatus that measures whether its required conditions hold on consumer silicon.

---

## 2. The Deflationary Framing: Apparatus over Hype

In September 2026, Databricks CEO Ali Ghodsi formulated a four-condition test to deflate existential superintelligence hype. Ghodsi pointed out that frontier industrial AI development is currently the **exact opposite of RSI**:
- Costs are scaling super-linearly ($5B–$10B clusters).
- Iteration cycles are growing slower, more complex, and more brittle.
- Infrastructure and energy constraints prevent autonomous recursive acceleration.
- The human-in-the-loop requirement is expanding rather than contracting.

`clj-einsum` adopts this deflationary reality as its foundational premise. **We explicitly do NOT claim the system "has achieved RSI."** Instead, our contribution is the **rigorous, reproducible measurement apparatus**—grounded in OpenXLA StableHLO MLIR, Java 25 Panama FFM, and physical consumer GPUs (AMD Radeon RX 7900 XTX 24GB / NVIDIA RTX 4090)—that measures these four quantities without marketing embellishment or datacenter subsidies.

---

## 3. Arithmetic vs. Hypothesis: The Physics of Consumer Silicon

To avoid the wishful thinking common in neural-symbolic literature, the apparatus maintains a strict boundary between what is mathematically guaranteed (arithmetic) and what must be verified empirically on hardware (hypothesis):

### Storage & Bandwidth: Arithmetic
- Uncompressed 16-bit floats require $2.0\text{ bytes/param}$. A 30B model requires $\approx 60\text{ GB}$ VRAM; a 70B model requires $\approx 140\text{ GB}$ VRAM. Neither fits on consumer silicon.
- 1.58-bit / ternary quantization ($\{-1, 0, 1\}$) reduces weight representation to $\approx 0.2\text{ to } 0.25\text{ bytes/param}$ ($1.58\text{ bits} + \text{scales}$).
- **The Arithmetic Win:** A 30B parameter model compresses to $\approx 7.5\text{ GB}$ VRAM; a 70B parameter model compresses to $\approx 17.5\text{ GB}$ VRAM. Both fit entirely within a single 24GB consumer GPU with generous headroom for KV caches. This storage reduction is mathematical certainty, not a research speculation.

### Compute & Throughput: Hypothesis
- Unpacking ternary weights into FP16 or BF16 registers via StableHLO unpack-and-GEMM reduces memory bandwidth pressure during autoregressive decoding.
- **The Empirical Reality:** Unpack-and-GEMM still executes floating-point matrix multiplications on hardware tensor cores. True ternary integer-addition speedups ($\text{GEMM} \to \text{ADD}$) require specialized fused native kernels and memory architectures.
- **The Measurement Obligation:** The framework never assumes compute speedups by default. It measures achieved token throughput ($\text{tok/s}$) and GFLOPs/watt against optimized FP16 and INT4 (EXL3) baselines on actual hardware.

---

## 4. What the Representation Covers — and What It Doesn't

Explicit, because coding agents and peer reviewers must know the boundaries before trusting results.

**Covered — the parts that work:**
- **Tensor contraction (einsum):** The shared mathematical core of deep learning: attention, multi-layer perceptrons, convolutions, embedding lookups, and relational joins. One construct, `[:= head & body-terms]`; a shared index contracted on both sides *is* the matmul, the relational join, the attention score.
- **Function-free Datalog-style rules:** Compiled to contractions over embedded relations. Lowered to StableHLO without runtime interpretation.
- **Elementwise nonlinearities and autodiff closure:** Activations are attributes on equations (`{:act :gelu}`); the gradient of a tensor-logic program is itself a pure tensor-logic program.
- **Bounded, statically-shaped dataflow:** Exact compilation via OpenXLA PJRT with a differential reference interpreter checking device correctness.

**Not covered — and not claimed:**
- Full first-order logic with function symbols, modal logic, or unconstrained probabilistic programming.
- *Absolute* soundness in continuous embedding space. Continuous representations yield PAC-style margins, not symbolic certainty.
- Gradient-discovered discrete predicates. Foundational negative result (synthesized in Paper 2): continuous gradients learn geometric rankings, not discrete selection. Predicate invention requires exact discrete priors and verified commit machinery living *around* the tensor core.

---

## 5. System Architecture: Pure Data to Accelerated Silicon

`clj-einsum` is a vertically integrated toolchain in Clojure with zero Java escape hatches and zero host Python dependencies:

```
tensor equations (Hiccup AST, pure data)
        │  expand · lower · autodiff — all as pure data transforms
        ▼
EDN SSA graph IR (Malli-schematized, verifiable)
        │
        ▼
StableHLO MLIR Text → OpenXLA PJRT Executables (ROCm / CUDA / CPU)
        │
        ▼
Physical Consumer Accelerator (AMD RX 7900 XTX 24GB / RTX 4090)
        │
        ▼
reference interpreter ⇄ physical device differential verification
```

- **`einsum.logic.*` (The Language):** AST expansion, lowering, symbolic relations, relational memory, and agent deliberation loops.
- **`einsum.compiler.*` (The Compiler):** Java 25 Project Panama FFM bindings to OpenXLA PJRT C API (`pjrt_c_api.h`), StableHLO MLIR codegen, SHA-256 graph caching, and off-heap memory arena lifecycle management.
- **`einsum.quant.*` (The Quantization Engine):** EXL3, QuIP, and ternary (CAT-Q) unpacking routines executing directly in StableHLO graphs.
- **`einsum.models.*` (Vehicle Architectures):** Canonical, pure Clojure implementations of Gemma 4, Gemma 3, SmolLM, and GPT-2 running on persistent VRAM buffers.

---

## 6. The 4 Research Pillars

- **Pillar 0 — The Compiler (Built):** Tensor Logic AST $\to$ StableHLO $\to$ PJRT, validated via differential testing. Documented in Paper 1 (*Executable Tensor Logic*).
- **Pillar 1 — Trainable Tensor Logic & The Continuous Refutation (Complete):** Structure learning on real-world corpora (WebNLG). Documented in Paper 2 (*The Limits of Trainable Tensor Logic*), establishing that continuous relaxation cannot clear Gate 3 for predicate invention without discrete commitment machinery.
- **Pillar 2 — The Agent Deliberation Loop (Built):** A three-tier agent loop (reflex contractions, device-resident deliberation, discrete host-side synthesis) where the agent's memory snapshot is verified by discrete schema checks, not raw token generation.
- **Pillar 3 — The RSI Measurement Apparatus (Active):** The automated closed loop testing Ghodsi's 4 gates on consumer silicon across generation lineage ($G_0 \to G_1 \to \dots$).

---

## 7. The Recursive Meta-Research Loop

The meta-research loop elevates the coding agent from executing predefined tasks to running autonomous, hardware-grounded scientific inquiry:

1. **Papers as Proposals:** The external literature (arXiv) is a noisy proposal distribution over model architectures.
2. **Homoiconic Reification:** Proposals are translated into declarative Clojure Hiccup ASTs. Because every component speaks the exact same representation, disparate techniques (e.g. in-graph dispatch, state reduction, prefix cache handover) compose cleanly without glue code.
3. **Hardware as the Incorruptible Verifier:** Executables compile to StableHLO and run on physical consumer GPUs. Claims must survive contact with silicon.
4. **Adversarial Critique & Verification:** Automated checks verify floor retention ($\ge 95\%$), test for confounded baselines, and measure achieved memory and throughput.
5. **The Immutable Catalog:** Verified pods commit to `resources/catalog/registry.edn` and `resources/catalog/gate*/<pod>/`. Subsequent generations read the catalog to avoid refuted lines and build on verified components.

---

## 8. Operating Principles

- **Ground Every Number:** Reference simulation, unquantized baseline, and on-device measurements must be explicitly labeled. No claims without reproducible metrics.
- **Verify on Silicon:** A claim about performance, latency, or accuracy is unverified until executed on physical accelerator hardware via PJRT.
- **Respect Mechanical Sympathy:** Place dense continuous math in device VRAM; place sparse discrete lookups and schema verifications in host CPU memory (justified by the host discrete lookup advantage: $16.4\times$ faster than device lookup).
- **The Derivative of Judgment:** The ultimate test of Gate 4 is whether the wall-clock time from proposal to verified merge and the ratio of human intervention decline across generations.
