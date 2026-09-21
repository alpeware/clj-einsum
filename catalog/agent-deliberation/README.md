# Arc 3: In-VRAM Autonomous Agent Deliberation & State Reduction (E21–E24)

**Generation**: $G_3$ (In-VRAM Autonomous Deliberation & Handover Apparatus)  
**Status**: `ACTIVE`  
**Minted IDs**: `e21` through `e24`  
**Primary Substrate**: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100, ROCm 6.2, `libjsig.so`)  
**RSI Gates Addressed**: Gate 1 (Resource), Gate 2 (Time), Gate 3 (Capability), Gate 4 (Repeatability)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective
Arc 3 builds the operational deliberative apparatus for Generation $G_3$ (`AGENT-LOOP.md`). Overcoming the continuous-relaxation failure of Generation $G_2$, it couples dense in-VRAM tensor execution with discrete host-side commitment gates and homogeneous instance-to-instance prefix caching on physical silicon.

## 2. RSI Gate Evaluations & Key Findings
- **Gate 1 (Resource Efficiency):**
  - **E24 (Zero Weight Duplication)**: Allocates **0 duplicate model weights** across 493 PJRT buffers during multi-instance deliberation, keeping a persistent 2.3B parameter footprint pinned resident in device VRAM.
- **Gate 2 (Time Efficiency):**
  - **E24 (Prefill Elimination)**: Replaces 74.46 ms of re-prefill latency with a **0.04 ms in-VRAM replacement transition** ($1861\times$ faster).
  - **E22 (In-Graph Query Dispatch)**: Achieves $100\%$ tool-token emission with Gemma 4 E2B, while proving a $16.4\times$ host advantage ($42.20\ \mu\text{s}$ host vs $691.94\ \mu\text{s}$ device) for sparse discrete routing, establishing the mechanical-sympathy placement rule.
- **Gate 3 (Capability Improvement):**
  - **E21 & E23 (Zero-Violation Invariants)**: The schema-constrained write path and reduce loop guarantee **0 illegal role assignments or RBAC violations** committed to long-term memory over adversarial horizons.
  - **E24 (Emergent Self-Correction)**: Demonstrates an emergent $+3.3\%$ accuracy lift on reasoning benchmarks (`arith-07`) through zero-overhead second-opinion deliberation.
- **Gate 4 (Repeatability / Closed Loop):**
  - **E23 & E24**: Proved that state accumulation does not drift, saturate, or degrade across extended multi-turn agent deliberations.

## 3. Contained Experiment Pods
- [`e21-kb-write-path/`](e21-kb-write-path/)
- [`e22-ingraph-dispatch/`](e22-ingraph-dispatch/)
- [`e23-state-reduce/`](e23-state-reduce/)
- [`e24-prefix-cache-handover/`](e24-prefix-cache-handover/)
