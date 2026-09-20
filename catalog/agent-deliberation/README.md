# Arc 3: In-VRAM Autonomous Agent Deliberation & State Reduction (E21–E24)

**Status**: `ACTIVE`  
**Minted IDs**: `e21` through `e24`  
**Primary Substrate**: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100, ROCm 6.2, `libjsig.so`)

---

## 1. Scope & Objective
Arc 3 builds the operational read/write memory and deliberation substrate for autonomous coding agents (`AGENT-LOOP.md`). Instead of relying on gradient descent for discrete truth, it couples continuous in-VRAM contractions with discrete host-side commitment gates and in-VRAM multi-instance prefix caching.

## 2. Key Findings & Empirical Trajectory
- **E21 (Schema-Constrained KB Write Path)**: Built the sans-IO verified commit pipeline, guaranteeing 0 illegal role assignments or contradictory facts commit to long-term memory.
- **E22 (In-Graph KB Query Dispatch)**: Demonstrated 100% emission of tool-call tokens in Gemma 4 E2B, while proving a $16.4\times$ host latency advantage ($42.20\ \mu\text{s}$ host vs $691.94\ \mu\text{s}$ device) for sparse discrete table routing (establishing the mechanical-sympathy placement rule).
- **E23 (Verified KB Accumulation Over Horizon / RBAC Reduce)**: Proved mathematical zero-violation accumulation over an adversarial horizon, preventing model degradation.
- **E24 (In-VRAM Prefix-Cache Handover)**: Realized homogeneous instance-to-instance handover in device memory. Eliminated 74.46 ms of host re-prefill (0.04 ms replacement handover), proved 96.7% semantic parity, and demonstrated emergent verifier self-correction (+3.3% lift on `arith-07`) with 0 duplicate weights allocated across 493 PJRT buffers.

## 3. Contained Experiment Pods
- [`e21-kb-write-path/`](e21-kb-write-path)
- [`e22-ingraph-dispatch/`](e22-ingraph-dispatch)
- [`e23-state-reduce/`](e23-state-reduce)
- [`e24-prefix-cache-handover/`](e24-prefix-cache-handover)
