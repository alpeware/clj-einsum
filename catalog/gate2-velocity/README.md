# Gate 2: Velocity & Low-Latency Adaptation (`catalog/gate2-velocity/`)

**Metric**: `Calibration, distillation & adaptation wall-clock time`  
**Target**: $< 30\text{ minutes}$ calibration; $< 2\text{ hours}$ parameter-efficient adaptation on single 24GB GPU  
**Primary Hardware Target**: AMD Radeon RX 7900 XTX (ROCm PJRT)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective

Gate 2 evaluates whether a model generation trains, adapts, or calibrates in **materially less time** ($T_{n+1} \ll T_n$):
- Eliminating multi-week datacenter pre-training runs in favor of rapid, zero-backpropagation calibration and in-VRAM fast weight updates.
- Measuring sub-millisecond execution kernels in OpenXLA StableHLO via Java 25 Panama FFM.

## 2. Historical & Active Pods (Generation $G_1$, E01–E10)

- **E01 (CAMP)**: Cross-Attention Memory Probe — attention concentration in leave-one-out CV.
- **E03**: Sub-second contrastive subspace pre-training on consumer GPU.
- **E04**: In-VRAM Datalog fixpoint state tracker executing multi-hop deduction in sub-2ms PJRT runtime.
- **E05**: Zero-gradient ephemeral online learning via PJRT with instantaneous fact retraction.
- **E06**: Unified TL-Transformer forward pass executing in 4.757 ms.
- **E08**: Dynamic in-VRAM relation induction via StableHLO tensor factorization (1.7 ms kernel).
- **E09**: Native TL-Nano pre-training at 553 tok/s on consumer silicon.
