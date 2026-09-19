# Arc 1: Relational Foundations & Contraction Primitives (E01–E10)

**Status**: `CLOSED`  
**Minted IDs**: `e01` through `e10`  
**Primary Substrate**: OpenXLA PJRT (CPU & AMD ROCm)

---

## 1. Scope & Objective
Arc 1 validated the foundational thesis of Pedro Domingos' mapping: function-free Datalog rules can be lowered directly into tensor contractions over embedded relation matrices, executed via OpenXLA PJRT with zero Java runtime overhead.

## 2. Key Findings & Empirical Trajectory
- **E01 (Cross-Attention Memory Probe / CAMP)**: Demonstrated that attention weights reliably concentrate on head entities in leave-one-out CV.
- **E02 (KG-Masked Attention)**: Implemented in-graph relational distractor suppression in StableHLO while preserving autoregressive causal masks.
- **E03 (Contrastive Subspace Pre-training)**: Proved top-3 generalization on unseen entities with sub-second pre-training on consumer GPUs.
- **E04 (In-VRAM Datalog Fixpoint State Tracker)**: Realized zero hallucinations across multi-hop deductive paths in sub-2ms PJRT runtime.
- **E06 (Unified TL-Transformer Layer Block)**: Unified dense transformer layers and relational memory into an end-to-end forward pass (4.757 ms latency on GPU).
- **E07 (SWE-Bench 100-Turn Horizon)**: Confirmed $O(1)$ VRAM state tracking prevents long-horizon amnesia in multi-turn software refactoring tasks.
- **E09 & E10 (TL-Nano Pre-training & CEO Corpus)**: Successfully trained native 2.8M parameter models on consumer silicon, demonstrating verifiable deductive logit boosts ($+0.8900$).

## 3. Paper Export
This arc constitutes the experimental core of **Paper 1: Executable Tensor Logic (Generalized Einsum on Accelerated Substrates)**.
