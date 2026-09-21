# Arc 1: Relational Foundations & Contraction Primitives (E01–E10)

**Generation**: $G_1$ (Relational Foundations)  
**Status**: `CLOSED`  
**Minted IDs**: `e01` through `e10`  
**Primary Substrate**: OpenXLA PJRT (CPU & AMD ROCm)  
**RSI Gates Addressed**: Gate 1 (Resource), Gate 2 (Time), Gate 3 (Capability), Gate 4 (Repeatability)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective
Arc 1 established the computational substrate for Generation $G_1$, validating Pedro Domingos' mapping: function-free Datalog rules lower directly into tensor contractions over embedded relation matrices, executed via OpenXLA PJRT with zero Java runtime overhead.

## 2. RSI Gate Evaluations & Key Findings
- **Gate 1 (Resource Efficiency):**
  - **E04 & E07**: Realized $O(1)$ constant-memory fixpoint state tracking on device, replacing quadratic KV context accumulation in long-horizon tasks (SWE-Bench 100-Turn challenge).
  - **E02 & E05**: Ephemeral zero-gradient learning and in-graph distractor suppression eliminating full parameter fine-tuning.
- **Gate 2 (Time Efficiency):**
  - **E04**: Sub-2ms PJRT fixpoint execution for multi-hop relational deduction.
  - **E06**: Unified TL-Transformer forward pass executing in 4.757 ms on consumer GPU.
- **Gate 3 (Capability Improvement):**
  - **E01 (CAMP)**: Attention weights concentrate on ground-truth head entities in leave-one-out CV.
  - **E03**: Contrastive subspace pre-training achieving Top-3 generalization on unseen entities.
  - **E10**: Real-world CEO corpus cloze QA demonstrating verifiable deductive logit boost ($+0.8900$).
- **Gate 4 (Repeatability / Closed Loop):**
  - **E07**: Multi-turn state tracking proved completely immune to memory saturation or catastrophic drift across 100-turn sequences.

## 3. Paper Export
This arc constitutes the experimental core of **Paper 1: Executable Tensor Logic (Generalized Einsum on Accelerated Substrates)**.
