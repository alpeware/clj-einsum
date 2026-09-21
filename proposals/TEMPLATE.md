# RFC: <Experiment Title>

**Experiment**: `<slug>`  
**Gate**: `<gate1-compression | gate2-velocity | gate3-evals | gate4-recursion>`  
**Generation**: `<target-generation-number, e.g. 1>`  
**Literature**: `["<Author et al. (Year) arXiv:XXXX.XXXXX>"]`  
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB)" :claim-shape "<e.g. >= 7.5x VRAM compression, zero NaN>"}`  
**Extends**: `"<prior-eNN-or-nil>"`  
**Refutes**: `"<refuted-claim-or-nil>"`  
**Supersedes**: `"<superseded-eNN-or-nil>"`  
**Reopens**: `"<closed-arc-if-applicable>"`  

---

## 1. Abstract & Motivation
*Briefly describe the proposed architectural modification, compression scheme, or deliberation mechanism. Explain why this technique is expected to advance the target generation without falling prey to ungrounded speculation.*

---

## 2. Pre-Registered Hypotheses & Gate Criteria

Every proposal must explicitly state falsifiable criteria across Ghodsi's 4 RSI Gates:

### Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$)
- **Storage / Memory (Arithmetic)**:
  - Target `bytes/param`: `<e.g. <= 0.25 bytes/param>`
  - Peak VRAM footprint: `<e.g. <= 8.0 GB for 30B model>`
- **Compute / Throughput (Hypothesis)**:
  - Baseline throughput (FP16 / INT4): `<e.g. XX tok/s>`
  - Target achieved throughput: `<e.g. >= YY tok/s>`
  - Falsification condition: `<e.g. Unpack overhead reduces decode throughput below INT4 baseline>`

### Gate 2: Time Efficiency ($T_{n+1} \ll T_n$)
- **Wall-Clock Latency**:
  - Target calibration / adaptation duration: `<e.g. < 30 minutes on single 24GB GPU>`
  - Training FLOPs / compute budget: `<e.g. Zero backpropagation, forward-only Hessians>`
  - Falsification condition: `<e.g. Calibration requires > 2 hours or exceeds 24GB memory ceiling>`

### Gate 3: Intelligence & Capability Floor ($A_{n+1} > A_n$)
- **Floor Retention ($\ge 95\%$)**:
  - Target retention percentage: `<e.g. >= 95% of uncompressed base model perplexity on WikiText-2>`
- **Absolute Capability Score**:
  - Benchmark suite: `<e.g. GSM8K / HumanEval-Clojure>`
  - Baseline score: `<e.g. 0.56>`
  - Target score: `<e.g. >= 0.54 (within 95% floor)>`
  - Falsification condition: `<e.g. Catastrophic accuracy collapse below 95% floor>`

### Gate 4: Continuous Recursion & The Compound Derivative
- **The Derivative of Judgment**:
  - Estimated proposal-to-merge cycle time: `<e.g. < 6.0 hours>`
  - Estimated human intervention: `<e.g. < 1.0 human hours for review>`
  - Deployment target: `<e.g. Deploys as resident vehicle model for Generation G1 agent loop>`

---

## 3. Mathematical Specification (Declarative Tensor Logic AST)

Declare the proposed transformation purely as Pedro Domingos' Declarative Tensor Logic equations:

```clojure
;; Example: Quantized Unpack-and-Contract AST
[:= [:out :b :dim]
    [:x :b :d]
    [:w_unpacked :d :dim]
    {:act :identity}]
```

---

## 4. Execution Harness & Silicon Verification Plan

- **Harness Namespace**: `tools.<slug>` or `einsum.experiments.<slug>.run`
- **Output Artifacts**: `results.edn`, `summary.csv`
- **Differential Parity**: Output must match reference Clojure CPU interpreter within epsilon $\le 10^{-3}$.
