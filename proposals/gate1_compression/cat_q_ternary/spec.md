# RFC: Cost-Efficient Accurate Ternary Quantization (CAT-Q / 1.58b) for 30B–70B Models in PJRT

**Experiment**: `cat-q-ternary`  
**Gate**: `gate1_compression`  
**Generation**: `1`  
**Literature**: `["Wang et al. (2024) The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (arXiv:2402.17764)", "Frantar et al. (2022) GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (arXiv:2210.17323)"]`  
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB) via ROCm 6.2 PJRT" :claim-shape ">= 7.5x VRAM compression (<= 0.25 bytes/param), >= 95% perplexity retention, zero backprop calibration < 30min"}`  
**Extends**: `nil`  
**Refutes**: `nil`  
**Supersedes**: `nil`  
**Reopens**: `nil`  

---

## 1. Abstract & Motivation

State-of-the-art open models with 30B to 70B parameters (e.g., LLaMA-3 70B, Gemma 2 27B, Qwen 2.5 32B/72B) require between $60\text{ GB}$ and $140\text{ GB}$ of VRAM at 16-bit precision, completely excluding them from single-GPU consumer hardware (24GB VRAM ceiling). 

Cost-Efficient Accurate Ternary Quantization (CAT-Q / 1.58-bit) constrains all model linear weight matrices to the ternary alphabet $W \in \{-1, 0, 1\}$, scaled by per-channel or per-block floating-point factors $\gamma \in \mathbb{R}$. This compresses raw parameters to $\approx 1.58\text{ bits}$ (packed into 2-bit bitfields or base-3 Huffman blocks), achieving an effective $\approx 0.20\text{ to } 0.25\text{ bytes/param}$.

This RFC proposes the implementation and empirical validation of ternary weight unpacking and execution directly inside OpenXLA StableHLO MLIR, establishing the foundation for **Generation 1 ($G_1$)** of the `clj-einsum` measurement apparatus.

---

## 2. Pre-Registered Hypotheses & Gate Criteria

### Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$)
- **Storage / VRAM Footprint (Arithmetic)**:
  - Target: $\le 0.25\text{ bytes/param}$ ($\ge 7.5\times$ VRAM reduction compared to uncompressed FP16 at $2.0\text{ bytes/param}$).
  - A 30B parameter model must occupy $\le 7.5\text{ GB}$ of device VRAM; a 70B model must occupy $\le 17.5\text{ GB}$ of device VRAM. Both must leave $\ge 6.5\text{ GB}$ of headroom for KV-cache state and execution workspaces on a single 24GB GPU.
- **Compute Throughput (Hypothesis)**:
  - StableHLO unpack-and-GEMM will reduce memory bandwidth pressure during autoregressive token generation.
  - Achieved token throughput ($\text{tok/s}$) will be measured against uncompressed FP16 and EXL3 (3-bit) baselines on AMD Radeon RX 7900 XTX.
  - **Falsification Condition**: If the instruction overhead of in-graph bit unpacking exceeds memory bandwidth savings, reducing decoding throughput below INT4 baselines, the compute hypothesis is refuted.

### Gate 2: Time Efficiency ($T_{n+1} \ll T_n$)
- **Unsupervised Calibration Latency**:
  - Target: Second-order Hessian calibration across 128 calibration sequences must complete in $< 30\text{ minutes}$ on a single AMD RX 7900 XTX.
  - Compute Discipline: Forward-only activations and Cholesky updates; **zero backpropagation**.
  - **Falsification Condition**: Calibration time exceeding 60 minutes or exceeding the 24GB VRAM workspace ceiling.

### Gate 3: Intelligence & Capability Floor ($A_{n+1} > A_n$)
- **Dual Evaluation Criteria**:
  1. **Perplexity Floor Retention ($\ge 95\%$)**: Perplexity on WikiText-2 and C4 test splits must retain $\ge 95\%$ of the uncompressed FP16 base model score ($\frac{\text{PPL}_{\text{FP16}}}{\text{PPL}_{\text{Ternary}}} \ge 0.95$).
  2. **Clojure Code Generation Floor**: Zero syntax errors or macro expansion regressions on HumanEval/Clojure benchmark suite.
  - **Falsification Condition**: Catastrophic collapse on multi-hop reasoning or AST structure generation ($< 95\%$ retention).

### Gate 4: Continuous Recursion & The Compound Derivative
- **Compounding Measurement**:
  - Successfully deploying the ternary 30B+ model as the resident deliberation engine in device VRAM.
  - Target Judgment Cost: Reducing total wall-clock time from proposal to verified merge to $< 6.0\text{ hours}$, with $< 1.0\text{ hours}$ of human review intervention required.

---

## 3. Mathematical Specification (Declarative Tensor Logic AST)

Ternary weights are stored as packed 2-bit bitfields in an integer tensor $W_{\text{packed}} \in \mathbb{Z}_4^{D_{\text{in}} \times \lceil D_{\text{out}} / 4 \rceil}$, with scale vectors $\gamma \in \mathbb{R}^{D_{\text{out}}}$.

The forward matrix contraction is expressed in `clj-einsum` Declarative Tensor Logic:

```clojure
;; Step 1: Unpack 2-bit packed integers to {-1, 0, 1} floats and scale by gamma
[:= [:w_unpacked :din :dout]
    [:w_packed :din :dout_packed]
    [:gamma :dout]
    {:unpack :ternary-2bit}]

;; Step 2: Dense contraction over activations
[:= [:y :batch :seq :dout]
    [:x :batch :seq :din]
    [:w_unpacked :din :dout]
    {:act :identity}]
```

In StableHLO lowering, `:unpack :ternary-2bit` expands into:
$$\text{Shift} \to \text{BitwiseAnd}(0x03) \to \text{Subtract}(1.0) \to \text{Multiply}(\gamma)$$
which OpenXLA's fusion engine compiles into a single fused GPU kernel before GEMM evaluation.

---

## 4. Execution Harness & Silicon Verification Plan

- **Module Namespace**: `einsum.quant.ternary`
- **Benchmarking Script**: `tools/benchmark_ternary.clj`
- **Reference Baselines**:
  1. Uncompressed FP16 Gemma 4 E2B / 27B.
  2. EXL3 3.0bpw unquantized matrix parity (`test/einsum/logic/exl3_test.clj`).
- **Target Hardware**: AMD Radeon RX 7900 XTX (Navi 31, 24GB VRAM, ROCm 6.2, `libjsig.so`).
- **Required Artifacts**: `catalog/gate1_compression/cat_q_ternary/results.edn`, `summary.csv`.
