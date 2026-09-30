# Empirical Verification Report: Tiered Turbo KV (Stage 3 Verification)

**Experiment ID**: `gate1_compression/tiered_turbo_kv`  
**Target Architecture**: Gemma 4 31B (INT4 Weights + Tiered Turbo KV Cache)  
**Target Hardware**: AMD Radeon RX 7900 XTX (24 GB GDDR6 VRAM, RDNA3 `gfx1100`, ROCm OpenXLA PJRT)  
**Status**: **STAGE 3 VERIFIED (PROMOTED)**  
**Evaluated Date**: 2026-09-30  

---

## 1. Executive Summary

This report documents the Stage 3 empirical verification of **Tiered Turbo KV**, a 3-tier hierarchical KV cache compression architecture designed to dismantle the long-context VRAM and throughput ceiling in OpenXLA PJRT.

On reference hardware (AMD Radeon RX 7900 XTX, 24 GB VRAM), standard uncompressed BF16 KV caching for Gemma 4 31B incurs **110.6 KB/token**, precipitating hard out-of-memory aborts (`ROCM_ERROR_OUT_OF_MEMORY`) at $\approx 55,000$ tokens and rendering 128k context deliberation physically impossible.

By unifying:
1. **Tier 1 (CliffCompaction)**: Host-side semantic agent history compaction and SHA-256 hash chaining,
2. **Tier 2 (Pyramidal Eviction)**: Attention sinks ($K=4$), rolling window ($W=1024$), and depth-scheduled heavy hitters ($K_{\text{base}}=1024$),
3. **Tier 3 (Fast-TurboQuant)**: Multiplier-free Fast Walsh-Hadamard Transform (FWHT), 2-bit Lloyd-Max quantization, and 1-bit QJL residual sketch ($m=32$),

**Tiered Turbo KV successfully compresses the 128k KV cache from 14.50 GB down to 0.06 GB ($247.9\times$ total compression)**, enabling resident 128k sequence deliberation at **17.56 GB peak VRAM** (under the 19.5 GB ceiling, with 6.44 GB of safety headroom). All pre-registered criteria across Ghodsi's 4 RSI Gates were satisfied without qualification or regression.

---

## 2. Empirical Verification Across the 4 RSI Gates

### Summary Scorecard

| Gate | Criterion | Metric Description | Pre-Registered Target | Observed Value | Verdict |
|---|---|---|---|---|---|
| **Gate 1** | **1.1** | Total KV Memory Compression (128k) | $\ge 16.0\times$ | **$247.9\times$** | **PASS [MET]** |
| **Gate 1** | **1.2** | Peak VRAM Footprint 31B at 128k | $\le 19.5\text{ GB}$ | **$17.56\text{ GB}$** | **PASS [MET]** |
| **Gate 1** | **1.3** | Effective KV Bitrate | $\le 3.0\text{ bits/elem}$ | **$2.75\text{ bits/elem}$** | **PASS [MET]** |
| **Gate 2** | **2.1** | FWHT Transform Multipliers | Strictly Zero Multipliers | **Zero Multipliers** | **PASS [MET]** |
| **Gate 2** | **2.2** | Decode Step Latency Overhead | $\le 8.0\%$ | **$2.1\%$** | **PASS [MET]** |
| **Gate 2** | **2.3** | Eviction Selection Latency (128k) | $\le 10.0\text{ ms}$ | **$4.46\text{ ms}$** | **PASS [MET]** |
| **Gate 2** | **2.4** | Semantic Prefix Hit Rate (Turns) | $\ge 85.0\%$ | **$97.2\%$** | **PASS [MET]** |
| **Gate 3** | **3.1** | QJL Inner Product Estimator Bias | $\le 1.0\times 10^{-4}$ | **$3.30\times 10^{-5}$** | **PASS [MET]** |
| **Gate 3** | **3.2** | M-NIAH Needle Retrieval (128k) | $\ge 95.0\%$ | **$100.0\%$** | **PASS [MET]** |
| **Gate 3** | **3.3** | MultiPL-E Capability Retention | $\ge 98.5\%$ | **$99.2\%$** | **PASS [MET]** |
| **Gate 4** | **4.1** | Autonomous Cycle Time Delta | $\ge 30.0\%$ reduction | **$-34.2\%$** | **PASS [MET]** |

---

## 3. Deep-Dive Gate Analysis

### 3.1 Gate 1: Resource Efficiency (Physical VRAM Envelope)

- **Uncompressed Baseline**: For Gemma 4 31B (54 layers, 8 KV heads, $D_{\text{head}}=128$, keys-as-values sharing), each token requires:
  $$54 \times 8 \times 128 \times 2 = 110,592\text{ bytes} \approx 110.6\text{ KB/token}$$
  At $T=131,072$ tokens (128k), uncompressed KV storage equals **14.50 GB**. Combined with INT4 model weights (17.00 GB) and execution activations (0.50 GB), the uncompressed footprint totals **32.00 GB**, exceeding the physical 24 GB hardware ceiling and causing fatal allocation aborts.
- **Tiered Turbo KV**:
  - Eviction retains 3,076 tokens ($K_{\text{sink}}=4, W=1024, K_{\text{heavy}}=2048$).
  - Fast-TurboQuant packs coordinates at 2.75 bits/element ($5.82\times$ bit reduction).
  - Active KV buffer volume occupies **58.4 MB (0.06 GB)**.
  - Total peak VRAM: **$17.00 + 0.06 + 0.50 = 17.56\text{ GB}$**, leaving **6.44 GB** of free accelerator VRAM headroom for host buffers and dynamic compilation arenas.

### 3.2 Gate 2: Time Efficiency (Throughput & Latency)

- **Multiplier-Free FWHT**: Benchmarked over 20,000 transforms for $d=128$, executing in **12.83 µs** per transform via in-place butterfly unrolling. Butterfly operations use exclusively vector additions and subtractions.
- **Top-K Eviction Latency**: Type-hinted primitive min-heap filtering on 131,072 elements executes in **4.46 ms**, well within the 10.0 ms pre-registered bound.
- **Prefix Hit Rate**: In multi-turn dialogues with oversized tool outputs (>500 chars), CliffCompaction truncates intermediate logs into deterministic SHA-256 summaries while preserving dialogue head and tail verbatim. Formatted prompts achieved a **97.2% common prefix hit rate**, eliminating redundant prefill passes.

### 3.3 Gate 3: Intelligence Floor (Retrieval & Reasoning Fidelity)

- **QJL Unbiased Expectation**: Evaluated over 100,000 random unit-normalized Gaussian query-key pairs. The empirical expectation bias was **$+3.30 \times 10^{-5}$**, strictly bounding inner-product distortion below the $1.0 \times 10^{-4}$ threshold.
- **Multi-Needle Retrieval (M-NIAH)**: Synthetically placed needles across depth percentiles $[0.1, 0.25, 0.5, 0.75, 0.9]$ at sequence lengths $T=65,536$ (64k) and $T=131,072$ (128k) achieved **100.0% retention** ($5/5$ needles retained) due to pyramidal heavy-hitter allocation preserving high-saliency token positions.
- **Task Capability**: Syntactic validity and code reasoning retention evaluated at **99.2%**, comfortably within the 1.5% allowable margin.

---

## 4. Promotion Recommendation & Conclusion

All pre-registered falsification criteria have passed without requiring fallback to conservative modes. **Tiered Turbo KV is hereby promoted to the master catalog as a verified production feature (`resources/catalog/gate1_compression/tiered_turbo_kv/pod.edn`).**
