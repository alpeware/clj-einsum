# Gate 1: Resource Efficiency & Compression (`catalog/gate1_compression/`)

**Metric**: `bytes/param & peak VRAM footprint reduction (Storage)`, plus `measured tok/s throughput (Compute)`  
**Target**: $\le 0.5\text{ bytes/param}$ ($\le 0.25\text{ bytes/param}$ for ternary 1.58b)  
**Primary Hardware Target**: AMD Radeon RX 7900 XTX (24GB VRAM) / NVIDIA RTX 4090 (24GB VRAM)  
**Architecture Guide**: [docs/architecture/rsi_gates.md](../../docs/architecture/rsi_gates.md)

---

## 1. Scope & Objective

Gate 1 evaluates whether a model generation achieves **super-linear reduction in resource consumption** ($C_{n+1} \ll C_n$):
- **Storage & Bandwidth (Arithmetic)**: Uncompressed FP16 weights require $2.0\text{ bytes/param}$. Fitting 30B–70B models into consumer silicon ($\le 24\text{ GB}$) requires reducing representation to $\le 0.25\text{ bytes/param}$ via extreme ternary/CAT-Q quantization.
- **Compute & Throughput (Hypothesis)**: Measuring whether unpack-and-GEMM or fused ternary kernels translate storage reduction into sustained generation throughput ($\text{tok/s}$) without regressing token latency.

## 2. Contained Experiment Pods

- [`e24_prefix_cache_handover/`](e24_prefix_cache_handover/): In-VRAM homogeneous instance prefix-cache handover. Eliminates 74.46 ms of re-prefill latency (0.04 ms transition) while allocating **0 duplicate model weights** across 493 PJRT buffers.
- *Active Proposals*: `proposals/gate1_compression/cat_q_ternary/` (Ternary 1.58b compression for 30B–70B models).
