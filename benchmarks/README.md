# Empirical Hardware Benchmark Suite & Cross-Framework Performance Reports

Welcome to the **`clj-einsum` Empirical Hardware Benchmark Registry**.

`clj-einsum` is designed to run across heterogeneous consumer hardware (laptops, desktops, workstations) and multi-backend PJRT drivers (CPU, Intel SYCL, AMD ROCm, NVIDIA CUDA). To systematically evaluate performance, track driver upgrades, and compare against Python reference engines (JAX / OpenXLA), this directory stores empirical benchmark reports for each hardware and driver target.

---

## 📊 Target Hardware Reports Index

| Hardware Target | Backend Driver | Telemetry / Driver Build | Empirical Report |
| :--- | :--- | :--- | :--- |
| **Lenovo ThinkPad X1 Carbon Gen 13** | Intel Arc 140V (SYCL Level-Zero) | Compute Runtime `26.22.038646` / Level-Zero `v1.30.0` | [Report](lenovo_x1_carbon_intel_sycl.md) |
| **AMD Ryzen Desktop (7900 XTX 24G)** | AMD RDNA3 (ROCm) | ROCm `7.2.0` / `6.0` Driver | [Report](amd_desktop_7900_xtx_rocm.md) |

---

## 🔬 Benchmark Methodology & Verification

All benchmarks measure standard workloads defined in [`src/einsum/runtime/benchmark/workloads.clj`](../src/einsum/runtime/benchmark/workloads.clj):
1. **GEMM FP32**: Matrix Multiplication $[1024, 1024] \times [1024, 1024]$ ($2.147 \text{ GFLOPs}$).
2. **GEMM BF16**: Matrix Multiplication $[1024, 1024] \times [1024, 1024]$ in bfloat16.
3. **RMSNorm**: Root Mean Square Normalization $[1, 2048, 4096]$.
4. **SwiGLU**: Gated Feed-Forward Activation $[1, 2048, 4096]$.
5. **GQA Causal Attention**: Grouped-Query Attention $[1, 128, 8, 256]$ Query, $[1, 128, 1, 256]$ KV.
6. **GPT-2 Block**: Full Transformer Layer Block $[1, 128, 768]$.
7. **Gemma 4 Block**: Full Transformer Layer Block $[1, 128, 1536]$.

### Verification & JAX Parity
To compare **JVM / Clojure Panama FFM OpenXLA (`clj-einsum`)** directly against **Python / XLA (JAX)**, reference Python benchmarks are maintained in [`verification/jax_benchmark.py`](../verification/jax_benchmark.py).
