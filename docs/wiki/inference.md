# Inference Dynamics: Prompt Prefill Phase vs. Autoregressive Decoding Phase ($B=1$)

LLM inference consists of two computationally distinct execution phases: the **Compute-Bound Prompt Prefill Phase** and the **Memory-Bandwidth Bound Autoregressive Decoding Phase**.

---

## 1. Comparative Dynamics

```mermaid
flowchart TD
    subgraph Prefill ["Phase 1: Prompt Prefill Phase (Compute-Bound)"]
        P_Input["Input Prompt: L tokens [1, L, d]"] --> P_GEMM["Batched Matrix Pass O(L² · d)"]
        P_GEMM --> P_KV["Populate KV Cache for L Tokens"]
        P_KV --> P_TTFT["First Token Generated (TTFT)"]
    end

    subgraph Decode ["Phase 2: Autoregressive Decoding Phase (Memory-Bandwidth Bound)"]
        D_Input["New Token xₜ [1, 1, d]"] --> D_Fetch["Fetch All Model Weights W from RAM"]
        D_Fetch --> D_KV["Update & Append to KV Cache"]
        D_KV --> D_Logit["Sample Next Token xₜ₊₁ (TPOT)"]
        D_Logit --> D_Input
    end
```

---

## 2. Mathematical Breakdown

| Execution Attribute | Phase 1: Prompt Prefill Phase | Phase 2: Autoregressive Decoding ($B=1$) |
| :--- | :--- | :--- |
| **Input Shape** | $[1, L, d]$ (Prompt of length $L$) | $[1, 1, d]$ (Single token step) |
| **Primary Metric** | **Time to First Token (TTFT)** | **Time Per Output Token (TPOT)** |
| **Compute Complexity** | $O(L^2 \cdot d)$ (Quadratic in prompt length) | $O(d)$ (Linear in model dimension) |
| **Arithmetic Intensity** | High ($\gg 50 \text{ FLOPs/byte}$) | Extremely Low ($\approx 1 - 2 \text{ FLOPs/byte}$) |
| **Hardware Bottleneck** | **Compute-Bound** (GPU Tensor Cores) | **Memory-Bandwidth Bound** (RAM/iGPU Bus) |
| **GPU Tensor Core Utilization** | $70\% - 90\%$ | $< 5\%$ (Waiting for weight bytes) |

---

## 3. Clojure Implementation in `clj-einsum`

- **Prefill Execution**: Handled in [`einsum.runtime.generation.autoregressive`](../../src/einsum/runtime/generation/autoregressive.clj) where prompt sequence tokens are evaluated in a single matrix pass.
- **Decoding Loop**: Executed iteratively in [`einsum.runtime.generation.autoregressive`](../../src/einsum/runtime/generation/autoregressive.clj) passing length-1 tokens alongside updated KV-cache handles.
