# Low-Bit Weight Quantization & In-Graph De-quantization

At single-batch size ($B=1$), token generation throughput is directly proportional to the memory bandwidth required to load weight tensors from RAM.

---

## 1. Post-Training Quantization (PTQ) Formats

| Format | Bits per Weight | Typical Compression Ratio | Speedup at $B=1$ |
| :--- | :--- | :--- | :--- |
| **FP16 / BF16** | 16 bits | $1.0\times$ (Baseline) | $1.0\times$ |
| **INT8 / FP8** | 8 bits | $2.0\times$ | $\sim 1.8\times - 2.0\times$ |
| **INT4 / AWQ / GPTQ** | 4 bits | $4.0\times$ | $\sim 3.2\times - 3.8\times$ |

---

## 2. In-Graph De-quantization in OpenXLA / StableHLO MLIR

In `clj-einsum`, quantized weights are stored as low-bit integer tensors (`:i8` or packed `:i4`) alongside per-channel/per-block scale vectors (`:f32` or `:bf16`).

Inside the trace graph, de-quantization is fused directly before the matrix multiplication:
```clojure
(defn dequantize-weight-i8
  [w-i8 scale]
  (t/* (t/cast w-i8 :f32) scale))

(defn quantized-linear
  [x w-i8 scale bias]
  (let [w-f32 (dequantize-weight-i8 w-i8 scale)
        out (t/matmul x (t/transpose w-f32 [1 0]))]
    (if bias (t/+ out bias) out)))
```

When compiled to PJRT via OpenXLA, the integer de-quantization and matrix multiplication compile into fused hardware tensor core kernels (XMX on Intel Arc, Tensor Cores on NVIDIA CUDA, Matrix Accelerators on AMD ROCm).

* **Clojure Reference**: [`einsum.runtime.safetensors`](../../src/einsum/runtime/safetensors.clj).
