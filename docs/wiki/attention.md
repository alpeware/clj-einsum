# Attention Mechanisms & Memory Optimization Algorithms

Attention mechanisms dictate the computational complexity and memory footprint of Transformer models. This document compares standard attention variants against single-batch ($B=1$) memory-bound optimizations.

---

## 1. Multi-Head Attention (MHA) vs. Grouped-Query Attention (GQA)

### Multi-Head Attention (MHA)
In standard MHA, every Query head has a corresponding Key and Value head:
$$\text{Query, Key, Value Dimensions} = [H, d_k]$$
* **KV-Cache Size per Token**: $2 \times H \times d_k \times \text{layers} \times \text{bytes}$.
* **Bottleneck**: At $B=1$, streaming huge KV-caches per token consumes excessive memory bandwidth.

### Grouped-Query Attention (GQA)
GQA shares a smaller set of $H_{kv}$ Key and Value heads across $H_q$ Query heads ($H_q = g \times H_{kv}$):
```
MHA (1:1 Query-to-KV ratio):
  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
  │  │  │  │  │  │  │  │
  K0 K1 K2 K3 K4 K5 K6 K7

GQA (8:1 Query-to-KV ratio):
  Q0 Q1 Q2 Q3 Q4 Q5 Q6 Q7
  └──┴──┴──┼──┴──┴──┴──┘
           K0 (Shared Key/Value Head)
```
* **Clojure Reference**: `[:gqa-attention ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

---

## 2. Gemma 4 Hybrid Sliding-Window Attention

Gemma 4 combines local sliding-window attention with global full-attention layers:
1. **Sliding Window Layers**: Attention is restricted to a local window of $W$ tokens (e.g. $W=4096$). Tokens older than $W$ are evicted from the KV cache in RAM.
2. **Global Layers**: Every $N$-th layer maintains full context attention across all tokens.
3. **Per-Layer Shared KV Cache**: Adjacent layers share Key and Value projections, reducing KV cache bytes by $2\times$.

* **Clojure Reference**: [`gemma4-layer-ast`](../../models/gemma.clj) in [`models.gemma`](../../models/gemma.clj).

---

## 3. DeepSeek Multi-Head Latent Attention (MLA)

DeepSeek MLA compresses the Key and Value heads into a low-rank latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c = 576$):
$$c_t^{KV} = W^{DKV} x_t$$
$$K_t^C = W^{UK} c_t^{KV}, \quad V_t^C = W^{UV} c_t^{KV}$$
* **Impact**: Instead of storing thousands of floats per token, the KV cache stores only 576 floats per token across all heads, yielding a **$>90\%$ KV cache memory reduction**.

---

## 4. FlashDecoding & Split-K Parallelism ($B=1$)

For single-request ($B=1$) autoregressive decoding, standard FlashAttention leaves GPU execution units underutilized. **FlashDecoding** splits the sequence length dimension of the KV cache across multiple GPU thread blocks (Split-K):
1. **Split**: Sequence $L$ is partitioned into $K$ chunks.
2. **Parallel Attention**: Each chunk computes local attention reduction in parallel.
3. **Merge**: A final reduction kernel combines local softmax statistics.
