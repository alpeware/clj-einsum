# Autoregressive Sampling Algorithms & KV Cache Management

During autoregressive generation, output logits $z \in \mathbb{R}^V$ are converted into probability distributions and sampled to select the next token $x_{t+1}$.

---

## 1. Sampling Algorithms

### A. Temperature Scaling
Modulates distribution entropy by dividing raw logits by temperature $T > 0$:
$$z'_i = \frac{z_i}{T}$$
* $T < 1.0$: Sharpens distribution (more deterministic).
* $T > 1.0$: Flattens distribution (more creative).

### B. Top-K Filtering
Restricts sampling candidates to the top $K$ highest-probability logits, setting all other logits to $-\infty$:
$$z'_i = \begin{cases} z_i & \text{if } \text{rank}(z_i) \le K \\ -\infty & \text{otherwise} \end{cases}$$

### C. Top-P (Nucleus) Sampling
Accumulates candidate token probabilities until the cumulative sum reaches threshold $P \in (0, 1]$:
$$\sum_{i \in S} \text{softmax}(z)_i \ge P$$

---

## 2. KV-Cache Management

The Key-Value (KV) cache stores historical key and value tensors across steps to avoid re-computing attention for past tokens:
```
Step t-1:  KV_cache = [K_{0..t-1}, V_{0..t-1}]
Step t:    Compute [K_t, V_t] for new token x_t
           Update: KV_cache = concat(KV_cache, [K_t, V_t])
```

---

## 3. Clojure Implementation in `clj-einsum`

- **Temperature Scaling**: [`einsum.runtime.sampling/apply-temperature`](../../src/einsum/runtime/sampling.clj) in [`einsum.runtime.sampling`](../../src/einsum/runtime/sampling.clj).
- **Top-K Filtering**: [`einsum.runtime.sampling/apply-top-k`](../../src/einsum/runtime/sampling.clj) in [`einsum.runtime.sampling`](../../src/einsum/runtime/sampling.clj).
- **Autoregressive Loop**: [`einsum.runtime.generation.autoregressive`](../../src/einsum/runtime/generation/autoregressive.clj).
- **KV Cache Invariants**: Tested in [`test/einsum/runtime/generation_test.clj`](../../test/einsum/runtime/generation_test.clj).
