# Gemma Model Architecture Specification & Verification

- **Status**: **Fully Supported** (Gemma 2B, Gemma 3 1B/4B/12B/27B, Gemma 4 E2B/E4B/12B/26B-A4B/31B)
- **Clojure Source**: [`models/gemma.clj`](../../models/gemma.clj)
- **CLI Runner**: [`tools/gemma4_inference.clj`](../../tools/gemma4_inference.clj)
- **Test Suite**: [`test/models/gemma_test.clj`](../../test/models/gemma_test.clj)

---

## 1. Model Configurations

| Variant | Hidden Dim | Layers | Query Heads | KV Heads | Head Dim | Per-Layer Gate Dim |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma 2B** | 2048 | 18 | 8 | 1 | 256 | N/A |
| **Gemma 3 1B** | 1536 | 22 | 8 | 1 | 256 | N/A |
| **Gemma 4 E2B** | 1536 | 22 | 8 | 1 | 256 | 256 |
| **Gemma 4 E4B** | 2560 | 34 | 10 | 2 | 256 | 256 |

---

## 2. Key Neural Network Components

1. **Gemma RMSNorm**: $+1.0$ weight scaling offset $y = \frac{x}{\text{RMS}(x)} \times (1.0 + w)$.
2. **Grouped-Query Attention (GQA)**: Multi-query attention with Rotary Position Embeddings (RoPE).
3. **SwiGLU MLP**: Gated Feed-Forward blocks ($\text{SiLU}(x W_{gate}) \odot (x W_{up}) W_{down}$).
4. **Per-Layer Input Residual Gating**: Gemma 4 per-layer scalar gating projections.

---

## 3. Verification & Execution

* **Interactive CLI Hardware Execution (SYCL / CPU)**:
  ```bash
  clojure -M tools/gemma4_inference.clj --model Gemma-4-E2B-it --backend sycl --precision int8
  ```
* **Generative Unit & Property Tests**:
  ```bash
  clojure -M:test -e "(require '[models.gemma-test]) (clojure.test/run-tests 'models.gemma-test)"
  ```
