# Gemma 3 Model Architecture Specification & StableHLO Graph

- **Status**: **Fully Supported** (Gemma 3 1B, Gemma 3 4B, Gemma 3 12B, Gemma 3 27B)
- **Clojure Source**: [`models/gemma3.clj`](../../models/gemma3.clj)
- **Test Suite**: [`test/models/gemma3_test.clj`](../../test/models/gemma3_test.clj)

---

## 1. Architectural Highlights

Gemma 3 builds on Gemma 2's GQA attention and $+1.0$ weight offset Gemma RMSNorm, offering an ultra-lightweight **Gemma 3 1B** footprint ($d_{model}=1536$, 22 layers) optimized for edge laptops and mobile devices.

---

## 2. Hyperparameter Variants

| Hyperparameter | Gemma 3 1B | Gemma 3 4B | Gemma 3 12B | Gemma 3 27B |
| :--- | :--- | :--- | :--- | :--- |
| **Hidden Dimension ($d_{model}$)** | 1536 | 2560 | 3840 | 4608 |
| **Layers ($N_{layers}$)** | 22 | 34 | 40 | 46 |
| **Query Heads ($H_q$)** | 8 | 10 | 16 | 32 |
| **KV Heads ($H_{kv}$)** | 1 | 2 | 8 | 16 |
| **Head Dimension ($d_k$)** | 256 | 256 | 256 | 128 |

---

## 3. Verification & Execution

```bash
clojure -M:test -e "(require '[models.gemma3-test]) (clojure.test/run-tests 'models.gemma3-test)"
```
