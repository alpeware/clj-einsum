# Normalization Algorithms & Clojure Implementations

Normalization layers stabilize deep neural network training and activation distributions. This document details the mathematical formulations and implementations in `clj-einsum`.

---

## 1. Standard Layer Normalization (LayerNorm)

Standard LayerNorm computes zero-mean, unit-variance normalization across feature dimensions:
$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \times \gamma + \beta$$
* **Clojure Reference**: `[:layer-norm ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

---

## 2. Root Mean Square Normalization (RMSNorm)

RMSNorm removes mean-subtraction, scaling inputs strictly by the root mean square of activations:
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$
$$\bar{x} = \frac{x}{\text{RMS}(x)} \times \gamma$$
* **Advantage**: Saves compute and memory traffic by eliminating mean calculation while maintaining stability.
* **Clojure Reference**: `[:rms-norm ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

---

## 3. Gemma RMSNorm ($+1.0$ Weight Scaling)

Google’s Gemma models (Gemma 2, Gemma 3, Gemma 4) modify RMSNorm by applying a $+1.0$ offset to the scale parameter $w$:
$$y = \frac{x}{\text{RMS}(x)} \times (1.0 + w)$$
* **Rationale**: Initializing weights $w$ to zero results in an identity scale $1.0$, preventing numerical instability in deep networks.
* **Clojure Reference**: `[:gemma-rms-norm ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).
