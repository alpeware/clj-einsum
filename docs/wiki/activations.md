# Activation Functions & Gated Sub-Blocks

Activation functions introduce non-linear expressivity into neural networks. Modern transformer architectures favor Gated Linear Units (GLU).

---

## 1. Standard Non-Linearities

* **GELU (Gaussian Error Linear Unit)**:
  $$\text{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$$
* **SiLU (Sigmoid Linear Unit / Swish)**:
  $$\text{SiLU}(x) = x \times \sigma(x) = \frac{x}{1 + e^{-x}}$$

* **Clojure Reference**: `[:gelu ...]` and `[:silu ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).

---

## 2. SwiGLU & GeGLU Gated Feed-Forward Blocks

Gated Linear Units split the feed-forward projection into a gate branch and an up-projection branch:
$$\text{SwiGLU}(x, W_{gate}, W_{up}, W_{down}) = \left( \text{SiLU}(x W_{gate}) \odot (x W_{up}) \right) W_{down}$$
$$\text{GeGLU}(x, W_{gate}, W_{up}, W_{down}) = \left( \text{GELU}(x W_{gate}) \odot (x W_{up}) \right) W_{down}$$
* **Advantage**: Superior gradient flow and representation quality compared to single-projection MLPs.
* **Clojure Reference**: `[:swiglu ...]` in [`einsum.logic.nn`](../../src/einsum/logic/nn.clj).
