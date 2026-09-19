# Design Document: Tensor Logic Intermediate Representation for clj-xla

## 1. System Overview

`clj-xla` is a pure Clojure compiler frontend targeting OpenXLA and StableHLO via the PJRT C API. Instead of relying on macro-based domain-specific languages (DSLs) or tape-based dynamic execution graphs, `clj-xla` adopts Pedro Domingos' **Tensor Logic** as its foundational Intermediate Representation (IR).

Computational graphs are represented as immutable S-expressions using Hiccup-style data structures. The system unifies deep learning forward/backward passes, attention mechanisms, and discrete relational routing into Einstein summation equations over multi-dimensional tensors.

```
               [Hiccup EDN AST]
                      │
                      ▼ (Recursive Subgraph Expansion)
           [Primitive Tensor Equations]
                      │
                      ▼ (Backward-Chaining DCE & Index Partitioning)
         [Normalized Canonical Einsum Nodes]
                      │
                      ▼ (StableHLO MLIR Bytecode Generator)
           [In-Memory StableHLO Module]
                      │
                      ▼ (PJRT / Java FFM Panama)
        [Hardware-Compiled Binary (ROCm / CUDA / CPU)]

```

---

## 2. Core Architectural Principles

* **Code as Pure Data (Macro-Free):** Every operation, layer, and network is an EDN data structure. AI coding agents modify, analyze, and synthesize graphs using standard associative operations (`assoc`, `update`, `clojure.walk`) without macro evaluation hazards.
* **Einsum as Universal Contraction:** Following Tensor Logic, multilinear layers (dense linear projections, attention projections, batched matrix products) compile to a unified contraction primitive: joining matching indices, summing over unprojected indices, and applying optional univariate nonlinearities.


* **Backward Chaining for Dead-Code Elimination (DCE):** Queries for specific target tensors trigger backward-chaining dependency traversals. Unused intermediate expressions and projections are automatically pruned prior to lowering.


* **Dual Graph Lifecycle:** Autoregressive models compile into two distinct, static StableHLO binaries: a parallel **Prefill** executable and a 1-token step **Decode** executable managing persistent on-device Key-Value (KV) cache buffers.

---

## 3. IR Specification: Hiccup Tensor Syntax

### 3.1 Primitive Equations

A primitive equation defines an Einstein summation join and projection:

```clojure
[:= [Head-Name & head-indices] ?attrs-map & body-terms]

```

* **Head:** Vector starting with output tensor identifier keyword, followed by symbolic index keywords (e.g., `[:out :b :p :d]`).
* **Body Terms:** Zero or more vectors specifying input tensors and their dimensional indices (e.g., `[:w :d :dff]`, `[:x :b :p :d]`).


* **Attributes Map (`?attrs-map`):** Optional map specifying elementwise post-activations (`:act :silu`), scaling factors (`:scale 0.125`), or reduction overrides.


* **Index Semantics:**
* **Batch Dimensions:** Indices present in all body terms and retained in the head.


* **Contracting Dimensions:** Indices present in multiple body terms but omitted from the head (implicitly summed out).


* **Free Dimensions:** Indices present in one body term and preserved in the head.




* **Implicit Accumulation:** Declaring multiple equations with an identical head signature implicitly sums their results, eliminating explicit addition nodes for skip/residual connections.



### 3.2 Containers and Subgraphs

Hierarchical networks use vector tags for composition:

```clojure
;; Sequential container
[:block {:name :layer-0}
  [:rmsnorm [:x-norm :b :p :d] [:x :b :p :d] [:gamma :d]]
  [:= [:q :b :p :h :dh] [:x-norm :b :p :d] [:w-q :d :h :dh]]]

;; Residual wrapper
[:residual
  [:out [:x-mid :b :p :d] [:x-in :b :p :d]]
  [:ffn-subgraph ...]]

```

---

## 4. Compiler Pipeline Implementation Contracts

### 4.1 Phase 1: Recursive Subgraph Expansion

Subgraphs register expansion rules via Clojure multimethods. The expander traverses the Hiccup AST using `clojure.walk/prewalk` until the tree contains only primitive `[:= ...]` equations or direct lowering hooks.

```clojure
(defmulti expand-node (fn [node ctx] (when (vector? node) (first node))))
(defmethod expand-node :default [_ _] nil)

(defn expand-ast [ctx ast]
  (clojure.walk/prewalk
    (fn [node]
      (if-let [expanded (expand-node node ctx)]
        (expand-ast ctx expanded)
        node))
    ast))

```

### 4.2 Phase 2: Index Partitioning & StableHLO Lowering

Every primitive contraction compiles to `stablehlo.dot_general` or elementwise primitives by partitioning indices via set operations:

| Index Category | Formula | StableHLO Target Attribute |
| --- | --- | --- |
| **Contracting** | $(I_{\text{lhs}} \cap I_{\text{rhs}}) \setminus I_{\text{head}}$<br> | `lhs_contracting_dimensions`, `rhs_contracting_dimensions` |
| **Batch** | $I_{\text{lhs}} \cap I_{\text{rhs}} \cap I_{\text{head}}$<br> | `lhs_batching_dimensions`, `rhs_batching_dimensions` |
| **Free LHS** | $I_{\text{lhs}} \setminus I_{\text{rhs}}$ | Preserved in output shape order |
| **Free RHS** | $I_{\text{rhs}} \setminus I_{\text{lhs}}$ | Preserved in output shape order |

```clojure
(defn partition-indices [lhs-idx rhs-idx head-idx]
  (let [lhs-set  (set lhs-idx)
        rhs-set  (set rhs-idx)
        head-set (set head-idx)
        common   (clojure.set/intersection lhs-set rhs-set)]
    {:contracting (vec (clojure.set/difference common head-set))
     :batch       (vec (clojure.set/intersection common head-set))
     :lhs-free    (vec (clojure.set/difference lhs-set rhs-set))
     :rhs-free    (vec (clojure.set/difference rhs-set lhs-set))}))

```

### 4.3 Phase 3: Autodiff (Reverse-Mode Derivative Derivation)

For fine-tuning and distillation, backward adjoint equations are generated algebraically without tape trackers. By the product rule over Tensor Logic RHS joins, the partial derivative with respect to any term is the product of remaining terms multiplied by the incoming adjoint:

$$\frac{\partial \mathcal{L}}{\partial T_i} = \sum_{\text{equations}} \delta_{\text{out}} \cdot \prod_{j \neq i} T_j$$

```clojure
(defn derive-adjoint-equations [{:keys [head body]}]
  (let [[out-tensor & out-idx] head
        d-out (keyword (str "adj/" (name out-tensor)))]
    (mapv
      (fn [[target-tensor & target-idx]]
        (let [other-inputs (remove #(= (first %) target-tensor) body)
              d-target     (keyword (str "adj/" (name target-tensor)))]
          [:= (into [d-target] target-idx)
              (into [[d-out out-idx]] other-inputs)]))
      body)))

```

---

## 5. Memory Management & Hardware Strategy

### 5.1 Dynamic State Updates (KV Caches)

Autoregressive decoding executes with a fixed sequence length context window to avoid buffer reallocations. KV cache insertion is lowered to `stablehlo.dynamic_update_slice`:

```clojure
{:op :stablehlo/dynamic_update_slice
 :operand       :kv-cache-buffer
 :update        :k-new-slice
 :start-indices [0 :current-pos-sym 0 0]}

```

### 5.2 Host/Device Partitioning via Project Panama

* **Host Layer (JVM):** Clojure manages tokenization, sampling (greedy/top-p), dynamic control flow, and PJRT C API calls via `java.lang.foreign`.
* **Device Layer (Accelerators):** Dense GEMM operations, RMSNorm kernel fusions, and attention heads remain resident in VRAM as pre-compiled StableHLO execution graphs.

---

## 6. Coding Agent Directives & Verification

AI coding agents contributing to `clj-xla` must adhere to the following implementation contracts:

1. **No Macros for Graph Construction:** All high-level layer definitions must be pure functions returning Hiccup vectors or multimethod extensions of `expand-node`.
2. **Deterministic Output Order:** Set operations over tensor indices must be deterministically sorted using vectors to guarantee reproducible StableHLO dimension number bindings.
3. **Malli Schema Conformance:** Every newly introduced node type must satisfy the core Hiccup schema:
```clojure
[:schema
 [:node
  [:cat :keyword
        [:? [:map-of :keyword :any]]
        [:* [:or [:vector :any] :keyword :number]]]]]

```


4. **Symbolic Shape Validation:** Before emitting StableHLO bytecode, shape unification passes must verify that contracting indices across LHS and RHS share identical static or symbolic bounds.

# Reference

https://arxiv.org/pdf/2510.12269
