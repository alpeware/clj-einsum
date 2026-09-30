# RFC: Tiered Turbo KV — 3-Tier Multiplier-Free Quantization and Prefix-Stable Agent Compaction for 128k+ Context

**Experiment**: `tiered_turbo_kv`  
**Type**: `experiment`  
**Arc**: `gate1_compression`  
**Gate**: `gate1_compression`  
**Generation**: `1`  
**Literature**: `["Pereira et al. (2026) Fast-TurboQuant: A Multiplier-Free Online Vector Quantization Approach (arXiv:2606.21448)", "Zandieh et al. (2025) TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (arXiv:2504.19874)", "Nguyen, Cho, Chen, Dettmers (2026) CliffCompaction: Cost-Efficient Compaction for Long-Horizon Coding Agents (arXiv:2609.26779)", "Xiao et al. (2024) Efficient Streaming Language Models with Attention Sinks (ICLR 2024, arXiv:2309.17453)", "Li et al. (2024) SnapKV: LLM Knows What You are Looking for Before Generation (arXiv:2404.14469)"]`  
**Hardware-Target**: `{:reference "AMD Radeon RX 7900 XTX (24GB) via ROCm 6.2 PJRT" :claim-shape ">= 16x combined KV compression, <= 64KB LDS workgroup ceiling, 128k context on 31B in < 24GB VRAM"}`  
**Extends**: `"cat-q-ternary"`  
**Refutes**: `nil`  
**Supersedes**: `nil`  
**Reopens**: `nil`  

---

## 1. Abstract & Motivation

As language model context lengths expand from local scratchpads (2k tokens) to long-horizon agentic task horizons (32k to 128k+ tokens), **the Key-Value (KV) cache becomes the primary capacity and throughput ceiling of the inference apparatus**, far surpassing model weights:

$$\text{KV Cache Volume} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times N_{\text{seq}} \times \text{bytes per element}$$

In our Generation 0 (`G0`) runtime:
- **Gemma 4 E4B** ($42\text{ layers}, 2\text{ kv-heads}, D_{\text{head}}=128$, BF16): **$43.0\text{ KB/token}$** ($1.41\text{ GB}$ at 32k tokens, $5.64\text{ GB}$ at 128k tokens).
- **Gemma 4 31B** ($54\text{ layers}, 8\text{ kv-heads}, D_{\text{head}}=128$, BF16): **$110.6\text{ KB/token}$** ($3.62\text{ GB}$ at 32k tokens, $14.50\text{ GB}$ at 128k tokens).

On our reference accelerator (AMD Radeon RX 7900 XTX, 24 GB VRAM, RDNA3 `gfx1100`), the 31B model with INT4 weights occupies $17.0\text{ GB}$ of VRAM. An uncompressed BF16 KV cache hits the 24 GB physical hardware ceiling at $\approx 55,000$ tokens, causing complete out-of-memory (`ROCM_ERROR_OUT_OF_MEMORY`) aborts and making 128k context physically impossible. Furthermore, ROCm RDNA3 strictly caps Local Data Share (LDS) at **64 KB per workgroup**, causing attention graph compilation failures when uncompressed sequence contexts expand.

This RFC proposes **Tiered Turbo KV**, a unified, 3-tier hierarchical compression architecture that spans the host agent loop down to the in-accelerator vector registers:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: Agent & Harness Semantic Compaction (CliffCompaction)                │
│ - Mechanical content-class pruning: tool results >500 chars dropped          │
│ - Tool calls truncated to 1-line signatures; human turns & head verbatim     │
│ - Hash-chained prefix stabilization guaranteeing 100% KV prefix reuse        │
│ - Compresses raw conversation history: N_raw -> N_prompt <= 16k tokens      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: Attention Graph & Token Eviction (StreamingLLM / SnapKV)              │
│ - 4 pinned initial Attention Sinks (<bos> + system framing tokens)           │
│ - Pyramidal observation window: retains top-k heavy-hitter tokens per head   │
│ - Bounded rolling local window W = 1024 for immediate context continuity    │
│ - Compresses active token set: N_prompt -> N_active <= 4k tokens (4x)        │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: Tensor Logic & In-Accelerator Quantization (Fast-TurboQuant)         │
│ - Multiplier-free Fast Walsh-Hadamard Transform (FWHT) + Rademacher phase    │
│ - 2-bit optimal Lloyd-Max scalar codebook (4 values per byte)                │
│ - 1-bit Quantized Johnson-Lindenstrauss (QJL) residual error correction      │
│ - Mathematically unbiased attention logits: E[<q, k_hat>] = <q, k>          │
│ - Compresses element precision: 16-bit BF16 -> 2.5–3.0 bits (5.33x)         │
└──────────────────────────────────────────────────────────────────────────────┘
```

By multiplying the orthogonal compression factors across tiers:

$$\text{Combined Compression Ratio} \approx \underbrace{4.0\times}_{\text{Tier 2 Eviction}} \times \underbrace{5.33\times}_{\text{Tier 3 Quantization}} \approx \mathbf{21.3\times}$$

Total 128k context KV cache memory on Gemma 4 31B drops from **$14.50\text{ GB}$ to $< 700\text{ MB}$**, unlocking sustained 128k+ long-horizon deliberation on a single 24 GB GPU with zero memory leaks and zero LDS compilation failures.

---

## 2. Pre-Registered Hypotheses & Falsifiable Gate Criteria

Every claim in this RFC is pre-registered across Ghodsi's 4 RSI Gates with mechanical falsification criteria:

### Gate 1: Resource Efficiency ($C_{n+1} \ll C_n$)

- **Storage / VRAM Footprint**:
  - **Criterion 1.1 (Combined Compression)**: Achieve $\ge 16.0\times$ total VRAM reduction in KV cache storage at sequence lengths $\ge 32,768$ tokens compared to uncompressed BF16 baseline.
  - **Criterion 1.2 (31B 128k VRAM Ceiling)**: Peak VRAM allocation for Gemma 4 31B-INT4 with a 131,072-token context must not exceed **$19.5\text{ GB}$** on the AMD RX 7900 XTX, leaving $\ge 4.5\text{ GB}$ of physical memory headroom.
  - **Criterion 1.3 (Sub-Byte Bitrate)**: Tier 3 Fast-TurboQuant alone must store key and value tensors at an effective bit-rate $\le 3.0\text{ bits/element}$ (inclusive of codebook indices, 1-bit QJL residual sketch, and scalar scaling norms).
  - **Falsification Condition**: Peak memory exceeds $22.0\text{ GB}$ at 128k context, or the runtime triggers `ROCM_ERROR_OUT_OF_MEMORY`.

### Gate 2: Time Efficiency ($T_{n+1} \ll T_n$)

- **Execution Latency & Hardware Roofline**:
  - **Criterion 2.1 (Multiplier-Free Overhead)**: The Tier 3 Fast-TurboQuant vector rotation and Lloyd-Max quantization pass must be completely multiplier-free, requiring strictly additions, subtractions, and bit shifts ($\mathcal{O}(d \log_2 d)$).
  - **Criterion 2.2 (Decode Step Latency Retention)**: Autoregressive single-token decode latency with Tier 3 active must remain within $\le 8.0\%$ of the uncompressed BF16 step latency ($< 1.5\text{ ms}$ overhead per step).
  - **Criterion 2.3 (RDNA3 LDS Workgroup Invariant)**: In-accelerator attention kernels must allocate $\le 64\text{ KB}$ of Local Data Share (LDS) per workgroup across all layers.
  - **Criterion 2.4 (Prefix Cache Hit Speedup)**: Tier 1 CliffCompaction hash-chaining must achieve $\ge 85\%$ KV prefix cache hit rate across multi-turn agent turns, reducing prompt re-evaluation time by $\ge 40\%$.
  - **Falsification Condition**: Step latency overhead exceeds $15.0\%$, or ROCm compiler fails with `requested > 65536 bytes LDS`.

### Gate 3: Capability Floor & Mathematical Unbiasedness ($A_{n+1} > A_n$)

- **Accuracy & Long-Context Needle Retention**:
  - **Criterion 3.1 (Unbiased Estimator Invariant)**: The QJL residual sketch must maintain mathematical zero-bias for attention inner products:
    $$\left| \frac{1}{N} \sum_{i=1}^N (\langle \mathbf{q}_i, \hat{\mathbf{k}}_i \rangle - \langle \mathbf{q}_i, \mathbf{k}_i \rangle) \right| \le 1.0\times 10^{-4}$$
  - **Criterion 3.2 (Needle-in-a-Haystack Floor)**: On a 100-sample synthetic Multi-Needle-in-a-Haystack (M-NIAH) test split spanning context depths up to 64k tokens, Tiered Turbo KV must retain $\ge 95.0\%$ of the uncompressed BF16 baseline retrieval accuracy.
  - **Criterion 3.3 (Clojure-Bench / MultiPL-E Non-Regression)**: On the 50-task stratified MultiPL-E Clojure benchmark, Tiered Turbo KV must produce **zero regressions** compared to baseline (McNemar exact test $p \ge 0.05$ with $b \le c$).
  - **Falsification Condition**: Retrieval accuracy drops below $90.0\%$, or statistically significant task capability regression occurs ($p < 0.05, b > c$).

### Gate 4: Continuous Recursion & The Compound Derivative

- **Compounding Cycle Time**:
  - Unlocking resident 31B long-context deliberation allows autonomous multi-turn bug-fixing agents to evaluate full multi-file test suites ($> 30\text{k}$ tokens of git diffs and logs) on local silicon without host memory thrashing.
  - Target: Proposal-to-verified-commit cycle time on long-horizon tasks reduced by $\ge 30\%$.

---

## 3. Mathematical Specification (Declarative Tensor Logic AST)

The transformation is declared purely in Pedro Domingos' Declarative Tensor Logic (`einsum.logic.*`), compiling into OpenXLA StableHLO MLIR without custom host C++ drivers.

### 3.1 Tier 3: Fast-TurboQuant Multiplier-Free Transform

Given an incoming activation vector $\mathbf{x} \in \mathbb{R}^d$ ($d=128$ for Gemma 4 head dimension):

1. **Rademacher Phase Inversion ($D$)**:
   $$\tilde{\mathbf{x}}_i = D_{ii} \cdot \mathbf{x}_i, \quad D_{ii} \in \{-1, +1\}$$
   Expressed in Tensor Logic as a sign bit-xor:
   $$\tilde{\mathbf{x}} = \text{select}(D < 0, -\mathbf{x}, \mathbf{x})$$

2. **Fast Walsh-Hadamard Transform ($\text{FWHT}$)**:
   Recursive butterfly reduction across $\log_2(d) = 7$ stages ($k \in \{0, \dots, 6\}$):
   $$\text{stride} = 2^k$$
   $$\mathbf{y}_{2j \cdot \text{stride} + l} = \mathbf{y}^{(k-1)}_{2j \cdot \text{stride} + l} + \mathbf{y}^{(k-1)}_{(2j+1) \cdot \text{stride} + l}$$
   $$\mathbf{y}_{(2j+1) \cdot \text{stride} + l} = \mathbf{y}^{(k-1)}_{2j \cdot \text{stride} + l} - \mathbf{y}^{(k-1)}_{(2j+1) \cdot \text{stride} + l}$$
   $$\text{for } j \in [0, d / 2^{k+1} - 1], \quad l \in [0, 2^k - 1]$$
   Normalized by $1/\sqrt{d}$:
   $$\mathbf{z} = \frac{1}{\sqrt{d}} \mathbf{y}$$

3. **2-Bit Lloyd-Max Scalar Quantization**:
   Since $\mathbf{z}$ coordinates asymptotically concentrate as i.i.d. $\mathcal{N}(0, \|\mathbf{x}\|^2/d)$, coordinates are quantized via precomputed Gaussian Lloyd-Max thresholds $\tau = \{-0.9674, 0.0, 0.9674\} \times \sigma$:
   $$c_i = \begin{cases} 
   0 & \text{if } \mathbf{z}_i < -\tau_1 \\
   1 & \text{if } -\tau_1 \le \mathbf{z}_i < 0 \\
   2 & \text{if } 0 \le \mathbf{z}_i < \tau_1 \\
   3 & \text{if } \mathbf{z}_i \ge \tau_1
   \end{cases}$$
   Reconstructed coordinate:
   $$\hat{\mathbf{z}}_i = \mu(c_i) \cdot \frac{\|\mathbf{x}\|}{\sqrt{d}}, \quad \mu \in \{-1.510, -0.453, 0.453, 1.510\}$$
   Primary reconstructed vector:
   $$\hat{\mathbf{x}} = D \cdot \text{FWHT}(\hat{\mathbf{z}}) \cdot \frac{1}{\sqrt{d}}$$

4. **1-Bit QJL Residual Error Sketch**:
   Residual error vector:
   $$\mathbf{r} = \mathbf{x} - \hat{\mathbf{x}}$$
   Random sign sketch projection ($S \in \{-1, +1\}^{m \times d}$, with $m=32$):
   $$\mathbf{s} = \text{sign}(S \mathbf{r}) \in \{0, 1\}^m$$
   Unbiased inner product estimator between query $\mathbf{q}$ and key $\mathbf{k}$:
   $$\langle \mathbf{q}, \mathbf{k} \rangle \approx \langle \mathbf{q}, \hat{\mathbf{k}} \rangle + \sqrt{\frac{\pi}{2}} \cdot \frac{\|\mathbf{r}_k\|_2}{m} \cdot \sum_{j=1}^m (S \mathbf{q})_j \cdot (2\mathbf{s}_j - 1)$$

### 3.2 Tier 2: Attention Sinks & Pyramidal Eviction

For layer $l$ and attention head $h$ at sequence length $T > W$:
- **Attention Sink Set**: First $K_{\text{sink}} = 4$ positions $\{0, 1, 2, 3\}$ permanently pinned in device memory.
- **Local Sliding Window**: Most recent $W = 1024$ positions $\{T-W, \dots, T-1\}$ retained verbatim.
- **Heavy-Hitter Selection**: For intermediate positions $j \in [K_{\text{sink}}, T-W-1]$, compute cumulative attention mass:
  $$M_j = \sum_{t=j+1}^T A_{t, j}$$
  Retain top $K_{\text{heavy}}$ indices where $K_{\text{heavy}}$ follows a pyramidal layer schedule:
  $$K_{\text{heavy}}(l) = K_{\text{base}} \times \left(1 + \frac{l}{L}\right)$$

### 3.3 Tier 1: CliffCompaction Semantic Agent Harness

At the agent loop boundary:
- Maintain conversation hash chain:
  $$H_0 = \text{sha256}(\text{system\_prompt})$$
  $$H_t = \text{sha256}(H_{t-1} \mathbin{\Vert} \text{turn}_t)$$
- When estimated token count exceeds threshold $\Theta = 16,384$:
  1. Extract $\text{Head} = \{\text{turn}_0, \text{turn}_1\}$ (system and user prompt).
  2. Extract $\text{Tail} = \{\text{turn}_{t-2}, \text{turn}_{t-1}, \text{turn}_t\}$ (last 3 turns).
  3. Intermediate turns $i \in [2, t-3]$ are mechanically compacted:
     - Tool results $> 500$ chars replaced with `"[output truncated: <sha256> ... N lines]"`
     - Tool calls compacted to function name and primary target file argument.
  4. Form compacted prompt: $\text{Prompt}_{\text{compact}} = \text{Head} \mathbin{\Vert} \text{Summary}(H_{\text{mid}}) \mathbin{\Vert} \text{Tail}$.
  5. The head and middle summary hash remain static across future turns, allowing the OpenXLA runtime to execute zero-recomputation prefix KV reuse.

---

## 4. Execution Harness & Silicon Verification Plan

### Step 1 — Fast-TurboQuant Implementation & TDD (`gate1_compression/tiered_turbo_kv`)

1. **FWHT AST Lowering**:
   - Implement `lower-fwht!` in `src/einsum/logic/lower.clj` using pure StableHLO addition and subtraction graph operators.
   - Generative Property Test: Verify FWHT round-trip orthogonality:
     $$\text{FWHT}(\text{FWHT}(\mathbf{x})) = d \cdot \mathbf{x}$$
2. **Lloyd-Max 2-bit & QJL Residual Operator**:
   - Implement fused `fast-turboquant-pack!` and `fast-turboquant-unpack!`.
   - Generative Property Test: Unbiased inner-product expectation test over $10^5$ random Gaussian vectors.
3. **KV Cache Buffer Compaction in Runtime**:
   - Extend `src/einsum/models/gemma4/kernels.clj` and `runtime.clj` to allocate 2-bit packed KV caches.

### Step 2 — Attention Sinks & Eviction Kernel

1. Implement index gather slicing for sink + heavy-hitter + local window in `runtime.clj`.
2. Verify needle-in-a-haystack accuracy across 8k, 16k, 32k, and 64k synthetic contexts.

### Step 3 — CliffCompaction Agent Harness Integration

1. Implement `compact-agent-history` in `src/einsum/agent/core.clj` with content-class truncation rules.
2. Verify prefix hash stability: assert prefix KV cache pointer reuse across 20 turns of agentic execution.

### Step 4 — Silicon Verification on AMD Radeon RX 7900 XTX

1. Launch full evaluation via `./tools/gemma4.sh run -M:tools -m experiments.gate1-compression.tiered-turbo-kv.run --backend rocm`.
2. Evaluate 128k context on `gemma-4-e4b-it-qat-int4` and `gemma-4-31b-it-qat-int4`.
3. Measure:
   - Peak VRAM footprint (GB)
   - Token decode throughput ($\text{tok/s}$)
   - MultiPL-E pass rate non-regression
   - M-NIAH needle retrieval accuracy

---

## 5. Decision Log

| Date | Event | Rationale |
|---|---|---|
| 2026-09-30 | `proposed` | Stage 1 RFC drafted. Following the NO-GO result of `perplexity_probe_v1` (confirming that E4B's gap is search dynamics rather than distribution blindness), Gate 1 pivot initiates memory compression for long-context deliberation. Synthesizes three breakthrough paradigms: CliffCompaction (Dettmers et al. 2026) at the agent loop, Attention Sinks + SnapKV at the sequence level, and Fast-TurboQuant (Pereira et al. 2026) multiplier-free FWHT + 2-bit Lloyd-Max + QJL at the silicon tensor level. Pre-registered target: >= 16x combined KV compression, enabling 128k context on 31B within 24GB VRAM. |
