# E24 — In-VRAM Multi-Instance Handover: Cache-to-Cache (C2C) Semantic Communication Between Gemma 4 Instances

Date: 2026-09-18  
Status: SPEC  
Depends on: E22 (in-graph query dispatch), E23 (state reduce loop), `AGENT-LOOP.md` §9 (Tier 2 deliberation, Stage A / Stage B)  
Inspiration: *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* (Fu et al., 2025, arXiv:2510.03215) and Hacker News discussion #49758615.

---

## 1. Background & Motivation

In current multi-agent and multi-LLM architectures (AutoGen, CrewAI, LangChain, ChatCompletions), inter-model communication is mediated entirely by text:
1. Model 1 generates token-by-token.
2. Device-to-Host (D2H) transfer copies tokens to CPU.
3. Host tokenizer decodes token IDs into a Unicode string.
4. Host application formats the string into a new conversational prompt template.
5. Host tokenizer re-encodes the string into token IDs.
6. Host-to-Device (H2D) transfer copies the entire concatenated token array back to the GPU.
7. Model 2 **re-prefills the entire prompt and Model 1 output from scratch**, recomputing $O(L^2)$ attention and activations that Model 1 already computed!

This text-mediated handover suffers from three major flaws:
- **Severe Latency Wall**: Re-prefill recomputation on Model 2 scales quadratically or linearly with sequence length, adding 50–200 ms of redundant GPU work per handover, alongside serialization overhead.
- **Information Bottleneck**: Transforming rich continuous internal representations (KV-cache and hidden activations) into discrete vocabulary tokens destroys soft probability distributions, nuance, and uncertainty margins.
- **Host Context Churn**: The host CPU is burdened with string concatenation, memory allocations, and parsing at every turn.

Fu et al. (2025) proposed **Cache-to-Cache (C2C)** for heterogeneous models, using a neural projection network to transfer KV-caches directly.
On Hacker News (#49758615), commentators noted a profound corollary:
> *"What stops us then from producing a model family where all models are 'KV aligned', and each model can utilize the KV cache of other models directly? An 'expensive' reasoning model can use its full faculties to plan, but 'delegate' simple subgoals to a smaller model... with no prefill recompute and no associated 'handover' latency."*

When multiple instances belong to the **same model architecture** (such as $n$ instances of **Gemma 4 E2B**), the opportunity is even cleaner:
**Their KV-caches and internal representations are 100% natively aligned without needing any learned projection network.**

Because all $n$ instances share the same weights, head dimensions ($d_{\text{head}} = 256/512$), RoPE frequencies, and layer count ($35$ layers), Instance 2 can **directly inherit the VRAM-resident KV-cache and token state of Instance 1**.

---

## 2. Experimental Goal

Demonstrate **In-VRAM Multi-Instance Handover** with $n=2$ instances of Gemma 4 E2B executing $m=1$ conversational exchange entirely in GPU VRAM:
1. **Single-Prompt Ingress**: The host submits the user prompt to device VRAM once.
2. **Instance 1 (The Proposer / Drafter)**: Executes autoregressive generation in VRAM via the OpenXLA While-Loop, populating its KV-cache and token buffer.
3. **In-VRAM Handover ($m=1$)**: Instance 2 (The Critic / Refiner) inherits Instance 1's output state **directly within device memory**:
   - Zero token decode to host text.
   - Zero token re-encode from host text.
   - Zero re-prefill of Instance 1's generated prefix.
   - Handover latency $\le 2.0\text{ ms}$ (vs $>80\text{ ms}$ for host-mediated re-prefill).
4. **Instance 2 Execution**: Refines, verifies, and finalizes the solution in VRAM.
5. **Single-Response Egress**: Only the final verified response is transferred to host memory and returned to the user.

---

## 3. System Architecture & VRAM Mechanics

```
                       ┌──────────────────────────────┐
                       │       User Prompt (P)        │
                       └──────────────┬───────────────┘
                                      │ Host-to-Device (H2D)
                                      ▼
┌─────────────────────────────────── GPU VRAM ──────────────────────────────────────┐
│                                                                                   │
│   Shared Model Weights: Gemma 4 E2B (35 layers, bf16, ~4.6 GB resident once)       │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────────┐   │
│   │ [Turn 0] Instance 1: Proposer / Draftsman                                 │   │
│   │                                                                           │   │
│   │ 1. 1-Shot Parallel Prefill on Prompt P                                    │   │
│   │    -> Populates KV-Cache buffers KV_0..34 for 0..|P|                      │   │
│   │ 2. In-VRAM While-Loop Decode -> Generates Draft D (k1 tokens)             │   │
│   │    -> Extends KV-Cache buffers to 0..(|P|+k1)                             │   │
│   │    -> Token Buffer: B_toks = [P | D | 0 0 ... 0]                          │   │
│   └─────────────────────────────────────┬─────────────────────────────────────┘   │
│                                         │                                         │
│                      IN-VRAM HANDOVER   │ (Zero D2H copy, Zero text decode,       │
│                      (C2C KV Transfer)  │  Zero re-prefill of [P | D])            │
│                                         ▼                                         │
│   ┌───────────────────────────────────────────────────────────────────────────┐   │
│   │ [Turn 1] Instance 2: Verifier / Refiner                                   │   │
│   │                                                                           │   │
│   │ 1. In-VRAM Splicing of Transition Delimiter (Delta prefill: ~6 tokens)    │   │
│   │    -> KV-Cache updated for transition tokens only at position |P|+k1      │   │
│   │ 2. In-VRAM While-Loop Decode -> Generates Final Verified Answer A         │   │
│   │    -> Attends over the entire shared KV history [P | D | Transition]      │   │
│   │    -> Extends KV-Cache buffers to final length                            │   │
│   └─────────────────────────────────────┬─────────────────────────────────────┘   │
│                                         │                                         │
└─────────────────────────────────────────┼─────────────────────────────────────────┘
                                          │ Device-to-Host (D2H)
                                          ▼
                       ┌──────────────────────────────┐
                       │    Final Verified Response   │
                       └──────────────────────────────┘
```

### 3.1 Pinned Weight Sharing ($O(1)$ VRAM Memory)
Both Instance 1 and Instance 2 execute the same underlying OpenXLA executable (`executable` and `prefill-executable`) against the **same pinned device buffers** `device-weights` in PJRT VRAM. Total model weight memory footprint is strictly constant regardless of $n$.

### 3.2 In-VRAM Handover Protocols
We evaluate two implementation variants for the VRAM handover:
1. **Protocol A: In-VRAM C2C (Prefix KV Reuse)**
   - Instance 2 inherits the `kv-buffers` vector directly from Instance 1.
   - Instance 2 prefills *only* the short transition tokens (e.g. `<turn|>\n<|turn>user\nVerify and correct the draft:<turn|>\n<|turn>model\n`), inserting them at step index $p_1$.
   - Instance 2 launches its while loop starting from $p_2 = p_1 + \Delta$.
2. **Protocol B: In-VRAM Chained Step (Unified In-VRAM Multi-Turn Loop)**
   - A single composite While-Loop execution carrying turn-state registers `cur_instance`, `turn_counter`, and dynamic role delimiters in device memory.

---

## 4. Experimental Sweep: The 3 Comparative Cells

To isolate the causal benefits of in-VRAM communication and multi-instance deliberation, E24 defines three comparative cells:

| Cell | Architecture | Communication Medium | Prefill Behavior | Handover Latency |
|:---|:---|:---|:---|:---:|
| **Cell H** | **In-VRAM C2C ($n=2, m=1$)** | **VRAM Only** (Device PjRtBuffers) | Instance 1 prefills $P$; Instance 2 prefills **only delta transition** ($\sim 8$ tokens) | **$\le 2.0\text{ ms}$** |
| **Cell B1** | **Host-Mediated Text ($n=2, m=1$)** | **Host Text** (D2H $\to$ string $\to$ H2D) | Instance 1 prefills $P$; Instance 2 **re-prefills entire $[P + D]$** from scratch | **$80 - 150\text{ ms}$** |
| **Cell B0** | **Single-Instance Baseline ($n=1, m=0$)** | **None** (Direct Generation) | Single model pass without verification/deliberation | **N/A** |

---

## 5. Benchmark Task & Dataset: Multi-Step Constraint Reasoning

To rigorously evaluate both latency and collaborative accuracy gain, the benchmark consists of 30 structured reasoning problems requiring multi-step verification:

1. **GSM8K-Style Arithmetic & Word Problems (10 problems)**:
   Multi-step financial and resource calculations prone to off-by-one or arithmetic slip.
2. **Logic & Constraint Satisfaction (10 problems)**:
   Separation-of-Duty scheduling, seating constraints, and transitive dependency orderings.
3. **Factual & Relational Verification (10 problems)**:
   Entity attribution and role verification queries evaluated against ground truth.

### Instance Roles:
- **Instance 1 (Proposer / Drafter)**:
  `"You are a rigorous problem solver. Formulate a complete step-by-step reasoning draft to solve the problem."`
- **Instance 2 (Verifier / Refiner)**:
  `"You are an expert verifier. Inspect the proposed draft, verify each calculation and constraint step, correct any errors, and state the final answer clearly as 'Final Answer: <val>'."`

---

## 6. Pre-Registered Acceptance Criteria

| Criterion | Metric | Target Threshold | Rationale |
|:---|:---|:---|:---|
| **1. In-VRAM Handover Latency** | Handover time between Instance 1 and Instance 2 | **$\le 2.0\text{ ms}$** | Proves handover occurs in VRAM without host string serialization or full re-prefill. |
| **2. Handover Speedup Ratio** | Latency ratio: $\text{Handover}(B1) / \text{Handover}(H)$ | **$\ge 25\times$ Speedup** | Quantifies elimination of host roundtrip and prefill recomputation. |
| **3. End-to-End Latency Speedup** | Total turn latency: $\text{Latency}(B1) - \text{Latency}(H)$ | **$\ge 50\text{ ms}$ savings** per query turn | Demonstrates user-perceptible latency reduction. |
| **4. Deliberation Accuracy Gain** | Task accuracy: $\text{Acc}(H) \text{ vs } \text{Acc}(B0)$ | **$\text{Acc}(H) - \text{Acc}(B0) \ge +15.0\%$** | Proves that 2 communicating instances outperform single-instance generation. |
| **5. C2C Parity** | Output semantic equivalence between Cell H and Cell B1 | **$\ge 90.0\%$ answer agreement** | Validates that VRAM KV-cache transfer preserves full conversational semantics. |
| **6. VRAM Footprint Invariance** | Weight memory allocated in device VRAM | **$= 4.6\text{ GB}$ (Single Model Footprint)** | Proves $n$ instances share device weights in PJRT memory. |

---

## 7. Scaling Horizon ($n$ Instances, $m$ Rounds)

Following validation of the base configuration ($n=2, m=1$), the runner supports parametric expansion:
- $n = 2, m \in \{1, 2, 3\}$ (Multi-turn iterative debate/refinement).
- $n = 3, m = 1$ (Generator $\to$ Critic $\to$ Arbiter).

All intermediate turns execute strictly within device memory, returning only the final converged answer to host.
