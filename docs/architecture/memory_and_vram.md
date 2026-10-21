# Memory Layout, VRAM Management & Signal Chaining

High-performance deep learning on consumer silicon demands strict mechanical sympathy between host CPU memory and device VRAM. This document specifies the memory architecture, arena lifecycle, RDNA3 hardware limits, and native signal handling in `clj-einsum`.

---

## 1. Mechanical Sympathy & Host vs. Device Boundary

`clj-einsum` enforces clear separation of concerns based on empirical hardware profiling:

```
┌────────────────────────────────────────────────────────┐
│                      Host Memory                       │
│  - Discrete graph traversals, trie lookups             │
│  - Schema validation (Malli), symbol registries        │
│  - Tokenization, commitment gates, agent state         │
│  - Zero-copy weight memory mapping (mmap)              │
└───────────────────────────┬────────────────────────────┘
                            │ PCIe DMA / hipHostRegister
                            ▼
┌────────────────────────────────────────────────────────┐
│                      Device VRAM                       │
│  - Dense tensor contractions (GEMM, GEMV)              │
│  - Attention projections, online streaming softmax     │
│  - RMSNorm, RoPE embeddings, gated activations         │
│  - In-VRAM autoregressive decode & ping-pong KV cache  │
└────────────────────────────────────────────────────────┘
```

- **Accelerators (GPU VRAM)**: Dense contractions, attention graphs, RMSNorm, RoPE embeddings, autoregressive token loops, and in-tensor relational memory execute inside pure-Clojure UOp schedules (`einsum.uop.*`) dispatched via Project Panama FFM (`libamdhip64.so` / `libhiprtc.so`) with zero host-device round-trips in the hot path.
- **Host (CPU Memory)**: Discrete index lookups, AST transformations, schema checks, and commitment gates execute in pure Clojure persistent data structures.
- **The E22 Grounding**: On a 64-entity discrete relation lookup, host hash indexing completes in **42.20 µs** vs. **691.94 µs** for in-graph device dynamic gather ($16.4\times$ host advantage). Discrete index lookups belong on the host.

---

## 2. UOp VRAM Arena Management (`einsum.runtime.arena`)

Device memory allocations are managed through hierarchical, scoped device arenas:

```
                  ┌──────────────────────────────┐
                  │        Session Arena         │
                  │  (Persistent Model Weights,  │
                  │   Static Ping-Pong Buffers)  │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │       Iteration Arena        │
                  │  (Transient Activation Maps, │
                  │   Intermediate Realizations) │
                  └──────────────────────────────┘
```

- **Scoped Lifecycles (`with-device-arena`)**: Memory buffers are scoped to arena blocks. At block exit, all untracked device buffers are automatically freed via `hipFree`.
- **Buffer Promotion (`arena/promote!`)**: Tokens and updated KV cache heads produced in an iteration arena are promoted into the persistent session arena before the iteration arena is reclaimed.
- **Explicit Tracking (`arena/track!`)**: Native memory allocations returned by kernel executions are registered with the active arena, ensuring leak-free cleanup even in the event of hardware or runtime exceptions.

---

## 3. Static-Shape Ping-Pong Step Buffers

Autoregressive token generation generates a new KV cache state at every step. Rather than continuously reallocating memory or updating dynamically sized buffers, `clj-einsum` employs **static-shape ping-pong buffers**:

```
Step 2k:   [Buffer A: Current KV]  ──(kernel dispatch)──>  [Buffer B: Next KV]
Step 2k+1: [Buffer B: Current KV]  ──(kernel dispatch)──>  [Buffer A: Next KV]
```

- **Pre-Allocation**: Two static buffers sized to the maximum sequence length are allocated inside the session arena during `init-agent-vram-session`.
- **Zero GC Overhead**: Pointers alternate between `Buffer A` and `Buffer B` per token step, eliminating intermediate VRAM allocations and host garbage collection pauses.
- **Graph Invariance**: Buffer shapes remain static throughout generation, preventing graph recompilation or pipeline flushes.

---

## 4. >4 GiB Zero-Copy Host Memory Registration (`hipHostRegister`)

Large-parameter models (such as Gemma 4 31B INT4, which requires 16.95 GB VRAM) exceed the standard 4 GiB single-allocation DMA chunk limits of legacy drivers:

- **Host Memory Pinning**: Model parameter files mapped into off-heap `MemorySegment` memory via Panama FFM are pinned and registered directly with the AMD ROCm driver via `hipHostRegister`:
  ```c
  hipHostRegister(ptr, size_bytes, hipHostRegisterMapped | hipHostRegisterPortable);
  ```
- **Direct DMA Transfers**: The GPU accesses pinned host memory directly over PCIe bus lines without intermediate heap staging or double-buffering.
- **Sub-Second Loading**: Ingests the complete 16.95 GB Gemma 4 31B weight set into device VRAM in under 1.2 seconds.

---

## 5. RDNA3 LDS Limits & Tile Budgets (AMD `gfx1100` / RX 7900 XTX)

Local Data Share (LDS / shared memory) per workgroup is strictly limited on AMD RDNA3 hardware:

- **Hardware Limit**: AMD RDNA3 Local Data Share per workgroup is strictly capped at **64 KB (65,536 bytes)**.
- **Compiler Tiling Budget**: The UOp tiling optimizer (`einsum.uop.opt`) enforces a conservative maximum allocation ceiling of **48 KB** per workgroup, leaving headroom for cooperative reduction buffers and compiler spill registers.
- **Bank-Conflict-Free Stride-18 Layout**:
  - AMD RDNA3 Compute Units contain 32 LDS banks (4 bytes per bank).
  - To prevent bank conflicts during Wave32 BF16 WMMA GEMM operations, weight and activation tiles are padded with a stride of 18 floats:
    ```c
    __shared__ float s_W[4][16][18];   // 4 * 16 * 18 * 4 = 4,608 bytes
    __shared__ float s_X[TM][16][18];  // TM * 16 * 18 * 4 <= 9,216 bytes (for TM <= 8)
    ```
  - Total shared memory for the GEMM kernel remains under 14 KB, allowing high workgroup occupancy across all 96 Compute Units.

---

## 6. Signal Chaining (`libjsig.so`)

The AMD ROCm HIP runtime (`libamdhip64.so` / `libhiprtc.so`) installs native POSIX signal handlers (`SIGSEGV`, `SIGBUS`, `SIGILL`) for hardware exception management. These native handlers can intercept signals utilized by the HotSpot JVM for thread safepoint checks and garbage collection barriers, leading to premature termination or deadlocks.

- **Mechanism**: Preloading `libjsig.so` via `LD_PRELOAD` intercepts signal handler installation and chains signals back to the JVM's internal signal dispatchers.
- **Automated Configuration**: The shell wrapper script [`tools/gemma4.sh`](../../tools/gemma4.sh) resolves JDK installation paths and automatically preloads `libjsig.so` along with required `--add-modules jdk.incubator.vector` and native access flags.
