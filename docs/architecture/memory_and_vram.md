# Memory Layout, VRAM Management & Signal Chaining

## 1. Mechanical Sympathy & Host vs. Device Boundary

`clj-einsum` enforces clear separation of concerns based on empirical hardware profiles:
- **Accelerators (GPU VRAM)**: Dense contractions, attention projections, RMSNorm, RoPE, autoregressive token loops, and KV cache mutations execute inside OpenXLA StableHLO.
- **Host (CPU Memory)**: Discrete graph traversals, trie lookups, symbol registries, schema checks, and commitment gates execute in Clojure.
- **The E22 Grounding**: On a 64-entity discrete relation lookup, host hash indexing completes in **42.20 µs** vs. **691.94 µs** for in-graph device dynamic gather ($16.4\times$ host advantage). Discrete index lookups belong on the host.

## 2. RDNA3 LDS Limits (AMD gfx1100 / RX 7900 XTX)

- **Hardware Limit**: AMD RDNA3 Local Data Share (LDS / shared memory) per workgroup is strictly capped at 64 KB (65,536 bytes).
- **Compilation Constraint**: In attention blocks with `head_dim=512` (such as Gemma 4 12B full attention layers), sequence lengths exceeding 448 can cause OpenXLA to request up to 98,304 bytes of shared memory, leading to compilation failure (`requested 98304, available 65536`).
- **Mitigation**: Cap `--max-seq-len` to 448 (or 400) for large head-dimension compilation on RDNA3 targets.

## 3. Signal Chaining (`libjsig.so`)

The OpenXLA PJRT ROCm plugin bundles an internal LLVM distribution that installs native POSIX signal handlers (`SIGSEGV`, `SIGBUS`, `SIGILL`). These conflict with JVM safepoint checks and garbage collection signals, leading to crashes unless chained properly.
- **Mechanism**: Preloading `libjsig.so` via `LD_PRELOAD` intercepts signal handler installation and chains signals back to HotSpot JVM.
- **Launcher**: [`tools/gemma4.sh`](../../tools/gemma4.sh) detects JDK installation paths and automatically preloads `libjsig.so`.
