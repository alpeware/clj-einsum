# Repository Rules & Coding Guidelines for AI Agents

> **Read [`PROCESS.md`](PROCESS.md) first.** It is the single source of truth for the Staircase milestones, 4 RSI gates, and lab operating rules.

## The 4 Non-Negotiable Invariants

* **Rule 1: Strict TDD.** Write generative tests (`clojure.test.check`) for invariants *before* implementing core logic.
* **Rule 2: Pure Functions.** The core must remain pure (Sans-IO). Side effects are strictly isolated to boundary shells.
* **Rule 3: Clean Local Commits.** Run format (`clojure -M:format`), lint (`clojure -M:lint`), and tests (`clojure -M:test fast` or `clojure -M:test all`) successfully before committing. Zero errors and zero linter warnings allowed.
* **Rule 4: Maximize Mechanical Sympathy (Pure UOp Device Core + Host Pragmatism).** Place workloads where they run best: dense contractions, neural network layers, attention graphs, and autoregressive generation loops compile via Declarative Tensor Logic (`einsum.logic.*`) into pure-Clojure UOp schedules (`einsum.uop.*`), rendering directly to HIP/C99 kernels dispatched on-device via Panama FFM (`libamdhip64.so` / `libhiprtc.so`) with zero host-device round-trips in the hot path. Discrete index lookups, schema checks, tokenization, and commitment gates run host-side in pure Clojure persistent data structures. Do not build custom host matrix-math engines or ad-hoc Java/C++ FFI classes to bypass UOp compilation, but do not force host-suited discrete operations onto device memory.

---

## Multi-Agent Direct-Commit Protocol

We do not use GitHub Pull Requests. Multiple agents work concurrently in this local repository and commit directly to `main`:

1. **Commit Message Format**:
   `[<gate>/<slug>] <stage>: <brief description>`
   - Examples:
     - `[gate1/cat-q] stage1-spec: pre-register ternary unpack falsification criteria`
     - `[gate1/cat-q] stage2-impl: implement uop ternary unpack and run benchmarks`
     - `[gate1/cat-q] stage3-verify: replicate throughput on rx 7900 xtx and catalog pod`
     - `[core/engine] fix: resolve panama off-heap arena leak in dynamic slice`
2. **Atomic Commits & Pre-Commit Invariant**:
   Never commit broken code or untested features. Every commit MUST pass `clojure -M:format`, `clojure -M:lint`, and `clojure -M:test fast`.
3. **Directory Ownership**:
   - `src/einsum/` — Core engine, tensor logic, pure-Clojure UOp compiler and Panama FFM runtime (`einsum.uop.*`), promoted models.
   - `src/tools/` — CLI entrypoints and model inference drivers (`tools.*` namespaces).
   - `src/experiments/<gate>/<slug>/` — Executable research code for active experiments.
   - `resources/proposals/<gate>/<slug>/` — RFC specs (`spec.md`), raw metrics (`results.edn`), summaries (`summary.csv`).
   - `resources/catalog/<gate>/<slug>/` — Verified research pods and master [`registry.edn`](resources/catalog/registry.edn).
   - `resources/golden/` — Post-XLA regression oracle [`gemma4_uop_oracle.edn`](resources/golden/gemma4_uop_oracle.edn) and archived final vs-XLA benchmark comparison [`xla_vs_uop_archive.edn`](resources/golden/xla_vs_uop_archive.edn).
   - `resources/data/` — Static corpora and test fixtures (loaded via `clojure.java.io/resource`).
   - `tools/` — Shell wrapper scripts ONLY (`gemma4.sh`, `benchmark.sh`, `quantize.sh`).

---

## Optimal Inference & Hardware Invariants

1. **Launch via Wrapper Scripts (`libjsig.so` Chaining)**:
   Always run inference and agent sessions through [`tools/gemma4.sh`](tools/gemma4.sh):
   ```bash
   ./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"
   ./tools/gemma4.sh --backend uop-rocm --model .models/gemma-4-31b-it-int4 --prompt "Explain monads in Clojure"
   ./tools/gemma4.sh agent --backend uop-rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and calculate total Clojure lines"
   ```
   *Why*: The pure-Clojure UOp engine utilizes Java 22+ Panama FFM (`java.lang.foreign`) and `jdk.incubator.vector` for direct dispatch into ROCm HIP (`libamdhip64.so` / `libhiprtc.so`). `tools/gemma4.sh` configures native JVM module exports and optionally exports `LD_PRELOAD=libjsig.so` to chain ROCm runtime signals safely with JVM safepoint signals.

2. **ROCm RDNA3 LDS Limits (gfx1100 / RX 7900 XTX)**:
   Local Data Share (LDS) per workgroup is strictly limited to 64 KB (65,536 bytes). When compiling UOp cooperative reductions and Wave32 BF16 WMMA GEMM tiles (`s_W[4][16][18]` + `s_X[TM][16][18]`), total shared memory allocation must remain within 64 KB to maintain workgroup occupancy across Compute Units.

3. **Persistent VRAM Sessions**:
   For multi-turn agent loops, pin weights resident in UOp ROCm VRAM once using `init-agent-vram-session` and allocate static-shape ping-pong step buffers (`run-cached-kv-generation`) to eliminate intermediate host allocations and prevent graph recompilation.
