# Repository Rules & Coding Guidelines for AI Agents

> **Read [`PROCESS.md`](PROCESS.md) first.** It is the single source of truth for the Staircase milestones, 4 RSI gates, and lab operating rules.

## The 4 Non-Negotiable Invariants

* **Rule 1: Strict TDD.** Write generative tests (`clojure.test.check`) for invariants *before* implementing core logic.
* **Rule 2: Pure Functions.** The core must remain pure (Sans-IO). Side effects are strictly isolated to boundary shells.
* **Rule 3: Clean Local Commits.** Run format (`clojure -M:format`), lint (`clojure -M:lint`), and tests (`clojure -M:test fast` or `clojure -M:test all`) successfully before committing. Zero errors and zero linter warnings allowed.
* **Rule 4: Maximize Mechanical Sympathy (Pure XLA Device Core + Host Pragmatism).** Place workloads where they run best: dense contractions, neural network layers, attention graphs, and autoregressive generation loops compile via Declarative Tensor Logic (`einsum.logic.*`) into StableHLO MLIR for in-accelerator OpenXLA PJRT execution (zero host-device round-trips in the hot path). Discrete index lookups, schema checks, tokenization, and commitment gates run host-side in pure Clojure persistent data structures. Do not build custom host matrix-math engines or ad-hoc Java classes to bypass XLA compilation, but do not force host-suited discrete operations onto device memory.

---

## Multi-Agent Direct-Commit Protocol

We do not use GitHub Pull Requests. Multiple agents work concurrently in this local repository and commit directly to `main`:

1. **Commit Message Format**:
   `[<gate>/<slug>] <stage>: <brief description>`
   - Examples:
     - `[gate1/cat-q] stage1-spec: pre-register ternary unpack falsification criteria`
     - `[gate1/cat-q] stage2-impl: implement stablehlo unpack and run benchmarks`
     - `[gate1/cat-q] stage3-verify: replicate throughput on rx 7900 xtx and catalog pod`
     - `[core/engine] fix: resolve panama off-heap arena leak in dynamic slice`
2. **Atomic Commits & Pre-Commit Invariant**:
   Never commit broken code or untested features. Every commit MUST pass `clojure -M:format`, `clojure -M:lint`, and `clojure -M:test fast`.
3. **Directory Ownership**:
   - `src/einsum/` — Core engine, tensor logic, PJRT Panama bindings, promoted models.
   - `src/tools/` — CLI entrypoints and model inference drivers (`tools.*` namespaces).
   - `src/experiments/<gate>/<slug>/` — Executable research code for active experiments.
   - `resources/proposals/<gate>/<slug>/` — RFC specs (`spec.md`), raw metrics (`results.edn`), summaries (`summary.csv`).
   - `resources/catalog/<gate>/<slug>/` — Verified research pods and master [`registry.edn`](resources/catalog/registry.edn).
   - `resources/data/` — Static corpora and test fixtures (loaded via `clojure.java.io/resource`).
   - `tools/` — Shell wrapper scripts ONLY (`gemma4.sh`, `benchmark.sh`, `quantize.sh`).

---

## Optimal Inference & Hardware Invariants

1. **Launch via Wrapper Scripts (`libjsig.so` Chaining)**:
   Always run inference and agent sessions through [`tools/gemma4.sh`](tools/gemma4.sh):
   ```bash
   ./tools/gemma4.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"
   ./tools/gemma4.sh agent --backend rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and calculate total Clojure lines"
   ```
   *Why*: The OpenXLA PJRT ROCm plugin bundles LLVM, which installs native signal handlers that clash with the JVM's crash/safepoint signals unless `libjsig.so` is preloaded. `tools/gemma4.sh` automatically exports `LD_PRELOAD`, preventing segmentation faults.

2. **ROCm RDNA3 LDS Limits (gfx1100 / RX 7900 XTX)**:
   Local Data Share (LDS) per workgroup is strictly limited to 64 KB (65,536 bytes). When compiling attention graphs for models with `head_dim=512` (such as Gemma 4 12B full attention layers), keep `--max-seq-len` at or below 448 to prevent compilation failure (`requested 98304, available 65536`).

3. **Persistent VRAM Sessions**:
   For multi-turn agent loops, pin weights resident in PJRT VRAM once using `init-agent-vram-session` and compile static-shape graphs with `:max-seq-len` padding to avoid recompilation.
