# Repository Rules & Coding Guidelines for AI Agents

* **Rule 1: Strict TDD.** Write generative tests (`clojure.test.check`) for invariants *before* implementing core logic.
* **Rule 2: Pure Functions.** The core must remain pure (Sans-IO). Side effects are strictly isolated to boundary shells.
* **Rule 3: Clean CI.** PRs must run `clojure -M:format`, `clojure -M:lint`, and tests (`clojure -M:test -m einsum.test-runner`) successfully before submission. Do not ignore linter warnings.
* **Rule 4: Zero Java Escape Hatches (Pure XLA Execution).** All tensor math, neural network layers, and full model forward passes MUST be written in pure Clojure using Pedro Domingos' Declarative Tensor Logic (`einsum.logic.*`) and compiled into StableHLO MLIR for OpenXLA execution via PJRT. Under NO circumstances should custom `.java` classes, host-side primitive float array loops (`float[][]`), or manual CPU matrix math engines be created to bypass XLA compilation.

---

## Optimal Inference & Agent Launching Guide

When launching inference or agent sessions, follow these operational patterns:

### 1. Launch via Wrapper Scripts (`libjsig.so` Chaining)
Always run inference and agent sessions through [`tools/gemma4.sh`](tools/gemma4.sh) or [`tools/gemma4_agent.sh`](tools/gemma4_agent.sh):
```bash
# Optimal Gemma 4 Text Generation
./tools/gemma4.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "Explain monads in Clojure"

# Optimal Gemma 4 Autonomous Agent Loop
./tools/gemma4.sh agent --backend rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and calculate total Clojure lines"
# Or using the agent symlink:
./tools/gemma4_agent.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "Inspect src/ and calculate total Clojure lines"
```
* **Why**: The OpenXLA PJRT ROCm plugin bundles LLVM, which installs native signal handlers that clash with the JVM's crash/safepoint signals unless `libjsig.so` is preloaded. `tools/gemma4.sh` automatically detects `libjsig.so` across JDK paths and exports `LD_PRELOAD`, preventing segmentation faults and avoiding in-process JVM re-execution.

### 2. ROCm Backend & Environment Autotuning
`einsum.core/determine-optimal-xla-flags` automatically sets required C runtime environment variables natively via Project Panama FFM (`setenv`) before PJRT client initialization:
- `HSA_OVERRIDE_GFX_VERSION` (defaults to `"11.0.0"` for RDNA3 / gfx1100 targets).
- `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` (defaults to `"0"` for discrete GPU selection).
- `TF_XLA_HSACO_CACHE_DIR` & `XLA_FLAGS` autotuning caches (`~/.cache/xla` and `~/.cache/hsa_cache`) with latency-hiding schedulers and disabled GPU command buffer overhead on ROCm.

To override target devices or architectures programmatically:
```clojure
(einsum.core/init-backend! :rocm {:gpu-device "0" :hsa-override-gfx-version "11.0.0"})
```

### 3. Agent VRAM Sessions (Persistent Memory & Dynamic Slicing)
For autonomous multi-turn agent loops, avoid recompiling graphs or reloading weights per turn:
- Use `init-agent-vram-session` from `tools.gemma4-inference` to pin model weights resident in PJRT VRAM once.
- Pre-compile a static-shape execution graph with `:max-seq-len` padding and `:last-token-only true`.
- Dynamic slice in-graph so only the exact active query slice is computed and transferred back to host, maintaining peak tok/s across multi-turn agent interactions.

### 4. EXL3 Quantized Models & RDNA3 Sequence Length Limits
EXL3 quantized models (such as `turboderp/gemma-4-12B-it-exl3` @ `3.00bpw_mul1`) can be downloaded directly from Hugging Face:
```bash
clojure -M:tools -m tools.download-hf turboderp/gemma-4-12B-it-exl3 --revision 3.00bpw_mul1
```
* **ROCm RDNA3 LDS Limits (gfx1100 / RX 7900 XTX)**: On AMD RDNA3 hardware, Local Data Share (shared memory) per workgroup is strictly limited to 64 KB (65,536 bytes). When compiling attention step graphs for models with `head_dim=512` (such as Gemma 4 12B full attention layers), keep `--max-seq-len` at or below 448 (e.g. `--max-seq-len 400`) to prevent OpenXLA from exceeding LDS limits (`requested 98304, available 65536`).
* **VRAM-Optimal Compilation Order**: For 12B models, PJRT graph compilation occurs *before* device weight allocation, preserving required memory workspace headroom for OpenXLA compilation.
