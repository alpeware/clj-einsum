# OpenXLA & PJRT Hardware & Compiler Limitations

This document provides an exhaustive, empirically verified technical specification of **OpenXLA** and **PJRT C-API** hardware limitations, compiler lowerings, native memory layouts, and signal interposition rules when accelerating LLMs with Clojure `clj-einsum`.

---

## 1. ⚡ Foreign Function & Memory (FFM) Struct Layouts & API ABI

OpenXLA PJRT plugins interact with host runtimes (Clojure JVM) through native C-API function pointers exposed via `PJRT_Api`. Strict compliance with native struct offsets and memory lifetimes is mandatory to prevent double-free segment faults and kernel driver crashes.

### `PJRT_ExecuteOptions` (112-Byte C-Struct)
In OpenXLA PJRT API version `0.112` / `24.0`, passing uninitialized bytes inside `PJRT_ExecuteOptions` causes native C++ plugins (e.g. `libpjrt_rocm.so`, `libpjrt_cuda.so`) to dereference invalid pointers.

| Offset (Bytes) | Data Type | Field Name | Description / Requirement |
| :--- | :--- | :--- | :--- |
| `0` | `int64_t` | `struct_size` | **Must be exactly 112**. Passing 144 bytes exposes internal uninitialized fields that crash the plugin. |
| `16` | `PJRT_Executable*` | `executable` | Handle to the compiled `PJRT_LoadedExecutable`. |
| `24` | `PJRT_ExecuteOptions*` | `options` | Pointer to zeroed `PJRT_ExecuteOptions` struct. |
| `32` | `PJRT_Buffer***` | `argument_lists` | Array of pointers to argument buffer lists. |
| `40` | `int64_t` | `num_args` | Number of input arguments per execution. |
| `48` | `int64_t` | `num_devices` | Number of execution target devices (typically `1`). |
| `56` | `PJRT_Buffer***` | `output_lists` | Array of pointers to output buffer lists (allocated with `num_outs` slots). |
| `64` | `PJRT_Event**` | `device_complete_events` | Pointer to allocated `PJRT_Event*` array (size `num_outs`). |

---

## 2. 🛑 Native Event Lifecycles & Double-Free Protection

OpenXLA manages the native lifetime of buffer execution events internally. Attempting manual destruction of execution output events leads to severe native memory corruption.

### Rules for `events-ptrs`:
1. **Allocation Size**: `events-ptrs` MUST be allocated with `num-outs` address slots (e.g., $71$ slots for Gemma 4 dual-graph output tuples). Allocating a single slot for multi-output graphs causes stack boundary corruption during native downcalls.
2. **Synchronization**: Call `PJRT_Event_Await` on `events-ptrs[0]` to synchronize host threads before reading device output buffers.
3. **No Manual Event Destruction**: Do **NOT** invoke `PJRT_Event_Destroy` on events returned inside `PJRT_LoadedExecutable_Execute_Args`. PJRT C++ runtimes free output completion events automatically. Explicit downcalls to `PJRT_Event_Destroy` trigger `SIGSEGV (0xb)` in `libhsa-runtime64.so` / `libc.so.6`.

---

## 3. 📐 128-Byte Hardware Off-Heap Memory Alignment

On modern GPU architectures (AMD RDNA3 / NVIDIA Hopper & Blackwell / Intel Xe-HPG), vectorized load/store instructions (e.g. AMDGPU `ds_read_b128` / `v_mov_b128`) require 128-byte hardware boundary alignment.

### Off-Heap Memory Invariants:
- **Independent Off-Heap Segments**: Key-Cache ($K_i$) and Value-Cache ($V_i$) host buffers uploaded via `buffer-from-host-buffer` MUST be backed by distinct, non-overlapping off-heap segments (`off-heap-k` and `off-heap-v`).
- **Alignment Requirement**: When allocating off-heap memory segments via Panama FFM `Arena`:
  ```clojure
  ;; CORRECT: Explicit 128-byte hardware alignment
  (let [off-heap-k (.allocate arena n-bytes 128)
        off-heap-v (.allocate arena n-bytes 128)]
    ...)
  ```
  Allocating default 8-byte aligned segments risks unaligned 128-bit SIMD vector reads during OpenXLA fused kernel execution, resulting in `ROCM_ERROR_ILLEGAL_ADDRESS`.

---

## 4. 🔀 Signal Chaining via `libjsig.so` (ROCm / Gentoo Linux Integration)

ROCm PJRT plugins (`libpjrt_rocm.so`) embed custom LLVM code generation engines that register native C signal handlers for `SIGSEGV`, `SIGBUS`, and `SIGILL`. These native handlers conflict with OpenJDK JVM signal handlers.

### Solution: JVM Signal Interposition
Before initializing the OpenXLA ROCm plugin via `System.load`, the JVM process MUST be launched with `LD_PRELOAD` pointing to `libjsig.so`:

```bash
./tools/gemma4.sh --backend rocm --model .models/gemma-4-E2B-it --prompt "..."
```

In `clj-einsum`, [`tools.gemma4-inference/needs-libjsig-reexec?`](../../tools/gemma4_inference.clj#L659) automatically detects missing `libjsig` interposition and re-executes the Clojure JVM process with `LD_PRELOAD` active.

---

## 5. 🧱 Compiler Lowering Limits: 32-Bit Indexing Boundary in `dynamic_update_slice`

When updating KV-cache tensors in StableHLO MLIR, `stablehlo.dynamic_update_slice` mutates a slice at dynamic position `%pos`:

```mlir
%updated = "stablehlo.dynamic_update_slice"(%cache, %update, %p0, %p1, %pos, %p3) 
           : (tensor<1x1x10240x256xf32>, tensor<1x1x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) 
           -> tensor<1x1x10240x256xf32>
```

### LLVM AMDGPU Lowering Limit
* **Un-chunked Index Boundary**: When sequence length $S > 2048$ tokens, OpenXLA GPU lowering generates single GCN block grid kernels.
* **32-Bit Pointer Overflow**: In un-chunked arrays, byte offset calculations (`pos * stride`) along sequence dimension ($S$) exceed 32-bit signed indexing limits inside AMDGPU LLVM lowering.
* **SOTA Solution**: Implement **PagedAttention Block Tables** or **Chunked 1024-Token Page Updates** to bound single-slice mutation offsets to 1024-token page boundaries.
