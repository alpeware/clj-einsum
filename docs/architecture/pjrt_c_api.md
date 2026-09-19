# OpenXLA PJRT C API & Java 25 Panama FFM Bindings

## Overview

`clj-einsum` interacts directly with OpenXLA via the C API defined in `pjrt_c_api.h` using Java 25's Foreign Function & Memory (FFM) API (Project Panama). Zero intermediate JNI C/C++ wrappers or custom shared libraries are required.

## Key Architectural Principles

1. **Zero Java Escape Hatches (Rule 4)**: All tensor operations and models are lowered into StableHLO MLIR byte streams and compiled through the PJRT plugin into native GPU machine code (HSACO on ROCm, PTX/CUBIN on CUDA).
2. **Dynamic Symbol Resolution**:
   - The PJRT plugin `.so` is loaded using `SymbolLookup/libraryLookup`.
   - The plugin entry point `GetPjrtApi` is resolved to obtain the static `PJRT_Api` struct containing function pointers.
3. **ABI Versioning & Compatibility**:
   - Plugin ABI major and minor versions are checked on initialization (`PJRT_Api_Major` and `PJRT_Api_Minor`).
   - Detailed ABI notes are preserved in [docs/architecture/pjrt/](pjrt/).
4. **Memory Management**:
   - Java 25 `Arena/ofConfined` and `Arena/ofShared` manage off-heap native memory segments.
   - Pinned device buffers (`PJRT_Buffer`) reside in accelerator VRAM and can be transferred without intermediate host array copying.
