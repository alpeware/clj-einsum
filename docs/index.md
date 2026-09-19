# Architectural Documentation & Papers

Welcome to the `clj-einsum` architectural documentation and research catalog exports.

## Directory Structure

- [**architecture/**](architecture/): Substrate engineering notes and technical specifications.
  - [pjrt_c_api.md](architecture/pjrt_c_api.md): Java 25 Project Panama FFM bindings to OpenXLA PJRT C API (`pjrt_c_api.h`), ABI versioning, and lifecycle management.
  - [memory_and_vram.md](architecture/memory_and_vram.md): VRAM layout, RDNA3 LDS limits, mechanical sympathy, and `libjsig.so` signal chaining.
  - [DESIGN.md](architecture/DESIGN.md): Overall compiler architecture and execution pipeline.
  - [TENSOR_LOGIC.md](architecture/TENSOR_LOGIC.md): Datalog-as-contraction semantics and Hiccup AST specification.
  - [agent_loop.md](architecture/agent_loop.md): In-VRAM autonomous multi-turn agent loop.
- [**papers/**](papers/): ArXiv paper exports compiled directly from verified catalog research arcs.
  - [paper1_executable_einsum/](papers/paper1_executable_einsum/): Executable generalized semiring einsum on accelerators.
  - [paper2_trainable_limits/](papers/paper2_trainable_limits/): Negative gradient results on predicate invention (Rankings != Truth).
