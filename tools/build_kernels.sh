#!/usr/bin/env bash
set -euo pipefail

mkdir -p bin
HIPCC="${HIPCC:-hipcc}"
ARCH="${ROCM_ARCH:-gfx1100}"

echo "Compiling native RDNA3 kernels with $HIPCC for $ARCH..."
$HIPCC -O3 --offload-arch="$ARCH" -fPIC -shared \
  native/kernels/q_gemm_rdna3.cu \
  -o bin/libclj_einsum_kernels.so
echo "Successfully built bin/libclj_einsum_kernels.so"
