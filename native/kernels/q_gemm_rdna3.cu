// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project, clj-einsum contributors
//
// W4A16 GEMV kernel for AMD RDNA3 (gfx1100 / RX 7900 XTX class), templated on
// activation dtype (half or __hip_bfloat16). Adapted from vLLM/exllamav2 for OpenXLA PJRT.
//
// De-torched and wrapped with OpenXLA PJRT GPU Custom Call ABI:
//   extern "C" void w4a16_gemv_rocm(hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace gptq_rdna3 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

template <>
__forceinline__ __device__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  float result = 0.0f;
  const half2* a2_ptr = (const half2*)a_ptr;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk4_bf16(bf16_t* addr, bf162_t v01,
                                                    bf162_t v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                             T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna3(const T* __restrict__ a,
                                     const uint32_t* __restrict__ b_q_weight,
                                     const uint32_t* __restrict__ b_qzeros,
                                     const T* __restrict__ b_scales,
                                     T* __restrict__ c, const int size_m,
                                     const int size_n, const int size_k,
                                     const int groups, const int zero_offset) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X (1 K element per thread)");

  constexpr bool USE_LDS_A = (M_COUNT > 1) || std::is_same<T, half>::value;
  if constexpr (USE_LDS_A) {
    if (offset_k + t < end_k) {
  #pragma unroll
      for (int m = 0; m < M_COUNT; ++m) {
        T av;
        if (offset_m + m < size_m) {
          const T* a_row = a + (offset_m + m) * size_k;
          av = a_row[offset_k + t];
        } else {
          av = tzero<T>();
        }
        block_a[m][t] = av;
      }
    }
    __syncthreads();
  }
  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  half2 z1z16_h[4][2], y1y16_h[4][2];
  float z_b_f[4], y_b_f[4];

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = b_qzeros ? (b_qzeros + g * (size_n / 8)) : nullptr;
    const T* sc_row = b_scales + g * size_n;
    int zeros[4];
    T scales[4];
    if (qz_row) {
      load4_zeros(qz_row, n, zeros);
    } else {
      zeros[0] = 8;
      zeros[1] = 8;
      zeros[2] = 8;
      zeros[3] = 8;
    }
    load4_scales<T>(sc_row, n, scales);
    if constexpr (std::is_same<T, half>::value) {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scales[i],
                             z1z16_h[i], y1y16_h[i]);
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_bf16_f32((uint32_t)(zeros[i] + zero_offset), scales[i],
                                 z_b_f[i], y_b_f[i]);
      }
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        half2 dq[4][4];
        dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0], y1y16_h[0]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1], y1y16_h[1]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2], y1y16_h[2]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3], y1y16_h[3]);

  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
        }
      } else if constexpr (M_COUNT == 1) {
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;  // bf162(128, 128)
        constexpr uint32_t BF16_ONES = 0x3F803F80u;   // bf162(1.0, 1.0)
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        pack4 a_pack;
        {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(a + offset_k + a_off);
          a_pack.u[0] = a_words[0];
          a_pack.u[1] = a_words[1];
          a_pack.u[2] = a_words[2];
          a_pack.u[3] = a_words[3];
        }

        float sum_a = 0.0f;
  #pragma unroll
        for (int b = 0; b < 4; ++b) {
          sum_a = __builtin_amdgcn_fdot2_f32_bf16(
              *((bf16x2_t*)(&a_pack.f[b])), *((const bf16x2_t*)&BF16_ONES),
              sum_a, /*clamp=*/false);
        }

  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

          float partial = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            partial = __builtin_amdgcn_fdot2_f32_bf16(
                *((bf16x2_t*)(&a_pack.f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                partial, /*clamp=*/false);
          }

          block_c[0][col] =
              __fmaf_rn(y_b_f[col], partial,
                        __fmaf_rn(z_b_f[col], sum_a, block_c[0][col]));
        }
      } else {
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;
        constexpr uint32_t BF16_ONES = 0x3F803F80u;
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        pack4 a_pack[M_COUNT];
  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(&block_a[m][a_off]);
          a_pack[m].u[0] = a_words[0];
          a_pack[m].u[1] = a_words[1];
          a_pack[m].u[2] = a_words[2];
          a_pack[m].u[3] = a_words[3];
        }

        float sum_a[M_COUNT];
  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          sum_a[m] = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            sum_a[m] = __builtin_amdgcn_fdot2_f32_bf16(
                *((bf16x2_t*)(&a_pack[m].f[b])),
                *((const bf16x2_t*)&BF16_ONES), sum_a[m], /*clamp=*/false);
          }
        }

  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

  #pragma unroll
          for (int m = 0; m < M_COUNT; ++m) {
            float partial = 0.0f;
  #pragma unroll
            for (int b = 0; b < 4; ++b) {
              partial = __builtin_amdgcn_fdot2_f32_bf16(
                  *((bf16x2_t*)(&a_pack[m].f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                  partial, /*clamp=*/false);
            }
            block_c[m][col] =
                __fmaf_rn(y_b_f[col], partial,
                          __fmaf_rn(z_b_f[col], sum_a[m], block_c[m][col]));
          }
        }
      }
    }
    k += 32;
  }

  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;
    T* out = c + (offset_m + m) * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      bf162_t r01;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      bf162_t r23;
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4_bf16(out, r01, r23);
    }
  }
}

#else

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna3(const T*, const uint32_t*, const uint32_t*,
                                     const T*, T*, const int, const int,
                                     const int, const int, const int) {}

#endif

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const T* b_scales,
                               T* c, int size_m, int size_n, int size_k,
                               int groups, int zero_offset,
                               hipStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna3<T, M_COUNT>
      <<<grid, block, 0, stream>>>(a, b_q_weight, b_qzeros, b_scales, c, size_m,
                                   size_n, size_k, groups, zero_offset);
}

template <typename T>
void launch_gemm_q4(const T* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const T* b_scales, T* c,
                    int size_m, int size_n, int size_k, int groups,
                    bool use_v2_format, hipStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  if (size_m == 1) {
    launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales, c,
                                    size_m, size_n, size_k, groups, zero_offset,
                                    stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales, c,
                                    size_m, size_n, size_k, groups, zero_offset,
                                    stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales, c,
                                    size_m, size_n, size_k, groups, zero_offset,
                                    stream);
  } else {
    launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales, c,
                                    size_m, size_n, size_k, groups, zero_offset,
                                    stream);
  }
}

}  // namespace gptq_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// OpenXLA PJRT GPU Custom Call Wrapper
// ABI: (hipStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
//
// buffers layout:
//   buffers[0]: a          (const __hip_bfloat16*, [M, K])
//   buffers[1]: b_q_weight (const uint32_t*, [K/8, N])
//   buffers[2]: b_qzeros   (const uint32_t*, [groups, N/8])
//   buffers[3]: b_scales   (const __hip_bfloat16*, [groups, N])
//   buffers[4]: c          (__hip_bfloat16*, [M, N]) - OUTPUT
//
// opaque config:
//   Either binary (4 x int32_t: {m, n, k, groups})
//   or ASCII string: "m,n,k,groups"
// ---------------------------------------------------------------------------

struct W4A16GemvDescriptor {
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t groups;
  int32_t has_zeros;
};

extern "C" void w4a16_gemv_rocm(hipStream_t stream,
                               void** buffers,
                               const char* opaque,
                               size_t opaque_len) {
  int32_t m = 1;
  int32_t n = 0;
  int32_t k = 0;
  int32_t groups = 1;
  int32_t has_zeros = 0;

  if (opaque_len == sizeof(W4A16GemvDescriptor)) {
    const auto* desc = reinterpret_cast<const W4A16GemvDescriptor*>(opaque);
    m = desc->m;
    n = desc->n;
    k = desc->k;
    groups = desc->groups;
    has_zeros = desc->has_zeros;
  } else if (opaque != nullptr && opaque_len > 0) {
    int parsed = std::sscanf(opaque, "%d,%d,%d,%d,%d", &m, &n, &k, &groups, &has_zeros);
    if (parsed < 5) {
      has_zeros = 0;
    }
  }

  const auto* a = reinterpret_cast<const __hip_bfloat16*>(buffers[0]);
  const auto* b_q_weight = reinterpret_cast<const uint32_t*>(buffers[1]);
  const uint32_t* b_qzeros = nullptr;
  const __hip_bfloat16* b_scales = nullptr;
  __hip_bfloat16* c = nullptr;

  if (has_zeros) {
    b_qzeros = reinterpret_cast<const uint32_t*>(buffers[2]);
    b_scales = reinterpret_cast<const __hip_bfloat16*>(buffers[3]);
    c = reinterpret_cast<__hip_bfloat16*>(buffers[4]);
  } else {
    // 3-input layout: (a, b_q_weight, b_scales) -> c
    // Implicit symmetric zero-point = 8, zero VRAM reads for qzeros!
    b_qzeros = nullptr;
    b_scales = reinterpret_cast<const __hip_bfloat16*>(buffers[2]);
    c = reinterpret_cast<__hip_bfloat16*>(buffers[3]);
  }

  // CRITICAL INVARIANT: The kernel writes via CAS-loop atomic adds into c.
  // XLA allocates output buffers uninitialized.
  // We MUST memset the output to zero on the stream before kernel launch!
  size_t out_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(__hip_bfloat16);
  (void)hipMemsetAsync(c, 0, out_bytes, stream);

  // Launch the tuned vLLM RDNA3 GEMV kernel
  vllm::gptq_rdna3::launch_gemm_q4<__hip_bfloat16>(
      a, b_q_weight, b_qzeros, b_scales, c,
      m, n, k, groups, /*use_v2_format=*/true, stream);
}
