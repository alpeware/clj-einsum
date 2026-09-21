#!/usr/bin/env python3
"""
JAX Workload Benchmark Suite for clj-xla Parity & Performance Gap Verification.

Measures JAX JIT compilation latency, execution warmup, mean/P99 latencies,
and FLOPs/TFLOPS for standard ML kernels and transformer layer blocks across CPU and SYCL backends.
"""

import sys
import time
import json
import argparse
import numpy as np

def setup_jax_backend(backend_name):
    from jax._src.lib import xla_client
    import jax
    import jax.numpy as jnp

    target_dev = None
    if backend_name == "sycl":
        try:
            xla_client.load_pjrt_plugin_dynamically("sycl", "bin/libpjrt_sycl.so")
            client = xla_client.make_c_api_client("sycl", {"allocator": "platform"})
            target_dev = client.devices()[0]
        except Exception as e:
            print(f"Warning initializing SYCL C API client: {e}", flush=True)
    elif backend_name == "rocm":
        try:
            xla_client.load_pjrt_plugin_dynamically("rocm", "bin/libpjrt_rocm.so")
            client = xla_client.make_c_api_client("rocm", {"allocator": "platform"})
            target_dev = client.devices()[0]
        except Exception as e:
            print(f"Warning initializing ROCm C API client: {e}", flush=True)
    elif backend_name == "cuda12" or backend_name == "cuda":
        try:
            xla_client.load_pjrt_plugin_dynamically("cuda", "bin/libpjrt_cuda.so")
            client = xla_client.make_c_api_client("cuda", {"allocator": "platform"})
            target_dev = client.devices()[0]
        except Exception as e:
            print(f"Warning initializing CUDA C API client: {e}", flush=True)

    if target_dev is None:
        target_dev = jax.devices("cpu")[0]

    return jax, jnp, target_dev

def calculate_stats(latencies_ms):
    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    p50_idx = int(n * 0.50)
    p90_idx = int(n * 0.90)
    p99_idx = int(min(n - 1, int(n * 0.99)))
    mean_ms = sum(sorted_ms) / n
    variance = sum((x - mean_ms) ** 2 for x in sorted_ms) / n
    std_dev_ms = variance ** 0.5
    return {
        "min_ms": round(sorted_ms[0], 3),
        "max_ms": round(sorted_ms[-1], 3),
        "mean_ms": round(mean_ms, 3),
        "std_dev_ms": round(std_dev_ms, 3),
        "p50_ms": round(sorted_ms[p50_idx], 3),
        "p90_ms": round(sorted_ms[p90_idx], 3),
        "p99_ms": round(sorted_ms[p99_idx], 3)
    }

def benchmark_fn(name, jit_fn, args, warmup_iters=5, measure_iters=50, flops=None):
    # Measure Cold JIT Compile Time
    t0 = time.perf_counter()
    res = jit_fn(*args)
    res.block_until_ready()
    t1 = time.perf_counter()
    cold_jit_ms = (t1 - t0) * 1000.0

    # Warmup Passes
    for _ in range(warmup_iters):
        res = jit_fn(*args)
        res.block_until_ready()

    # Measured Iterations
    latencies = []
    for _ in range(measure_iters):
        st = time.perf_counter()
        res = jit_fn(*args)
        res.block_until_ready()
        et = time.perf_counter()
        latencies.append((et - st) * 1000.0)

    stats = calculate_stats(latencies)
    stats["cold_jit_ms"] = round(cold_jit_ms, 3)

    if flops:
        mean_sec = stats["mean_ms"] / 1000.0
        tflops = (flops / mean_sec) / 1e12
        stats["tflops"] = round(tflops, 3)
    else:
        stats["tflops"] = 0.0

    return stats

def dev_put(tensor, jax, dev):
    return jax.device_put(tensor, dev)

# --- Workload Definitions in JAX ---

def make_gemm_fp32(jnp, jax, dev):
    m, k, n = 1024, 1024, 1024
    a = dev_put(jnp.ones((m, k), dtype=jnp.float32), jax, dev)
    b = dev_put(jnp.ones((k, n), dtype=jnp.float32), jax, dev)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-fp32", run, (a, b), flops

def make_gemm_bf16(jnp, jax, dev):
    m, k, n = 1024, 1024, 1024
    a = dev_put(jnp.ones((m, k), dtype=jnp.bfloat16), jax, dev)
    b = dev_put(jnp.ones((k, n), dtype=jnp.bfloat16), jax, dev)
    flops = 2 * m * n * k
    @jax.jit
    def run(a_t, b_t):
        return jnp.matmul(a_t, b_t)
    return "gemm-bf16", run, (a, b), flops

def make_rms_norm(jnp, jax, dev):
    b, s, d = 1, 2048, 4096
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    gamma = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    @jax.jit
    def run(x_t, g_t):
        ms = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        return (x_t / jnp.sqrt(ms + 1e-6)) * g_t
    return "rms-norm", run, (x, gamma), None

def make_swiglu(jnp, jax, dev):
    b, s, d = 1, 2048, 4096
    inter = 4 * d
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    w_gate = dev_put(jnp.ones((inter, d), dtype=jnp.float32), jax, dev)
    w_up = dev_put(jnp.ones((inter, d), dtype=jnp.float32), jax, dev)
    flops = 2 * (2 * b * s * d * inter)

    @jax.jit
    def run(x_t, wg, wu):
        gate = jax.nn.silu(jnp.matmul(x_t, wg.T))
        up = jnp.matmul(x_t, wu.T)
        return gate * up

    return "swiglu", run, (x, w_gate, w_up), flops

def make_gqa_causal_attn(jnp, jax, dev):
    b, seq, h_q, h_kv, d_k = 1, 128, 8, 1, 256
    q_dim = h_q * d_k
    kv_dim = h_kv * d_k
    hidden_dim = q_dim

    x = dev_put(jnp.ones((b, seq, hidden_dim), dtype=jnp.float32), jax, dev)
    qw = dev_put(jnp.ones((q_dim, hidden_dim), dtype=jnp.float32), jax, dev)
    kw = dev_put(jnp.ones((kv_dim, hidden_dim), dtype=jnp.float32), jax, dev)
    vw = dev_put(jnp.ones((kv_dim, hidden_dim), dtype=jnp.float32), jax, dev)
    ow = dev_put(jnp.ones((hidden_dim, q_dim), dtype=jnp.float32), jax, dev)
    qn = dev_put(jnp.ones((d_k,), dtype=jnp.float32), jax, dev)
    kn = dev_put(jnp.ones((d_k,), dtype=jnp.float32), jax, dev)

    flops = 2 * (b * seq * hidden_dim * q_dim + 2 * b * seq * hidden_dim * kv_dim + b * seq * q_dim * hidden_dim)

    @jax.jit
    def run(x_t, q_w, k_w, v_w, o_w, q_n, k_n):
        q_proj = jnp.matmul(x_t, q_w.T).reshape(b, seq, h_q, d_k)
        k_proj = jnp.matmul(x_t, k_w.T).reshape(b, seq, h_kv, d_k)
        v_proj = jnp.matmul(x_t, v_w.T).reshape(b, seq, h_kv, d_k)

        # Per-head RMSNorm
        q_ms = jnp.mean(q_proj ** 2, axis=-1, keepdims=True)
        k_ms = jnp.mean(k_proj ** 2, axis=-1, keepdims=True)
        q_norm = (q_proj / jnp.sqrt(q_ms + 1e-6)) * (1.0 + q_n)
        k_norm = (k_proj / jnp.sqrt(k_ms + 1e-6)) * (1.0 + k_n)

        # Repeat KV heads for GQA
        k_exp = jnp.repeat(k_norm, h_q // h_kv, axis=2).transpose(0, 2, 3, 1)
        v_exp = jnp.repeat(v_proj, h_q // h_kv, axis=2).transpose(0, 2, 1, 3)
        q_trans = q_norm.transpose(0, 2, 1, 3)

        # Multi-Head Causal Attention
        scores = jnp.matmul(q_trans, k_exp) / (d_k ** 0.5)
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(probs, v_exp).transpose(0, 2, 1, 3).reshape(b, seq, q_dim)
        return jnp.matmul(context, o_w.T)

    return "gqa-causal-attn", run, (x, qw, kw, vw, ow, qn, kn), flops

def make_gpt2_block(jnp, jax, dev):
    b, s, d = 1, 128, 768
    num_heads = 12
    head_dim = d // num_heads
    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    ln1_g = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    ln1_b = dev_put(jnp.zeros((d,), dtype=jnp.float32), jax, dev)
    c_attn_w = dev_put(jnp.ones((d, 3 * d), dtype=jnp.float32), jax, dev)
    c_attn_b = dev_put(jnp.zeros((3 * d,), dtype=jnp.float32), jax, dev)
    c_proj_w = dev_put(jnp.ones((d, d), dtype=jnp.float32), jax, dev)
    c_proj_b = dev_put(jnp.zeros((d,), dtype=jnp.float32), jax, dev)
    ln2_g = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    ln2_b = dev_put(jnp.zeros((d,), dtype=jnp.float32), jax, dev)
    mlp_fc_w = dev_put(jnp.ones((d, 4 * d), dtype=jnp.float32), jax, dev)
    mlp_fc_b = dev_put(jnp.zeros((4 * d,), dtype=jnp.float32), jax, dev)
    mlp_proj_w = dev_put(jnp.ones((4 * d, d), dtype=jnp.float32), jax, dev)
    mlp_proj_b = dev_put(jnp.zeros((d,), dtype=jnp.float32), jax, dev)
    flops = 2 * (b * s * d * (3 * d) + b * s * d * d + b * s * d * (4 * d) + b * s * (4 * d) * d)

    @jax.jit
    def run(x_t, g1, b1, c_attn, cb, c_proj, pb, g2, b2, mlp_fc, fcb, mlp_proj, pb2):
        # LayerNorm 1
        m1 = jnp.mean(x_t, axis=-1, keepdims=True)
        v1 = jnp.mean((x_t - m1) ** 2, axis=-1, keepdims=True)
        ln1 = (x_t - m1) / jnp.sqrt(v1 + 1e-5) * g1 + b1
        # QKV Projection
        qkv = jnp.matmul(ln1, c_attn) + cb
        q = qkv[:, :, :d].reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, d:2*d].reshape(b, s, num_heads, head_dim).transpose(0, 2, 3, 1)
        v = qkv[:, :, 2*d:].reshape(b, s, num_heads, head_dim).transpose(0, 2, 1, 3)
        # Multi-Head Causal Self-Attention
        scores = jnp.matmul(q, k) / (head_dim ** 0.5)
        probs = jax.nn.softmax(scores, axis=-1)
        attn_context = jnp.matmul(probs, v).transpose(0, 2, 1, 3).reshape(b, s, d)
        attn_out = jnp.matmul(attn_context, c_proj) + pb
        res1 = x_t + attn_out
        # LayerNorm 2 & MLP
        m2 = jnp.mean(res1, axis=-1, keepdims=True)
        v2 = jnp.mean((res1 - m2) ** 2, axis=-1, keepdims=True)
        ln2 = (res1 - m2) / jnp.sqrt(v2 + 1e-5) * g2 + b2
        fc = jax.nn.gelu(jnp.matmul(ln2, mlp_fc) + fcb)
        mlp_out = jnp.matmul(fc, mlp_proj) + pb2
        return res1 + mlp_out

    args = (x, ln1_g, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_g, ln2_b, mlp_fc_w, mlp_fc_b, mlp_proj_w, mlp_proj_b)
    return "gpt2-block", run, args, flops

def make_gemma4_block(jnp, jax, dev):
    b, s, d = 1, 128, 1536
    num_heads, num_kv_heads, head_dim = 8, 1, 256
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    mlp_dim = 4 * d
    pl_dim = 256

    x = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)
    in_ln = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    ls = dev_put(jnp.ones((1,), dtype=jnp.float32), jax, dev)
    qw = dev_put(jnp.ones((q_dim, d), dtype=jnp.float32), jax, dev)
    kw = dev_put(jnp.ones((kv_dim, d), dtype=jnp.float32), jax, dev)
    vw = dev_put(jnp.ones((kv_dim, d), dtype=jnp.float32), jax, dev)
    ow = dev_put(jnp.ones((d, q_dim), dtype=jnp.float32), jax, dev)
    qn = dev_put(jnp.ones((head_dim,), dtype=jnp.float32), jax, dev)
    kn = dev_put(jnp.ones((head_dim,), dtype=jnp.float32), jax, dev)
    post_attn = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    pre_mlp = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    post_mlp = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    gw = dev_put(jnp.ones((mlp_dim, d), dtype=jnp.float32), jax, dev)
    uw = dev_put(jnp.ones((mlp_dim, d), dtype=jnp.float32), jax, dev)
    dw = dev_put(jnp.ones((d, mlp_dim), dtype=jnp.float32), jax, dev)
    plg = dev_put(jnp.ones((pl_dim, d), dtype=jnp.float32), jax, dev)
    plp = dev_put(jnp.ones((d, pl_dim), dtype=jnp.float32), jax, dev)
    pln = dev_put(jnp.ones((d,), dtype=jnp.float32), jax, dev)
    pl_in = dev_put(jnp.ones((b, s, d), dtype=jnp.float32), jax, dev)

    flops = 2 * (b * s * d * 2048 + b * s * d * 256 + b * s * d * 256 + b * s * 2048 * d + 2 * b * s * d * 6144 + b * s * 6144 * d)

    @jax.jit
    def run(x_t, in_ln_w, ls_w, q_w, k_w, v_w, o_w, q_n, k_n, post_attn_w, pre_mlp_w, post_mlp_w, g_w, u_w, d_w, plg_w, plp_w, pln_w, plin):
        # 1. Input RMSNorm (1+w)
        ms1 = jnp.mean(x_t ** 2, axis=-1, keepdims=True)
        norm1 = (x_t / jnp.sqrt(ms1 + 1e-6)) * (1.0 + in_ln_w)
        # 2. Multi-Head GQA Attention
        q_proj = jnp.matmul(norm1, q_w.T).reshape(b, s, num_heads, head_dim)
        k_proj = jnp.matmul(norm1, k_w.T).reshape(b, s, num_kv_heads, head_dim)
        v_proj = jnp.matmul(norm1, v_w.T).reshape(b, s, num_kv_heads, head_dim)
        # Per-head RMSNorm (1+w)
        q_ms = jnp.mean(q_proj ** 2, axis=-1, keepdims=True)
        k_ms = jnp.mean(k_proj ** 2, axis=-1, keepdims=True)
        q_norm = (q_proj / jnp.sqrt(q_ms + 1e-6)) * (1.0 + q_n)
        k_norm = (k_proj / jnp.sqrt(k_ms + 1e-6)) * (1.0 + k_n)
        # Repeat KV heads for GQA
        k_exp = jnp.repeat(k_norm, num_heads // num_kv_heads, axis=2).transpose(0, 2, 3, 1)
        v_exp = jnp.repeat(v_proj, num_heads // num_kv_heads, axis=2).transpose(0, 2, 1, 3)
        q_trans = q_norm.transpose(0, 2, 1, 3)
        # Multi-Head Attention
        scores = jnp.matmul(q_trans, k_exp) / (head_dim ** 0.5)
        probs = jax.nn.softmax(scores, axis=-1)
        context = jnp.matmul(probs, v_exp).transpose(0, 2, 1, 3).reshape(b, s, q_dim)
        attn_out = jnp.matmul(context, o_w.T)
        post_attn_ms = jnp.mean(attn_out ** 2, axis=-1, keepdims=True)
        post_attn_res = (attn_out / jnp.sqrt(post_attn_ms + 1e-6)) * (1.0 + post_attn_w)
        res1 = x_t + post_attn_res * ls_w
        # 3. Pre-MLP RMSNorm & SwiGLU Gated MLP
        pre_mlp_ms = jnp.mean(res1 ** 2, axis=-1, keepdims=True)
        pre_mlp_norm = (res1 / jnp.sqrt(pre_mlp_ms + 1e-6)) * (1.0 + pre_mlp_w)
        gate = jax.nn.silu(jnp.matmul(pre_mlp_norm, g_w.T))
        up = jnp.matmul(pre_mlp_norm, u_w.T)
        mlp_out = jnp.matmul(gate * up, d_w.T)
        post_mlp_ms = jnp.mean(mlp_out ** 2, axis=-1, keepdims=True)
        post_mlp_res = (mlp_out / jnp.sqrt(post_mlp_ms + 1e-6)) * (1.0 + post_mlp_w)
        res2 = res1 + post_mlp_res
        # 4. Gemma 4 Per-Layer Projection
        pl_gate = jax.nn.silu(jnp.matmul(plin, plg_w.T))
        pl_out = jnp.matmul(pl_gate, plp_w.T)
        pl_ms = jnp.mean(pl_out ** 2, axis=-1, keepdims=True)
        pl_norm = (pl_out / jnp.sqrt(pl_ms + 1e-6)) * (1.0 + pln_w)
        return res2 + pl_norm

    args = (x, in_ln, ls, qw, kw, vw, ow, qn, kn, post_attn, pre_mlp, post_mlp, gw, uw, dw, plg, plp, pln, pl_in)
    return "gemma4-block", run, args, flops

def run_benchmarks_for_backend(backend_name):
    print(f"\n==========================================================================", flush=True)
    print(f"               Running JAX Benchmarks for Backend: [{backend_name}]        ", flush=True)
    print(f"==========================================================================", flush=True)
    jax, jnp, dev = setup_jax_backend(backend_name)
    print(f"Active JAX Device: {dev}", flush=True)

    workloads = [
        make_gemm_fp32(jnp, jax, dev),
        make_gemm_bf16(jnp, jax, dev),
        make_rms_norm(jnp, jax, dev),
        make_swiglu(jnp, jax, dev),
        make_gqa_causal_attn(jnp, jax, dev),
        make_gpt2_block(jnp, jax, dev),
        make_gemma4_block(jnp, jax, dev)
    ]
    results = {}
    print(f"{'Kernel':<18} | {'Mean (ms)':<10} | {'P99 (ms)':<10} | {'TFLOPS':<8} | {'Cold JIT (ms)':<12}", flush=True)
    print("-" * 70, flush=True)
    for name, fn, args, flops in workloads:
        stats = benchmark_fn(name, fn, args, flops=flops)
        stats["backend"] = str(backend_name)
        results[name] = stats
        print(f"{name:<18} | {stats['mean_ms']:<10.3f} | {stats['p99_ms']:<10.3f} | {stats['tflops']:<8.3f} | {stats['cold_jit_ms']:<12.3f}", flush=True)

    return results

def main():
    parser = argparse.ArgumentParser(description="JAX Benchmark Suite")
    parser.add_argument("--backend", "-b", choices=["auto", "cpu", "sycl", "rocm", "cuda12", "all"], default="auto",
                        help="Target backend platform")
    args = parser.parse_args()

    backends = []
    if args.backend == "all":
        backends = ["sycl", "cpu"]
    elif args.backend == "auto":
        backends = ["sycl", "cpu"]
    else:
        backends = [args.backend]

    all_results = {}
    for b in backends:
        try:
            res = run_benchmarks_for_backend(b)
            all_results[b] = res
        except Exception as e:
            print(f"Error benchmarking JAX backend {b}: {e}")

    import os
    os.makedirs("target/benchmark-reports", exist_ok=True)
    report_path = "target/benchmark-reports/jax_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJAX benchmark report saved to: {report_path}")

if __name__ == "__main__":
    main()
