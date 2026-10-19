(ns einsum.uop.render-test
  "Generative property and unit tests for Phase 4: Declarative OptOps (`opt.clj`) and C99 / HIP C++ Source Renderer (`render.clj`)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gpt2 :as gpt2]
            [einsum.models.smollm :as smollm]
            [einsum.uop.core :as uop]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.opt :as opt]
            [einsum.uop.rangeify :as rangeify]
            [einsum.uop.render :as render]
            [einsum.uop.schedule :as schedule]))

(deftest test-opt-and-render-c99-and-hip-kernels
  (testing "OptOps scheduler and C99/HIP renderer emit syntactically valid C99 and HIP C++ kernel sources"
    (let [invars [[:A [:tensor [4 16] :f32]]
                  [:B [:tensor [16 8] :f32]]]
          ast [:= [:C :i :j] [:A :i :k] [:B :k :j]]
          uop-graph (uop-lower/ast->uop-graph "mm_render" invars ast [:C])
          sched (schedule/schedule-graph uop-graph)
          k0 (first (:kernels sched))
          opt-k0 (opt/optimize-kernel k0 {:target :rocm})
          c99-prog (render/render-kernel opt-k0 {:target :cpu})
          hip-prog (render/render-kernel opt-k0 {:target :rocm})]
      (is (string? (:source c99-prog)))
      (is (str/includes? (:source c99-prog) "void "))
      (is (str/includes? (:source c99-prog) "for (int "))
      (is (string? (:source hip-prog)))
      (is (str/includes? (:source hip-prog) "extern \"C\" __global__ void "))
      (is (str/includes? (:source hip-prog) "blockIdx.x"))
      (is (str/includes? (:source hip-prog) "threadIdx.x"))
      (is (vector? (:global-size hip-prog)))
      (is (vector? (:local-size hip-prog))))))

(defspec prop-opt-preserves-rangeified-kernel-semantics
  40
  (prop/for-all [m (gen/choose 1 8)
                 k (gen/choose 1 16)
                 n (gen/choose 1 8)]
                (let [a-data (float-array (repeatedly (* m k) #(- (rand 2.0) 1.0)))
                      b-data (float-array (repeatedly (* k n) #(- (rand 2.0) 1.0)))
                      p0 (uop/uop-param 0 :f32 [m k])
                      p1 (uop/uop-param 1 :f32 [k n])
                      e0 (uop/uop-expand (uop/uop-reshape p0 [m 1 k]) [m n k])
                      e1 (-> p1
                             (uop/uop-permute [1 0])
                             (uop/uop-reshape [1 n k])
                             (uop/uop-expand [m n k]))
                      prod (uop/uop-alu :uop/mul e0 e1)
                      red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [m n])
                      rk (rangeify/rangeify-kernel red {:out-slot 2})
                      opt-rk (:rangeified (opt/optimize-kernel {:name "k_test" :rangeified rk} {:target :rocm}))
                      raw-out ^floats (rangeify/eval-rangeified-kernel rk {0 a-data 1 b-data})
                      opt-out ^floats (rangeify/eval-rangeified-kernel opt-rk {0 a-data 1 b-data})]
                  (and (= (alength raw-out) (alength opt-out))
                       (every? true?
                               (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-5))
                                    (seq raw-out)
                                    (seq opt-out)))))))

(deftest test-render-all-gpt2-and-smollm-kernels
  (testing "All scheduled kernels for GPT-2 and SmolLM render cleanly to both C99 and HIP C++"
    (let [max-seq-len 8
          gpt2-ast (gpt2/gpt2-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          gpt2-invars [[:x [:tensor [1 max-seq-len] :i32]]
                       [:pos_ids [:tensor [1 max-seq-len] :i32]]
                       [:wte [:tensor [64 768] :f32]]
                       [:wpe [:tensor [16 768] :f32]]
                       [:ln_f_g [:tensor [768] :f32]]
                       [:ln_f_b [:tensor [768] :f32]]
                       [:ln1_g_0 [:tensor [768] :f32]]
                       [:ln1_b_0 [:tensor [768] :f32]]
                       [:attn_w_0 [:tensor [768 2304] :f32]]
                       [:attn_b_0 [:tensor [2304] :f32]]
                       [:proj_w_0 [:tensor [768 768] :f32]]
                       [:proj_b_0 [:tensor [768] :f32]]
                       [:ln2_g_0 [:tensor [768] :f32]]
                       [:ln2_b_0 [:tensor [768] :f32]]
                       [:mlp_fc_w_0 [:tensor [768 3072] :f32]]
                       [:mlp_fc_b_0 [:tensor [3072] :f32]]
                       [:mlp_proj_w_0 [:tensor [3072 768] :f32]]
                       [:mlp_proj_b_0 [:tensor [768] :f32]]]
          gpt2-sched (schedule/schedule-graph (uop-lower/ast->uop-graph "gpt2_1l" gpt2-invars gpt2-ast [:logits]))
          smollm-ast (smollm/smollm-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          smollm-invars [[:x [:tensor [1 max-seq-len] :i32]]
                         [:embed_tokens [:tensor [64 576] :f32]]
                         [:final_norm_w [:tensor [576] :f32]]
                         [:lm_head_w [:tensor [64 576] :f32]]
                         [:input_ln_w_0 [:tensor [576] :f32]]
                         [:q_w_0 [:tensor [576 576] :f32]]
                         [:k_w_0 [:tensor [192 576] :f32]]
                         [:v_w_0 [:tensor [192 576] :f32]]
                         [:o_w_0 [:tensor [576 576] :f32]]
                         [:post_attn_ln_w_0 [:tensor [576] :f32]]
                         [:gate_w_0 [:tensor [1536 576] :f32]]
                         [:up_w_0 [:tensor [1536 576] :f32]]
                         [:down_w_0 [:tensor [576 1536] :f32]]]
          smollm-sched (schedule/schedule-graph (uop-lower/ast->uop-graph "smollm_1l" smollm-invars smollm-ast [:logits]))]
      (doseq [k (concat (:kernels gpt2-sched) (:kernels smollm-sched))]
        (let [opt-k (opt/optimize-kernel k {:target :rocm})
              c99 (render/render-kernel opt-k {:target :cpu})
              hip (render/render-kernel opt-k {:target :rocm})]
          (is (not (str/blank? (:source c99))))
          (is (not (str/blank? (:source hip))))
          (is (seq (:buffer-slots c99)))
          (is (= (:buffer-slots c99) (:buffer-slots hip))))))))

(deftest test-render-bf16-storage-and-bitcast-helpers
  (testing "Kernel with bf16 storage emits unsigned short pointers and bf16 bitcast helper calls"
    (let [invars [[:A [:tensor [4 16] :f32]]
                  [:B [:tensor [16 8] :bf16]]]
          ast [:= [:C :i :j] [:A :i :k] [:B :k :j]]
          uop-graph (uop-lower/ast->uop-graph "mm_bf16" invars ast [:C])
          sched (schedule/schedule-graph uop-graph)
          k0 (first (:kernels sched))
          opt-k0 (opt/optimize-kernel k0 {:target :rocm})
          c99-prog (render/render-kernel opt-k0 {:target :cpu})
          hip-prog (render/render-kernel opt-k0 {:target :rocm})]
      ;; Param 2 (:B) has dtype :bf16 -> declared as const unsigned short* __restrict__ buf_2
      (is (str/includes? (:source c99-prog) "const unsigned short* __restrict__ buf_2"))
      (is (str/includes? (:source hip-prog) "const unsigned short* __restrict__ buf_2"))
      ;; Headers declare bitcast functions
      (is (str/includes? (:source c99-prog) "bf16_to_f32"))
      (is (str/includes? (:source hip-prog) "bf16_to_f32"))
      ;; Kernel body calls bf16_to_f32
      (is (str/includes? (:source c99-prog) "bf16_to_f32("))
      (is (str/includes? (:source hip-prog) "bf16_to_f32(")))))

(defspec prop-bf16-bitcast-roundtrip
  50
  (prop/for-all [f (gen/fmap (fn [[s exp mant]]
                               (let [sign (if (pos? (int s)) 1.0 -1.0)
                                     v (* sign (Math/pow 2.0 (double exp)) (+ 1.0 (/ (double mant) 128.0)))]
                                 (float v)))
                             (gen/tuple (gen/choose -1 1) (gen/choose -10 10) (gen/choose 0 127)))]
                (let [bits (Float/floatToIntBits f)
                      bf16-short (short (bit-shift-right bits 16))
                      f-back (Float/intBitsToFloat (unchecked-int (bit-shift-left (bit-and (int bf16-short) 0xffff) 16)))]
                  (or (zero? (Float/compare f f-back))
                      (< (Math/abs (- (double f) (double f-back))) (* (Math/abs (double f)) 0.01))))))

(deftest test-render-2d-prefill-gemm
  (testing "2D prefill GEMM kernel renders 2D grid and token/channel coordinate bindings for short-batch"
    (let [invars [[:A [:tensor [1 4 128] :f32]]
                  [:B [:tensor [256 128] :f32]]]
          ast [:= [:C :b :p :d] [:A :b :p :k] [:B :d :k]]
          uop-graph (uop-lower/ast->uop-graph "prefill_gemm" invars ast [:C])
          sched (schedule/schedule-graph uop-graph)
          k0 (first (:kernels sched))
          opt-k0 (opt/optimize-kernel k0 {:target :rocm})
          hip-prog (render/render-kernel opt-k0 {:target :rocm})]
      (is (string? (:source hip-prog)))
      (is (true? (:has-group? opt-k0)))
      (is (vector? (:global-size hip-prog)))
      (is (vector? (:local-size hip-prog)))
      ;; When 2D short-batch prefill GEMM is active, local-size has group-size along K and local-rows along N
      (is (= 64 (first (:local-size hip-prog))))
      (is (= 4 (second (:local-size hip-prog))))
      ;; Source must contain shared memory LDS declaration and synchronization
      (is (str/includes? (:source hip-prog) "__shared__"))
      (is (str/includes? (:source hip-prog) "__syncthreads()")))))

(deftest test-render-2d-tiled-gemm
  (testing "2D tiled GEMM kernel (M >= 16, N >= 16) renders 16x16 local geometry, direct token/channel coordinates, and WMMA intrinsic header"
    (let [invars [[:A [:tensor [1 16 128] :f32]]
                  [:B [:tensor [256 128] :f32]]]
          ast [:= [:C :b :p :d] [:A :b :p :k] [:B :d :k]]
          uop-graph (uop-lower/ast->uop-graph "tiled_gemm" invars ast [:C])
          sched (schedule/schedule-graph uop-graph)
          k0 (first (:kernels sched))
          opt-k0 (opt/optimize-kernel k0 {:target :rocm})
          hip-prog (render/render-kernel opt-k0 {:target :rocm})]
      (is (string? (:source hip-prog)))
      (is (true? (:gemm-2d? opt-k0)))
      (is (false? (:has-group? opt-k0)))
      (is (= [16 16 1] (:local-size hip-prog)))
      (is (= [16 1 1] (:global-size hip-prog)))
      ;; Direct coordinate bindings (zero div/mod in kernel preamble)
      (is (str/includes? (:source hip-prog) "int r_token = (int)(blockIdx.y * 16 + threadIdx.y);"))
      (is (str/includes? (:source hip-prog) "int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);"))
      (is (str/includes? (:source hip-prog) "if (r_token >= 16 || r_channel >= 256) return;"))
      ;; WMMA intrinsic in ROCm header
      (is (str/includes? (:source hip-prog) "rdna3_wmma_bf16_16x16x16"))
      ;; Zero LDS overhead
      (is (not (str/includes? (:source hip-prog) "__syncthreads()"))))))

(defspec prop-2d-tiled-gemm-invariants
  30
  (prop/for-all [m (gen/choose 16 32)
                 k (gen/choose 16 32)
                 n (gen/choose 16 32)]
                (let [p0 (uop/uop-param 0 :f32 [m k])
                      p1 (uop/uop-param 1 :f32 [k n])
                      e0 (uop/uop-expand (uop/uop-reshape p0 [m 1 k]) [m n k])
                      e1 (-> p1
                             (uop/uop-permute [1 0])
                             (uop/uop-reshape [1 n k])
                             (uop/uop-expand [m n k]))
                      prod (uop/uop-alu :uop/mul e0 e1)
                      red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [m n])
                      rk (rangeify/rangeify-kernel red {:out-slot 2})
                      opt-k (opt/optimize-kernel {:name "k_gemm2d" :rangeified rk} {:target :rocm})
                      hip-prog (render/render-kernel opt-k {:target :rocm})]
                  (and (true? (:gemm-2d? opt-k))
                       (= [16 16 1] (:local-size hip-prog))
                       (or (str/includes? (:source hip-prog) "int r_token = (int)(blockIdx.y * 16 + threadIdx.y);")
                           (str/includes? (:source hip-prog) "r_token_base"))
                       (or (str/includes? (:source hip-prog) "int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);")
                           (str/includes? (:source hip-prog) "raw_channel"))))))

(deftest test-render-multi-row-upcast-prefill-gemm
  (testing "Multi-row register-upcast prefill GEMM emits tile-m accumulators, hoisted weights, and workgroup store"
    (let [invars [[:A [:tensor [1 32 256] :bf16]]
                  [:B [:tensor [256 256] :bf16]]]
          ast [:= [:C :b :p :d] [:A :b :p :k] [:B :d :k]]
          uop-graph (uop-lower/ast->uop-graph "prefill_upcast_gemm" invars ast [:C])
          sched (schedule/schedule-graph uop-graph)
          k0 (first (:kernels sched))
          opt-k0 (opt/optimize-kernel k0 {:target :rocm})
          hip-prog (render/render-kernel opt-k0 {:target :rocm})
          src (:source hip-prog)]
      (is (true? (:gemm-2d? opt-k0)))
      (is (= 2 (:tile-m opt-k0)))
      (is (= [16 16 1] (:local-size hip-prog)))
      (is (= [16 1 1] (:global-size hip-prog)))
      (is (str/includes? src "__shared__ float s_W[16][17];"))
      (is (str/includes? src "__shared__ float s_X[2][16][17];"))
      (is (str/includes? src "s_W[threadIdx.y][threadIdx.x]"))
      (is (str/includes? src "s_W[threadIdx.x][k_sub]"))
      (is (str/includes? src "r_token_base"))
      (is (str/includes? src "r_tok_0"))
      (is (str/includes? src "r_tok_1"))
      (is (str/includes? src "acc_1_0"))
      (is (str/includes? src "acc_1_1"))
      (let [loop-idx (.indexOf src "for (int rk_")
            idx-buf2 (.indexOf src "buf_2" loop-idx)
            idx-buf1 (.indexOf src "buf_1" loop-idx)]
        (is (pos? idx-buf2))
        (is (pos? idx-buf1))
        (is (< idx-buf2 idx-buf1))))))

(deftest test-multi-row-rmsnorm-routing
  (testing "3D and 4D multi-row RMSNorm sum-sq reductions route to :gemma4-gemv (group 64, local-rows 4)"
    (let [ke-3d {:name "rmsnorm_3d" :rangeified {:out-shape [1 16 1] :out-size 16 :red-shape [256]}}
          opt-3d (opt/optimize-kernel ke-3d {:target :rocm})

          ke-4d {:name "rmsnorm_4d" :rangeified {:out-shape [1 16 8 1] :out-size 128 :red-shape [256]}}
          opt-4d (opt/optimize-kernel ke-4d {:target :rocm})]
      (is (true? (:has-group? opt-3d)))
      (is (= 4 (:local-rows opt-3d)))
      (is (= [64 4 1] (:local-size opt-3d)))
      (is (not (:gemm-2d? opt-3d)))

      (is (true? (:has-group? opt-4d)))
      (is (= 4 (:local-rows opt-4d)))
      (is (= [64 4 1] (:local-size opt-4d)))
      (is (not (:gemm-2d? opt-4d))))))

(defspec prop-multi-row-upcast-prefill-gemm-hip-structure
  25
  (prop/for-all [s (gen/choose 24 80)
                 n (gen/elements [16 32 64])]
                (let [k 256
                      p0 (uop/uop-param 0 :f32 [s k])
                      p1 (uop/uop-param 1 :f32 [k n])
                      e0 (uop/uop-expand (uop/uop-reshape p0 [s 1 k]) [s n k])
                      e1 (-> p1
                             (uop/uop-permute [1 0])
                             (uop/uop-reshape [1 n k])
                             (uop/uop-expand [s n k]))
                      prod (uop/uop-alu :uop/mul e0 e1)
                      red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [s n])
                      rk (rangeify/rangeify-kernel red {:out-slot 2})
                      opt-k (opt/optimize-kernel {:name "k_prefill_upcast" :rangeified rk} {:target :rocm})
                      hip-prog (render/render-kernel opt-k {:target :rocm})
                      tile-m (long (or (:tile-m opt-k) 1))
                      block-m (* 16 tile-m)
                      expected-grid-n (max 1 (long (quot (+ n 16 -1) 16)))
                      expected-grid-m (max 1 (long (quot (+ s block-m -1) block-m)))]
                  (and (> tile-m 1)
                       (true? (:gemm-2d? opt-k))
                       (= [16 16 1] (:local-size hip-prog))
                       (= [expected-grid-n expected-grid-m 1] (:global-size hip-prog))
                       (str/includes? (:source hip-prog) "r_token_base")))))

(defn- reference-fused-rms-matmul
  "Computes reference fused RMSNorm scale + matmul:
   out[i, j] = (1 / sqrt(s[i] + 1e-5)) * sum_k (A[i, k] * B[k, j])"
  [^floats a ^floats b ^floats s m k n]
  (let [c (float-array (* m n))]
    (dotimes [i m]
      (let [rstd (float (/ 1.0 (Math/sqrt (+ (double (aget s i)) 1e-5))))]
        (dotimes [j n]
          (let [acc (atom 0.0)]
            (dotimes [kk k]
              (let [va (aget a (+ (* i k) kk))
                    vb (aget b (+ (* kk n) j))]
                (swap! acc + (* va rstd vb))))
            (aset c (+ (* i n) j) (float @acc))))))
    c))

(deftest test-fused-rmsnorm-matmul-hoisting
  (testing "Fused RMSNorm + matmul hoists sqrtf and (1.0f / before the reduction loop in HIP C++ source"
    (let [m 32 k 64 n 32
          p0 (uop/uop-param 0 :f32 [m k])
          p1 (uop/uop-param 1 :f32 [k n])
          p2 (uop/uop-param 2 :f32 [m 1])
          e0 (uop/uop-expand (uop/uop-reshape p0 [m 1 k]) [m n k])
          e1 (-> p1
                 (uop/uop-permute [1 0])
                 (uop/uop-reshape [1 n k])
                 (uop/uop-expand [m n k]))
          eps-const (uop/uop-const :f32 1e-5 [m 1])
          var-eps (uop/uop-alu :uop/add p2 eps-const)
          rstd (uop/uop-alu :uop/recip (uop/uop-alu :uop/sqrt var-eps))
          rstd-bcast (uop/uop-expand (uop/uop-reshape rstd [m 1 1]) [m n k])
          normed (uop/uop-alu :uop/mul e0 rstd-bcast)
          prod (uop/uop-alu :uop/mul normed e1)
          red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [m n])
          rk (rangeify/rangeify-kernel red {:out-slot 3})
          opt-k (opt/optimize-kernel {:name "k_fused_rmsnorm_matmul" :rangeified rk} {:target :rocm})
          hip-prog (render/render-kernel opt-k {:target :rocm})
          src (:source hip-prog)
          loop-idx (.indexOf src "for (int rk_")
          idx-sqrt (.indexOf src "sqrtf")
          idx-recip (.indexOf src "(1.0f /")]
      (is (pos? loop-idx))
      (is (pos? idx-sqrt))
      (is (pos? idx-recip))
      (is (< idx-sqrt loop-idx))
      (is (< idx-recip loop-idx)))))

(defspec prop-fused-rmsnorm-matmul-evaluation-parity
  25
  (prop/for-all [m (gen/choose 1 16)
                 k (gen/choose 2 32)
                 n (gen/choose 1 16)]
                (let [a-data (float-array (repeatedly (* m k) #(- (rand 2.0) 1.0)))
                      b-data (float-array (repeatedly (* k n) #(- (rand 2.0) 1.0)))
                      s-data (float-array (repeatedly m #(+ 0.1 (rand 2.0))))
                      p0 (uop/uop-param 0 :f32 [m k])
                      p1 (uop/uop-param 1 :f32 [k n])
                      p2 (uop/uop-param 2 :f32 [m 1])
                      e0 (uop/uop-expand (uop/uop-reshape p0 [m 1 k]) [m n k])
                      e1 (-> p1
                             (uop/uop-permute [1 0])
                             (uop/uop-reshape [1 n k])
                             (uop/uop-expand [m n k]))
                      eps-const (uop/uop-const :f32 1e-5 [m 1])
                      var-eps (uop/uop-alu :uop/add p2 eps-const)
                      rstd (uop/uop-alu :uop/recip (uop/uop-alu :uop/sqrt var-eps))
                      rstd-bcast (uop/uop-expand (uop/uop-reshape rstd [m 1 1]) [m n k])
                      normed (uop/uop-alu :uop/mul e0 rstd-bcast)
                      prod (uop/uop-alu :uop/mul normed e1)
                      red (uop/uop-reshape (uop/uop-reduce prod :add [2] false) [m n])
                      rk (rangeify/rangeify-kernel red {:out-slot 3})
                      opt-k (opt/optimize-kernel {:name "k_fused_rmsnorm_matmul" :rangeified rk} {:target :rocm})
                      opt-rk (:rangeified opt-k)
                      hip-prog (render/render-kernel opt-k {:target :rocm})
                      src (:source hip-prog)
                      loop-idx (.indexOf src "for (int rk_")
                      idx-sqrt (.indexOf src "sqrtf")
                      idx-recip (.indexOf src "(1.0f /")
                      ref-out (reference-fused-rms-matmul a-data b-data s-data m k n)
                      raw-out ^floats (rangeify/eval-rangeified-kernel rk {0 a-data 1 b-data 2 s-data})
                      opt-out ^floats (rangeify/eval-rangeified-kernel opt-rk {0 a-data 1 b-data 2 s-data})]
                  (and (pos? loop-idx)
                       (pos? idx-sqrt)
                       (pos? idx-recip)
                       (< idx-sqrt loop-idx)
                       (< idx-recip loop-idx)
                       (= (alength raw-out) (alength opt-out))
                       (every? true?
                               (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-4))
                                    (seq raw-out)
                                    (seq opt-out)))
                       (every? true?
                               (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-4))
                                    (seq opt-out)
                                    (seq ref-out)))))))

(deftest test-4d-batched-attention-gemm-tiling
  (testing "4D batched attention GEMM routes to 2D GEMM tiling, decomposes r_head and r_token without inner div/mod, and hoists invariant K load"
    (let [b 1 h 8 s 32 n 48 k 64
          q (uop/uop-param 0 :bf16 [b h s k])
          k-param (uop/uop-param 1 :bf16 [b n 1 k])
          e-q (-> q
                  (uop/uop-cast :f32)
                  (uop/uop-reshape [b h s 1 k])
                  (uop/uop-expand [b h s n k]))
          e-k (-> k-param
                  (uop/uop-cast :f32)
                  (uop/uop-reshape [1 1 1 n k])
                  (uop/uop-expand [b h s n k]))
          prod (uop/uop-alu :uop/mul e-q e-k)
          red (uop/uop-reduce prod :add [4] false)
          rk (rangeify/rangeify-kernel red {:out-slot 2})
          opt-k (opt/optimize-kernel {:name "k_batched_attn" :rangeified rk} {:target :rocm})
          hip (render/render-kernel opt-k {:target :rocm})
          src (:source hip)
          loop-idx (.indexOf src "for (int rk_")
          inner-src (if (pos? loop-idx) (subs src loop-idx) "")]
      (is (true? (:gemm-2d? opt-k)))
      (is (= [16 16 1] (:local-size hip)))
      (is (= 2 (:tile-m opt-k)))
      (is (= [3 8 1] (:global-size hip)))
      (is (str/includes? src "__shared__ float s_W[16][17];"))
      (is (str/includes? src "__shared__ float s_X[2][16][17];"))
      (is (str/includes? src "s_W[threadIdx.y][threadIdx.x]"))
      (is (str/includes? src "s_W[threadIdx.x][k_sub]"))
      (is (str/includes? src "int r_head = (int)(blockIdx.y / 1);"))
      (is (str/includes? src "int block_token_base = (int)((blockIdx.y % 1) * 32);"))
      (is (str/includes? src "int raw_token_base = block_token_base + (int)threadIdx.y;"))
      (is (let [k-idx (.indexOf src "buf_2[" loop-idx)
                q-idx (.indexOf src "buf_1[" loop-idx)]
            (and (pos? k-idx) (pos? q-idx) (< k-idx q-idx))))
      (is (not (str/includes? inner-src "/ 32")))
      (is (not (str/includes? inner-src "% 32"))))))

(defspec prop-4d-batched-attention-gemm-tiling
  20
  (prop/for-all [h (gen/choose 2 4)
                 s (gen/choose 16 36)
                 n (gen/choose 16 32)
                 k (gen/choose 16 32)]
                (let [b 1
                      q-data (float-array (repeatedly (* b h s k) #(- (rand 2.0) 1.0)))
                      k-data (float-array (repeatedly (* b n 1 k) #(- (rand 2.0) 1.0)))
                      q (uop/uop-param 0 :f32 [b h s k])
                      k-param (uop/uop-param 1 :f32 [b n 1 k])
                      e-q (-> q
                              (uop/uop-reshape [b h s 1 k])
                              (uop/uop-expand [b h s n k]))
                      e-k (-> k-param
                              (uop/uop-reshape [1 1 1 n k])
                              (uop/uop-expand [b h s n k]))
                      prod (uop/uop-alu :uop/mul e-q e-k)
                      red (uop/uop-reduce prod :add [4] false)
                      rk (rangeify/rangeify-kernel red {:out-slot 2})
                      opt-k (opt/optimize-kernel {:name "k_batched_attn_gen" :rangeified rk} {:target :rocm})
                      opt-rk (:rangeified opt-k)
                      hip (render/render-kernel opt-k {:target :rocm})
                      raw-out ^floats (rangeify/eval-rangeified-kernel rk {0 q-data 1 k-data})
                      opt-out ^floats (rangeify/eval-rangeified-kernel opt-rk {0 q-data 1 k-data})
                      tile-m (:tile-m opt-k)
                      expected-tile-m (min 8 (long (quot (+ s 15) 16)))
                      expected-grid-n (max 1 (long (quot (+ n 16 -1) 16)))
                      block-m (* 16 tile-m)
                      expected-grid-q (max 1 (long (quot (+ s block-m -1) block-m)))
                      expected-grid-m (* h expected-grid-q)]
                  (and (true? (:gemm-2d? opt-k))
                       (= [16 16 1] (:local-size hip))
                       (= expected-tile-m tile-m)
                       (= [expected-grid-n expected-grid-m 1] (:global-size hip))
                       (= (alength raw-out) (alength opt-out))
                       (every? true?
                               (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-4))
                                    (seq raw-out)
                                    (seq opt-out)))))))

(deftest test-4d-gemm-with-singleton-axis-before-channel
  (testing "Output shape [1, 32, 1, 64] with a singleton axis before n-dim gets batch-heads = 1, tile-m = 2, global-size = [4 1 1]"
    (let [b 1 s 32 n 64 k 64
          x (uop/uop-param 0 :f32 [b s 1 k])
          w (uop/uop-param 1 :f32 [k n])
          e-x (-> x
                  (uop/uop-reshape [b s 1 1 k])
                  (uop/uop-expand [b s 1 n k]))
          e-w (-> w
                  (uop/uop-reshape [1 1 1 n k])
                  (uop/uop-expand [b s 1 n k]))
          prod (uop/uop-alu :uop/mul e-x e-w)
          red (uop/uop-reduce prod :add [4] false)
          rk (rangeify/rangeify-kernel red {:out-slot 2})
          opt-k (opt/optimize-kernel {:name "k_singleton_gemm" :rangeified rk} {:target :rocm})
          hip (render/render-kernel opt-k {:target :rocm})]
      (is (true? (:gemm-2d? opt-k)))
      (is (= 2 (:tile-m opt-k)))
      (is (= [4 1 1] (:global-size hip))))))

(defspec prop-lds-gemm-unaligned-parity
  20
  (prop/for-all [s (gen/choose 17 35)
                 n (gen/choose 16 35)
                 k (gen/choose 16 35)]
                (let [a-data (float-array (repeatedly (* s k) #(- (rand 2.0) 1.0)))
                      b-data (float-array (repeatedly (* k n) #(- (rand 2.0) 1.0)))
                      p0 (uop/uop-param 0 :f32 [1 s k])
                      p1 (uop/uop-param 1 :f32 [k n])
                      e0 (uop/uop-expand (uop/uop-reshape p0 [1 s 1 k]) [1 s n k])
                      e1 (-> p1
                             (uop/uop-permute [1 0])
                             (uop/uop-reshape [1 1 n k])
                             (uop/uop-expand [1 s n k]))
                      prod (uop/uop-alu :uop/mul e0 e1)
                      red (uop/uop-reshape (uop/uop-reduce prod :add [3] false) [1 s n])
                      rk (rangeify/rangeify-kernel red {:out-slot 2})
                      opt-k (opt/optimize-kernel {:name "k_lds_unaligned" :rangeified rk} {:target :rocm})
                      opt-rk (:rangeified opt-k)
                      hip-prog (render/render-kernel opt-k {:target :rocm})
                      src (:source hip-prog)
                      raw-out ^floats (rangeify/eval-rangeified-kernel rk {0 a-data 1 b-data})
                      opt-out ^floats (rangeify/eval-rangeified-kernel opt-rk {0 a-data 1 b-data})]
                  (and (true? (:gemm-2d? opt-k))
                       (= [16 16 1] (:local-size hip-prog))
                       (str/includes? src "__shared__ float s_W[16][17];")
                       (str/includes? src "s_W[threadIdx.x][k_sub]")
                       (= (alength raw-out) (alength opt-out))
                       (every? true?
                               (map (fn [x y] (< (Math/abs (- (double x) (double y))) 1e-4))
                                    (seq raw-out)
                                    (seq opt-out)))))))
