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
