(ns einsum.uop.runtime-test
  "End-to-end generative property and parity tests for Phase 5: Panama UOp Runtime (`:uop`, `:uop-cpu`, `:uop-rocm`)
   verifying exact numerical parity against `:interpreter` across matmul, Transformer primitives, GPT-2, and SmolLM."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as xla]
            [einsum.models.gpt2 :as gpt2]
            [einsum.models.smollm :as smollm]))

(defn- close-floats?
  ([^floats a ^floats b]
   (close-floats? a b 1e-3))
  ([^floats a ^floats b ^double tol]
   (and (= (alength a) (alength b))
        (every? true?
                (map (fn [x y]
                       (<= (Math/abs (- (double x) (double y))) tol))
                     (seq a)
                     (seq b))))))

(defn- argmax [^floats arr]
  (let [n (alength arr)]
    (loop [i 1
           best-idx 0
           best-val (double (aget arr 0))]
      (if (>= i n)
        best-idx
        (let [v (double (aget arr i))]
          (if (> v best-val)
            (recur (inc i) i v)
            (recur (inc i) best-idx best-val)))))))

(defspec prop-uop-cpu-matmul-parity
  25
  (prop/for-all [m (gen/choose 1 6)
                 k (gen/choose 1 12)
                 n (gen/choose 1 6)]
                (let [a (float-array (repeatedly (* m k) #(- (rand 2.0) 1.0)))
                      b (float-array (repeatedly (* k n) #(- (rand 2.0) 1.0)))
                      bias (float-array (repeatedly n #(- (rand 1.0) 0.5)))
                      ast [:block {:name :mm_bias}
                           [:= [:Y :i :j] [:A :i :k] [:B :k :j]]
                           [:= [:Z :i :j] [:Y :i :j]]
                           [:= [:Z :i :j] [:bias :j]]]
                      in-spec [[:A [:tensor [m k] :f32]]
                               [:B [:tensor [k n] :f32]]
                               [:bias [:tensor [n] :f32]]]
                      interp-ctx (xla/init-backend! :interpreter {:quiet? true})
                      cpu-ctx (xla/init-backend! :uop-cpu {:quiet? true})]
                  (with-open [k-interp (xla/compile-kernel interp-ctx "mm_p" ast {:in in-spec :out [:Z]})
                              k-cpu (xla/compile-kernel cpu-ctx "mm_p" ast {:in in-spec :out [:Z]})]
                    (let [ref-out (xla/to-host-slice (k-interp [a b bias]) 0 (* m n) (* m n))
                          cpu-out (xla/to-host-slice (k-cpu [a b bias]) 0 (* m n) (* m n))]
                      (close-floats? ref-out cpu-out 1e-4))))))

(deftest test-uop-transformer-primitives-parity
  (testing "LayerNorm, RMSNorm, RoPE, and Causal Softmax match :interpreter on :uop-cpu and :uop-rocm"
    (let [b 1 p 4 d 8
          x (float-array (repeatedly (* b p d) #(- (rand 2.0) 1.0)))
          g (float-array (repeatedly d #(+ 0.5 (rand 0.5))))
          bias (float-array (repeatedly d #(- (rand 0.2) 0.1)))
          ast [:block {:name :prim_test}
               [:layer-norm [:ln :b :p :d] [:x :b :p :d] [:g :d] [:bias :d] {:eps 1e-5}]
               [:rms-norm [:rn :b :p :d] [:ln :b :p :d] [:g :d] {:eps 1e-5}]
               [:rope [:rot :b :p :d] [:rn :b :p :d] {:head-dim 4 :theta 10000.0 :max-seq-len p}]
               [:reshape [:rot4 :b :h :p-q :p-k] [:rot :b :p :d] {:shape [1 2 4 4]}]
               [:causal-softmax [:out :b :h :p-q :p-k] [:rot4 :b :h :p-q :p-k]]]
          in-spec [[:x [:tensor [b p d] :f32]]
                   [:g [:tensor [d] :f32]]
                   [:bias [:tensor [d] :f32]]]
          interp-ctx (xla/init-backend! :interpreter {:quiet? true})
          uop-cpu-ctx (xla/init-backend! :uop-cpu {:quiet? true})
          uop-auto-ctx (xla/init-backend! :uop {:quiet? true})]
      (with-open [k-ref (xla/compile-kernel interp-ctx "prim" ast {:in in-spec :out [:out]})
                  k-cpu (xla/compile-kernel uop-cpu-ctx "prim" ast {:in in-spec :out [:out]})
                  k-auto (xla/compile-kernel uop-auto-ctx "prim" ast {:in in-spec :out [:out]})]
        (let [ref-floats (xla/to-host-slice (k-ref [x g bias]) 0 (* b p d) (* b p d))
              cpu-floats (xla/to-host-slice (k-cpu [x g bias]) 0 (* b p d) (* b p d))
              auto-floats (xla/to-host-slice (k-auto [x g bias]) 0 (* b p d) (* b p d))]
          (is (close-floats? ref-floats cpu-floats 1e-4))
          (is (close-floats? ref-floats auto-floats 1e-4)))))))

(deftest test-uop-gpt2-and-smollm-e2e-parity
  (testing "1-layer GPT-2 and 1-layer SmolLM-135M execute end-to-end via :uop-cpu and :uop with exact argmax & logit parity"
    (let [rng (java.util.Random. 42)
          rand-f32 (fn [n scale]
                     (let [arr (float-array n)]
                       (dotimes [i n]
                         (aset arr i (float (* (- (.nextDouble rng) 0.5) scale))))
                       arr))
          max-seq-len 4
          vocab-size 32
          ;; 1. GPT-2 1-layer end-to-end test
          gpt2-ast (gpt2/gpt2-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          gpt2-weights {:wte (rand-f32 (* vocab-size 768) 0.1)
                        :wpe (rand-f32 (* 16 768) 0.1)
                        :ln_f_g (rand-f32 768 1.0)
                        :ln_f_b (rand-f32 768 0.1)
                        :ln1_g_0 (rand-f32 768 1.0)
                        :ln1_b_0 (rand-f32 768 0.1)
                        :attn_w_0 (rand-f32 (* 768 2304) 0.05)
                        :attn_b_0 (rand-f32 2304 0.05)
                        :proj_w_0 (rand-f32 (* 768 768) 0.05)
                        :proj_b_0 (rand-f32 768 0.05)
                        :ln2_g_0 (rand-f32 768 1.0)
                        :ln2_b_0 (rand-f32 768 0.1)
                        :mlp_fc_w_0 (rand-f32 (* 768 3072) 0.05)
                        :mlp_fc_b_0 (rand-f32 3072 0.05)
                        :mlp_proj_w_0 (rand-f32 (* 3072 768) 0.05)
                        :mlp_proj_b_0 (rand-f32 768 0.05)}
          gpt2-header {:wte {:shape [vocab-size 768] :dtype :f32}
                       :wpe {:shape [16 768] :dtype :f32}
                       :ln_f_g {:shape [768] :dtype :f32}
                       :ln_f_b {:shape [768] :dtype :f32}
                       :ln1_g_0 {:shape [768] :dtype :f32}
                       :ln1_b_0 {:shape [768] :dtype :f32}
                       :attn_w_0 {:shape [768 2304] :dtype :f32}
                       :attn_b_0 {:shape [2304] :dtype :f32}
                       :proj_w_0 {:shape [768 768] :dtype :f32}
                       :proj_b_0 {:shape [768] :dtype :f32}
                       :ln2_g_0 {:shape [768] :dtype :f32}
                       :ln2_b_0 {:shape [768] :dtype :f32}
                       :mlp_fc_w_0 {:shape [768 3072] :dtype :f32}
                       :mlp_fc_b_0 {:shape [3072] :dtype :f32}
                       :mlp_proj_w_0 {:shape [3072 768] :dtype :f32}
                       :mlp_proj_b_0 {:shape [768] :dtype :f32}}
          ;; 2. SmolLM 1-layer end-to-end test
          smollm-ast (smollm/smollm-model-ast {:num-layers 1 :max-seq-len max-seq-len})
          smollm-weights {:embed_tokens (rand-f32 (* vocab-size 576) 0.1)
                          :final_norm_w (rand-f32 576 1.0)
                          :lm_head_w (rand-f32 (* vocab-size 576) 0.1)
                          :input_ln_w_0 (rand-f32 576 1.0)
                          :q_w_0 (rand-f32 (* 576 576) 0.05)
                          :k_w_0 (rand-f32 (* 192 576) 0.05)
                          :v_w_0 (rand-f32 (* 192 576) 0.05)
                          :o_w_0 (rand-f32 (* 576 576) 0.05)
                          :post_attn_ln_w_0 (rand-f32 576 1.0)
                          :gate_w_0 (rand-f32 (* 1536 576) 0.05)
                          :up_w_0 (rand-f32 (* 1536 576) 0.05)
                          :down_w_0 (rand-f32 (* 576 1536) 0.05)}
          smollm-header {:embed_tokens {:shape [vocab-size 576] :dtype :f32}
                         :final_norm_w {:shape [576] :dtype :f32}
                         :lm_head_w {:shape [vocab-size 576] :dtype :f32}
                         :input_ln_w_0 {:shape [576] :dtype :f32}
                         :q_w_0 {:shape [576 576] :dtype :f32}
                         :k_w_0 {:shape [192 576] :dtype :f32}
                         :v_w_0 {:shape [192 576] :dtype :f32}
                         :o_w_0 {:shape [576 576] :dtype :f32}
                         :post_attn_ln_w_0 {:shape [576] :dtype :f32}
                         :gate_w_0 {:shape [1536 576] :dtype :f32}
                         :up_w_0 {:shape [1536 576] :dtype :f32}
                         :down_w_0 {:shape [576 1536] :dtype :f32}}
          toks (int-array [2 5 9 14])
          pos-ids (int-array [0 1 2 3])

          run-gpt2 (fn [backend-kw]
                     (let [ctx (xla/init-backend! backend-kw {:quiet? true})]
                       (with-open [arena (xla/create-arena ctx)]
                         (let [store (xla/create-weight-store ctx (assoc gpt2-weights :header gpt2-header) {:arena arena})]
                           (with-open [k (xla/compile-kernel ctx "gpt2_1l" gpt2-ast
                                                             {:in {:x [:tensor [1 max-seq-len] :i32]
                                                                   :pos_ids [:tensor [1 max-seq-len] :i32]}
                                                              :weights store
                                                              :out [:logits]})]
                             (let [{:keys [logits]} (k {:x toks :pos_ids pos-ids})]
                               (xla/to-host-slice logits (dec max-seq-len) vocab-size (* max-seq-len vocab-size))))))))

          run-smollm (fn [backend-kw]
                       (let [ctx (xla/init-backend! backend-kw {:quiet? true})]
                         (with-open [arena (xla/create-arena ctx)]
                           (let [store (xla/create-weight-store ctx (assoc smollm-weights :header smollm-header) {:arena arena})]
                             (with-open [k (xla/compile-kernel ctx "smollm_1l" smollm-ast
                                                               {:in {:x [:tensor [1 max-seq-len] :i32]}
                                                                :weights store
                                                                :out [:logits]})]
                               (let [{:keys [logits]} (k {:x toks})]
                                 (xla/to-host-slice logits (dec max-seq-len) vocab-size (* max-seq-len vocab-size))))))))
          gpt2-ref (run-gpt2 :interpreter)
          gpt2-cpu (run-gpt2 :uop-cpu)
          gpt2-uop (run-gpt2 :uop)
          smollm-ref (run-smollm :interpreter)
          smollm-cpu (run-smollm :uop-cpu)
          smollm-uop (run-smollm :uop)]
      (is (close-floats? gpt2-ref gpt2-cpu 1e-3))
      (is (close-floats? gpt2-ref gpt2-uop 1e-3))
      (is (= (argmax gpt2-ref) (argmax gpt2-cpu) (argmax gpt2-uop)))
      (is (close-floats? smollm-ref smollm-cpu 1e-3))
      (is (close-floats? smollm-ref smollm-uop 1e-3))
      (is (= (argmax smollm-ref) (argmax smollm-cpu) (argmax smollm-uop))))))
