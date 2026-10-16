(ns einsum.uop.core-test
  "Generative property and unit tests for UOp IR, hash-consing, symbolic rewriter, and Tensor Logic AST -> UOp lowering."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.models.gpt2 :as gpt2]
            [einsum.models.smollm :as smollm]
            [einsum.uop.core :as uop]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.rewrite :as rewrite]))

;; -----------------------------------------------------------------------------
;; Phase 1 Tests: UOp Hash-Consing, Invariants, and Symbolic Rewriting
;; -----------------------------------------------------------------------------

(deftest test-uop-hash-consing-and-validation
  (testing "Identical UOps are interned to the exact same canonical node (hash-consed)"
    (uop/clear-uop-cache!)
    (let [c1 (uop/uop-const :f32 2.5)
          c2 (uop/uop-const :f32 2.5)
          c3 (uop/uop-const :f32 3.5)
          p0 (uop/uop-param 0 :f32 [4 8])
          p0-dup (uop/uop-param 0 :f32 [4 8])
          m1 (uop/uop-alu :uop/mul p0 (uop/uop-expand (uop/uop-reshape c1 [1 1]) [4 8]))
          m2 (uop/uop-alu :uop/mul p0-dup (uop/uop-expand (uop/uop-reshape c2 [1 1]) [4 8]))]
      (is (identical? c1 c2))
      (is (= (:id c1) (:id c2)))
      (is (not= (:id c1) (:id c3)))
      (is (identical? p0 p0-dup))
      (is (identical? m1 m2))
      (is (uop/valid-uop? m1)))))

(deftest test-uop-movement-shape-inference
  (testing "Movement ops infer exact shapes per tinyspec"
    (let [p (uop/uop-param 0 :f32 [2 3 4])
          perm (uop/uop-permute p [2 0 1])
          rsh (uop/uop-reshape perm [4 1 6])
          exp (uop/uop-expand rsh [4 5 6])
          shr (uop/uop-shrink exp [[1 3] [0 5] [2 6]])
          pad (uop/uop-pad shr [[1 1] [0 0] [0 2]])
          flp (uop/uop-flip pad [true false true])]
      (is (= [4 2 3] (:shape perm)))
      (is (= [4 1 6] (:shape rsh)))
      (is (= [4 5 6] (:shape exp)))
      (is (= [2 5 4] (:shape shr)))
      (is (= [4 5 6] (:shape pad)))
      (is (= [4 5 6] (:shape flp))))))

(deftest test-uop-uir-formatting
  (testing "Rendering gemm_256x256 to geohot's .uir format"
    (uop/clear-uop-cache!)
    (let [b0 (uop/uop-buffer 0 :f32 65536)
          b1 (uop/uop-buffer 1 :f32 65536)
          r0 (uop/uop-reshape b0 [256 256 1])
          r1 (uop/uop-reshape b1 [256 1 256])
          e0 (uop/uop-expand r0 [256 256 256])
          e1 (uop/uop-expand r1 [256 256 256])
          m (uop/uop-alu :uop/mul e0 e1)
          red (uop/uop-reduce m :add [1])
          uir-str (uop/uop->uir-string red)]
      (is (string? uir-str))
      (is (.contains uir-str "buffer"))
      (is (.contains uir-str "reshape"))
      (is (.contains uir-str "mul"))
      (is (.contains uir-str "reduce"))
      (is (= [256 1 256] (:shape red))))))

(defspec prop-symbolic-rewrite-preserves-integer-semantics
  100
  (prop/for-all [a-val (gen/choose 0 31)
                 b-val (gen/choose 0 7)
                 k-val (gen/choose 8 16)
                 c-val (gen/choose 1 10)]
                (let [r-a (uop/uop-range (uop/uop-const :i32 32) 0)
                      r-b (uop/uop-range (uop/uop-const :i32 k-val) 1)
                      k-const (uop/uop-const :i32 k-val)
                      zero (uop/uop-const :i32 0)
                      one (uop/uop-const :i32 1)
                      c-const (uop/uop-const :i32 c-val)
                      ;; Construct compound index expression:
                      ;; (((r_a * k + r_b) // k) + 0) * 1 + ((r_a * k + r_b) % k) + (c - c)
                      lin (uop/uop-alu :uop/add
                                       (uop/uop-alu :uop/mul r-a k-const)
                                       r-b)
                      q (uop/uop-alu :uop/idiv lin k-const)
                      rem (uop/uop-alu :uop/mod lin k-const)
                      expr (uop/uop-alu :uop/add
                                        (uop/uop-alu :uop/add
                                                     (uop/uop-alu :uop/mul
                                                                  (uop/uop-alu :uop/add q zero)
                                                                  one)
                                                     rem)
                                        (uop/uop-alu :uop/sub c-const c-const))
                      simplified (rewrite/simplify-uop expr)
                      env {r-a a-val r-b b-val}
                      raw-eval (rewrite/eval-scalar-uop expr env)
                      simp-eval (rewrite/eval-scalar-uop simplified env)]
                  ;; Should simplify (r_a * k + r_b) // k -> r_a and (r_a * k + r_b) % k -> r_b
                  ;; so simplified expression is just (r_a + r_b)!
                  (and (= raw-eval simp-eval (+ a-val b-val))
                       (<= (count (uop/toposort simplified))
                           (count (uop/toposort expr)))))))

(defspec prop-movement-canonicalization-invariants
  50
  (prop/for-all [d0 (gen/choose 2 6)
                 d1 (gen/choose 2 6)
                 d2 (gen/choose 2 6)]
                (let [n (* d0 d1 d2)
                      p (uop/uop-param 0 :f32 [d0 d1 d2])
                      ;; reshape -> reshape collapses to single reshape
                      r1 (uop/uop-reshape p [(* d0 d1) d2])
                      r2 (uop/uop-reshape r1 [n])
                      r3 (uop/uop-reshape r2 [d0 d1 d2])
                      simp-r (rewrite/simplify-uop r3)
                      ;; permute -> inverse permute collapses to identity
                      p1 (uop/uop-permute p [2 0 1])
                      p2 (uop/uop-permute p1 [1 2 0])
                      simp-p (rewrite/simplify-uop p2)]
                  (and (= simp-r p)
                       (= simp-p p)))))

;; -----------------------------------------------------------------------------
;; Phase 2 Tests: Tensor Logic AST -> High-Level UOp Lowering
;; -----------------------------------------------------------------------------

(deftest test-lower-binary-contraction-to-uops
  (testing "Binary contraction [:= [:C :b :m :n] [:A :b :m :k] [:B :k :n]] lowers to movement + mul + reduce UOps"
    (let [invars [[:A [:tensor [2 4 8] :f32]]
                  [:B [:tensor [8 6] :f32]]]
          ast [:= [:C :b :m :n] [:A :b :m :k] [:B :k :n]]
          uop-graph (uop-lower/ast->uop-graph "gemm_uop" invars ast [:C])
          c-uop (get-in uop-graph [:outputs :C])
          nodes (uop/toposort c-uop)
          ops (set (map :op nodes))]
      (is (some? c-uop))
      (is (= [2 4 6] (:shape c-uop)))
      (is (= :f32 (:dtype c-uop)))
      (is (contains? ops :uop/mul))
      (is (contains? ops :uop/reduce))
      (is (contains? ops :uop/expand))
      (is (contains? ops :uop/reshape)))))

(deftest test-lower-gpt2-layer-and-full-model-to-uops
  (testing "Full GPT-2 model AST lowers cleanly to a valid UOp DAG"
    (let [max-seq-len 16
          num-layers 2
          ast (gpt2/gpt2-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
          base-invars [[:x [:tensor [1 max-seq-len] :i32]]
                       [:pos_ids [:tensor [1 max-seq-len] :i32]]
                       [:wte [:tensor [50257 768] :f32]]
                       [:wpe [:tensor [1024 768] :f32]]
                       [:ln_f_g [:tensor [768] :f32]]
                       [:ln_f_b [:tensor [768] :f32]]]
          layer-invars (vec (mapcat (fn [i]
                                      [[(keyword (str "ln1_g_" i)) [:tensor [768] :f32]]
                                       [(keyword (str "ln1_b_" i)) [:tensor [768] :f32]]
                                       [(keyword (str "attn_w_" i)) [:tensor [768 2304] :f32]]
                                       [(keyword (str "attn_b_" i)) [:tensor [2304] :f32]]
                                       [(keyword (str "proj_w_" i)) [:tensor [768 768] :f32]]
                                       [(keyword (str "proj_b_" i)) [:tensor [768] :f32]]
                                       [(keyword (str "ln2_g_" i)) [:tensor [768] :f32]]
                                       [(keyword (str "ln2_b_" i)) [:tensor [768] :f32]]
                                       [(keyword (str "mlp_fc_w_" i)) [:tensor [768 3072] :f32]]
                                       [(keyword (str "mlp_fc_b_" i)) [:tensor [3072] :f32]]
                                       [(keyword (str "mlp_proj_w_" i)) [:tensor [3072 768] :f32]]
                                       [(keyword (str "mlp_proj_b_" i)) [:tensor [768] :f32]]])
                                    (range num-layers)))
          invars (into base-invars layer-invars)
          uop-graph (uop-lower/ast->uop-graph "gpt2_2layer" invars ast [:logits])
          logits-uop (get-in uop-graph [:outputs :logits])]
      (is (some? logits-uop))
      (is (uop/valid-uop? logits-uop))
      (is (= [1 max-seq-len 50257] (:shape logits-uop)))
      (is (= :f32 (:dtype logits-uop))))))

(deftest test-lower-smollm-model-to-uops
  (testing "Full SmolLM model AST (with RMSNorm, RoPE, GQA, SwiGLU) lowers cleanly to a valid UOp DAG"
    (let [max-seq-len 16
          num-layers 2
          ast (smollm/smollm-model-ast {:num-layers num-layers :max-seq-len max-seq-len})
          base-invars [[:x [:tensor [1 max-seq-len] :i32]]
                       [:embed_tokens [:tensor [49152 576] :f32]]
                       [:final_norm_w [:tensor [576] :f32]]
                       [:lm_head_w [:tensor [49152 576] :f32]]]
          layer-invars (vec (mapcat (fn [i]
                                      [[(keyword (str "input_ln_w_" i)) [:tensor [576] :f32]]
                                       [(keyword (str "q_w_" i)) [:tensor [576 576] :f32]]
                                       [(keyword (str "k_w_" i)) [:tensor [192 576] :f32]]
                                       [(keyword (str "v_w_" i)) [:tensor [192 576] :f32]]
                                       [(keyword (str "o_w_" i)) [:tensor [576 576] :f32]]
                                       [(keyword (str "post_attn_ln_w_" i)) [:tensor [576] :f32]]
                                       [(keyword (str "gate_w_" i)) [:tensor [1536 576] :f32]]
                                       [(keyword (str "up_w_" i)) [:tensor [1536 576] :f32]]
                                       [(keyword (str "down_w_" i)) [:tensor [576 1536] :f32]]])
                                    (range num-layers)))
          invars (into base-invars layer-invars)
          uop-graph (uop-lower/ast->uop-graph "smollm_2layer" invars ast [:logits])
          logits-uop (get-in uop-graph [:outputs :logits])]
      (is (some? logits-uop))
      (is (uop/valid-uop? logits-uop))
      (is (= [1 max-seq-len 49152] (:shape logits-uop)))
      (is (= :f32 (:dtype logits-uop))))))
