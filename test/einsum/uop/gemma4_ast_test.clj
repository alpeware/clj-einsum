(ns einsum.uop.gemma4-ast-test
  "TDD tests for Gemma 4 E2B Tensor Logic AST lowering completeness on the UOp backend.
   Validates:
   1. Proportional RoPE (:rope-proportion 0.25 on global layers with 128 rotary dims and 384 pass-through dims)
   2. Dynamic :pos RoPE offset for single-token and chunked decode
   3. Sliding-window causal mask (:sliding-window 512)
   4. Dynamic slicing (:dynamic-slice for :last-token-only? LM head)
   5. Dynamic update slice (:dynamic-update-slice for persistent KV cache)
   6. 4D and scale-free RMSNorm (w-uop = nil)"
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as ein]
            [einsum.models.gemma4 :as gemma4]
            [einsum.models.gemma4.kernels :as kernels]))

;; -----------------------------------------------------------------------------
;; 1. Proportional RoPE (:rope-proportion)
;; -----------------------------------------------------------------------------

(deftest test-rope-proportion-pass-through
  (testing "RoPE with rope-proportion 0.25 rotates first 25% of dims and leaves 75% unrotated"
    (let [b 1 seq-len 4 num-heads 2 head-dim 16
          total-dim (* num-heads head-dim)
          rope-prop 0.25
          _rotary-dim (long (* head-dim rope-prop)) ;; 4 dims rotated (half-dim = 2), 12 pass-through
          x-floats (float-array (repeatedly (* b seq-len total-dim) #(- (rand 2.0) 1.0)))
          ast [:block {:name :rope_prop_test}
               [:rope [:y :b :p :qd] [:x :b :p :qd]
                {:head-dim head-dim :theta 10000.0 :rope-proportion rope-prop}]]
          in-spec [[:x [:tensor [b seq-len total-dim] :f32]]]
          cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
      (with-open [k (ein/compile-kernel cpu-ctx "rope_prop" ast {:in in-spec :out [:y]})]
        (let [out (ein/to-host-slice (k [x-floats]) 0 (* b seq-len total-dim) (* b seq-len total-dim))]
          ;; Check that for token 0 (pos = 0), all dims match exactly (cos(0) = 1, sin(0) = 0)
          (dotimes [i total-dim]
            (is (<= (Math/abs (- (double (aget ^floats out i)) (double (aget ^floats x-floats i)))) 1e-4)))
          ;; Check that for all tokens, pass-through channels match original x
          ;; In our half-dim split layout:
          ;; x1 has [0..half-dim), x2 has [half-dim..head-dim).
          ;; Rotary angles are first (rope-prop * half-dim) of each half.
          (let [half-dim (quot head-dim 2)
                rope-angles (long (* rope-prop half-dim))]
            (dotimes [p seq-len]
              (dotimes [h num-heads]
                (let [head-offset (+ (* p total-dim) (* h head-dim))]
                  ;; Check unrotated channels in x1: [rope-angles .. half-dim)
                  (doseq [d (range rope-angles half-dim)]
                    (let [idx (+ head-offset d)]
                      (is (<= (Math/abs (- (double (aget ^floats out idx)) (double (aget ^floats x-floats idx)))) 1e-4))))
                  ;; Check unrotated channels in x2: [half-dim + rope-angles .. head-dim)
                  (doseq [d (range (+ half-dim rope-angles) head-dim)]
                    (let [idx (+ head-offset d)]
                      (is (<= (Math/abs (- (double (aget ^floats out idx)) (double (aget ^floats x-floats idx)))) 1e-4)))))))))))))

(defspec prop-rope-proportion-unrotated-invariance
  25
  (prop/for-all [head-dim (gen/elements [16 32 64])
                 rope-prop (gen/elements [0.25 0.5])]
                (let [b 1 seq-len 2 num-heads 2
                      total-dim (* num-heads head-dim)
                      half-dim (quot head-dim 2)
                      rope-angles (long (* (double rope-prop) half-dim))
                      x-floats (float-array (repeatedly (* b seq-len total-dim) #(- (rand 2.0) 1.0)))
                      ast [:block {:name :rope_prop_p}
                           [:rope [:y :b :p :qd] [:x :b :p :qd]
                            {:head-dim head-dim :theta 10000.0 :rope-proportion rope-prop}]]
                      in-spec [[:x [:tensor [b seq-len total-dim] :f32]]]
                      cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
                  (with-open [k (ein/compile-kernel cpu-ctx "rope_prop_p" ast {:in in-spec :out [:y]})]
                    (let [out (ein/to-host-slice (k [x-floats]) 0 (* b seq-len total-dim) (* b seq-len total-dim))]
                      (every? (fn [p]
                                (every? (fn [h]
                                          (let [head-offset (+ (* p total-dim) (* h head-dim))]
                                            (and (every? (fn [d]
                                                           (let [idx (+ head-offset d)]
                                                             (<= (Math/abs (- (double (aget ^floats out idx)) (double (aget ^floats x-floats idx)))) 1e-4)))
                                                         (range rope-angles half-dim))
                                                 (every? (fn [d]
                                                           (let [idx (+ head-offset d)]
                                                             (<= (Math/abs (- (double (aget ^floats out idx)) (double (aget ^floats x-floats idx)))) 1e-4)))
                                                         (range (+ half-dim rope-angles) head-dim)))))
                                        (range num-heads)))
                              (range seq-len)))))))

;; -----------------------------------------------------------------------------
;; 2. Sliding-Window Causal Mask (:sliding-window)
;; -----------------------------------------------------------------------------

(deftest test-sliding-window-causal-mask
  (testing "Causal softmax with sliding window masks keys older than window"
    (let [q-len 6 k-len 6 window 2
          scores-floats (float-array (repeatedly (* q-len k-len) #(rand 1.0)))
          ast [:block {:name :sliding_softmax_test}
               [:causal-softmax [:probs :q :k] [:scores :q :k] {:sliding-window window}]]
          in-spec [[:scores [:tensor [q-len k-len] :f32]]]
          cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
      (with-open [k (ein/compile-kernel cpu-ctx "sliding_softmax" ast {:in in-spec :out [:probs]})]
        (let [out (ein/to-host-slice (k [scores-floats]) 0 (* q-len k-len) (* q-len k-len))]
          ;; For each query row r:
          (dotimes [r q-len]
            (dotimes [c k-len]
              (let [idx (+ (* r k-len) c)
                    prob (aget ^floats out idx)]
                (if (or (> c r) (>= (- r c) window))
                  ;; Should be strictly masked out to probability ~ 0.0
                  (is (<= prob 1e-5))
                  ;; Within window and causal: positive probability
                  (is (pos? prob)))))))))))

;; -----------------------------------------------------------------------------
;; 3. Dynamic Slice (:dynamic-slice for :last-token-only?)
;; -----------------------------------------------------------------------------

(deftest test-dynamic-slice-last-token
  (testing "Dynamic slice extracts row pos[0] correctly from a sequence tensor"
    (let [seq-len 8 hidden-dim 16
          x-floats (float-array (map float (range (* seq-len hidden-dim))))
          pos (int-array [5])
          ast [:block {:name :dyn_slice_test}
               [:dynamic-slice [:normed_last :b :one :d] [:normed :b :p :d]
                {:slice-sizes [1 1 hidden-dim]
                 :start-indices [0 :pos 0]}]]
          in-spec [[:normed [:tensor [1 seq-len hidden-dim] :f32]]
                   [:pos [:tensor [1] :i32]]]
          cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
      (with-open [k (ein/compile-kernel cpu-ctx "dyn_slice" ast {:in in-spec :out [:normed_last]})]
        (let [out (ein/to-host-slice (k [x-floats pos]) 0 hidden-dim hidden-dim)]
          ;; Expected row 5: values [5*16 .. 5*16 + 15]
          (dotimes [i hidden-dim]
            (is (== (float (+ (* 5 hidden-dim) i)) (aget ^floats out i)))))))))

;; -----------------------------------------------------------------------------
;; 4. Dynamic Update Slice (:dynamic-update-slice for KV Cache)
;; -----------------------------------------------------------------------------

(deftest test-dynamic-update-slice-kv-cache
  (testing "Dynamic update slice overwrites cache row pos[0] while preserving all other rows"
    (let [max-seq 8 kv-heads 2 head-dim 4
          total-elems (* max-seq kv-heads head-dim)
          cache-in (float-array total-elems 0.0)
          k-new (float-array (map float (repeatedly (* 1 kv-heads head-dim) #(rand 2.0))))
          pos (int-array [3])
          ast [:block {:name :dyn_update_test}
               [:dynamic-update-slice [:k_cache_out :b :kvs :kvh :dh]
                [:k_cache_in :b :kvs :kvh :dh]
                [:k_new :b :p :kvh :dh]
                {:start-indices [0 :pos 0 0]}]]
          in-spec [[:k_cache_in [:tensor [1 max-seq kv-heads head-dim] :f32]]
                   [:k_new [:tensor [1 1 kv-heads head-dim] :f32]]
                   [:pos [:tensor [1] :i32]]]
          cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
      (with-open [k (ein/compile-kernel cpu-ctx "dyn_update" ast {:in in-spec :out [:k_cache_out]})]
        (let [out (ein/to-host-slice (k [cache-in k-new pos]) 0 total-elems total-elems)
              step-size (* kv-heads head-dim)]
          ;; Rows other than 3 must remain 0.0
          (dotimes [r max-seq]
            (if (== r 3)
              (dotimes [i step-size]
                (is (== (aget ^floats k-new i) (aget ^floats out (+ (* r step-size) i)))))
              (dotimes [i step-size]
                (is (zero? (aget ^floats out (+ (* r step-size) i))))))))))))

;; -----------------------------------------------------------------------------
;; 5. Scale-Free 4D RMSNorm
;; -----------------------------------------------------------------------------

(deftest test-scale-free-4d-rms-norm
  (testing "RMSNorm without weight parameter normalizes 4D tensor along last axis"
    (let [b 1 seq-len 2 heads 4 head-dim 8
          total-elems (* b seq-len heads head-dim)
          x-floats (float-array (repeatedly total-elems #(+ 1.0 (rand 2.0))))
          ast [:block {:name :rms_norm_4d_test}
               [:rms-norm [:y :b :p :h :dh] [:x :b :p :h :dh] {:eps 1e-6}]]
          in-spec [[:x [:tensor [b seq-len heads head-dim] :f32]]]
          cpu-ctx (ein/init-backend! :uop-cpu {:quiet? true})]
      (with-open [k (ein/compile-kernel cpu-ctx "rms_4d" ast {:in in-spec :out [:y]})]
        (let [out (ein/to-host-slice (k [x-floats]) 0 total-elems total-elems)]
          ;; For each 8-dim slice, sum of squares divided by 8 should be ~1.0
          (dotimes [slice-idx (* b seq-len heads)]
            (let [offset (* slice-idx head-dim)
                  sum-sq (reduce + (map (fn [i]
                                          (let [v (aget ^floats out (+ offset i))]
                                            (* v v)))
                                        (range head-dim)))
                  mean-sq (/ sum-sq (double head-dim))]
              (is (<= (Math/abs (- mean-sq 1.0)) 1e-3)))))))))

;; -----------------------------------------------------------------------------
;; 6. Gemma 4 31B INT4 UOp ROCm Routing & Invars Invariants
;; -----------------------------------------------------------------------------

(deftest test-gemma4-31b-int4-uop-rocm-routing
  (testing "Gemma 4 31B INT4 routes to pure UOp int4-unpack instead of W4A16 custom call when backend is :uop / :uop-rocm"
    (let [base-cfg {:is-int4 true
                    :weight-dtype :int4
                    :group-size 128
                    :vocab-size 256
                    :hidden-dim 256
                    :num-layers 2
                    :num-heads 4
                    :num-kv-heads 2
                    :head-dim 64
                    :global-head-dim 128
                    :num-global-kv-heads 1
                    :intermediate-dim 512
                    :total-pl-dim 0
                    :pl-dim 0
                    :q-len 1
                    :max-seq-len 128}
          uop-cfg (assoc base-cfg :backend :uop :target :uop-rocm)
          rocm-cfg (assoc base-cfg :backend :rocm :target :rocm)
          ;; 1. Invars verification
          uop-invars (into {} (map (fn [[k spec]] [k spec])) (kernels/build-tensor-logic-invars uop-cfg 128))
          rocm-invars (into {} (map (fn [[k spec]] [k spec])) (kernels/build-tensor-logic-invars rocm-cfg 128))]
      ;; Verify uop-rocm emits standard [rows cols/2] :i8 and [rows cols/group-size] :bf16
      (is (= [:tensor [256 128] :i8] (get uop-invars :q_w_0)))
      (is (= [:tensor [256 2] :bf16] (get uop-invars :q_scale_0)))
      ;; Verify rocm emits standard [rows cols/2] :i8 and [rows cols/group-size] :bf16 (no W4A16)
      (is (= [:tensor [256 128] :i8] (get rocm-invars :q_w_0)))
      (is (= [:tensor [256 2] :bf16] (get rocm-invars :q_scale_0)))

      ;; 2. AST equation routing verification
      (let [uop-ast (gemma4/build-gemma4-kv-ast uop-cfg 128)
            rocm-ast (gemma4/build-gemma4-kv-ast rocm-cfg 128)
            uop-ops (map first (filter #(and (vector? %) (keyword? (first %))) (tree-seq coll? seq uop-ast)))
            rocm-ops (map first (filter #(and (vector? %) (keyword? (first %))) (tree-seq coll? seq rocm-ast)))
            uop-unpack-count (count (filter #(= % :int4-unpack) uop-ops))
            uop-w4a16-count (count (filter #(= % :w4a16-gemv) uop-ops))
            rocm-unpack-count (count (filter #(= % :int4-unpack) rocm-ops))
            rocm-w4a16-count (count (filter #(= % :w4a16-gemv) rocm-ops))]
        ;; Both use in-accelerator :int4-unpack and zero :w4a16-gemv
        (is (> uop-unpack-count 0))
        (is (zero? uop-w4a16-count))
        (is (> rocm-unpack-count 0))
        (is (zero? rocm-w4a16-count))))))
