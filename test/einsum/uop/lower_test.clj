(ns einsum.uop.lower-test
  "Generative property and unit tests for UOp lowering rules (Phase 7A):
   Coordinate-Aware :uop/iota, unsigned :i8 shift fix, :int4-unpack, and :ternary-unpack."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.uop.lower :as lower]
            [einsum.uop.rangeify :as rangeify]
            [einsum.uop.schedule :as schedule]))

;; -----------------------------------------------------------------------------
;; 1. Generative Property Test: INT4 Unpack & Fused Contraction Parity
;; -----------------------------------------------------------------------------

(defspec prop-int4-unpack-contraction-parity
  25
  (prop/for-all [m (gen/choose 1 4)
                 n (gen/choose 8 16)
                 num-groups (gen/choose 1 2)
                 group-size (gen/fmap (fn [g] (* 2 g)) (gen/choose 4 8))]
                (let [k (* num-groups group-size)
                      hk (quot k 2)
                      invars [[:x [:tensor [m k] :f32]]
                              [:w_packed [:tensor [n hk] :i8]]
                              [:scale [:tensor [n num-groups] :f32]]]
                      ast [:block {}
                           [:int4-unpack [:w_scaled :j :k] [:w_packed :j :hk] [:scale :j :g]]
                           [:= [:out :i :j] [:x :i :k] [:w_scaled :j :k]]]
                      ug (lower/ast->uop-graph "prop_int4_unpack" invars ast [:out])
                      sched (schedule/schedule-graph ug)
                      kernels (:kernels sched)
                      k0 (first kernels)
                      rk (:rangeified k0)
          ;; Generate random test inputs
                      x-arr (float-array (repeatedly (* m k) #(float (- (rand 2.0) 1.0))))
                      raw-nibbles (vec (repeatedly (* n k) #(rand-int 16)))
          ;; Pack two 4-bit nibbles per byte: low nibble at 2*h, high nibble at 2*h+1
                      packed-bytes (byte-array
                                    (for [j (range n)
                                          h (range hk)]
                                      (let [q0 (get raw-nibbles (+ (* j k) (* 2 h)))
                                            q1 (get raw-nibbles (+ (* j k) (* 2 h) 1))]
                                        (unchecked-byte (bit-or q0 (bit-shift-left q1 4))))))
                      scale-arr (float-array (repeatedly (* n num-groups) #(float (+ 0.5 (rand 1.0)))))
                      input-map {0 x-arr 1 packed-bytes 2 scale-arr}
                      sched-res ^floats (rangeify/eval-rangeified-kernel rk input-map)
          ;; Reference calculation
                      ref-res (float-array (* m n))]
                  (dotimes [i m]
                    (dotimes [j n]
                      (let [acc (atom 0.0)]
                        (dotimes [col k]
                          (let [nibble (get raw-nibbles (+ (* j k) col))
                                w-unscaled (- (double nibble) 8.0)
                                g (quot col group-size)
                                sc (double (aget scale-arr (+ (* j num-groups) g)))
                                w-val (* w-unscaled sc)
                                x-val (double (aget x-arr (+ (* i k) col)))]
                            (swap! acc + (* x-val w-val))))
                        (aset ref-res (+ (* i n) j) (float @acc)))))
                  (let [diffs (map (fn [a b] (Math/abs (- (double a) (double b))))
                                   (seq sched-res)
                                   (seq ref-res))
                        max-err (apply max diffs)]
                    (and (= 1 (count kernels))
                         (< max-err 1e-4))))))

;; -----------------------------------------------------------------------------
;; 2. Unit Test: Ternary Unpack & Fused Contraction Parity
;; -----------------------------------------------------------------------------

(deftest test-ternary-unpack-contraction-parity
  (testing "Ternary unpack lowers to zero :uop/cat view+iota+ALU ops and fuses into single contraction kernel"
    (let [m 2
          n 4
          num-groups 2
          group-size 8
          k (* num-groups group-size) ;; 16
          qk (quot k 4) ;; 4
          invars [[:x [:tensor [m k] :f32]]
                  [:w_packed [:tensor [n qk] :i8]]
                  [:scale [:tensor [n num-groups] :f32]]]
          ast [:block {}
               [:ternary-unpack [:w_ternary :j :k] [:w_packed :j :qk] [:scale :j :g]]
               [:= [:out :i :j] [:x :i :k] [:w_ternary :j :k]]]
          ug (lower/ast->uop-graph "test_ternary_unpack" invars ast [:out])
          sched (schedule/schedule-graph ug)
          kernels (:kernels sched)
          k0 (first kernels)
          rk (:rangeified k0)
          ;; Prepare inputs
          x-arr (float-array (repeatedly (* m k) #(float (- (rand 2.0) 1.0))))
          raw-trits (vec (repeatedly (* n k) #(rand-int 4)))
          ;; Pack four 2-bit trits per byte: trits 0..3
          packed-bytes (byte-array
                        (for [j (range n)
                              q (range qk)]
                          (let [t0 (get raw-trits (+ (* j k) (* 4 q)))
                                t1 (get raw-trits (+ (* j k) (* 4 q) 1))
                                t2 (get raw-trits (+ (* j k) (* 4 q) 2))
                                t3 (get raw-trits (+ (* j k) (* 4 q) 3))]
                            (unchecked-byte (bit-or t0
                                                    (bit-shift-left t1 2)
                                                    (bit-shift-left t2 4)
                                                    (bit-shift-left t3 6))))))
          scale-arr (float-array (repeatedly (* n num-groups) #(float (+ 0.5 (rand 1.0)))))
          input-map {0 x-arr 1 packed-bytes 2 scale-arr}
          sched-res ^floats (rangeify/eval-rangeified-kernel rk input-map)
          ref-res (float-array (* m n))]
      (dotimes [i m]
        (dotimes [j n]
          (let [acc (atom 0.0)]
            (dotimes [col k]
              (let [trit (get raw-trits (+ (* j k) col))
                    w-unscaled (- (double trit) 1.0)
                    g (quot col group-size)
                    sc (double (aget scale-arr (+ (* j num-groups) g)))
                    w-val (* w-unscaled sc)
                    x-val (double (aget x-arr (+ (* i k) col)))]
                (swap! acc + (* x-val w-val))))
            (aset ref-res (+ (* i n) j) (float @acc)))))
      (let [diffs (map (fn [a b] (Math/abs (- (double a) (double b))))
                       (seq sched-res)
                       (seq ref-res))
            max-err (apply max diffs)]
        (is (= 1 (count kernels)))
        (is (< max-err 1e-4))))))
