(ns clj-xla.interpret-diff-test
  "Differential tests: the JVM interpreter (clj-xla.interpret) must agree
   with PJRT CPU on identical small graphs, op by op. This is the semantic
   parity gate before wiring Gemma inference to the JVM backend."
  (:require [clj-xla.interpret :as interp]
            [clj-xla.pjrt :as pjrt]
            [clj-xla.stablehlo :as shlo]
            [clojure.test :refer [deftest is use-fixtures]]))

;; ---------------------------------------------------------------------------
;; PJRT CPU fixture (loaded once per test run)
;; ---------------------------------------------------------------------------

(def ^:private api-ctx (atom nil))
(def ^:private client (atom nil))

(defn- ensure-pjrt! []
  (when (nil? @api-ctx)
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          cli (pjrt/create-client api)]
      (reset! api-ctx api)
      (reset! client cli)))
  [@api-ctx @client])

(use-fixtures :once (fn [f] (ensure-pjrt!) (f)))

;; ---------------------------------------------------------------------------
;; Runners
;; ---------------------------------------------------------------------------

(def ^:private dtype-enum {:f32 11 :bf16 13 :i32 4 :pred 1})

(defn- pjrt-run
  "Compiles single-output `graph` and runs it on PJRT CPU. `inputs` maps
   invar keywords to {:shape [...] :dtype kw :data flat-sequential}.
   `out-count` is the expected number of output elements.
   Returns a float vector of the output."
  [graph inputs out-count out-dtype]
  (let [[api cli] (ensure-pjrt!)
        mlir (shlo/graph->mlir-text graph)
        exec (pjrt/compile-mlir api cli mlir)
        bufs (mapv (fn [[v _]]
                     (let [{:keys [shape dtype data]} (get inputs v)
                           arr (if (= dtype :i32)
                                 (int-array (map int data))
                                 (float-array (map float data)))]
                       (pjrt/buffer-from-host-buffer api cli arr (vec shape)
                                                     (get dtype-enum dtype 11))))
                   (:invars graph))]
    (try
      (let [out-buf (pjrt/execute-executable api exec bufs 1)]
        (try
          (if (= out-dtype :i32)
            (vec (map float (pjrt/buffer-to-host-int-buffer api out-buf (long out-count))))
            (vec (pjrt/buffer-to-host-buffer api out-buf (long out-count) :f32)))
          (finally (pjrt/destroy-buffer! api out-buf))))
      (finally
        (doseq [b bufs] (pjrt/destroy-buffer! api b))
        (pjrt/destroy-loaded-executable! api exec)))))

(defn- interp-run
  "Runs single-output `graph` on the JVM interpreter. `inputs` as in pjrt-run.
   Returns the output tensor."
  [graph inputs]
  (let [bindings (into {}
                       (map (fn [[v _]]
                              (let [{:keys [shape dtype data]} (get inputs v)]
                                [v (interp/tensor dtype (vec shape) (vec data))]))
                            (:invars graph)))
        out (interp/execute graph bindings)]
    (get out (first (:outvars graph)))))

(defn- max-abs-diff [xs ys]
  (reduce max 0.0 (map (fn [x y] (Math/abs (double (- x y)))) xs ys)))

(defn- assert-parity!
  "Runs single-output `graph` on both backends and asserts the outputs agree
   within `tol`. `inputs` maps invar -> {:shape [...] :dtype kw :data [...]}.
   `out-shape` is the expected output shape."
  [graph inputs out-shape tol]
  (let [t (interp-run graph inputs)
        want-count (long (reduce * 1 out-shape))
        got (vec (take want-count (mapv float (interp/as-floats t))))
        pjrt-got (pjrt-run graph inputs want-count (:dtype t))
        d (max-abs-diff got pjrt-got)]
    (is (= (vec out-shape) (vec (:shape t))) "interpreter shape mismatch")
    (is (= want-count (count pjrt-got)) "pjrt output count mismatch")
    (is (< d tol) (str "parity failed: max-abs-diff=" d " tol=" tol))))

;; ---------------------------------------------------------------------------
;; Elementwise arithmetic
;; ---------------------------------------------------------------------------

(deftest diff-multiply-constant
  (let [graph {:name "diff_mul_const"
               :invars [[:x [:tensor [1 64] :f32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/constant :value 1.5 :outvars [:c0]}
                      {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
        xs (mapv float (range 64))]
    (assert-parity! graph {:x {:shape [1 64] :dtype :f32 :data xs}} [1 64] 1e-5)))

(deftest diff-binary-broadcast
  ;; [2 3] op [3] with broadcasting; divide avoids zeros
  (doseq [op [:stablehlo/add :stablehlo/subtract
              :stablehlo/multiply :stablehlo/divide]]
    (let [graph {:name (str "diff_" (name op))
                 :invars [[:x [:tensor [2 3] :f32]] [:y [:tensor [3] :f32]]]
                 :outvars [:z]
                 :eqns [{:op op :invars [:x :y] :outvars [:z]}]}
          xs (mapv float (range 1 7))
          ys [2.0 4.0 8.0]]
      (assert-parity! graph
                      {:x {:shape [2 3] :dtype :f32 :data xs}
                       :y {:shape [3] :dtype :f32 :data ys}}
                      [2 3] 1e-5))))

(deftest diff-unary-math
  (doseq [op [:stablehlo/negate :stablehlo/rsqrt :stablehlo/exp
              :stablehlo/tanh :stablehlo/sine :stablehlo/cosine]]
    (let [graph {:name (str "diff_" (name op))
                 :invars [[:x [:tensor [8] :f32]]]
                 :outvars [:y]
                 :eqns [{:op op :invars [:x] :outvars [:y]}]}
          ;; positive inputs keep rsqrt finite; modest range keeps exp finite
          xs (mapv #(+ 0.25 (* 0.5 %)) (range 8))]
      (assert-parity! graph {:x {:shape [8] :dtype :f32 :data xs}} [8] 1e-4))))

(deftest diff-dot-general
  ;; 4x8 @ 8x5, contracting dims lhs[1] rhs[0]
  (let [m 4 n 5 k 8
        graph {:name "diff_dot"
               :invars [[:a [:tensor [4 8] :f32]] [:b [:tensor [8 5] :f32]]]
               :outvars [:c]
               :eqns [{:op :stablehlo/dot_general :invars [:a :b] :outvars [:c]
                       :attrs {:contracting_dims {:lhs [1] :rhs [0]}
                               :batch_dims {:lhs [] :rhs []}}}]}
        xs (mapv #(/ (double %) 8.0) (range (* m k)))
        ys (mapv #(/ (double %) 8.0) (range (* k n)))]
    (assert-parity! graph
                    {:a {:shape [4 8] :dtype :f32 :data xs}
                     :b {:shape [8 5] :dtype :f32 :data ys}}
                    [4 5] 1e-4)))

(deftest diff-dot-general-batched
  ;; [2 4 8] @ [2 8 5], batch [0], contracting lhs[2] rhs[1]
  (let [graph {:name "diff_dot_batch"
               :invars [[:a [:tensor [2 4 8] :f32]] [:b [:tensor [2 8 5] :f32]]]
               :outvars [:c]
               :eqns [{:op :stablehlo/dot_general :invars [:a :b] :outvars [:c]
                       :attrs {:contracting_dims {:lhs [2] :rhs [1]}
                               :batch_dims {:lhs [0] :rhs [0]}}}]}
        xs (mapv #(/ (double %) 64.0) (range (* 2 4 8)))
        ys (mapv #(/ (double %) 64.0) (range (* 2 8 5)))]
    (assert-parity! graph
                    {:a {:shape [2 4 8] :dtype :f32 :data xs}
                     :b {:shape [2 8 5] :dtype :f32 :data ys}}
                    [2 4 5] 1e-4)))

(deftest diff-structural-ops
  (let [xs (mapv float (range 24))]
    ;; reshape [2 12] -> [4 3 2]
    (assert-parity!
     {:name "diff_reshape"
      :invars [[:x [:tensor [2 12] :f32]]]
      :outvars [:y]
      :eqns [{:op :stablehlo/reshape :invars [:x] :outvars [:y]
              :attrs {:shape [4 3 2]}}]}
     {:x {:shape [2 12] :dtype :f32 :data xs}} [4 3 2] 1e-6)
    ;; transpose [2 3 4] permutation [2 0 1] -> [4 2 3]
    (assert-parity!
     {:name "diff_transpose"
      :invars [[:x [:tensor [2 3 4] :f32]]]
      :outvars [:y]
      :eqns [{:op :stablehlo/transpose :invars [:x] :outvars [:y]
              :attrs {:permutation [2 0 1]}}]}
     {:x {:shape [2 3 4] :dtype :f32 :data xs}} [4 2 3] 1e-6)
    ;; broadcast_in_dim [3] -> [2 3], dims [1]
    (assert-parity!
     {:name "diff_bcast"
      :invars [[:x [:tensor [3] :f32]]]
      :outvars [:y]
      :eqns [{:op :stablehlo/broadcast_in_dim :invars [:x] :outvars [:y]
              :attrs {:broadcast_dimensions [1] :target_shape [2 3]}}]}
     {:x {:shape [3] :dtype :f32 :data [1.0 2.0 3.0]}} [2 3] 1e-6)
    ;; concatenate [2 3] + [2 5] along dim 1 -> [2 8]
    (assert-parity!
     {:name "diff_concat"
      :invars [[:a [:tensor [2 3] :f32]] [:b [:tensor [2 5] :f32]]]
      :outvars [:y]
      :eqns [{:op :stablehlo/concatenate :invars [:a :b] :outvars [:y]
              :attrs {:dimension 1}}]}
     {:a {:shape [2 3] :dtype :f32 :data (mapv float (range 6))}
      :b {:shape [2 5] :dtype :f32 :data (mapv float (range 10 20))}}
     [2 8] 1e-6)
    ;; slice [4 6] -> [2 3] starting at [1 2]
    (assert-parity!
     {:name "diff_slice"
      :invars [[:x [:tensor [4 6] :f32]]]
      :outvars [:y]
      :eqns [{:op :stablehlo/slice :invars [:x] :outvars [:y]
              :attrs {:start_indices [1 2] :limit_indices [3 5] :strides [1 1]}}]}
     {:x {:shape [4 6] :dtype :f32 :data xs}} [2 3] 1e-6)))

(deftest diff-iota
  ;; MLIR emitter only supports 1D iota; match that.
  (let [graph {:name "diff_iota"
               :invars []
               :outvars [:y]
               :eqns [{:op :stablehlo/iota :invars [] :outvars [:y]
                       :attrs {:iota_dimension 0 :dtype :f32 :len 10}}]}]
    (assert-parity! graph {} [10] 1e-6)))

(deftest diff-convert-roundtrip
  ;; f32 -> bf16 -> f32 loses precision; compare against PJRT's own bf16 path
  (let [xs (mapv #(/ (double %) 3.0) (range 24))
        graph {:name "diff_convert"
               :invars [[:x [:tensor [24] :f32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/convert :invars [:x] :outvars [:b]
                       :attrs {:target_dtype :bf16}}
                      {:op :stablehlo/convert :invars [:b] :outvars [:y]
                       :attrs {:target_dtype :f32}}]}]
    (assert-parity! graph {:x {:shape [24] :dtype :f32 :data xs}} [24] 1e-2)))

(deftest diff-compare-select
  (let [xs (mapv float (range 8))
        ys (mapv #(+ 3.5 %) (range 8))
        graph {:name "diff_cmp_sel"
               :invars [[:x [:tensor [8] :f32]] [:y [:tensor [8] :f32]]]
               :outvars [:z]
               :eqns [{:op :stablehlo/compare :invars [:x :y] :outvars [:p]
                       :attrs {:comparison_direction "GT"}}
                      {:op :stablehlo/select :invars [:p :x :y] :outvars [:z]}]}]
    (assert-parity! graph
                    {:x {:shape [8] :dtype :f32 :data xs}
                     :y {:shape [8] :dtype :f32 :data ys}}
                    [8] 1e-6)))

(deftest diff-reductions
  (let [xs (mapv #(+ 1.0 %) (range 24))]
    (doseq [[op kw] [[:stablehlo/reduce_sum :sum]
                     [:stablehlo/reduce_mean :mean]
                     [:stablehlo/reduce_max :max]]]
      (assert-parity!
       {:name (str "diff_" (name kw))
        :invars [[:x [:tensor [4 6] :f32]]]
        :outvars [:y]
        :eqns [{:op op :invars [:x] :outvars [:y]
                :attrs {:axes [1] :keep_dims false}}]}
       {:x {:shape [4 6] :dtype :f32 :data xs}} [4] 1e-4))))

(deftest diff-gather-embedding
  ;; operand [8 4], indices [3 1] -> [3 4] embedding lookup
  (let [graph {:name "diff_gather"
               :invars [[:emb [:tensor [8 4] :f32]] [:idx [:tensor [3 1] :i32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/gather :invars [:emb :idx] :outvars [:y]
                       :attrs {:offset_dims [1]
                               :collapsed_slice_dims [0]
                               :start_index_map [0]
                               :index_vector_dim 1
                               :slice_sizes [1 4]}}]}
        emb (mapv float (range 32))]
    (assert-parity! graph
                    {:emb {:shape [8 4] :dtype :f32 :data emb}
                     :idx {:shape [3 1] :dtype :i32 :data [5 0 7]}}
                    [3 4] 1e-6)))

(deftest diff-dynamic-update-slice
  ;; operand [4 4], update [2 2] at [1 1]
  (let [graph {:name "diff_dus"
               :invars [[:x [:tensor [4 4] :f32]] [:u [:tensor [2 2] :f32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/dynamic_update_slice
                       :invars [:x :u] :outvars [:y]
                       :attrs {:start_indices [1 1]}}]}
        xs (mapv float (range 16))
        us [100.0 101.0 102.0 103.0]]
    (assert-parity! graph
                    {:x {:shape [4 4] :dtype :f32 :data xs}
                     :u {:shape [2 2] :dtype :f32 :data us}}
                    [4 4] 1e-6)))

(deftest diff-integer-bitwise
  ;; int4 unpack pattern: (packed >> 4) & 15, low nibble via & 15
  (let [packed [0xAB 0x3C 0xF0 0x12 0x7E 0x94 0x5A 0xC3]
        graph {:name "diff_intops"
               :invars [[:p [:tensor [8] :i32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/constant :value 4 :outvars [:four] :type [:tensor [] :i32]}
                      {:op :stablehlo/shift_right_logical :invars [:p :four] :outvars [:hi]}
                      {:op :stablehlo/constant :value 15 :outvars [:mask] :type [:tensor [] :i32]}
                      {:op :stablehlo/and :invars [:hi :mask] :outvars [:y]}]}]
    (assert-parity! graph {:p {:shape [8] :dtype :i32 :data packed}} [8] 1e-6)))

(deftest diff-maximum-broadcast
  (let [graph {:name "diff_max_bcast"
               :invars [[:x [:tensor [2 3] :f32]] [:y [:tensor [3] :f32]]]
               :outvars [:z]
               :eqns [{:op :stablehlo/maximum :invars [:x :y] :outvars [:z]}]}
        xs (mapv float (range 6))
        ys [2.5 1.0 4.0]]
    (assert-parity! graph
                    {:x {:shape [2 3] :dtype :f32 :data xs}
                     :y {:shape [3] :dtype :f32 :data ys}}
                    [2 3] 1e-6)))

(deftest diff-or
  ;; (p | q) on i32, exercising op-or alongside op-and
  (let [graph {:name "diff_or"
               :invars [[:p [:tensor [8] :i32]] [:q [:tensor [8] :i32]]]
               :outvars [:y]
               :eqns [{:op :stablehlo/or :invars [:p :q] :outvars [:y]}]}
        ps [0x0F 0xF0 0x33 0xCC 0x55 0xAA 0x01 0x80]
        qs [0xF0 0x0F 0xCC 0x33 0xAA 0x55 0xFE 0x7F]]
    (assert-parity! graph
                    {:p {:shape [8] :dtype :i32 :data ps}
                     :q {:shape [8] :dtype :i32 :data qs}}
                    [8] 1e-6)))
