(ns einsum.logic.quip-test
  "Generative property and unit tests for QuIP# (Hadamard Incoherence and Lattice Codebooks)."
  (:require [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [einsum.logic.lower :as lower]
            [einsum.logic.quip :as quip]
            [einsum.logic.shape :as shape]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; --- Mathematical Reference Helpers ---

(defn- l2-norm [v]
  (Math/sqrt (reduce + (map (fn [x] (* x x)) v))))

(defn- close?
  ([a b] (close? a b 1e-4))
  ([a b eps]
   (let [diff (Math/abs (double (- a b)))]
     (<= diff eps))))

(defn- vec-close?
  ([v1 v2] (vec-close? v1 v2 1e-4))
  ([v1 v2 eps]
   (and (= (count v1) (count v2))
        (every? true? (map #(close? %1 %2 eps) v1 v2)))))

;; Explicit Sylvester-Hadamard recursive matrix generator
(defn sylvester-hadamard [n]
  (if (= n 1)
    [[1.0]]
    (let [half (/ n 2)
          h-half (sylvester-hadamard half)
          inv-sqrt2 (/ 1.0 (Math/sqrt 2.0))]
      (vec
       (concat
        (mapv (fn [row]
                (vec (mapv #(* % inv-sqrt2) (concat row row))))
              h-half)
        (mapv (fn [row]
                (vec (concat (mapv #(* % inv-sqrt2) row)
                             (mapv #(* (- %) inv-sqrt2) row))))
              h-half))))))

(defn mat-vec-mul [m v]
  (mapv (fn [row] (reduce + (map * row v))) m))

;; --- Generative Property Tests (Rule 1: Strict TDD) ---

(def gen-power-of-2-dim
  (gen/elements [2 4 8 16 32 64]))

(defspec prop-fwht-involution
  50
  (prop/for-all [d gen-power-of-2-dim
                 v (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 64)]
                (let [x (vec (take d v))
                      h1 (quip/fwht-ref x)
                      h2 (quip/fwht-ref h1)]
                  (vec-close? x h2 1e-4))))

(defspec prop-fwht-norm-conservation
  50
  (prop/for-all [d gen-power-of-2-dim
                 v (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 64)]
                (let [x (vec (take d v))
                      norm-in (l2-norm x)
                      norm-out (l2-norm (quip/fwht-ref x))]
                  (close? norm-in norm-out 1e-4))))

(defspec prop-rht-involution
  50
  (prop/for-all [d gen-power-of-2-dim
                 v (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 64)
                 signs (gen/vector (gen/elements [-1.0 1.0]) 64)]
                (let [x (vec (take d v))
                      s (vec (take d signs))
                      r1 (quip/rht-ref x s)
                      r2 (quip/rht-inv-ref r1 s)]
                  (vec-close? x r2 1e-4))))

;; --- Unit Tests for Mathematics & Codebooks ---

(deftest test-fwht-sylvester-parity
  (testing "Verify FWHT reference matches explicit Sylvester-Hadamard matrix multiplication."
    (doseq [d [2 4 8 16 32]]
      (let [x (mapv double (range 1 (inc d)))
            h-mat (sylvester-hadamard d)
            expected (mat-vec-mul h-mat x)
            actual (quip/fwht-ref x)]
        (is (vec-close? expected actual 1e-4))))))

(deftest test-e8p-codebook-properties
  (testing "Verify E8P codebook construction according to QuIP# paper Appendix C.1."
    (let [cb (quip/generate-e8p-codebook)]
      (is (= 256 (count cb)) "E8P codebook must contain exactly 256 codewords")
      (is (every? #(= 8 (count %)) cb) "Every codeword must be 8-dimensional")
      ;; The first 227 elements must have norm squared <= 10
      (doseq [v (take 227 cb)]
        (let [norm-sq (reduce + (map (fn [x] (* x x)) v))]
          (is (<= norm-sq (+ 10.0 1e-5)) (str "Codeword " v " norm squared must be <= 10"))))
      ;; The 29 padding elements must have norm squared 12
      (doseq [v (drop 227 cb)]
        (let [norm-sq (reduce + (map (fn [x] (* x x)) v))]
          (is (close? norm-sq 12.0 1e-4) (str "Padding codeword " v " norm squared must be 12")))))))

;; --- AST Shape Unification & Lowering Tests ---

(deftest test-fwht-ast-shape-unification
  (let [in-shapes {:x [2 16 64]}
        eqns [[:fwht [:y :b :s :d] [:x :b :s :d]]]
        resolved (shape/unify-shapes in-shapes eqns)]
    (is (= [2 16 64] (get resolved :y)))))

(deftest test-rht-ast-shape-unification
  (let [in-shapes {:x [2 16 64] :s [64]}
        eqns [[:rht [:y :b :s :d] [:x :b :s :d] [:s :d]]]
        resolved (shape/unify-shapes in-shapes eqns)]
    (is (= [2 16 64] (get resolved :y)))))

(deftest test-quip-dequant-shape-unification
  (let [in-shapes {:codes [64 16] :cb [256 8] :scales [128]}
        eqns [[:quip-dequant [:w_deq :k :n] [:codes :k :n8] [:cb :c :dim8] [:scales :n]]]
        resolved (shape/unify-shapes in-shapes eqns)]
    ;; 16 blocks * 8 = 128 out-features
    (is (= [64 128] (get resolved :w_deq)))))

(deftest test-fwht-lowering-produces-valid-graph
  (let [invars [[:x [:tensor [1 4 16] :f32]]]
        ast [:fwht [:y :b :s :d] [:x :b :s :d]]
        graph (lower/ast->graph "fwht_graph" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (is (= [:y] (:outvars graph)))))

(deftest test-rht-lowering-produces-valid-graph
  (let [invars [[:x [:tensor [1 4 16] :f32]]
                [:s [:tensor [16] :f32]]]
        ast [:rht [:y :b :s :d] [:x :b :s :d] [:s :d]]
        graph (lower/ast->graph "rht_graph" invars ast #{:y})]
    (is (shlo/validate-graph graph))
    (is (= [:y] (:outvars graph)))))

(deftest test-quip-dequant-lowering-produces-valid-graph
  (let [invars [[:codes [:tensor [32 4] :i32]]
                [:cb [:tensor [256 8] :f32]]
                [:scales [:tensor [32] :f32]]]
        ast [:quip-dequant [:w_deq :k :n] [:codes :k :n8] [:cb :c :dim8] [:scales :n]]
        graph (lower/ast->graph "quip_dequant_graph" invars ast #{:w_deq})]
    (is (shlo/validate-graph graph))
    (is (= [:w_deq] (:outvars graph)))))

;; --- End-to-End OpenXLA PJRT Compilation and Execution ---

(deftest test-fwht-pjrt-execution-e2e
  (let [ctx (xla/get-context)
        d 8
        invars [[:x [:tensor [1 d] :f32]]]
        ast [:fwht [:y :b :d] [:x :b :d]]
        compiled (logic/compile-ast ctx "fwht_e2e" invars ast #{:y})
        input (float-array [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])
        out-buf (xla/execute compiled input)
        res (vec (xla/to-host-slice out-buf 0 d 4))
        expected (quip/fwht-ref [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])]
    (is (vec-close? expected res 1e-4))
    (xla/destroy-buffer! out-buf)))

(deftest test-rht-pjrt-execution-e2e
  (let [ctx (xla/get-context)
        d 8
        invars [[:x [:tensor [1 d] :f32]]
                [:s [:tensor [d] :f32]]]
        ast [:rht [:y :b :d] [:x :b :d] [:s :d]]
        compiled (logic/compile-ast ctx "rht_e2e" invars ast #{:y})
        input (float-array [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])
        signs (float-array [1.0 -1.0 1.0 -1.0 1.0 1.0 -1.0 -1.0])
        out-buf (xla/execute compiled input signs)
        res (vec (xla/to-host-slice out-buf 0 d 4))
        expected (quip/rht-ref [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
                               [1.0 -1.0 1.0 -1.0 1.0 1.0 -1.0 -1.0])]
    (is (vec-close? expected res 1e-4))
    (xla/destroy-buffer! out-buf)))

(deftest test-quip-linear-e2e-gemm-parity
  (testing "Verify that a QuIP# linear layer with exact weights reproduces unquantized GEMM."
    (let [ctx (xla/get-context)
          batch 1
          k 16
          n 16
          w-flat (vec (map double (range (* k n))))
          w-mat (mapv vec (partition n w-flat))
          su (vec (repeat k 1.0))
          sv (vec (repeat n 1.0))
          w-col-had (mapv (fn [col-idx]
                            (quip/fwht-ref (mapv #(nth % col-idx) w-mat)))
                          (range n))
          w-tilde-cols (mapv (fn [row-idx]
                               (quip/fwht-ref (mapv #(nth % row-idx) w-col-had)))
                             (range k))
          w-tilde-flat (float-array (flatten w-tilde-cols))
          ast [:block {:name :quip_linear_block}
               [:rht [:x_rot :b :k] [:x :b :k] [:su :k]]
               [:= [:y_rot :b :n] [:x_rot :b :k] [:w_tilde :k :n]]
               [:rht [:y :b :n] [:y_rot :b :n] [:sv :n]]]
          invars [[:x [:tensor [batch k] :f32]]
                  [:su [:tensor [k] :f32]]
                  [:w_tilde [:tensor [k n] :f32]]
                  [:sv [:tensor [n] :f32]]]
          compiled (logic/compile-ast ctx "quip_linear_e2e" invars ast #{:y})
          x-vec (vec (map double (range 1 (inc k))))
          x-arr (float-array x-vec)
          su-arr (float-array su)
          sv-arr (float-array sv)
          out-buf (xla/execute compiled x-arr su-arr w-tilde-flat sv-arr)
          actual-res (vec (xla/to-host-slice out-buf 0 n 4))
          expected-res (mapv (fn [j]
                               (reduce + (map (fn [i] (* (nth x-vec i) (nth (nth w-mat i) j))) (range k))))
                             (range n))]
      (is (vec-close? expected-res actual-res 1e-3))
      (xla/destroy-buffer! out-buf))))

(deftest test-quip-linear-ast-with-e8p-codebook-e2e
  (testing "Verify complete QuIP# linear layer forward pass with E8P codebook vector dequantization."
    (let [ctx (xla/get-context)
          batch 1
          k 16
          n 16
          n8 2
          cb-matrix (quip/generate-e8p-codebook)
          cb-flat (float-array (flatten cb-matrix))
          codes-vec (vec (take (* k n8) (cycle [10 42 100 220 5 17 88 150])))
          codes-arr (int-array codes-vec)
          scales-vec (vec (repeat n 0.125))
          scales-arr (float-array scales-vec)
          su-vec (vec (take k (cycle [1.0 -1.0 1.0 1.0 -1.0 1.0 -1.0 -1.0])))
          su-arr (float-array su-vec)
          sv-vec (vec (take n (cycle [-1.0 1.0 1.0 -1.0 -1.0 1.0 1.0 -1.0])))
          sv-arr (float-array sv-vec)
          x-vec (mapv double (range 1 (inc k)))
          x-arr (float-array x-vec)
          ast (quip/quip-linear-ast [:y :b :n] [:x :b :k] [:su :k] [:codes :k :n8] [:cb :c :dim8] [:scales :n] [:sv :n])
          invars [[:x [:tensor [batch k] :f32]]
                  [:su [:tensor [k] :f32]]
                  [:codes [:tensor [k n8] :i32]]
                  [:cb [:tensor [256 8] :f32]]
                  [:scales [:tensor [n] :f32]]
                  [:sv [:tensor [n] :f32]]]
          compiled (logic/compile-ast ctx "quip_linear_ast_e2e" invars ast #{:y})
          out-buf (xla/execute compiled x-arr su-arr codes-arr cb-flat scales-arr sv-arr)
          actual-res (vec (xla/to-host-slice out-buf 0 n 4))
          codes-partitioned (partition n8 (vec codes-arr))
          w-deq (mapv (fn [row-codes]
                        (let [row-unscaled (vec (mapcat #(nth cb-matrix %) row-codes))]
                          (mapv * row-unscaled scales-vec)))
                      codes-partitioned)
          x-rot (quip/rht-ref x-vec (vec su-arr))
          y-rot (mapv (fn [j]
                        (reduce + (map (fn [i] (* (nth x-rot i) (nth (nth w-deq i) j))) (range k))))
                      (range n))
          expected-res (quip/rht-inv-ref y-rot (vec sv-arr))]
      (is (vec-close? expected-res actual-res 1e-3))
      (xla/destroy-buffer! out-buf))))

