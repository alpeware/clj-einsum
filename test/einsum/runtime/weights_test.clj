(ns einsum.runtime.weights-test
  "Generative property tests and unit tests for WeightStore and automated invar inference."
  (:require [einsum.compiler.stablehlo :as shlo]
            [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.weights :as weights]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest test-weight-store-resolution
  (testing "create-weight-store normalizes dtypes and resolves weights"
    (let [header {"model.embed_tokens.weight" {"shape" [1000 64] "dtype" "BF16"}
                  "model.layers.0.mlp.weight" {"shape" [64 256] "dtype" "F32"}}
          store (weights/create-weight-store header {:aliases {:embed_tokens "model.embed_tokens.weight"
                                                               :mlp_w "model.layers.0.mlp.weight"}})]
      (is (= {:shape [1000 64] :dtype :bf16} (weights/tensor-info store :embed_tokens)))
      (is (= {:shape [64 256] :dtype :f32} (weights/tensor-info store :mlp_w))))))

(deftest test-weight-store-fn-alias-resolution
  (testing "create-weight-store supports procedural resolver functions for aliases"
    (let [header {"layer.0.weight" {"shape" [32 32] "dtype" "F32"}
                  "layer.1.weight" {"shape" [32 32] "dtype" "F32"}}
          resolver (fn [k]
                     (let [k-str (name k)]
                       (when-let [[_ idx] (re-matches #"^w_(\d+)$" k-str)]
                         (format "layer.%s.weight" idx))))
          store (weights/create-weight-store header {:aliases resolver})]
      (is (= "layer.0.weight" (weights/resolve-tensor-name store :w_0)))
      (is (= "layer.1.weight" (weights/resolve-tensor-name store :w_1)))
      (is (= {:shape [32 32] :dtype :f32} (weights/tensor-info store :w_0)))
      (is (= {:shape [32 32] :dtype :f32} (weights/tensor-info store :w_1))))))

(deftest test-weight-store-get-device-buffer-with-arrays
  (testing "get-device-buffer transfers host array to device, tracks in arena, and caches"
    (let [ctx (xla/init-backend! :cpu)
          w-floats (float-array [1.0 2.0 3.0 4.0])
          source {"w.weight" w-floats}
          header {"w.weight" {"shape" [2 2] "dtype" "F32"}}]
      (with-open [arena (arena/create-arena ctx)]
        (let [store (weights/create-weight-store ctx {:header header :source source}
                                                 {:aliases {:w "w.weight"}
                                                  :arena arena})]
          (is (= 0 (count @(:device-buffers store))))
          (let [buf (weights/get-device-buffer store :w)]
            (is (some? buf))
            (is (= 1 (count @(:device-buffers store))))
            (is (= buf (get @(:device-buffers store) :w)))
            (is (contains? (arena/tracked-buffers arena) buf))
            ;; Second call returns cached buffer directly
            (is (identical? buf (weights/get-device-buffer store :w)))))))))

(deftest test-infer-invars-basic-gemm
  (testing "infer-invars automatically resolves weight invars from AST free variables"
    (let [ast [:= [:y :b :d] [:x :b :k] [:w :k :d]]
          dynamic-inputs {:x [:tensor [1 64] :f32]}
          weights-map {:w {:shape [64 128] :dtype :f32}}
          invars (weights/infer-invars ast dynamic-inputs weights-map)]
      (is (= [[:x [:tensor [1 64] :f32]]
              [:w [:tensor [64 128] :f32]]]
             invars))
      ;; Verify the inferred invars lower into a valid StableHLO graph via 4-arity
      (let [graph (lower/ast->graph "inferred_gemm" invars ast #{:y})]
        (is (some? (shlo/validate-graph graph))))
      ;; Verify lowering directly via 3-arity ast->graph with :inputs and :weights
      (let [graph-3 (lower/ast->graph "inferred_gemm_3" ast {:inputs dynamic-inputs
                                                             :weights weights-map
                                                             :targets #{:y}})]
        (is (some? (shlo/validate-graph graph-3)))
        (is (= invars (:invars graph-3)))))))

(defspec prop-inferred-invars-match-free-vars
  50
  (prop/for-all [batch (gen/choose 1 4)
                 dim-k (gen/choose 8 32)
                 dim-d (gen/choose 8 32)]
                (let [ast [:block {}
                           [:= [:h :b :k] [:x :b :k] [:w1 :k :k]]
                           [:= [:y :b :d] [:h :b :k] [:w2 :k :d]]]
                      dynamic-inputs {:x [:tensor [batch dim-k] :f32]}
                      weights-map {:w1 {:shape [dim-k dim-k] :dtype :f32}
                                   :w2 {:shape [dim-k dim-d] :dtype :f32}}
                      invars (weights/infer-invars ast dynamic-inputs weights-map)
                      invar-names (set (map first invars))]
                  (and (= invar-names #{:x :w1 :w2})
                       (= (get (into {} invars) :x) [:tensor [batch dim-k] :f32])
                       (= (get (into {} invars) :w1) [:tensor [dim-k dim-k] :f32])
                       (= (get (into {} invars) :w2) [:tensor [dim-k dim-d] :f32])))))

(defspec prop-resolver-fn-matches-map
  30
  (prop/for-all [layers (gen/choose 1 6)
                 dim (gen/choose 8 32)]
                (let [header (into {} (map (fn [i]
                                             [(format "h.%d.weight" i) {"shape" [dim dim] "dtype" "F32"}])
                                           (range layers)))
                      alias-map (into {} (map (fn [i]
                                                [(keyword (str "w_" i)) (format "h.%d.weight" i)])
                                              (range layers)))
                      alias-fn (fn [k]
                                 (when-let [[_ idx] (re-matches #"^w_(\d+)$" (name k))]
                                   (format "h.%s.weight" idx)))
                      store-map (weights/create-weight-store header {:aliases alias-map})
                      store-fn (weights/create-weight-store header {:aliases alias-fn})]
                  (every? (fn [i]
                            (let [kw (keyword (str "w_" i))]
                              (and (= (weights/resolve-tensor-name store-map kw)
                                      (weights/resolve-tensor-name store-fn kw))
                                   (= (weights/tensor-info store-map kw)
                                      (weights/tensor-info store-fn kw)))))
                          (range layers)))))
