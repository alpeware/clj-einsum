(ns einsum.integration.rocm-e2e-test
  "End-to-end integration test suite for AMD ROCm GPUs, multi-device topology, and CPU vs ROCm execution parity."
  (:require [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.pjrt.version :as v]
            [einsum.compiler.stablehlo :as shlo]
            [einsum.test-helpers.parity :as p]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]))

(def ^:private rocm-ctx
  (delay
    (let [probe (v/probe-system-driver)
          rocm-so (io/file "bin/libpjrt_rocm.so")]
      (when (and (contains? (:detected-backends probe) :rocm) (.exists rocm-so))
        (let [api (pjrt/load-plugin! "bin/libpjrt_rocm.so")
              client (pjrt/create-client api)]
          {:api api :client client})))))

(deftest test-rocm-gpu-integration
  (if-let [{:keys [api client]} @rocm-ctx]
    (testing "AMD ROCm GPU initialization, device discovery, and hardware compilation"
      (is (some? (:api-ptr api)))
      (is (some? client))
      (let [devs (pjrt/addressable-devices api client)]
        (is (pos? (count devs)) "At least 1 ROCm device handle must be addressable")
        (when (>= (count devs) 2)
          (println (format "Dual AMD GPU Topology Detected: %d addressable ROCm devices." (count devs))))
        (let [graph {:name "rocm_e2e_module"
                     :invars [[:x [:tensor [1 64] :f32]]
                              [:y [:tensor [1 64] :f32]]]
                     :outvars [:z]
                     :eqns [{:op :stablehlo/multiply :invars [:x :y] :outvars [:z]}]}
              mlir (shlo/graph->mlir-text graph)
              exec (pjrt/compile-mlir api client mlir)]
          (is (some? exec) "ROCm hardware binary compilation must succeed"))))
    (println "Skipping ROCm E2E hardware test (ROCm backend or bin/libpjrt_rocm.so not loaded).")))

(deftest test-rocm-execution-parity
  (if-let [{:keys [api client]} @rocm-ctx]
    (testing "ROCm GPU hardware execution parity against Clojure reference math"
      (let [graph {:name "parity_test_graph"
                   :invars [[:x [:tensor [1 16] :f32]]]
                   :outvars [:y]
                   :eqns [{:op :stablehlo/constant :value 2.5 :outvars [:c0]}
                          {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
            inputs {:x (float-array (range 16))}
            actual-out (p/evaluate-graph-on-backend api client graph inputs)
            expected-out (mapv #(* (double %) 2.5) (range 16))
            match? (p/approx-equal? actual-out expected-out {:atol 1e-4 :rtol 1e-4})]
        (is (true? match?) (str "ROCm execution mismatch. Expected: " expected-out " Actual: " actual-out))
        (is (< (p/max-abs-diff actual-out expected-out) 1e-4))))
    (println "Skipping ROCm parity test (ROCm backend or bin/libpjrt_rocm.so not loaded).")))
