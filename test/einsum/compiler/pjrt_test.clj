(ns einsum.compiler.pjrt-test
  "Unit tests for PJRT plugin loading and compilation."
  (:require [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.stablehlo :as shlo]
            [clojure.test :refer [deftest is testing]]))

(deftest pjrt-companion-lib-test
  (testing "companion-lib? filters out main PJRT plugin binaries and rocprofiler while allowing runtime shared libraries"
    (is (false? (pjrt/companion-lib? "xla_cpu_pjrt.so")))
    (is (false? (pjrt/companion-lib? "pjrt_plugin_xpu.so")))
    (is (false? (pjrt/companion-lib? "libpjrt_cpu.so")))
    (is (false? (pjrt/companion-lib? "libpjrt_sycl.so")))
    (is (false? (pjrt/companion-lib? "libpjrt_rocm.so")))
    (is (false? (pjrt/companion-lib? "rocprofiler.so")))
    (is (true? (pjrt/companion-lib? "libsycl.so")))
    (is (true? (pjrt/companion-lib? "libmkl_core.so.2")))
    (is (true? (pjrt/companion-lib? "libccl.so")))))

(deftest pjrt-cpu-plugin-test
  (testing "Loading CPU PJRT plugin and initializing client"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")]
      (is (some? (:api-ptr api)))
      (is (some? (:linker api)))
      (let [[major minor] (pjrt/api-version api)]
        (is (integer? major))
        (is (integer? minor))
        (is (>= major 0))
        (is (>= minor 0)))
      (let [client (pjrt/create-client api)]
        (is (some? client))
        (is (= "cpu" (pjrt/platform-name api client)))))))

(deftest pjrt-compile-mlir-test
  (testing "Compiling StableHLO MLIR text to hardware executable via PJRT"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "test_compile_module"
                 :invars [[:x [:tensor [1 64] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 1.5 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          mlir (shlo/graph->mlir-text graph)
          exec (pjrt/compile-mlir api client mlir)]
      (is (some? exec)))))
