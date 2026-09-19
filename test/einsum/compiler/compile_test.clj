(ns einsum.compiler.compile-test
  "Unit tests for compilation caching."
  (:require [einsum.compiler.compile :as compile]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.test-helpers.parity :as p]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn- clean-cache-fixture [f]
  (let [tmp-dir (doto (io/file (System/getProperty "java.io.tmpdir") (str "clj_xla_test_cache_" (System/nanoTime)))
                  .mkdirs)]
    (try
      (compile/set-cache-dir! tmp-dir)
      (compile/clear-cache!)
      (f)
      (finally
        (compile/clear-cache!)
        (compile/set-cache-dir! nil)
        (doseq [file (.listFiles tmp-dir)] (.delete file))
        (.delete tmp-dir)))))

(use-fixtures :each clean-cache-fixture)

(deftest compilation-caching-test
  (testing "SHA-256 graph compilation caching"
    (compile/clear-cache!)
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "cached_graph"
                 :invars [[:x [:tensor [2 32] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 3.0 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          exec1 (compile/compile-graph api client graph)
          exec2 (compile/compile-graph api client graph)]
      (is (some? exec1))
      (is (= exec1 exec2) "Cached compilation must return identical executable handle")
      (compile/clear-cache!))))

(deftest pjrt-serialization-roundtrip-test
  (testing "Serialization and deserialization of PJRT executable preserves execution behavior"
    (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "ser_test_graph"
                 :invars [[:x [:tensor [4] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 2.5 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
          exec (compile/compile-graph api client graph)
          ser-bytes (pjrt/serialize-executable api (:handle exec))]
      (is (bytes? ser-bytes))
      (is (pos? (alength ^bytes ser-bytes)))
      (let [deser-handle (pjrt/deserialize-and-load api client ser-bytes)
            x-buf (pjrt/buffer-from-host-buffer api client (float-array [1.0 2.0 3.0 4.0]) [4] 11)
            out-orig (pjrt/execute-executable api (:handle exec) [x-buf] 1)
            out-deser (pjrt/execute-executable api deser-handle [x-buf] 1)
            res-orig (vec (pjrt/buffer-to-host-buffer api out-orig 4 :f32))
            res-deser (vec (pjrt/buffer-to-host-buffer api out-deser 4 :f32))]
        (is (= [2.5 5.0 7.5 10.0] res-orig))
        (is (= res-orig res-deser))))))

(defspec prop-serialization-roundtrip-invariant
  20
  (let [api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
        client (pjrt/create-client api)]
    (prop/for-all [c (gen/double* {:NaN? false :infinite? false :min -100.0 :max 100.0})]
                  (let [graph {:name "prop_ser_graph"
                               :invars [[:x [:tensor [2] :f32]]]
                               :outvars [:y]
                               :eqns [{:op :stablehlo/constant :value (float c) :outvars [:c0]}
                                      {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}
                        exec (compile/compile-graph api client graph)
                        ser-bytes (pjrt/serialize-executable api (:handle exec))
                        deser-handle (pjrt/deserialize-and-load api client ser-bytes)
                        x-buf (pjrt/buffer-from-host-buffer api client (float-array [2.0 4.0]) [2] 11)
                        out-orig (pjrt/execute-executable api (:handle exec) [x-buf] 1)
                        out-deser (pjrt/execute-executable api deser-handle [x-buf] 1)
                        res-orig (vec (pjrt/buffer-to-host-buffer api out-orig 2 :f32))
                        res-deser (vec (pjrt/buffer-to-host-buffer api out-deser 2 :f32))]
                    (p/approx-equal? res-orig res-deser {:atol 1e-4 :rtol 1e-4})))))

(deftest persistent-disk-caching-test
  (testing "Compiled executables are saved to and reloaded from disk cache across in-memory cache clears"
    (let [tmp-dir (doto (io/file (System/getProperty "java.io.tmpdir") (str "clj_xla_test_cache_" (System/currentTimeMillis)))
                    .mkdirs)
          _ (compile/set-cache-dir! tmp-dir)
          api (pjrt/load-plugin! "bin/libpjrt_cpu.so")
          client (pjrt/create-client api)
          graph {:name "disk_cached_test_graph"
                 :invars [[:x [:tensor [3] :f32]]]
                 :outvars [:y]
                 :eqns [{:op :stablehlo/constant :value 10.0 :outvars [:c0]}
                        {:op :stablehlo/multiply :invars [:x :c0] :outvars [:y]}]}]
      (try
        (compile/clear-cache!)
        (compile/clear-disk-cache!)
        ;; 1. First compile: should be fresh compile and write to disk
        (let [exec1 (compile/compile-graph api client graph)
              cached-files (filter #(.isFile ^java.io.File %) (.listFiles tmp-dir))]
          (is (false? (:from-disk-cache? exec1)))
          (is (= 1 (count cached-files)) "Exactly one binary cache file should exist on disk")

          ;; 2. Clear in-memory cache only (simulating fresh process startup)
          (compile/clear-cache!)

          ;; 3. Second compile: should hit disk cache and deserialize without recompiling
          (let [exec2 (compile/compile-graph api client graph)
                x-buf (pjrt/buffer-from-host-buffer api client (float-array [1.0 2.0 3.0]) [3] 11)
                out-buf (pjrt/execute-executable api (:handle exec2) [x-buf] 1)
                res (vec (pjrt/buffer-to-host-buffer api out-buf 3 :f32))]
            (is (true? (:from-disk-cache? exec2)))
            (is (= [10.0 20.0 30.0] res))))
        (finally
          (compile/set-cache-dir! nil)
          (doseq [f (.listFiles tmp-dir)] (.delete f))
          (.delete tmp-dir))))))

