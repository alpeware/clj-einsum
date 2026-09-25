(ns einsum.core-test
  "Unit and generative property tests for dynamic multi-backend PJRT plugin initialization in einsum.core."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as xla]
            [einsum.logic.interpret :as interpret]
            [einsum.logic.lower :as lower])
  (:import [java.lang.foreign Arena ValueLayout]))

(deftest test-backend-library-map-structure
  (testing "BACKEND-LIBRARY-MAP contains all supported hardware targets"
    (is (contains? xla/BACKEND-LIBRARY-MAP :cpu))
    (is (contains? xla/BACKEND-LIBRARY-MAP :sycl))
    (is (contains? xla/BACKEND-LIBRARY-MAP :rocm))
    (is (contains? xla/BACKEND-LIBRARY-MAP :cuda12))
    (doseq [[target config] xla/BACKEND-LIBRARY-MAP]
      (is (string? (:default config)) (str "Target " target " missing string default"))
      (is (re-find #"^bin/libpjrt_" (:default config)) (str "Target " target " default must start with bin/libpjrt_"))
      (is (string? (:env config)) (str "Target " target " missing string env"))
      (is (re-find #"^PJRT_.*_PATH$" (:env config)) (str "Target " target " env must match PJRT_*_PATH")))))

(def gen-target-keyword
  (gen/elements [:cpu :sycl :rocm :cuda12]))

(defspec prop-backend-resolution-invariants
  50
  (prop/for-all [target gen-target-keyword]
                (let [config (get xla/BACKEND-LIBRARY-MAP target)]
                  (and (some? config)
                       (string? (:default config))
                       (string? (:env config))))))

(defspec prop-flag-generation-invariants
  50
  (prop/for-all [target gen-target-keyword
                 autotune-level (gen/choose 1 4)]
                (let [res (xla/determine-optimal-xla-flags target {} {:autotune-level autotune-level})
                      {:keys [xla-flags env-vars cache-dir]} res]
                  (and (map? res)
                       (string? xla-flags)
                       (map? env-vars)
                       (or (nil? cache-dir) (string? cache-dir))
                       (case target
                         :rocm (re-find #"xla_gpu_autotune_level=" xla-flags)
                         :cuda12 (re-find #"xla_gpu_enable_cublaslt=true" xla-flags)
                         :sycl (re-find #"xla_gpu_enable_highest_priority_async_stream=true" xla-flags)
                         :cpu (re-find #"xla_cpu_multi_thread_eigen=true" xla-flags))))))

(deftest test-determine-optimal-xla-flags-overrides
  (testing "User explicit options override auto-detected XLA flags"
    (let [res (xla/determine-optimal-xla-flags :rocm {} {:xla-flags "--custom_flag=1" :autotune-level 2})
          {:keys [xla-flags autotune-level]} res]
      (is (= 2 autotune-level))
      (is (str/includes? xla-flags "--custom_flag=1"))
      (is (str/includes? xla-flags "--xla_gpu_autotune_level=2"))))

  (testing "Disabling auto defaults preserves user flags only"
    (let [res (xla/determine-optimal-xla-flags :rocm {} {:disable-defaults? true :xla-flags "--custom_only"})
          {:keys [xla-flags]} res]
      (is (= "--custom_only" xla-flags)))))

(deftest test-determine-optimal-xla-flags-rocm-env-vars
  (testing "determine-optimal-xla-flags sets ROCm environment variables including HSA_OVERRIDE_GFX_VERSION"
    (let [res (xla/determine-optimal-xla-flags :rocm {} {:gfx-version "11.0.0" :rocr-visible-devices "0" :hip-visible-devices "0"})
          {:keys [env-vars]} res]
      (is (= "11.0.0" (get env-vars "HSA_OVERRIDE_GFX_VERSION")))
      (is (= "0" (get env-vars "ROCR_VISIBLE_DEVICES")))
      (is (= "0" (get env-vars "HIP_VISIBLE_DEVICES")))
      (is (= "11.0.0" (System/getProperty "HSA_OVERRIDE_GFX_VERSION")))
      (is (= "0" (System/getProperty "ROCR_VISIBLE_DEVICES")))
      (is (= "0" (System/getProperty "HIP_VISIBLE_DEVICES"))))))

(deftest test-init-interpreter-backend
  (testing "init-backend! with :interpreter or :interp initializes pure-JVM backend without PJRT plugin"
    (let [ctx (xla/init-backend! :interpreter {:quiet? true})]
      (is (= :interpreter (:backend ctx)))
      (is (= :interpreter (:target ctx)))
      (is (= "interpreter" (:platform ctx)))
      (is (nil? (:client ctx)))
      (is (fn? (:destroy-fn ctx)))
      ;; Calling destroy-fn is a safe no-op
      (is (nil? ((:destroy-fn ctx) ctx nil)))
      (is (= ctx (xla/get-context))))
    (let [ctx2 (xla/init-backend! :interp {:quiet? true})]
      (is (= :interpreter (:backend ctx2))))
    (let [ctx3 (xla/init-interpreter! {:quiet? true})]
      (is (= :interpreter (:backend ctx3))))))

(deftest test-interpreter-compile-graph-and-execute
  (testing "compile-graph and execute on interpreter backend runs pure-JVM StableHLO"
    (let [ctx (xla/init-backend! :interpreter {:quiet? true})
          graph (lower/ast->graph "add_test"
                                  [[:x [:tensor [2] :f32]]
                                   [:y [:tensor [2] :f32]]]
                                  [:= [:z :d] [:x :d] [:y :d]]
                                  [:z])
          exec (xla/compile-graph ctx graph)]
      (is (map? exec))
      (is (= :interpreter (:backend exec)))
      (is (some? (:graph exec)))
      (let [x-arr (float-array [1.0 2.0])
            y-arr (float-array [3.0 4.0])
            res (xla/execute exec [x-arr y-arr])]
        (is (map? res))
        (is (= [3.0 8.0] (vec (:data res))))
        (let [floats (xla/to-host-slice res 0 2 2 :f32)]
          (is (= [3.0 8.0] (vec floats))))))))

(deftest test-interpreter-to-host-slice
  (testing "to-host-slice handles interpreter tensor maps, primitive arrays, and memory segments"
    ;; Direct float array
    (let [f-arr (float-array [1.0 2.0 3.0 4.0])
          sl0 (xla/to-host-slice f-arr 0 2 4 :f32)
          sl1 (xla/to-host-slice f-arr 1 2 4 :f32)]
      (is (= [1.0 2.0] (vec sl0)))
      (is (= [3.0 4.0] (vec sl1))))

    ;; Tensor map with float array
    (let [t-map {:dtype :f32 :shape [4] :data (float-array [10.0 20.0 30.0 40.0])}
          sl (xla/to-host-slice t-map 1 2 4 :f32)]
      (is (= [30.0 40.0] (vec sl))))

    ;; Tensor map with bf16 shorts
    (let [bf-shorts (short-array [(interpret/f32->bf16 1.5)
                                  (interpret/f32->bf16 2.5)
                                  (interpret/f32->bf16 3.5)])
          t-map {:dtype :bf16 :shape [3] :data bf-shorts}
          sl (xla/to-host-slice t-map 0 2 3 :bf16)]
      (is (= [1.5 2.5] (vec sl))))

    ;; Tensor map with MemorySegment
    (with-open [arena (Arena/ofConfined)]
      (let [seg (.allocate arena (* 4 4) 4)]
        (dotimes [i 4]
          (.setAtIndex seg ValueLayout/JAVA_FLOAT (long i) (float (* 10 (inc i)))))
        (let [t-map {:dtype :f32 :shape [4] :data seg}
              sl (xla/to-host-slice t-map 1 2 4 :f32)]
          (is (= [30.0 40.0] (vec sl))))))))



