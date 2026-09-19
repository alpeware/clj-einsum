(ns einsum.core-test
  "Unit and generative property tests for dynamic multi-backend PJRT plugin initialization in einsum.core."
  (:require [einsum.core :as xla]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

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


