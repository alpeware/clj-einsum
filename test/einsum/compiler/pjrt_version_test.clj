(ns einsum.compiler.pjrt-version-test
  "Generative property tests for PJRT versioning, compatibility validation, and driver probing."
  (:require [einsum.compiler.pjrt.version :as v]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(def gen-version-pair
  (gen/tuple gen/small-integer gen/small-integer))

(defspec prop-compatibility-validation-invariant
  100
  (prop/for-all [[major minor] gen-version-pair
                 min-minor gen/small-integer]
                (let [res (v/validate-version-compatibility major minor min-minor)
                      expected-compat (and (= major 0) (>= minor min-minor))]
                  (and (boolean? (:compatible? res))
                       (= expected-compat (:compatible? res))
                       (= major (:major res))
                       (= minor (:minor res))
                       (string? (:reason res))))))

(defspec prop-plugin-attributes-parsing
  50
  (prop/for-all [attrs (gen/map (gen/not-empty gen/string-alphanumeric)
                                (gen/one-of [(gen/not-empty gen/string-alphanumeric) gen/small-integer]))]
                (let [parsed (v/parse-plugin-attributes attrs)]
                  (and (map? parsed)
                       (contains? parsed :raw-attributes)
                       (or (nil? (:driver-version parsed)) (string? (:driver-version parsed)))))))

(deftest test-system-driver-probing
  (testing "System driver probe returns structured non-nil diagnostic map"
    (let [probe (v/probe-system-driver)]
      (is (map? probe))
      (is (set? (:detected-backends probe)))
      (is (map? (:details probe)))
      (is (contains? (:detected-backends probe) :cpu) "CPU backend must always be detected as fallback")
      (is (contains? (:details probe) :sycl) "SYCL backend details map must exist")
      (is (boolean? (get-in probe [:details :sycl :detected?])))
      (when (contains? (:detected-backends probe) :rocm)
        (is (string? (get-in probe [:details :rocm :version])) "ROCm version string must be populated when detected"))
      (when (contains? (:detected-backends probe) :sycl)
        (is (true? (get-in probe [:details :sycl :detected?])) "SYCL detected? must be true when in detected-backends")
        (is (string? (get-in probe [:details :sycl :version])) "SYCL version string must be populated when detected")
        (is (or (str/includes? (get-in probe [:details :sycl :version]) "Driver")
                (str/includes? (get-in probe [:details :sycl :version]) "Level-Zero")) "SYCL version string must include driver or Level-Zero version info")))))

(deftest test-ensure-hsaco-cache-dir
  (testing "HSACO cache directory environment configuration"
    (let [cache-path (v/ensure-hsaco-cache-dir!)]
      (is (string? cache-path))
      (is (.exists (io/file cache-path))))))
