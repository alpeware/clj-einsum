(ns einsum.tools.fetch-pjrt-test
  "Generative property tests for fetch_pjrt_binaries sources and target resolution."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [tools.fetch-pjrt :as fetch]))

(deftest test-sources-structure
  (testing "SOURCES map contains required targets and non-nil attributes"
    (is (contains? fetch/SOURCES :cpu))
    (is (contains? fetch/SOURCES :cuda12))
    (is (contains? fetch/SOURCES :sycl))
    (is (contains? fetch/SOURCES :rocm))
    (doseq [[target config] fetch/SOURCES]
      (is (string? (:url config)) (str "Target " target " missing string url"))
      (is (re-find #"^https?://" (:url config)) (str "Target " target " url must start with http(s)://"))
      (is (string? (:so-name config)) (str "Target " target " missing string so-name"))
      (is (instance? java.util.regex.Pattern (:entry-pattern config)) (str "Target " target " missing entry-pattern regex"))))
  (testing "SYCL target configuration"
    (let [sycl-config (get fetch/SOURCES :sycl)]
      (is (re-find #"intel_extension_for_openxla-0\.7\.0" (:url sycl-config)) "SYCL URL should point to 0.7.0 wheel")
      (is (vector? (:deps sycl-config)) "SYCL target should have deps vector")
      (is (>= (count (:deps sycl-config)) 8) "SYCL target should have at least 8 companion wheel dependencies")
      (doseq [dep-url (:deps sycl-config)]
        (is (string? dep-url))
        (is (re-find #"^https?://" dep-url))))))

(def gen-backend-target
  (gen/elements (conj (vec (keys fetch/SOURCES)) :all)))

(defspec prop-backend-target-resolution
  50
  (prop/for-all [target gen-backend-target]
                (if (= target :all)
                  (= (set (keys fetch/SOURCES)) #{:cpu :cuda12 :sycl :rocm})
                  (some? (get fetch/SOURCES target)))))

(defspec prop-dependencies-urls-invariant
  50
  (prop/for-all [target gen-backend-target]
                (if (= target :all)
                  true
                  (let [config (get fetch/SOURCES target)]
                    (if-let [deps (or (:deps config) (:dependencies config))]
                      (and (vector? deps)
                           (every? #(and (string? %) (re-find #"^https?://" %)) deps))
                      true)))))
