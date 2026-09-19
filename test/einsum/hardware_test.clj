(ns einsum.hardware-test
  "Unit and generative tests for hardware capability detection and quantization strategy selection."
  (:require [einsum.hardware :as hw]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-select-quant-strategy-valid-result 100
  (prop/for-all [vram-gb (gen/choose 4 128)
                 param-count-billions (gen/choose 1 100)]
                (let [vram-bytes (* (long vram-gb) 1024 1024 1024)
                      param-count (* (long param-count-billions) 1000000000)
                      strategy (hw/select-quant-strategy {:vram-bytes vram-bytes} {:param-count param-count})]
                  (and (keyword? strategy)
                       (contains? #{:bf16 :int8 :int4} strategy)))))

(defspec prop-quant-strategy-monotonicity 50
  (prop/for-all [vram-gb (gen/choose 16 64)
                 small-params (gen/choose 1 10)
                 large-params (gen/choose 30 100)]
                (let [vram-bytes (* (long vram-gb) 1024 1024 1024)
                      strat-small (hw/select-quant-strategy {:vram-bytes vram-bytes} {:param-count (* (long small-params) 1000000000)})
                      strat-large (hw/select-quant-strategy {:vram-bytes vram-bytes} {:param-count (* (long large-params) 1000000000)})
                      precision-order {:bf16 3 :int8 2 :int4 1}]
                  (>= (get precision-order strat-small)
                      (get precision-order strat-large)))))

(deftest hardware-quant-strategy-unit-tests
  (testing "Gemma 4 models on 24 GB VRAM GPU (AMD Radeon RX 7900 XTX)"
    (let [gpu-24gb {:vram-bytes (* 24 1024 1024 1024) :arch :rdna3}]
      ;; Gemma 4 31B (~62 GB BF16 -> INT8 is ~32 GB -> must use INT4 ~17.5 GB to leave >6 GB KV cache headroom)
      (is (= :int4 (hw/select-quant-strategy gpu-24gb {:param-count 31000000000})))
      ;; Gemma 4 12B (~24 GB BF16 -> INT8 is ~13 GB -> comfortably fits in 24 GB)
      (is (= :int8 (hw/select-quant-strategy gpu-24gb {:param-count 12000000000})))
      ;; Gemma 4 E2B (~4 GB BF16 -> comfortably fits in 24 GB unquantized)
      (is (= :bf16 (hw/select-quant-strategy gpu-24gb {:param-count 2000000000})))))

  (testing "Gemma 4 31B on 80 GB VRAM GPU (e.g. NVIDIA A100/H100 / AMD MI300)"
    (let [gpu-80gb {:vram-bytes (* 80 1024 1024 1024)}]
      (is (= :bf16 (hw/select-quant-strategy gpu-80gb {:param-count 31000000000})))))

  (testing "Gemma 4 31B on 48 GB VRAM GPU (e.g. RTX 6000 Ada / A6000)"
    (let [gpu-48gb {:vram-bytes (* 48 1024 1024 1024)}]
      (is (= :int8 (hw/select-quant-strategy gpu-48gb {:param-count 31000000000}))))))

(deftest detect-hardware-test
  (testing "detect-hardware returns valid structured profile"
    (let [hw-info (hw/detect-hardware)]
      (is (map? hw-info))
      (is (contains? #{:rocm :cuda12 :sycl :cpu} (:backend hw-info)))
      (is (some? (:primary-device hw-info)))
      (let [dev (:primary-device hw-info)]
        (is (number? (:vram-bytes dev)))
        (is (pos? (:vram-bytes dev)))))))
