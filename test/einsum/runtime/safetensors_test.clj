(ns einsum.runtime.safetensors-test
  "Unit and generative tests for einsum.runtime.safetensors Panama off-heap weight loader."
  (:require [einsum.runtime.safetensors :as st]
            [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-header-json-parsing 50
  (prop/for-all [shape-len (gen/choose 1 4)
                 dim-val (gen/choose 1 128)]
                (let [header-map {"weight_1" {"dtype" "F32" "shape" (vec (repeat shape-len dim-val)) "data_offsets" [0 100]}}
                      parsed (st/parse-header-json (json/write-str header-map))]
                  (some? (get parsed "weight_1")))))

(deftest safetensors-header-parser-test
  (testing "Parsing safetensors header JSON metadata"
    (let [header-json "{\"weight_a\":{\"dtype\":\"F32\",\"shape\":[1,128,768],\"data_offsets\":[0,393216]}}"
          parsed (st/parse-header-json header-json)]
      (is (= "F32" (get-in parsed ["weight_a" "dtype"])))
      (is (= [1 128 768] (get-in parsed ["weight_a" "shape"])))
      (is (= [0 393216] (get-in parsed ["weight_a" "data_offsets"]))))))

(defspec prop-build-header-invariants 50
  (prop/for-all [dim1 (gen/choose 1 64)
                 dim2 (gen/choose 1 64)
                 dim3 (gen/choose 1 64)]
                (let [tensors [{:name "w0" :dtype "I8" :shape [dim1 dim2] :byte-len (* dim1 dim2)}
                               {:name "w1" :dtype "BF16" :shape [dim2 dim3] :byte-len (* dim2 dim3 2)}
                               {:name "w2" :dtype "F32" :shape [dim3] :byte-len (* dim3 4)}]
                      {:keys [header-map total-payload-bytes]} (st/build-header-map tensors {})
                      w0-offsets (get-in header-map ["w0" "data_offsets"])
                      w1-offsets (get-in header-map ["w1" "data_offsets"])
                      w2-offsets (get-in header-map ["w2" "data_offsets"])]
                  (and (= 0 (first w0-offsets))
                       (= (* dim1 dim2) (second w0-offsets))
                       (= (second w0-offsets) (first w1-offsets))
                       (= (+ (first w1-offsets) (* dim2 dim3 2)) (second w1-offsets))
                       (= (second w1-offsets) (first w2-offsets))
                       (= (+ (first w2-offsets) (* dim3 4)) (second w2-offsets))
                       (= (second w2-offsets) total-payload-bytes)))))

(deftest safetensors-write-and-read-roundtrip-test
  (testing "Writing and reading back safetensors with Panama FFM zero-copy"
    (let [temp-file (java.io.File/createTempFile "test_model" ".safetensors")
          temp-path (.getAbsolutePath temp-file)]
      (try
        (let [w-bytes (byte-array [10 20 30 40 50 60])
              scale-shorts (short-array [16256 16384])
              tensors [{:name "layer0.weight" :dtype "I8" :shape [2 3] :data w-bytes}
                       {:name "layer0.weight.scales" :dtype "BF16" :shape [2] :data scale-shorts}]
              _ (st/write-safetensors temp-path tensors {:metadata {"quantization" "int4"}})]
          (with-open [arena (java.lang.foreign.Arena/ofConfined)]
            (let [mapped (st/map-safetensors-weights temp-path arena)
                  header (:header mapped)]
              (is (contains? header "layer0.weight"))
              (is (contains? header "layer0.weight.scales"))
              (is (= "int4" (get-in header ["__metadata__" "quantization"])))
              (let [w-slice (st/get-tensor-slice mapped "layer0.weight")
                    out-bytes (byte-array 6)]
                (java.lang.foreign.MemorySegment/copy w-slice java.lang.foreign.ValueLayout/JAVA_BYTE 0 out-bytes 0 6)
                (is (= (vec w-bytes) (vec out-bytes))))
              (let [s-arr (st/get-tensor-bf16-shorts mapped "layer0.weight.scales")]
                (is (= (vec scale-shorts) (vec s-arr)))))))
        (finally
          (.delete temp-file))))))
