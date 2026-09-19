(ns einsum.quantize-test
  "Unit and generative tests for offline model quantization pipeline."
  (:require [einsum.runtime.safetensors :as st]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [tools.quantize :as quant])
  (:import [java.lang.foreign Arena]))

(defn- sample-float-matrix [rows cols max-val]
  (let [total (* rows cols)
        arr (float-array total)]
    (dotimes [i total]
      (let [val (* max-val (- (* 2.0 (rand)) 1.0))]
        (aset arr i (float val))))
    arr))

(defspec prop-quantize-weight-int4-invariants 50
  (prop/for-all [rows (gen/choose 2 32)
                 half-cols (gen/choose 2 32)]
                (let [cols (* half-cols 2)
                      w-arr (sample-float-matrix rows cols 1.5)
                      {:keys [data scales]} (quant/quantize-projection-weight w-arr rows cols :int4)
                      packed-len (* rows half-cols)]
                  (and (= packed-len (alength ^bytes data))
                       (= rows (alength ^shorts scales))))))

(defspec prop-quantize-weight-block-int4-invariants 50
  (prop/for-all [rows (gen/choose 2 16)
                 group-size (gen/elements [16 32])
                 num-groups (gen/choose 1 4)]
                (let [cols (* group-size num-groups)
                      w-arr (sample-float-matrix rows cols 2.0)
                      {:keys [data scales shape scale-shape]} (quant/quantize-projection-weight w-arr rows cols :int4 {:group-size group-size})
                      packed-len (* rows (quot cols 2))]
                  (and (= packed-len (alength ^bytes data))
                       (= (* rows num-groups) (alength ^shorts scales))
                       (= [rows (quot cols 2)] shape)
                       (= [rows num-groups] scale-shape)))))

(defspec prop-quantize-weight-int8-invariants 50
  (prop/for-all [rows (gen/choose 2 32)
                 cols (gen/choose 2 32)]
                (let [w-arr (sample-float-matrix rows cols 2.0)
                      {:keys [data scales]} (quant/quantize-projection-weight w-arr rows cols :int8)
                      total (* rows cols)]
                  (and (= total (alength ^bytes data))
                       (= rows (alength ^shorts scales))))))

(deftest quantizable-weight-filter-test
  (testing "Correctly categorizes weights eligible for INT4/INT8 quantization"
    (is (true? (quant/quantizable-weight? "model.layers.0.mlp.gate_proj.weight" [4096 1024])))
    (is (true? (quant/quantizable-weight? "model.layers.15.self_attn.o_proj.weight" [2048 2048])))
    ;; Embeddings must remain in BF16
    (is (false? (quant/quantizable-weight? "model.embed_tokens.weight" [262144 5120])))
    ;; RMSNorms must remain in BF16
    (is (false? (quant/quantizable-weight? "model.norm.weight" [5120])))
    (is (false? (quant/quantizable-weight? "model.layers.0.input_layernorm.weight" [5120])))
    (is (false? (quant/quantizable-weight? "model.layers.0.post_attention_layernorm.weight" [5120])))))

(deftest end-to-end-synthetic-quantization-test
  (testing "Quantizes a small synthetic model safetensors into native INT4 safetensors"
    (let [tmp-dir (io/file (System/getProperty "java.io.tmpdir") (str "clj_xla_quant_test_" (System/currentTimeMillis)))
          out-dir (io/file tmp-dir "quantized")
          _ (.mkdirs tmp-dir)
          source-file (io/file tmp-dir "model.safetensors")
          config-file (io/file tmp-dir "config.json")]
      (try
        ;; 1. Create a dummy synthetic model
        (spit config-file (json/write-str {"model_type" "gemma4", "hidden_size" 64, "num_hidden_layers" 1}))
        (let [embed-shorts (short-array (* 32 64))
              norm-shorts (short-array 64)
              gate-shorts (short-array (* 128 64))
              tensors [{:name "model.embed_tokens.weight" :dtype "BF16" :shape [32 64] :data embed-shorts}
                       {:name "model.norm.weight" :dtype "BF16" :shape [64] :data norm-shorts}
                       {:name "model.layers.0.mlp.gate_proj.weight" :dtype "BF16" :shape [128 64] :data gate-shorts}]]
          (st/write-safetensors (.getAbsolutePath source-file) tensors))

        ;; 2. Run quantization pipeline
        (let [res (quant/quantize-model! {:model-path (.getAbsolutePath tmp-dir)
                                          :output-path (.getAbsolutePath out-dir)
                                          :precision :int4
                                          :quiet true})]
          (is (= :int4 (:precision res)))
          (is (.exists (io/file out-dir "model.safetensors")))
          (is (.exists (io/file out-dir "config.json")))
          (is (.exists (io/file out-dir "quant_config.edn")))

          ;; 3. Verify output safetensors header and contents
          (with-open [arena (Arena/ofConfined)]
            (let [mapped (st/map-safetensors-weights (.getAbsolutePath out-dir) arena)
                  header (:header mapped)]
              ;; embed_tokens kept as BF16
              (is (= "BF16" (get-in header ["model.embed_tokens.weight" "dtype"])))
              (is (= [32 64] (get-in header ["model.embed_tokens.weight" "shape"])))
              ;; norm kept as BF16
              (is (= "BF16" (get-in header ["model.norm.weight" "dtype"])))
              ;; gate_proj quantized to INT4 (packed I8 [128 32])
              (is (= "I8" (get-in header ["model.layers.0.mlp.gate_proj.weight" "dtype"])))
              (is (= [128 32] (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"])))
              ;; gate_proj.scales present in BF16 [128]
              (is (= "BF16" (get-in header ["model.layers.0.mlp.gate_proj.weight.scales" "dtype"])))
              (is (= [128] (get-in header ["model.layers.0.mlp.gate_proj.weight.scales" "shape"]))))))
        (finally
          ;; Clean up temp dir
          (doseq [f (reverse (file-seq tmp-dir))]
            (.delete f))))))

  (testing "Quantizes a synthetic model into block-wise INT4 safetensors with 2D scales"
    (let [tmp-dir (io/file (System/getProperty "java.io.tmpdir") (str "clj_xla_quant_block_test_" (System/currentTimeMillis)))
          out-dir (io/file tmp-dir "quantized")
          _ (.mkdirs tmp-dir)
          source-file (io/file tmp-dir "model.safetensors")
          config-file (io/file tmp-dir "config.json")]
      (try
        (spit config-file (json/write-str {"model_type" "gemma4", "hidden_size" 256, "num_hidden_layers" 1}))
        (let [gate-shorts (short-array (* 64 256))
              tensors [{:name "model.layers.0.mlp.gate_proj.weight" :dtype "BF16" :shape [64 256] :data gate-shorts}]]
          (st/write-safetensors (.getAbsolutePath source-file) tensors))

        (let [res (quant/quantize-model! {:model-path (.getAbsolutePath tmp-dir)
                                          :output-path (.getAbsolutePath out-dir)
                                          :precision :int4
                                          :group-size 128
                                          :quiet true})]
          (is (= :int4 (:precision res)))
          (with-open [arena (Arena/ofConfined)]
            (let [mapped (st/map-safetensors-weights (.getAbsolutePath out-dir) arena)
                  header (:header mapped)]
              (is (= "128" (get-in header ["__metadata__" "group_size"])))
              (is (= "I8" (get-in header ["model.layers.0.mlp.gate_proj.weight" "dtype"])))
              (is (= [64 128] (get-in header ["model.layers.0.mlp.gate_proj.weight" "shape"])))
              ;; cols=256, group-size=128 => num-groups=2 => scale shape [64 2]
              (is (= "BF16" (get-in header ["model.layers.0.mlp.gate_proj.weight.scales" "dtype"])))
              (is (= [64 2] (get-in header ["model.layers.0.mlp.gate_proj.weight.scales" "shape"]))))))
        (finally
          (doseq [f (reverse (file-seq tmp-dir))]
            (.delete f)))))))
