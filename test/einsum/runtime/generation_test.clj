(ns einsum.runtime.generation-test
  "Unit and generative tests for autoregressive text generation strategy."
  (:require [einsum.runtime.generation.autoregressive :as ar]
            [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.protocol :refer [encode]]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest autoregressive-generate-mock-test
  (testing "Autoregressive generation step loop with dummy step function"
    (let [tokenizer (bpe/load-bpe-tokenizer ".models/gpt2/vocab.json" ".models/gpt2/merges.txt")
          prompt-ids (encode tokenizer "Hello world")
          step-fn (fn [_context-ids]
                    (vec (concat (repeat 50255 0.0) [100.0])))
          gen-ids (ar/generate-tokens step-fn prompt-ids {:max-new-tokens 5 :eos-token-id 50256})
          out-text (einsum.runtime.tokenizer.protocol/decode tokenizer gen-ids)]
      (is (vector? gen-ids))
      (is (> (count gen-ids) (count prompt-ids)))
      (is (string? out-text)))))

(defspec prop-autoregressive-max-tokens 20
  (prop/for-all [max-tokens (gen/choose 1 5)]
                (let [prompt-ids [15496 995]
                      step-fn (fn [_ctx] [1.0 2.0 3.0 4.0])
                      gen-ids (ar/generate-tokens step-fn prompt-ids {:max-new-tokens max-tokens :eos-token-id 9999})]
                  (= (+ (count prompt-ids) max-tokens) (count gen-ids)))))

(defspec prop-autoregressive-cached-generation-invariants 20
  (prop/for-all [max-tokens (gen/choose 1 5)]
                (let [prompt-ids [15496 995]
                      model-fn (fn [_ids kv-caches _pos]
                                 (let [dummy-logits (vec (repeat 5000 0.0))]
                                   [dummy-logits (or kv-caches [(vec (repeat 2 nil))])]))
                      gen-ids (ar/generate-tokens-cached model-fn prompt-ids {:max-new-tokens max-tokens :eos-token-id 9999})]
                  (= (+ (count prompt-ids) max-tokens) (count gen-ids)))))

(defspec prop-common-prefix-len-reflexive 50
  (prop/for-all [tokens (gen/vector gen/nat 0 50)]
                (= (count tokens) (ar/common-prefix-len tokens tokens))))

(defspec prop-common-prefix-len-symmetric 50
  (prop/for-all [xs (gen/vector gen/nat 0 30)
                 ys (gen/vector gen/nat 0 30)]
                (= (ar/common-prefix-len xs ys) (ar/common-prefix-len ys xs))))

(defspec prop-common-prefix-len-prefix-preservation 50
  (prop/for-all [prefix (gen/vector gen/nat 0 20)
                 suffix-a (gen/vector gen/nat 0 20)
                 suffix-b (gen/vector gen/nat 0 20)]
                (let [xs (vec (concat prefix suffix-a))
                      ys (vec (concat prefix suffix-b))]
                  (>= (ar/common-prefix-len xs ys) (count prefix)))))

