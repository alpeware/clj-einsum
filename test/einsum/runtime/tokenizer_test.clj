(ns einsum.runtime.tokenizer-test
  "Unit and generative tests for BPE and HuggingFace Tokenizers."
  (:require [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.hf-json :as hf]
            [einsum.runtime.tokenizer.protocol :refer [decode encode]]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(deftest gpt2-vocab-bpe-test
  (testing "Loading GPT-2 vocab.json and merges.txt"
    (let [tokenizer (bpe/load-bpe-tokenizer ".models/gpt2/vocab.json" ".models/gpt2/merges.txt")
          tokens (encode tokenizer "Hello world")
          text (decode tokenizer tokens)]
      (is (vector? tokens))
      (is (seq tokens))
      (is (= "Hello world" text)))))

(deftest hf-json-tokenizer-test
  (testing "Loading HuggingFace tokenizer.json for GPT-2"
    (let [tokenizer (hf/load-hf-json-tokenizer ".models/gpt2/tokenizer.json")
          tokens (encode tokenizer "Hello world")
          text (decode tokenizer tokens)]
      (is (vector? tokens))
      (is (seq tokens))
      (is (= "Hello world" text)))))

(defspec prop-tokenizer-encode-decode 20
  (prop/for-all [words (gen/vector (gen/elements ["Hello" "world" "Clojure" "XLA" "compiler" "AI"]) 1 5)]
                (let [input-text (str/join " " words)
                      tokenizer (bpe/load-bpe-tokenizer ".models/gpt2/vocab.json" ".models/gpt2/merges.txt")
                      ids (encode tokenizer input-text)
                      decoded (decode tokenizer ids)]
                  (and (vector? ids)
                       (= input-text decoded)))))
