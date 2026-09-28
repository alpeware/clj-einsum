(ns einsum.runtime.tokenizer-test
  "Unit and generative tests for BPE and HuggingFace Tokenizers."
  (:require [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.hf-json :as hf]
            [einsum.runtime.tokenizer.protocol :refer [decode encode]]
            [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defn- create-temp-bpe-fixture! []
  (let [v-file (doto (java.io.File/createTempFile "vocab" ".json") .deleteOnExit)
        m-file (doto (java.io.File/createTempFile "merges" ".txt") .deleteOnExit)
        vocab (into {"<|endoftext|>" 0}
                    (map (fn [[b u]] [u (inc (long b))]) bpe/bytes->unicode))]
    (spit v-file (json/write-str vocab))
    (spit m-file "#version: 0.2\n")
    [(.getAbsolutePath v-file) (.getAbsolutePath m-file)]))

(defn- create-temp-hf-json-fixture! []
  (let [hf-file (doto (java.io.File/createTempFile "tokenizer" ".json") .deleteOnExit)
        vocab (into {"<|endoftext|>" 0}
                    (map (fn [[b u]] [u (inc (long b))]) bpe/bytes->unicode))
        hf-data {"model" {"vocab" vocab "merges" []}}]
    (spit hf-file (json/write-str hf-data))
    (.getAbsolutePath hf-file)))

(def ^:private test-bpe-tokenizer
  (delay
    (let [[v-path m-path] (create-temp-bpe-fixture!)]
      (bpe/load-bpe-tokenizer v-path m-path))))

(deftest bpe-tokenizer-test
  (testing "Loading synthetic BPE vocab and merges (standalone, no models needed)"
    (let [tokenizer @test-bpe-tokenizer
          tokens (encode tokenizer "Hello world")
          text (decode tokenizer tokens)]
      (is (vector? tokens))
      (is (seq tokens))
      (is (= "Hello world" text))))
  (when (.exists (io/file ".models/gpt2/vocab.json"))
    (testing "Loading real GPT-2 vocab.json and merges.txt when present on disk"
      (let [tokenizer (bpe/load-bpe-tokenizer ".models/gpt2/vocab.json" ".models/gpt2/merges.txt")
            tokens (encode tokenizer "Hello world")
            text (decode tokenizer tokens)]
        (is (vector? tokens))
        (is (seq tokens))
        (is (= "Hello world" text))))))

(defn- create-temp-hf-json-with-specials-fixture! []
  (let [hf-file (doto (java.io.File/createTempFile "tokenizer-specials" ".json") .deleteOnExit)
        vocab (into {"<|endoftext|>" 0
                     "<|tool_call>" 48
                     "<tool_call|>" 49
                     "<|tool_response>" 50
                     "<tool_response|>" 51
                     "<|\"|>" 52
                     ":<" 61916
                     "}<" 22762}
                    (map (fn [[b u]] [u (inc (long b))]) bpe/bytes->unicode))
        added-tokens [{"id" 48 "content" "<|tool_call>" "special" true}
                      {"id" 49 "content" "<tool_call|>" "special" true}
                      {"id" 50 "content" "<|tool_response>" "special" true}
                      {"id" 51 "content" "<tool_response|>" "special" true}
                      {"id" 52 "content" "<|\"|>" "special" true}]
        hf-data {"model" {"vocab" vocab "merges" []}
                 "added_tokens" added-tokens}]
    (spit hf-file (json/write-str hf-data))
    (.getAbsolutePath hf-file)))

(deftest hf-json-tokenizer-test
  (testing "Loading synthetic HuggingFace tokenizer.json (standalone, no models needed)"
    (let [hf-path (create-temp-hf-json-fixture!)
          tokenizer (hf/load-hf-json-tokenizer hf-path)
          tokens (encode tokenizer "Hello world")
          text (decode tokenizer tokens)]
      (is (vector? tokens))
      (is (seq tokens))
      (is (= "Hello world" text))))
  (testing "Special tokens are preserved atomically and not merged with adjacent :, }, >"
    (let [hf-path (create-temp-hf-json-with-specials-fixture!)
          tokenizer (hf/load-hf-json-tokenizer hf-path)
          tokens-call (encode tokenizer "code:<|\"|>(+ 1 2)<|\"|>}<tool_call|>")
          tokens-resp (encode tokenizer "response:{output:<|\"|>3<|\"|>}<tool_response|>")]
      (is (some #{49} tokens-call) "Closing tool_call tag must be preserved as token 49")
      (is (some #{51} tokens-resp) "Closing tool_response tag must be preserved as token 51")
      (is (some #{52} tokens-call) "Quote delimiter <|\"|> must be preserved as token 52")
      (is (= 51 (last tokens-resp)) "tool_response|> must be the exact final token ID 51")))
  (when (.exists (io/file ".models/gpt2/tokenizer.json"))
    (testing "Loading real HuggingFace tokenizer.json for GPT-2 when present on disk"
      (let [tokenizer (hf/load-hf-json-tokenizer ".models/gpt2/tokenizer.json")
            tokens (encode tokenizer "Hello world")
            text (decode tokenizer tokens)]
        (is (vector? tokens))
        (is (seq tokens))
        (is (= "Hello world" text))))))

(def ^:private test-specials-tokenizer
  (delay
    (let [hf-path (create-temp-hf-json-with-specials-fixture!)]
      (hf/load-hf-json-tokenizer hf-path))))

(defspec prop-tokenizer-encode-decode 20
  (prop/for-all [words (gen/vector (gen/elements ["Hello" "world" "Clojure" "XLA" "compiler" "AI"]) 1 5)]
                (let [input-text (str/join " " words)
                      tokenizer @test-bpe-tokenizer
                      ids (encode tokenizer input-text)
                      decoded (decode tokenizer ids)]
                  (and (vector? ids)
                       (= input-text decoded)))))

(defspec prop-special-tokens-preserved 30
  (prop/for-all [prefix (gen/elements ["" "}" ":" "\"" ">" "<" "\n" "abc"])
                 suffix (gen/elements ["" "{" ":" "<" ">" "\n" "xyz"])
                 [tok-str expected-id] (gen/elements [["<|tool_call>" 48]
                                                      ["<tool_call|>" 49]
                                                      ["<|tool_response>" 50]
                                                      ["<tool_response|>" 51]
                                                      ["<|\"|>" 52]])]
                (let [tokenizer @test-specials-tokenizer
                      input-str (str prefix tok-str suffix)
                      ids (encode tokenizer input-str)]
                  (and (vector? ids)
                       (some #{expected-id} ids)))))
