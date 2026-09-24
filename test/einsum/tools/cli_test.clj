(ns einsum.tools.cli-test
  "Unit and generative tests for tools.cli argument parsing, model discovery, and input tensor preparation."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [tools.cli :as cli]))

(defspec prop-normalize-args-equality 50
  (prop/for-all [key-suffix (gen/elements ["foo" "bar" "max-tokens" "model"])
                 val-str (gen/elements ["123" "abc" "path/to/dir" "true"])]
                (let [flag (str "--" key-suffix)
                      combined (str flag "=" val-str)
                      normalized (vec (cli/normalize-args [combined]))]
                  (= [flag val-str] normalized))))

(defspec prop-parse-cli-args-numbers 50
  (prop/for-all [max-tokens (gen/choose 1 1024)
                 temp (gen/fmap #(+ 0.1 (Math/abs (double %))) gen/double)
                 top-k (gen/choose 1 100)]
                (let [args ["--max-new-tokens" (str max-tokens)
                            "--temperature" (str temp)
                            "--top-k" (str top-k)]
                      parsed (cli/parse-cli-args args)]
                  (and (= max-tokens (:max-new-tokens parsed))
                       (== temp (:temperature parsed))
                       (= top-k (:top-k parsed))))))

(defspec prop-prepare-input-tensor-length 50
  (prop/for-all [token-count (gen/choose 0 32)
                 max-len (gen/choose 32 128)]
                (let [tokens (vec (range token-count))
                      arr (cli/prepare-input-tensor tokens max-len)]
                  (and (instance? (Class/forName "[I") arr)
                       (= max-len (alength ^ints arr))
                       (= (take token-count (vec arr)) tokens)
                       (every? zero? (drop token-count (vec arr)))))))

(deftest test-parse-cli-args-comprehensive
  (testing "Parses standard string, file, number, keyword, and switch flags"
    (let [args ["--prompt" "Hello world"
                "--model" "turboderp/gemma-4-12B-it-exl3"
                "--max-new-tokens" "42"
                "--temp" "0.85"
                "--top-k" "50"
                "--backend" ":rocm"
                "--precision" ":bf16"
                "--verbose"
                "--chat"
                "--skip-layers" "1,2,3"
                "--custom-flag" "custom-val"]
          parsed (cli/parse-cli-args args)]
      (is (= "Hello world" (:prompt parsed)))
      (is (= ".models/gemma-4-12B-it-exl3" (:model-dir parsed)))
      (is (= ".models/gemma-4-12B-it-exl3" (:model parsed)))
      (is (= 42 (:max-new-tokens parsed)))
      (is (== 0.85 (:temperature parsed)))
      (is (= 50 (:top-k parsed)))
      (is (= :rocm (:backend parsed)))
      (is (= :bf16 (:precision parsed)))
      (is (true? (:verbose parsed)))
      (is (true? (:chat parsed)))
      (is (= #{1 2 3} (:skip-layers parsed)))
      (is (= "custom-val" (:custom-flag parsed)))))

  (testing "Supports custom handlers"
    (let [custom-handler (fn [opts remaining]
                           [(assoc opts :handled-flag (second remaining)) 2])
          parsed (cli/parse-cli-args ["--my-flag" "special" "--prompt" "test"]
                                     cli/DEFAULT_INFERENCE_OPTS
                                     {"--my-flag" custom-handler})]
      (is (= "special" (:handled-flag parsed)))
      (is (= "test" (:prompt parsed))))))

(deftest test-find-model-dir-resolution
  (testing "Resolves existing models by family keyword"
    (is (string? (cli/find-model-dir :gpt2)))
    (is (string? (cli/find-model-dir :smollm)))
    (is (string? (cli/find-model-dir :gemma-2)))
    (is (string? (cli/find-model-dir :gemma-3)))
    (is (string? (cli/find-model-dir :gemma-4))))

  (testing "Resolves explicit existing path"
    (is (= ".models/gpt2" (cli/find-model-dir ".models/gpt2" :gpt2))))

  (testing "Throws descriptive exception when candidate directories do not exist"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Model directory not found in candidates"
                          (cli/find-model-dir ["non-existent-dir-12345/foo"])))))
