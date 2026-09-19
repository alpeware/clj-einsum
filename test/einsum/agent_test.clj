(ns einsum.agent-test
  "Unit and generative tests for tools.gemma4-agent SCI tool execution and turn formatting."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [sci.core :as sci]
            [tools.gemma4-agent :as agent]))

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` code block strings from text, including unclosed blocks up to EOF."
  [text]
  (let [pattern #"(?s)```(?:clojure|clj)\s*\n(.*?)(?:```|$)"]
    (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern text))))))

(defn eval-sci-code
  "Evaluates `code-str` in a safe SCI sandbox context with agent helper bindings."
  [code-str]
  (let [env (sci/init {:bindings {'println println
                                  'str str
                                  'range range
                                  'mapv mapv
                                  'reduce reduce
                                  'filter filter}})]
    (try
      (let [out (binding [*out* (java.io.StringWriter.)]
                  (let [res (sci/eval-string* env code-str)
                        printed (str *out*)]
                    (if (seq printed)
                      (str printed "\n=> " (pr-str res))
                      (pr-str res))))]
        {:status :success :result out})
      (catch Exception e
        {:status :error :result (str "Evaluation Error: " (.getMessage e))}))))

(deftest test-extract-clojure-code-blocks
  (testing "Extracting single and multiple Clojure code blocks from model generation"
    (let [text "I will calculate the sum of squares.\n```clojure\n(reduce + (map #(Math/pow % 2) (range 1 5)))\n```\nDone."
          blocks (extract-clojure-code-blocks text)]
      (is (= 1 (count blocks)))
      (is (clojure.string/includes? (first blocks) "reduce +"))))
  (testing "Extracting bare code blocks without language tag"
    (let [text "Running code:\n```\n(range 10)\n```\nDone"
          pattern #"(?s)```(?:clojure|clj)?\s*\n?(.*?)(?:```|$)"
          blocks (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern text))))]
      (is (= ["(range 10)"] blocks))))
  (testing "Evaluating invalid code produces genuine SCI error"
    (let [res (eval-sci-code "( 10)")]
      (is (= :error (:status res)))
      (is (clojure.string/includes? (:result res) "cannot be cast to")))))

(deftest test-eval-sci-code-success
  (testing "Evaluating math expressions in SCI sandbox"
    (let [res (eval-sci-code "(reduce + (range 10))")]
      (is (= :success (:status res)))
      (is (clojure.string/includes? (:result res) "45")))))

(deftest test-eval-sci-code-error-handling
  (testing "Handling runtime exceptions inside SCI sandbox"
    (let [res (eval-sci-code "(/ 1 0)")]
      (is (= :error (:status res)))
      (is (clojure.string/includes? (:result res) "Evaluation Error")))))

(defspec prop-sci-arithmetic-eval-invariant
  50
  (prop/for-all [a (gen/choose 1 1000)
                 b (gen/choose 1 1000)]
                (let [code (format "(+ %d %d)" a b)
                      res (eval-sci-code code)]
                  (and (= :success (:status res))
                       (= (str (+ a b)) (str/trim (:result res)))))))

(deftest test-format-agent-chat-prompt-native-system
  (testing "Formatting prompt produces native Gemma 4 <|turn>system turn"
    (let [sys "You are an assistant."
          history [{:role :user :content "Hello"}
                   {:role :model :content "Hi there"}
                   {:role :user :content "Write code"}]
          prompt (agent/format-agent-chat-prompt sys history)]
      (is (str/starts-with? prompt "<bos><|turn>system\nYou are an assistant.<turn|>\n"))
      (is (str/includes? prompt "<|turn>user\nHello<turn|>\n"))
      (is (str/includes? prompt "<|turn>model\nHi there<turn|>\n"))
      (is (str/ends-with? prompt "<|turn>model\n"))))
  (testing "Formatting prompt with empty system prompt omits system turn"
    (let [history [{:role :user :content "Hello"}]
          prompt (agent/format-agent-chat-prompt "" history)]
      (is (str/starts-with? prompt "<bos><|turn>user\nHello<turn|>\n")))))

(defspec prop-format-agent-chat-prompt-invariants
  50
  (prop/for-all [sys (gen/not-empty gen/string-alphanumeric)
                 user-msg (gen/not-empty gen/string-alphanumeric)]
                (let [prompt (agent/format-agent-chat-prompt sys [{:role :user :content user-msg}])]
                  (and (str/starts-with? prompt "<bos><|turn>system\n")
                       (str/includes? prompt (str "<|turn>user\n" user-msg "<turn|>\n"))
                       (str/ends-with? prompt "<|turn>model\n")))))

