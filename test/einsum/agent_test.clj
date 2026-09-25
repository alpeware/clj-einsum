(ns einsum.agent-test
  "Unit and generative tests for tools.gemma4-agent SCI tool execution, thinking trace extraction, and turn formatting."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [tools.gemma4-agent :as agent]))

(deftest test-extract-clojure-code-blocks
  (testing "Extracting single and multiple Clojure code blocks from model generation"
    (let [text "I will calculate the sum of squares.\n```clojure\n(reduce + (map #(Math/pow % 2) (range 1 5)))\n```\nDone."
          blocks (agent/extract-clojure-code-blocks text)]
      (is (= 1 (count blocks)))
      (is (str/includes? (first blocks) "reduce +"))))
  (testing "Extracting bare code blocks without language tag"
    (let [text "Running code:\n```\n(range 10)\n```\nDone"
          blocks (agent/extract-clojure-code-blocks text)]
      (is (= ["(range 10)"] blocks))))
  (testing "Evaluating invalid code produces genuine SCI error"
    (let [sci-ctx (agent/create-agent-sci-ctx)
          res (agent/eval-tool-code sci-ctx "( 10)")]
      (is (= :error (:status res)))
      (is (str/includes? (:output res) "cannot be cast to")))))

(deftest test-extract-tool-call
  (testing "Extracting native Gemma 4 tool call with <|\"|> string delimiter"
    (let [text "<|tool_call>call:eval_clojure{code:<|\"|>(+ 10 20)<|\"|>}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(+ 10 20)" (:code tc)))))

  (testing "Extracting tool call with standard quotes"
    (let [text "<|tool_call>call:eval_clojure{code:\"(range 10)\"}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(range 10)" (:code tc)))))

  (testing "Extracting tool call with bare s-expression"
    (let [text "<|tool_call>call:eval_clojure{(println \"hello\")}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(println \"hello\")" (:code tc)))))

  (testing "Extracting tool call when stopped at EOF without closing token"
    (let [text "<|tool_call>call:eval_clojure{code:<|\"|>(mapv inc [1 2 3])<|\"|>}"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(mapv inc [1 2 3])" (:code tc)))))

  (testing "Extracting tool call with thinking trace present"
    (let [text "<|channel>thought\nWe should compute 2+2.\n<channel|>\n<|tool_call>call:eval_clojure{code:<|\"|>(+ 2 2)<|\"|>}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(+ 2 2)" (:code tc)))))

  (testing "Returns nil when response is plain text without a tool call"
    (let [text "The sum of 10 and 20 is 30. You can also write (+ 10 20)."]
      (is (nil? (agent/extract-tool-call text)))))

  (testing "Returns nil when response has markdown code block but no tool call tag"
    (let [text "Here is the solution:\n```clojure\n(defn square [x] (* x x))\n```\nEnjoy!"]
      (is (nil? (agent/extract-tool-call text))))))

(deftest test-format-tool-response
  (testing "Formatting success execution result with native tool response tokens"
    (let [res {:status :success :output "30"}
          resp (agent/format-tool-response "eval_clojure" res)]
      (is (str/starts-with? resp "<|tool_response>response:eval_clojure{output:<|\"|>"))
      (is (str/includes? resp "30"))
      (is (str/ends-with? resp "<|\"|>}<tool_response|>"))))

  (testing "Formatting error execution result"
    (let [res {:status :error :output "Execution Exception: Divide by zero"}
          resp (agent/format-tool-response "eval_clojure" res)]
      (is (str/includes? resp "Divide by zero"))
      (is (str/ends-with? resp "<|\"|>}<tool_response|>")))))

(deftest test-eval-tool-code-success
  (testing "Evaluating math expressions in SCI sandbox"
    (let [sci-ctx (agent/create-agent-sci-ctx)
          res (agent/eval-tool-code sci-ctx "(reduce + (range 10))")]
      (is (= :success (:status res)))
      (is (str/includes? (:output res) "45")))))

(deftest test-eval-tool-code-error-handling
  (testing "Handling runtime exceptions inside SCI sandbox"
    (let [sci-ctx (agent/create-agent-sci-ctx)
          res (agent/eval-tool-code sci-ctx "(/ 1 0)")]
      (is (= :error (:status res)))
      (is (str/includes? (:output res) "Execution Exception")))))

(defspec prop-sci-arithmetic-eval-invariant
  50
  (prop/for-all [a (gen/choose 1 1000)
                 b (gen/choose 1 1000)]
                (let [sci-ctx (agent/create-agent-sci-ctx)
                      code (format "(+ %d %d)" a b)
                      res (agent/eval-tool-code sci-ctx code)]
                  (and (= :success (:status res))
                       (str/includes? (:output res) (str (+ a b)))))))

(deftest test-extract-thinking-trace
  (testing "Extracting Gemma 4 native channel thought"
    (let [text "<|channel>thought\nAnalyzing the problem...\nWe should use mapv.<channel|>\nHere is the code:\n```clojure\n(mapv inc [1 2 3])\n```"
          thought (agent/extract-thinking-trace text)]
      (is (= "Analyzing the problem...\nWe should use mapv." thought))))

  (testing "Extracting thought tags <|thought|> ... <thought|>"
    (let [text "<|thought|>\nConsidering options.\n<thought|>\nDone."
          thought (agent/extract-thinking-trace text)]
      (is (= "Considering options." thought))))

  (testing "Extracting XML <thought> tags"
    (let [text "<thought>Let's plan out the steps.</thought>\nPlan ready."
          thought (agent/extract-thinking-trace text)]
      (is (= "Let's plan out the steps." thought))))

  (testing "Extracting XML <think> tags"
    (let [text "<think>Reasoning step 1.\nReasoning step 2.</think>\nResult."
          thought (agent/extract-thinking-trace text)]
      (is (= "Reasoning step 1.\nReasoning step 2." thought))))

  (testing "Extracting unclosed thinking trace up to EOF"
    (let [text "<|channel>thought\nStill thinking midway through generation..."
          thought (agent/extract-thinking-trace text)]
      (is (= "Still thinking midway through generation..." thought))))

  (testing "Returns nil when no thinking tags are present"
    (let [text "Just plain response with no thoughts."
          thought (agent/extract-thinking-trace text)]
      (is (nil? thought)))))

(deftest test-strip-thinking-trace
  (testing "Stripping thoughts removes thought blocks cleanly"
    (let [text "<|channel>thought\nLet me think about this.\n<channel|>\nHere is the answer: 42."
          stripped (agent/strip-thinking-trace text)]
      (is (= "Here is the answer: 42." stripped))))

  (testing "Stripping multiple thoughts"
    (let [text "<think>Thought 1</think>\nPart 1\n<think>Thought 2</think>\nPart 2"
          stripped (agent/strip-thinking-trace text)]
      (is (not (str/includes? stripped "Thought 1")))
      (is (not (str/includes? stripped "Thought 2")))
      (is (str/includes? stripped "Part 1"))
      (is (str/includes? stripped "Part 2")))))

(deftest test-code-extraction-with-thinking
  (testing "Code blocks in response are extracted even when thoughts precede them"
    (let [text "<|channel>thought\nI think I will use (range 10) or (mapv inc).\n<channel|>\n```clojure\n(filter pos? [-1 0 1 2])\n```"
          blocks (agent/extract-clojure-code-blocks text)]
      (is (= ["(filter pos? [-1 0 1 2])"] blocks))))

  (testing "Does not extract bare sexpr from thoughts when no code block is present in final answer"
    (let [text "<|channel>thought\nMaybe (range 10) would work, but the answer is obvious.<channel|>\nThe answer is forty-two."
          blocks (agent/extract-clojure-code-blocks text)]
      (is (empty? blocks)))))

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
      (is (str/starts-with? prompt "<bos><|turn>user\nHello<turn|>\n"))))

  (testing "Formatting prompt with thinking enabled and system prompt"
    (let [sys "You are an assistant."
          history [{:role :user :content "Calculate 2+2"}]
          prompt (agent/format-agent-chat-prompt sys history 8 true)]
      (is (str/starts-with? prompt "<bos><|turn>system\n<|think|>\nYou are an assistant.<turn|>\n"))
      (is (str/includes? prompt "<|turn>user\nCalculate 2+2<turn|>\n"))
      (is (str/ends-with? prompt "<|turn>model\n"))))

  (testing "Formatting prompt with thinking enabled and empty system prompt"
    (let [history [{:role :user :content "Calculate 2+2"}]
          prompt (agent/format-agent-chat-prompt "" history 8 true)]
      (is (str/starts-with? prompt "<bos><|turn>system\n<|think|><turn|>\n"))
      (is (str/includes? prompt "<|turn>user\nCalculate 2+2<turn|>\n"))
      (is (str/ends-with? prompt "<|turn>model\n"))))

  (testing "Formatting prompt with tool call and tool response in history"
    (let [history [{:role :user :content "Calculate 2+2"}
                   {:role :model :content "<|tool_call>call:eval_clojure{code:\"(+ 2 2)\"}<tool_call|>"}
                   {:role :tool :content "<|tool_response>response:eval_clojure{output:<|\"|>4<|\"|>}<tool_response|>"}]
          prompt (agent/format-agent-chat-prompt "You are an assistant." history)]
      (is (str/includes? prompt "<|turn>model\n<|tool_call>call:eval_clojure{code:\"(+ 2 2)\"}<tool_call|><|tool_response>response:eval_clojure{output:<|\"|>4<|\"|>}<tool_response|>"))
      (is (not (str/includes? prompt "<turn|>\n<|turn>user\n<|tool_response>")))
      (is (str/ends-with? prompt "<tool_response|>\n"))))

  (testing "Formatting prompt consolidates tool declaration inside system turn"
    (let [decl "<|tool>declaration:eval_clojure{...}<tool|>"
          history [{:role :user :content "Hi"}]
          prompt (agent/format-agent-chat-prompt "You are a bot." history 8 false decl)]
      (is (str/starts-with? prompt "<bos><|turn>system\nYou are a bot.\n<|tool>declaration:eval_clojure{...}<tool|><turn|>\n"))
      (is (not (str/starts-with? prompt "<bos><|tool>"))))))

(defspec prop-format-agent-chat-prompt-invariants
  50
  (prop/for-all [sys (gen/not-empty gen/string-alphanumeric)
                 user-msg (gen/not-empty gen/string-alphanumeric)]
                (let [prompt (agent/format-agent-chat-prompt sys [{:role :user :content user-msg}])]
                  (and (str/starts-with? prompt "<bos><|turn>system\n")
                       (str/includes? prompt (str "<|turn>user\n" user-msg "<turn|>\n"))
                       (str/ends-with? prompt "<|turn>model\n")
                       (not (str/includes? prompt "<|think|>"))))))

(defspec prop-format-agent-chat-prompt-thinking-invariant
  50
  (prop/for-all [sys (gen/not-empty gen/string-alphanumeric)
                 user-msg (gen/not-empty gen/string-alphanumeric)]
                (let [prompt (agent/format-agent-chat-prompt sys [{:role :user :content user-msg}] 8 true)]
                  (and (str/starts-with? prompt "<bos><|turn>system\n<|think|>\n")
                       (str/includes? prompt "<|think|>")
                       (str/includes? prompt (str "<|turn>user\n" user-msg "<turn|>\n"))
                       (str/ends-with? prompt "<|turn>model\n")))))

(defspec prop-extract-thinking-trace-invariant
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 response (gen/not-empty gen/string-alphanumeric)]
                (let [text (str "<|channel>thought\n" thought "<channel|>\n" response)
                      extracted (agent/extract-thinking-trace text)
                      stripped (agent/strip-thinking-trace text)]
                  (and (= (str/trim thought) (str/trim extracted))
                       (= (str/trim response) (str/trim stripped))
                       (not (str/includes? stripped "<|channel>"))
                       (not (str/includes? stripped "<channel|>"))))))

(defspec prop-extract-tool-call-roundtrip
  50
  (prop/for-all [a (gen/choose 1 1000)
                 b (gen/choose 1 1000)]
                (let [code (format "(+ %d %d)" a b)
                      text (format "<|tool_call>call:eval_clojure{code:<|\"|>%s<|\"|>}<tool_call|>" code)
                      tc (agent/extract-tool-call text)]
                  (and (= "eval_clojure" (:name tc))
                       (= code (:code tc))))))

(defspec prop-plain-text-no-tool-call
  50
  (prop/for-all [text (gen/not-empty gen/string-alphanumeric)]
                (nil? (agent/extract-tool-call text))))

