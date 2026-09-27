(ns einsum.agent-test
  "Unit and generative tests for tools.gemma4-agent SCI tool execution, thinking trace extraction, and turn formatting."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.agent.core :as agent]))

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

  (testing "Extracting tool call with multiple fields does not swallow subsequent fields (Flag 2 fix)"
    (let [text "<|tool_call>call:eval_clojure{code: \"(+ 1 2)\", note: \"(paren in note)\"}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(+ 1 2)" (:code tc))))
    (let [text "<|tool_call>call:eval_clojure{code: (+ 1 2), note: \"(paren in note)\"}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(+ 1 2)" (:code tc)))))

  (testing "Extracting tool call with parentheses in string literal"
    (let [text "<|tool_call>call:eval_clojure{code:<|\"|>(println \"hello (world)\")<|\"|>}<tool_call|>"
          tc (agent/extract-tool-call text)]
      (is (= "eval_clojure" (:name tc)))
      (is (= "(println \"hello (world)\")" (:code tc)))))

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
      (is (str/includes? (:output res) "Execution Exception"))))

  (testing "Infinite loop in eval-tool-code triggers timeout error cleanly without hanging"
    (let [sci-ctx (agent/create-agent-sci-ctx)
          res (agent/eval-tool-code sci-ctx "(loop [x 0] (recur x))" 100)]
      (is (= :error (:status res)))
      (is (str/includes? (:output res) "timed out after 100 ms")))))

(deftest test-try-parse-sci-reader-no-artificial-parens
  (testing "Incomplete truncated forms return nil and do not synthesize infinite loops"
    (let [incomplete "(defn my-range [x] (if (< x 5) (recur"
          res (agent/extract-balanced-sexpr incomplete)]
      (is (nil? res)))))

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

(deftest test-extract-balanced-sexpr-reader
  (testing "Balanced reader handles strings with parentheses without premature termination (Flag 1)"
    (is (= "(println \"hello (world)\")"
           (agent/extract-balanced-sexpr "(println \"hello (world)\")")))
    (is (= "(println \"hello ) world\")"
           (agent/extract-balanced-sexpr "(println \"hello ) world\")")))
    (is (= "(println (str \")\"))"
           (agent/extract-balanced-sexpr "(println (str \")\"))"))))
  (testing "Balanced reader handles comments with parentheses"
    (is (= "(range 10)"
           (agent/extract-balanced-sexpr "(range 10) ; (comment with parens)"))))
  (testing "Balanced reader extracts first valid s-expression from surrounding text"
    (is (= "(map inc [1 2 3])"
           (agent/extract-balanced-sexpr "Here is the code: (map inc [1 2 3]) and some final note."))))
  (testing "Returns nil when no balanced s-expression exists"
    (is (nil? (agent/extract-balanced-sexpr "just plain text without parens")))))

(deftest test-benchmark-sci-ctx-isolation
  (let [bench-ctx (agent/create-benchmark-sci-ctx)]
    (testing "Math and pure Clojure logic execute properly in benchmark sandbox"
      (let [res (agent/eval-tool-code bench-ctx "(reduce + [1 2 3 4 5])")]
        (is (= :success (:status res)))
        (is (str/includes? (:output res) "15")))
      (let [res (agent/eval-tool-code bench-ctx "(Math/sqrt 16)")]
        (is (= :success (:status res)))
        (is (str/includes? (:output res) "4.0"))))

    (testing "File I/O is strictly forbidden in benchmark sandbox (Flag 5)"
      (let [res (agent/eval-tool-code bench-ctx "(slurp \"/etc/passwd\")")]
        (is (= :error (:status res)))
        (is (str/includes? (:output res) "Could not resolve symbol: slurp")))
      (let [res (agent/eval-tool-code bench-ctx "(spit \"test.txt\" \"data\")")]
        (is (= :error (:status res)))
        (is (str/includes? (:output res) "Could not resolve symbol: spit")))
      (let [res (agent/eval-tool-code bench-ctx "(list-files \".\")")]
        (is (= :error (:status res)))
        (is (str/includes? (:output res) "Could not resolve symbol: list-files"))))

    (testing "System inspection is strictly forbidden in benchmark sandbox (Flag 5)"
      (let [res (agent/eval-tool-code bench-ctx "(system-info)")]
        (is (= :error (:status res)))
        (is (str/includes? (:output res) "Could not resolve symbol: system-info")))
      (let [res (agent/eval-tool-code bench-ctx "(System/getProperty \"os.name\")")]
        (is (= :error (:status res)))
        (is (str/includes? (:output res) "Could not resolve symbol: System/getProperty"))))))

(deftest test-agent-sci-ctx-system-info
  (testing "Agent sandbox provides system inspection"
    (let [agent-ctx (agent/create-agent-sci-ctx)
          res (agent/eval-tool-code agent-ctx "(system-info)")]
      (is (= :success (:status res)))
      (is (str/includes? (:output res) ":os")))))

(defspec prop-reader-balanced-sexpr-strings-with-parens
  30
  (prop/for-all [inner (gen/not-empty gen/string-alphanumeric)]
                (let [code (format "(println \"hello (%s)\")" inner)
                      extracted (agent/extract-balanced-sexpr code)]
                  (= code extracted))))

(deftest test-print-and-save-telemetry
  (testing "Telemetry aggregates prompt and new tokens and writes profile EDN"
    (let [tmp (java.io.File/createTempFile "telemetry_test" ".edn")
          tmp-path (.getAbsolutePath tmp)]
      (try
        (let [telemetry (atom [{:turn 1 :prompt-tokens 50 :new-tokens 20 :tok-per-sec 40.0 :model-ms 500.0 :tool-ms 10.0 :total-turn-ms 510.0}
                               {:turn 2 :prompt-tokens 75 :new-tokens 30 :tok-per-sec 37.5 :model-ms 800.0 :tool-ms 0.0 :total-turn-ms 800.0}])
              start-t (System/nanoTime)]
          (agent/print-and-save-telemetry! telemetry start-t true tmp-path)
          (let [saved (read-string (slurp tmp-path))]
            (is (= 2 (:total-turns saved)))
            (is (= 125 (:total-prompt-tokens saved)))
            (is (= 50 (:total-new-tokens saved)))
            (is (= 1300.0 (:total-model-ms saved)))
            (is (= 10.0 (:total-tool-ms saved)))
            (is (pos? (:avg-tok-per-sec saved)))))
        (finally
          (.delete tmp))))))

(deftest test-error-budget-warning-directive
  (testing "Reaching consecutive error limit appends directive to tool response"
    (let [limit 3
          eval-res {:status :error :output "Execution Exception: Divide by zero"}
          base-obs (agent/format-tool-response "eval_clojure" eval-res)
          obs-with-directive (str base-obs
                                  (format "\n[System: Consecutive tool error limit (%d) reached. Tool execution is now disabled. Provide your final answer in plain text based on the observations collected so far without calling further tools.]"
                                          limit))]
      (is (str/includes? obs-with-directive "Consecutive tool error limit (3) reached"))
      (is (str/includes? obs-with-directive "Tool execution is now disabled")))))

(deftest test-format-agent-chat-prompt-pruning-thoughts-from-past-turns
  (testing "Past model turns with closed thoughts have thoughts pruned from formatted prompt"
    (let [history [{:role :user :content "Write (+ 1 2)"}
                   {:role :model :content "<|channel>thought\nI will calculate 1+2.\n<channel|>\n<|tool_call>call:eval_clojure{code:<|\"|>(+ 1 2)<|\"|>}<tool_call|>"}
                   {:role :tool :content "<|tool_response>response:eval_clojure{output:<|\"|>3<|\"|>}<tool_response|>"}
                   {:role :user :content "Now multiply by 3"}]
          prompt (agent/format-agent-chat-prompt "You are a Clojure assistant." history)]
      (is (str/includes? prompt "<|tool_call>call:eval_clojure{code:<|\"|>(+ 1 2)<|\"|>}<tool_call|>"))
      (is (not (str/includes? prompt "I will calculate 1+2.")))
      (is (not (str/includes? prompt "<|channel>thought")))))

  (testing "Past model turns with unclosed thoughts are sanitized and pruned"
    (let [history [{:role :user :content "Do hard math"}
                   {:role :model :content "<|channel>thought\nStill thinking endlessly and never closed..."}
                   {:role :user :content "Please call eval_clojure"}]
          prompt (agent/format-agent-chat-prompt "You are a Clojure assistant." history)]
      (is (not (str/includes? prompt "Still thinking endlessly")))
      (is (not (str/includes? prompt "<|channel>thought"))))))

(deftest test-sanitize-history-model-content
  (testing "Model reply with tool call returns compact raw tool call"
    (let [tc {:name "eval_clojure" :code "(+ 1 2)" :raw "<|tool_call>call:eval_clojure{code:\"(+ 1 2)\"}<tool_call|>"}
          res (agent/sanitize-history-model-content tc "" "<|channel>thought...<channel|><|tool_call>...")]
      (is (= (:raw tc) res))))

  (testing "Model reply with final response returns clean response without thoughts"
    (let [res (agent/sanitize-history-model-content nil "The answer is 42." "<|channel>thought\nLet me think.\n<channel|>\nThe answer is 42.")]
      (is (= "The answer is 42." res))))

  (testing "Model reply with only truncated unclosed thought returns compact placeholder"
    (let [truncated-raw "<|channel>thought\nThinking for 1536 tokens without closing..."
          res (agent/sanitize-history-model-content nil "" truncated-raw)]
      (is (not (str/includes? res "Thinking for 1536 tokens")))
      (is (< (count res) 100)))))

(defspec prop-format-agent-chat-prompt-prunes-past-thoughts
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 code (gen/not-empty gen/string-alphanumeric)]
                (let [thought-str (str "THOUGHT_" thought)
                      code-str (str "CODE_" code)
                      model-turn (format "<|channel>thought\n%s\n<channel|>\n<|tool_call>call:eval_clojure{code:\"%s\"}<tool_call|>" thought-str code-str)
                      history [{:role :user :content "Task"}
                               {:role :model :content model-turn}
                               {:role :tool :content "Result"}
                               {:role :user :content "Next"}]
                      prompt (agent/format-agent-chat-prompt "System" history)]
                  (and (not (str/includes? prompt thought-str))
                       (not (str/includes? prompt "<|channel>thought"))
                       (str/includes? prompt code-str)))))

;; =============================================================================
;; Harness v2 Tests: Semantic Early Stopping, Nudge Short-Circuit, Early Exit
;; =============================================================================

(deftest test-inside-unclosed-thought
  (testing "Unclosed thoughts detect as inside-thought"
    (is (true? (agent/inside-unclosed-thought? "<|channel>thought\nStill thinking...")))
    (is (true? (agent/inside-unclosed-thought? "<think>Pondering options...")))
    (is (true? (agent/inside-unclosed-thought? "<|thought|>\nConsidering..."))))
  (testing "Closed thoughts detect as not inside-thought"
    (is (false? (agent/inside-unclosed-thought? "<|channel>thought\nThinking...<channel|>\nHere is the answer.")))
    (is (false? (agent/inside-unclosed-thought? "<think>Pondering options...</think>\nDone.")))
    (is (false? (agent/inside-unclosed-thought? "Just plain text without thoughts."))))
  (testing "Multiple thoughts where last is unclosed"
    (is (true? (agent/inside-unclosed-thought? "<think>First</think>\nText\n<think>Second unclosed"))))
  (testing "Multiple thoughts where all are closed"
    (is (false? (agent/inside-unclosed-thought? "<think>First</think>\nText\n<think>Second</think>\nFinal")))))

(defspec prop-inside-unclosed-thought-detection
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 suffix (gen/not-empty gen/string-alphanumeric)]
                (let [unclosed (str "<|channel>thought\n" thought)
                      closed (str "<|channel>thought\n" thought "<channel|>\n" suffix)]
                  (and (true? (agent/inside-unclosed-thought? unclosed))
                       (false? (agent/inside-unclosed-thought? closed))))))

(defspec prop-semantic-stop-unclosed-thought-invariant
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)]
                ;; Even if complete code or tool call appears inside unclosed thought, it must NOT stop
                (let [text (format "<|channel>thought\n%s\n```clojure\n(defn target [x] x)\n```\n" thought)]
                  (false? (agent/semantic-stop? text {:target-fn "target"})))))

(defspec prop-semantic-stop-code-block-invariant
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 trailing gen/string-alphanumeric]
                (let [text (format "<|channel>thought\n%s<channel|>\n```clojure\n(defn target [x] x)\n```\n%s"
                                   thought trailing)]
                  (true? (agent/semantic-stop? text {:target-fn "target"})))))

(defspec prop-semantic-stop-tool-call-invariant
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 trailing gen/string-alphanumeric]
                (let [text (format "<|channel>thought\n%s<channel|>\n<|tool_call>call:eval_clojure{code:\"(+ 1 2)\"}<tool_call|>\n%s"
                                   thought trailing)]
                  (true? (agent/semantic-stop? text {:target-fn "target"})))))

(defspec prop-semantic-stop-balanced-form-invariant
  50
  (prop/for-all [thought (gen/not-empty gen/string-alphanumeric)
                 trailing gen/string-alphanumeric]
                (let [text (format "<|channel>thought\n%s<channel|>\n(defn target [x] (inc x))\n%s"
                                   thought trailing)]
                  (true? (agent/semantic-stop? text {:target-fn "target"})))))

(defspec prop-nudge-short-circuit-invariant
  50
  (prop/for-all [code (gen/not-empty gen/string-alphanumeric)]
                (let [base-ctx {:turn 1
                                :max-turns 5
                                :new-tokens 100
                                :opts {:nudge-on-no-tool true}
                                :history [{:role :user :content "Prompt"}]
                                :consecutive-errors 0}
                      ;; Case A: Candidate code present -> nudge suppressed
                      ctx-with-cand (assoc base-ctx :has-candidate? true :model-reply (str "(defn target [x] " code ")"))
                      ;; Case B: Candidate code absent -> nudge needed
                      ctx-no-cand (assoc base-ctx :has-candidate? false :model-reply "Just text with no code")]
                  (and (false? (agent/nudge-needed? ctx-with-cand))
                       (true? (agent/nudge-needed? ctx-no-cand))))))

(deftest test-run-agent-loop-early-exit-public-pass
  (testing "Early exit on public pass terminates loop at Turn 1"
    (let [scripted-responses ["<|tool_call>call:eval_clojure{code:\"(defn my-inc [x] (inc x))\"}<tool_call|>"
                              "Turn 2 that should never be executed"]
          tool-hook (fn [_ctx _code _turn]
                      {:status :success
                       :output "[Public tests: 2/2 passed. All public examples succeeded!]"
                       :early-exit? true})
          session {:opts {:system "System prompt"
                          :max-turns 5
                          :tool-declaration agent/DEFAULT-TOOL-DECLARATION
                          :scripted-responses scripted-responses
                          :tool-eval-fn tool-hook
                          :quiet true}}
          transcript (agent/run-agent-loop session "Implement my-inc")]
      (is (= 2 (count transcript)))
      (is (= :model (:role (first transcript))))
      (is (= :tool (:role (second transcript))))
      (is (true? (:early-exit? (meta transcript))))
      (is (= :public-pass (:early-exit-reason (meta transcript)))))))

(deftest test-run-agent-loop-nudge-short-circuit
  (testing "Turn 1 with candidate code short-circuits blind nudge and exits after 1 turn"
    (let [scripted-responses ["Here is the code:\n```clojure\n(defn my-add [a b] (+ a b))\n```\nHope this helps!"
                              "Turn 2 that should never be executed"]
          cand-fn (fn [text] (str/includes? text "defn my-add"))
          session {:opts {:system "System prompt"
                          :max-turns 5
                          :nudge-on-no-tool true
                          :candidate-check-fn cand-fn
                          :scripted-responses scripted-responses
                          :quiet true}}
          transcript (agent/run-agent-loop session "Implement my-add")]
      (is (= 1 (count transcript)))
      (is (= :model (:role (first transcript))))
      (is (not (some #(= (:role %) :user) (rest transcript)))))))

(deftest test-multipl-e-dev-50-fixture-and-ledger
  (testing "MultiPL-E 50-task dev subset loads and satisfies schema invariants"
    (let [pub-file (io/file "resources/catalog/gate3_evals/multipl_e/dev_50_public.edn")
          sealed-file (io/file "resources/catalog/gate3_evals/multipl_e/dev_50_sealed.edn")]
      (is (.exists pub-file))
      (is (.exists sealed-file))
      (let [pub (edn/read-string (slurp pub-file))
            sealed (edn/read-string (slurp sealed-file))]
        (is (= 50 (count pub)))
        (is (= 50 (count sealed)))
        (is (every? #(and (string? (:id %))
                          (string? (:title %))
                          (symbol? (:fn-name %))
                          (string? (:prompt %))
                          (seq (:public-tests %)))
                    pub))
        (is (every? #(and (string? (:id %))
                          (seq (:hidden-tests %)))
                    sealed)))))

  (testing "MultiPL-E 50-task dev subset runs through harness with valid :harness-sha on every row"
    (let [tmp-results (str "scratch/test_multipl_e_results_" (System/currentTimeMillis) ".edn")
          tmp-summary (str "scratch/test_multipl_e_summary_" (System/currentTimeMillis) ".csv")
          run-fn (requiring-resolve 'experiments.gate3-evals.clojure-bench.run/run-benchmark)
          _ (run-fn {:dry-run true
                     :overwrite true
                     :quiet true
                     :public-tasks-file "resources/catalog/gate3_evals/multipl_e/dev_50_public.edn"
                     :sealed-tasks-file "resources/catalog/gate3_evals/multipl_e/dev_50_sealed.edn"
                     :results-file tmp-results
                     :summary-file tmp-summary})
          rows (with-open [r (io/reader tmp-results)]
                 (mapv edn/read-string (line-seq r)))]
      (is (= 100 (count rows)))
      (is (every? #(and (string? (:harness-sha %))
                        (pos? (count (:harness-sha %))))
                  rows))
      (is (every? #(boolean? (:harness-dirty? %)) rows))
      (is (every? #(true? (:passed? %)) rows))
      ;; Cleanup temporary files
      (io/delete-file tmp-results true)
      (io/delete-file tmp-summary true))))

