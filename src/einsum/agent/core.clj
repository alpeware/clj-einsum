(ns einsum.agent.core
  "Autonomous software architecture agent loop, SCI Clojure tool calling,
   prompt synthesis, thinking trace extraction, and semantic early stopping."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [sci.core :as sci]))

;; =============================================================================
;; 1. Default Prompts & Options
;; =============================================================================

(def DEFAULT-TOOL-DECLARATION
  "<|tool>declaration:eval_clojure{description:<|\"|>Execute Clojure code in the SCI Clojure sandbox and return the evaluation result. Code must be valid Clojure s-expressions.<|\"|>,parameters:{properties:{code:{description:<|\"|>The Clojure code to evaluate.<|\"|>,type:<|\"|>string<|\"|>}},required:[<|\"|>code<|\"|>],type:<|\"|>object<|\"|>}}<tool|>")

(def DEFAULT-SYSTEM-PROMPT
  "You are a Clojure assistant with access to the eval_clojure tool.
Call eval_clojure to execute or test Clojure code. Once you have the result, provide your final response to the user in plain text without further tool calls.
Syntax rules: use square brackets for bindings and parameters: [x], [k v], vectors: [1 2 3], and (range start end).")

(def DEFAULT-SYSTEM-PROMPT-XML
  "You are a Clojure assistant with access to an interactive Clojure REPL sandbox.
To evaluate or test Clojure code, output your code wrapped in <clojure>...</clojure> tags:
<clojure>
(defn my-fn [x]
  (* x 2))
</clojure>
Execution results will be returned in <clojure_result>...</clojure_result>.
Once you have the result, provide your final response to the user in plain text without further code tags.
Syntax rules: use square brackets for bindings and parameters: [x], [k v], vectors: [1 2 3], and (range start end).")

(def DEFAULT-SYSTEM-PROMPT-FENCED
  "You are a Clojure assistant with access to an interactive Clojure REPL sandbox.
To evaluate or test Clojure code, output your code wrapped in a ```clojure ... ``` markdown fence:
```clojure
(defn my-fn [x]
  (* x 2))
```
Execution results will be returned in ```clojure_result ... ```.
Once you have the result, provide your final response to the user in plain text without further code tags.
Syntax rules: use square brackets for bindings and parameters: [x], [k v], vectors: [1 2 3], and (range start end).")

(def DEFAULT-AGENT-OPTS
  {:prompt "Write a Clojure function returning the first 10 integers."
   :model ".models/gemma-4-E2B-it"
   :system DEFAULT-SYSTEM-PROMPT
   :tool-declaration DEFAULT-TOOL-DECLARATION
   :tool-syntax :native
   :max-new-tokens 512
   :max-seq-len 1024
   :max-turns 5
   :max-consecutive-errors 3
   :sandbox :agent
   :temperature 0.0
   :top-k 10
   :repetition-penalty 1.0
   :method nil
   :backend :rocm
   :precision :bf16
   :thinking false
   :out "scratch/output_agent_loop.txt"
   :profile-out "scratch/gemma4_agent_profile.edn"
   :chrome-trace-out "scratch/gemma4_agent_chrome_trace.json"
   :quiet false})

;; =============================================================================
;; 2. Reader-Based S-Expression & Code Parsing
;; =============================================================================

(def ^:private default-reader-ctx (sci/init {}))

(defn try-parse-sci-reader
  "Attempts to parse the first form from string `s` using SCI's reader.
   Returns the string form if complete, or nil if incomplete or malformed.
   Does not synthesize artificial closing parens on truncated code to avoid creating broken loops."
  [s]
  (try
    (second (sci/parse-next+string default-reader-ctx (sci/source-reader s)))
    (catch Throwable _ nil)))

(defn extract-balanced-sexpr
  "Finds the first balanced s-expression string starting with `(` in text using SCI's reader.
   Correctly handles strings with parentheses and comments without naive paren counting.
   If a top-level definition form (defn, def, defmacro) is incomplete, returns nil to avoid extracting inner fragments."
  [text]
  (when (string? text)
    (loop [offset 0]
      (let [start (.indexOf ^String text "(" offset)]
        (when (>= start 0)
          (let [candidate (subs text start)]
            (if-let [parsed (try-parse-sci-reader candidate)]
              parsed
              (if (re-find #"^\((?:defn|defn-|defmacro|def)\b" candidate)
                nil
                (recur (inc start))))))))))

(defn- unescape-json-string
  "Unescapes standard JSON string escape sequences."
  [s]
  (if (string? s)
    (-> s
        (str/replace #"\\\"" "\"")
        (str/replace #"\\n" "\n")
        (str/replace #"\\t" "\t")
        (str/replace #"\\\\" "\\"))
    s))

(defn parse-tool-call-code
  "Extracts the Clojure code string from a Gemma 4 tool call argument block.
   Employs reader-based balanced extraction when available and falls back gracefully
   to raw unescaped code blocks to ensure syntax errors are caught by the evaluator
   rather than dropping the tool call."
  [args-str]
  (let [trimmed (str/trim (or args-str ""))
        unbraced (-> trimmed
                     (str/replace #"^\{" "")
                     (str/replace #"\}$" "")
                     str/trim)]
    (cond
      ;; 1. Gemma 4 native string delimiter: code:<|"|>...<|"|> or code:=<|"|>...
      (re-find #"(?s)code\s*[:=]+\s*<\|\"\|>(.*?)(?:<\|\"\|>|$)" unbraced)
      (second (re-find #"(?s)code\s*[:=]+\s*<\|\"\|>(.*?)(?:<\|\"\|>|$)" unbraced))

      ;; 2. Standard JSON quotes with escaped string support: code:"..." or code:="..."
      (re-find #"(?s)code\s*[:=]+\s*\"([^\"\\]*(?:\\.[^\"\\]*)*)\"" unbraced)
      (unescape-json-string (second (re-find #"(?s)code\s*[:=]+\s*\"([^\"\\]*(?:\\.[^\"\\]*)*)\"" unbraced)))

      ;; 3. Single quotes: code:'...' or code:='...'
      (re-find #"(?s)code\s*[:=]+\s*\'([^\']*)\'" unbraced)
      (second (re-find #"(?s)code\s*[:=]+\s*\'([^\']*)\'" unbraced))

      ;; 4. code: or code:= followed by an s-expression or raw code
      (re-find #"(?s)code\s*[:=]+" unbraced)
      (let [m (re-matcher #"(?s)code\s*[:=]+\s*" unbraced)]
        (when (.find m)
          (let [after-code (subs unbraced (.end m))
                code-segment (first (str/split after-code #",\s*[a-zA-Z_][a-zA-Z0-9_-]*\s*:" 2))
                clean-after (-> (or code-segment after-code)
                                (str/replace #"(?:<\|\"\|>|\"|\'|[,\}\s])+$" "")
                                str/trim)
                balanced (try (extract-balanced-sexpr after-code) (catch Throwable _ nil))]
            (or (when (and (seq clean-after) (str/starts-with? clean-after "("))
                  clean-after)
                (when (seq balanced) balanced)
                (when (seq clean-after) clean-after)))))

      ;; 5. Any bare s-expression or raw code within the tool call argument block
      :else
      (let [code-segment (first (str/split unbraced #",\s*[a-zA-Z_][a-zA-Z0-9_-]*\s*:" 2))
            clean-unbraced (-> (or code-segment unbraced)
                               (str/replace #"(?:<\|\"\|>|\"|\'|[,\}\s])+$" "")
                               str/trim)
            balanced (try (extract-balanced-sexpr unbraced) (catch Throwable _ nil))]
        (or (when (and (seq clean-unbraced) (str/starts-with? clean-unbraced "("))
              clean-unbraced)
            (when (seq balanced) balanced))))))

;; =============================================================================
;; 3. Thought & Reasoning Trace Extraction
;; =============================================================================

(def ^:private THOUGHT-TAG-PAIRS
  [["<|channel>thought" "<channel|>"]
   ["<|thought|>" "<thought|>"]
   ["<thought>" "</thought>"]
   ["<think>" "</think>"]])

(defn inside-unclosed-thought?
  "Returns true if `text` contains an active unclosed thought channel or block.
   Determined by finding the latest thought opening tag and verifying whether
   a matching or corresponding closing tag succeeds it."
  [text]
  (when (string? text)
    (loop [pairs THOUGHT-TAG-PAIRS]
      (if (empty? pairs)
        false
        (let [[open-tag close-tag] (first pairs)
              last-open (.lastIndexOf ^String text ^String open-tag)]
          (if (>= last-open 0)
            (let [last-close (.lastIndexOf ^String text ^String close-tag)]
              (if (or (< last-close 0) (< last-close last-open))
                true
                (recur (rest pairs))))
            (recur (rest pairs))))))))

(defn extract-thinking-trace
  "Extracts reasoning thoughts from model output text.
   Supports Gemma 4 native channel (<|channel>thought ... <channel|>),
   XML tags (<thought>...</thought>, <think>...</think>),
   and Gemma thought tags (<|thought|>...<thought|>)."
  [text]
  (when (string? text)
    (let [patterns [#"(?s)<\|channel>thought\s*(.*?)(?:<channel\|>|$)"
                    #"(?s)<\|thought\|>\s*(.*?)(?:<thought\|>|$)"
                    #"(?s)<thought>\s*(.*?)(?:</thought>|$)"
                    #"(?s)<think>\s*(.*?)(?:</think>|$)"]
          matches (mapcat (fn [pat]
                            (mapv second (re-seq pat text)))
                          patterns)
          cleaned (mapv str/trim (filter #(seq (str/trim %)) matches))]
      (when (seq cleaned)
        (str/join "\n\n" cleaned)))))

(defn strip-thinking-trace
  "Removes thinking and reasoning blocks from text to isolate final response or tool calls."
  [text]
  (if (string? text)
    (let [patterns [#"(?s)<\|channel>thought\s*.*?(?:<channel\|>|$)"
                    #"(?s)<\|thought\|>\s*.*?(?:<thought\|>|$)"
                    #"(?s)<thought>\s*.*?(?:</thought>|$)"
                    #"(?s)<think>\s*.*?(?:</think>|$)"]]
      (str/trim (reduce (fn [acc pat] (str/replace acc pat "")) text patterns)))
    text))

(defn extract-tool-call
  "Extracts the tool call from text if present.
   Supports :native Gemma 4 syntax (<|tool_call>call:eval_clojure{code:...}<tool_call|>),
   :xml syntax (<clojure>...</clojure> or <clj>...</clj>),
   and :fenced syntax (```clojure ... ``` or ```clj ... ```).
   Returns a map {:name fn-name :code clojure-code-str :raw raw-tool-call-str} or nil if text is a plain text response or malformed."
  ([text]
   (extract-tool-call text :native))
  ([text tool-syntax]
   (let [clean (strip-thinking-trace (or text ""))
         syntax (keyword (or tool-syntax :native))]
     (case syntax
       :xml
       (when-let [[raw code] (re-find #"(?s)<(?:clojure|clj)>\s*(.*?)\s*(?:</(?:clojure|clj)>|$)" clean)]
         (let [trimmed-code (str/trim (or code ""))]
           (when (seq trimmed-code)
             {:name "eval_clojure"
              :code trimmed-code
              :raw (if (or (str/ends-with? (str/trim raw) "</clojure>")
                           (str/ends-with? (str/trim raw) "</clj>"))
                     (str/trim raw)
                     (str "<clojure>\n" trimmed-code "\n</clojure>"))})))

       :fenced
       (when-let [[raw code] (or (re-find #"(?s)```(?:clojure|clj)(?!_)\s*\n?(.*?)(?:```|$)" clean)
                                 (re-find #"(?s)```\s*\n(.*?)\n```" clean))]
         (let [trimmed-code (str/trim (or code ""))]
           (when (seq trimmed-code)
             {:name "eval_clojure"
              :code trimmed-code
              :raw (if (str/ends-with? (str/trim raw) "```")
                     (str/trim raw)
                     (str "```clojure\n" trimmed-code "\n```"))})))

       ;; default :native
       (when-let [[raw fn-name args] (re-find #"(?s)<\|tool_call>(?:call:)?(\w+)?\s*(.*?)(?:<tool_call\|>|$)" clean)]
         (when-let [code (parse-tool-call-code args)]
           {:name (if (seq fn-name) fn-name "eval_clojure")
            :code (str/trim code)
            :raw (if (str/ends-with? raw "<tool_call|>")
                   raw
                   (str raw "<tool_call|>"))}))))))

(defn sanitize-history-model-content
  "Sanitizes model output before recording into conversation history.
   When a tool call is present, retains the thought process preceding the tool call
   in accordance with Google's Gemma 4 Function Calling Exception specification.
   When output is a final response, strips thinking traces to avoid multi-turn drift.
   When output consists solely of truncated, unclosed thoughts, substitutes a compact reminder
   to prevent context window bloat and compounding prefill latency."
  [tool-call final-response model-reply]
  (cond
    tool-call
    (let [thinking (extract-thinking-trace model-reply)]
      (if (seq thinking)
        (str "<|channel>thought\n" (str/trim thinking) "\n<channel|>" (:raw tool-call))
        (:raw tool-call)))

    (and (string? final-response) (seq (str/trim final-response)))
    (str/trim final-response)

    :else
    "[Incomplete generation: token limit reached without tool call]"))

(defn- format-tool-response-value
  [v]
  (cond
    (nil? v)
    "<|\"|><|\"|>"

    (or (number? v) (boolean? v))
    (str v)

    (keyword? v)
    (format "<|\"|>%s<|\"|>" (name v))

    :else
    (format "<|\"|>%s<|\"|>" (str/trim (str v)))))

(defn format-tool-response
  "Formats the tool execution result as a tool response observation.
   Supports :native Gemma 4 syntax (<|tool_response>response:...<tool_response|>)
   and :xml syntax (<clojure_result>...</clojure_result>)."
  ([res]
   (format-tool-response "eval_clojure" res :native))
  ([tool-name res]
   (if (keyword? res)
     (format-tool-response "eval_clojure" tool-name res)
     (format-tool-response tool-name res :native)))
  ([tool-name res tool-syntax]
   (let [syntax (keyword (or tool-syntax :native))]
     (if (or (= syntax :xml) (= syntax :fenced))
       (let [content (cond
                       (string? res) (str/trim res)
                       (map? res) (let [out (str/trim (or (:output res) ""))
                                        already-has-summary? (or (str/includes? out "Public tests:")
                                                                 (str/includes? out "Failing tests:"))
                                        failures (when (and (not already-has-summary?) (:failures res))
                                                   (str "Failures: " (str/trim (str (:failures res)))))
                                        nudge (when (and (not already-has-summary?) (:nudge res))
                                                (str/trim (str (:nudge res))))
                                        directive (when (or (= (:status res) :failed)
                                                            (= (:status res) :error)
                                                            (seq failures))
                                                    (if (= syntax :fenced)
                                                      "Please inspect the test results and provide your revised implementation in ```clojure ... ```."
                                                      "Please inspect the test results and provide your revised implementation in <clojure>...</clojure>."))
                                        pieces (filter seq [out failures nudge directive])]
                                    (if (seq pieces)
                                      (str/join "\n\n" pieces)
                                      (pr-str (dissoc res :early-exit?))))
                       :else (str/trim (str res)))]
         (if (= syntax :fenced)
           (format "```clojure_result\n%s\n```" content)
           (format "<clojure_result>\n%s\n</clojure_result>" content)))
       (let [t-name (if (seq tool-name) tool-name "eval_clojure")]
         (cond
           (string? res)
           (format "<|tool_response>response:%s{output:<|\"|>%s<|\"|>}<tool_response|>"
                   t-name (str/trim res))

           (map? res)
           (let [priority-keys [:status :failures :nudge :tests_passed :tests_total :output :system_note]
                 all-keys (distinct (concat (filter #(contains? res %) priority-keys)
                                            (sort (keys (apply dissoc res (conj priority-keys :early-exit?))))))
                 entries (keep (fn [k]
                                 (when-let [v (get res k)]
                                   (when (not (and (string? v) (str/blank? v)))
                                     (str (name k) ":" (format-tool-response-value v)))))
                               all-keys)
                 body (if (seq entries)
                        (str/join "," entries)
                        (str "output:" (format-tool-response-value (or (:output res) ""))))]
             (format "<|tool_response>response:%s{%s}<tool_response|>" t-name body))

           :else
           (format "<|tool_response>response:%s{output:<|\"|>%s<|\"|>}<tool_response|>"
                   t-name (str/trim (str res)))))))))

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` or ```clj ... ``` code block strings, XML <clojure> tags, Gemma 4 <|tool_call> tags, or raw S-expressions from text.
   Maintained for backward compatibility."
  [text]
  (let [extract-raw (fn [s allow-loose?]
                      (let [tagged-matches (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq #"(?s)```(?:clojure|clj)(?!_)\s*\n?(.*?)(?:```|$)" s))))
                            bare-matches (when (empty? tagged-matches)
                                           (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq #"(?s)```\s*\n(.*?)\n```" s)))))
                            raw-matches (vec (concat tagged-matches bare-matches))
                            cleaned-blocks (mapv (fn [block]
                                                   (-> block
                                                       (str/replace #"^```[a-z]*>?" "")
                                                       (str/replace #"```$" "")
                                                       str/trim))
                                                 raw-matches)
                            xml-pattern #"(?s)<(?:clojure|clj)>\s*\n?(.*?)\s*(?:</(?:clojure|clj)>|$)"
                            xml-matches (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq xml-pattern s))))
                            tool-call-pattern #"(?s)<\|tool_call>call:(\w+)(.*?)(?:<tool_call\|>|$)"
                            tool-call-matches (mapv (fn [[_ fn-name args]]
                                                      (let [clean-args (str/trim (str/replace args #"^\{|\}$" ""))]
                                                        (if (seq clean-args)
                                                          (str "(" fn-name " " clean-args ")")
                                                          (str "(" fn-name ")"))))
                                                    (re-seq tool-call-pattern s))]
                        (cond
                          (seq cleaned-blocks) (vec cleaned-blocks)
                          (seq xml-matches) (vec xml-matches)
                          (seq tool-call-matches) (vec tool-call-matches)
                          allow-loose?
                          (let [raw-fn-pattern #"(?s)\((?:defn|def|range|take|filter|map|reduce|\+|\-|\*|\/|list-files|slurp|system-info|println)\b[^\)]*\)"
                                raw-matches (mapv str/trim (re-seq raw-fn-pattern s))]
                            (cond
                              (seq raw-matches) (vec raw-matches)
                              :else (if-let [sexpr (extract-balanced-sexpr s)]
                                      (let [trimmed (str/trim sexpr)]
                                        (if (and (> (count trimmed) 3) (re-find #"^\([a-zA-Z\+\-\*\/0-9]" trimmed))
                                          [trimmed]
                                          []))
                                      [])))
                          :else [])))
        stripped (strip-thinking-trace text)
        stripped-blocks (extract-raw stripped true)]
    (if (seq stripped-blocks)
      stripped-blocks
      (extract-raw text false))))

;; =============================================================================
;; 4. Semantic Early Stopping (Optimization 2)
;; =============================================================================

(defn- contains-target-definition?
  "Checks whether candidate code defines target-fn (symbol or string)."
  [code-str target-fn]
  (if-not target-fn
    true
    (let [target-name (name target-fn)]
      (boolean (re-find (re-pattern (str "(?s)\\((?:defn|defn-|defmacro|def)\\s+"
                                         (java.util.regex.Pattern/quote target-name)
                                         "(?![\\w\\-\\?\\!\\*\\+\\/])"))
                        (or code-str ""))))))

(defn semantic-stop?
  "Evaluates whether text satisfies semantic early stopping conditions:
   1. Text is NOT actively inside an unclosed thought channel.
   2. Outside thought, text contains:
      - A complete XML tool call tag: `<clojure>...</clojure>` (when tool-syntax is :xml).
      - A complete Gemma 4 tool call tag: `<|tool_call>...<tool_call|>`.
      - A complete closed markdown code block (```` ```...``` ````) containing target definition or valid form.
      - A balanced top-level definition form (defn, defmacro) outside markdown.
   Returns true when generation should halt, false otherwise."
  ([text]
   (semantic-stop? text nil))
  ([text opts]
   (when (string? text)
     (if (inside-unclosed-thought? text)
       false
       (let [body (strip-thinking-trace text)
             target-fn (:target-fn opts)
             tool-syntax (keyword (or (:tool-syntax opts) :native))]
         (cond
           ;; Case 0a: Complete XML tool call tag outside thought
           (and (= tool-syntax :xml)
                (boolean (re-find #"(?s)<(?:clojure|clj)>.*?</(?:clojure|clj)>" body)))
           true

           ;; Case 0b: Complete fenced markdown tool call block outside thought
           (and (= tool-syntax :fenced)
                (boolean (or (re-find #"(?s)```(?:clojure|clj)(?!_)\s*\n?\s*(\S.*?)\s*```" body)
                             (re-find #"(?s)```\s*\n\s*(\S.*?)\s*```" body))))
           true

           ;; Case 1: Complete tool call tag
           (boolean (re-find #"(?s)<\|tool_call>(?:call:)?\w*?\s*\{.*?\}<tool_call\|>" body))
           true

           ;; Case 2: Complete closed markdown code block
           (when-let [[_ code-content] (or (re-find #"(?s)```(?:clojure|clj)(?!_)\s*\n(.*?)\n```" body)
                                           (re-find #"(?s)```\s*\n(.*?)\n```" body))]
             (if target-fn
               (contains-target-definition? code-content target-fn)
               (some? (try-parse-sci-reader (str/trim code-content)))))
           true

           ;; Case 3: Balanced top-level definition form
           (when-let [sexpr (extract-balanced-sexpr body)]
             (let [trimmed (str/trim sexpr)]
               (and (re-find #"^\((?:defn|defn-|defmacro|def)\b" trimmed)
                    (if target-fn
                      (contains-target-definition? trimmed target-fn)
                      true))))
           true

           :else false))))))

;; =============================================================================
;; 5. Prompt Formatting (Chat Turns & Thinking Injection)
;; =============================================================================

(defn format-agent-chat-prompt
  "Formats conversation history into Gemma 4 Turn syntax, placing tool declarations and system instructions in native Gemma 4 turns.
   Applies sliding window context retention for long histories to preserve the initial task and alternating turns.
   When `thinking?` is enabled, injects the native Gemma 4 `<|think|>` token into the system turn.
   Supports :native Gemma 4 tool calling turns as well as :xml alternating user/model turns."
  ([system-prompt history]
   (format-agent-chat-prompt system-prompt history 8 false nil :native))
  ([system-prompt history opts-or-max-turns]
   (cond
     (map? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history
                               (long (get opts-or-max-turns :max-recent-turns 8))
                               (boolean (or (:thinking opts-or-max-turns) (:thinking? opts-or-max-turns)))
                               (get opts-or-max-turns :tool-declaration)
                               (keyword (or (:tool-syntax opts-or-max-turns) :native)))

     (boolean? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history 8 opts-or-max-turns nil :native)

     :else
     (format-agent-chat-prompt system-prompt history (long opts-or-max-turns) false nil :native)))
  ([system-prompt history max-recent-turns thinking?]
   (format-agent-chat-prompt system-prompt history max-recent-turns thinking? nil :native))
  ([system-prompt history max-recent-turns thinking? tool-declaration]
   (format-agent-chat-prompt system-prompt history max-recent-turns thinking? tool-declaration :native))
  ([system-prompt history max-recent-turns thinking? tool-declaration tool-syntax]
   (let [syntax (keyword (or tool-syntax :native))
         history-vec (vec history)
         max-turns (long (or max-recent-turns 8))
         trimmed (if (<= (count history-vec) (inc max-turns))
                   history-vec
                   (let [initial (first history-vec)
                         tail-candidates (take-last max-turns (rest history-vec))
                         clean-tail (if (= (:role (first tail-candidates)) :user)
                                      (vec (rest tail-candidates))
                                      (vec tail-candidates))]
                     (into [initial] clean-tail)))
         has-system? (boolean (seq system-prompt))
         clean-system (when has-system? (str/trim system-prompt))
         tool-decl-str (when (and (seq tool-declaration) (not= syntax :xml) (not= syntax :fenced))
                         (str (str/trim tool-declaration)))
         already-has-think? (and has-system? (str/includes? clean-system "<|think|>"))
         thinking-needed? (and thinking? (not already-has-think?))
         base-system (cond
                       (and has-system? thinking-needed?)
                       (str "<|think|>\n" clean-system)

                       has-system?
                       clean-system

                       thinking-needed?
                       "<|think|>"

                       :else nil)
         system-content (cond
                          (and base-system (seq tool-decl-str))
                          (str base-system "\n" tool-decl-str)

                          base-system
                          base-system

                          (seq tool-decl-str)
                          tool-decl-str

                          :else nil)
         system-turn (when system-content
                       (str "<|turn>system\n" system-content "<turn|>\n"))
         last-role (:role (last trimmed))
         turns-str (loop [idx 0
                          acc (StringBuilder.)]
                     (if (>= idx (count trimmed))
                       (.toString acc)
                       (let [{:keys [role content]} (nth trimmed idx)
                             prev-role (when (pos? idx) (:role (nth trimmed (dec idx))))
                             next-role (when (< (inc idx) (count trimmed))
                                         (:role (nth trimmed (inc idx))))
                             clean-content (if (= role :model)
                                             (let [stripped (strip-thinking-trace (or content ""))]
                                               (if (seq (str/trim stripped))
                                                 (str/trim stripped)
                                                 "[Incomplete generation]"))
                                             (str/trim (or content "")))]
                         (cond
                           (= role :user)
                           (do
                             (.append acc "<|turn>user\n")
                             (.append acc clean-content)
                             (.append acc "<turn|>\n")
                             (recur (inc idx) acc))

                           (= role :model)
                           (if (or (= syntax :xml) (= syntax :fenced))
                             (do
                               (.append acc "<|turn>model\n")
                               (.append acc clean-content)
                               (.append acc "<turn|>\n")
                               (recur (inc idx) acc))
                             (do
                               (when-not (= prev-role :tool)
                                 (.append acc "<|turn>model\n"))
                               (.append acc clean-content)
                               (if (= next-role :tool)
                                 ;; Native Gemma 4 in-flow tool interaction: do not close model turn
                                 (recur (inc idx) acc)
                                 (do
                                   (.append acc "<turn|>\n")
                                   (recur (inc idx) acc)))))

                           (= role :tool)
                           (if (or (= syntax :xml) (= syntax :fenced))
                             (do
                               (.append acc "<|turn>user\n")
                               (.append acc clean-content)
                               (.append acc "<turn|>\n")
                               (recur (inc idx) acc))
                             (do
                               ;; Native Gemma 4 tool response directly attaches to model tool call without trailing newline
                               (.append acc clean-content)
                               (recur (inc idx) acc)))

                           :else
                           (do
                             (.append acc (str "<|turn>" (name role) "\n" clean-content "<turn|>\n"))
                             (recur (inc idx) acc))))))
         model-prefix (if (and (= last-role :tool) (not= syntax :xml) (not= syntax :fenced))
                        ""
                        "<|turn>model\n")]
     (str "<bos>" system-turn turns-str model-prefix))))

;; =============================================================================
;; 6. SCI Evaluation Sandboxes
;; =============================================================================

(defn create-benchmark-sci-ctx
  "Creates a hermetic, deterministic SCI sandbox context for benchmarking and grading.
   Strictly isolated: math and core pure Clojure logic only.
   No file I/O (slurp, spit, list-files), no system inspection (system-info, System/), and no reflection."
  []
  (sci/init
   {:classes {'Math Math}
    :bindings {'println println
               'print print
               'prn prn
               'str str}}))

(defn create-agent-sci-ctx
  "Creates a safe SCI sandbox context for the general coding agent, populated with safe file and system introspection helpers."
  []
  (sci/init
   {:classes {'Math Math}
    :bindings {'println println
               'print print
               'prn prn
               'str str
               'slurp (fn [f] (try (slurp f) (catch Exception e (str "Error reading file: " (.getMessage e)))))
               'spit (fn [f c] (try (spit f c) (str "Successfully wrote to " f) (catch Exception e (str "Error writing file: " (.getMessage e)))))
               'list-files (fn [dir]
                             (try
                               (let [d (io/file dir)]
                                 (mapv (fn [^java.io.File f]
                                         {:name (.getName f)
                                          :dir? (.isDirectory f)
                                          :size (.length f)})
                                       (.listFiles d)))
                               (catch Exception e (str "Error listing files: " (.getMessage e)))))
               'system-info (fn []
                              {:os (System/getProperty "os.name")
                               :arch (System/getProperty "os.arch")
                               :java (System/getProperty "java.version")
                               :cpus (.availableProcessors (Runtime/getRuntime))
                               :free-mem (.freeMemory (Runtime/getRuntime))})}}))

(defn eval-tool-code
  "Evaluates `code-str` in the SCI sandbox with an execution timeout guard and returns formatted execution result."
  ([sci-ctx code-str]
   (eval-tool-code sci-ctx code-str 5000))
  ([sci-ctx code-str timeout-ms]
   (try
     (let [clean-code (str/trim code-str)
           out-writer (java.io.StringWriter.)
           limit (long (or timeout-ms 5000))
           eval-fn (fn []
                     (binding [*out* out-writer]
                       (sci/eval-string* sci-ctx clean-code)))
           f (future (try (eval-fn) (catch Throwable t t)))
           res (deref f limit :timeout)]
       (if (= res :timeout)
         (do
           (future-cancel f)
           {:status :error
            :output (format "Execution Exception: Tool evaluation timed out after %d ms (potential infinite loop)." limit)})
         (if (instance? Throwable res)
           (throw res)
           (let [printed (str out-writer)
                 formatted-res (if (seq printed)
                                 (str printed "\n=> " (pr-str res))
                                 (pr-str res))]
             {:status :success :output formatted-res}))))
     (catch Throwable e
       {:status :error :output (str "Execution Exception: " (.getMessage e) " -- Please output valid Clojure s-expressions.")}))))

;; =============================================================================
;; 7. Telemetry Reporting
;; =============================================================================

(defn print-and-save-telemetry!
  "Prints multi-turn telemetry report including prompt/new token counts, decode speeds, and latencies,
   and writes summary EDN to `profile-out` if specified."
  [turn-telemetry loop-start-t quiet? profile-out]
  (let [telemetry-vec @turn-telemetry
        total-loop-ms (/ (- (System/nanoTime) loop-start-t) 1e6)
        total-model-ms (reduce + 0.0 (map :model-ms telemetry-vec))
        total-tool-ms (reduce + 0.0 (map :tool-ms telemetry-vec))
        total-prompt-tokens (reduce + 0 (map :prompt-tokens telemetry-vec))
        total-new-tokens (reduce + 0 (map :new-tokens telemetry-vec))
        avg-tok-per-sec (if (pos? total-model-ms)
                          (/ (* total-new-tokens 1000.0) total-model-ms)
                          0.0)]
    (when-not quiet?
      (println "\n==================================================")
      (println "=== Gemma 4 Agent Multi-Turn Telemetry Report ===")
      (println "==================================================")
      (doseq [{:keys [turn prompt-tokens new-tokens tok-per-sec model-ms tool-ms total-turn-ms]} telemetry-vec]
        (println (format "  Turn %d: Prompt=%4d tok | Gen=%4d tok (%5.1f tok/s) | Model=%8.2f ms | Tool=%8.2f ms | Turn Total=%8.2f ms"
                         turn (or prompt-tokens 0) (or new-tokens 0) (or tok-per-sec 0.0)
                         (or model-ms 0.0) (or tool-ms 0.0) (or total-turn-ms 0.0))))
      (println "--------------------------------------------------")
      (println (format "  Total Turns           : %d" (count telemetry-vec)))
      (println (format "  Total Prompt Tokens   : %d tok" total-prompt-tokens))
      (println (format "  Total Generated Tokens: %d tok" total-new-tokens))
      (println (format "  Average Decode Speed  : %5.1f tok/s" avg-tok-per-sec))
      (println (format "  Total Model Inference : %8.2f ms" total-model-ms))
      (println (format "  Total Tool Execution  : %8.2f ms" total-tool-ms))
      (println (format "  Total Agent Session   : %8.2f ms" total-loop-ms))
      (println "=================================================="))
    (when (seq profile-out)
      (spit profile-out (pr-str {:turns telemetry-vec
                                 :total-turns (count telemetry-vec)
                                 :total-prompt-tokens total-prompt-tokens
                                 :total-new-tokens total-new-tokens
                                 :avg-tok-per-sec avg-tok-per-sec
                                 :total-model-ms total-model-ms
                                 :total-tool-ms total-tool-ms
                                 :total-session-ms total-loop-ms})))))

;; =============================================================================
;; 8. Autonomous Multi-Turn Agent Loop (Optimizations 3 & 4)
;; =============================================================================

(defn nudge-needed?
  "Determines whether the blind nudge for Turn 2 should be emitted.
   Optimization 3 (Nudge Short-Circuit): Returns false if candidate code defining fn-name
   is already present in the Turn 1 reply."
  [{:keys [turn max-turns new-tokens opts history consecutive-errors has-candidate?]}]
  (and (< turn max-turns)
       (pos? new-tokens)
       (:nudge-on-no-tool opts)
       (not has-candidate?)
       (not (some #(= (:role %) :tool) history))
       (not (some #(and (= (:role %) :user)
                        (str/includes? (or (:content %) "") "Please test your Clojure implementation"))
                  history))
       (zero? consecutive-errors)))

(defn run-agent-loop
  "Runs autonomous agent loop with SCI Clojure tool calling across multiple turns.
   Implements:
   - Optimization 3: Nudge short-circuit (skips blind nudge if candidate code is present)
   - Optimization 4: Early exit on public test pass (halts immediately when tool returns early-exit? true)."
  ([session initial-prompt]
   (run-agent-loop session initial-prompt nil nil))
  ([session initial-prompt custom-sci-ctx]
   (run-agent-loop session initial-prompt custom-sci-ctx nil))
  ([session initial-prompt custom-sci-ctx custom-tool-eval-fn]
   (let [{:keys [opts]} session
         {:keys [system max-turns out quiet profile-out thinking tool-declaration max-consecutive-errors sandbox tool-eval-fn candidate-check-fn tool-syntax]} opts
         syntax (keyword (or tool-syntax :native))
         tool-eval (or custom-tool-eval-fn tool-eval-fn eval-tool-code)
         candidate-fn (or candidate-check-fn (constantly false))
         thinking? (boolean thinking)
         tool-decl (if (or (= syntax :xml) (= syntax :fenced))
                     nil
                     (or tool-declaration DEFAULT-TOOL-DECLARATION))
         sys-prompt (or system
                        (case syntax
                          :xml DEFAULT-SYSTEM-PROMPT-XML
                          :fenced DEFAULT-SYSTEM-PROMPT-FENCED
                          DEFAULT-SYSTEM-PROMPT))
         sci-ctx (or custom-sci-ctx
                     (case sandbox
                       :benchmark (create-benchmark-sci-ctx)
                       (create-agent-sci-ctx)))
         consecutive-error-limit (long (or max-consecutive-errors 3))
         history (atom [{:role :user :content initial-prompt}])
         transcript (atom [])
         turn-telemetry (atom [])
         loop-start-t (System/nanoTime)
         scripted-atom (when-let [scripted (:scripted-responses opts)]
                         (atom (vec scripted)))
         gen-fn (or (:generate-fn opts)
                    (when scripted-atom
                      (fn [_sess _prompt]
                        (let [resp (first @scripted-atom)]
                          (when (seq @scripted-atom)
                            (swap! scripted-atom subvec 1))
                          (if (map? resp)
                            resp
                            {:text (or resp "") :prompt-tokens 10 :new-tokens 10}))))
                    ;; Dynamically resolve gemma4-inf to keep einsum.agent.core decoupled
                    (when-let [res-fn (try (requiring-resolve 'einsum.models.gemma4.runtime/generate-new-tokens-and-text)
                                           (catch Throwable _ nil))]
                      res-fn)
                    (fn [sess prompt]
                      (let [f (requiring-resolve 'tools.gemma4-inference/generate-new-tokens-and-text)]
                        (f sess prompt))))]
     (loop [turn 1
            consecutive-errors 0]
       (if (> turn max-turns)
         (do
           (when-not quiet (println (format "\n[Agent] Reached max-turns limit (%d)." max-turns)))
           (print-and-save-telemetry! turn-telemetry loop-start-t quiet profile-out)
           (when (seq out)
             (spit out (str/join "\n\n" (map :content @transcript)))
             (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
           (with-meta @transcript {:turn-telemetry @turn-telemetry :history @history}))
         (do
           (when-not quiet (println "\n=================================================="))
           (when-not quiet (println (format "=== Agent Turn %d/%d ===" turn max-turns)))
           (when-not quiet (println "=================================================="))
           (let [formatted-prompt (format-agent-chat-prompt sys-prompt @history 8 thinking? tool-decl syntax)
                 _ (when-not quiet (println "Executing Gemma 4 Agent Forward Pass..."))
                 t-gen-0 (System/nanoTime)
                 gen-res (gen-fn session formatted-prompt)
                 t-gen-1 (System/nanoTime)
                 gen-ms (/ (- t-gen-1 t-gen-0) 1e6)
                 new-gen (:text gen-res)
                 prompt-tokens (long (or (:prompt-tokens gen-res) 0))
                 new-tokens (long (or (:new-tokens gen-res) 0))
                 tok-per-sec (if (pos? gen-ms) (/ (* new-tokens 1000.0) gen-ms) 0.0)
                 model-reply (str/trim (str/replace (or new-gen "") #"<bos>|<eos>|<turn\|>|<\|turn>" ""))
                 thinking-trace (extract-thinking-trace model-reply)
                 final-response (strip-thinking-trace model-reply)
                 tool-call (extract-tool-call model-reply syntax)]

             (when (and thinking? (seq thinking-trace) (not quiet))
               (println "\n--------------------------------------------------")
               (println "[Agent Thought Process]:")
               (println thinking-trace)
               (println "--------------------------------------------------"))

             (let [history-content (sanitize-history-model-content tool-call final-response model-reply)]
               (swap! history conj {:role :model :content history-content}))

             (swap! transcript conj {:turn turn
                                     :role :model
                                     :content (if (seq thinking-trace)
                                                (str "[Thought Process]\n" thinking-trace "\n\n[Model Response]\n" final-response)
                                                model-reply)
                                     :thought thinking-trace
                                     :response final-response
                                     :raw model-reply
                                     :tool-call tool-call})

             (if-not tool-call
               (let [has-candidate? (boolean (candidate-fn final-response))]
                 (if (nudge-needed? {:turn turn
                                     :max-turns max-turns
                                     :new-tokens new-tokens
                                     :opts opts
                                     :history @history
                                     :consecutive-errors consecutive-errors
                                     :model-reply final-response
                                     :has-candidate? has-candidate?})
                   (let [nudge-msg (or (:nudge-prompt opts)
                                       (case syntax
                                         :xml "Please test your Clojure implementation by outputting code in <clojure>...</clojure>."
                                         :fenced "Please test your Clojure implementation by outputting code in ```clojure ... ```."
                                         "Please test your Clojure implementation by calling the eval_clojure tool."))]
                     (when-not quiet (println (format "\n[Agent Loop Nudge (Turn %d)]: Emitting tool reminder." turn)))
                     (swap! history conj {:role :user :content nudge-msg})
                     (swap! transcript conj {:turn turn :role :user :content nudge-msg})
                     (recur (inc turn) consecutive-errors))
                   (do
                     (swap! turn-telemetry conj {:turn turn
                                                 :prompt-tokens prompt-tokens
                                                 :new-tokens new-tokens
                                                 :tok-per-sec tok-per-sec
                                                 :model-ms gen-ms
                                                 :tool-ms 0.0
                                                 :total-turn-ms gen-ms})
                     (when-not quiet
                       (when (seq final-response)
                         (println "\n[Agent Response]:")
                         (println final-response))
                       (if has-candidate?
                         (println "\n[Agent] Candidate code present in response. Nudge short-circuited!")
                         (println "\n[Agent] No further tool calls requested. Task completed!")))
                     (print-and-save-telemetry! turn-telemetry loop-start-t quiet profile-out)
                     (when (seq out)
                       (spit out (str/join "\n\n" (map :content @transcript)))
                       (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
                     (with-meta @transcript {:turn-telemetry @turn-telemetry :history @history}))))

               (if (>= consecutive-errors consecutive-error-limit)
                 ;; Model attempted another tool call after error budget was exhausted
                 (do
                   (when-not quiet
                     (println (format "\n[Agent] Consecutive tool error budget exhausted (%d/%d consecutive errors). Halting loop."
                                      consecutive-errors consecutive-error-limit)))
                   (swap! turn-telemetry conj {:turn turn
                                               :prompt-tokens prompt-tokens
                                               :new-tokens new-tokens
                                               :tok-per-sec tok-per-sec
                                               :model-ms gen-ms
                                               :tool-ms 0.0
                                               :total-turn-ms gen-ms})
                   (print-and-save-telemetry! turn-telemetry loop-start-t quiet profile-out)
                   (when (seq out)
                     (spit out (str/join "\n\n" (map :content @transcript)))
                     (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
                   (with-meta @transcript {:turn-telemetry @turn-telemetry :history @history}))

                 (let [tool-code (:code tool-call)
                       tool-name (:name tool-call)
                       _ (when-not quiet
                           (println "\n--------------------------------------------------")
                           (println (format "[Agent Tool Call (%s -> SCI Clojure)]:" tool-name))
                           (println (or tool-code "<no code>"))
                           (println "--------------------------------------------------"))
                       t-tool-0 (System/nanoTime)
                       eval-res (if (seq tool-code)
                                  (try
                                    (tool-eval sci-ctx tool-code turn)
                                    (catch clojure.lang.ArityException e
                                      (if (= 3 (.actual e))
                                        (tool-eval sci-ctx tool-code)
                                        (throw e))))
                                  {:status :error :output "Error: No code provided to eval_clojure."})
                       t-tool-1 (System/nanoTime)
                       tool-ms (/ (- t-tool-1 t-tool-0) 1e6)
                       turn-total-ms (+ gen-ms tool-ms)
                       new-consecutive-errors (if (= (:status eval-res) :error)
                                                (inc consecutive-errors)
                                                0)
                       error-budget-reached? (and (= (:status eval-res) :error)
                                                  (>= new-consecutive-errors consecutive-error-limit))
                       early-exit? (boolean (:early-exit? eval-res))
                       augmented-res (if error-budget-reached?
                                       (assoc eval-res :system_note
                                              (format "Consecutive tool error limit (%d) reached. Tool execution is now disabled. Provide your final answer in plain text based on the observations collected so far without calling further tools."
                                                      consecutive-error-limit))
                                       eval-res)
                       obs-str (format-tool-response tool-name augmented-res syntax)]

                   (swap! turn-telemetry conj {:turn turn
                                               :prompt-tokens prompt-tokens
                                               :new-tokens new-tokens
                                               :tok-per-sec tok-per-sec
                                               :model-ms gen-ms
                                               :tool-ms tool-ms
                                               :total-turn-ms turn-total-ms})
                   (when-not quiet
                     (println "\n[Tool Observation Output]:")
                     (println (:output eval-res))
                     (println (format "  ↳ [Turn %d Latency: Model=%.2f ms (%d tok, %.1f tok/s), Tool=%.2f ms, Total=%.2f ms]"
                                      turn gen-ms new-tokens tok-per-sec tool-ms turn-total-ms)))

                   (swap! history conj {:role :tool :content obs-str})
                   (swap! transcript conj {:turn turn :role :tool :content obs-str})
                   (when (seq out)
                     (spit out (str/join "\n\n" (map :content @transcript))))

                   (if early-exit?
                     (do
                       (when-not quiet
                         (println (format "\n[Agent Loop Early Exit (Turn %d)]: All public tests passed! Terminating loop." turn)))
                       (print-and-save-telemetry! turn-telemetry loop-start-t quiet profile-out)
                       (with-meta @transcript {:turn-telemetry @turn-telemetry
                                               :history @history
                                               :early-exit? true
                                               :early-exit-reason :public-pass}))
                     (recur (inc turn) new-consecutive-errors))))))))))))
