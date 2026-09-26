(ns tools.gemma4-agent
  "Autonomous software architecture agent loop powered by Gemma 4, XLA execution, and SCI Clojure tool calling."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [tools.cli :as cli]
            [tools.gemma4-inference :as gemma4-inf]
            [sci.core :as sci]))

(def DEFAULT_TOOL_DECLARATION
  "<|tool>declaration:eval_clojure{description:<|\"|>Execute Clojure code in the SCI Clojure sandbox and return the evaluation result. Code must be valid Clojure s-expressions.<|\"|>,parameters:{properties:{code:{description:<|\"|>The Clojure code to evaluate.<|\"|>,type:<|\"|>string<|\"|>}},required:[<|\"|>code<|\"|>],type:<|\"|>object<|\"|>}}<tool|>")

(def DEFAULT_SYSTEM_PROMPT
  "You are a Clojure assistant with access to the eval_clojure tool.
Call eval_clojure to execute or test Clojure code. Once you have the result, provide your final response to the user in plain text without further tool calls.
Syntax rules: use square brackets for bindings and parameters: [x], [k v], vectors: [1 2 3], and (range start end).")

(def DEFAULT_AGENT_OPTS
  {:prompt "Write a Clojure function returning the first 10 integers."
   :model ".models/gemma-4-E2B-it"
   :system DEFAULT_SYSTEM_PROMPT
   :tool-declaration DEFAULT_TOOL_DECLARATION
   :max-new-tokens 512
   :max-seq-len 1024
   :max-turns 5
   :max-consecutive-errors 3
   :sandbox :agent
   :temperature 0.0
   :top-k 10
   :repetition-penalty 1.15
   :method nil
   :backend :rocm
   :precision :bf16
   :thinking false
   :out "scratch/output_agent_loop.txt"
   :profile-out "scratch/gemma4_agent_profile.edn"
   :chrome-trace-out "scratch/gemma4_agent_chrome_trace.json"
   :quiet false})

(def ^:private default-reader-ctx (sci/init {}))

(defn- try-parse-sci-reader
  "Attempts to parse the first form from string `s` using SCI's reader.
   If parsing fails due to EOF (e.g. truncated closing parens), tries appending closing parens."
  [s]
  (try
    (second (sci/parse-next+string default-reader-ctx (sci/source-reader s)))
    (catch Throwable _
      (loop [depth 1]
        (when (<= depth 8)
          (if-let [res (try
                         (second (sci/parse-next+string default-reader-ctx
                                                        (sci/source-reader (str s (apply str (repeat depth ")"))))))
                         (catch Throwable _ nil))]
            res
            (recur (inc depth))))))))

(defn extract-balanced-sexpr
  "Finds the first balanced s-expression string starting with `(` in text using SCI's reader.
   Correctly handles strings with parentheses and comments without naive paren counting."
  [text]
  (when (string? text)
    (loop [offset 0]
      (let [start (.indexOf ^String text "(" offset)]
        (when (>= start 0)
          (let [candidate (subs text start)]
            (if-let [parsed (try-parse-sci-reader candidate)]
              parsed
              (recur (inc start)))))))))

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
   Employs reader-based balanced extraction to prevent greedy swallowing of subsequent fields."
  [args-str]
  (let [trimmed (str/trim (or args-str ""))
        unbraced (-> trimmed
                     (str/replace #"^\{" "")
                     (str/replace #"\}$" "")
                     str/trim)]
    (cond
      ;; 1. Gemma 4 native string delimiter: code:<|"|>...<|"|>
      (re-find #"(?s)code\s*:\s*<\|\"\|>(.*?)(?:<\|\"\|>|$)" unbraced)
      (second (re-find #"(?s)code\s*:\s*<\|\"\|>(.*?)(?:<\|\"\|>|$)" unbraced))

      ;; 2. Standard JSON quotes with escaped string support: code:"..."
      (re-find #"(?s)code\s*:\s*\"([^\"\\]*(?:\\.[^\"\\]*)*)\"" unbraced)
      (unescape-json-string (second (re-find #"(?s)code\s*:\s*\"([^\"\\]*(?:\\.[^\"\\]*)*)\"" unbraced)))

      ;; 3. Single quotes: code:'...'
      (re-find #"(?s)code\s*:\s*\'([^\']*)\'" unbraced)
      (second (re-find #"(?s)code\s*:\s*\'([^\']*)\'" unbraced))

      ;; 4. code: followed by an s-expression
      (re-find #"(?s)code\s*:\s*\(" unbraced)
      (let [code-idx (.indexOf ^String unbraced "code:")
            after-code (subs unbraced (+ code-idx 5))]
        (extract-balanced-sexpr after-code))

      ;; 5. Any bare s-expression within the tool call argument block (Reader-based balanced extractor)
      (extract-balanced-sexpr unbraced)
      (extract-balanced-sexpr unbraced)

      :else nil)))

(defn extract-tool-call
  "Extracts the tool call from text if present.
   Returns a map {:name fn-name :code clojure-code-str :raw raw-tool-call-str} or nil if text is a plain text response or malformed."
  [text]
  (let [clean (strip-thinking-trace (or text ""))]
    (when-let [[raw fn-name args] (re-find #"(?s)<\|tool_call>(?:call:)?(\w+)?\s*(.*?)(?:<tool_call\|>|$)" clean)]
      (when-let [code (parse-tool-call-code args)]
        {:name (if (seq fn-name) fn-name "eval_clojure")
         :code (str/trim code)
         :raw (if (str/ends-with? raw "<tool_call|>")
                raw
                (str raw "<tool_call|>"))}))))

(defn format-tool-response
  "Formats the SCI execution result as a native Gemma 4 tool response observation."
  ([res]
   (format-tool-response "eval_clojure" res))
  ([tool-name res]
   (let [content (str/trim (or (:output res) ""))]
     (format "<|tool_response>response:%s{output:<|\"|>%s<|\"|>}<tool_response|>"
             (if (seq tool-name) tool-name "eval_clojure")
             content))))

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` or ```clj ... ``` code block strings, Gemma 4 <|tool_call> tags, or raw S-expressions from text.
   Maintained for backward compatibility."
  [text]
  (let [extract-raw (fn [s allow-loose?]
                      (let [pattern #"(?s)```(?:clojure|clj)?\s*\n?(.*?)(?:```|$)"
                            raw-matches (mapv str/trim (filter #(seq (str/trim %)) (mapv second (re-seq pattern s))))
                            cleaned-blocks (mapv (fn [block]
                                                   (-> block
                                                       (str/replace #"^```[a-z]*>?" "")
                                                       (str/replace #"```$" "")
                                                       str/trim))
                                                 raw-matches)
                            tool-call-pattern #"(?s)<\|tool_call>call:(\w+)(.*?)(?:<tool_call\|>|$)"
                            tool-call-matches (mapv (fn [[_ fn-name args]]
                                                      (let [clean-args (str/trim (str/replace args #"^\{|\}$" ""))]
                                                        (if (seq clean-args)
                                                          (str "(" fn-name " " clean-args ")")
                                                          (str "(" fn-name ")"))))
                                                    (re-seq tool-call-pattern s))]
                        (cond
                          (seq cleaned-blocks) (vec cleaned-blocks)
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
  "Evaluates `code-str` in the SCI sandbox and returns formatted execution result."
  [sci-ctx code-str]
  (try
    (let [clean-code (str/trim code-str)
          out-writer (java.io.StringWriter.)
          eval-res (binding [*out* out-writer]
                     (sci/eval-string* sci-ctx clean-code))
          printed (str out-writer)
          formatted-res (if (seq printed)
                          (str printed "\n=> " (pr-str eval-res))
                          (pr-str eval-res))]
      {:status :success :output formatted-res})
    (catch Throwable e
      {:status :error :output (str "Execution Exception: " (.getMessage e) " -- Please output valid Clojure s-expressions.")})))

(defn parse-agent-cli-args
  "Parses CLI flags for gemma4_agent."
  [args]
  (let [opts (cli/parse-cli-args args DEFAULT_AGENT_OPTS)]
    (cond-> opts
      (string? (:max-consecutive-errors opts))
      (update :max-consecutive-errors #(Long/parseLong %))
      (string? (:sandbox opts))
      (update :sandbox #(keyword (str/replace % #"^:+" ""))))))

(defn format-agent-chat-prompt
  "Formats conversation history into Gemma 4 Turn syntax, placing tool declarations and system instructions in native Gemma 4 turns.
   Applies sliding window context retention for long histories to preserve the initial task and alternating turns.
   When `thinking?` is enabled, injects the native Gemma 4 `<|think|>` token into the system turn."
  ([system-prompt history]
   (format-agent-chat-prompt system-prompt history 8 false nil))
  ([system-prompt history opts-or-max-turns]
   (cond
     (map? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history
                               (long (get opts-or-max-turns :max-recent-turns 8))
                               (boolean (or (:thinking opts-or-max-turns) (:thinking? opts-or-max-turns)))
                               (get opts-or-max-turns :tool-declaration))

     (boolean? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history 8 opts-or-max-turns nil)

     :else
     (format-agent-chat-prompt system-prompt history (long opts-or-max-turns) false nil)))
  ([system-prompt history max-recent-turns thinking?]
   (format-agent-chat-prompt system-prompt history max-recent-turns thinking? nil))
  ([system-prompt history max-recent-turns thinking? tool-declaration]
   (let [history-vec (vec history)
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
         tool-decl-str (when (seq tool-declaration)
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
                             clean-content (str/trim (or content ""))
                             prev-role (when (pos? idx) (:role (nth trimmed (dec idx))))
                             next-role (when (< (inc idx) (count trimmed))
                                         (:role (nth trimmed (inc idx))))]
                         (cond
                           (= role :user)
                           (do
                             (.append acc "<|turn>user\n")
                             (.append acc clean-content)
                             (.append acc "<turn|>\n")
                             (recur (inc idx) acc))

                           (= role :model)
                           (do
                             (when-not (= prev-role :tool)
                               (.append acc "<|turn>model\n"))
                             (.append acc clean-content)
                             (if (= next-role :tool)
                               ;; Native Gemma 4 in-flow tool interaction: do not close model turn
                               (recur (inc idx) acc)
                               (do
                                 (.append acc "<turn|>\n")
                                 (recur (inc idx) acc))))

                           (= role :tool)
                           (do
                             ;; Native Gemma 4 tool response directly attaches to model tool call
                             (.append acc clean-content)
                             (if (str/ends-with? clean-content "\n")
                               nil
                               (.append acc "\n"))
                             (recur (inc idx) acc))

                           :else
                           (do
                             (.append acc (str "<|turn>" (name role) "\n" clean-content "<turn|>\n"))
                             (recur (inc idx) acc))))))
         model-prefix (if (= last-role :tool)
                        ""
                        "<|turn>model\n")]
     (str "<bos>" system-turn turns-str model-prefix))))

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

(defn run-agent-loop
  "Runs autonomous agent loop with SCI Clojure tool calling across multiple turns."
  ([session initial-prompt]
   (run-agent-loop session initial-prompt nil nil))
  ([session initial-prompt custom-sci-ctx]
   (run-agent-loop session initial-prompt custom-sci-ctx nil))
  ([session initial-prompt custom-sci-ctx custom-tool-eval-fn]
   (let [{:keys [opts]} session
         {:keys [system max-turns out quiet profile-out thinking tool-declaration max-consecutive-errors sandbox tool-eval-fn]} opts
         tool-eval (or custom-tool-eval-fn tool-eval-fn eval-tool-code)
         thinking? (boolean thinking)
         tool-decl (or tool-declaration DEFAULT_TOOL_DECLARATION)
         sci-ctx (or custom-sci-ctx
                     (case sandbox
                       :benchmark (create-benchmark-sci-ctx)
                       (create-agent-sci-ctx)))
         consecutive-error-limit (long (or max-consecutive-errors 3))
         history (atom [{:role :user :content initial-prompt}])
         transcript (atom [])
         turn-telemetry (atom [])
         loop-start-t (System/nanoTime)]
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
           (let [formatted-prompt (format-agent-chat-prompt system @history 8 thinking? tool-decl)
                 _ (when-not quiet (println "Executing Gemma 4 Agent Forward Pass..."))
                 t-gen-0 (System/nanoTime)
                 gen-res (gemma4-inf/generate-new-tokens-and-text session formatted-prompt)
                 t-gen-1 (System/nanoTime)
                 gen-ms (/ (- t-gen-1 t-gen-0) 1e6)
                 new-gen (:text gen-res)
                 prompt-tokens (long (or (:prompt-tokens gen-res) 0))
                 new-tokens (long (or (:new-tokens gen-res) 0))
                 tok-per-sec (if (pos? gen-ms) (/ (* new-tokens 1000.0) gen-ms) 0.0)
                 model-reply (str/trim (str/replace (or new-gen "") #"<bos>|<eos>|<turn\|>|<\|turn>" ""))
                 thinking-trace (extract-thinking-trace model-reply)
                 final-response (strip-thinking-trace model-reply)
                 tool-call (extract-tool-call model-reply)]

             (when (and thinking? (seq thinking-trace) (not quiet))
               (println "\n--------------------------------------------------")
               (println "[Agent Thought Process]:")
               (println thinking-trace)
               (println "--------------------------------------------------"))

             (let [history-content (if tool-call
                                     (:raw tool-call)
                                     (if (seq final-response) final-response model-reply))]
               (swap! history conj {:role :model :content history-content}))

             (swap! transcript conj {:turn turn
                                     :role :model
                                     :content (if (seq thinking-trace)
                                                (str "[Thought Process]\n" thinking-trace "\n\n[Model Response]\n" final-response)
                                                model-reply)
                                     :thought thinking-trace})

             (if-not tool-call
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
                   (println "\n[Agent] No further tool calls requested. Task completed!"))
                 (print-and-save-telemetry! turn-telemetry loop-start-t quiet profile-out)
                 (when (seq out)
                   (spit out (str/join "\n\n" (map :content @transcript)))
                   (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
                 (with-meta @transcript {:turn-telemetry @turn-telemetry :history @history}))

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
                                  (tool-eval sci-ctx tool-code)
                                  {:status :error :output "Error: No code provided to eval_clojure."})
                       t-tool-1 (System/nanoTime)
                       tool-ms (/ (- t-tool-1 t-tool-0) 1e6)
                       turn-total-ms (+ gen-ms tool-ms)
                       new-consecutive-errors (if (= (:status eval-res) :error)
                                                (inc consecutive-errors)
                                                0)
                       error-budget-reached? (and (= (:status eval-res) :error)
                                                  (>= new-consecutive-errors consecutive-error-limit))
                       base-obs (format-tool-response tool-name eval-res)
                       obs-str (if error-budget-reached?
                                 (str base-obs
                                      (format "\n[System: Consecutive tool error limit (%d) reached. Tool execution is now disabled. Provide your final answer in plain text based on the observations collected so far without calling further tools.]"
                                              consecutive-error-limit))
                                 base-obs)]

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
                   (recur (inc turn) new-consecutive-errors)))))))))))

(defn -main
  "CLI Entrypoint for Gemma 4 Agent."
  [& args]
  (try
    (let [opts (parse-agent-cli-args args)
          model-dir (cli/find-model-dir (or (:model-dir opts) (:model opts)) :gemma-4)
          max-seq-len (long (or (:max-seq-len opts) 1024))
          opts (assoc opts :model-dir model-dir :model model-dir :mode :agent :max-seq-len max-seq-len)
          metrics-atom (atom {})
          trace-spans-atom (atom [])
          session (assoc (gemma4-inf/init-agent-vram-session opts max-seq-len)
                         :metrics-atom metrics-atom
                         :trace-spans-atom trace-spans-atom)]
      (try
        (run-agent-loop session (:prompt opts))
        (finally
          (gemma4-inf/close-agent-session! session))))
    (catch Throwable e
      (println "\nAgent Exception:" (.getMessage e))
      (.printStackTrace e))
    (finally
      (.. Runtime getRuntime (halt 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
