(ns tools.gemma4-agent
  "Autonomous software architecture agent loop powered by Gemma 4, XLA execution, and SCI Clojure tool calling."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [tools.cli :as cli]
            [tools.gemma4-inference :as gemma4-inf]
            [sci.core :as sci]))

(def DEFAULT_SYSTEM_PROMPT
  "You are an autonomous Clojure engineering assistant with access to a live SCI Clojure sandbox environment.
Solve programming tasks methodically:
1. When you need to test, run, or verify code, output executable Clojure s-expressions enclosed in a single ```clojure ... ``` code block.
2. The environment will execute your code in SCI and return the evaluation result in a Tool Execution Observation.
3. If an error occurs, inspect the error message and output revised Clojure code in a ```clojure ... ``` block to fix it.
4. When the observation demonstrates that your code has successfully solved the task, provide your final response to the user as plain text explaining the solution, without any code blocks. Never use JavaScript or Python.

Clojure syntax rules:
- Always use square brackets for parameter lists and bindings: `(defn f [x] ...)`, `(fn [x] ...)`, `(let [x 1] ...)`, `(loop [i 0] ...)`. Never use `lambda` or double parentheses `((var val))`.
- For vectors, use `[...]`, `(vec ...)`, or `(filterv ...)`.
- To generate integer sequences, use `(range start end)` or `(range n)`.")

(def DEFAULT_AGENT_OPTS
  {:prompt "Write a Clojure function returning the first 10 integers."
   :system DEFAULT_SYSTEM_PROMPT
   :max-new-tokens 256
   :max-seq-len 1024
   :max-turns 5
   :temperature 0.0
   :top-k 10
   :repetition-penalty 1.15
   :method nil
   :backend :cpu
   :precision :bf16
   :thinking false
   :out "scratch/output_agent_loop.txt"
   :profile-out "scratch/gemma4_agent_profile.edn"
   :chrome-trace-out "scratch/gemma4_agent_chrome_trace.json"
   :quiet false})

(defn extract-balanced-sexpr
  "Finds the first balanced s-expression string starting with `(` in text."
  [text]
  (let [start (.indexOf ^String text "(")]
    (when (>= start 0)
      (loop [i start depth 0]
        (if (>= i (count text))
          (when (pos? depth) (str (subs text start) (apply str (repeat depth ")"))))
          (let [ch (.charAt ^String text i)]
            (cond
              (= ch \() (recur (inc i) (inc depth))
              (= ch \)) (if (= depth 1)
                          (subs text start (inc i))
                          (recur (inc i) (dec depth)))
              :else (recur (inc i) depth))))))))

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

(defn extract-clojure-code-blocks
  "Extracts all ```clojure ... ``` or ```clj ... ``` code block strings, Gemma 4 <|tool_call> tags, or raw S-expressions from text."
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

(defn create-agent-sci-ctx
  "Creates a safe SCI sandbox context populated with useful Clojure agent helper functions."
  []
  (sci/init
   {:bindings {'println println
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
  (cli/parse-cli-args args DEFAULT_AGENT_OPTS))

(defn format-agent-chat-prompt
  "Formats conversation history into Gemma 4 Turn syntax, placing system instructions in a native system turn.
   Applies sliding window context retention for long histories to preserve the initial task and alternating turns.
   When `thinking?` is enabled, injects the native Gemma 4 `<|think|>` token into the system turn."
  ([system-prompt history]
   (format-agent-chat-prompt system-prompt history 8 false))
  ([system-prompt history opts-or-max-turns]
   (cond
     (map? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history
                               (long (get opts-or-max-turns :max-recent-turns 8))
                               (boolean (or (:thinking opts-or-max-turns) (:thinking? opts-or-max-turns))))

     (boolean? opts-or-max-turns)
     (format-agent-chat-prompt system-prompt history 8 opts-or-max-turns)

     :else
     (format-agent-chat-prompt system-prompt history (long opts-or-max-turns) false)))
  ([system-prompt history max-recent-turns thinking?]
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
         already-has-think? (and has-system? (str/includes? clean-system "<|think|>"))
         thinking-needed? (and thinking? (not already-has-think?))
         system-content (cond
                          (and has-system? thinking-needed?)
                          (str "<|think|>\n" clean-system)

                          has-system?
                          clean-system

                          thinking-needed?
                          "<|think|>"

                          :else nil)
         system-turn (when system-content
                       (str "<|turn>system\n" system-content "<turn|>\n"))
         turns (mapv (fn [{:keys [role content]}]
                       (str "<|turn>" (name role) "\n" (str/trim content) "<turn|>\n"))
                     trimmed)]
     (str "<bos>" system-turn (str/join "" turns) "<|turn>model\n"))))

(defn run-agent-loop
  "Runs autonomous agent loop with SCI Clojure tool calling across multiple turns."
  [session initial-prompt]
  (let [{:keys [opts]} session
        {:keys [system max-turns out quiet profile-out thinking]} opts
        thinking? (boolean thinking)
        sci-ctx (create-agent-sci-ctx)
        history (atom [{:role :user :content initial-prompt}])
        transcript (atom [])
        turn-telemetry (atom [])
        loop-start-t (System/nanoTime)]
    (loop [turn 1]
      (if (> turn max-turns)
        (do
          (when-not quiet (println (format "\n[Agent] Reached max-turns limit (%d)." max-turns)))
          (when (seq out)
            (spit out (str/join "\n\n" (map :content @transcript)))
            (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
          @transcript)
        (do
          (when-not quiet (println "\n=================================================="))
          (when-not quiet (println (format "=== Agent Turn %d/%d ===" turn max-turns)))
          (when-not quiet (println "=================================================="))
          (let [formatted-prompt (format-agent-chat-prompt system @history 8 thinking?)
                _ (when-not quiet (println "Executing Gemma 4 Agent Forward Pass..."))
                t-gen-0 (System/nanoTime)
                full-gen (gemma4-inf/generate-text-string session formatted-prompt)
                t-gen-1 (System/nanoTime)
                gen-ms (/ (- t-gen-1 t-gen-0) 1e6)
                model-text (if (re-find #"<\|turn>model" full-gen)
                             (last (str/split full-gen #"<\|turn>model\r?\n?"))
                             full-gen)
                model-reply (str/trim (str/replace model-text #"<bos>|<eos>|<turn\|>|<\|turn>" ""))
                thinking-trace (extract-thinking-trace model-reply)
                final-response (strip-thinking-trace model-reply)
                code-blocks (extract-clojure-code-blocks model-reply)]

            (when (and thinking? (seq thinking-trace) (not quiet))
              (println "\n--------------------------------------------------")
              (println "[Agent Thought Process]:")
              (println thinking-trace)
              (println "--------------------------------------------------"))

            (swap! history conj {:role :model :content model-reply})
            (swap! transcript conj {:turn turn
                                    :role :model
                                    :content (if (seq thinking-trace)
                                               (str "[Thought Process]\n" thinking-trace "\n\n[Model Response]\n" final-response)
                                               model-reply)
                                    :thought thinking-trace})

            (if (empty? code-blocks)
              (do
                (swap! turn-telemetry conj {:turn turn :model-ms gen-ms :tool-ms 0.0 :total-turn-ms gen-ms})
                (when-not quiet
                  (when (seq final-response)
                    (println "\n[Agent Response]:")
                    (println final-response))
                  (println "\n[Agent] No further tool calls requested. Task completed!"))
                (let [total-loop-ms (/ (- (System/nanoTime) loop-start-t) 1e6)
                      total-model-ms (reduce + (map :model-ms @turn-telemetry))
                      total-tool-ms (reduce + (map :tool-ms @turn-telemetry))]
                  (when-not quiet
                    (println "\n==================================================")
                    (println "=== Gemma 4 Agent Multi-Turn Telemetry Report ===")
                    (println "==================================================")
                    (doseq [{:keys [turn model-ms tool-ms total-turn-ms]} @turn-telemetry]
                      (println (format "  Turn %d: Model=%8.2f ms | Tool=%8.2f ms | Turn Total=%8.2f ms"
                                       turn model-ms tool-ms total-turn-ms)))
                    (println "--------------------------------------------------")
                    (println (format "  Total Turns           : %d" (count @turn-telemetry)))
                    (println (format "  Total Model Inference : %8.2f ms" total-model-ms))
                    (println (format "  Total Tool Execution  : %8.2f ms" total-tool-ms))
                    (println (format "  Total Agent Session   : %8.2f ms" total-loop-ms))
                    (println "=================================================="))
                  (when (seq profile-out)
                    (spit profile-out (pr-str {:turns @turn-telemetry
                                               :total-turns (count @turn-telemetry)
                                               :total-model-ms total-model-ms
                                               :total-tool-ms total-tool-ms
                                               :total-session-ms total-loop-ms}))))
                (when (seq out)
                  (spit out (str/join "\n\n" (map :content @transcript)))
                  (when-not quiet (println (format "  ↳ Saved agent transcript to [%s]" out))))
                @transcript)

              (let [tool-code (first code-blocks)
                    _ (when-not quiet
                        (println "\n--------------------------------------------------")
                        (println "[Agent Tool Call (SCI Clojure)]:")
                        (println tool-code)
                        (println "--------------------------------------------------"))
                    t-tool-0 (System/nanoTime)
                    eval-res (eval-tool-code sci-ctx tool-code)
                    t-tool-1 (System/nanoTime)
                    tool-ms (/ (- t-tool-1 t-tool-0) 1e6)
                    turn-total-ms (+ gen-ms tool-ms)
                    obs-str (if (= (:status eval-res) :success)
                              (str "Tool Execution Observation:\n" (:output eval-res)
                                   "\n[Note: If this output answers the request, provide your final response in plain text without code blocks.]")
                              (str "Tool Execution Observation:\n" (:output eval-res)))]

                (swap! turn-telemetry conj {:turn turn :model-ms gen-ms :tool-ms tool-ms :total-turn-ms turn-total-ms})
                (when-not quiet
                  (println "\n[Tool Observation Output]:")
                  (println (:output eval-res))
                  (println (format "  ↳ [Turn %d Latency: Model=%.2f ms, Tool=%.2f ms, Total=%.2f ms]"
                                   turn gen-ms tool-ms turn-total-ms)))

                (swap! history conj {:role :user :content obs-str})
                (swap! transcript conj {:turn turn :role :tool :content obs-str})
                (when (seq out)
                  (spit out (str/join "\n\n" (map :content @transcript))))
                (recur (inc turn))))))))))

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
