(ns tools.gemma4-agent
  "CLI entrypoint and wrapper for Gemma 4 autonomous coding agent loop."
  (:require [clojure.string :as str]
            [einsum.agent.core :as core]
            [tools.cli :as cli]
            [tools.gemma4-inference :as gemma4-inf]))

;; =============================================================================
;; Re-Exports for Backward Compatibility
;; =============================================================================

(def DEFAULT_TOOL_DECLARATION core/DEFAULT-TOOL-DECLARATION)
(def DEFAULT-TOOL-DECLARATION core/DEFAULT-TOOL-DECLARATION)
(def DEFAULT_SYSTEM_PROMPT core/DEFAULT-SYSTEM-PROMPT)
(def DEFAULT-SYSTEM-PROMPT core/DEFAULT-SYSTEM-PROMPT)
(def DEFAULT_SYSTEM_PROMPT_XML core/DEFAULT-SYSTEM-PROMPT-XML)
(def DEFAULT-SYSTEM-PROMPT-XML core/DEFAULT-SYSTEM-PROMPT-XML)
(def DEFAULT_AGENT_OPTS core/DEFAULT-AGENT-OPTS)
(def DEFAULT-AGENT-OPTS core/DEFAULT-AGENT-OPTS)

(def try-parse-sci-reader core/try-parse-sci-reader)
(def extract-balanced-sexpr core/extract-balanced-sexpr)
(def parse-tool-call-code core/parse-tool-call-code)
(def extract-thinking-trace core/extract-thinking-trace)
(def strip-thinking-trace core/strip-thinking-trace)
(def extract-tool-call core/extract-tool-call)
(def sanitize-history-model-content core/sanitize-history-model-content)
(def format-tool-response core/format-tool-response)
(def extract-clojure-code-blocks core/extract-clojure-code-blocks)
(def inside-unclosed-thought? core/inside-unclosed-thought?)
(def semantic-stop? core/semantic-stop?)
(def format-agent-chat-prompt core/format-agent-chat-prompt)
(def create-benchmark-sci-ctx core/create-benchmark-sci-ctx)
(def create-agent-sci-ctx core/create-agent-sci-ctx)
(def eval-tool-code core/eval-tool-code)
(def print-and-save-telemetry! core/print-and-save-telemetry!)
(def nudge-needed? core/nudge-needed?)
(def run-agent-loop core/run-agent-loop)

;; =============================================================================
;; CLI Entrypoint & Flag Parsing
;; =============================================================================

(defn parse-agent-cli-args
  "Parses CLI flags for gemma4_agent."
  [args]
  (let [opts (cli/parse-cli-args args DEFAULT_AGENT_OPTS)]
    (cond-> opts
      (string? (:max-consecutive-errors opts))
      (update :max-consecutive-errors #(Long/parseLong %))
      (string? (:sandbox opts))
      (update :sandbox #(keyword (str/replace % #"^:+" "")))
      (string? (:tool-syntax opts))
      (update :tool-syntax #(keyword (str/replace % #"^:+" ""))))))

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
