(ns tools.gemma4-inference
  "Top-level runnable CLI entrypoint and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]
            [einsum.models.gemma4.weights :as gemma-weights]
            [tools.cli :as cli]))

;; ==============================================================================
;; CLI Defaults & Argument Parsing
;; ==============================================================================

(def DEFAULT_CLI_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 20
   :temperature 0.7
   :top-k 10
   :backend :cpu
   :precision :bf16
   :method :kv-cache
   :compare false
   :verbose false
   :quiet false
   :thinking false
   :profile true
   :profile-out "scratch/gemma4_profile.edn"
   :chrome-trace-out "scratch/gemma4_chrome_trace.json"})

(def normalize-args cli/normalize-args)

(defn parse-cli-args
  "Parses command-line flags (--prompt, --model/--model-dir, --max-new-tokens, --temperature, --top-k, --backend, --precision, --verbose, --quiet)."
  [args]
  (cli/parse-cli-args args DEFAULT_CLI_OPTS))

;; ==============================================================================
;; High-Level Text Generation
;; ==============================================================================

(def generate-text rt/generate-text)
(def generate-text-string rt/generate-text-string)
(def generate-new-tokens-and-text rt/generate-new-tokens-and-text)
(def generate-new-text-string rt/generate-new-text-string)

;; ==============================================================================
;; Backward-Compatible Re-Exports for Existing Callers
;; ==============================================================================

(def DEFAULT_MODEL_DIRS cfg/DEFAULT_MODEL_DIRS)
(def find-model-dir cfg/find-model-dir)
(def load-model-config cfg/load-model-config)
(def resolve-weight-shape cfg/resolve-weight-shape)
(def max-safe-prefill-seq-len cfg/max-safe-prefill-seq-len)

(def floats->bf16-shorts gemma-weights/floats->bf16-shorts)
(def quantize-bf16-to-int8 gemma-weights/quantize-bf16-to-int8)
(def load-weight-buffer gemma-weights/load-weight-buffer)
(def load-linear-projection-buffers gemma-weights/load-linear-projection-buffers)
(def allocate-device-weights gemma-weights/allocate-device-weights)
(def allocate-tensor-logic-weights gemma-weights/allocate-device-weights)
(def allocate-relational-buffers gemma-weights/allocate-relational-buffers)
(def destroy-relational-buffers! gemma-weights/destroy-relational-buffers!)

(def GEMMA4-STOP-TOKEN-IDS kernels/GEMMA4-STOP-TOKEN-IDS)
(def build-tensor-logic-invars kernels/build-tensor-logic-invars)
(def build-gemma4-kv-invars kernels/build-gemma4-kv-invars)
(def build-gemma4-kv-outvars kernels/build-gemma4-kv-outvars)
(def build-gemma4-prefill-outvars kernels/build-gemma4-prefill-outvars)
(def allocate-kv-cache-buffers kernels/allocate-kv-cache-buffers)
(def compile-tensor-logic-executable kernels/compile-tensor-logic-executable)
(def compile-gemma4-prefill-executable kernels/compile-gemma4-prefill-executable)
(def compile-gemma4-kv-executable kernels/compile-gemma4-kv-executable)
(def compile-in-vram-loop-executable kernels/compile-in-vram-loop-executable)
(def compile-executable kernels/compile-tensor-logic-executable)

(def argmax-host rt/argmax-host)
(def argmax-with-penalty rt/argmax-with-penalty)
(def sample-next-token rt/sample-next-token)
(def common-prefix-len rt/common-prefix-len)
(def run-vram-loop-generation rt/run-vram-loop-generation)
(def run-cached-kv-generation rt/run-cached-kv-generation)
(def run-autoregressive-generation-logic rt/run-autoregressive-generation-logic)
(def run-autoregressive-generation rt/run-autoregressive-generation)
(def init-inference-session rt/init-inference-session)
(def init-agent-vram-session rt/init-agent-vram-session)
(def close-agent-session! rt/close-agent-session!)

;; Deprecated signal stubs (signal chaining is now exclusively handled by tools/gemma4.sh)
(defn needs-libjsig-reexec? [_] false)
(defn reexec-with-libjsig! [& _] nil)

;; ==============================================================================
;; CLI Entrypoint
;; ==============================================================================

(defn -main
  "CLI entrypoint for Gemma 4 text generation."
  [& args]
  (try
    (let [opts (parse-cli-args args)]
      (when-not (:quiet opts)
        (println "==================================================================")
        (println (str "  clj-xla Gemma 4 Single-Pass Prefill & " (case (:precision opts) :int8 "INT8" :int4 "INT4" "BF16") " Generation "))
        (println "=================================================================="))
      (let [session (init-inference-session opts)]
        (generate-text session (:prompt opts))))
    (catch Throwable e
      (println "\nExecution Exception:" (.getMessage e))
      (.printStackTrace e))
    (finally
      (.. Runtime getRuntime (halt 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
