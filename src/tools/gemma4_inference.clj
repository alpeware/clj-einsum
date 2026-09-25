(ns tools.gemma4-inference
  "Top-level runnable CLI entrypoint and REPL API for end-to-end Gemma 4 text generation via pure XLA execution."
  (:require [clojure.pprint :refer [pprint]]
            [clojure.string :as str]
            [einsum.core :as xla]
            [einsum.models.gemma4.config :as cfg]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]
            [einsum.models.gemma4.weights :as gemma-weights]
            [einsum.runtime.profile :as profile]
            [einsum.runtime.tokenizer.protocol :refer [bos-id decode encode]]
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

(defn generate-text
  "Generates text response using Gemma 4 model via pure Tensor Logic execution."
  [session prompt]
  (let [{:keys [tokenizer opts]} session
        {:keys [max-new-tokens temperature top-k model quiet]} opts
        clean-prompt (or prompt "The capital of France is")
        model-str (or model (get-in session [:config :model-dir]) "")
        is-it-model (str/includes? (str/lower-case model-str) "-it")
        is-already-templated (or (str/includes? clean-prompt "<|turn>user") (str/includes? clean-prompt "<|turn>model"))
        prompt-ids (cond
                     (and is-it-model (not is-already-templated))
                     (let [raw-ids (encode tokenizer clean-prompt)
                           clean-ids (if (= (first raw-ids) (bos-id tokenizer)) (rest raw-ids) raw-ids)
                           prefix (if (:thinking opts)
                                    ;; <bos><|turn>system\n<|think|><turn|>\n<|turn>user\n
                                    [(bos-id tokenizer) 105 9731 107 98 106 107 105 2364 107]
                                    [(bos-id tokenizer) 105 2364 107])]
                       (vec (concat prefix clean-ids [106 107 105 4368 107])))

                     :else
                     (let [raw-ids (encode tokenizer clean-prompt)]
                       (if (= (first raw-ids) (bos-id tokenizer))
                         (vec raw-ids)
                         (vec (cons (bos-id tokenizer) raw-ids)))))
        prompt-len (count prompt-ids)
        max-seq-len (long (or (:max-seq-len session) (:max-seq-len opts) (min 2048 (+ prompt-len max-new-tokens 16))))]
    (when-not quiet
      (let [prompt-str (if (> (count clean-prompt) 200)
                         (str (subs clean-prompt 0 100) " ... [truncated " (count clean-prompt) " chars] ... " (subs clean-prompt (- (count clean-prompt) 100)))
                         clean-prompt)
            tok-str (if (> prompt-len 30)
                      (str "[" (str/join " " (take 10 prompt-ids)) " ... " (str/join " " (take-last 5 prompt-ids)) "]")
                      (str prompt-ids))]
        (println (format "Prompt: \"%s\"" prompt-str))
        (println (format "Generation Options: max-new-tokens=%d, temperature=%.2f, top-k=%d, precision=%s, method=%s"
                         max-new-tokens temperature top-k (name (get-in session [:config :weight-dtype]))
                         (name (or (:method opts) :kv-cache))))
        (println (format "Encoded Token IDs (%d tokens): %s" prompt-len tok-str))))

    (let [metrics-atom (or (:metrics-atom session) (atom {}))
          trace-spans-atom (or (:trace-spans-atom session) (atom []))
          reuse-weights? (some? (:device-weights session))
          reuse-exec? (some? (:executable session))
          exec (binding [profile/*active-trace-spans* trace-spans-atom]
                 (if reuse-exec?
                   (:executable session)
                   (profile/with-profile metrics-atom "graph_compilation"
                     (cond
                       (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                       (kernels/compile-in-vram-loop-executable session max-seq-len)

                       (= (:method opts) :tensor-logic-full)
                       (kernels/compile-tensor-logic-executable session max-seq-len)

                       :else
                       (kernels/compile-gemma4-kv-executable session max-seq-len)))))
          safe-prefill-len (cfg/max-safe-prefill-seq-len (:config session))
          prefill-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                         (if (:prefill-executable session)
                           (:prefill-executable session)
                           (when (and (or (= (or (:method opts) :kv-cache) :kv-cache)
                                          (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                                      (<= max-seq-len safe-prefill-len))
                             (profile/with-profile metrics-atom "graph_compilation"
                               (kernels/compile-gemma4-prefill-executable session max-seq-len)))))
          step-exec (binding [profile/*active-trace-spans* trace-spans-atom]
                      (when (and (or (:vram-loop? opts) (:vram-loop? session) (= (:method opts) :vram-loop))
                                 (> max-seq-len safe-prefill-len))
                        (profile/with-profile metrics-atom "graph_compilation"
                          (kernels/compile-gemma4-kv-executable session max-seq-len))))
          active-session (assoc session
                                :prefill-executable prefill-exec
                                :step-executable step-exec)
          device-weights (binding [profile/*active-trace-spans* trace-spans-atom]
                           (if reuse-weights?
                             (:device-weights session)
                             (profile/with-profile metrics-atom "weight_transfer"
                               (gemma-weights/allocate-device-weights session))))]
      (when-not quiet
        (println "\nGenerating tokens autoregressively with pure Tensor Logic Gemma 4 Kernel..."))
      (let [final-context (binding [profile/*active-trace-spans* trace-spans-atom]
                            (profile/with-profile metrics-atom "autoregressive_generation"
                              (rt/run-autoregressive-generation active-session exec device-weights prompt-ids max-seq-len)))
            generated-str (decode tokenizer final-context)]
        (when-not quiet
          (println)
          (println generated-str))
        (when-not reuse-weights?
          (if-let [sa (:session-arena session)]
            (xla/close-arena! sa)
            (doseq [w device-weights] (xla/destroy-buffer! (:ctx session) w))))
        (when-let [out-path (:out opts)]
          (spit out-path generated-str)
          (when-not quiet
            (println (format "\n  ↳ Written generated output to [%s]" out-path))))
        (when-let [profile-path (:profile-out opts)]
          (spit profile-path (with-out-str (pprint @metrics-atom)))
          (when-not quiet
            (println (format "  ↳ Saved telemetry profile report to [%s]" profile-path))))
        (when-let [trace-path (:chrome-trace-out opts)]
          (profile/save-chrome-trace! @trace-spans-atom trace-path)
          (when-not quiet
            (println (format "  ↳ Saved Chrome tracing JSON to [%s]" trace-path))))
        (when-not quiet
          (println "\n==================================================================")
          (println "=== Tensor Logic Gemma 4 Generation Verification Passed! ===")
          (println "=================================================================="))
        final-context))))

(defn generate-text-string
  "Generates text response using Gemma 4 model session and returns decoded text string."
  [session prompt]
  (let [{:keys [tokenizer]} session
        final-context (generate-text session prompt)]
    (decode tokenizer final-context)))

(defn generate-new-text-string
  "Generates text response using Gemma 4 model session and returns ONLY newly generated text string without prompt prefix."
  [session prompt]
  (let [{:keys [tokenizer]} session
        raw-ids (encode tokenizer prompt)
        prompt-ids (if (= (first raw-ids) (bos-id tokenizer))
                     raw-ids
                     (vec (cons (bos-id tokenizer) raw-ids)))
        prompt-len (count prompt-ids)
        final-context (generate-text session prompt)
        new-ids (subvec (vec final-context) prompt-len)]
    (decode tokenizer new-ids)))

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
