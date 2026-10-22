(ns einsum.test-runner
  "Main CLI entrypoint for running the complete test suite or targeted sub-groups."
  (:require [einsum.agent-test]
            [einsum.agent.graph-test]
            [einsum.core-test]
            [einsum.graph.binder-test]
            [einsum.graph.runtime-test]
            [einsum.hardware-test]
            [einsum.kb.device-test]
            [einsum.kb.rbac-test]
            [einsum.kb.store-test]
            [einsum.logic.agent.state-tracker-test]
            [einsum.logic.agent.swe-benchmark-test]
            [einsum.logic.ast-test]
            [einsum.logic.attention.kg-masked-test]
            [einsum.logic.autodiff-test]
            [einsum.logic.constrained-search-test]
            [einsum.logic.core-test]
            [einsum.logic.data.family-trees-test]
            [einsum.logic.dce-test]
            [einsum.logic.exl3-test]
            [einsum.logic.expand-test]
            [einsum.logic.gather-test]
            [einsum.logic.index-test]
            [einsum.logic.interpret-test]
            [einsum.logic.memory.anchored-memory-test]
            [einsum.logic.memory.camp-test]
            [einsum.logic.memory.contrastive-test]
            [einsum.logic.memory.ephemeral-test]
            [einsum.logic.memory.factorization-test]
            [einsum.logic.memory.poc-builders-test]
            [einsum.logic.memory.relation-test]
            [einsum.logic.memory.sampler-test]
            [einsum.logic.memory.span-probe-test]
            [einsum.logic.memory.train-mem-projection-test]
            [einsum.logic.nn-test]
            [einsum.logic.predicate-crystallization-test]
            [einsum.logic.quip-test]
            [einsum.logic.semiring-test]
            [einsum.logic.shape-test]
            [einsum.logic.stratified-negation-test]
            [einsum.logic.symbolic-e2e-test]
            [einsum.logic.symbolic-scaling-test]
            [einsum.logic.symbolic-test]
            [einsum.models.gemma-test]
            [einsum.models.gemma3-test]
            [einsum.models.gemma4-test]
            [einsum.models.gpt2-test]
            [einsum.models.predicate-invention-test]
            [einsum.models.smollm-test]
            [einsum.models.tl-block-test]
            [einsum.models.tl-nano-test]
            [einsum.models.transformer-test]
            [einsum.quant.catq-test]
            [einsum.quant.eviction-test]
            [einsum.quant.ternary-test]
            [einsum.quant.turboquant-test]
            [einsum.quantize-test]
            [einsum.runtime.benchmark.core-test]
            [einsum.runtime.benchmark.runner-test]
            [einsum.runtime.arena-test]
            [einsum.runtime.weights-test]
            [einsum.runtime.benchmark.workloads-test]
            [einsum.runtime.c2c-test]
            [einsum.runtime.generation-test]
            [einsum.runtime.opt-test]
            [einsum.runtime.profile-test]
            [einsum.runtime.safetensors-test]
            [einsum.runtime.sampling-test]
            [einsum.runtime.tokenizer-test]
            [einsum.test-helpers.generators-test]
            [einsum.test-helpers.isolated-runner-test]
            [einsum.test-helpers.parity-test]
            [einsum.test-helpers.telemetry :as telemetry]
            [einsum.test-helpers.telemetry-test]
            [einsum.tools.cli-test]
            [einsum.tools.download-hf-test]
            [einsum.uop.core-test]
            [einsum.uop.lower-test]
            [einsum.uop.gemma4-ast-test]
            [einsum.uop.opt-test]
            [einsum.uop.beam-test]
            [einsum.uop.rangeify-test]
            [einsum.uop.render-test]
            [einsum.uop.runtime-test]
            [einsum.uop.schedule-test]
            [einsum.uop.e2b-integration-test]
            [einsum.uop.golden-test]
            [experiments.gate1-compression.tiered-turbo-kv-test]
            [experiments.gate3-evals.clojure-bench-test]
            [experiments.gate3-evals.multipl-e-test]
            [experiments.gate3-evals.prompt-tuning-v1-test]
            [experiments.gate3-evals.perplexity-probe-v1-test]
            [clojure.string :as str]
            [clojure.test :refer [run-tests]]))

;; -----------------------------------------------------------------------------
;; Test Group Definitions
;; -----------------------------------------------------------------------------

(def fast-namespaces
  "Pure, fast unit and generative property tests with zero hardware dependencies (<3s)."
  ['einsum.logic.interpret-test
   'einsum.test-helpers.generators-test
   'einsum.test-helpers.parity-test
   'einsum.test-helpers.telemetry-test
   'einsum.runtime.arena-test
   'einsum.runtime.weights-test
   'einsum.runtime.benchmark.core-test
   'einsum.runtime.opt-test
   'einsum.runtime.safetensors-test
   'einsum.runtime.sampling-test
   'einsum.runtime.tokenizer-test
   'einsum.runtime.generation-test
   'einsum.tools.cli-test
   'einsum.tools.download-hf-test
   'einsum.runtime.profile-test
   'einsum.agent-test
   'einsum.quantize-test
   'einsum.quant.catq-test
   'einsum.quant.eviction-test
   'einsum.quant.ternary-test
   'einsum.quant.turboquant-test
   'einsum.models.gemma4-test
   'einsum.core-test
   'einsum.graph.runtime-test
   'einsum.graph.binder-test
   'einsum.agent.graph-test
   'experiments.gate3-evals.clojure-bench-test
   'experiments.gate3-evals.multipl-e-test
   'experiments.gate3-evals.prompt-tuning-v1-test
   'experiments.gate3-evals.perplexity-probe-v1-test
   'experiments.gate1-compression.tiered-turbo-kv-test
   'einsum.logic.gather-test
   'einsum.uop.core-test
   'einsum.uop.lower-test
   'einsum.uop.gemma4-ast-test
   'einsum.uop.opt-test
   'einsum.uop.beam-test
   'einsum.uop.rangeify-test
   'einsum.uop.render-test
   'einsum.uop.runtime-test
   'einsum.uop.schedule-test
   'einsum.uop.e2b-integration-test
   'einsum.uop.golden-test])

(def logic-namespaces
  "Pedro Domingos' Declarative Tensor Logic, semirings, Horn-clause symbolic reasoning, memory relations, KB."
  ['einsum.logic.ast-test
   'einsum.logic.autodiff-test
   'einsum.logic.core-test
   'einsum.logic.data.family-trees-test
   'einsum.logic.dce-test
   'einsum.logic.expand-test
   'einsum.logic.gather-test
   'einsum.logic.index-test
   'einsum.logic.interpret-test
   'einsum.logic.nn-test
   'einsum.logic.quip-test
   'einsum.logic.exl3-test
   'einsum.logic.shape-test
   'einsum.logic.semiring-test
   'einsum.logic.stratified-negation-test
   'einsum.logic.symbolic-test
   'einsum.logic.symbolic-e2e-test
   'einsum.logic.symbolic-scaling-test
   'einsum.logic.memory.relation-test
   'einsum.logic.memory.poc-builders-test
   'einsum.logic.memory.span-probe-test
   'einsum.logic.memory.train-mem-projection-test
   'einsum.logic.memory.anchored-memory-test
   'einsum.logic.memory.camp-test
   'einsum.logic.memory.ephemeral-test
   'einsum.logic.memory.contrastive-test
   'einsum.logic.memory.factorization-test
   'einsum.logic.memory.sampler-test
   'einsum.logic.attention.kg-masked-test
   'einsum.logic.predicate-crystallization-test
   'einsum.logic.constrained-search-test
   'einsum.kb.store-test
   'einsum.kb.device-test
   'einsum.kb.rbac-test
   'einsum.logic.agent.state-tracker-test
   'einsum.logic.agent.swe-benchmark-test])

(def model-namespaces
  "Model architectures, layer lowerings, transformer blocks, and predicate invention."
  ['einsum.models.gemma-test
   'einsum.models.gemma3-test
   'einsum.models.gemma4-test
   'einsum.models.gpt2-test
   'einsum.models.smollm-test
   'einsum.models.tl-block-test
   'einsum.models.tl-nano-test
   'einsum.models.transformer-test
   'einsum.models.predicate-invention-test])

(def hardware-namespaces
  "Hardware capability detection, benchmarks, C2C memory, and isolated runners."
  ['einsum.hardware-test
   'einsum.runtime.c2c-test
   'einsum.runtime.benchmark.workloads-test
   'einsum.runtime.benchmark.runner-test
   'einsum.test-helpers.isolated-runner-test])

(def all-namespaces
  "Complete suite of in-process test namespaces."
  (vec (distinct (concat fast-namespaces logic-namespaces model-namespaces hardware-namespaces))))

(def test-groups
  {"fast" fast-namespaces
   "unit" fast-namespaces
   "logic" logic-namespaces
   "models" model-namespaces
   "model" model-namespaces
   "hardware" hardware-namespaces
   "hw" hardware-namespaces
   "all" all-namespaces})

;; -----------------------------------------------------------------------------
;; CLI Entrypoint
;; -----------------------------------------------------------------------------

(defn -main
  "CLI entrypoint for running test suite.
   Accepts optional group name(s):
     'fast' / 'unit' : Pure unit & generative tests (<3s)
     'logic'         : Declarative Tensor Logic, semirings, fixpoints, memory relations, KB
     'models'        : Model architectures & lowerings (Gemma, GPT-2, SmolLM, TL)
     'hardware'      : Hardware capability detection, benchmarks, C2C memory, and isolated runners
     'all' (default) : Complete namespace suite + hardware telemetry"
  [& args]
  (let [group-arg (str/lower-case (or (first args) "all"))
        is-all? (= group-arg "all")
        selected-nses (get test-groups group-arg)
        nses-to-run (if (seq selected-nses)
                      selected-nses
                      ;; If user passed a specific namespace symbol directly
                      (let [sym (symbol group-arg)]
                        (try (require sym) (catch Throwable _ nil))
                        (if (find-ns sym)
                          [sym]
                          (do
                            (println (str "Unknown test group or namespace: '" group-arg "'. Available groups: " (str/join ", " (sort (keys test-groups)))))
                            (System/exit 1)))))]
    (println (str "Running test group [" group-arg "] (" (count nses-to-run) " namespaces)..."))
    (let [results (apply run-tests nses-to-run)
          {:keys [fail error]} results
          _ (when is-all?
              (telemetry/generate-edn-report {:in-process-results results}))]
      (if (and (zero? fail) (zero? error))
        (do (println (str "Test group [" group-arg "] completed successfully."))
            (System/exit 0))
        (do (println (str "Test failures detected: " fail " failures, " error " errors"))
            (System/exit 1))))))
