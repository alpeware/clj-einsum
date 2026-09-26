(ns einsum.test-runner
  "Main CLI entrypoint for running the complete test suite or targeted sub-groups."
  (:require [einsum.agent-test]
            [einsum.compiler.autodiff-test]
            [einsum.compiler.compile-test]
            [einsum.compiler.fuse-test]
            [einsum.compiler.kernel-test]
            [einsum.compiler.pjrt-test]
            [einsum.compiler.pjrt-version-test]
            [einsum.compiler.stablehlo-test]
            [einsum.core-test]
            [einsum.hardware-test]
            [einsum.integration.rocm-e2e-test]
            [einsum.integration.sycl-e2e-test]
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
            [einsum.logic.index-test]
            [einsum.logic.interpret-test]
            [einsum.logic.interpret-diff-test]
            [einsum.logic.lower-test]
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
            [einsum.quant.ternary-test]
            [einsum.quant.w4a16-custom-call-test]
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
            [einsum.test-helpers.isolated-runner :as isolated-runner]
            [einsum.test-helpers.isolated-runner-test]
            [einsum.test-helpers.parity-test]
            [einsum.test-helpers.telemetry :as telemetry]
            [einsum.test-helpers.telemetry-test]
            [einsum.tools.cli-test]
            [einsum.tools.download-hf-test]
            [einsum.tools.fetch-pjrt-test]
            [experiments.gate3-evals.clojure-bench-test]
            [clojure.string :as str]
            [clojure.test :refer [run-tests]]))

;; -----------------------------------------------------------------------------
;; Test Group Definitions
;; -----------------------------------------------------------------------------

(def fast-namespaces
  "Pure, fast unit and generative property tests with zero hardware dependencies (<3s)."
  ['einsum.compiler.stablehlo-test
   'einsum.compiler.fuse-test
   'einsum.compiler.kernel-test
   'einsum.compiler.pjrt-version-test
   'einsum.logic.interpret-test
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
   'einsum.tools.fetch-pjrt-test
   'einsum.runtime.profile-test
   'einsum.agent-test
   'einsum.quantize-test
   'einsum.quant.catq-test
   'einsum.quant.ternary-test
   'einsum.quant.w4a16-custom-call-test
   'einsum.models.gemma4-test
   'einsum.core-test
   'experiments.gate3-evals.clojure-bench-test])

(def logic-namespaces
  "Pedro Domingos' Declarative Tensor Logic, semirings, Horn-clause symbolic reasoning, memory relations, KB."
  ['einsum.logic.ast-test
   'einsum.logic.autodiff-test
   'einsum.logic.core-test
   'einsum.logic.data.family-trees-test
   'einsum.logic.dce-test
   'einsum.logic.expand-test
   'einsum.logic.index-test
   'einsum.logic.interpret-test
   'einsum.logic.lower-test
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
  "PJRT plugin backend initialization, graph compilation, benchmarks, C2C memory, and isolated runners."
  ['einsum.compiler.pjrt-test
   'einsum.compiler.compile-test
   'einsum.compiler.autodiff-test
   'einsum.hardware-test
   'einsum.logic.interpret-diff-test
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
     'hardware'      : PJRT backend kernels, compilation, C2C, benchmarks & isolated device tests
     'all' (default) : Complete 50+ namespace suite + hardware telemetry"
  [& args]
  (let [group-arg (str/lower-case (or (first args) "all"))
        is-all? (= group-arg "all")
        is-hw? (or is-all? (= group-arg "hardware") (= group-arg "hw"))
        selected-nses (get test-groups group-arg)
        nses-to-run (if (seq selected-nses)
                      selected-nses
                      ;; If user passed a specific namespace symbol directly
                      (let [sym (symbol group-arg)]
                        (if (find-ns sym)
                          [sym]
                          (do
                            (println (str "Unknown test group or namespace: '" group-arg "'. Available groups: " (str/join ", " (sort (keys test-groups)))))
                            (System/exit 1)))))]
    (println (str "Running test group [" group-arg "] (" (count nses-to-run) " namespaces)..."))
    (let [results (apply run-tests nses-to-run)
          {:keys [fail error]} results
          [rocm-res sycl-res] (if is-hw?
                                [(isolated-runner/run-isolated-test 'einsum.integration.rocm-e2e-test {"HIP_VISIBLE_DEVICES" "0" "ROCR_VISIBLE_DEVICES" "0"})
                                 (isolated-runner/run-isolated-test 'einsum.integration.sycl-e2e-test)]
                                [{:status :skipped} {:status :skipped}])
          rocm-ok? (or (not is-hw?) (= (:status rocm-res) :pass))
          sycl-ok? (or (not is-hw?) (= (:status sycl-res) :pass))
          _ (when is-all?
              (telemetry/generate-edn-report {:in-process-results results
                                              :isolated-hardware-results {:rocm rocm-res :sycl sycl-res}}))]
      (if (and (zero? fail) (zero? error) rocm-ok? sycl-ok?)
        (do (println (str "Test group [" group-arg "] completed successfully."))
            (System/exit 0))
        (do (println (str "Test failures detected: " fail " failures, " error " errors"
                          (when is-hw? (str ", ROCm status: " (:status rocm-res) ", SYCL status: " (:status sycl-res)))))
            (System/exit 1))))))
