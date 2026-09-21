(ns einsum.test-runner
  "Main CLI entrypoint for running the complete test suite."
  (:require [einsum.agent-test]
            [einsum.compiler.autodiff-test]
            [einsum.runtime.benchmark.core-test]
            [einsum.runtime.benchmark.runner-test]
            [einsum.runtime.benchmark.workloads-test]
            [einsum.compiler.compile-test]
            [einsum.core-test]
            [einsum.tools.download-hf-test]
            [einsum.tools.fetch-pjrt-test]
            [einsum.runtime.generation-test]
            [einsum.hardware-test]
            [einsum.integration.rocm-e2e-test]
            [einsum.integration.sycl-e2e-test]
            [einsum.logic.ast-test]
            [einsum.logic.autodiff-test]
            [einsum.logic.core-test]
            [einsum.logic.data.family-trees-test]
            [einsum.logic.dce-test]
            [einsum.logic.expand-test]
            [einsum.logic.exl3-test]
            [einsum.quant.ternary-test]
            [einsum.models.gemma-test]
            [einsum.models.gemma3-test]
            [einsum.models.gpt2-test]
            [einsum.logic.index-test]
            [einsum.logic.lower-test]
            [einsum.logic.nn-test]
            [einsum.logic.quip-test]
            [einsum.logic.shape-test]
            [einsum.logic.memory.relation-test]
            [einsum.logic.memory.poc-builders-test]
            [einsum.logic.memory.span-probe-test]
            [einsum.logic.memory.train-mem-projection-test]
            [einsum.logic.memory.anchored-memory-test]
            [einsum.logic.memory.camp-test]
            [einsum.logic.memory.ephemeral-test]
            [einsum.logic.memory.contrastive-test]
            [einsum.logic.memory.factorization-test]
            [einsum.logic.memory.sampler-test]
            [einsum.logic.attention.kg-masked-test]
            [einsum.models.tl-block-test]
            [einsum.models.tl-nano-test]
            [einsum.models.transformer-test]
            [einsum.models.predicate-invention-test]
            [einsum.logic.predicate-crystallization-test]
            [einsum.logic.constrained-search-test]
            [einsum.kb.store-test]
            [einsum.kb.device-test]
            [einsum.runtime.c2c-test]
            [einsum.kb.rbac-test]
            [einsum.logic.agent.state-tracker-test]
            [einsum.logic.agent.swe-benchmark-test]
            [einsum.logic.semiring-test]
            [einsum.models.smollm-test]
            [einsum.logic.stratified-negation-test]
            [einsum.logic.symbolic-test]
            [einsum.logic.symbolic-e2e-test]
            [einsum.logic.symbolic-scaling-test]
            [einsum.runtime.opt-test]
            [einsum.compiler.pjrt-test]
            [einsum.compiler.pjrt-version-test]
            [einsum.runtime.profile-test]
            [einsum.quantize-test]
            [einsum.runtime.safetensors-test]
            [einsum.runtime.sampling-test]
            [einsum.compiler.stablehlo-test]
            [einsum.test-helpers.generators-test]
            [einsum.test-helpers.isolated-runner :as isolated-runner]
            [einsum.test-helpers.isolated-runner-test]
            [einsum.test-helpers.parity-test]
            [einsum.test-helpers.telemetry :as telemetry]
            [einsum.test-helpers.telemetry-test]
            [einsum.runtime.tokenizer-test]
            [clojure.test :refer [run-tests]]))

(defn -main
  "Runs all unit, generative, and hardware integration tests, outputting an EDN telemetry report."
  [& _args]
  (let [results (run-tests 'einsum.compiler.stablehlo-test
                           'einsum.compiler.pjrt-test
                           'einsum.compiler.pjrt-version-test
                           'einsum.test-helpers.generators-test
                           'einsum.test-helpers.isolated-runner-test
                           'einsum.test-helpers.parity-test
                           'einsum.test-helpers.telemetry-test
                           'einsum.runtime.benchmark.core-test
                           'einsum.runtime.benchmark.workloads-test
                           'einsum.runtime.benchmark.runner-test
                           'einsum.compiler.compile-test
                           'einsum.compiler.autodiff-test
                           'einsum.hardware-test
                           'einsum.quantize-test
                           'einsum.runtime.opt-test
                           'einsum.runtime.safetensors-test
                           'einsum.runtime.sampling-test
                           'einsum.runtime.tokenizer-test
                           'einsum.runtime.generation-test
                           'einsum.tools.download-hf-test
                           'einsum.tools.fetch-pjrt-test
                           'einsum.runtime.profile-test
                           'einsum.agent-test
                           'einsum.logic.ast-test
                           'einsum.logic.autodiff-test
                           'einsum.logic.core-test
                           'einsum.logic.data.family-trees-test
                           'einsum.logic.dce-test
                           'einsum.logic.expand-test
                           'einsum.quant.ternary-test
                           'einsum.models.gemma-test
                           'einsum.models.gemma3-test
                           'einsum.models.gpt2-test
                           'einsum.logic.index-test
                           'einsum.logic.lower-test
                           'einsum.logic.nn-test
                           'einsum.logic.quip-test
                           'einsum.logic.exl3-test
                           'einsum.logic.shape-test
                           'einsum.logic.semiring-test
                           'einsum.models.smollm-test
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
                           'einsum.models.tl-block-test
                           'einsum.models.tl-nano-test
                           'einsum.models.transformer-test
                           'einsum.models.predicate-invention-test
                           'einsum.logic.predicate-crystallization-test
                           'einsum.logic.constrained-search-test
                           'einsum.kb.store-test
                           'einsum.kb.device-test
                           'einsum.runtime.c2c-test
                           'einsum.kb.rbac-test
                           'einsum.logic.agent.state-tracker-test
                           'einsum.logic.agent.swe-benchmark-test
                           'einsum.core-test)
        rocm-res (isolated-runner/run-isolated-test 'einsum.integration.rocm-e2e-test {"HIP_VISIBLE_DEVICES" "0" "ROCR_VISIBLE_DEVICES" "0"})
        sycl-res (isolated-runner/run-isolated-test 'einsum.integration.sycl-e2e-test)
        {:keys [fail error]} results
        rocm-ok? (= (:status rocm-res) :pass)
        sycl-ok? (= (:status sycl-res) :pass)
        _ (telemetry/generate-edn-report {:in-process-results results
                                          :isolated-hardware-results {:rocm rocm-res :sycl sycl-res}})]
    (if (and (zero? fail) (zero? error) rocm-ok? sycl-ok?)
      (do (println "All unit, generative, and hardware integration tests passed successfully.")
          (System/exit 0))
      (do (println (str "Test failures detected: " fail " failures, " error " errors, ROCm status: " (:status rocm-res) ", SYCL status: " (:status sycl-res)))
          (System/exit 1)))))
