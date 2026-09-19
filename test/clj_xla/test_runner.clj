(ns clj-xla.test-runner
  "Main CLI entrypoint for running the complete test suite."
  (:require [clj-xla.agent-test]
            [clj-xla.autodiff-test]
            [clj-xla.benchmark.core-test]
            [clj-xla.benchmark.runner-test]
            [clj-xla.benchmark.workloads-test]
            [clj-xla.compile-test]
            [clj-xla.core-test]
            [clj-xla.download-hf-test]
            [clj-xla.fetch-pjrt-binaries-test]
            [clj-xla.generation-test]
            [clj-xla.hardware-test]
            [clj-xla.integration.rocm-e2e-test]
            [clj-xla.integration.sycl-e2e-test]
            [clj-xla.logic.ast-test]
            [clj-xla.logic.autodiff-test]
            [clj-xla.logic.core-test]
            [clj-xla.logic.data.family-trees-test]
            [clj-xla.logic.dce-test]
            [clj-xla.logic.expand-test]
            [clj-xla.logic.exl3-test]
            [clj-xla.logic.gemma-test]
            [clj-xla.logic.gemma3-test]
            [clj-xla.logic.gpt2-test]
            [clj-xla.logic.index-test]
            [clj-xla.logic.lower-test]
            [clj-xla.logic.nn-test]
            [clj-xla.logic.quip-test]
            [clj-xla.logic.shape-test]
            [clj-xla.logic.memory.relation-test]
            [clj-xla.logic.memory.poc-builders-test]
            [clj-xla.logic.memory.span-probe-test]
            [clj-xla.logic.memory.train-mem-projection-test]
            [clj-xla.logic.memory.anchored-memory-test]
            [clj-xla.logic.memory.camp-test]
            [clj-xla.logic.memory.ephemeral-test]
            [clj-xla.logic.memory.contrastive-test]
            [clj-xla.logic.memory.factorization-test]
            [clj-xla.logic.memory.sampler-test]
            [clj-xla.logic.attention.kg-masked-test]
            [clj-xla.logic.models.tl-block-test]
            [clj-xla.logic.models.tl-nano-test]
            [clj-xla.logic.predicate-invention-test]
            [clj-xla.logic.predicate-crystallization-test]
            [clj-xla.logic.constrained-search-test]
            [clj-xla.logic.kb-test]
            [clj-xla.logic.kb-device-test]
            [clj-xla.logic.c2c-test]
            [clj-xla.logic.rbac-test]
            [clj-xla.logic.agent.state-tracker-test]
            [clj-xla.logic.agent.swe-benchmark-test]
            [clj-xla.logic.semiring-test]
            [clj-xla.logic.smollm-test]
            [clj-xla.logic.stratified-negation-test]
            [clj-xla.logic.symbolic-test]
            [clj-xla.logic.symbolic-e2e-test]
            [clj-xla.logic.symbolic-scaling-test]
            [clj-xla.opt-test]
            [clj-xla.pjrt-test]
            [clj-xla.pjrt.version-test]
            [clj-xla.profile-test]
            [clj-xla.quantize-test]
            [clj-xla.safetensors-test]
            [clj-xla.sampling-test]
            [clj-xla.stablehlo-test]
            [clj-xla.test.generators-test]
            [clj-xla.test.isolated-runner :as isolated-runner]
            [clj-xla.test.isolated-runner-test]
            [clj-xla.test.parity-test]
            [clj-xla.test.telemetry :as telemetry]
            [clj-xla.test.telemetry-test]
            [clj-xla.tokenizer-test]
            [clojure.test :refer [run-tests]]))

(defn -main
  "Runs all unit, generative, and hardware integration tests, outputting an EDN telemetry report."
  [& _args]
  (let [results (run-tests 'clj-xla.stablehlo-test
                           'clj-xla.pjrt-test
                           'clj-xla.pjrt.version-test
                           'clj-xla.test.generators-test
                           'clj-xla.test.isolated-runner-test
                           'clj-xla.test.parity-test
                           'clj-xla.test.telemetry-test
                           'clj-xla.benchmark.core-test
                           'clj-xla.benchmark.workloads-test
                           'clj-xla.benchmark.runner-test
                           'clj-xla.compile-test
                           'clj-xla.autodiff-test
                           'clj-xla.hardware-test
                           'clj-xla.quantize-test
                           'clj-xla.opt-test
                           'clj-xla.safetensors-test
                           'clj-xla.sampling-test
                           'clj-xla.tokenizer-test
                           'clj-xla.generation-test
                           'clj-xla.download-hf-test
                           'clj-xla.fetch-pjrt-binaries-test
                           'clj-xla.profile-test
                           'clj-xla.agent-test
                           'clj-xla.logic.ast-test
                           'clj-xla.logic.autodiff-test
                           'clj-xla.logic.core-test
                           'clj-xla.logic.data.family-trees-test
                           'clj-xla.logic.dce-test
                           'clj-xla.logic.expand-test
                           'clj-xla.logic.gemma-test
                           'clj-xla.logic.gemma3-test
                           'clj-xla.logic.gpt2-test
                           'clj-xla.logic.index-test
                           'clj-xla.logic.lower-test
                           'clj-xla.logic.nn-test
                           'clj-xla.logic.quip-test
                           'clj-xla.logic.exl3-test
                           'clj-xla.logic.shape-test
                           'clj-xla.logic.semiring-test
                           'clj-xla.logic.smollm-test
                           'clj-xla.logic.stratified-negation-test
                           'clj-xla.logic.symbolic-test
                           'clj-xla.logic.symbolic-e2e-test
                           'clj-xla.logic.symbolic-scaling-test
                           'clj-xla.logic.memory.relation-test
                           'clj-xla.logic.memory.poc-builders-test
                           'clj-xla.logic.memory.span-probe-test
                           'clj-xla.logic.memory.train-mem-projection-test
                           'clj-xla.logic.memory.anchored-memory-test
                           'clj-xla.logic.memory.camp-test
                           'clj-xla.logic.memory.ephemeral-test
                           'clj-xla.logic.memory.contrastive-test
                           'clj-xla.logic.memory.factorization-test
                           'clj-xla.logic.memory.sampler-test
                           'clj-xla.logic.attention.kg-masked-test
                           'clj-xla.logic.models.tl-block-test
                           'clj-xla.logic.models.tl-nano-test
                           'clj-xla.logic.predicate-invention-test
                           'clj-xla.logic.predicate-crystallization-test
                           'clj-xla.logic.constrained-search-test
                           'clj-xla.logic.kb-test
                           'clj-xla.logic.kb-device-test
                           'clj-xla.logic.c2c-test
                           'clj-xla.logic.rbac-test
                           'clj-xla.logic.agent.state-tracker-test
                           'clj-xla.logic.agent.swe-benchmark-test
                           'clj-xla.core-test)
        rocm-res (isolated-runner/run-isolated-test 'clj-xla.integration.rocm-e2e-test {"HIP_VISIBLE_DEVICES" "0" "ROCR_VISIBLE_DEVICES" "0"})
        sycl-res (isolated-runner/run-isolated-test 'clj-xla.integration.sycl-e2e-test)
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
