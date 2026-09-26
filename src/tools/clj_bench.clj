(ns tools.clj-bench
  "CLI entrypoint for clojure_bench (Gate 3 Evals).
   Evaluates Gemma 4 models on Clojure code synthesis and agentic tool-use loops."
  (:require [experiments.gate3-evals.clojure-bench.run :as bench-run]))

(defn -main
  "Main entrypoint delegating to experiments.gate3-evals.clojure-bench.run."
  [& args]
  (apply bench-run/-main args))
