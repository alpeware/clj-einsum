(ns tools.benchmark-ternary
  "Convenience CLI wrapper for running CAT-Q (1.58b ternary) benchmark and evaluation suite."
  (:require [experiments.gate1-compression.cat-q-ternary.run :as cat-q-run]))

(defn -main
  [& args]
  (apply cat-q-run/-main args))
