(ns proposals.gate1-compression.cat-q-ternary.run
  "Stage 2 pod runner delegating to tools.benchmark-ternary."
  (:require [tools.benchmark-ternary :as bench]))

(defn -main
  [& args]
  (apply bench/-main args))
