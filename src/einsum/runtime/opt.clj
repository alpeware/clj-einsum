(ns einsum.runtime.opt
  "Frontend EDN SSA graph optimizations (Dead Code Elimination & Constant Folding)."
  (:require [einsum.compiler.stablehlo :as shlo]))

(defn dce
  "Performs Dead Code Elimination (DCE) on `graph`.
   Prunes unused equations whose outputs are not transitively connected to `:outvars`."
  [graph]
  (shlo/validate-graph graph)
  (let [{:keys [name invars outvars eqns]} graph
        used-vars (atom (set outvars))
        ;; Backward pass to mark all required variables
        _ (doseq [{:keys [invars outvars]} (reverse eqns)]
            (when (some @used-vars outvars)
              (swap! used-vars into invars)))
        ;; Filter equations emitting required variables
        pruned-eqns (filterv (fn [{:keys [outvars]}]
                               (some @used-vars outvars))
                             eqns)
        pruned-graph {:name name
                      :invars invars
                      :outvars outvars
                      :eqns pruned-eqns}]
    (shlo/validate-graph pruned-graph)))
