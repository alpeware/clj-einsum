(ns einsum.logic.autodiff
  "Algebraic reverse-mode derivative derivation (VJP) for Tensor Logic equations."
  (:require [einsum.logic.ast :as ast]))

(defn derive-adjoint-equations
  "Derives backward adjoint equations algebraically for an equation.
   For each target input term T_i in body, the adjoint equation is:
   d(T_i) = d(Head) * prod_{j != i} T_j"
  [eqn-or-map]
  (let [eqn (if (map? eqn-or-map)
              (ast/eqn (:head eqn-or-map) (:attrs eqn-or-map) (:body eqn-or-map))
              eqn-or-map)
        head (ast/head eqn)
        [out-tensor & out-idx] head
        body (ast/body-terms eqn)
        attrs (ast/attrs eqn)
        d-out (into [(keyword (str "adj/" (name out-tensor)))] out-idx)]
    (mapv
     (fn [target-term]
       (let [target-name (first target-term)
             target-idxs (vec (rest target-term))
             other-inputs (vec (remove #(= (first %) target-name) body))
             d-target (into [(keyword (str "adj/" (name target-name)))] target-idxs)
             adjoint-body (into [d-out] other-inputs)]
         (cond-> [:= d-target]
           attrs (conj attrs)
           true (into adjoint-body))))
     body)))
