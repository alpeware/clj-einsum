(ns einsum.logic.index
  "Deterministic Einstein summation index partitioning for StableHLO dot_general lowering."
  (:require [clojure.set :as set]))

(defn partition-indices
  "Deterministically partitions indices for StableHLO dot_general lowering.
   All sets are sorted and converted to vectors to guarantee deterministic dimension bindings:
   - contracting: (lhs ∩ rhs) \\ head
   - batch: lhs ∩ rhs ∩ head
   - lhs-free: lhs \\ rhs
   - rhs-free: rhs \\ lhs"
  [lhs-idx rhs-idx head-idx]
  (let [lhs-set  (set lhs-idx)
        rhs-set  (set rhs-idx)
        head-set (set head-idx)
        common   (set/intersection lhs-set rhs-set)]
    {:contracting (vec (sort (set/difference common head-set)))
     :batch       (vec (sort (set/intersection common head-set)))
     :lhs-free    (vec (filter #(contains? (set/difference lhs-set rhs-set) %) lhs-idx))
     :rhs-free    (vec (filter #(contains? (set/difference rhs-set lhs-set) %) rhs-idx))}))
