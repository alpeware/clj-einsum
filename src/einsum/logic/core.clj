(ns einsum.logic.core
  "High-level Public Clojure API for compiling Tensor Logic ASTs into native UOp executables."
  (:require [einsum.uop.runtime :as uop-rt]))

(defn compile-ast
  "Compiles a Tensor Logic Hiccup AST into a native UOp loaded executable.
   Arity [graph-name invars ast target-heads]: uses default UOp CPU context.
   Arity [ctx graph-name invars ast target-heads]: uses specified context."
  ([graph-name invars ast target-heads]
   (compile-ast (uop-rt/init-uop-context :uop-cpu {:quiet? true}) graph-name invars ast target-heads))
  ([ctx graph-name invars ast target-heads]
   (uop-rt/compile-uop-executable ctx graph-name invars ast target-heads)))
