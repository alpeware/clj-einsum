(ns einsum.logic.core
  "High-level Public Clojure API for compiling Tensor Logic ASTs into OpenXLA PJRT executables."
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [einsum.compiler.stablehlo :as shlo]))

(defn compile-ast
  "Compiles a Tensor Logic Hiccup AST into a PJRT loaded executable.
   Arity [graph-name invars ast target-heads]: uses default PJRT context.
   Arity [ctx graph-name invars ast target-heads]: uses specified context."
  ([graph-name invars ast target-heads]
   (compile-ast (xla/get-context) graph-name invars ast target-heads))
  ([ctx graph-name invars ast target-heads]
   (let [graph (lower/ast->graph graph-name invars ast target-heads)]
     (xla/compile-graph ctx graph))))

(defn ast->mlir-text
  "Lowers AST to StableHLO SSA graph and returns serialized MLIR textual module string."
  [graph-name invars ast target-heads]
  (let [graph (lower/ast->graph graph-name invars ast target-heads)]
    (shlo/graph->mlir-text graph)))
