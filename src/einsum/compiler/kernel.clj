(ns einsum.compiler.kernel
  "Callable Compiled Kernel abstraction for OpenXLA PJRT.
   Wraps AST lowering, compilation, device buffer marshaling, and execution
   into an idiomatic, invokable Clojure function (clojure.lang.IFn)."
  (:require [einsum.compiler.compile :as compile]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.logic.lower :as lower]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.weights :as weights])
  (:import [java.lang.foreign MemorySegment]))

(declare run-kernel)

(defrecord CompiledKernel [name ast graph exec in-spec in-keys out-spec ctx opts]
  clojure.lang.IFn
  (invoke [this m]
    (cond
      (map? m) (run-kernel this m :map)
      (sequential? m) (run-kernel this (vec m) :positional)
      :else (run-kernel this [m] :positional)))
  (invoke [this a b]
    (run-kernel this [a b] :positional))
  (invoke [this a b c]
    (run-kernel this [a b c] :positional))
  (invoke [this a b c d]
    (run-kernel this [a b c d] :positional))
  (invoke [this a b c d e]
    (run-kernel this [a b c d e] :positional))
  (applyTo [this args]
    (cond
      (and (= 1 (count args)) (map? (first args)))
      (run-kernel this (first args) :map)

      (and (= 1 (count args)) (sequential? (first args)))
      (run-kernel this (vec (first args)) :positional)

      :else
      (run-kernel this (vec args) :positional)))

  java.lang.AutoCloseable
  (close [_this]
    (when-let [handle (or (:handle exec) exec)]
      (try
        (pjrt/destroy-loaded-executable! ctx handle)
        (catch Exception _ nil)))))

(defn- to-device-buffer [ctx _in-var spec val]
  (cond
    (instance? MemorySegment val)
    val

    :else
    (let [tensor-spec (if (and (vector? spec) (= (count spec) 2) (keyword? (first spec)) (vector? (second spec)))
                        (second spec)
                        spec)
          [_tag shape dtype] tensor-spec
          dt-enum (arena/dtype->enum (or dtype :f32))
          cli (or (:client ctx) ctx)
          buf (pjrt/buffer-from-host-buffer ctx cli val shape dt-enum)]
      (when arena/*active-arena*
        (arena/track! arena/*active-arena* buf))
      buf)))

(defn run-kernel
  "Executes `kernel` with `inputs` (either a keyword-indexed map or ordered vector)."
  [kernel inputs call-style]
  (let [{:keys [exec in-spec in-keys out-spec ctx opts]} kernel
        exec-handle (or (:handle exec) exec)
        spec-map (into {} in-spec)
        weights-store (or (:weights opts) (:weight-store opts))

        ;; 1. Collect inputs in the exact canonical invars order
        input-buffers (if (= call-style :map)
                        (mapv (fn [k]
                                (if (contains? inputs k)
                                  (to-device-buffer ctx k (get spec-map k) (get inputs k))
                                  (if (and weights-store (weights/weight-store? weights-store))
                                    (weights/get-device-buffer weights-store k)
                                    (throw (ex-info (str "Missing required kernel input: " k)
                                                    {:required k :available (keys inputs) :kernel-name (:name kernel)})))))
                              in-keys)
                        (mapv (fn [idx val]
                                (let [k (nth in-keys idx)
                                      spec (nth in-spec idx)]
                                  (to-device-buffer ctx k spec val)))
                              (range (count inputs))
                              inputs))

        ;; 2. Execute on PJRT device
        num-outs (max 1 (count out-spec))
        raw-outs (pjrt/execute-executable ctx exec-handle input-buffers num-outs)
        outs-vec (if (vector? raw-outs) raw-outs [raw-outs])]

    ;; 3. Register outputs in active arena if bound
    (when arena/*active-arena*
      (doseq [b outs-vec]
        (arena/track! arena/*active-arena* b)))

    ;; 4. Package return value based on call-style and output spec
    (cond
      (= call-style :map)
      (into {} (map vector out-spec outs-vec))

      (= 1 (count out-spec))
      (first outs-vec)

      :else
      outs-vec)))

(defn compile-kernel
  "Compiles a Tensor Logic Hiccup AST into an invokable, callable CompiledKernel.
   Supports automatic invar inference when :weights is passed in `opts`."
  ([name-str ast opts]
   (let [ctx (if-let [f (resolve 'einsum.core/get-context)]
               (f)
               {})]
     (compile-kernel ctx name-str ast opts)))
  ([ctx name-str ast opts]
   (let [in-arg (or (:in opts) (:inputs opts) (:invars opts))
         weights-store (or (:weights opts) (:weight-store opts))
         out-arg (or (:out opts) (:outputs opts) (:targets opts) [:logits])
         out-spec (cond
                    (vector? out-arg) out-arg
                    (set? out-arg) (vec out-arg)
                    (keyword? out-arg) [out-arg]
                    :else (vec out-arg))

         ;; Derive invars either directly or via weights/infer-invars
         in-spec (cond
                   (and (vector? in-arg) (seq in-arg) (vector? (first in-arg)))
                   in-arg

                   weights-store
                   (weights/infer-invars ast (or in-arg {}) weights-store)

                   (map? in-arg)
                   (mapv (fn [[k spec]] [k spec]) in-arg)

                   :else
                   (throw (ex-info "Missing or invalid :in / :inputs specification for compile-kernel"
                                   {:opts opts})))

         in-keys (mapv first in-spec)

         ;; Lower AST to validated StableHLO graph
         graph (lower/ast->graph name-str in-spec ast out-spec)

         ;; Compile graph to native PJRT executable
         cli (or (:client ctx) ctx)
         exec (compile/compile-graph ctx cli graph)]
     (->CompiledKernel name-str ast graph exec in-spec in-keys out-spec ctx opts))))

(defmacro defkernel
  "Defines a var bound to a compiled kernel function."
  ([name-sym ast opts]
   `(def ~name-sym (compile-kernel (name '~name-sym) ~ast ~opts)))
  ([name-sym ctx ast opts]
   `(def ~name-sym (compile-kernel ~ctx (name '~name-sym) ~ast ~opts))))
