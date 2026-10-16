(ns einsum.compiler.kernel
  "Callable Compiled Kernel abstraction for OpenXLA PJRT.
   Wraps AST lowering, compilation, device buffer marshaling, and execution
   into an idiomatic, invokable Clojure function (clojure.lang.IFn)."
  (:require [einsum.compiler.compile :as compile]
            [einsum.compiler.fuse :as fuse]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.logic.interpret :as interpret]
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
    (cond
      (= (:backend ctx) :interpreter) nil
      (= (:backend ctx) :uop)
      (when-let [destroy-fn (requiring-resolve 'einsum.uop.runtime/destroy-uop-executable!)]
        (try (destroy-fn ctx exec) (catch Exception _ nil)))
      :else
      (when-let [handle (or (:handle exec) exec)]
        (try
          (pjrt/destroy-loaded-executable! ctx handle)
          (catch Exception _ nil))))))

(defn- to-device-buffer [ctx _in-var spec val]
  (cond
    (= (:backend ctx) :interpreter)
    (let [tensor-spec (if (and (vector? spec) (= (count spec) 2) (keyword? (first spec)) (vector? (second spec)))
                        (second spec)
                        spec)
          [_tag shape dtype] tensor-spec
          buf (cond
                (and (map? val) (contains? val :data))
                val

                :else
                {:dtype (or dtype :f32)
                 :shape (vec (or shape []))
                 :data val})]
      (when arena/*active-arena*
        (arena/track! arena/*active-arena* buf))
      buf)

    (= (:backend ctx) :uop)
    (let [uop-buf? (requiring-resolve 'einsum.uop.runtime/uop-buffer?)]
      (if (uop-buf? val)
        val
        (let [tensor-spec (if (and (vector? spec) (= (count spec) 2) (keyword? (first spec)) (vector? (second spec)))
                            (second spec)
                            spec)
              [_tag shape dtype] tensor-spec
              host->dev (requiring-resolve 'einsum.uop.runtime/host->device)
              buf (host->dev ctx val (vec (or shape [])) (or dtype :f32))]
          (when arena/*active-arena*
            (arena/track! arena/*active-arena* buf))
          buf)))

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
  "Executes `kernel` with `inputs` (either a keyword-indexed map or ordered vector).
   - Map-style `(kernel {:x input})`: Automatically resolves remaining parameters from `:weights` Store.
   - Positional vector `(kernel [x w1 w2 ...])`: High-throughput, zero-overhead manual assembly fast-path."
  [kernel inputs call-style]
  (let [{:keys [exec in-spec in-keys out-spec ctx opts]} kernel
        spec-map (into {} in-spec)
        weights-store (or (:weights opts) (:weight-store opts))
        interpreter? (= (:backend ctx) :interpreter)
        uop? (= (:backend ctx) :uop)

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
                              inputs))]

    (cond
      interpreter?
      ;; 2a. Execute on pure-JVM StableHLO interpreter
      (let [bindings (zipmap in-keys input-buffers)
            out-map (interpret/execute (:graph exec) bindings)
            outs-vec (mapv #(get out-map %) out-spec)]
        (when arena/*active-arena*
          (doseq [b outs-vec]
            (arena/track! arena/*active-arena* b)))
        (cond
          (= call-style :map)
          (into {} (map vector out-spec outs-vec))

          (= 1 (count out-spec))
          (first outs-vec)

          :else
          outs-vec))

      uop?
      ;; 2b. Execute on native UOp backend (:uop-cpu or :uop-rocm)
      (let [exec-fn (requiring-resolve 'einsum.uop.runtime/execute-uop-executable)
            out-map (exec-fn ctx exec input-buffers)
            outs-vec (mapv #(get out-map %) out-spec)]
        (when arena/*active-arena*
          (doseq [b outs-vec]
            (arena/track! arena/*active-arena* b)))
        (cond
          (= call-style :map)
          (into {} (map vector out-spec outs-vec))

          (= 1 (count out-spec))
          (first outs-vec)

          :else
          outs-vec))

      :else
      ;; 2c. Execute on PJRT device
      (let [exec-handle (or (:handle exec) exec)
            num-outs (max 1 (count out-spec))
            raw-outs (pjrt/execute-executable ctx exec-handle input-buffers num-outs)
            outs-vec (if (vector? raw-outs) raw-outs [raw-outs])]
        ;; Register outputs in active arena if bound
        (when arena/*active-arena*
          (doseq [b outs-vec]
            (arena/track! arena/*active-arena* b)))
        (cond
          (= call-style :map)
          (into {} (map vector out-spec outs-vec))

          (= 1 (count out-spec))
          (first outs-vec)

          :else
          outs-vec)))))

(defn compile-kernel
  "Compiles a Tensor Logic Hiccup AST into an invokable, callable CompiledKernel.
   - When `:weights` is passed in `opts`, automatically infers missing input signatures via `weights/infer-invars`
     and enables seamless map-style execution `(kernel {:x data})`.
   - When explicit `:in` vector is provided without `:weights`, returns a positional kernel function."
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

         ;; Lower AST to validated StableHLO graph (or UOp executable when :backend is :uop)
         graph (when-not (= (:backend ctx) :uop)
                 (lower/ast->graph name-str in-spec ast out-spec))

         ;; Compile graph to native PJRT executable, UOp executable, or pure-JVM interpreter executable
         exec (cond
                (= (:backend ctx) :interpreter)
                (let [fused (fuse/fuse-graph graph)]
                  {:handle fused
                   :graph fused
                   :backend :interpreter
                   :ctx ctx})

                (= (:backend ctx) :uop)
                (let [comp-fn (requiring-resolve 'einsum.uop.runtime/compile-uop-executable)]
                  (comp-fn ctx name-str in-spec ast out-spec))

                :else
                (let [cli (or (:client ctx) ctx)]
                  (compile/compile-graph ctx cli graph)))]
     (->CompiledKernel name-str ast (or graph (:uop-graph exec)) exec in-spec in-keys out-spec ctx opts))))

(defmacro defkernel
  "Defines a var bound to a compiled kernel function."
  ([name-sym ast opts]
   `(def ~name-sym (compile-kernel (name '~name-sym) ~ast ~opts)))
  ([name-sym ctx ast opts]
   `(def ~name-sym (compile-kernel ~ctx (name '~name-sym) ~ast ~opts))))
