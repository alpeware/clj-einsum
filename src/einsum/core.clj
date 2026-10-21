(ns einsum.core
  "High-level Public Clojure API for Declarative UOp compilation, execution, and scoped memory management."
  (:refer-clojure :exclude [compile])
  (:require [clojure.string :as str]
            [einsum.logic.interpret :as interpret]
            [einsum.runtime.arena :as arena]
            [einsum.runtime.weights :as weights]
            [einsum.uop.lower :as uop-lower]
            [einsum.uop.runtime :as uop-rt])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(def ^:dynamic *default-context* nil)

;; -----------------------------------------------------------------------------
;; Backend Context Management
;; -----------------------------------------------------------------------------

(defn init-backend!
  "Initializes UOp backend (:uop-cpu, :uop-rocm, :uop) or pure-JVM interpreter (:interpreter, :interp).
   Accepts optional `opts` map. Sets and returns the thread-root default context *default-context*."
  ([] (init-backend! :uop-cpu {}))
  ([target] (init-backend! target {}))
  ([target opts]
   (let [ctx (cond
               (contains? #{:interpreter :interp} target)
               {:backend :interpreter
                :target :interpreter
                :platform "interpreter"
                :client nil
                :destroy-fn (fn [_ _] nil)}

               (contains? #{:uop :uop-cpu :uop-rocm} target)
               (uop-rt/init-uop-context! target opts)

               :else
               (uop-rt/init-uop-context! :uop-cpu opts))]
     (alter-var-root #'*default-context* (constantly ctx))
     (when-not (or (Boolean/getBoolean "clj-einsum.quiet") (:quiet? opts))
       (println (format "clj-einsum initialized Backend: [%s]" (or (:platform ctx) (name target)))))
     ctx)))

(defn init-interpreter!
  "Initializes and returns a pure-JVM reference interpreter backend context."
  ([] (init-interpreter! {}))
  ([opts] (init-backend! :interpreter opts)))

(defn init-cpu!
  "Initializes and returns a default UOp CPU backend context."
  ([] (init-cpu! {}))
  ([opts] (init-backend! :uop-cpu opts)))

(defn init-rocm!
  "Initializes and returns a default UOp ROCm backend context."
  ([] (init-rocm! {}))
  ([opts] (init-backend! :uop-rocm opts)))

(defn init-uop!
  "Initializes and returns a native UOp backend context."
  ([] (init-uop! :uop-cpu {}))
  ([target] (init-uop! target {}))
  ([target opts] (init-backend! target opts)))

(defn get-context
  "Returns the currently active backend context, initializing default `:uop-cpu` if nil."
  []
  (or *default-context* (init-backend! :uop-cpu {:quiet? true})))

;; -----------------------------------------------------------------------------
;; Classic Einsum Spec & Planning
;; -----------------------------------------------------------------------------

(defn parse-spec
  "Parses an Einstein summation subscript string (e.g. \"ij,jk->ik\" or \"i,j->ij\")
   into input and output index vector sequences."
  [spec-str]
  (let [spec-str (str/trim spec-str)
        [in-part out-part] (str/split spec-str #"->")
        in-subs (mapv (fn [s] (mapv str (remove str/blank? (str/split (str/trim s) #""))))
                      (str/split in-part #","))
        out-sub (if out-part
                  (mapv str (remove str/blank? (str/split (str/trim out-part) #"")))
                  (let [freqs (frequencies (apply concat in-subs))]
                    (vec (sort (for [[idx cnt] freqs :when (= cnt 1)] idx)))))]
    {:inputs in-subs :output out-sub}))

(defn plan
  "Creates an execution contraction plan from a spec string or parsed spec and operand shapes."
  [spec-str-or-parsed shapes]
  (let [parsed (if (string? spec-str-or-parsed) (parse-spec spec-str-or-parsed) spec-str-or-parsed)
        {:keys [inputs output]} parsed
        idx->dim (reduce (fn [acc [in-idxs shape]]
                           (reduce (fn [m [idx dim]] (assoc m idx dim))
                                   acc
                                   (map vector in-idxs shape)))
                         {}
                         (map vector inputs shapes))
        out-shape (mapv #(get idx->dim %) output)]
    {:spec parsed
     :inputs inputs
     :output output
     :idx->dim idx->dim
     :out-shape out-shape}))

;; -----------------------------------------------------------------------------
;; UOp Compilation & Execution
;; -----------------------------------------------------------------------------

(declare execute)

(defrecord CompiledExecutable [backend name graph invars outvars uop-executable uop-graph ctx schedule module handle prepared-kernels num-slots slot-ptrs intermediate-buffers output-pool exec-turn exec-arena warmed-up? lock]
  java.lang.AutoCloseable
  (close [this]
    (when (or (:uop-executable? this) (:intermediate-buffers this))
      (try
        (uop-rt/destroy-uop-executable! ctx this)
        (catch Exception _ nil))))
  clojure.lang.IFn
  (invoke [this]
    (execute this []))
  (invoke [this a]
    (execute this a))
  (invoke [this a b]
    (execute this [a b]))
  (invoke [this a b c]
    (execute this [a b c]))
  (invoke [this a b c d]
    (execute this [a b c d]))
  (invoke [this a b c d e]
    (execute this [a b c d e]))
  (applyTo [this args]
    (execute this (vec args))))

(defn compile-graph
  "Compiles an AST or lowered UOp graph into an executable for the given or current context."
  ([graph] (compile-graph (get-context) graph))
  ([ctx graph]
   (let [lowered-graph (if (or (contains? graph :outputs) (contains? graph :eqns))
                         graph
                         (uop-lower/ast->uop-graph (:name graph "graph") (:invars graph) (or (:ast graph) graph) (:outvars graph)))
         res (if (contains? lowered-graph :eqns)
               {:backend :interpreter :graph lowered-graph :ctx ctx :invars (:invars lowered-graph) :outvars (:outvars lowered-graph)}
               (let [uop-target (if (or (= (:backend ctx) :interpreter) (= (:target ctx) :interpreter))
                                  :cpu
                                  (or (:uop-target ctx) :cpu))
                     exec-ctx (assoc ctx :uop-target uop-target)]
                 (uop-rt/compile-uop-executable exec-ctx lowered-graph)))]
     (map->CompiledExecutable res))))

(defn compile-kernel
  "Compiles a named kernel AST or graph with signature options into an executable.
   Supports automatic invar inference and weight binding when :weights is passed in `opts`."
  ([ctx name-str ast-or-graph] (compile-kernel ctx name-str ast-or-graph {}))
  ([ctx name-str ast-or-graph opts]
   (let [in-arg (or (:in opts) (:inputs opts) (:invars opts) (:invars ast-or-graph))
         weights-store (or (:weights opts) (:weight-store opts))
         out-arg (or (:out opts) (:outputs opts) (:targets opts) (:outvars ast-or-graph) [:logits])
         out-spec (cond
                    (vector? out-arg) out-arg
                    (set? out-arg) (vec out-arg)
                    (keyword? out-arg) [out-arg]
                    :else (vec out-arg))
         in-spec (cond
                   (and (vector? in-arg) (seq in-arg) (vector? (first in-arg)))
                   (if weights-store
                     (weights/infer-invars ast-or-graph in-arg weights-store)
                     in-arg)

                   weights-store
                   (weights/infer-invars ast-or-graph (or in-arg {}) weights-store)

                   (map? in-arg)
                   (mapv (fn [[k spec]] [k spec]) in-arg)

                   :else
                   (or in-arg []))
         lowered-graph (if (or (contains? ast-or-graph :outputs) (contains? ast-or-graph :eqns))
                         ast-or-graph
                         (uop-lower/ast->uop-graph name-str in-spec ast-or-graph out-spec))
         res (if (contains? lowered-graph :eqns)
               {:backend :interpreter
                :name name-str
                :graph lowered-graph
                :opts opts
                :weights weights-store
                :ctx ctx
                :invars in-spec
                :outvars out-spec}
               (let [uop-target (if (or (= (:backend ctx) :interpreter) (= (:target ctx) :interpreter))
                                  :cpu
                                  (or (:uop-target ctx) :cpu))
                     exec-ctx (assoc ctx :uop-target uop-target)
                     uop-exec (uop-rt/compile-uop-executable exec-ctx name-str in-spec lowered-graph out-spec)]
                 (assoc uop-exec :opts opts :weights weights-store :backend (or (:backend ctx) :uop))))]
     (map->CompiledExecutable res))))

(defn compile
  "Compiles an AST or lowered UOp graph into an executable."
  ([ast-or-graph] (compile-graph (get-context) ast-or-graph))
  ([ctx ast-or-graph] (compile-graph ctx ast-or-graph)))

(defn- normalize-invars
  [raw-invars]
  (let [entries (cond
                  (vector? raw-invars) raw-invars
                  (map? raw-invars) (vec raw-invars)
                  (sequential? raw-invars) (vec raw-invars)
                  :else [])]
    (mapv (fn [entry]
            (if (and (vector? entry) (= 2 (count entry)))
              (let [[v-name spec] entry]
                (cond
                  (vector? spec)
                  [v-name (second spec) (nth spec 2 :f32)]
                  (map? spec)
                  [v-name (:shape spec) (:dtype spec :f32)]
                  :else
                  [v-name nil :f32]))
              [(first entry) nil :f32]))
          entries)))

(defn execute
  "Executes a compiled executable, interpreter graph, or callable with given inputs."
  [exec & inputs]
  (let [ctx (or (when (map? exec) (:ctx exec)) (get-context))
        weights-store (or (:weights exec) (get-in exec [:opts :weights]))
        raw-invars (or (get-in exec [:graph :invars])
                       (get-in exec [:uop-graph :invars])
                       (:invars exec)
                       [])
        norm-invars (normalize-invars raw-invars)
        single-map-input? (and (= 1 (count inputs))
                               (map? (first inputs))
                               (not (:uop-buffer? (first inputs)))
                               (not (contains? (first inputs) :data)))
        input-map (when single-map-input? (first inputs))
        flat-inputs (cond
                      single-map-input?
                      (mapv (fn [[v-name _shape _dtype]]
                              (if (contains? input-map v-name)
                                (get input-map v-name)
                                (when weights-store (weights/get-device-buffer weights-store v-name))))
                            norm-invars)

                      (and (= 1 (count inputs)) (vector? (first inputs)))
                      (let [in-vec (first inputs)]
                        (if (and weights-store (> (count norm-invars) (count in-vec)))
                          (mapv (fn [idx [v-name _shape _dtype]]
                                  (if (< idx (count in-vec))
                                    (nth in-vec idx)
                                    (weights/get-device-buffer weights-store v-name)))
                                (range (count norm-invars))
                                norm-invars)
                          in-vec))

                      :else
                      (let [in-vec (vec inputs)]
                        (if (and weights-store (> (count norm-invars) (count in-vec)))
                          (mapv (fn [idx [v-name _shape _dtype]]
                                  (if (< idx (count in-vec))
                                    (nth in-vec idx)
                                    (weights/get-device-buffer weights-store v-name)))
                                (range (count norm-invars))
                                norm-invars)
                          in-vec)))]
    (cond
      (and (fn? exec) (not (instance? CompiledExecutable exec)))
      (if single-map-input?
        (exec input-map)
        (exec flat-inputs))

      (contains? (:graph exec) :eqns)
      (let [graph (:graph exec)
            bindings (into {}
                           (map-indexed (fn [idx [v-name shape dtype]]
                                          (let [val (nth flat-inputs idx)
                                                t-val (cond
                                                        (and (map? val) (contains? val :data))
                                                        val

                                                        (and (map? val) (:uop-buffer? val))
                                                        {:dtype (or dtype :f32)
                                                         :shape (when shape (vec shape))
                                                         :data (uop-rt/device->host-floats ctx val (reduce * 1 (or shape [1])))}

                                                        :else
                                                        {:dtype (weights/normalize-dtype (or dtype :f32))
                                                         :shape (when shape (vec shape))
                                                         :data val})]
                                            [v-name t-val]))
                                        norm-invars))
            out-map (interpret/execute graph bindings)
            out-spec (or (:outvars exec)
                         (:outvars graph)
                         (vec (keys out-map)))]
        (cond
          single-map-input?
          out-map

          (= 1 (count out-spec))
          (get out-map (first out-spec))

          :else
          (mapv #(get out-map %) out-spec)))

      :else
      (let [out-map (uop-rt/execute-uop-executable ctx exec flat-inputs)
            out-spec (or (:outvars exec)
                         (when-let [outs (get-in exec [:uop-graph :outputs])] (vec (keys outs)))
                         (vec (keys out-map)))]
        (cond
          single-map-input?
          out-map

          (= 1 (count out-spec))
          (get out-map (first out-spec))

          :else
          (mapv #(get out-map %) out-spec))))))

(def execute! execute)

;; -----------------------------------------------------------------------------
;; Scoped Arena & Buffer Transfers
;; -----------------------------------------------------------------------------

(defn create-arena
  ([] (create-arena (get-context)))
  ([ctx] (arena/create-arena ctx)))

(defn close-arena! [arena] (arena/close-arena! arena))

(defmacro with-device-arena [[binding-sym & [ctx]] & body]
  `(arena/with-device-arena [~binding-sym ~(or ctx `(get-context))]
     ~@body))

(defn device-buffer
  ([data shape dtype] (device-buffer (get-context) data shape dtype {}))
  ([ctx-or-arena data shape dtype]
   (if (arena/arena? ctx-or-arena)
     (arena/device-buffer ctx-or-arena data shape dtype)
     (device-buffer ctx-or-arena data shape dtype {})))
  ([ctx data shape dtype _opts]
   (uop-rt/host->device ctx data shape dtype)))

(defn track! [arena buf] (arena/track! arena buf))
(defn disown! [arena buf] (arena/disown! arena buf))
(defn promote! [src-arena dst-arena buf] (arena/promote! src-arena dst-arena buf))
(defn destroy! [buf] (arena/destroy! buf))
(defn destroy-buffer!
  ([buf] (destroy-buffer! (get-context) buf))
  ([ctx buf] (uop-rt/destroy-buffer! ctx buf)))

(defn create-weight-store
  ([weights-map] (create-weight-store (get-context) weights-map {}))
  ([ctx weights-map] (create-weight-store ctx weights-map {}))
  ([ctx weights-map opts] (weights/create-weight-store ctx weights-map opts)))

(def ^:private ^Class FLOAT-ARRAY-CLASS (Class/forName "[F"))
(def ^:private ^Class SHORT-ARRAY-CLASS (Class/forName "[S"))
(def ^:private ^Class INT-ARRAY-CLASS (Class/forName "[I"))

(defn to-host-slice
  "Transfers a slice of UOp device buffer, primitive array, or interpreter tensor back to host float array."
  ([out-buf] (to-host-slice out-buf 0 256000 (* 128 256000) :f32))
  ([out-buf slice-idx] (to-host-slice out-buf slice-idx 256000 (* 128 256000) :f32))
  ([out-buf slice-idx vocab-size] (to-host-slice out-buf slice-idx vocab-size (* 128 vocab-size) :f32))
  ([out-buf slice-idx vocab-size total-elements] (to-host-slice out-buf slice-idx vocab-size total-elements :f32))
  ([out-buf slice-idx vocab-size total-elements dtype]
   (let [ctx (get-context)
         target-buf (if (and (map? out-buf)
                             (not (:uop-buffer? out-buf))
                             (not (contains? out-buf :data))
                             (pos? (count out-buf)))
                      (first (vals out-buf))
                      out-buf)
         n-elements (max (long total-elements) (long (* (inc slice-idx) vocab-size)))
         all-floats (cond
                      (instance? FLOAT-ARRAY-CLASS target-buf)
                      ^floats target-buf

                      (and (map? target-buf) (true? (:uop-buffer? target-buf)))
                      ^floats (uop-rt/device->host-floats ctx target-buf n-elements)

                      (and (map? target-buf) (contains? target-buf :data))
                      (let [data (:data target-buf)
                            dt (or (:dtype target-buf) dtype :f32)]
                        (cond
                          (instance? FLOAT-ARRAY-CLASS data) ^floats data
                          (instance? SHORT-ARRAY-CLASS data)
                          (let [^shorts s-arr data
                                len (alength s-arr)
                                f-arr (float-array len)]
                            (dotimes [i len] (aset f-arr i (interpret/bf16->f32 (aget s-arr i))))
                            f-arr)
                          (instance? MemorySegment data)
                          (let [^MemorySegment seg data
                                elem-size (if (= dt :bf16) 2 4)
                                len (long (min (quot (.byteSize seg) elem-size) n-elements))
                                f-arr (float-array len)]
                            (if (= dt :bf16)
                              (dotimes [i len] (aset f-arr i (interpret/bf16->f32 (.getAtIndex seg ValueLayout/JAVA_SHORT (long i)))))
                              (dotimes [i len] (aset f-arr i (.getAtIndex seg ValueLayout/JAVA_FLOAT (long i)))))
                            f-arr)
                          :else (float-array (map float data))))

                      (instance? MemorySegment target-buf)
                      (let [^MemorySegment seg target-buf
                            len (long (min (quot (.byteSize seg) 4) n-elements))
                            f-arr (float-array len)]
                        (dotimes [i len] (aset f-arr i (.getAtIndex seg ValueLayout/JAVA_FLOAT (long i))))
                        f-arr)

                      :else
                      (float-array (map float (if (sequential? target-buf) target-buf [target-buf]))))
         start-idx (* slice-idx vocab-size)
         slice-len (min (long vocab-size) (long (max 0 (- (alength ^floats all-floats) start-idx))))
         target (float-array vocab-size)]
     (when (pos? slice-len)
       (if (instance? INT-ARRAY-CLASS all-floats)
         (let [^ints i-arr (cast ints all-floats)]
           (dotimes [i slice-len] (aset target i (float (aget i-arr (+ start-idx i))))))
         (System/arraycopy all-floats (int start-idx) target 0 (int slice-len))))
     target)))

(defn buffer-from-host-buffer
  "Compatibility helper for allocating and transferring host buffer to device memory."
  ([ctx data shape dtype-or-code]
   (uop-rt/buffer-from-host-buffer ctx data shape dtype-or-code))
  ([ctx client data shape dtype-or-code]
   (uop-rt/buffer-from-host-buffer ctx client data shape dtype-or-code)))

(defn buffer-to-host-buffer
  "Compatibility helper for transferring device memory back to host floats."
  ([ctx buf] (uop-rt/buffer-to-host-buffer ctx buf))
  ([ctx buf n-elements] (uop-rt/buffer-to-host-buffer ctx buf n-elements))
  ([ctx buf n-elements dtype] (uop-rt/buffer-to-host-buffer ctx buf n-elements dtype)))

(defn execute-executable
  "Compatibility helper for executing compiled executables."
  ([ctx exec inputs] (uop-rt/execute-executable ctx exec inputs))
  ([ctx exec inputs n-outputs] (uop-rt/execute-executable ctx exec inputs n-outputs)))

