(ns einsum.runtime.arena
  "Scoped Device Memory Arenas for deterministic, leak-free OpenXLA PJRT buffer management.
   Inspired by Project Panama FFM Arenas (java.lang.foreign.Arena) and babashka.ffi."
  (:require [einsum.compiler.pjrt :as pjrt])
  (:import [java.lang.foreign MemorySegment]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicBoolean]))

(def ^:dynamic *active-arena*
  "Dynamically bound active DeviceArena within a `with-device-arena` scope."
  nil)

(def DTYPE->ENUM
  "Maps symbolic data type keywords to PJRT buffer type enum integers."
  {:i1 1 :bool 1 :pred 1
   :i8 2 :byte 2
   :i32 4 :int 4
   :f16 10
   :f32 11 :float 11
   :f64 12 :double 12
   :bf16 13 :bfloat16 13})

(defn dtype->enum
  "Converts keyword `dt` or integer to PJRT data type enum integer."
  [dt]
  (cond
    (integer? dt) (int dt)
    (keyword? dt) (get DTYPE->ENUM dt 11)
    :else 11))

(defn- destroy-device-buffer!
  [ctx buf]
  (when (and (some? buf) (not= MemorySegment/NULL buf))
    (try
      (cond
        (:destroy-fn ctx) ((:destroy-fn ctx) ctx buf)
        (:mock-destroy ctx) ((:mock-destroy ctx) ctx buf)
        :else (pjrt/destroy-buffer! ctx buf))
      (catch Exception _ nil))))

(defrecord DeviceArena [ctx ^java.util.Set buffers-set parent-arena ^AtomicBoolean closed?]
  java.lang.AutoCloseable
  (close [_this]
    (when (.compareAndSet closed? false true)
      (doseq [buf (vec buffers-set)]
        (destroy-device-buffer! ctx buf))
      (.clear buffers-set))))

(defn arena?
  "Returns true if `x` is an instance of DeviceArena."
  [x]
  (instance? DeviceArena x))

(defn closed?
  "Returns true if `arena` has been closed."
  [^DeviceArena arena]
  (if arena (.get ^AtomicBoolean (:closed? arena)) true))

(defn tracked-buffers
  "Returns a persistent set of all device buffers currently tracked by `arena`."
  [^DeviceArena arena]
  (if arena
    (set (vec ^java.util.Set (:buffers-set arena)))
    #{}))

(defn create-arena
  "Creates an open DeviceArena associated with PJRT `ctx` and optional `parent-arena`."
  ([ctx]
   (create-arena ctx nil))
  ([ctx parent-arena]
   (let [ctx (cond
               (arena? ctx) (:ctx ctx)
               :else ctx)
         parent (cond
                  (arena? parent-arena) parent-arena
                  (arena? ctx) ctx
                  :else nil)]
     (->DeviceArena ctx (ConcurrentHashMap/newKeySet) parent (AtomicBoolean. false)))))

(defn close!
  "Explicitly closes `arena`, destroying all tracked device buffers."
  [^DeviceArena arena]
  (when arena
    (.close arena)))

(defn track!
  "Registers `buffer` (or collection of buffers) in `arena` (or `*active-arena*`).
   Throws ExceptionInfo if `arena` is closed. Returns `buffer`."
  ([buffer]
   (track! *active-arena* buffer))
  ([^DeviceArena arena buffer]
   (when arena
     (when (closed? arena)
       (throw (ex-info "Cannot track buffer in closed DeviceArena"
                       {:arena arena :buffer buffer})))
     (let [^java.util.Set set-ref (:buffers-set arena)]
       (if (sequential? buffer)
         (doseq [b buffer] (when b (.add set-ref b)))
         (when buffer (.add set-ref buffer)))))
   buffer))

(defn disown!
  "Removes `buffer` (or collection of buffers) from `arena` (or `*active-arena*`),
   preventing it from being destroyed when the arena closes. Returns `buffer`."
  ([buffer]
   (disown! *active-arena* buffer))
  ([^DeviceArena arena buffer]
   (when arena
     (let [^java.util.Set set-ref (:buffers-set arena)]
       (if (sequential? buffer)
         (doseq [b buffer] (when b (.remove set-ref b)))
         (when buffer (.remove set-ref buffer)))))
   buffer))

(defn promote!
  "Transfers ownership of `buffer` (or collection of buffers) from `source-arena`
   to `target-arena`. The buffer survives when `source-arena` closes."
  ([target-arena buffer]
   (promote! *active-arena* target-arena buffer))
  ([^DeviceArena source-arena ^DeviceArena target-arena buffer]
   (disown! source-arena buffer)
   (track! target-arena buffer)
   buffer))

(defn destroy!
  "Immediately destroys `buffer` and removes it from `arena` tracking."
  ([buffer]
   (destroy! *active-arena* buffer))
  ([^DeviceArena arena buffer]
   (when buffer
     (disown! arena buffer)
     (if (sequential? buffer)
       (doseq [b buffer] (destroy-device-buffer! (:ctx arena) b))
       (destroy-device-buffer! (:ctx arena) buffer)))
   nil))

(defn device-buffer
  "Transfers host data into a PJRT device buffer or interpreter tensor and tracks it in `arena` (or `*active-arena*`)."
  ([host-data shape dtype]
   (device-buffer *active-arena* host-data shape dtype))
  ([^DeviceArena arena host-data shape dtype]
   (let [ctx (or (:ctx arena)
                 (when-let [v (resolve 'einsum.core/*default-context*)] @v)
                 (when-let [f (resolve 'einsum.core/get-context)] (f))
                 {})]
     (if (= (:backend ctx) :interpreter)
       (let [buf {:dtype (or dtype :f32)
                  :shape (vec shape)
                  :data host-data}]
         (when arena (track! arena buf))
         buf)
       (let [cli (or (:client ctx) (:client arena))
             dt-enum (dtype->enum dtype)
             buf (pjrt/buffer-from-host-buffer ctx cli host-data shape dt-enum)]
         (when arena (track! arena buf))
         buf)))))

(defmacro with-device-arena
  "Executes `body` in a scoped DeviceArena. Automatically and deterministically
   destroys all tracked device buffers when the lexical block exits (even on exception)."
  [[arena-sym & [ctx-or-parent]] & body]
  `(let [arg# ~ctx-or-parent
         ctx# (cond
                (arena? arg#) (:ctx arg#)
                (nil? arg#) (try
                              (if-let [f# (resolve 'einsum.core/get-context)]
                                (f#)
                                {})
                              (catch Exception _# {}))
                :else arg#)
         parent# (when (arena? arg#) arg#)
         ~arena-sym (create-arena ctx# parent#)]
     (binding [*active-arena* ~arena-sym]
       (try
         ~@body
         (finally
           (.close ~arena-sym))))))
