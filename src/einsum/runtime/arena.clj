(ns einsum.runtime.arena
  "Scoped Device Memory Arenas for deterministic, leak-free buffer management.
   Inspired by Project Panama FFM Arenas (java.lang.foreign.Arena) and babashka.ffi."
  (:require [einsum.uop.runtime :as uop-rt])
  (:import [java.lang.foreign MemorySegment]
           [java.util.concurrent ConcurrentHashMap]
           [java.util.concurrent.atomic AtomicBoolean]))

(def ^:dynamic *active-arena*
  "Dynamically bound active DeviceArena within a `with-device-arena` scope."
  nil)

(def DTYPE->ENUM
  "Maps symbolic data type keywords to buffer type enum integers."
  {:i1 1 :bool 1 :pred 1
   :i8 2 :byte 2
   :i32 4 :int 4
   :f16 10
   :f32 11 :float 11
   :f64 12 :double 12
   :bf16 13 :bfloat16 13})

(defn dtype->enum
  "Converts keyword `dt` or integer to data type enum integer."
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
        (uop-rt/uop-buffer? buf)
        (uop-rt/destroy-buffer! ctx buf)
        (:destroy-fn ctx) ((:destroy-fn ctx) ctx buf)
        (:mock-destroy ctx) ((:mock-destroy ctx) ctx buf)
        :else nil)
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
  (when arena
    (.get ^AtomicBoolean (:closed? arena))))

(defn close-arena!
  "Closes the `arena`, destroying all tracked resident device buffers."
  [^DeviceArena arena]
  (when (and arena (not (closed? arena)))
    (.close arena)))

(def close! close-arena!)

(defn create-arena
  "Creates a new scoped DeviceArena associated with backend `ctx` and optional `parent-arena`."
  ([] (create-arena nil nil))
  ([ctx] (create-arena ctx nil))
  ([ctx parent-arena]
   (->DeviceArena ctx (ConcurrentHashMap/newKeySet) parent-arena (AtomicBoolean. false))))

(defn track!
  "Registers device buffer `buf` to be owned and freed by `arena` upon scope exit.
   Throws ExceptionInfo if `arena` is closed."
  [^DeviceArena arena buf]
  (when (and arena buf)
    (if (closed? arena)
      (throw (ex-info "Cannot track buffer in closed DeviceArena"
                      {:arena arena :buffer buf}))
      (do (.add ^java.util.Set (:buffers-set arena) buf)
          buf))))

(defn disown!
  "Removes device buffer `buf` from `arena` management without destroying it."
  [^DeviceArena arena buf]
  (when (and arena buf)
    (.remove ^java.util.Set (:buffers-set arena) buf)
    buf))

(defn promote!
  "Transfers ownership of device buffer or sequential collection of buffers `buf-or-bufs`
   from `source-arena` to `target-arena`."
  [^DeviceArena source-arena ^DeviceArena target-arena buf-or-bufs]
  (when (and source-arena target-arena buf-or-bufs)
    (if (sequential? buf-or-bufs)
      (doseq [b buf-or-bufs]
        (promote! source-arena target-arena b))
      (do (disown! source-arena buf-or-bufs)
          (track! target-arena buf-or-bufs)))
    buf-or-bufs))

(defn destroy!
  "Destroys a resident device buffer immediately and disowns it from its arena."
  ([buf]
   (when *active-arena*
     (disown! *active-arena* buf))
   (let [ctx (or (when *active-arena* (:ctx *active-arena*))
                 (when-let [v (resolve 'einsum.core/*default-context*)] @v)
                 (when-let [f (resolve 'einsum.core/get-context)] (f))
                 {})]
     (destroy-device-buffer! ctx buf)))
  ([arena-or-ctx buf]
   (if (arena? arena-or-ctx)
     (let [arena ^DeviceArena arena-or-ctx]
       (disown! arena buf)
       (destroy-device-buffer! (:ctx arena) buf))
     (destroy-device-buffer! arena-or-ctx buf))))

(defn count-buffers
  "Returns the number of resident buffers currently tracked by `arena`."
  [^DeviceArena arena]
  (if (or (nil? arena) (closed? arena))
    0
    (.size ^java.util.Set (:buffers-set arena))))

(defn tracked-buffers
  "Returns a persistent set of all buffers currently tracked by `arena`."
  [^DeviceArena arena]
  (if (or (nil? arena) (closed? arena))
    #{}
    (set (:buffers-set arena))))

(defn device-buffer
  "Allocates a device buffer for `host-data`, tracking it within `arena` (or *active-arena*).
   Returns the device buffer."
  ([host-data shape dtype]
   (device-buffer *active-arena* host-data shape dtype))
  ([^DeviceArena arena host-data shape dtype]
   (let [ctx (or (:ctx arena)
                 (when-let [v (resolve 'einsum.core/*default-context*)] @v)
                 (when-let [f (resolve 'einsum.core/get-context)] (f))
                 {})]
     (cond
       (= (:backend ctx) :interpreter)
       (let [buf {:dtype (or dtype :f32)
                  :shape (vec shape)
                  :data host-data}]
         (when arena (track! arena buf))
         buf)

       :else
       (let [buf (uop-rt/host->device ctx host-data shape dtype)]
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
