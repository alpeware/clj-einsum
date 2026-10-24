(ns einsum.graph.cas
  "Content-Addressed Storage (CAS) Pointer Store for off-heap and immutable data structures."
  (:require [einsum.graph.schema :as schema])
  (:import [java.security MessageDigest]))

;; -----------------------------------------------------------------------------
;; 1. Cryptographic Digest Helpers
;; -----------------------------------------------------------------------------

(def ^:private ^java.util.HexFormat hex-format (java.util.HexFormat/of))

(defn- digest->hex
  ^String [^bytes digest]
  (.formatHex hex-format digest))

(defn sha256-bytes
  "Computes a 64-character lowercase hexadecimal SHA-256 digest of byte array b."
  [^bytes b]
  (let [md (MessageDigest/getInstance "SHA-256")]
    (digest->hex (.digest md b))))

(defn sha256-str
  "Computes a 64-character lowercase hexadecimal SHA-256 digest of string s (UTF-8)."
  [^String s]
  (sha256-bytes (.getBytes s "UTF-8")))

;; -----------------------------------------------------------------------------
;; 2. CAS Pointer Predicate
;; -----------------------------------------------------------------------------

(defn cas-ptr?
  "Returns true if x conforms to a CAS pointer map."
  [x]
  (schema/cas-ptr? x))

(def ^:private float-array-class (Class/forName "[F"))
(def ^:private int-array-class (Class/forName "[I"))
(def ^:private long-array-class (Class/forName "[J"))
(def ^:private double-array-class (Class/forName "[D"))
(def ^:private short-array-class (Class/forName "[S"))

(defn primitive-array?
  "Returns true if x is a primitive Java array ([F, [I, [J, [D, [S, or [B)."
  [x]
  (or (instance? float-array-class x)
      (instance? int-array-class x)
      (instance? long-array-class x)
      (instance? double-array-class x)
      (instance? short-array-class x)
      (bytes? x)))

(defn- argmax-float-array
  ^long [^floats arr]
  (let [n (alength arr)]
    (if (zero? n)
      0
      (loop [i 1
             max-idx 0
             max-val (aget arr 0)]
        (if (< i n)
          (let [v (aget arr i)]
            (if (> v max-val)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx)))))

(defn- argmax-int-array
  ^long [^ints arr]
  (let [n (alength arr)]
    (if (zero? n)
      0
      (loop [i 1
             max-idx 0
             max-val (aget arr 0)]
        (if (< i n)
          (let [v (aget arr i)]
            (if (> v max-val)
              (recur (inc i) i v)
              (recur (inc i) max-idx max-val)))
          max-idx)))))

(def ^:private ^ThreadLocal tl-scratch
  (ThreadLocal/withInitial
   (reify java.util.function.Supplier
     (get [_]
       (let [ba (byte-array 16384)
             bb (.order (java.nio.ByteBuffer/wrap ba) java.nio.ByteOrder/LITTLE_ENDIAN)]
         {:ba ba
          :fb (.asFloatBuffer bb)
          :ib (.asIntBuffer bb)
          :lb (.asLongBuffer bb)
          :db (.asDoubleBuffer bb)
          :sb (.asShortBuffer bb)
          :md (MessageDigest/getInstance "SHA-256")})))))

(defn- digest-primitive-array
  "Computes the big-endian SHA-256 hex digest, byte length, shape, and optional argmax
   of a primitive array using a zero-allocation thread-local 16 KB L1-cache scratch buffer."
  [val]
  (let [{:keys [^bytes ba ^java.nio.FloatBuffer fb ^java.nio.IntBuffer ib
                ^java.nio.LongBuffer lb ^java.nio.DoubleBuffer db
                ^java.nio.ShortBuffer sb ^MessageDigest md]} (.get tl-scratch)]
    (.reset md)
    (cond
      (instance? float-array-class val)
      (let [arr ^floats val
            n (alength arr)]
        (loop [off 0]
          (when (< off n)
            (let [chunk (min 4096 (- n off))]
              (.clear fb)
              (.put fb arr off chunk)
              (.update md ba 0 (* 4 chunk))
              (recur (+ off chunk)))))
        {:hex (digest->hex (.digest md))
         :bytes-cnt (* 4 n)
         :kind :tensor/f32
         :shape [n]
         :argmax (when (pos? n) (int (argmax-float-array arr)))})

      (instance? int-array-class val)
      (let [arr ^ints val
            n (alength arr)]
        (loop [off 0]
          (when (< off n)
            (let [chunk (min 4096 (- n off))]
              (.clear ib)
              (.put ib arr off chunk)
              (.update md ba 0 (* 4 chunk))
              (recur (+ off chunk)))))
        {:hex (digest->hex (.digest md))
         :bytes-cnt (* 4 n)
         :kind :tensor/i32
         :shape [n]
         :argmax (when (pos? n) (int (argmax-int-array arr)))})

      (instance? long-array-class val)
      (let [arr ^longs val
            n (alength arr)]
        (loop [off 0]
          (when (< off n)
            (let [chunk (min 2048 (- n off))]
              (.clear lb)
              (.put lb arr off chunk)
              (.update md ba 0 (* 8 chunk))
              (recur (+ off chunk)))))
        {:hex (digest->hex (.digest md))
         :bytes-cnt (* 8 n)
         :kind :tensor/i64
         :shape [n]})

      (instance? double-array-class val)
      (let [arr ^doubles val
            n (alength arr)]
        (loop [off 0]
          (when (< off n)
            (let [chunk (min 2048 (- n off))]
              (.clear db)
              (.put db arr off chunk)
              (.update md ba 0 (* 8 chunk))
              (recur (+ off chunk)))))
        {:hex (digest->hex (.digest md))
         :bytes-cnt (* 8 n)
         :kind :tensor/f64
         :shape [n]})

      (instance? short-array-class val)
      (let [arr ^shorts val
            n (alength arr)]
        (loop [off 0]
          (when (< off n)
            (let [chunk (min 8192 (- n off))]
              (.clear sb)
              (.put sb arr off chunk)
              (.update md ba 0 (* 2 chunk))
              (recur (+ off chunk)))))
        {:hex (digest->hex (.digest md))
         :bytes-cnt (* 2 n)
         :kind :tensor/i16
         :shape [n]})

      (bytes? val)
      (let [arr ^bytes val
            n (alength arr)]
        {:hex (sha256-bytes arr)
         :bytes-cnt n
         :kind :bytes/raw
         :shape [n]})

      :else nil)))

(defn primitive-array-cid
  "Computes the deterministic 'sha256:<hex>' content identifier for a primitive array."
  [val]
  (when-let [{:keys [hex]} (digest-primitive-array val)]
    (str "sha256:" hex)))

;; -----------------------------------------------------------------------------
;; 3. Store Construction & Operations
;; -----------------------------------------------------------------------------

(defn create-cas-store
  "Creates a new in-memory Content-Addressed Pointer Store wrapping an atom."
  []
  {:entries (atom {})})

(defn- store-atom
  [store]
  (if (instance? clojure.lang.IAtom store)
    store
    (or (:entries store)
        (throw (ex-info "Invalid CAS store: missing atom or :entries" {:store store})))))

(defn put!
  "Stores value val in the CAS store, returning an immutable CasPtr map.
   Options can specify:
     :cid   - custom CID (defaults to \"sha256:\" + SHA-256 of raw bytes)
     :kind  - keyword (defaults to :text/utf8 if string?, :tensor/* if primitive array, or :edn/value)
     :uri   - string URI (defaults to \"mem://cas/\" + hex hash)
     :bytes - integer byte size
     :shape - optional shape vector (e.g. for tensor representations)"
  ([store val]
   (put! store val {}))
  ([store val opts]
   (let [prim-info (digest-primitive-array val)
         raw-bytes (when-not prim-info
                     (if (string? val)
                       (.getBytes ^String val "UTF-8")
                       (.getBytes (pr-str val) "UTF-8")))
         cid (or (:cid opts)
                 (if prim-info
                   (str "sha256:" (:hex prim-info))
                   (str "sha256:" (sha256-bytes raw-bytes))))
         hash-part (if (.startsWith cid "sha256:") (subs cid 7) cid)
         kind (or (:kind opts)
                  (:kind prim-info)
                  (if (string? val) :text/utf8 :edn/value))
         uri (or (:uri opts) (str "mem://cas/" hash-part))
         bytes-cnt (or (:bytes opts)
                       (:bytes-cnt prim-info)
                       (alength ^bytes raw-bytes))
         shape (or (:shape opts) (:shape prim-info))
         argmax (or (:argmax opts) (:argmax prim-info))
         ptr (cond-> {:cas/cid cid
                      :cas/kind kind
                      :cas/uri uri
                      :cas/bytes bytes-cnt}
               shape (assoc :cas/shape shape)
               (some? argmax) (assoc :cas/argmax (int argmax)))
         atm (store-atom store)]
     (swap! atm assoc cid {:ptr ptr :value val})
     ptr)))

(defn resolve-ptr
  "Resolves a pointer x in the CAS store.
   If x is a cas-ptr?, returns the stored :value (or nil if not found).
   If x is not a cas-ptr?, returns x unchanged."
  [store x]
  (if (and store (cas-ptr? x))
    (let [atm (store-atom store)
          entry (get @atm (:cas/cid x))]
      (:value entry))
    x))

(def ^:private primitive-cas-kinds
  #{:tensor/f32 :tensor/i32 :tensor/i64 :tensor/f64 :tensor/i16 :bytes/raw})

(defn externalize-primitives!
  "Recursively traverses `payload` and stores any Java primitive arrays ([F, [I, [J, [D, [S, [B)
   in `store` via `put!`, replacing them with pure-EDN `CasPtr` maps."
  [store payload]
  (if (nil? store)
    payload
    (cond
      (primitive-array? payload)
      (put! store payload)

      (vector? payload)
      (mapv #(externalize-primitives! store %) payload)

      (and (map? payload) (not (cas-ptr? payload)))
      (persistent!
       (reduce-kv (fn [acc k v]
                    (assoc! acc k (externalize-primitives! store v)))
                  (transient {})
                  payload))

      :else
      payload)))

(defn resolve-payload
  "Recursively traverses `payload` and resolves any primitive-array `CasPtr` (`:tensor/*` or `:bytes/raw`)
   whose `:cas/cid` is present in `store`. If `store` does not contain the entry (e.g. during
   unhydrated cross-process replay), leaves the `CasPtr` intact."
  [store payload]
  (if (nil? store)
    payload
    (cond
      (and (cas-ptr? payload) (contains? primitive-cas-kinds (:cas/kind payload)))
      (let [v (resolve-ptr store payload)]
        (if (some? v) v payload))

      (vector? payload)
      (mapv #(resolve-payload store %) payload)

      (and (map? payload) (not (cas-ptr? payload)))
      (persistent!
       (reduce-kv (fn [acc k v]
                    (assoc! acc k (resolve-payload store v)))
                  (transient {})
                  payload))

      :else
      payload)))

(defn maybe-externalize!
  "If payload is a string exceeding max-inline-chars in length, stores it in the CAS
   store and returns the resulting CasPtr; otherwise returns payload unchanged."
  [store payload max-inline-chars]
  (if (and (string? payload) (> (count payload) (long max-inline-chars)))
    (put! store payload {:kind :text/utf8})
    payload))
