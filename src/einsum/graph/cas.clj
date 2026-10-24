(ns einsum.graph.cas
  "Content-Addressed Storage (CAS) Pointer Store for off-heap and immutable data structures."
  (:require [einsum.graph.schema :as schema])
  (:import [java.security MessageDigest]))

;; -----------------------------------------------------------------------------
;; 1. Cryptographic Digest Helpers
;; -----------------------------------------------------------------------------

(defn sha256-bytes
  "Computes a 64-character lowercase hexadecimal SHA-256 digest of byte array b."
  [^bytes b]
  (let [md (MessageDigest/getInstance "SHA-256")
        digest (.digest md b)
        sb (StringBuilder. 64)]
    (doseq [x digest]
      (.append sb (format "%02x" (bit-and x 0xff))))
    (.toString sb)))

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

(defn- serialize-primitive-array
  "Serializes a primitive array into a big-endian byte array and returns
   {:bytes <byte[]> :kind <kw> :shape [<len>] :argmax <int-or-nil>}."
  [val]
  (cond
    (instance? float-array-class val)
    (let [arr ^floats val
          n (alength arr)
          bb (java.nio.ByteBuffer/allocate (* 4 n))]
      (.put (.asFloatBuffer bb) arr)
      {:bytes (.array bb)
       :kind :tensor/f32
       :shape [n]
       :argmax (when (pos? n) (int (argmax-float-array arr)))})

    (instance? int-array-class val)
    (let [arr ^ints val
          n (alength arr)
          bb (java.nio.ByteBuffer/allocate (* 4 n))]
      (.put (.asIntBuffer bb) arr)
      {:bytes (.array bb)
       :kind :tensor/i32
       :shape [n]
       :argmax (when (pos? n) (int (argmax-int-array arr)))})

    (instance? long-array-class val)
    (let [arr ^longs val
          n (alength arr)
          bb (java.nio.ByteBuffer/allocate (* 8 n))]
      (.put (.asLongBuffer bb) arr)
      {:bytes (.array bb)
       :kind :tensor/i64
       :shape [n]})

    (instance? double-array-class val)
    (let [arr ^doubles val
          n (alength arr)
          bb (java.nio.ByteBuffer/allocate (* 8 n))]
      (.put (.asDoubleBuffer bb) arr)
      {:bytes (.array bb)
       :kind :tensor/f64
       :shape [n]})

    (instance? short-array-class val)
    (let [arr ^shorts val
          n (alength arr)
          bb (java.nio.ByteBuffer/allocate (* 2 n))]
      (.put (.asShortBuffer bb) arr)
      {:bytes (.array bb)
       :kind :tensor/i16
       :shape [n]})

    (bytes? val)
    (let [arr ^bytes val]
      {:bytes arr
       :kind :bytes/raw
       :shape [(alength arr)]})

    :else nil))

(defn primitive-array-cid
  "Computes the deterministic 'sha256:<hex>' content identifier for a primitive array."
  [val]
  (when-let [{:keys [^bytes bytes]} (serialize-primitive-array val)]
    (str "sha256:" (sha256-bytes bytes))))

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
   (let [prim-info (serialize-primitive-array val)
         raw-bytes (cond
                     prim-info (:bytes prim-info)
                     (string? val) (.getBytes ^String val "UTF-8")
                     :else (.getBytes (pr-str val) "UTF-8"))
         cid (or (:cid opts) (str "sha256:" (sha256-bytes raw-bytes)))
         hash-part (if (.startsWith cid "sha256:") (subs cid 7) cid)
         kind (or (:kind opts)
                  (:kind prim-info)
                  (if (string? val) :text/utf8 :edn/value))
         uri (or (:uri opts) (str "mem://cas/" hash-part))
         bytes-cnt (or (:bytes opts) (alength ^bytes raw-bytes))
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
