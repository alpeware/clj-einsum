(ns einsum.graph.cas
  "Content-Addressed Storage (CAS) Pointer Store for off-heap and immutable data structures."
  (:require [einsum.graph.schema :as schema])
  (:import [java.security MessageDigest]))

;; -----------------------------------------------------------------------------
;; 1. Cryptographic Digest Helpers
;; -----------------------------------------------------------------------------

(defn sha256-str
  "Computes a 64-character lowercase hexadecimal SHA-256 digest of string s (UTF-8)."
  [^String s]
  (let [md (MessageDigest/getInstance "SHA-256")
        bytes (.digest md (.getBytes s "UTF-8"))
        sb (StringBuilder. 64)]
    (doseq [b bytes]
      (.append sb (format "%02x" (bit-and b 0xff))))
    (.toString sb)))

;; -----------------------------------------------------------------------------
;; 2. CAS Pointer Predicate
;; -----------------------------------------------------------------------------

(defn cas-ptr?
  "Returns true if x conforms to a CAS pointer map."
  [x]
  (schema/cas-ptr? x))

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
     :cid   - custom CID (defaults to \"sha256:\" + SHA-256 of (pr-str val))
     :kind  - keyword (defaults to :text/utf8 if string? or :edn/value)
     :uri   - string URI (defaults to \"mem://cas/\" + hex hash)
     :bytes - integer byte size
     :shape - optional shape vector (e.g. for tensor representations)"
  ([store val]
   (put! store val {}))
  ([store val opts]
   (let [val-str (if (string? val) val (pr-str val))
         cid (or (:cid opts) (str "sha256:" (sha256-str (pr-str val))))
         hash-part (if (.startsWith cid "sha256:") (subs cid 7) cid)
         kind (or (:kind opts) (if (string? val) :text/utf8 :edn/value))
         uri (or (:uri opts) (str "mem://cas/" hash-part))
         bytes-cnt (or (:bytes opts) (count (.getBytes val-str "UTF-8")))
         shape (:shape opts)
         ptr (cond-> {:cas/cid cid
                      :cas/kind kind
                      :cas/uri uri
                      :cas/bytes bytes-cnt}
               shape (assoc :cas/shape shape))
         atm (store-atom store)]
     (swap! atm assoc cid {:ptr ptr :value val})
     ptr)))

(defn resolve-ptr
  "Resolves a pointer x in the CAS store.
   If x is a cas-ptr?, returns the stored :value (or nil if not found).
   If x is not a cas-ptr?, returns x unchanged."
  [store x]
  (if (cas-ptr? x)
    (let [atm (store-atom store)
          entry (get @atm (:cas/cid x))]
      (:value entry))
    x))

(defn maybe-externalize!
  "If payload is a string exceeding max-inline-chars in length, stores it in the CAS
   store and returns the resulting CasPtr; otherwise returns payload unchanged."
  [store payload max-inline-chars]
  (if (and (string? payload) (> (count payload) (long max-inline-chars)))
    (put! store payload {:kind :text/utf8})
    payload))
