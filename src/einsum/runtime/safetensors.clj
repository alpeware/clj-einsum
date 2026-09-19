(ns einsum.runtime.safetensors
  "Panama FFM zero-copy off-heap .safetensors weight loader."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.io RandomAccessFile]
           [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteBuffer]
           [java.nio.channels FileChannel FileChannel$MapMode]
           [java.nio.file Path StandardOpenOption]))

(defn parse-header-json
  "Parses safetensors UTF-8 JSON header metadata string."
  [json-str]
  (json/read-str json-str))

(defn read-header
  "Reads the safetensors 8-byte unsigned header size and JSON metadata map from `file-path`."
  [file-path]
  (with-open [raf (RandomAccessFile. ^String file-path "r")]
    (let [header-size (.readLong raf)
          header-size-le (Long/reverseBytes header-size)
          header-bytes (byte-array header-size-le)]
      (.readFully raf header-bytes)
      (let [json-str (String. header-bytes "UTF-8")]
        {:header-size header-size-le
         :metadata (parse-header-json json-str)}))))

(declare map-safetensors-weights)

(defn map-safetensors-directory
  "Memory-maps all `.safetensors` files in `dir-path` using `arena`."
  [dir-path arena]
  (let [dir (io/file dir-path)
        safetensor-files (filter #(and (.isFile %) (.endsWith (.getName %) ".safetensors"))
                                 (.listFiles dir))
        mapped-shards (mapv (fn [f]
                              (let [path (.getAbsolutePath f)
                                    {:keys [header segment]} (map-safetensors-weights path arena)]
                                {:header header :segment segment :file-path path}))
                            safetensor-files)
        tensor-map (into {}
                         (for [shard mapped-shards
                               [t-name t-info] (:header shard)]
                           [t-name {:info t-info :segment (:segment shard)}]))
        header-map (into {} (map (fn [[k v]] [k (:info v)]) tensor-map))]
    {:header header-map
     :tensors tensor-map
     :shards mapped-shards}))

(defn map-safetensors-weights
  "Maps `file-path` (or directory) off-heap into Java 25 Panama MemorySegments using `arena`.
   Returns {:header header-metadata :segment memory-segment} or {:header ... :tensors ...}."
  [^String file-path ^Arena arena]
  (let [f (io/file file-path)]
    (cond
      (.isDirectory f)
      (map-safetensors-directory file-path arena)

      (and (not (.exists f))
           (some? (.getParent f))
           (.exists (io/file (.getParent f) "model-00001-of-00002.safetensors")))
      (map-safetensors-directory (.getParent f) arena)

      :else
      (let [{:keys [header-size metadata]} (read-header file-path)
            path (Path/of file-path (into-array String []))
            fc (FileChannel/open path (into-array [StandardOpenOption/READ]))
            total-len (.size fc)
            weight-bytes-len (- total-len (+ 8 header-size))
            ^MemorySegment mapped-seg (.map fc FileChannel$MapMode/READ_ONLY (+ 8 (long header-size)) (long weight-bytes-len) arena)]
        {:header metadata
         :segment mapped-seg}))))

(defn get-tensor-slice
  "Extracts off-heap memory segment slice for tensor `tensor-name` from mapped weights."
  [mapped-weights tensor-name]
  (if-let [t-entry (get (:tensors mapped-weights) tensor-name)]
    (let [{:keys [info segment]} t-entry
          [start end] (get info "data_offsets")
          len (- end start)]
      (.asSlice ^MemorySegment segment (long start) (long len)))
    (if-let [tensor-info (get (:header mapped-weights) tensor-name)]
      (let [segment (:segment mapped-weights)
            [start end] (get tensor-info "data_offsets")
            len (- end start)]
        (.asSlice ^MemorySegment segment (long start) (long len)))
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name})))))

(def ^:private UNALIGNED-SHORT (.. ValueLayout/JAVA_SHORT (withByteAlignment 1)))
(def ^:private UNALIGNED-FLOAT (.. ValueLayout/JAVA_FLOAT (withByteAlignment 1)))

(defn get-tensor-floats
  "Reads Float32 or BF16 tensor values from `mapped-weights` into a Java float array using Little-Endian byte order."
  [mapped-weights tensor-name]
  (let [t-entry (or (get (:tensors mapped-weights) tensor-name)
                    (when-let [info (get (:header mapped-weights) tensor-name)]
                      {:info info :segment (:segment mapped-weights)}))]
    (if-not t-entry
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))
      (let [{:keys [info segment]} t-entry
            [start end] (get info "data_offsets")
            dtype (get info "dtype" "F32")
            len (- end start)
            ^MemorySegment slice (.asSlice ^MemorySegment segment (long start) (long len))]
        (cond
          (or (= dtype "BF16") (= dtype "BFLOAT16"))
          (let [num-elements (quot len 2)
                chunk-size 50000000
                target-len (min num-elements Integer/MAX_VALUE)
                arr (float-array target-len)]
            (loop [offset 0]
              (when (< offset target-len)
                (let [cur-len (min chunk-size (- target-len offset))
                      sa (short-array cur-len)
                      byte-offset (* (long offset) 2)
                      seg-slice (.asSlice slice byte-offset (* (long cur-len) 2))]
                  (MemorySegment/copy seg-slice UNALIGNED-SHORT (long 0) sa (long 0) (long cur-len))
                  (dotimes [i cur-len]
                    (let [s (int (aget sa i))
                          bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
                      (aset arr (+ offset i) (Float/intBitsToFloat bits))))
                  (recur (+ offset cur-len)))))
            arr)

          :else
          (let [num-floats (quot len 4)
                arr (float-array (min num-floats Integer/MAX_VALUE))]
            (MemorySegment/copy slice UNALIGNED-FLOAT (long 0) arr (long 0) (long (count arr)))
            arr))))))

(defn get-tensor-bf16-shorts
  "Reads BF16 or Float32 tensor values from `mapped-weights` into a Java short array (BF16 raw 16-bit values)."
  [mapped-weights tensor-name]
  (let [t-entry (or (get (:tensors mapped-weights) tensor-name)
                    (when-let [info (get (:header mapped-weights) tensor-name)]
                      {:info info :segment (:segment mapped-weights)}))]
    (if-not t-entry
      (throw (ex-info "Tensor key not found in Safetensors header" {:tensor tensor-name}))
      (let [{:keys [info segment]} t-entry
            [start end] (get info "data_offsets")
            dtype (get info "dtype" "F32")
            len (- end start)
            ^MemorySegment slice (.asSlice ^MemorySegment segment (long start) (long len))]
        (cond
          (or (= dtype "BF16") (= dtype "BFLOAT16"))
          (let [num-shorts (quot len 2)
                arr (short-array num-shorts)]
            (MemorySegment/copy slice UNALIGNED-SHORT (long 0) arr (long 0) (long num-shorts))
            arr)

          :else
          (let [num-floats (quot len 4)
                fa (float-array num-floats)
                _ (MemorySegment/copy slice UNALIGNED-FLOAT (long 0) fa (long 0) (long num-floats))
                arr (short-array num-floats)]
            (dotimes [i num-floats]
              (let [f (aget fa i)
                    bits (Float/floatToIntBits f)
                    s (short (bit-shift-right bits 16))]
                (aset arr i s)))
            arr))))))

(defn- tensor-byte-len
  "Computes payload byte length for given data array, segment, or shape/dtype."
  [data shape dtype]
  (cond
    (instance? (Class/forName "[B") data) (long (alength ^bytes data))
    (instance? (Class/forName "[S") data) (* 2 (long (alength ^shorts data)))
    (instance? (Class/forName "[I") data) (* 4 (long (alength ^ints data)))
    (instance? (Class/forName "[F") data) (* 4 (long (alength ^floats data)))
    (instance? (Class/forName "[D") data) (* 8 (long (alength ^doubles data)))
    (instance? (Class/forName "[J") data) (* 8 (long (alength ^longs data)))
    (instance? MemorySegment data) (.byteSize ^MemorySegment data)
    (and shape dtype)
    (let [n (long (reduce * 1 shape))]
      (case (str/upper-case (name dtype))
        ("BOOL" "U8" "I8") n
        ("F16" "BF16" "BFLOAT16" "I16" "U16") (* n 2)
        ("F32" "I32" "U32") (* n 4)
        ("F64" "I64" "U64") (* n 8)
        n))
    :else 0))

(defn build-header-map
  "Pure function: calculates safetensors JSON header metadata and continuous data offsets
   for a sequence of tensor specifications: `[{:name string :dtype string :shape [...] :data array :byte-len num}]`.
   Returns `{:header-map ... :total-payload-bytes ...}`."
  [tensors metadata]
  (let [entries (loop [tensors tensors
                       curr-offset 0
                       res []]
                  (if (empty? tensors)
                    {:res res :total curr-offset}
                    (let [t (first tensors)
                          t-name (:name t)
                          dtype (str/upper-case (name (:dtype t)))
                          shape (vec (:shape t))
                          len (or (:byte-len t)
                                  (tensor-byte-len (:data t) shape dtype))
                          next-offset (+ curr-offset len)]
                      (recur (rest tensors)
                             next-offset
                             (conj res [t-name {"dtype" dtype
                                                "shape" shape
                                                "data_offsets" [curr-offset next-offset]}])))))
        header-map (into (cond-> {} (seq metadata) (assoc "__metadata__" metadata))
                         (:res entries))]
    {:header-map header-map
     :total-payload-bytes (:total entries)}))

(defn write-segment-to-channel!
  "Writes MemorySegment `seg` to `fc` in chunks of up to 1 GB to avoid Java ByteBuffer 2 GB limit."
  [^FileChannel fc ^MemorySegment seg]
  (let [total-size (.byteSize seg)
        chunk-size (* 1024 1024 1024)]
    (loop [offset (long 0)]
      (when (< offset total-size)
        (let [curr-len (min (long chunk-size) (- total-size offset))
              slice (.asSlice seg offset curr-len)]
          (.write fc (.asByteBuffer slice))
          (recur (+ offset curr-len)))))))

(defn write-safetensors
  "Writes a safetensors binary file to `file-path` with zero-copy/streaming tensor payloads.
   `tensors` is a sequence of maps: `{:name string :dtype string :shape [dims] :data array-or-segment}`.
   `opts` may contain `:metadata` (a map of key-value metadata strings)."
  ([file-path tensors]
   (write-safetensors file-path tensors {}))
  ([file-path tensors opts]
   (let [{:keys [header-map]} (build-header-map tensors (:metadata opts))
         json-str (json/write-str header-map :escape-slash false)
         header-bytes (.getBytes json-str "UTF-8")
         header-size (count header-bytes)
         path (Path/of file-path (into-array String []))
         _ (when-let [p (.getParentFile (io/file file-path))] (.mkdirs p))]
     (with-open [fc (FileChannel/open path (into-array [StandardOpenOption/CREATE
                                                        StandardOpenOption/WRITE
                                                        StandardOpenOption/TRUNCATE_EXISTING]))]
       ;; 1. Write 8-byte LE header size
       (let [header-size-le (Long/reverseBytes (long header-size))
             size-buf (ByteBuffer/allocate 8)]
         (.putLong size-buf header-size-le)
         (.flip size-buf)
         (.write fc size-buf))
       ;; 2. Write UTF-8 JSON header metadata
       (.write fc (ByteBuffer/wrap header-bytes))
       ;; 3. Write tensor payloads
       (with-open [arena (Arena/ofConfined)]
         (doseq [{:keys [data]} tensors]
           (cond
             (instance? MemorySegment data)
             (write-segment-to-channel! fc ^MemorySegment data)

             (instance? (Class/forName "[B") data)
             (.write fc (ByteBuffer/wrap ^bytes data))

             (instance? (Class/forName "[S") data)
             (let [^shorts sa data
                   n (alength sa)
                   seg (.allocate arena (* (long n) 2) (long 1))]
               (MemorySegment/copy sa 0 seg ValueLayout/JAVA_SHORT (long 0) n)
               (.write fc (.asByteBuffer seg)))

             (instance? (Class/forName "[F") data)
             (let [^floats fa data
                   n (alength fa)
                   seg (.allocate arena (* (long n) 4) (long 1))]
               (MemorySegment/copy fa 0 seg ValueLayout/JAVA_FLOAT (long 0) n)
               (.write fc (.asByteBuffer seg)))

             (instance? (Class/forName "[I") data)
             (let [^ints ia data
                   n (alength ia)
                   seg (.allocate arena (* (long n) 4) (long 1))]
               (MemorySegment/copy ia 0 seg ValueLayout/JAVA_INT (long 0) n)
               (.write fc (.asByteBuffer seg)))

             (instance? (Class/forName "[D") data)
             (let [^doubles da data
                   n (alength da)
                   seg (.allocate arena (* (long n) 8) (long 1))]
               (MemorySegment/copy da 0 seg ValueLayout/JAVA_DOUBLE (long 0) n)
               (.write fc (.asByteBuffer seg)))

             :else
             (throw (ex-info "Unsupported tensor data type for safetensors serialization"
                             {:data-type (type data)})))))))))
