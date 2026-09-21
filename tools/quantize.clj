(ns tools.quantize
  "Offline hardware-aware model quantization pipeline for clj-xla.
   Automatically detects GPU VRAM and architecture, selects optimal quantization,
   and streams quantized weights into native Safetensors format."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.hardware :as hw]
            [einsum.logic.exl3 :as exl3]
            [einsum.quant.ternary :as ternary]
            [einsum.runtime.safetensors :as st])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteBuffer]
           [java.nio.channels FileChannel]
           [java.nio.file Path StandardCopyOption StandardOpenOption]))

;; --- Pure Quantization Functions ---

(defn quantizable-weight?
  "Pure predicate: returns true if tensor is an attention or MLP projection weight eligible for INT4/INT8 quantization."
  [tensor-name shape]
  (boolean
   (and (vector? shape)
        (= (count shape) 2)
        (str/ends-with? tensor-name ".weight")
        (or (re-find #"\.(?:q|k|v|o|gate|up|down)_proj\.weight$" tensor-name)
            (and (or (str/includes? tensor-name "self_attn")
                     (str/includes? tensor-name "mlp"))
                 (not (str/includes? tensor-name "norm"))
                 (not (str/includes? tensor-name "scale"))))
        (not (str/includes? tensor-name "embed"))
        (not (str/includes? tensor-name "norm"))
        (not (str/includes? tensor-name "scale"))
        (not (str/ends-with? tensor-name "bias"))
        (even? (second shape)))))

(defn quantize-projection-weight
  "Pure function: quantizes 2D projection weight array `w-arr` of shape `[rows cols]` to `precision` (:int4 or :int8).
   Opts may contain `:group-size` (default 128 for :int4).
   Returns `{:data byte-array :scales short-array :shape [...] :scale-shape [...] :dtype ...}`."
  ([w-arr rows cols precision]
   (quantize-projection-weight w-arr rows cols precision {}))
  ([w-arr rows cols precision opts]
   (case precision
     :int4
     (let [group-size (get opts :group-size 128)
           {:keys [data scales scale-shape]} (exl3/quantize-weights-per-row-int4 w-arr rows cols {:as :bf16 :group-size group-size})]
       {:data data
        :scales scales
        :shape [rows (quot cols 2)]
        :dtype "I8"
        :scale-shape scale-shape
        :scale-dtype "BF16"})

     :int8
     (let [{:keys [data scales]} (exl3/quantize-weights-per-row-int8 w-arr rows cols {:as :bf16})]
       {:data data
        :scales scales
        :shape [rows cols]
        :dtype "I8"
        :scale-shape [rows]
        :scale-dtype "BF16"})

     :ternary
     (let [group-size (get opts :group-size nil)
           {:keys [data scales scale-shape]} (ternary/quantize-weights-per-row-ternary w-arr rows cols {:as :bf16 :group-size group-size})]
       {:data data
        :scales scales
        :shape [rows (quot cols 4)]
        :dtype "I8"
        :scale-shape (or scale-shape [rows])
        :scale-dtype "BF16"})

     (throw (ex-info "Unsupported quantization precision" {:precision precision})))))

(defn parse-model-param-count
  "Extracts or estimates total parameter count from model `config.json` map."
  [config-json]
  (let [text-cfg (get config-json "text_config" config-json)
        hidden (get text-cfg "hidden_size" 2048)
        _inter (get text-cfg "intermediate_size" (* 4 hidden))
        layers (get text-cfg "num_hidden_layers" 24)
        _vocab (get text-cfg "vocab_size" 262144)]
    (cond
      ;; If 31B
      (or (>= (long hidden) 5000) (>= (long layers) 50)) 31000000000
      ;; If 12B
      (or (>= (long hidden) 3800) (>= (long layers) 40)) 12000000000
      ;; If 4B
      (or (>= (long hidden) 2500) (>= (long layers) 30)) 4000000000
      ;; If 2B
      :else 2000000000)))

;; --- Streaming Quantization Engine (Boundary Shell) ---

(defn- copy-model-file-if-exists [src-dir dst-dir file-name]
  (let [src-f (io/file src-dir file-name)
        dst-f (io/file dst-dir file-name)]
    (when (.exists src-f)
      (java.nio.file.Files/copy (.toPath src-f)
                                (.toPath dst-f)
                                (into-array [StandardCopyOption/REPLACE_EXISTING])))))

(defn quantize-model!
  "Quantizes unquantized safetensors checkpoint at `model-path` into native pre-quantized format at `output-path`.
   Automatically detects hardware capabilities if `precision` is `:auto` (or not provided)."
  [{:keys [model-path output-path precision quiet group-size]}]
  (let [model-dir (io/file model-path)
        _ (when-not (.exists model-dir)
            (throw (ex-info "Model directory does not exist" {:model-path model-path})))

        ;; 1. Detect hardware & select strategy
        hw-profile (hw/detect-hardware)
        primary-dev (:primary-device hw-profile)
        config-file (io/file model-dir "config.json")
        model-cfg (when (.exists config-file)
                    (json/read-str (slurp config-file)))
        param-count (when model-cfg (parse-model-param-count model-cfg))
        selected-precision (if (and precision (not= precision :auto))
                             precision
                             (hw/select-quant-strategy hw-profile param-count))
        group-size (long (or group-size 128))
        resolved-out-dir (or (when output-path (io/file output-path))
                             (io/file (.getParentFile model-dir) (str (.getName model-dir) "-" (name selected-precision))))
        _ (.mkdirs resolved-out-dir)]

    (when-not quiet
      (println "=== clj-xla Hardware-Aware Offline Model Quantizer ===")
      (println (format "Detected Primary Device : %s (%s, %d MB VRAM)"
                       (or (:name primary-dev) "Unknown")
                       (name (or (:arch primary-dev) :generic))
                       (or (:vram-total-mb primary-dev) 0)))
      (println (format "Target Model Directory  : %s (estimated %s params)"
                       (.getAbsolutePath model-dir)
                       (if param-count (format "%.1fB" (/ (double param-count) 1e9)) "unknown")))
      (println (format "Selected Strategy       : %s (group-size: %s) -> %s"
                       (str/upper-case (name selected-precision))
                       (if (= selected-precision :int4) (str group-size) "per-row")
                       (.getAbsolutePath resolved-out-dir))))

    (with-open [arena (Arena/ofConfined)]
      ;; 2. Memory-map source weights
      (let [mapped-weights (st/map-safetensors-weights (.getAbsolutePath model-dir) arena)
            raw-header (or (:header mapped-weights) {})
            ;; Filter out any existing __metadata__ from header
            tensor-header (dissoc raw-header "__metadata__")
            ;; Deterministically sort tensor names
            tensor-names (vec (sort (keys tensor-header)))]

        (when-not quiet
          (println (format "Found %d tensors in source model. Computing quantized header layout..." (count tensor-names))))

        ;; 3. Build target tensor specs and offsets
        (let [tensor-specs (loop [names tensor-names
                                  curr-offset 0
                                  specs []]
                             (if (empty? names)
                               {:specs specs :total-bytes curr-offset}
                               (let [t-name (first names)
                                     info (get tensor-header t-name)
                                     orig-shape (get info "shape")
                                     orig-dtype (get info "dtype")
                                     [start end] (get info "data_offsets")
                                     orig-len (- end start)]
                                 (if (quantizable-weight? t-name orig-shape)
                                   (let [[rows cols] orig-shape
                                         half-cols (quot cols 2)
                                         w-len (if (= selected-precision :int4)
                                                 (* rows half-cols)
                                                 (* rows cols))
                                         w-offset-end (+ curr-offset w-len)
                                         scale-name (str t-name ".scales")
                                         num-groups (if (and (= selected-precision :int4) group-size (zero? (mod cols group-size)))
                                                      (quot cols group-size)
                                                      1)
                                         scale-shape (if (> num-groups 1) [rows num-groups] [rows])
                                         num-scales (if (> num-groups 1) (* rows num-groups) rows)
                                         scale-len (* (long num-scales) 2) ;; BF16 = 2 bytes per scale
                                         scale-offset-end (+ w-offset-end scale-len)
                                         w-spec {:name t-name
                                                 :source-name t-name
                                                 :quantize? true
                                                 :shape (if (= selected-precision :int4) [rows half-cols] [rows cols])
                                                 :dtype "I8"
                                                 :scale-name scale-name
                                                 :scale-shape scale-shape
                                                 :scale-dtype "BF16"
                                                 :data-offsets [curr-offset w-offset-end]
                                                 :scale-offsets [w-offset-end scale-offset-end]}]
                                     (recur (rest names)
                                            scale-offset-end
                                            (conj specs w-spec)))
                                   ;; Non-quantizable tensor: preserve exact bytes
                                   (let [next-offset (+ curr-offset orig-len)
                                         spec {:name t-name
                                               :source-name t-name
                                               :quantize? false
                                               :shape orig-shape
                                               :dtype orig-dtype
                                               :data-offsets [curr-offset next-offset]}]
                                     (recur (rest names)
                                            next-offset
                                            (conj specs spec)))))))

              {:keys [specs total-bytes]} tensor-specs
              ;; Build header map
              header-tensors (into {}
                                   (mapcat (fn [spec]
                                             (if (:quantize? spec)
                                               [[(:name spec) {"dtype" (:dtype spec)
                                                               "shape" (:shape spec)
                                                               "data_offsets" (:data-offsets spec)}]
                                                [(:scale-name spec) {"dtype" (:scale-dtype spec)
                                                                     "shape" (:scale-shape spec)
                                                                     "data_offsets" (:scale-offsets spec)}]]
                                               [[(:name spec) {"dtype" (:dtype spec)
                                                               "shape" (:shape spec)
                                                               "data_offsets" (:data-offsets spec)}]]))
                                           specs))
              metadata (cond-> {"quantization" (name selected-precision)
                                "format" "clj-xla"
                                "producer" "einsum.hardware"
                                "target_arch" (name (get primary-dev :arch :generic))}
                         (= selected-precision :int4) (assoc "group_size" (str group-size)))
              full-header-map (assoc header-tensors "__metadata__" metadata)
              json-str (json/write-str full-header-map :escape-slash false)
              header-bytes (.getBytes json-str "UTF-8")
              header-size (count header-bytes)
              out-file (io/file resolved-out-dir "model.safetensors")
              out-path (Path/of (.getAbsolutePath out-file) (into-array String []))]

          (when-not quiet
            (println (format "Quantized payload size: %.2f GB (header: %.2f KB)"
                             (/ (double total-bytes) 1024.0 1024.0 1024.0)
                             (/ (double header-size) 1024.0)))
            (println "Streaming weights into native Safetensors..."))

          ;; 4. Stream write into output file
          (with-open [fc (FileChannel/open out-path (into-array [StandardOpenOption/CREATE
                                                                 StandardOpenOption/WRITE
                                                                 StandardOpenOption/TRUNCATE_EXISTING]))]
            ;; 4a. Write 8-byte LE header size
            (let [header-size-le (Long/reverseBytes (long header-size))
                  size-buf (ByteBuffer/allocate 8)]
              (.putLong size-buf header-size-le)
              (.flip size-buf)
              (.write fc size-buf))
            ;; 4b. Write JSON header bytes
            (.write fc (ByteBuffer/wrap header-bytes))

            ;; 4c. Process and stream tensors sequentially
            (let [total-specs (count specs)
                  progress-interval (max 1 (quot total-specs 10))]
              (doseq [[idx spec] (map-indexed vector specs)]
                (if (:quantize? spec)
                  (let [src-name (:source-name spec)
                        [rows cols] (get-in raw-header [src-name "shape"])
                        raw-shorts (st/get-tensor-bf16-shorts mapped-weights src-name)
                        {:keys [data scales]} (quantize-projection-weight raw-shorts rows cols selected-precision {:group-size group-size})
                        ^bytes data-bytes data
                        ^shorts scale-shorts scales
                        num-scales (alength scale-shorts)]
                    ;; Write INT4/INT8 packed bytes
                    (.write fc (ByteBuffer/wrap data-bytes))
                    ;; Write BF16 scales
                    (let [scale-seg (.allocate arena (* (long num-scales) 2) (long 1))]
                      (MemorySegment/copy scale-shorts 0 scale-seg ValueLayout/JAVA_SHORT (long 0) num-scales)
                      (.write fc (.asByteBuffer scale-seg))))
                  ;; Non-quantizable tensor: zero-copy stream direct from mapped source
                  (let [slice (st/get-tensor-slice mapped-weights (:source-name spec))]
                    (st/write-segment-to-channel! fc ^MemorySegment slice)))

                (when (and (not quiet) (zero? (mod (inc idx) progress-interval)))
                  (println (format "  [%3d%%] Processed %d / %d tensors..."
                                   (int (* 100 (/ (double (inc idx)) (double total-specs))))
                                   (inc idx)
                                   total-specs))))))

          (when-not quiet
            (println "Weight streaming complete! Copying tokenizer and configuration files..."))

          ;; 5. Copy configuration and tokenizer files
          (doseq [f ["config.json" "generation_config.json" "processor_config.json" "tokenizer.json" "tokenizer_config.json"]]
            (copy-model-file-if-exists model-dir resolved-out-dir f))

          ;; 6. Write quant_config.edn
          (spit (io/file resolved-out-dir "quant_config.edn")
                (pr-str {:quantization selected-precision
                         :group-size (when (= selected-precision :int4) group-size)
                         :source-model (.getAbsolutePath model-dir)
                         :hardware-profile hw-profile
                         :timestamp (str (java.time.Instant/now))
                         :total-tensors (count specs)}))

          (when-not quiet
            (println (format "Successfully generated pre-quantized %s model at [%s]"
                             (str/upper-case (name selected-precision))
                             (.getAbsolutePath resolved-out-dir)))
            (println "=================================================================="))

          {:status :ok
           :precision selected-precision
           :output-path (.getAbsolutePath resolved-out-dir)
           :total-tensors (count specs)})))))

;; --- CLI Entrypoint ---

(defn parse-cli-args [args]
  (loop [remaining args
         opts {:precision :auto}]
    (if (empty? remaining)
      opts
      (let [arg (first remaining)]
        (cond
          (or (= arg "--model") (= arg "-m"))
          (recur (drop 2 remaining) (assoc opts :model-path (second remaining)))

          (or (= arg "--output") (= arg "-o"))
          (recur (drop 2 remaining) (assoc opts :output-path (second remaining)))

          (or (= arg "--precision") (= arg "-p"))
          (recur (drop 2 remaining) (assoc opts :precision (keyword (second remaining))))

          (or (= arg "--group-size") (= arg "-g"))
          (recur (drop 2 remaining) (assoc opts :group-size (parse-long (second remaining))))

          (or (= arg "--quiet") (= arg "-q"))
          (recur (rest remaining) (assoc opts :quiet true))

          (not (:model-path opts))
          (recur (rest remaining) (assoc opts :model-path arg))

          :else
          (recur (rest remaining) opts))))))

(defn -main [& args]
  (let [opts (parse-cli-args args)]
    (if-not (:model-path opts)
      (do
        (println "Usage: clojure -M:quantize --model <model-path> [--output <output-path>] [--precision auto|int4|int8]")
        (System/exit 1))
      (do
        (quantize-model! opts)
        (System/exit 0)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
