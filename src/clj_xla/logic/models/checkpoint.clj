(ns clj-xla.logic.models.checkpoint
  "Pure Clojure binary checkpoint serializer and deserializer for TL-Nano models.
   Provides fast, self-contained persistence of model hyperparameters, vocabularies,
   relational memory operators, and tensor weights with zero external dependencies."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io])
  (:import [java.io BufferedInputStream BufferedOutputStream DataInputStream DataOutputStream FileInputStream FileOutputStream]))

(defn save-checkpoint
  "Saves TL-Nano checkpoint map `ckpt` to binary file `file-path`.
   `ckpt` must contain:
     :config    - model architecture hyperparameters map
     :metadata  - training run telemetry & provenance map
     :vocab     - active sub-vocabulary mappings {:bpe->active ... :active->bpe ...}
     :entities  - candidate entity metadata vector
     :relations - vector of relation keywords
     :r-maps    - map of {relation-keyword float-array}
     :params    - map of {tensor-keyword float-array}"
  [file-path ckpt]
  (let [target-file (io/file file-path)]
    (when-let [p (.getParentFile target-file)]
      (.mkdirs p))
    (with-open [dos (DataOutputStream. (BufferedOutputStream. (FileOutputStream. target-file) 65536))]
      ;; 1. Header Magic & Version
      (.writeUTF dos "TLNANO_V1")

      ;; 2. Serialized EDN Metadata (config, metadata, vocab, entities, relations)
      (let [meta-edn (pr-str (select-keys ckpt [:config :metadata :vocab :entities :relations]))
            meta-bytes (.getBytes ^String meta-edn "UTF-8")]
        (.writeInt dos (alength meta-bytes))
        (.write dos meta-bytes))

      ;; 3. Relational Memory Maps (:r-maps)
      (let [r-maps (:r-maps ckpt)]
        (.writeInt dos (count r-maps))
        (doseq [[rel ^floats r-arr] r-maps]
          (.writeUTF dos (name rel))
          (.writeInt dos (alength r-arr))
          (dotimes [i (alength r-arr)]
            (.writeFloat dos (aget r-arr i)))))

      ;; 4. Model Parameters (:params)
      (let [params (:params ckpt)]
        (.writeInt dos (count params))
        (doseq [[param-name ^floats p-arr] params]
          (.writeUTF dos (name param-name))
          (.writeInt dos (alength p-arr))
          (dotimes [i (alength p-arr)]
            (.writeFloat dos (aget p-arr i))))))
    (.length target-file)))

(defn load-checkpoint
  "Loads binary TL-Nano checkpoint from `file-path`.
   Returns a map with :config, :metadata, :vocab, :entities, :relations, :r-maps, and :params."
  [file-path]
  (let [target-file (io/file file-path)]
    (when-not (.exists target-file)
      (throw (ex-info (str "Checkpoint file not found: " file-path) {:file-path file-path})))
    (with-open [dis (DataInputStream. (BufferedInputStream. (FileInputStream. target-file) 65536))]
      (let [magic (.readUTF dis)]
        (when-not (= magic "TLNANO_V1")
          (throw (ex-info (str "Invalid checkpoint magic header: " magic) {:magic magic})))

        ;; 1. Read Metadata
        (let [meta-len (.readInt dis)
              meta-bytes (byte-array meta-len)]
          (.readFully dis meta-bytes)
          (let [meta-map (edn/read-string (String. meta-bytes "UTF-8"))

                ;; 2. Read Relational Memory Maps
                num-rel (.readInt dis)
                r-maps (into {}
                             (for [_ (range num-rel)]
                               (let [rel-str (.readUTF dis)
                                     arr-len (.readInt dis)
                                     arr (float-array arr-len)]
                                 (dotimes [i arr-len]
                                   (aset-float arr i (.readFloat dis)))
                                 [(keyword rel-str) arr])))

                ;; 3. Read Model Parameters
                num-params (.readInt dis)
                params (into {}
                             (for [_ (range num-params)]
                               (let [param-str (.readUTF dis)
                                     arr-len (.readInt dis)
                                     arr (float-array arr-len)]
                                 (dotimes [i arr-len]
                                   (aset-float arr i (.readFloat dis)))
                                 [(keyword param-str) arr])))]

            (merge meta-map
                   {:r-maps r-maps
                    :params params})))))))
