(ns einsum.runtime.tokenizer.bpe
  "GPT-2 Byte-Level Byte Pair Encoding (BPE) Tokenizer."
  (:require [einsum.runtime.tokenizer.protocol :refer [Tokenizer]]
            [clojure.data.json :as json]
            [clojure.string :as str]))

;; -----------------------------------------------------------------------------
;; Byte-to-Unicode Mapping (GPT-2 Byte Encoder)
;; -----------------------------------------------------------------------------

(def bytes->unicode
  "Creates the GPT-2 byte to unicode character mapping."
  (let [bs (concat (range (int \!) (inc (int \~)))
                   (range (int \¡) (inc (int \¬)))
                   (range (int \®) (inc (int \ÿ))))
        cs (vec bs)
        all-bs (range 256)
        missing-bs (remove (set bs) all-bs)
        new-cs (map (fn [i] (+ 256 i)) (range (count missing-bs)))
        b->u (into {} (map vector (concat bs missing-bs) (concat cs new-cs)))]
    (into {} (map (fn [[b u]] [b (str (char u))]) b->u))))

(def unicode->bytes
  "Inverse of bytes->unicode map."
  (into {} (map (fn [[b u]] [(first u) (unchecked-byte b)]) bytes->unicode)))

(defn string->byte-encoded
  "Encodes string bytes into GPT-2 byte-unicode representation."
  [s]
  (let [utf8-bytes (.getBytes ^String s "UTF-8")]
    (str/join (map #(get bytes->unicode (bit-and % 0xFF)) utf8-bytes))))

(defn byte-encoded->string
  "Decodes GPT-2 byte-unicode representation back into a UTF-8 string."
  [s]
  (let [bs (mapv #(get unicode->bytes % (byte 0)) s)
        arr (byte-array (count bs) (map byte bs))]
    (String. arr "UTF-8")))

;; -----------------------------------------------------------------------------
;; BPE Merge Algorithm
;; -----------------------------------------------------------------------------

(defn- get-pairs [word]
  (mapv vector word (rest word)))

(defn- bpe-word [word bpe-ranks]
  (loop [w (mapv str word)]
    (let [pairs (get-pairs w)]
      (if (empty? pairs)
        (str/join " " w)
        (let [valid-pairs (filter #(contains? bpe-ranks %) pairs)]
          (if (empty? valid-pairs)
            (str/join " " w)
            (let [pair (first (sort-by #(get bpe-ranks %) valid-pairs))
                  [first-part second-part] pair
                  new-w (loop [in w out []]
                          (cond
                            (empty? in) out
                            (and (= (first in) first-part)
                                 (> (count in) 1)
                                 (= (second in) second-part))
                            (recur (subvec in 2) (conj out (str first-part second-part)))
                            :else
                            (recur (subvec in 1) (conj out (first in)))))]
              (recur new-w))))))))

;; -----------------------------------------------------------------------------
;; BPETokenizer Record & Protocol Implementation
;; -----------------------------------------------------------------------------

(defrecord BPETokenizer [vocab encoder bpe-ranks bos-token-id eos-token-id]
  Tokenizer
  (encode [this text]
    (einsum.runtime.tokenizer.protocol/encode this text false))
  (encode [_this text _add-special-tokens?]
    (if (str/blank? text)
      []
      (let [regex #"'(?:[sdt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            matcher (re-matcher regex text)
            matches (loop [acc []]
                      (if (.find matcher)
                        (recur (conj acc (.group matcher)))
                        acc))
            tokens (flatten
                    (for [token matches]
                      (let [bpe-str (string->byte-encoded token)
                            merged (bpe-word bpe-str bpe-ranks)]
                        (str/split merged #" "))))]
        (mapv #(get encoder % (get encoder "<|endoftext|>" 50256)) tokens))))

  (decode [this token-ids]
    (einsum.runtime.tokenizer.protocol/decode this token-ids true))
  (decode [_this token-ids _skip-special-tokens?]
    (let [raw-str (str/join (map #(get vocab % "") token-ids))]
      (byte-encoded->string raw-str)))

  (bos-id [_this] bos-token-id)
  (eos-id [_this] eos-token-id))

(defn load-bpe-tokenizer
  "Loads GPT-2 BPE Tokenizer from `vocab-path` (vocab.json) and `merges-path` (merges.txt)."
  [vocab-path merges-path]
  (let [encoder (json/read-str (slurp vocab-path))
        vocab (into {} (map (fn [[k v]] [v k]) encoder))
        merges-lines (str/split-lines (slurp merges-path))
        merge-pairs (for [line (remove #(str/starts-with? % "#") merges-lines)
                          :let [parts (str/split line #" ")]
                          :when (= 2 (count parts))]
                      [(first parts) (second parts)])
        bpe-ranks (into {} (map-indexed (fn [i pair] [pair i]) merge-pairs))
        eos-id (get encoder "<|endoftext|>" 50256)]
    (->BPETokenizer vocab encoder bpe-ranks eos-id eos-id)))
