(ns einsum.runtime.tokenizer.hf-json
  "HuggingFace fast tokenizer.json loader supporting both GPT-2 BPE and SentencePiece formats."
  (:require [einsum.runtime.tokenizer.bpe :as bpe]
            [einsum.runtime.tokenizer.protocol :refer [Tokenizer]]
            [clojure.data.json :as json]
            [clojure.string :as str]))

(defn- encode-sentencepiece-style [encoder text]
  (if (str/blank? text)
    []
    (let [sp-text (str/replace text " " "\u2581")
          len (count sp-text)]
      (loop [idx 0
             acc []]
        (if (>= idx len)
          acc
          (let [match (loop [l (min len (+ idx 64))]
                        (if (<= l idx)
                          nil
                          (let [sub (subs sp-text idx l)]
                            (if-let [id (get encoder sub)]
                              [id (- l idx)]
                              (recur (dec l))))))]
            (if match
              (let [[id match-len] match]
                (recur (+ idx match-len) (conj acc id)))
              (let [single-ch (subs sp-text idx (inc idx))
                    id (get encoder single-ch (get encoder "<|endoftext|>" 0))]
                (recur (inc idx) (conj acc id))))))))))

(defrecord HFJsonTokenizer [vocab encoder bpe-ranks bos-token-id eos-token-id is-sp?]
  Tokenizer
  (encode [this text]
    (einsum.runtime.tokenizer.protocol/encode this text false))
  (encode [_this text _add-special-tokens?]
    (if is-sp?
      (encode-sentencepiece-style encoder text)
      (let [bpe-tok (bpe/->BPETokenizer vocab encoder bpe-ranks bos-token-id eos-token-id)]
        (einsum.runtime.tokenizer.protocol/encode bpe-tok text _add-special-tokens?))))

  (decode [this token-ids]
    (einsum.runtime.tokenizer.protocol/decode this token-ids true))
  (decode [_this token-ids _skip-special-tokens?]
    (if is-sp?
      (let [raw-str (str/join (map #(get vocab % "") token-ids))]
        (str/replace raw-str "\u2581" " "))
      (let [bpe-tok (bpe/->BPETokenizer vocab encoder bpe-ranks bos-token-id eos-token-id)]
        (einsum.runtime.tokenizer.protocol/decode bpe-tok token-ids _skip-special-tokens?))))

  (bos-id [_this] bos-token-id)
  (eos-id [_this] eos-token-id))

(defn load-hf-json-tokenizer
  "Loads HuggingFace `tokenizer.json` file keeping raw string tokens intact."
  [json-path]
  (let [data (json/read-str (slurp json-path))
        vocab-map (or (get-in data ["model" "vocab"]) {})
        merges (or (get-in data ["model" "merges"]) [])
        encoder (into {} (map (fn [[k v]] [k (int v)]) vocab-map))
        vocab (into {} (map (fn [[k v]] [(int v) k]) vocab-map))
        is-sp? (some #(str/includes? % "\u2581") (keys encoder))
        merge-pairs (for [item merges
                          :let [parts (cond
                                        (string? item) (str/split item #" ")
                                        (sequential? item) item
                                        :else [])]
                          :when (= 2 (count parts))]
                      [(first parts) (second parts)])
        bpe-ranks (into {} (map-indexed (fn [i pair] [pair i]) merge-pairs))
        eos-id (or (get encoder "</s>")
                   (get encoder "<eos>")
                   (get encoder "<|endoftext|>")
                   (get encoder "<|im_end|>")
                   1)
        bos-id (or (get encoder "<bos>")
                   (get encoder "<s>")
                   (get encoder "<|endoftext|>")
                   (get encoder "<|im_start|>")
                   2)]
    (->HFJsonTokenizer vocab encoder bpe-ranks bos-id eos-id is-sp?)))
