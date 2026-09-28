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

(defn- encode-with-specials [base-encode-fn special-tokens special-pattern text]
  (cond
    (nil? text) []
    (empty? text) []
    (nil? special-pattern) (base-encode-fn text)
    :else
    (let [matcher (re-matcher special-pattern text)]
      (loop [last-idx 0
             acc []]
        (if (.find matcher)
          (let [start (.start matcher)
                end (.end matcher)
                prefix (subs text last-idx start)
                tok-str (.group matcher)
                prefix-ids (if (empty? prefix) [] (base-encode-fn prefix))
                special-id (get special-tokens tok-str)]
            (recur end (conj (into acc prefix-ids) special-id)))
          (let [rem (subs text last-idx)]
            (if (empty? rem)
              acc
              (into acc (base-encode-fn rem)))))))))

(defrecord HFJsonTokenizer [vocab encoder bpe-ranks bos-token-id eos-token-id is-sp? special-tokens special-tokens-pattern]
  Tokenizer
  (encode [this text]
    (einsum.runtime.tokenizer.protocol/encode this text false))
  (encode [_this text _add-special-tokens?]
    (let [base-encode (fn [chunk]
                        (if is-sp?
                          (encode-sentencepiece-style encoder chunk)
                          (let [bpe-tok (bpe/->BPETokenizer vocab encoder bpe-ranks bos-token-id eos-token-id)]
                            (einsum.runtime.tokenizer.protocol/encode bpe-tok chunk _add-special-tokens?))))]
      (encode-with-specials base-encode special-tokens special-tokens-pattern text)))

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
        added-tokens (or (get data "added_tokens") [])
        added-encoder (into {} (map (fn [item] [(get item "content") (int (get item "id"))]) added-tokens))
        added-vocab (into {} (map (fn [item] [(int (get item "id")) (get item "content")]) added-tokens))
        encoder (merge (into {} (map (fn [[k v]] [k (int v)]) vocab-map)) added-encoder)
        vocab (merge (into {} (map (fn [[k v]] [(int v) k]) vocab-map)) added-vocab)
        special-tokens added-encoder
        sorted-specials (when (seq special-tokens)
                          (sort-by (comp - count) (keys special-tokens)))
        special-tokens-pattern (when (seq sorted-specials)
                                 (re-pattern (str "(" (str/join "|" (map #(java.util.regex.Pattern/quote %) sorted-specials)) ")")))
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
    (->HFJsonTokenizer vocab encoder bpe-ranks bos-id eos-id is-sp? special-tokens special-tokens-pattern)))
