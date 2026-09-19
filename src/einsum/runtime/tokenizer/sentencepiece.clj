(ns einsum.runtime.tokenizer.sentencepiece
  "SentencePiece / Unigram tokenizer stub for Gemma architecture."
  (:require [einsum.runtime.tokenizer.protocol :refer [Tokenizer]]))

(defrecord SentencePieceTokenizer [vocab-path bos-token-id eos-token-id]
  Tokenizer
  (encode [_this _text]
    (throw (ex-info "SentencePiece tokenizer encoding not implemented yet" {:path vocab-path})))
  (encode [_this _text _add-special-tokens?]
    (throw (ex-info "SentencePiece tokenizer encoding not implemented yet" {:path vocab-path})))
  (decode [_this _token-ids]
    (throw (ex-info "SentencePiece tokenizer decoding not implemented yet" {:path vocab-path})))
  (decode [_this _token-ids _skip-special-tokens?]
    (throw (ex-info "SentencePiece tokenizer decoding not implemented yet" {:path vocab-path})))
  (bos-id [_this] bos-token-id)
  (eos-id [_this] eos-token-id))

(defn load-sentencepiece-tokenizer
  "Stub constructor for SentencePiece tokenizer."
  [vocab-path]
  (->SentencePieceTokenizer vocab-path 2 1))
