(ns einsum.runtime.tokenizer.protocol
  "Polymorphic Tokenizer protocol definition.")

(defprotocol Tokenizer
  (encode [this text] [this text add-special-tokens?]
    "Encodes a string into a vector of integer token IDs.")
  (decode [this token-ids] [this token-ids skip-special-tokens?]
    "Decodes a sequence of integer token IDs into a UTF-8 string.")
  (bos-id [this] "Returns the Beginning-Of-Sequence token ID if present.")
  (eos-id [this] "Returns the End-Of-Sequence token ID if present."))
