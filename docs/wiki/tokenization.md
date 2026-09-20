# Tokenization Algorithms & Special Tokens

Tokenization converts natural language text into numerical token ID sequences used by neural network embedding layers.

---

## 1. Tokenization Algorithms

### A. Byte-Pair Encoding (BPE)
- **Used by**: GPT-2, SmolLM, Llama 2/3.
- **Algorithm**: Iteratively merges frequent character pairs into subword vocabulary entries based on trained frequency tables.
- **Special Tokens**: `<|endoftext|>`.

### B. SentencePiece / Unigram
- **Used by**: Gemma 2, Gemma 3, Gemma 4, T5.
- **Algorithm**: Probabilistic subword tokenization treating input as a raw byte stream (preserving whitespace via ` ` underscore prefix).
- **Special Tokens**: `<bos>` ($1$), `<eos>` ($2$), `<turn|>` context control tokens for multi-turn instruct dialogs.

---

## 2. Clojure Implementation in `clj-einsum`

Tokenization in `clj-einsum` is managed via pure Clojure wrapper modules interfacing with SentencePiece / Hugging Face tokenizers:
- **Encoding**: [`einsum.runtime.tokenizer.core/encode`](../../src/einsum/runtime/tokenizer/core.clj) in [`einsum.runtime.tokenizer.core`](../../src/einsum/runtime/tokenizer/core.clj).
- **Decoding**: [`einsum.runtime.tokenizer.core/decode`](../../src/einsum/runtime/tokenizer/core.clj) in [`einsum.runtime.tokenizer.core`](../../src/einsum/runtime/tokenizer/core.clj).
- **Unit & Property Tests**: [`test/einsum/runtime/tokenizer_test.clj`](../../test/einsum/runtime/tokenizer_test.clj).
