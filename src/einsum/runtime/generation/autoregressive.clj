(ns einsum.runtime.generation.autoregressive
  "Autoregressive token-by-token generation strategy."
  (:require [einsum.runtime.sampling :as sampling]))

(defn common-prefix-len
  "Returns the number of leading items shared by sequences xs and ys."
  [xs ys]
  (let [n (min (count xs) (count ys))]
    (loop [i 0]
      (if (and (< i n) (= (nth xs i) (nth ys i)))
        (recur (inc i))
        i))))

(defn init-kv-caches
  "Pre-allocates empty KV-cache descriptors up to `max-seq-len` for `num-layers`."
  ([num-layers _shape]
   (vec (repeat num-layers [nil nil])))
  ([num-layers batch num-kv-heads max-seq-len head-dim]
   (init-kv-caches num-layers [batch num-kv-heads max-seq-len head-dim])))

(defn generate-tokens-cached
  "Generates tokens autoregressively using KV-caching.
   `model-fn` is `(fn [token-ids kv-caches pos] [logits updated-kv-caches])`.
   `prompt-ids` is a vector of prompt token IDs.
   Opts: {:max-new-tokens :max-seq-len :num-layers :cache-shape :temperature :top-k :top-p :eos-token-id :callback}"
  [model-fn prompt-ids {:keys [max-new-tokens max-seq-len num-layers cache-shape
                               temperature top-k top-p eos-token-id callback]
                        :or {max-new-tokens 32 temperature 1.0 top-k 0 top-p 1.0}}]
  (let [prompt-len (count prompt-ids)
        _total-max-len (or max-seq-len (+ prompt-len max-new-tokens))
        init-caches (if (and num-layers cache-shape)
                      (init-kv-caches num-layers cache-shape)
                      nil)
        ;; Phase 1: Prompt Prefill
        [prefill-logits prefill-caches] (model-fn prompt-ids init-caches 0)
        first-next-token (sampling/sample-logits prefill-logits {:temperature temperature :top-k top-k :top-p top-p})]
    (when callback
      (callback first-next-token))
    (if (and eos-token-id (= first-next-token eos-token-id))
      (conj (vec prompt-ids) first-next-token)
      ;; Phase 2: Single-token Decoding Loop
      (loop [context (conj (vec prompt-ids) first-next-token)
             kv-caches prefill-caches
             pos prompt-len
             step 1]
        (if (>= step max-new-tokens)
          context
          (let [last-token (last context)
                [step-logits updated-caches] (model-fn [last-token] kv-caches pos)
                next-tok (sampling/sample-logits step-logits {:temperature temperature :top-k top-k :top-p top-p})
                next-context (conj context next-tok)]
            (when callback
              (callback next-tok))
            (if (and eos-token-id (= next-tok eos-token-id))
              next-context
              (recur next-context updated-caches (inc pos) (inc step)))))))))

(defn generate-tokens
  "Generates a sequence of token IDs autoregressively given a model step function, prompt token IDs, and generation options.
   `model-step-fn` is a function `(fn [context-token-ids] logits-vector)`.
   Opts: {:max-new-tokens :temperature :top-k :top-p :eos-token-id :callback}"
  [model-step-fn prompt-ids {:keys [max-new-tokens temperature top-k top-p eos-token-id callback]
                             :or {max-new-tokens 32 temperature 1.0 top-k 0 top-p 1.0} :as opts}]
  (if (:use-kv? opts)
    (generate-tokens-cached model-step-fn prompt-ids opts)
    (loop [context (vec prompt-ids)
           step 0]
      (if (>= step max-new-tokens)
        context
        (let [logits (model-step-fn context)
              next-token (sampling/sample-logits logits {:temperature temperature :top-k top-k :top-p top-p})
              next-context (conj context next-token)]
          (when callback
            (callback next-token))
          (if (and eos-token-id (= next-token eos-token-id))
            next-context
            (recur next-context (inc step))))))))

