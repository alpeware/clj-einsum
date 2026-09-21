(ns einsum.logic.memory.span
  "Subsequence search, span location, and score pooling operations over token spans."
  (:require [einsum.runtime.tokenizer.protocol :refer [bos-id encode]]
            [clojure.string :as str]))

(defn find-subsequence
  "Finds the 0-based start index of `needle` in `haystack`. Returns nil if not found."
  [needle haystack]
  (let [n-count (count needle)
        h-count (count haystack)]
    (cond
      (zero? n-count) 0
      (> n-count h-count) nil
      :else
      (first (keep-indexed (fn [i part] (when (= part needle) i))
                           (partition n-count 1 haystack))))))

(defn locate-head-span
  "Locates the [start-idx span-len] of `head` in `prompt-ids` using tokenizer.
   Tries both space-prefixed and direct tokenizations. Fails loudly if absent."
  [tokenizer prompt-ids head]
  (let [head-clean (str/trim head)
        b-id (bos-id tokenizer)
        strip-bos (fn [ids] (vec (remove #(= % b-id) ids)))
        h-ids-spaced (strip-bos (encode tokenizer (str " " head-clean)))
        h-ids-direct (strip-bos (encode tokenizer head-clean))
        idx-spaced (find-subsequence h-ids-spaced prompt-ids)
        idx-direct (find-subsequence h-ids-direct prompt-ids)]
    (cond
      (some? idx-spaced) [idx-spaced (count h-ids-spaced)]
      (some? idx-direct) [idx-direct (count h-ids-direct)]
      :else
      (throw (ex-info (str "Could not locate head entity span in prompt for: " head-clean)
                      {:head head-clean :prompt-ids prompt-ids})))))

(defn pool-scores-last-pos
  "Extracts entity scores at the last token position (dec prompt-len) from [P N] scores array."
  [^floats scores prompt-len n]
  (let [last-pos (dec (long prompt-len))
        out (float-array n)
        offset (* last-pos (long n))]
    (dotimes [j n]
      (aset-float out j (aget scores (+ offset j))))
    out))

(defn pool-scores-span-mean
  "Computes mean entity scores across the head entity token span [start-idx, start-idx + span-len)."
  [^floats scores start-idx span-len n]
  (let [start (long start-idx)
        slen (long span-len)
        n-long (long n)
        out (float-array n)]
    (dotimes [j n]
      (let [sum (loop [k 0 s 0.0]
                  (if (>= k slen)
                    s
                    (let [idx (+ (* (+ start k) n-long) j)]
                      (recur (inc k) (+ s (double (aget scores idx)))))))]
        (aset-float out j (float (/ sum (double slen))))))
    out))

(defn pool-scores-span-max
  "Computes max entity scores across the head entity token span [start-idx, start-idx + span-len)."
  [^floats scores start-idx span-len n]
  (let [start (long start-idx)
        slen (long span-len)
        n-long (long n)
        out (float-array n)]
    (dotimes [j n]
      (let [max-val (loop [k 1 m (double (aget scores (+ (* start n-long) j)))]
                      (if (>= k slen)
                        m
                        (let [idx (+ (* (+ start k) n-long) j)
                              v (double (aget scores idx))]
                          (recur (inc k) (Math/max m v)))))]
        (aset-float out j (float max-val))))
    out))

(defn top-entity-id
  "Returns index j in [0, n-1] of the top scoring entity."
  [^floats pooled-scores n]
  (loop [j 1
         best-id 0
         best-score (double (aget pooled-scores 0))]
    (if (>= j (long n))
      best-id
      (let [score (double (aget pooled-scores j))]
        (if (> score best-score)
          (recur (inc j) j score)
          (recur (inc j) best-id best-score))))))
