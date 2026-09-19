(ns einsum.kb.device
  "Experiment E22: In-Graph KB Query Dispatch.
   Tier 2 KB-query arm conforming to Repository Rules 1, 2, and 4 (Pure XLA Execution, Sans-IO core).
   Implements:
   - VRAM-resident relation matrices [3 N N] (PARENT, SIBLING, ANCESTOR)
   - Compiled OpenXLA PJRT dynamic-slice query executable (row gather)
   - Protocol query window recognition via token-ID compare
   - Device-resident verbalization bridge (entity indices -> token IDs)
   - Exact parity with E21 Knowledge Base (Gate P0b)"
  (:require [einsum.core :as xla]
            [einsum.logic.lower :as lower]
            [clojure.string :as str]))

;; ==============================================================================
;; 1. Relation Mapping & Host Reference Queries
;; ==============================================================================

(def RELATION-INDICES
  {:parent 0
   :sibling 1
   :ancestor 2})

(def INDEX-RELATIONS
  {0 :parent
   1 :sibling
   2 :ancestor})

(defn relation->index [rel-kw]
  (get RELATION-INDICES rel-kw))

(defn index->relation [idx]
  (get INDEX-RELATIONS idx))

(defn query-host-kb-reference
  "Computes ground-truth relational answers directly from E21 KB facts for entity c."
  [kb rel-kw ^long c ^long _num-entities]
  (let [parents-facts (or (get-in kb [:facts :parent]) #{})
        child->parents (reduce (fn [acc [p ch]] (update acc ch (fnil conj #{}) p)) {} parents-facts)]
    (case rel-kw
      :parent
      (vec (sort (get child->parents c #{})))

      :sibling
      (let [ps (get child->parents c #{})]
        (if (empty? ps)
          []
          (vec (sort (filter (fn [s]
                               (and (not= s c)
                                    (= ps (get child->parents s #{}))))
                             (keys child->parents))))))

      :ancestor
      (loop [frontier (get child->parents c #{})
             acc #{}]
        (if (empty? frontier)
          (vec (sort acc))
          (let [curr (first frontier)
                next-acc (conj acc curr)
                next-parents (remove next-acc (get child->parents curr #{}))]
            (recur (into (disj frontier curr) next-parents) next-acc)))))))

;; ==============================================================================
;; 2. VRAM Relation Matrix Construction
;; ==============================================================================

(defn build-relation-matrices
  "Builds stacked [3 N N] float matrix for PARENT, SIBLING, and ANCESTOR relations.
   Row i corresponds to query entity i; entry [rel, i, j] = 1.0f if j is in rel(i)."
  [kb ^long num-entities]
  (let [n num-entities
        total-size (* 3 n n)
        table-data (float-array total-size)]
    (doseq [rel-kw [:parent :sibling :ancestor]]
      (let [rel-idx (long (relation->index rel-kw))
            rel-offset (* rel-idx n n)]
        (dotimes [c n]
          (let [answers (query-host-kb-reference kb rel-kw c n)
                row-offset (+ rel-offset (* c n))]
            (doseq [ans answers]
              (when (< (long ans) n)
                (aset table-data (int (+ row-offset (long ans))) (float 1.0))))))))
    {:table table-data
     :shape [3 n n]
     :num-entities n}))

;; ==============================================================================
;; 3. OpenXLA PJRT Query Graph Compilation & Device Execution
;; ==============================================================================

(defn compile-kb-query-executable
  "Compiles OpenXLA PJRT dynamic-slice query executable for gathering row [1 1 N] from [3 N N]."
  ([num-entities]
   (compile-kb-query-executable (xla/get-context) num-entities))
  ([ctx ^long num-entities]
   (let [n num-entities
         invars [[:KB [:tensor [3 n n] :f32]]
                 [:rel [:tensor [1] :i32]]
                 [:arg [:tensor [1] :i32]]]
         ast [:dynamic-slice [:row :one_r :one_c :dim] [:KB :r :c :dim]
              {:slice-sizes [1 1 n] :start-indices [:rel :arg 0]}]
         graph (lower/ast->graph "kb_device_query" invars ast #{:row})]
     (xla/compile-graph ctx graph))))

(defn allocate-device-kb!
  "Pins the [3 N N] relation table resident in PJRT VRAM."
  [ctx ^floats table-data ^long num-entities]
  (let [n num-entities]
    (xla/buffer-from-host-buffer ctx (:client ctx) table-data [3 n n] 11)))

(defn execute-device-query!
  "Executes in-graph row gather on device-resident relation table.
   Returns vector or float-array of length num-entities."
  [ctx exec device-kb-buf rel-idx arg-idx num-entities]
  (let [rel-arr (int-array [(long rel-idx)])
        arg-arr (int-array [(long arg-idx)])
        out-buf (xla/execute exec device-kb-buf rel-arr arg-arr)
        row-slice (xla/to-host-slice out-buf 0 (long num-entities) (long num-entities) :f32)]
    (xla/destroy-buffer! ctx out-buf)
    row-slice))

(defn row-slice->entity-indices
  "Extracts non-zero entity indices where gathered activation > 0.5."
  [row-data]
  (let [n (count row-data)]
    (vec
     (sort
      (filter (fn [i]
                (> (double (nth row-data i)) 0.5))
              (range n))))))

;; ==============================================================================
;; 4. Token-ID Protocol Detection & Verbalization Bridge
;; ==============================================================================

(def DEFAULT-DIGIT-TOKENS
  ;; Gemma 4 single-token digit mappings
  {0 236771
   1 236770
   2 236778
   3 236800
   4 236812
   5 236810
   6 236825
   7 236832
   8 236828
   9 236819})

(def TOKEN->DIGIT
  (into {} (map (fn [[k v]] [v k]) DEFAULT-DIGIT-TOKENS)))

(def RELATION-TOKENS
  ;; Gemma 4 token IDs for relation tokens (with or without leading space)
  {41903 :ancestor    ;; " ancestors"
   209410 :ancestor   ;; "ancest"
   36791 :sibling     ;; " siblings"
   223663 :sibling    ;; "siblings"
   108032 :sibling    ;; "sibling"
   3724 :parent       ;; " parent"
   5432 :parent       ;; "parent"
   33059 :parent})    ;; "parents"

(defn find-kb-query-window
  "Inspects token stream for a completed query window delimited by open-tag-id and close-tag-id.
   Returns {:relation rel-kw, :rel-idx rel-idx, :arg entity-arg} or nil."
  [tokens ^long open-tag-id ^long close-tag-id]
  (let [tok-vec (vec tokens)
        n (count tok-vec)]
    (when (and (>= n 3) (= (peek tok-vec) close-tag-id))
      ;; Search backward for open-tag-id
      (let [open-idx
            (loop [i (- n 2)]
              (cond
                (< i 0) nil
                (= (nth tok-vec i) open-tag-id) i
                (> (- n i) 12) nil ;; Max query window 12 tokens
                :else (recur (dec i))))]
        (when open-idx
          (let [inner-tokens (subvec tok-vec (inc open-idx) (dec n))
                ;; Filter whitespace token 236743
                content-tokens (filterv #(not= % 236743) inner-tokens)
                rel-kw (some #(get RELATION-TOKENS %) content-tokens)
                arg-val (some #(get TOKEN->DIGIT %) content-tokens)]
            (when (and rel-kw arg-val)
              {:relation rel-kw
               :rel-idx (relation->index rel-kw)
               :arg arg-val
               :open-idx open-idx
               :close-idx (dec n)})))))))

(defn format-answer-tokens
  "Formats entity indices into response tokens enclosed by ans-open-id and ans-close-id.
   Example: <|tool_response> 1 0 <tool_response|>"
  ([ans-open-id ans-close-id index-token-map entity-indices]
   (format-answer-tokens ans-open-id ans-close-id index-token-map entity-indices 236743))
  ([ans-open-id ans-close-id index-token-map entity-indices space-tok-id]
   (let [sorted-indices (sort entity-indices)
         digit-tokens (mapv #(get index-token-map % (get DEFAULT-DIGIT-TOKENS %)) sorted-indices)
         inner (if (seq digit-tokens)
                 (vec (interleave (repeat space-tok-id) digit-tokens))
                 [space-tok-id])]
     (vec (concat [ans-open-id] inner [ans-close-id])))))

(defn extract-answer-entities
  "Extracts the set of returned entity IDs from the final verbalized answer.
   Handles formats like:
   - 'The ancestors of person 3 are persons 2 and 6.' -> #{2 6}
   - 'Person 2 has no ancestors.' -> #{}
   - 'The siblings are 5 and 7' -> #{5 7}
   - 'none' -> #{}"
  [text query-arg]
  (let [verbalized (cond
                     (str/includes? text "<tool_response|>")
                     (last (str/split text #"<tool_response\|>"))

                     (str/includes? text "<|turn>model")
                     (last (str/split text #"<\|turn>model\n?"))

                     :else text)
        lower (str/lower-case verbalized)]
    (if (or (re-find #"\b(no|none|neither|not have|has no)\b" lower)
            (re-find #"no (ancestors|siblings|parents)" lower))
      #{}
      (let [target-part (if-let [idx (or (let [i (.indexOf verbalized " are ")] (when (>= i 0) (+ i 5)))
                                         (let [i (.indexOf verbalized " is ")] (when (>= i 0) (+ i 4)))
                                         (let [i (.indexOf verbalized " are:")] (when (>= i 0) (+ i 5))))]
                          (subs verbalized idx)
                          (str/replace verbalized (re-pattern (str "(?i)(?:of )?person\\s*" query-arg)) ""))
            digits (mapv #(Integer/parseInt %) (re-seq #"\d+" target-part))]
        (set digits)))))
