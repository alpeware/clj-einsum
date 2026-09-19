(ns einsum.logic.shape
  "Symbolic and concrete shape unification and dimension consistency validation for Tensor Logic."
  (:require [einsum.logic.ast :as ast]))

(defn unify-shapes
  "Unifies symbolic dimension indices with concrete input shapes across all equations.
   Validates dimension consistency on contracting indices and computes output shapes for all heads.
   Returns a map from tensor identifier keyword to concrete shape vector."
  [in-shapes equations]
  (reduce
   (fn [known-shapes eqn]
     (cond
       (= (first eqn) :while)
       (let [out-spec (second eqn)
             in-spec (nth eqn 2)
             out-names (mapv #(if (vector? %) (first %) %) (if (vector? out-spec) out-spec [out-spec]))
             in-names (mapv #(if (vector? %) (first %) %) (if (vector? in-spec) in-spec [in-spec]))]
         (reduce (fn [acc [out-n in-n]]
                   (if-let [s (get acc in-n)]
                     (assoc acc out-n s)
                     acc))
                 known-shapes
                 (map vector out-names in-names)))

       (= (first eqn) :constant)
       (let [h (second eqn)
             h-name (if (vector? h) (first h) h)
             attrs (ast/attrs eqn)
             explicit-shape (or (:shape attrs) [])]
         (assoc known-shapes h-name (vec explicit-shape)))

       (= (first eqn) :dynamic-update-slice)
       (let [head (ast/head eqn)
             head-name (first head)
             body (ast/body-terms eqn)
             op-name (first (first body))]
         (assoc known-shapes head-name (get known-shapes op-name)))

       (or (= (first eqn) :compare)
           (= (first eqn) :not)
           (= (first eqn) :and)
           (= (first eqn) :or)
           (= (first eqn) :cond)
           (= (first eqn) :+)
           (= (first eqn) :-)
           (= (first eqn) :add)
           (= (first eqn) :subtract)
           (= (first eqn) :*)
           (= (first eqn) :multiply)
           (= (first eqn) :/)
           (= (first eqn) :divide)
           (= (first eqn) :min)
           (= (first eqn) :minimum)
           (= (first eqn) :max)
           (= (first eqn) :maximum)
           (= (first eqn) :convert)
           (= (first eqn) :softmax)
           (= (first eqn) :fwht)
           (= (first eqn) :rht)
           (= (first eqn) :hadamard-block-128))
       (let [head (ast/head eqn)
             head-name (if (vector? head) (first head) head)
             head-idxs (when (vector? head) (vec (rest head)))
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             first-body-name (when (seq body) (first (first body)))
             first-body-shape (get known-shapes first-body-name [])
             shape (or (:shape attrs)
                       (if (empty? head-idxs)
                         first-body-shape
                         first-body-shape))]
         (assoc known-shapes head-name (vec shape)))

       (= (first eqn) :quip-dequant)
       (let [head (ast/head eqn)
             head-name (if (vector? head) (first head) head)
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             codes-name (first (first body))
             cb-name (first (second body))
             codes-shape (get known-shapes codes-name [1 1])
             cb-shape (get known-shapes cb-name [256 8])
             k (first codes-shape)
             n8 (second codes-shape)
             dim8 (if (>= (count cb-shape) 2) (second cb-shape) 8)
             out-shape (or (:shape attrs) [k (* n8 dim8)])]
         (assoc known-shapes head-name (vec out-shape)))

       (= (first eqn) :exl3-dequant)
       (let [head (ast/head eqn)
             head-name (if (vector? head) (first head) head)
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             trellis-name (first (first body))
             trellis-shape (get known-shapes trellis-name [1 1 48])
             in-tiles (first trellis-shape)
             out-tiles (second trellis-shape)
             out-shape (or (:shape attrs) [(* in-tiles 16) (* out-tiles 16)])]
         (assoc known-shapes head-name (vec out-shape)))

       (= (first eqn) :int4-unpack)
       (let [head (ast/head eqn)
             head-name (if (vector? head) (first head) head)
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             packed-name (first (first body))
             packed-shape (get known-shapes packed-name [1 1])
             rows (first packed-shape)
             half-cols (second packed-shape)
             out-shape (or (:shape attrs) [rows (* (long half-cols) 2)])]
         (assoc known-shapes head-name (vec out-shape)))

       (= (first eqn) :argmax)
       (let [head (ast/head eqn)
             head-name (if (vector? head) (first head) head)
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             in-name (first (first body))
             in-shape (get known-shapes in-name [1 1])
             rank (count in-shape)
             raw-axis (long (or (:axis attrs) (:dimension attrs) (dec rank)))
             axis (if (neg? raw-axis) (+ rank raw-axis) raw-axis)
             out-shape (or (:shape attrs)
                           (vec (concat (subvec (vec in-shape) 0 axis)
                                        (subvec (vec in-shape) (inc axis)))))]
         (assoc known-shapes head-name (vec out-shape)))

       :else
       (let [body (ast/body-terms eqn)
             head (ast/head eqn)
             head-name (first head)
             head-idxs (vec (rest head))
             attrs (ast/attrs eqn)
             explicit-shape (or (:shape attrs)
                                (:slice_sizes attrs)
                                (:slice-sizes attrs)
                                (when (and (= (first eqn) :slice)
                                           (or (:limit attrs) (:limit_indices attrs))
                                           (or (:start attrs) (:start_indices attrs)))
                                  (let [starts (or (:start attrs) (:start_indices attrs))
                                        limits (or (:limit attrs) (:limit_indices attrs))]
                                    (mapv - limits starts))))]
         (if explicit-shape
           (assoc known-shapes head-name (vec explicit-shape))
           (let [bindings
                 (reduce
                  (fn [binds term]
                    (let [t-name (first term)
                          t-idxs (vec (rest term))
                          t-shape (get known-shapes t-name)]
                      (if-not t-shape
                        (throw (ex-info "Unknown tensor shape for body term"
                                        {:tensor t-name :equation eqn :known (keys known-shapes)}))
                        (if (not= (count t-idxs) (count t-shape))
                          (throw (ex-info "Rank mismatch between term indices and tensor shape"
                                          {:tensor t-name :indices t-idxs :shape t-shape}))
                          (reduce
                           (fn [b [idx dim]]
                             (if (contains? b idx)
                               (let [existing (get b idx)]
                                 (if (not= existing dim)
                                   (throw (ex-info "Dimension mismatch for symbolic index"
                                                   {:index idx :expected existing :actual dim :tensor t-name}))
                                   b))
                               (assoc b idx dim)))
                           binds
                           (map vector t-idxs t-shape))))))
                  {}
                  body)
                 out-shape (mapv (fn [idx pos]
                                   (or (get bindings idx)
                                       (when-let [existing (get known-shapes head-name)]
                                         (nth existing pos nil))
                                       (throw (ex-info "Unbound symbolic index in head"
                                                       {:index idx :head head :bindings bindings}))))
                                 head-idxs
                                 (range (count head-idxs)))]
             (assoc known-shapes head-name out-shape))))))
   in-shapes
   equations))
