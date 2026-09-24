(ns einsum.logic.nn
  "High-level neural network layer constructors generating pure Tensor Logic Hiccup ASTs.")

(defn linear
  "Linear projection layer: [:= out-term ?attrs x-term w-term]."
  ([out-term x-term w-term]
   (linear out-term x-term w-term nil))
  ([out-term x-term w-term attrs]
   (cond-> [:= out-term]
     attrs (conj attrs)
     true (conj x-term w-term))))

(defn mlp
  "Two-layer Feed-Forward MLP block with intermediate activation:
   h = Act(x @ fc_w)
   out = h @ proj_w"
  ([out-term x-term fc-w-term proj-w-term]
   (mlp out-term x-term fc-w-term proj-w-term {:act :silu}))
  ([out-term x-term fc-w-term proj-w-term attrs]
   (let [h-idxs (vec (concat (drop-last (rest x-term)) [(last (rest fc-w-term))]))
         h-term (into [[:mlp_h (first out-term)]] h-idxs)]
     [:block {:name :mlp}
      (cond-> [:= h-term]
        attrs (conj attrs)
        true (conj x-term fc-w-term))
      [:= out-term h-term proj-w-term]])))

(defn causal-attention
  "Multi-head causal attention block in pure Einstein summation equations."
  [out-term x-term w-q-term w-k-term w-v-term w-o-term & [opts]]
  (let [scale (or (:scale opts) 0.125)
        prefix (first out-term)
        p-dim (or (second (rest x-term)) :p)
        p-q (keyword (str (name p-dim) "-q"))
        p-k (keyword (str (name p-dim) "-k"))]
    [:block {:name [:attention prefix]}
     [:= [[:q prefix] :b p-dim :h :dh] x-term w-q-term]
     [:= [[:k prefix] :b p-dim :h :dh] x-term w-k-term]
     [:= [[:v prefix] :b p-dim :h :dh] x-term w-v-term]
     [:= [[:scores prefix] :b :h p-q p-k] {:scale scale} [[:q prefix] :b p-q :h :dh] [[:k prefix] :b p-k :h :dh]]
     [:= [[:ctx prefix] :b p-q :h :dh] [[:scores prefix] :b :h p-q p-k] [[:v prefix] :b p-k :h :dh]]
     [:= out-term [[:ctx prefix] :b p-q :h :dh] w-o-term]]))

(defn residual-block
  "Wraps layers in a residual block, accumulating skip connection with layer outputs."
  [out-term in-term layers]
  [:block {:name :residual}
   [:= out-term in-term]
   (if (vector? (first layers))
     layers
     [layers])])
