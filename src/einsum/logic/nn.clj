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
   (let [head-name (first out-term)
         h-name (keyword (str (name head-name) "_mlp_h"))
         h-idxs (vec (concat (drop-last (rest x-term)) [(last (rest fc-w-term))]))
         h-term (into [h-name] h-idxs)]
     [:block {:name :mlp}
      (cond-> [:= h-term]
        attrs (conj attrs)
        true (conj x-term fc-w-term))
      [:= out-term h-term proj-w-term]])))

(defn causal-attention
  "Multi-head causal attention block in pure Einstein summation equations."
  [out-term x-term w-q-term w-k-term w-v-term w-o-term & [opts]]
  (let [scale (or (:scale opts) 0.125)
        prefix (name (first out-term))
        q-name (keyword (str prefix "_q"))
        k-name (keyword (str prefix "_k"))
        v-name (keyword (str prefix "_v"))
        scores-name (keyword (str prefix "_scores"))
        ctx-name (keyword (str prefix "_ctx"))]
    [:block {:name :attention}
     [:= [q-name :b :p :h :dh] x-term w-q-term]
     [:= [k-name :b :p :h :dh] x-term w-k-term]
     [:= [v-name :b :p :h :dh] x-term w-v-term]
     [:= [scores-name :b :h :p-q :p-k] {:scale scale} [q-name :b :p-q :h :dh] [k-name :b :p-k :h :dh]]
     [:= [ctx-name :b :p-q :h :dh] [scores-name :b :h :p-q :p-k] [v-name :b :p-k :h :dh]]
     [:= out-term [ctx-name :b :p-q :h :dh] w-o-term]]))

(defn residual-block
  "Wraps layers in a residual block, accumulating skip connection with layer outputs."
  [out-term in-term layers]
  [:block {:name :residual}
   [:= out-term in-term]
   (if (vector? (first layers))
     layers
     [layers])])
