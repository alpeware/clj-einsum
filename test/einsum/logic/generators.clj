(ns einsum.logic.generators
  "Generative test.check generators for Tensor Logic Hiccup ASTs, Einsum equations, and symbolic indices."
  (:require [clojure.test.check.generators :as gen]))

(def gen-index-kw
  "Generator for symbolic tensor dimension index keywords (e.g. :b, :p, :d, :h)."
  (gen/elements [:b :p :q :k :v :d :h :dh :m :n :k-dim :in-dim :out-dim :x-dim :y-dim]))

(def gen-tensor-name
  "Generator for tensor names as keywords."
  (gen/elements [:x :y :z :w :q :k :v :out :attn :ffn :gate :up :down :gamma :beta :bias]))

(def gen-indices
  "Generator for a distinct vector of symbolic indices."
  (gen/vector-distinct gen-index-kw {:min-elements 1 :max-elements 4}))

(def gen-head
  "Generator for an Einsum equation head vector [:name & indices]."
  (gen/fmap
   (fn [[t-name idxs]]
     (into [t-name] idxs))
   (gen/tuple gen-tensor-name gen-indices)))

(def gen-body-term
  "Generator for an Einsum equation body input term [:name & indices]."
  (gen/fmap
   (fn [[t-name idxs]]
     (into [t-name] idxs))
   (gen/tuple gen-tensor-name gen-indices)))

(def gen-attrs-map
  "Generator for optional equation attributes map."
  (gen/one-of
   [(gen/return nil)
    (gen/hash-map :scale (gen/double* {:min 0.01 :max 10.0 :NaN? false :infinite? false}))
    (gen/hash-map :act (gen/elements [:silu :gelu :relu :tanh]))
    (gen/hash-map :scale (gen/double* {:min 0.01 :max 10.0 :NaN? false :infinite? false})
                  :act (gen/elements [:silu :gelu :relu :tanh]))]))

(def gen-primitive-eqn
  "Generator for a valid primitive Einsum equation [:= head ?attrs & body-terms]."
  (gen/fmap
   (fn [[head attrs body-terms]]
     (cond-> [:= head]
       attrs (conj attrs)
       true (into body-terms)))
   (gen/tuple gen-head gen-attrs-map (gen/vector gen-body-term 1 3))))

(def gen-binary-contraction
  "Generator for a binary contraction equation with matching contracting dimensions."
  (gen/fmap
   (fn [[batch-dim contract-dim lhs-free rhs-free lhs-name rhs-name head-name]]
     (let [lhs-idx [batch-dim lhs-free contract-dim]
           rhs-idx [batch-dim contract-dim rhs-free]
           head-idx [batch-dim lhs-free rhs-free]]
       [:= (into [head-name] head-idx)
        (into [lhs-name] lhs-idx)
        (into [rhs-name] rhs-idx)]))
   (gen/tuple (gen/return :b)
              (gen/return :k)
              (gen/elements [:m :p :q :i])
              (gen/elements [:n :d :v :j])
              gen-tensor-name
              gen-tensor-name
              gen-tensor-name)))
