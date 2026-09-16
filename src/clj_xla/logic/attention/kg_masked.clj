(ns clj-xla.logic.attention.kg-masked
  "Knowledge-Graph Masked Self-Attention in StableHLO and Declarative Tensor Logic (Experiment E2).
   Encodes symbolic relations from resident VRAM knowledge graphs directly into attention scores:
   Attn(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M_causal + gamma * (T R T^T) ) V.
   Constrains autoregressive attention to valid relational links, suppressing hallucinations
   and ignoring adversarial distractor entities."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

;; ==============================================================================
;; 1. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn kg-token-adjacency-ast
  "Constructs Tensor Logic AST for projecting entity relations to token adjacency:
   TR = T * R        [L, N] x [N, N] -> [L, N]
   M_raw = TR * T^T  [L, N] x [L, N] -> [L, L]
   M_kg = gamma * M_raw"
  ([_seq-len _entity-count]
   (kg-token-adjacency-ast _seq-len _entity-count nil))
  ([_seq-len _entity-count opts]
   (let [g (double (or (:gamma opts) 1.0))]
     [:block {:name :kg_token_adjacency}
      [:= [:TR :l :e2] [:T :l :e1] [:R :e1 :e2]]
      [:= [:M_raw :p_q :p_k] [:TR :p_q :e2] [:T :p_k :e2]]
      [:= [:M_kg :p_q :p_k] {:scale g} [:M_raw :p_q :p_k]]])))

(defn kg-masked-self-attention-ast
  "Constructs Tensor Logic AST for Knowledge-Graph Masked Self-Attention:
   1. Project KG to token adjacency:
      TR = T * R
      M_raw = TR * T^T
      M_kg = gamma * M_raw
   2. Scaled Dot-Product Attention:
      scores = (Q . K^T) / sqrt(d_k)
   3. Apply KG bias:
      scores_biased = scores + M_kg
   4. Causal Softmax:
      probs = causal-softmax(scores_biased)
   5. Context contraction:
      out = probs @ V"
  ([batch num-heads seq-len head-dim entity-count]
   (kg-masked-self-attention-ast batch num-heads seq-len head-dim entity-count nil))
  ([_batch _num-heads _seq-len head-dim _entity-count opts]
   (let [g (double (or (:gamma opts) 1.0))
         scale (/ 1.0 (Math/sqrt (double head-dim)))
         window (:window opts)]
     [:block {:name :kg_masked_self_attention}
      [:= [:TR :l :e2] [:T :l :e1] [:R :e1 :e2]]
      [:= [:M_raw :p_q :p_k] [:TR :p_q :e2] [:T :p_k :e2]]
      [:= [:M_kg :p_q :p_k] {:scale g} [:M_raw :p_q :p_k]]
      [:= [:scores :b :h :p_q :p_k] {:scale scale} [:Q :b :p_q :h :dh] [:K :b :p_k :h :dh]]
      [:= [:scores_biased :b :h :p_q :p_k] {:op :add} [:scores :b :h :p_q :p_k] [:M_kg :p_q :p_k]]
      [:causal-softmax [:probs :b :h :p_q :p_k] [:scores_biased :b :h :p_q :p_k] (if window {:sliding-window window} {})]
      [:= [:out :b :p_q :h :dh] [:probs :b :h :p_q :p_k] [:V :b :p_k :h :dh]]])))

;; ==============================================================================
;; 2. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn compile-kg-token-adjacency
  "Compiles OpenXLA PJRT executable for computing token-level relational adjacency."
  ([seq-len entity-count]
   (compile-kg-token-adjacency (xla/get-context) seq-len entity-count nil))
  ([a b c]
   (if (map? a)
     (compile-kg-token-adjacency a b c nil)
     (compile-kg-token-adjacency (xla/get-context) a b c)))
  ([ctx seq-len entity-count opts]
   (let [l (long seq-len)
         n (long entity-count)
         invars [[:T [:tensor [l n] :f32]]
                 [:R [:tensor [n n] :f32]]]
         ast (kg-token-adjacency-ast l n opts)]
     (sym/compile-query ctx "kg_token_adjacency" invars ast [:M_kg]))))

(defn compile-kg-masked-self-attention
  "Compiles OpenXLA PJRT executable for full Knowledge-Graph Masked Self-Attention."
  ([batch num-heads seq-len head-dim entity-count]
   (compile-kg-masked-self-attention (xla/get-context) batch num-heads seq-len head-dim entity-count nil))
  ([a b c d e f]
   (if (map? a)
     (compile-kg-masked-self-attention a b c d e f nil)
     (compile-kg-masked-self-attention (xla/get-context) a b c d e f)))
  ([ctx batch num-heads seq-len head-dim entity-count opts]
   (let [b (long batch)
         h (long num-heads)
         l (long seq-len)
         d (long head-dim)
         n (long entity-count)
         invars [[:Q [:tensor [b l h d] :f32]]
                 [:K [:tensor [b l h d] :f32]]
                 [:V [:tensor [b l h d] :f32]]
                 [:T [:tensor [l n] :f32]]
                 [:R [:tensor [n n] :f32]]]
         ast (kg-masked-self-attention-ast b h l d n opts)]
     (sym/compile-query ctx "kg_masked_self_attention" invars ast [:out :probs :M_kg]))))

;; ==============================================================================
;; 3. Impure Execution Wrappers (PJRT Dispatch)
;; ==============================================================================

(defn compute-kg-token-adjacency!
  "Executes token adjacency computation on PJRT device runtime."
  [exec t-matrix r-matrix]
  (sym/run-query! exec {:T t-matrix :R r-matrix}))

(defn run-kg-masked-self-attention!
  "Executes KG-masked self-attention on PJRT device runtime."
  [exec q k-mat v-mat t-mat r-mat]
  (sym/run-query! exec {:Q q :K k-mat :V v-mat :T t-mat :R r-mat}))
