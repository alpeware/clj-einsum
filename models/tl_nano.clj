(ns models.tl-nano
  "Native TL-Nano Open-Weights Pre-training on Consumer Hardware (Experiment E9).
   Implements a consumer-hardware native model architecture designed from first principles
   with Declarative Tensor Logic layers (hybrid KG-attention and relational memory unbinding)
   and joint autoregressive LM + InfoNCE contrastive pre-training."
  (:require [einsum.core :as xla]
            [einsum.logic.memory.contrastive :as contrast]
            [einsum.logic.symbolic :as sym]
            [einsum.compiler.pjrt :as pjrt]))

;; ==============================================================================
;; 1. Architecture Configurations
;; ==============================================================================

(def DEFAULT_TL_NANO_CONFIG
  "Default lightweight configuration for rapid testing and consumer GPU iteration."
  {:vocab-size 1024
   :hidden-dim 256
   :intermediate-dim 512 ;; 50% parameter reduction in MLP (2x rather than 4x)
   :num-layers 4
   :num-heads 4
   :head-dim 64
   :dim-mem 64
   :entity-count 32
   :cand-count 16
   :hybrid-layers #{1 3} ;; 0-indexed: layers 1 and 3 are hybrid TL blocks
   :max-seq-len 64
   :lambda-tl 0.3
   :tau 0.2
   :threshold 0.5
   :gamma 1.0
   :lambda-mem 0.2
   :norm-eps 1e-5})

(def TL_NANO_1B_CONFIG
  "Full 1B parameter configuration optimized for 24GB VRAM consumer GPUs (AMD RX 7900 XTX / RTX 4090)."
  {:vocab-size 32000
   :hidden-dim 2048
   :intermediate-dim 4096 ;; 2x D vs standard 8192 (4x D)
   :num-layers 16
   :num-heads 16
   :head-dim 128
   :dim-mem 256
   :entity-count 1024
   :cand-count 64
   :hybrid-layers #{3 7 11 15}
   :max-seq-len 512
   :lambda-tl 0.3
   :tau 0.2
   :threshold 0.5
   :gamma 1.0
   :lambda-mem 0.2
   :norm-eps 1e-5})

(defn tl-nano-config
  "Returns TL-Nano configuration map with optional overrides."
  ([] (tl-nano-config {}))
  ([overrides] (merge DEFAULT_TL_NANO_CONFIG overrides)))

;; ==============================================================================
;; 2. Pure AST Constructors (Sans-IO)
;; ==============================================================================

(defn tl-nano-layer-ast
  "Constructs Tensor Logic AST for layer `i` of TL-Nano:
   - If `i` is in `hybrid-layers`:
     KG-Masked Attention + Relational Memory Unbinding + GeGLU MLP.
   - Else:
     Standard Causal Attention + GeGLU MLP."
  [layer-idx batch seq-len cfg]
  (let [i (long layer-idx)
        b (long batch)
        l (long seq-len)
        h (long (:num-heads cfg))
        dh (long (:head-dim cfg))
        d (long (:hidden-dim cfg))
        _dff (long (:intermediate-dim cfg))
        _dm (long (:dim-mem cfg))
        _nc (long (:cand-count cfg))
        hybrid? (contains? (:hybrid-layers cfg) i)
        gamma (double (or (:gamma cfg) 1.0))
        lambda-mem (double (or (:lambda-mem cfg) 0.2))
        tau (double (or (:tau cfg) 0.2))
        inv-tau (/ 1.0 tau)
        attn-scale (/ 1.0 (Math/sqrt (double dh)))

        h-in (keyword (str "h_" i))
        h-out (keyword (str "h_" (inc i)))
        x-norm1 (keyword (str "x_norm1_" i))
        w-q (keyword (str "W_q_" i))
        w-k (keyword (str "W_k_" i))
        w-v (keyword (str "W_v_" i))
        w-o (keyword (str "W_o_" i))
        w-gate (keyword (str "W_gate_" i))
        w-up (keyword (str "W_up_" i))
        w-down (keyword (str "W_down_" i))

        q-flat (keyword (str "q_flat_" i))
        k-flat (keyword (str "k_flat_" i))
        v-flat (keyword (str "v_flat_" i))
        q (keyword (str "q_" i))
        k (keyword (str "k_" i))
        v (keyword (str "v_" i))
        scores-raw (keyword (str "scores_raw_" i))
        scores (keyword (str "scores_" i))
        attn-probs (keyword (str "attn_probs_" i))
        out (keyword (str "out_" i))
        out-flat (keyword (str "out_flat_" i))
        attn-out (keyword (str "attn_out_" i))
        h-attn (keyword (str "H_attn_" i))

        h-tl (keyword (str "H_tl_" i))
        x-norm2 (keyword (str "x_norm2_" i))
        gate-out (keyword (str "gate_out_" i))
        up-out (keyword (str "up_out_" i))
        mlp-act (keyword (str "mlp_act_" i))
        mlp-out (keyword (str "mlp_out_" i))]

    [:block {:name (keyword (str "tl_nano_layer_" i))}
     ;; 1. Pre-Attention RMSNorm
     [:rms-norm [x-norm1 :b :p :d] [h-in :b :p :d]]

     ;; 2. Attention Projections
     [:= [q-flat :b :p :d2] [x-norm1 :b :p :d1] [w-q :d1 :d2]]
     [:reshape [q :b :p :h :dh] [q-flat :b :p :d2] {:shape [b l h dh]}]
     [:= [k-flat :b :p :d2] [x-norm1 :b :p :d1] [w-k :d1 :d2]]
     [:reshape [k :b :p :h :dh] [k-flat :b :p :d2] {:shape [b l h dh]}]
     [:= [v-flat :b :p :d2] [x-norm1 :b :p :d1] [w-v :d1 :d2]]
     [:reshape [v :b :p :h :dh] [v-flat :b :p :d2] {:shape [b l h dh]}]

     ;; 3. Attention Scoring (with optional KG Mask)
     (if hybrid?
       [:block {:name (keyword (str "kg_attention_" i))}
        [:= [(keyword (str "TR_" i)) :p_q :e2] [:T :p_q :e1] [:R_adj :e1 :e2]]
        [:= [(keyword (str "M_raw_" i)) :p_q :p_k] [(keyword (str "TR_" i)) :p_q :e2] [:T :p_k :e2]]
        [:= [(keyword (str "M_kg_" i)) :p_q :p_k] {:scale gamma} [(keyword (str "M_raw_" i)) :p_q :p_k]]
        [:= [scores-raw :b :h :p_q :p_k] {:scale attn-scale} [q :b :p_q :h :dh] [k :b :p_k :h :dh]]
        [:= [scores :b :h :p_q :p_k] {:op :add} [scores-raw :b :h :p_q :p_k] [(keyword (str "M_kg_" i)) :p_q :p_k]]]
       [:= [scores :b :h :p_q :p_k] {:scale attn-scale} [q :b :p_q :h :dh] [k :b :p_k :h :dh]])

     ;; 4. Causal Softmax & Context Contraction
     [:causal-softmax [attn-probs :b :h :p_q :p_k] [scores :b :h :p_q :p_k]]
     [:= [out :b :p_q :h :dh] [attn-probs :b :h :p_q :p_k] [v :b :p_k :h :dh]]
     [:reshape [out-flat :b :p_q :d1] [out :b :p_q :h :dh] {:shape [b l d]}]
     [:= [attn-out :b :p_q :d] [out-flat :b :p_q :d1] [w-o :d1 :d]]
     [:+ [h-attn :b :p_q :d] [h-in :b :p_q :d] [attn-out :b :p_q :d]]

     ;; 5. Relational Memory Unbinding (Hybrid Layers Only)
     (if hybrid?
       (let [u-q (keyword (str "u_q_" i))
             u-target (keyword (str "u_target_" i))
             u-target-norm (keyword (str "u_target_norm_" i))
             u-cand (keyword (str "U_cand_" i))
             cand-scores (keyword (str "cand_scores_" i))
             valid-mask (keyword (str "valid_mask_" i))
             valid-weight (keyword (str "valid_weight_" i))
             clamped-scores (keyword (str "clamped_scores_" i))
             v-grounded (keyword (str "v_grounded_" i))
             v-bias (keyword (str "v_bias_" i))]
         [:block {:name (keyword (str "rel_memory_unbinding_" i))}
          [:= [u-q :b :p_q :dm] [h-attn :b :p_q :d] [:W_mem :d :dm]]
          [:= [u-target :b :p_q :dm2] [u-q :b :p_q :dm1] [:R_mem :dm1 :dm2]]
          [:rms-norm [u-target-norm :b :p_q :dm2] [u-target :b :p_q :dm2]]
          [:= [u-cand :nc :dm] [:E_cand :nc :d] [:W_mem :d :dm]]
          [:= [cand-scores :b :p_q :nc] {:scale inv-tau} [u-target-norm :b :p_q :dm] [u-cand :nc :dm]]
          [:compare [valid-mask :b :p_q :nc] [cand-scores :b :p_q :nc] [:threshold :one] {:comparison_direction "GT"}]
          [:convert [valid-weight :b :p_q :nc] {:target-dtype :f32} [valid-mask :b :p_q :nc]]
          [:= [clamped-scores :b :p_q :nc] [cand-scores :b :p_q :nc] [valid-weight :b :p_q :nc]]
          [:= [v-grounded :b :p_q :d] [clamped-scores :b :p_q :nc] [:E_cand :nc :d]]
          [:= [v-bias :b :p_q :d] {:scale lambda-mem} [v-grounded :b :p_q :d]]
          [:+ [h-tl :b :p_q :d] [h-attn :b :p_q :d] [v-bias :b :p_q :d]]])
       [:= [h-tl :b :p_q :d] [h-attn :b :p_q :d]])

     ;; 6. GeGLU Feed-Forward Network
     [:rms-norm [x-norm2 :b :p_q :d] [h-tl :b :p_q :d]]
     [:= [gate-out :b :p_q :dff] {:act :gelu} [x-norm2 :b :p_q :d] [w-gate :d :dff]]
     [:= [up-out :b :p_q :dff] [x-norm2 :b :p_q :d] [w-up :d :dff]]
     [:= [mlp-act :b :p_q :dff] [gate-out :b :p_q :dff] [up-out :b :p_q :dff]]
     [:= [mlp-out :b :p_q :d] [mlp-act :b :p_q :dff] [w-down :dff :d]]
     [:+ [h-out :b :p_q :d] [h-tl :b :p_q :d] [mlp-out :b :p_q :d]]]))

(defn tl-nano-model-ast
  "Generates full TL-Nano forward model AST."
  ([cfg]
   (tl-nano-model-ast 1 (:max-seq-len cfg) cfg))
  ([batch seq-len cfg]
   (let [b (long batch)
         l (long seq-len)
         num-layers (long (:num-layers cfg))
         h-last (keyword (str "h_" num-layers))]
     [:block {:name :tl_nano_model}
      ;; 1. Token Embedding via Gather
      [:gather [:h_0 :b :p :d] [:W_embed :v :d] [:x :b :p]]

      ;; 2. Layer Blocks
      (mapv (fn [i] (tl-nano-layer-ast i b l cfg)) (range num-layers))

      ;; 3. Final RMSNorm
      [:rms-norm [:H_final :b :p :d] [h-last :b :p :d]]

      ;; 4. Tied LM Head Logits: H_final [b p d] @ W_embed [v d]^T = logits [b p v]
      [:= [:logits :b :p :v] [:H_final :b :p :d] [:W_embed :v :d]]])))

;; ==============================================================================
;; 3. OpenXLA PJRT Compilation Boundary
;; ==============================================================================

(defn build-tl-nano-invars
  "Constructs typed input shape signature for TL-Nano compilation."
  [batch seq-len cfg]
  (let [b (long batch)
        l (long seq-len)
        v (long (:vocab-size cfg))
        d (long (:hidden-dim cfg))
        dff (long (:intermediate-dim cfg))
        dm (long (:dim-mem cfg))
        ne (long (:entity-count cfg))
        nc (long (:cand-count cfg))
        num-layers (long (:num-layers cfg))
        base-invars [[:x [:tensor [b l] :i32]]
                     [:W_embed [:tensor [v d] :f32]]
                     [:W_mem [:tensor [d dm] :f32]]
                     [:R_mem [:tensor [dm dm] :f32]]
                     [:E_cand [:tensor [nc d] :f32]]
                     [:T [:tensor [l ne] :f32]]
                     [:R_adj [:tensor [ne ne] :f32]]
                     [:threshold [:tensor [1] :f32]]]
        layer-invars (mapcat
                      (fn [i]
                        [[(keyword (str "W_q_" i)) [:tensor [d d] :f32]]
                         [(keyword (str "W_k_" i)) [:tensor [d d] :f32]]
                         [(keyword (str "W_v_" i)) [:tensor [d d] :f32]]
                         [(keyword (str "W_o_" i)) [:tensor [d d] :f32]]
                         [(keyword (str "W_gate_" i)) [:tensor [d dff] :f32]]
                         [(keyword (str "W_up_" i)) [:tensor [d dff] :f32]]
                         [(keyword (str "W_down_" i)) [:tensor [dff d] :f32]]])
                      (range num-layers))]
    (vec (concat base-invars layer-invars))))

(defn compile-tl-nano-forward
  "Compiles OpenXLA PJRT executable for full TL-Nano forward pass."
  ([batch seq-len cfg]
   (compile-tl-nano-forward (xla/get-context) batch seq-len cfg))
  ([ctx batch seq-len cfg]
   (let [b (long batch)
         l (long seq-len)
         invars (build-tl-nano-invars b l cfg)
         ast (tl-nano-model-ast b l cfg)
         hybrid-layers (or (:hybrid-layers cfg) #{})
         hybrid-targets (mapcat
                         (fn [i]
                           [(keyword (str "cand_scores_" i))
                            (keyword (str "clamped_scores_" i))])
                         hybrid-layers)
         target-heads (vec (concat [:logits :H_final] hybrid-targets))]
     (sym/compile-query ctx "tl_nano_forward" invars ast target-heads))))

(defn run-tl-nano-forward!
  "Executes compiled TL-Nano forward pass on PJRT device runtime."
  [exec inputs]
  (sym/run-query! exec inputs))

;; ==============================================================================
;; 4. Parameters & Joint Pre-training Optimization Loop
;; ==============================================================================

(defn- sample-gaussian-array
  [^java.util.Random rnd size scale]
  (let [sz (long size)
        sc (double scale)
        arr (float-array sz)]
    (dotimes [i sz]
      (aset-float arr i (float (* (.nextGaussian rnd) sc))))
    arr))

(defn init-tl-nano-params
  "Initializes all trainable parameters for TL-Nano."
  ([cfg]
   (init-tl-nano-params cfg (java.util.Random. 42)))
  ([cfg ^java.util.Random rnd]
   (let [v (long (:vocab-size cfg))
         d (long (:hidden-dim cfg))
         dff (long (:intermediate-dim cfg))
         dm (long (:dim-mem cfg))
         nc (long (:cand-count cfg))
         ne (long (:entity-count cfg))
         l (long (:max-seq-len cfg))
         num-layers (long (:num-layers cfg))
         scale (/ 1.0 (Math/sqrt (double d)))

         base {:W_embed (sample-gaussian-array rnd (* v d) scale)
               :W_mem (sample-gaussian-array rnd (* d dm) scale)
               :R_mem (sample-gaussian-array rnd (* dm dm) scale)
               :E_cand (sample-gaussian-array rnd (* nc d) scale)
               :T (float-array (* l ne) (float 0.0))
               :R_adj (float-array (* ne ne) (float 0.0))
               :threshold (float-array 1 (float (or (:threshold cfg) 0.5)))}
         layer-params (into {}
                            (mapcat
                             (fn [i]
                               [[(keyword (str "W_q_" i)) (sample-gaussian-array rnd (* d d) scale)]
                                [(keyword (str "W_k_" i)) (sample-gaussian-array rnd (* d d) scale)]
                                [(keyword (str "W_v_" i)) (sample-gaussian-array rnd (* d d) scale)]
                                [(keyword (str "W_o_" i)) (sample-gaussian-array rnd (* d d) scale)]
                                [(keyword (str "W_gate_" i)) (sample-gaussian-array rnd (* d dff) scale)]
                                [(keyword (str "W_up_" i)) (sample-gaussian-array rnd (* d dff) scale)]
                                [(keyword (str "W_down_" i)) (sample-gaussian-array rnd (* dff d) scale)]])
                             (range num-layers)))]
     (merge base layer-params))))

(defn pin-params-in-vram
  "Transfers and pins static layer weights in PJRT device VRAM.
   Returns a params map where static layer weights are MemorySegment device buffers."
  [ctx params cfg]
  (let [d (long (:hidden-dim cfg))
        dff (long (:intermediate-dim cfg))
        num-layers (long (:num-layers cfg))
        client (:client ctx)
        pin (fn [arr shape]
              (if (instance? java.lang.foreign.MemorySegment arr)
                arr
                (pjrt/buffer-from-host-buffer ctx client arr shape 11)))
        layer-pins
        (into {}
              (mapcat
               (fn [i]
                 [[(keyword (str "W_q_" i)) (pin (get params (keyword (str "W_q_" i))) [d d])]
                  [(keyword (str "W_k_" i)) (pin (get params (keyword (str "W_k_" i))) [d d])]
                  [(keyword (str "W_v_" i)) (pin (get params (keyword (str "W_v_" i))) [d d])]
                  [(keyword (str "W_o_" i)) (pin (get params (keyword (str "W_o_" i))) [d d])]
                  [(keyword (str "W_gate_" i)) (pin (get params (keyword (str "W_gate_" i))) [d dff])]
                  [(keyword (str "W_up_" i)) (pin (get params (keyword (str "W_up_" i))) [d dff])]
                  [(keyword (str "W_down_" i)) (pin (get params (keyword (str "W_down_" i))) [dff d])]])
               (range num-layers)))
        base-pins {:threshold (pin (:threshold params) [1])
                   :T (pin (:T params) [(:max-seq-len cfg) (:entity-count cfg)])
                   :R_adj (pin (:R_adj params) [(:entity-count cfg) (:entity-count cfg)])}]
    (merge params layer-pins base-pins)))

(defn free-pinned-params!
  "Releases VRAM device buffers allocated by pin-params-in-vram."
  [params]
  (doseq [[_k v] params]
    (when (instance? java.lang.foreign.MemorySegment v)
      (try (xla/destroy-buffer! v) (catch Exception _ nil)))))

(defn compute-joint-loss
  "Computes joint autoregressive LM cross-entropy loss + InfoNCE relational contrastive loss:
   L_total = L_LM + lambda_TL * L_InfoNCE.
   Supports optional skip-rel? flag when unified in-VRAM kernel handles relational step."
  ([exec params batch cfg]
   (compute-joint-loss exec params batch cfg false))
  ([exec params batch cfg skip-rel?]
   (let [tokens (:tokens batch)
         targets (:targets batch)
         triples (:triples batch)
         b (long (or (:batch-size batch) 1))
         l (long (:max-seq-len cfg))
         v (long (:vocab-size cfg))
         d (long (:hidden-dim cfg))
         dm (long (:dim-mem cfg))
         lambda-tl (double (or (:lambda-tl cfg) 0.3))
         tau (double (or (:tau cfg) 0.2))

         inputs (assoc params :x tokens)
         outputs (run-tl-nano-forward! exec inputs)
         ^floats logits (:logits outputs)
         ^floats h-final (:H_final outputs)

         ;; 1. Autoregressive Language Model Loss (over L-1 positions)
         lm-loss-acc (double-array 1)
         valid-pos (* b (dec l))
         probs (float-array (* b l v))]

     (dotimes [bi b]
       (dotimes [pos (dec l)]
         (let [row-idx (+ (* bi l v) (* pos v))
               target-tok (aget ^ints targets (+ (* bi l) pos))
               ;; Find max logit for stable softmax
               max-l (loop [vi 1 m (double (aget logits row-idx))]
                       (if (>= vi v)
                         m
                         (recur (inc vi) (Math/max m (double (aget logits (+ row-idx vi)))))))
               exp-sum (loop [vi 0 s 0.0]
                         (if (>= vi v)
                           s
                           (recur (inc vi) (+ s (Math/exp (- (double (aget logits (+ row-idx vi))) max-l))))))]
           (dotimes [vi v]
             (let [p (/ (Math/exp (- (double (aget logits (+ row-idx vi))) max-l)) exp-sum)]
               (aset-float probs (+ row-idx vi) (float p))))
           (let [p-target (double (aget probs (+ row-idx target-tok)))
                 nll (- (Math/log (Math/max 1e-12 p-target)))]
             (aset-double lm-loss-acc 0 (+ (aget lm-loss-acc 0) nll))))))

     (let [lm-loss (if (pos? valid-pos) (/ (aget lm-loss-acc 0) (double valid-pos)) 0.0)

           ;; 2. Relational Subspace Alignment InfoNCE Loss (on Knowledge Triples)
           infonce-info
           (if (and (not skip-rel?) (seq triples) (:W_mem params) (:R_mem params))
             (let [k-cnt (count triples)
                   ^floats w-mem (:W_mem params)
                   ^floats r-mem (:R_mem params)
                   ^floats w-embed (:W_embed params)
                   u-h (float-array (* k-cnt dm))
                   u-t (float-array (* k-cnt dm))
                   u-hr (float-array (* k-cnt dm))]
              ;; Project head & tail entities: u = e @ W_mem
               (doseq [k-idx (range k-cnt)]
                 (let [{:keys [head tail]} (nth triples k-idx)
                       h-off (* (long head) d)
                       t-off (* (long tail) d)
                       uk-off (* k-idx dm)]
                   (dotimes [j dm]
                     (let [sum-h (loop [i 0 s 0.0]
                                   (if (>= i d) s
                                       (recur (inc i) (+ s (* (double (aget w-embed (+ h-off i)))
                                                              (double (aget w-mem (+ (* i dm) j))))))))
                           sum-t (loop [i 0 s 0.0]
                                   (if (>= i d) s
                                       (recur (inc i) (+ s (* (double (aget w-embed (+ t-off i)))
                                                              (double (aget w-mem (+ (* i dm) j))))))))]
                       (aset-float u-h (+ uk-off j) (float sum-h))
                       (aset-float u-t (+ uk-off j) (float sum-t))))
                  ;; Contract head with R_mem: u_hr = u_h @ R_mem
                   (dotimes [j dm]
                     (let [sum-hr (loop [i 0 s 0.0]
                                    (if (>= i dm) s
                                        (recur (inc i) (+ s (* (double (aget u-h (+ uk-off i)))
                                                               (double (aget r-mem (+ (* i dm) j))))))))]
                       (aset-float u-hr (+ uk-off j) (float sum-hr))))))
              ;; Compute scores S_ij = (u_hr_i @ u_t_j) / tau
               (let [scores (float-array (* k-cnt k-cnt))
                     infonce-probs (float-array (* k-cnt k-cnt))]
                 (dotimes [i k-cnt]
                   (dotimes [j k-cnt]
                     (let [dot (loop [m 0 s 0.0]
                                 (if (>= m dm) s
                                     (recur (inc m) (+ s (* (double (aget u-hr (+ (* i dm) m)))
                                                            (double (aget u-t (+ (* j dm) m))))))))]
                       (aset-float scores (+ (* i k-cnt) j) (float (/ dot tau))))))
                ;; Cross-entropy over in-batch negatives with exact softmax probabilities
                 (let [pos-cnt (min k-cnt (long (or (:pos-count batch) k-cnt)))
                       loss-sum (loop [i 0 s 0.0]
                                  (if (>= i pos-cnt) s
                                      (let [row-off (* i k-cnt)
                                            max-s (loop [j 1 m (double (aget scores row-off))]
                                                    (if (>= j k-cnt) m
                                                        (recur (inc j) (Math/max m (double (aget scores (+ row-off j)))))))
                                            exp-sum (loop [j 0 es 0.0]
                                                      (if (>= j k-cnt) es
                                                          (recur (inc j) (+ es (Math/exp (- (double (aget scores (+ row-off j))) max-s))))))]
                                        (dotimes [j k-cnt]
                                          (let [p (/ (Math/exp (- (double (aget scores (+ row-off j))) max-s)) exp-sum)]
                                            (aset-float infonce-probs (+ row-off j) (float p))))
                                        (let [pos-s (double (aget scores (+ row-off i)))
                                              prob (/ (Math/exp (- pos-s max-s)) exp-sum)]
                                          (recur (inc i) (+ s (- (Math/log (Math/max 1e-12 prob)))))))))]
                   {:loss (/ loss-sum (double (max 1 pos-cnt)))
                    :probs infonce-probs
                    :u-h u-h
                    :u-t u-t
                    :u-hr u-hr})))
             {:loss 0.0 :probs nil :u-h nil :u-t nil :u-hr nil})

           total-loss (+ lm-loss (* lambda-tl (:loss infonce-info)))]
       {:total-loss total-loss
        :lm-loss lm-loss
        :infonce-loss (:loss infonce-info)
        :infonce-probs (:probs infonce-info)
        :u-h (:u-h infonce-info)
        :u-t (:u-t infonce-info)
        :u-hr (:u-hr infonce-info)
        :logits logits
        :probs probs
        :H_final h-final
        :outputs outputs}))))

(defn compile-embedding-update
  "Compiles OpenXLA PJRT executable for in-VRAM dW_embed contraction and SGD update:
   dW_embed[v, d] = G_logit[n, v]^T @ H_final[n, d]
   W_embed_new = W_embed - lr * dW_embed"
  ([batch-size seq-len cfg]
   (compile-embedding-update (xla/get-context) batch-size seq-len cfg (or (:lr cfg) 0.01)))
  ([a b c d]
   (if (map? a)
     (compile-embedding-update a b c d (or (:lr d) 0.01))
     (compile-embedding-update (xla/get-context) a b c d)))
  ([ctx batch-size seq-len cfg lr]
   (let [n (* (long batch-size) (dec (long seq-len)))
         v (long (:vocab-size cfg))
         d (long (:hidden-dim cfg))
         invars [[:G [:tensor [n v] :f32]]
                 [:H [:tensor [n d] :f32]]
                 [:W [:tensor [v d] :f32]]]
         ast [:block {:name :embedding_update}
              [:= [:dW :v :d] [:G :n :v] [:H :n :d]]
              [:= [:scaled_dW :v :d] {:scale (double lr)} [:dW :v :d]]
              [:- [:W_new :v :d] [:W :v :d] [:scaled_dW :v :d]]]
         target-heads [:W_new]]
     (sym/compile-query ctx "embedding_update" invars ast target-heads))))

(defn compile-relational-forward
  "Compiles OpenXLA PJRT executable for in-VRAM forward relational subspace projections:
   U_h = V_h * W, U_t = V_t * W, U_hr = U_h * R, Scores = (U_hr * U_t^T) / tau."
  ([batch-size hidden-dim dim-mem opts]
   (contrast/compile-contrastive-forward (xla/get-context) batch-size hidden-dim dim-mem opts))
  ([ctx batch-size hidden-dim dim-mem opts]
   (contrast/compile-contrastive-forward ctx batch-size hidden-dim dim-mem opts)))

(defn compile-relational-backward
  "Compiles OpenXLA PJRT executable for in-VRAM relational InfoNCE adjoint contractions:
   adj_Ut = G_S^T * U_hr, adj_Uhr = G_S * U_t, dR = U_h^T * adj_Uhr,
   adj_Uh = adj_Uhr * R^T, dW_h = V_h^T * adj_Uh, dW_t = V_t^T * adj_Ut, dW = dW_h + dW_t."
  ([batch-size hidden-dim dim-mem]
   (contrast/compile-contrastive-backward (xla/get-context) batch-size hidden-dim dim-mem))
  ([ctx batch-size hidden-dim dim-mem]
   (contrast/compile-contrastive-backward ctx batch-size hidden-dim dim-mem)))

(defn compile-in-vram-contrastive-step
  "Compiles OpenXLA PJRT executable for unified, 100% in-VRAM InfoNCE training step:
   Gathers embeddings, projects subspaces, in-graph softmax, analytical adjoints,
   backward contractions, and SGD updates directly in GPU memory."
  ([vocab-size hidden-dim k-triples dim-mem]
   (compile-in-vram-contrastive-step (xla/get-context) vocab-size hidden-dim k-triples dim-mem nil))
  ([ctx vocab-size hidden-dim k-triples dim-mem]
   (compile-in-vram-contrastive-step ctx vocab-size hidden-dim k-triples dim-mem nil))
  ([ctx vocab-size hidden-dim k-triples dim-mem opts]
   (contrast/compile-in-vram-contrastive-step ctx vocab-size hidden-dim k-triples dim-mem opts)))

(defn train-step
  "Executes an OpenXLA pre-training step via algebraic autodiff adjoints.
   Updates W_embed, W_mem, R_mem, and output weights with learning rate `lr`.
   Accepts optional `embed-exec` and `rel-exec` compiled OpenXLA kernels for in-VRAM contractions."
  ([exec params batch cfg lr]
   (train-step exec params batch cfg lr nil nil))
  ([exec params batch cfg lr embed-exec]
   (train-step exec params batch cfg lr embed-exec nil))
  ([exec params batch cfg lr embed-exec rel-bwd-exec]
   (let [unified-rel? (and rel-bwd-exec (= (get-in rel-bwd-exec [:graph :name]) "in_vram_contrastive_step"))
         loss-res (compute-joint-loss exec params batch cfg unified-rel?)
         ^floats probs (:probs loss-res)
         ^floats h-final (:H_final loss-res)
         ^ints targets (:targets batch)
         triples (:triples batch)
         b (long (or (:batch-size batch) 1))
         l (long (:max-seq-len cfg))
         v (long (:vocab-size cfg))
         d (long (:hidden-dim cfg))
         dm (long (:dim-mem cfg))
         lambda-tl (double (or (:lambda-tl cfg) 0.3))
         tau (double (or (:tau cfg) 0.2))
         eta (double lr)
         valid-pos (double (* b (dec l)))
         ^floats grad-w-mem (float-array (* d dm))
         ^floats grad-r-mem (float-array (* dm dm))

         ;; 1. Language Model Head Gradients: In-VRAM GPU vs CPU
         updated-embed
         (if embed-exec
           (let [dec-l (dec l)
                 n (long (* b dec-l))
                 g-arr (float-array (* n v))
                 h-flat (float-array (* n d))
                 inv-n (/ 1.0 valid-pos)
                 chunk-len (* dec-l d)]
             (dotimes [bi b]
               (System/arraycopy h-final (* bi l d) h-flat (* bi dec-l d) chunk-len)
               (dotimes [pos dec-l]
                 (let [flat-idx (+ (* bi dec-l) pos)
                       src-row (+ (* bi l v) (* pos v))
                       tgt (aget targets (+ (* bi l) pos))
                       dst-row (* flat-idx v)]
                   (dotimes [vi v]
                     (let [p (double (aget probs (+ src-row vi)))
                           diff (if (= vi tgt) (- p 1.0) p)]
                       (aset-float g-arr (+ dst-row vi) (float (* diff inv-n))))))))
             (let [update-res (sym/run-query! embed-exec {:G g-arr :H h-flat :W (:W_embed params)})]
               (:W_new update-res)))
           (let [^floats grad-embed (float-array (* v d))]
             (dotimes [bi b]
               (dotimes [pos (dec l)]
                 (let [row-idx (+ (* bi l v) (* pos v))
                       target-tok (aget targets (+ (* bi l) pos))
                       h-off (+ (* bi l d) (* pos d))]
                   (dotimes [vi v]
                     (let [p (double (aget probs (+ row-idx vi)))
                           is-target? (= vi target-tok)]
                       (when (or is-target? (> p 1e-4))
                         (let [target-val (if is-target? 1.0 0.0)
                               g-logit (/ (- p target-val) valid-pos)
                               v-off (* vi d)]
                           (dotimes [di d]
                             (let [h-val (double (aget h-final (+ h-off di)))
                                   idx (+ v-off di)]
                               (aset-float grad-embed idx
                                           (float (+ (double (aget grad-embed idx)) (* g-logit h-val)))))))))))))
             (let [orig ^floats (:W_embed params)
                   len (alength orig)
                   upd (float-array len)]
               (dotimes [i len]
                 (aset-float upd i (float (- (double (aget orig i)) (* eta (double (aget grad-embed i)))))))
               upd)))

         ;; 2. Relational Contrastive Subspace Step: 100% In-VRAM vs Legacy Backward vs CPU
         [updated-w-mem updated-r-mem rel-loss infonce-probs]
         (cond
           (and unified-rel? (seq triples) (:W_mem params) (:R_mem params))
           (let [k-cnt (count triples)
                 pos-cnt (min k-cnt (long (or (:pos-count batch) k-cnt)))
                 ih (int-array k-cnt)
                 it (int-array k-cnt)
                 _ (dotimes [i k-cnt]
                     (let [t (nth triples i)]
                       (aset-int ih i (int (:head t)))
                       (aset-int it i (int (:tail t)))))
                 target (float-array (* k-cnt k-cnt))
                 _ (dotimes [i pos-cnt]
                     (aset-float target (+ (* i k-cnt) i) (float 1.0)))
                 mask-scale (float-array (* k-cnt k-cnt))
                 scale-val (/ 1.0 (* (double (max 1 pos-cnt)) tau))
                 _ (dotimes [i pos-cnt]
                     (dotimes [j k-cnt]
                       (aset-float mask-scale (+ (* i k-cnt) j) (float scale-val))))
                 step-res (contrast/run-in-vram-contrastive-step!
                           rel-bwd-exec (:W_embed params) ih it (:W_mem params) (:R_mem params) target mask-scale)
                 ^floats p (:P step-res)
                 inf-loss (if (pos? pos-cnt)
                            (let [loss-sum (loop [i 0 s 0.0]
                                             (if (>= i pos-cnt) s
                                                 (let [pos-p (double (aget p (+ (* i k-cnt) i)))]
                                                   (recur (inc i) (+ s (- (Math/log (Math/max 1e-12 pos-p))))))))]
                              (/ loss-sum (double pos-cnt)))
                            0.0)]
             [(:W_new step-res) (:R_new step-res) inf-loss p])

           (and (seq triples) (:W_mem params) (:R_mem params) (:infonce-probs loss-res))
           (let [k-cnt (count triples)
                 ^floats r-mem (:R_mem params)
                 ^floats w-embed (:W_embed params)
                 ^floats u-h (:u-h loss-res)
                 ^floats u-t (:u-t loss-res)
                 ^floats u-hr (:u-hr loss-res)
                 ^floats infonce-p (:infonce-probs loss-res)
                 pos-cnt (min k-cnt (long (or (:pos-count batch) k-cnt)))
                 inv-k-tau (/ 1.0 (* (double (max 1 pos-cnt)) tau))]
             (if rel-bwd-exec
               (let [g-s-arr (float-array (* k-cnt k-cnt))
                     vh (float-array (* k-cnt d))
                     vt (float-array (* k-cnt d))]
                 (dotimes [k-idx k-cnt]
                   (let [{:keys [head tail]} (nth triples k-idx)]
                     (System/arraycopy w-embed (* (long head) d) vh (* k-idx d) d)
                     (System/arraycopy w-embed (* (long tail) d) vt (* k-idx d) d)))
                 (dotimes [i pos-cnt]
                   (let [row-off (* i k-cnt)]
                     (dotimes [j k-cnt]
                       (let [target (if (= i j) 1.0 0.0)
                             prob (double (aget infonce-p (+ row-off j)))
                             g-s (* (- prob target) inv-k-tau)]
                         (aset-float g-s-arr (+ row-off j) (float g-s))))))
                 (let [bwd-res (sym/run-query! rel-bwd-exec
                                               {:G_S g-s-arr
                                                :U_hr u-hr
                                                :U_t u-t
                                                :U_h u-h
                                                :V_h vh
                                                :V_t vt
                                                :R r-mem})
                       ^floats dw (:dW bwd-res)
                       ^floats dr (:dR bwd-res)
                       dw-len (alength dw)
                       dr-len (alength dr)]
                   (dotimes [i dw-len]
                     (aset-float grad-w-mem i (float (* lambda-tl (double (aget dw i))))))
                   (dotimes [i dr-len]
                     (aset-float grad-r-mem i (float (* lambda-tl (double (aget dr i))))))))
               ;; CPU Fallback Loop
               (let [delta-u-hr (float-array (* k-cnt dm))
                     delta-u-t (float-array (* k-cnt dm))]
                 (dotimes [i pos-cnt]
                   (let [uk-i (* i dm)
                         row-off (* i k-cnt)]
                     (dotimes [j k-cnt]
                       (let [uk-j (* j dm)
                             target (if (= i j) 1.0 0.0)
                             prob (double (aget infonce-p (+ row-off j)))
                             g-s (* (- prob target) inv-k-tau)]
                         (dotimes [m1 dm]
                           (let [v-uh (double (aget u-h (+ uk-i m1)))]
                             (dotimes [m2 dm]
                               (let [idx (+ (* m1 dm) m2)
                                     v-ut (double (aget u-t (+ uk-j m2)))]
                                 (aset-float grad-r-mem idx (float (+ (double (aget grad-r-mem idx))
                                                                      (* lambda-tl g-s v-uh v-ut))))))))
                         (dotimes [m dm]
                           (let [v-ut (double (aget u-t (+ uk-j m)))
                                 v-uhr (double (aget u-hr (+ uk-i m)))]
                             (aset-float delta-u-hr (+ uk-i m) (float (+ (double (aget delta-u-hr (+ uk-i m)))
                                                                         (* g-s v-ut))))
                             (aset-float delta-u-t (+ uk-j m) (float (+ (double (aget delta-u-t (+ uk-j m)))
                                                                        (* g-s v-uhr))))))))))
                 (dotimes [k-idx k-cnt]
                   (let [{:keys [head tail]} (nth triples k-idx)
                         h-off (* (long head) d)
                         t-off (* (long tail) d)
                         uk-off (* k-idx dm)
                         delta-u-h (float-array dm)]
                     (dotimes [m1 dm]
                       (let [s (loop [m2 0 acc 0.0]
                                 (if (>= m2 dm) acc
                                     (recur (inc m2) (+ acc (* (double (aget delta-u-hr (+ uk-off m2)))
                                                               (double (aget r-mem (+ (* m1 dm) m2))))))))]
                         (aset-float delta-u-h m1 (float s))))
                     (dotimes [di d]
                       (let [vh (double (aget w-embed (+ h-off di)))
                             vt (double (aget w-embed (+ t-off di)))]
                         (dotimes [dj dm]
                           (let [idx (+ (* di dm) dj)
                                 duh (double (aget delta-u-h dj))
                                 dut (double (aget delta-u-t (+ uk-off dj)))]
                             (aset-float grad-w-mem idx (float (+ (double (aget grad-w-mem idx))
                                                                  (* lambda-tl (+ (* vh duh) (* vt dut))))))))))))))
             (let [update-array (fn [^floats orig ^floats grad]
                                  (let [len (alength orig)
                                        updated (float-array len)]
                                    (dotimes [i len]
                                      (aset-float updated i (float (- (double (aget orig i)) (* eta (double (aget grad i)))))))
                                    updated))]
               [(update-array (:W_mem params) grad-w-mem)
                (update-array (:R_mem params) grad-r-mem)
                (or (:infonce-loss loss-res) 0.0)
                infonce-p]))

           :else
           [(:W_mem params) (:R_mem params) (or (:infonce-loss loss-res) 0.0) (:infonce-probs loss-res)])

         final-loss (assoc loss-res
                           :infonce-loss rel-loss
                           :infonce-probs infonce-probs
                           :total-loss (+ (:lm-loss loss-res) (* lambda-tl rel-loss)))]
     (assoc params
            :W_embed updated-embed
            :W_mem updated-w-mem
            :R_mem updated-r-mem
            :loss final-loss))))

(defn estimate-tl-nano-vram
  "Calculates theoretical VRAM footprint and parameter count for TL-Nano configurations."
  [cfg]
  (let [v (long (:vocab-size cfg))
        d (long (:hidden-dim cfg))
        dff (long (:intermediate-dim cfg))
        dm (long (:dim-mem cfg))
        num-layers (long (:num-layers cfg))
        hybrid-count (long (count (:hybrid-layers cfg)))

        embed-params (* v d)
        attn-params-per-layer (* 4 d d)
        mlp-params-per-layer (+ (* 2 d dff) (* dff d))
        hybrid-params-per-layer (+ (* d dm) (* dm dm))

        total-layer-params (+ (* num-layers (+ attn-params-per-layer mlp-params-per-layer))
                              (* hybrid-count hybrid-params-per-layer))
        total-params (+ embed-params total-layer-params)
        fp16-bytes (* total-params 2)
        optimizer-bytes (* total-params 12) ;; FP32 weights, momentum, variance (AdamW)
        total-vram-bytes (+ fp16-bytes optimizer-bytes)]
    {:total-params total-params
     :embed-params embed-params
     :layer-params total-layer-params
     :weights-fp16-mb (/ (double fp16-bytes) 1048576.0)
     :train-vram-mb (/ (double total-vram-bytes) 1048576.0)
     :fits-in-24gb? (< (/ (double total-vram-bytes) 1073741824.0) 24.0)}))
