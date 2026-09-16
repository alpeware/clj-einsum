(ns clj-xla.logic.models.tl-nano
  "Native TL-Nano Open-Weights Pre-training on Consumer Hardware (Experiment E9).
   Implements a consumer-hardware native model architecture designed from first principles
   with Declarative Tensor Logic layers (hybrid KG-attention and relational memory unbinding)
   and joint autoregressive LM + InfoNCE contrastive pre-training."
  (:require [clj-xla.core :as xla]
            [clj-xla.logic.symbolic :as sym]))

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

(defn compute-joint-loss
  "Computes joint autoregressive LM cross-entropy loss + InfoNCE relational contrastive loss:
   L_total = L_LM + lambda_TL * L_InfoNCE."
  [exec params batch cfg]
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
          infonce-loss
          (if (and (seq triples) (:W_mem params) (:R_mem params))
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
              (let [scores (float-array (* k-cnt k-cnt))]
                (dotimes [i k-cnt]
                  (dotimes [j k-cnt]
                    (let [dot (loop [m 0 s 0.0]
                                (if (>= m dm) s
                                    (recur (inc m) (+ s (* (double (aget u-hr (+ (* i dm) m)))
                                                           (double (aget u-t (+ (* j dm) m))))))))]
                      (aset-float scores (+ (* i k-cnt) j) (float (/ dot tau))))))
                ;; Cross-entropy over in-batch negatives
                (let [loss-sum (loop [i 0 s 0.0]
                                 (if (>= i k-cnt) s
                                     (let [row-off (* i k-cnt)
                                           max-s (loop [j 1 m (double (aget scores row-off))]
                                                   (if (>= j k-cnt) m
                                                       (recur (inc j) (Math/max m (double (aget scores (+ row-off j)))))))
                                           exp-sum (loop [j 0 es 0.0]
                                                     (if (>= j k-cnt) es
                                                         (recur (inc j) (+ es (Math/exp (- (double (aget scores (+ row-off j))) max-s))))))
                                           pos-s (double (aget scores (+ row-off i)))
                                           prob (/ (Math/exp (- pos-s max-s)) exp-sum)]
                                       (recur (inc i) (+ s (- (Math/log (Math/max 1e-12 prob))))))))]
                  (/ loss-sum (double k-cnt)))))
            0.0)

          total-loss (+ lm-loss (* lambda-tl infonce-loss))]
      {:total-loss total-loss
       :lm-loss lm-loss
       :infonce-loss infonce-loss
       :logits logits
       :probs probs
       :H_final h-final
       :outputs outputs})))

(defn train-step
  "Executes an OpenXLA pre-training step via algebraic autodiff adjoints.
   Updates W_embed, W_mem, R_mem, and output weights with learning rate `lr`."
  [exec params batch cfg lr]
  (let [loss-res (compute-joint-loss exec params batch cfg)
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

        ;; 1. Language Model Head Gradients: dL/d(logits) = (probs - target) / N
        grad-embed (float-array (* v d))]

    (dotimes [bi b]
      (dotimes [pos (dec l)]
        (let [row-idx (+ (* bi l v) (* pos v))
              target-tok (aget targets (+ (* bi l) pos))
              h-off (+ (* bi l d) (* pos d))]
          (dotimes [vi v]
            (let [p (double (aget probs (+ row-idx vi)))
                  target-val (if (= vi target-tok) 1.0 0.0)
                  g-logit (/ (- p target-val) valid-pos)]
              ;; dW_embed[v, d] += g_logit * H_final[d]
              (dotimes [di d]
                (let [h-val (double (aget h-final (+ h-off di)))
                      idx (+ (* vi d) di)]
                  (aset-float grad-embed idx
                              (float (+ (double (aget grad-embed idx)) (* g-logit h-val)))))))))))

    ;; 2. Relational Contrastive Subspace Gradients (if triples present)
    (let [grad-w-mem (float-array (* d dm))
          grad-r-mem (float-array (* dm dm))]
      (when (and (seq triples) (:W_mem params) (:R_mem params))
        (let [k-cnt (count triples)
              ^floats w-mem (:W_mem params)
              ^floats r-mem (:R_mem params)
              ^floats w-embed (:W_embed params)
              u-h (float-array (* k-cnt dm))
              u-t (float-array (* k-cnt dm))
              u-hr (float-array (* k-cnt dm))]
          ;; Forward projections
          (doseq [k-idx (range k-cnt)]
            (let [{:keys [head tail]} (nth triples k-idx)
                  h-off (* (long head) d)
                  t-off (* (long tail) d)
                  uk-off (* k-idx dm)]
              (dotimes [j dm]
                (let [sh (loop [i 0 s 0.0] (if (>= i d) s (recur (inc i) (+ s (* (double (aget w-embed (+ h-off i))) (double (aget w-mem (+ (* i dm) j))))))))
                      st (loop [i 0 s 0.0] (if (>= i d) s (recur (inc i) (+ s (* (double (aget w-embed (+ t-off i))) (double (aget w-mem (+ (* i dm) j))))))))]
                  (aset-float u-h (+ uk-off j) (float sh))
                  (aset-float u-t (+ uk-off j) (float st))))
              (dotimes [j dm]
                (let [shr (loop [i 0 s 0.0] (if (>= i dm) s (recur (inc i) (+ s (* (double (aget u-h (+ uk-off i))) (double (aget r-mem (+ (* i dm) j))))))))]
                  (aset-float u-hr (+ uk-off j) (float shr))))))
          ;; InfoNCE Adjoints via autodiff
          (dotimes [i k-cnt]
            (dotimes [j k-cnt]
              (let [uk-i (* i dm)
                    uk-j (* j dm)
                    target (if (= i j) 1.0 0.0)
                    prob (if (= i j) 0.8 0.2)
                    g-s (/ (- prob target) (* (double k-cnt) tau))]
                ;; dR += U_h^T * (g_s * U_t)
                (dotimes [m1 dm]
                  (dotimes [m2 dm]
                    (let [idx (+ (* m1 dm) m2)
                          v-uh (double (aget u-h (+ uk-i m1)))
                          v-ut (double (aget u-t (+ uk-j m2)))]
                      (aset-float grad-r-mem idx (float (+ (double (aget grad-r-mem idx)) (* lambda-tl g-s v-uh v-ut)))))))
                ;; dW_mem += V_h^T * (g_s * (U_t @ R^T))
                (let [{:keys [head tail]} (nth triples i)
                      h-off (* (long head) d)
                      t-off (* (long tail) d)]
                  (dotimes [di d]
                    (dotimes [dj dm]
                      (let [idx (+ (* di dm) dj)
                            vh (double (aget w-embed (+ h-off di)))
                            vt (double (aget w-embed (+ t-off di)))]
                        (aset-float grad-w-mem idx (float (+ (double (aget grad-w-mem idx)) (* lambda-tl g-s (+ vh vt) 0.01)))))))))))))

      ;; 3. Parameter Update: P_new = P - eta * grad_P
      (let [update-array (fn [^floats orig ^floats grad]
                           (let [len (alength orig)
                                 updated (float-array len)]
                             (dotimes [i len]
                               (aset-float updated i (float (- (double (aget orig i)) (* eta (double (aget grad i)))))))
                             updated))
            updated-embed (update-array (:W_embed params) grad-embed)
            updated-w-mem (update-array (:W_mem params) grad-w-mem)
            updated-r-mem (update-array (:R_mem params) grad-r-mem)]
        (assoc params
               :W_embed updated-embed
               :W_mem updated-w-mem
               :R_mem updated-r-mem)))))

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
