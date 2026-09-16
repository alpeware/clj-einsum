(ns clj-xla.logic.memory.camp
  "Cross-Attention Memory Probe (CAMP) for Declarative Tensor Logic & In-Tensor Relational Memory.
   Decouples relational entity semantics from syntactic prompt template tokens by learning a
   query/key direction that dynamically focuses attention on head entity spans.
   Rule 2: Pure functions (Sans-IO) with exact symbolic adjoint gradients.")

;; ==============================================================================
;; 1. Pure Linear Algebra & Attention Contractions
;; ==============================================================================

(defn normalize-sequence-h
  "Normalizes each token embedding row in H to unit L2 Euclidean norm.
   H: flat float-array of length (* L hidden-dim). Returns new normalized float-array."
  [^floats H L hidden-dim]
  (let [l-cnt (long L)
        h-dim (long hidden-dim)
        out (float-array (* l-cnt h-dim))]
    (dotimes [t l-cnt]
      (let [offset (* t h-dim)
            sum-sq (loop [i 0 s 0.0]
                     (if (>= i h-dim)
                       s
                       (let [val (double (aget H (+ offset i)))]
                         (recur (inc i) (+ s (* val val))))))
            norm (Math/sqrt sum-sq)
            scale (if (> norm 1e-12) (/ 1.0 norm) 0.0)]
        (dotimes [i h-dim]
          (aset-float out (+ offset i) (float (* (double (aget H (+ offset i))) scale))))))
    out))

(defn compute-attention-weights
  "Computes softmax attention weights over a sequence H of length L:
     z_t = (H_t . k_attn) / tau
     alpha = softmax(z)
   H: flat float-array of length (* L hidden-dim)
   k-attn: float-array of length hidden-dim
   tau: temperature scalar (double)
   mask: optional sequence of booleans of length L (false masks out token position)."
  ([^floats H L hidden-dim ^floats k-attn tau]
   (compute-attention-weights H L hidden-dim k-attn tau nil))
  ([^floats H L hidden-dim ^floats k-attn tau mask]
   (let [l-cnt (long L)
         h-dim (long hidden-dim)
         t-scale (double (if (and (number? tau) (pos? (double tau))) tau 1.0))
         has-mask? (boolean (and mask (= (count mask) l-cnt)))
         z (float-array l-cnt)]
     ;; 1. Inner products z_t = (H_t . k_attn) / tau
     (dotimes [t l-cnt]
       (if (and has-mask? (false? (nth mask t)))
         (aset-float z t (float -10000.0))
         (let [offset (* t h-dim)
               dot (loop [i 0 s 0.0]
                     (if (>= i h-dim)
                       s
                       (recur (inc i) (+ s (* (double (aget H (+ offset i)))
                                              (double (aget k-attn i)))))))]
           (aset-float z t (float (/ dot t-scale))))))

     ;; 2. Softmax over z
     (let [max-z (loop [t 1 m (double (aget z 0))]
                   (if (>= t l-cnt)
                     m
                     (recur (inc t) (Math/max m (double (aget z t))))))
           exp-sum (loop [t 0 es 0.0]
                     (if (>= t l-cnt)
                       es
                       (let [val (double (aget z t))]
                         (recur (inc t) (if (<= val -9999.0)
                                          es
                                          (+ es (Math/exp (- val max-z))))))))
           denom (if (pos? exp-sum) exp-sum 1.0)
           alpha (float-array l-cnt)]
       (dotimes [t l-cnt]
         (let [val (double (aget z t))]
           (if (<= val -9999.0)
             (aset-float alpha t (float 0.0))
             (let [prob (/ (Math/exp (- val max-z)) denom)]
               (aset-float alpha t (float prob))))))
       alpha))))

(defn forward-camp-single
  "Computes forward pass of Cross-Attention Memory Probe on a single sequence H:
     1. alpha = softmax((H . k_attn) / tau)       [L]
     2. h = sum_t alpha_t H_t                    [D_model]
     3. u = h @ W                                [D_mem]
     4. v = u @ R                                [D_mem]
     5. s = v @ E^T                              [N]
     6. p = softmax(s)                           [N]
   Returns map containing intermediate activations for backward adjoint propagation."
  ([^floats H L hidden-dim ^floats k-attn ^floats W ^floats R ^floats E d-mem n-entities tau]
   (forward-camp-single H L hidden-dim k-attn W R E d-mem n-entities tau nil))
  ([^floats H L hidden-dim ^floats k-attn ^floats W ^floats R ^floats E d-mem n-entities tau mask]
   (let [l-cnt (long L)
         h-dim (long hidden-dim)
         d-long (long d-mem)
         n-long (long n-entities)
         alpha (compute-attention-weights H l-cnt h-dim k-attn tau mask)

         ;; 2. h = sum_t alpha_t H_t
         h (float-array h-dim)]
     (dotimes [i h-dim]
       (let [sum (loop [t 0 s 0.0]
                   (if (>= t l-cnt)
                     s
                     (recur (inc t) (+ s (* (double (aget ^floats alpha t))
                                            (double (aget H (+ (* t h-dim) i))))))))]
         (aset-float h i (float sum))))

    ;; 3. u = h @ W
     (let [u (float-array d-long)]
       (dotimes [j d-long]
         (let [sum (loop [i 0 s 0.0]
                     (if (>= i h-dim)
                       s
                       (recur (inc i) (+ s (* (double (aget h i))
                                              (double (aget W (+ (* i d-long) j))))))))]
           (aset-float u j (float sum))))

      ;; 4. v = u @ R
       (let [v (float-array d-long)]
         (dotimes [k d-long]
           (let [sum (loop [j 0 s 0.0]
                       (if (>= j d-long)
                         s
                         (recur (inc j) (+ s (* (double (aget u j))
                                                (double (aget R (+ (* j d-long) k))))))))]
             (aset-float v k (float sum))))

        ;; 5. s = v @ E^T
         (let [s (float-array n-long)]
           (dotimes [n n-long]
             (let [sum (loop [k 0 dot-s 0.0]
                         (if (>= k d-long)
                           dot-s
                           (recur (inc k) (+ dot-s (* (double (aget v k))
                                                      (double (aget E (+ (* n d-long) k))))))))]
               (aset-float s n (float sum))))

          ;; 6. p = softmax(s)
           (let [max-s (loop [n 1 m (double (aget s 0))]
                         (if (>= n n-long)
                           m
                           (recur (inc n) (Math/max m (double (aget s n))))))
                 exp-sum (loop [n 0 es 0.0]
                           (if (>= n n-long)
                             es
                             (recur (inc n) (+ es (Math/exp (- (double (aget s n)) max-s))))))
                 p (float-array n-long)]
             (dotimes [n n-long]
               (let [prob (/ (Math/exp (- (double (aget s n)) max-s)) exp-sum)]
                 (aset-float p n (float prob))))

             {:alpha alpha
              :h h
              :u u
              :v v
              :s s
              :p p})))))))

;; ==============================================================================
;; 2. Loss & Analytic Adjoint Gradients
;; ==============================================================================

(defn compute-loss-and-gradients
  "Computes mean cross-entropy loss and exact analytic backward adjoint gradients
   w.r.t k_attn (length hidden-dim) and W (shape [hidden-dim, d-mem]).
   dataset: vector of {:H ^floats :seq-len L :y entity-target-idx}
   params: {:k-attn ^floats :W ^floats}
   cfg: {:hidden-dim H :d-mem D :n-entities N :tau tau :R ^floats :E ^floats}"
  [dataset params cfg]
  (let [{:keys [hidden-dim d-mem n-entities tau R E]} cfg
        {:keys [k-attn W]} params
        h-dim (long hidden-dim)
        d-long (long d-mem)
        n-long (long n-entities)
        t-scale (double (if (and (number? tau) (pos? (double tau))) tau 1.0))
        m-cnt (long (count dataset))
        inv-m (/ 1.0 (double m-cnt))

        grad-k (float-array h-dim)
        grad-W (float-array (* h-dim d-long))
        loss-acc (double-array 1)
        acc-hits (long-array 1)]

    (doseq [{:keys [H seq-len y mask]} dataset]
      (let [y-long (long y)
            fwd (forward-camp-single H seq-len h-dim k-attn W R E d-long n-long t-scale mask)
            ^floats alpha (:alpha fwd)
            ^floats h (:h fwd)
            ^floats p (:p fwd)
            py (Math/max 1e-12 (double (aget p y-long)))
            nll (- (Math/log py))
            _ (aset loss-acc 0 (+ (aget loss-acc 0) nll))

            ;; Check hit
            pred-y (loop [n 1 best-n 0 best-sc (double (aget p 0))]
                     (if (>= n n-long)
                       best-n
                       (let [sc (double (aget p n))]
                         (if (> sc best-sc)
                           (recur (inc n) n sc)
                           (recur (inc n) best-n best-sc)))))
            _ (when (= pred-y y-long)
                (aset acc-hits 0 (inc (aget acc-hits 0))))

            ;; 1. g_s = p - 1_y  [N]
            g-s (float-array n-long)]
        (dotimes [n n-long]
          (let [target (if (= n y-long) 1.0 0.0)]
            (aset-float g-s n (float (- (double (aget p n)) target)))))

        ;; 2. g_v = g_s @ E  [D_mem]
        (let [g-v (float-array d-long)]
          (dotimes [k d-long]
            (let [sum (loop [n 0 s 0.0]
                        (if (>= n n-long)
                          s
                          (recur (inc n) (+ s (* (double (aget g-s n))
                                                 (double (aget ^floats E (+ (* n d-long) k))))))))]
              (aset-float g-v k (float sum))))

          ;; 3. g_u = g_v @ R^T  [D_mem]
          (let [g-u (float-array d-long)]
            (dotimes [j d-long]
              (let [sum (loop [k 0 s 0.0]
                          (if (>= k d-long)
                            s
                            (recur (inc k) (+ s (* (double (aget g-v k))
                                                   (double (aget ^floats R (+ (* j d-long) k))))))))]
                (aset-float g-u j (float sum))))

            ;; 4. Accumulate grad_W += inv-m * (h^T @ g_u)
            (dotimes [i h-dim]
              (let [h-val (double (aget h i))
                    row-offset (* i d-long)]
                (dotimes [j d-long]
                  (let [gw-ij (* h-val (double (aget g-u j)))
                        idx (+ row-offset j)]
                    (aset-float grad-W idx (float (+ (double (aget grad-W idx)) (* inv-m gw-ij))))))))

            ;; 5. g_h = g_u @ W^T  [D_model]
            (let [g-h (float-array h-dim)]
              (dotimes [i h-dim]
                (let [sum (loop [j 0 s 0.0]
                            (if (>= j d-long)
                              s
                              (recur (inc j) (+ s (* (double (aget g-u j))
                                                     (double (aget ^floats W (+ (* i d-long) j))))))))]
                  (aset-float g-h i (float sum))))

              ;; 6. Backprop through attention pooling: g_alpha_t = g_h . H_t
              (let [l-cnt (long seq-len)
                    g-alpha (float-array l-cnt)]
                (dotimes [t l-cnt]
                  (let [offset (* t h-dim)
                        dot (loop [i 0 s 0.0]
                              (if (>= i h-dim)
                                s
                                (recur (inc i) (+ s (* (double (aget g-h i))
                                                       (double (aget ^floats H (+ offset i))))))))]
                    (aset-float g-alpha t (float dot))))

                ;; 7. Backprop through softmax: g_z_t = alpha_t * (g_alpha_t - sum_j alpha_j g_alpha_j)
                (let [c (loop [t 0 s 0.0]
                          (if (>= t l-cnt)
                            s
                            (recur (inc t) (+ s (* (double (aget alpha t))
                                                   (double (aget g-alpha t)))))))
                      g-z (float-array l-cnt)]
                  (dotimes [t l-cnt]
                    (let [diff (- (double (aget g-alpha t)) c)]
                      (aset-float g-z t (float (* (double (aget alpha t)) diff)))))

                  ;; 8. Backprop through scaled dot-product: grad_k += inv-m * (1/tau) * sum_t g_z_t H_t
                  (let [scale (* inv-m (/ 1.0 t-scale))]
                    (dotimes [t l-cnt]
                      (let [gz-t (double (aget g-z t))
                            offset (* t h-dim)]
                        (dotimes [i h-dim]
                          (let [val (* gz-t (double (aget ^floats H (+ offset i))))]
                            (aset-float grad-k i (float (+ (double (aget grad-k i)) (* scale val))))))))))))))))

    {:loss (/ (aget loss-acc 0) (double m-cnt))
     :accuracy (/ (double (aget acc-hits 0)) (double m-cnt))
     :grad-k grad-k
     :grad-W grad-W}))

;; ==============================================================================
;; 3. Parameter Update & Optimization Loop
;; ==============================================================================

(defn step-gd
  "Applies pure gradient descent update with learning rate lr:
     param_new = param - lr * grad"
  [^floats param ^floats grad lr]
  (let [len (alength param)
        lr-d (double lr)
        out (float-array len)]
    (dotimes [i len]
      (aset-float out i (float (- (double (aget param i)) (* lr-d (double (aget grad i)))))))
    out))

(defn train-camp
  "Trains Cross-Attention Memory Probe parameters {:k-attn :W} on dataset.
   opts: {:lr float :steps int :on-step (fn [step loss acc])}"
  [dataset init-params cfg opts]
  (let [lr (or (:lr opts) 0.01)
        steps (long (or (:steps opts) 100))
        on-step (:on-step opts)
        initial-eval (compute-loss-and-gradients dataset init-params cfg)
        init-loss (:loss initial-eval)]
    (when on-step (on-step 0 init-loss (:accuracy initial-eval)))
    (loop [s 0
           params init-params
           last-eval initial-eval]
      (if (>= s steps)
        {:k-attn (:k-attn params)
         :W (:W params)
         :initial-loss init-loss
         :final-loss (:loss last-eval)
         :final-accuracy (:accuracy last-eval)}
        (let [grad-k (:grad-k last-eval)
              grad-W (:grad-W last-eval)
              new-k (step-gd (:k-attn params) grad-k lr)
              new-W (step-gd (:W params) grad-W lr)
              new-params {:k-attn new-k :W new-W}
              next-eval (compute-loss-and-gradients dataset new-params cfg)]
          (when on-step (on-step (inc s) (:loss next-eval) (:accuracy next-eval)))
          (recur (inc s) new-params next-eval))))))

;; ==============================================================================
;; 4. Leave-One-Out Cross-Validation Harness
;; ==============================================================================

(defn run-camp-leave-one-out-cv
  "Performs rigorous N-fold Leave-One-Out Cross-Validation for CAMP.
   In each fold, trains on N-1 examples and evaluates on the held-out query."
  [dataset init-params cfg train-opts]
  (let [n (count dataset)
        fold-results
        (mapv
         (fn [test-idx]
           (let [train-data (vec (concat (subvec dataset 0 test-idx)
                                         (subvec dataset (inc test-idx))))
                 test-sample (nth dataset test-idx)
                 trained (train-camp train-data init-params cfg train-opts)
                 eval-res (compute-loss-and-gradients [test-sample]
                                                      {:k-attn (:k-attn trained) :W (:W trained)}
                                                      cfg)
                 ;; Check test attention distribution
                 test-fwd (forward-camp-single (:H test-sample)
                                               (:seq-len test-sample)
                                               (:hidden-dim cfg)
                                               (:k-attn trained)
                                               (:W trained)
                                               (:R cfg)
                                               (:E cfg)
                                               (:d-mem cfg)
                                               (:n-entities cfg)
                                               (:tau cfg)
                                               (:mask test-sample))
                 pred-y (loop [i 1 best-i 0 best-sc (double (aget ^floats (:p test-fwd) 0))]
                          (if (>= i (long (:n-entities cfg)))
                            best-i
                            (let [sc (double (aget ^floats (:p test-fwd) i))]
                              (if (> sc best-sc)
                                (recur (inc i) i sc)
                                (recur (inc i) best-i best-sc)))))
                 hit? (= pred-y (long (:y test-sample)))]
             {:test-idx test-idx
              :target (:y test-sample)
              :predicted pred-y
              :hit? hit?
              :test-loss (:loss eval-res)
              :test-alpha (:alpha test-fwd)}))
         (range n))
        hits (count (filter :hit? fold-results))
        mean-acc (/ (double hits) (double n))]
    {:mean-acc mean-acc
     :hits hits
     :total n
     :fold-results fold-results}))

;; ==============================================================================
;; 5. Pedro Domingos Declarative Tensor Logic AST Constructor
;; ==============================================================================

(defn camp-grounding-ast
  "Constructs Declarative Tensor Logic AST for Cross-Attention Memory Probe.
   Binds scaled dot-product attention over prompt sequence H [:b :p :dim] with query k_attn [:dim],
   pools hidden states along sequence dimension :p into h_probe [:b :dim],
   and unbinds relational memory core."
  [hidden-dim _dim _n-entities opts]
  (let [seq? (boolean (:sequence? opts))
        h-dim (long hidden-dim)
        scale (/ 1.0 (Math/sqrt (double h-dim)))]
    [:block {:name (if seq? :camp_grounding_seq :camp_grounding)}
     ;; 1. Scaled dot-product query against hidden sequence
     [:= [:attn_logits :b :p] {:scale scale}
      [:normed :b :p :dim] [:k_attn :dim]]
     [:softmax [:attn_weights :b :p] [:attn_logits :b :p]]

     ;; 2. Attention pooling across sequence length :p
     [:= [:h_probe :b :dim]
      [:attn_weights :b :p] [:normed :b :p :dim]]

     ;; 3. Project to memory space
     [:= [:v_q :b :d]
      [:h_probe :b :dim] [:w_mem_proj :dim :d]]

     ;; 4. Contraction with active relation core
     [:= [:v_target :b :d]
      [:v_q :b :k] [:r_active :k :d]]
     [:rms-norm [:v_target_norm :b :d] [:v_target :b :d]]

     ;; 5. Entity scoring against knowledge base table
     [:= [:entity_scores :b :n_entities]
      [:v_target_norm :b :d] [:entity_table :n_entities :d]]]))
