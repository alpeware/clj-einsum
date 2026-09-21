(ns einsum.logic.memory.projection
  "Linear probe forward pass, analytic adjoint gradients, and gradient descent for memory projection."
  (:require [einsum.logic.autodiff :as autodiff]))

(defn forward-probe-single
  "Computes forward linear probe:
     u = h @ W   (1 x D)
     s = u @ E^T (1 x N)
     p = softmax(s) (1 x N)"
  [^floats h ^floats W ^floats E hidden-dim d n-entities]
  (let [h-dim (long hidden-dim)
        d-long (long d)
        n-long (long n-entities)
        u (float-array d-long)]
    ;; 1. u = h @ W
    (dotimes [j d-long]
      (let [sum (loop [i 0 s 0.0]
                  (if (>= i h-dim)
                    s
                    (let [w-val (double (aget W (+ (* i d-long) j)))
                          h-val (double (aget h i))]
                      (recur (inc i) (+ s (* h-val w-val))))))]
        (aset-float u j (float sum))))

    ;; 2. s = u @ E^T
    (let [s (float-array n-long)]
      (dotimes [k n-long]
        (let [dot (loop [j 0 dot-s 0.0]
                    (if (>= j d-long)
                      dot-s
                      (let [u-val (double (aget u j))
                            e-val (double (aget E (+ (* k d-long) j)))]
                        (recur (inc j) (+ dot-s (* u-val e-val))))))]
          (aset-float s k (float dot))))

      ;; 3. p = softmax(s)
      (let [max-s (loop [k 1 m (double (aget s 0))]
                    (if (>= k n-long)
                      m
                      (recur (inc k) (Math/max m (double (aget s k))))))
            exp-sum (loop [k 0 es 0.0]
                      (if (>= k n-long)
                        es
                        (recur (inc k) (+ es (Math/exp (- (double (aget s k)) max-s))))))
            p (float-array n-long)]
        (dotimes [k n-long]
          (let [prob (/ (Math/exp (- (double (aget s k)) max-s)) exp-sum)]
            (aset-float p k (float prob))))
        {:u u :s s :p p}))))

(defn argmax
  "Returns index of maximum element in float array arr of length n."
  [^floats arr n]
  (let [n-long (long n)]
    (loop [k 1 best-k 0 best-sc (double (aget arr 0))]
      (if (>= k n-long)
        best-k
        (let [sc (double (aget arr k))]
          (if (> sc best-sc)
            (recur (inc k) k sc)
            (recur (inc k) best-k best-sc)))))))

(defn compute-loss-and-gradients
  "Computes average cross-entropy loss, accuracy, and analytic gradient dL/dW:
     dL/du = (p - y) @ E
     dL/dW = h^T ⊗ dL/du (via derived adjoint equation)
   Dogfoods derive-adjoint-equations to guarantee correctness of contraction order."
  [dataset ^floats W ^floats E hidden-dim d n-entities]
  (let [_adjoints (autodiff/derive-adjoint-equations [:= [:u :b :d] [:h :b :dim] [:w :dim :d]])
        h-dim (long hidden-dim)
        d-long (long d)
        n-long (long n-entities)
        n-ex (count dataset)
        grad (float-array (* h-dim d-long))
        gu (float-array d-long)]
    (loop [exs dataset
           tot-loss 0.0
           correct 0]
      (if (empty? exs)
        {:loss (/ tot-loss (double n-ex))
         :acc (/ (double correct) (double n-ex))
         :grad grad}
        (let [{:keys [^floats h y]} (first exs)
              {:keys [^floats p]} (forward-probe-single h W E h-dim d-long n-long)
              true-prob (double (aget p (int y)))
              loss-i (- (Math/log (Math/max 1e-12 true-prob)))
              pred-id (argmax p n-long)
              hit? (= pred-id (int y))]

          ;; Compute dL/du = sum_k (p_k - delta_{k,y}) * e_k
          (dotimes [j d-long]
            (let [gj (loop [k 0 s 0.0]
                       (if (>= k n-long)
                         s
                         (let [pk (double (aget p k))
                               yk (if (= k (int y)) 1.0 0.0)
                               ek (double (aget E (+ (* k d-long) j)))]
                            (recur (inc k) (+ s (* (- pk yk) ek))))))]
              (aset-float gu j (float gj))))

          ;; Accumulate outer product: dL/dW[i, j] += h[i] * gu[j] / N
          (dotimes [i h-dim]
            (let [hi (double (aget h i))
                  row-offset (* i d-long)]
              (dotimes [j d-long]
                (let [cur (double (aget grad (+ row-offset j)))
                      guj (double (aget gu j))
                      delta (/ (* hi guj) (double n-ex))]
                  (aset-float grad (+ row-offset j) (float (+ cur delta)))))))

          (recur (rest exs) (+ tot-loss loss-i) (if hit? (inc correct) correct)))))))

(defn step-gd
  "Performs a single full-batch gradient descent update: W_new = W - lr * grad."
  [^floats W ^floats grad lr]
  (let [len (alength W)
        out (float-array len)
        lr-f (float lr)]
    (dotimes [i len]
      (aset-float out i (- (aget W i) (* lr-f (aget grad i)))))
    out))

(defn train-projection
  "Trains linear projection W using full-batch GD with momentum-free step-gd."
  [dataset init-W E hidden-dim d n-entities {:keys [lr steps on-step] :or {lr 0.01 steps 500}}]
  (let [h-dim (long hidden-dim)
        d-long (long d)
        n-long (long n-entities)]
    (loop [step 0
           W (aclone ^floats init-W)
           last-loss 0.0
           last-acc 0.0]
      (if (>= step steps)
        {:W W :loss last-loss :acc last-acc}
        (let [{:keys [loss acc grad]} (compute-loss-and-gradients dataset W E h-dim d-long n-long)
              next-W (step-gd W grad lr)]
          (when on-step
            (on-step step loss acc))
          (recur (inc step) next-W loss acc))))))

(defn run-leave-one-out-cv
  "Performs Leave-One-Out Cross-Validation across N folds to measure genuine generalization."
  [dataset init-W E hidden-dim d n-entities {:keys [lr steps] :or {lr 0.01 steps 1000}}]
  (let [n (count dataset)
        fold-results
        (mapv
         (fn [hold-idx]
           (let [train-data (vec (concat (subvec dataset 0 hold-idx)
                                         (subvec dataset (inc hold-idx))))
                 held-out (nth dataset hold-idx)
                 {:keys [W]} (train-projection train-data init-W E hidden-dim d n-entities {:lr lr :steps steps})
                 {:keys [^floats p]} (forward-probe-single (:h held-out) W E hidden-dim d n-entities)
                 pred-id (argmax p n-entities)
                 hit? (= pred-id (:y held-out))]
             {:fold hold-idx
              :target (:head held-out)
              :pred-id pred-id
              :expected-id (:y held-out)
              :hit? hit?}))
         (range n))
        total-hits (count (filter :hit? fold-results))
        mean-acc (/ (double total-hits) (double n))]
    {:fold-results fold-results
     :total-hits total-hits
     :mean-acc mean-acc}))
