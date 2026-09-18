(ns clj-xla.fuse
  "Operation fusion for the JVM StableHLO backend.

  Pure graph -> graph transformations that fuse common patterns into
  single fused ops.

  Fusions:
  - :fused/matmul-bias: dot_general + broadcast_in_dim + add
  - :fused/layer-norm: full LayerNorm (10 ops -> 1)
  - :fused/softmax: softmax core (5 ops -> 1)")

(defn- build-use-map
  [eqns]
  (reduce (fn [m [idx eq]]
            (reduce (fn [acc v] (update acc v (fnil conj []) idx))
                    m (:invars eq)))
          {} (map-indexed vector eqns)))

(defn- single-use?
  [use-map v]
  (= 1 (count (get use-map v []))))

(defn- find-eq
  [eqns var op]
  (some (fn [[i e]]
          (when (and (= op (:op e))
                     (= var (first (:outvars e))))
            i))
        (map-indexed vector eqns)))

(defn- match-matmul-bias
  [eqns use-map idx]
  (let [eq (nth eqns idx nil)]
    (when (and eq (= :stablehlo/dot_general (:op eq)))
      (let [t1 (first (:outvars eq))]
        (when (single-use? use-map t1)
          (let [add-idx (first (get use-map t1))
                add-eq (nth eqns add-idx)]
            (when (and (= :stablehlo/add (:op add-eq)) (> add-idx idx))
              (let [[a1 a2] (:invars add-eq)
                    bias-var (if (= a1 t1) a2 a1)
                    bias-idx (find-eq eqns bias-var :stablehlo/broadcast_in_dim)]
                (when (and bias-idx (single-use? use-map bias-var)
                           (> bias-idx idx) (< bias-idx add-idx))
                  (let [b-var (first (:invars (nth eqns bias-idx)))]
                    {:fused-op :fused/matmul-bias
                     :invars (vec (concat (:invars eq) [b-var]))
                     :outvars (:outvars add-eq)
                     :attrs {:contracting_dims (get-in eq [:attrs :contracting_dims])
                             :batch_dims (get-in eq [:attrs :batch_dims])}
                     :remove-idxs #{idx bias-idx add-idx}
                     :add-idx add-idx}))))))))))

(defn- match-layer-norm
  [eqns use-map idx]
  (let [eq0 (nth eqns idx nil)]
    (when (and eq0
               (= :stablehlo/reduce_mean (:op eq0))
               (= [-1] (get-in eq0 [:attrs :axes]))
               (get-in eq0 [:attrs :keep_dims]))
      (let [mean-var (first (:outvars eq0))
            x-var (first (:invars eq0))
            sub-idx (when (single-use? use-map mean-var)
                      (first (get use-map mean-var)))
            sub-eq (when sub-idx (nth eqns sub-idx))
            sub-ok? (and sub-eq
                         (= :stablehlo/subtract (:op sub-eq))
                         (= x-var (first (:invars sub-eq)))
                         (= mean-var (second (:invars sub-eq)))
                         (> sub-idx idx))]
        (when sub-ok?
          (let [diff-var (first (:outvars sub-eq))
                diff-user-idxs (distinct (get use-map diff-var))
                mul-idx (when (= 2 (count diff-user-idxs))
                          (some #(let [e (nth eqns %)]
                                   (when (and (= :stablehlo/multiply (:op e))
                                              (= diff-var (first (:invars e)))
                                              (= diff-var (second (:invars e))))
                                     %))
                                diff-user-idxs))
                div-idx (when (= 2 (count diff-user-idxs))
                          (some #(when (= :stablehlo/divide (:op (nth eqns %))) %) diff-user-idxs))]
            (when (and mul-idx div-idx (> mul-idx sub-idx))
              (let [diff-sq-var (first (:outvars (nth eqns mul-idx)))
                    var-idx (when (single-use? use-map diff-sq-var)
                              (first (get use-map diff-sq-var)))
                    var-eq (when var-idx (nth eqns var-idx))]
                (when (and var-eq
                           (= :stablehlo/reduce_mean (:op var-eq))
                           (= [-1] (get-in var-eq [:attrs :axes]))
                           (> var-idx mul-idx))
                  (let [var-var (first (:outvars var-eq))
                        add1-idx (when (single-use? use-map var-var)
                                   (first (get use-map var-var)))
                        add1-eq (when add1-idx (nth eqns add1-idx))]
                    (when (and add1-eq (= :stablehlo/add (:op add1-eq)) (> add1-idx var-idx))
                      (let [[v1 v2] (:invars add1-eq)
                            eps-var (if (= v1 var-var) v2 v1)
                            eps-idx (find-eq eqns eps-var :stablehlo/constant)]
                        (when (and eps-idx (single-use? use-map eps-var))
                          (let [eps-val (double (:value (nth eqns eps-idx)))
                                var-eps-var (first (:outvars add1-eq))
                                sqrt-idx (when (single-use? use-map var-eps-var)
                                           (first (get use-map var-eps-var)))
                                sqrt-eq (when sqrt-idx (nth eqns sqrt-idx))]
                            (when (and sqrt-eq (= :stablehlo/sqrt (:op sqrt-eq)) (> sqrt-idx add1-idx))
                              (let [std-var (first (:outvars sqrt-eq))]
                                (when (and (single-use? use-map std-var)
                                           (= div-idx (first (get use-map std-var))))
                                  (let [xhat-var (first (:outvars (nth eqns div-idx)))
                                        mul2-idx (when (single-use? use-map xhat-var)
                                                   (first (get use-map xhat-var)))
                                        mul2-eq (when mul2-idx (nth eqns mul2-idx))]
                                    (when (and mul2-eq (= :stablehlo/multiply (:op mul2-eq)) (> mul2-idx div-idx))
                                      (let [[m1 m2] (:invars mul2-eq)
                                            gamma-var (if (= m1 xhat-var) m2 m1)
                                            scaled-var (first (:outvars mul2-eq))
                                            add2-idx (when (single-use? use-map scaled-var)
                                                       (first (get use-map scaled-var)))
                                            add2-eq (when add2-idx (nth eqns add2-idx))]
                                        (when (and add2-eq (= :stablehlo/add (:op add2-eq)) (> add2-idx mul2-idx))
                                          (let [[s1 s2] (:invars add2-eq)
                                                beta-var (if (= s1 scaled-var) s2 s1)]
                                            {:fused-op :fused/layer-norm
                                             :invars [x-var gamma-var beta-var]
                                             :outvars (:outvars add2-eq)
                                             :attrs {:eps eps-val}
                                             :remove-idxs #{idx sub-idx mul-idx var-idx
                                                            eps-idx add1-idx sqrt-idx
                                                            div-idx mul2-idx add2-idx}
                                             :add-idx add2-idx}))))))))))))))))))))))

(defn- match-softmax
  "Matches softmax core: reduce_max -> subtract -> exp -> reduce_sum -> divide."
  [eqns use-map idx]
  (let [eq (nth eqns idx nil)]
    (when (and eq
               (= :stablehlo/reduce_max (:op eq))
               (= [-1] (get-in eq [:attrs :axes]))
               (get-in eq [:attrs :keep_dims]))
      (let [max-var (first (:outvars eq))
            x-var (first (:invars eq))]
        (when (single-use? use-map max-var)
          (let [sub-idx (first (get use-map max-var))
                sub-eq (nth eqns sub-idx)]
            (when (and (= :stablehlo/subtract (:op sub-eq))
                       (= x-var (first (:invars sub-eq)))
                       (= max-var (second (:invars sub-eq)))
                       (> sub-idx idx))
              (let [diff-var (first (:outvars sub-eq))]
                (when (single-use? use-map diff-var)
                  (let [exp-idx (first (get use-map diff-var))
                        exp-eq (nth eqns exp-idx)]
                    (when (and (= :stablehlo/exp (:op exp-eq)) (> exp-idx sub-idx))
                      (let [exp-var (first (:outvars exp-eq))
                            exp-uses (get use-map exp-var)
                            sum-idx (some #(when (= :stablehlo/reduce_sum (:op (nth eqns %))) %) exp-uses)
                            div-idx (some #(when (= :stablehlo/divide (:op (nth eqns %))) %) exp-uses)
                            sum-eq (when sum-idx (nth eqns sum-idx))]
                        (when (and sum-idx div-idx sum-eq
                                   (= [-1] (get-in sum-eq [:attrs :axes]))
                                   (> sum-idx exp-idx) (> div-idx sum-idx))
                          (let [sum-var (first (:outvars sum-eq))
                                div-eq (nth eqns div-idx)
                                [d1 d2] (:invars div-eq)]
                            (when (and (= :stablehlo/divide (:op div-eq))
                                       (or (and (= d1 exp-var) (= d2 sum-var))
                                           (and (= d1 sum-var) (= d2 exp-var)))
                                       (single-use? use-map sum-var))
                              {:fused-op :fused/softmax
                               :invars [x-var]
                               :outvars (:outvars div-eq)
                               :attrs {}
                               :remove-idxs #{idx sub-idx exp-idx sum-idx div-idx}
                               :add-idx div-idx})))))))))))))))

(defn fuse-graph
  "Applies fusion rewrites. Pure function: graph -> graph."
  [graph]
  (let [eqns (:eqns graph)
        use-map (build-use-map eqns)
        mm-fusions (keep #(match-matmul-bias eqns use-map %) (range (count eqns)))
        ln-fusions (keep #(match-layer-norm eqns use-map %) (range (count eqns)))
        sm-fusions (keep #(match-softmax eqns use-map %) (range (count eqns)))
        fusions (concat mm-fusions ln-fusions sm-fusions)
        remove-idxs (reduce into #{} (map :remove-idxs fusions))
        fused-by-pos (into {} (map (fn [f] [(:add-idx f) f]) fusions))
        new-eqns (vec
                  (keep-indexed
                   (fn [idx eq]
                     (if (contains? remove-idxs idx)
                       (when-let [f (get fused-by-pos idx)]
                         {:op (:fused-op f)
                          :invars (:invars f)
                          :outvars (:outvars f)
                          :attrs (:attrs f)})
                       eq))
                   eqns))]
    (assoc graph :eqns new-eqns)))
