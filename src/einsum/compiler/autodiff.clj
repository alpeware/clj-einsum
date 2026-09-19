(ns einsum.compiler.autodiff
  "Reverse-mode automatic differentiation (Vector-Jacobian Products) for EDN SSA graphs."
  (:require [einsum.compiler.stablehlo :as shlo]))

(defn- cotangent-var [v]
  (keyword (str "d" (name v))))

(defn- gen-tmp-var [v idx]
  (keyword (str "d" (name v) "_" idx)))

(defn vjp
  "Generates reverse-mode VJP (Vector-Jacobian Product) equations for `forward-graph`.
   Returns a combined forward+backward EDN SSA graph containing gradient computations."
  [forward-graph]
  (shlo/validate-graph forward-graph)
  (let [{:keys [name invars outvars eqns]} forward-graph
        ;; Seed initial cotangents for outvars
        out-cotangents (into {} (map (fn [ov] [ov [(cotangent-var ov)]]) outvars))

        ;; Process equations in reverse order
        [bwd-eqns raw-cotangents]
        (reduce
         (fn [[b-eqns cot-map] {:keys [op invars outvars]}]
           (let [out-v (first outvars)
                 out-cots (get cot-map out-v [(cotangent-var out-v)])
                 curr-cot (if (> (count out-cots) 1)
                            [(first out-cots)]
                            out-cots)
                 active-d (first curr-cot)
                 [in1 in2] invars]
             (cond
               (= op :stablehlo/add)
               (let [d-in1 (gen-tmp-var in1 (count (get cot-map in1 [])))
                     d-in2 (gen-tmp-var in2 (count (get cot-map in2 [])))
                     e1 {:op :stablehlo/constant :value 1.0 :outvars [(keyword (str "c1_" (name in1)))]}
                     e2 {:op :stablehlo/multiply :invars [active-d (keyword (str "c1_" (name in1)))] :outvars [d-in1]}
                     e3 {:op :stablehlo/multiply :invars [active-d (keyword (str "c1_" (name in1)))] :outvars [d-in2]}
                     c-map' (-> cot-map
                                (update in1 (fnil conj []) d-in1)
                                (update in2 (fnil conj []) d-in2))]
                 [(into b-eqns [e1 e2 e3]) c-map'])

               (= op :stablehlo/multiply)
               (let [d-in1 (gen-tmp-var in1 (count (get cot-map in1 [])))
                     d-in2 (gen-tmp-var in2 (count (get cot-map in2 [])))
                     e1 {:op :stablehlo/multiply :invars [active-d in2] :outvars [d-in1]}
                     e2 {:op :stablehlo/multiply :invars [active-d in1] :outvars [d-in2]}
                     c-map' (-> cot-map
                                (update in1 (fnil conj []) d-in1)
                                (update in2 (fnil conj []) d-in2))]
                 [(into b-eqns [e1 e2]) c-map'])

               (= op :stablehlo/constant)
               [b-eqns cot-map]

               :else
               [b-eqns cot-map])))
         [[] out-cotangents]
         (reverse eqns))

        ;; Cotangent accumulation step for multi-use variables
        accum-step
        (reduce
         (fn [[accum-eqns final-cots] [in-var c-list]]
           (if (> (count c-list) 1)
             (let [final-d (cotangent-var in-var)
                   sum-eqn (reduce (fn [prev c-item]
                                     {:op :stablehlo/add :invars [(:outvars prev (first c-list)) c-item] :outvars [final-d]})
                                   c-list)]
               [(conj accum-eqns sum-eqn) (assoc final-cots in-var final-d)])
             (let [final-d (or (first c-list) (cotangent-var in-var))]
               [accum-eqns (assoc final-cots in-var final-d)])))
         [[] {}]
         raw-cotangents)

        [accum-eqns final-grad-vars] accum-step
        in-var-keys (map first invars)
        grad-outvars (mapv #(get final-grad-vars % (cotangent-var %)) in-var-keys)
        all-eqns (vec (concat eqns bwd-eqns accum-eqns))]
    {:name (str name "_vjp")
     :invars invars
     :outvars (vec (concat outvars grad-outvars))
     :eqns all-eqns}))
