(ns einsum.compiler.stablehlo
  "EDN SSA Graph Schema Definition, Validation, and MLIR Serialization."
  (:require [clojure.string :as str]
            [malli.core :as m]))

(def TensorType
  [:cat [:enum :tensor] [:vector :int] [:enum :f32 :f64 :i32 :i64 :f16 :bf16 :i8 :i16 :u8 :u16 :u32 :u64 :i1 :bool]])

(def VariableDecl
  [:tuple :keyword TensorType])

(def Equation
  [:map
   [:op keyword?]
   [:invars {:optional true} [:vector keyword?]]
   [:outvars [:vector keyword?]]
   [:value {:optional true} any?]
   [:attrs {:optional true} map?]])

(def GraphSchema
  [:map
   [:name string?]
   [:invars [:vector VariableDecl]]
   [:outvars [:vector keyword?]]
   [:eqns [:vector Equation]]
   [:f {:optional true} any?]])

(defn validate-graph
  "Validates graph structure against Malli GraphSchema."
  [graph]
  (if (m/validate GraphSchema graph)
    graph
    (throw (ex-info "Invalid SSA Graph EDN schema"
                    {:errors (m/explain GraphSchema graph)}))))

(defn type->mlir-string
  "Converts EDN SSA tensor type tuple to MLIR type string."
  [[_dims shape dtype]]
  (let [dtype-str (name dtype)
        dims-str (str/join "x" shape)]
    (if (seq shape)
      (str "tensor<" dims-str "x" dtype-str ">")
      (str "tensor<" dtype-str ">"))))

(defn- format-op-name [op-kw]
  (case op-kw
    :stablehlo/exp "stablehlo.exponential"
    (let [n (name op-kw)]
      (if (str/includes? n "/")
        (str/replace n "/" ".")
        (str "stablehlo." n)))))

(defn- parse-tensor-dims [type-str]
  (when type-str
    (if-let [m (re-find #"tensor<([0-9x]+)x([a-z0-9]+)>" type-str)]
      (let [dims (mapv #(Integer/parseInt %) (str/split (nth m 1) #"x"))
            dtype (nth m 2)]
        [dims dtype])
      (when-let [m (re-find #"tensor<([a-z0-9]+)>" type-str)]
        [[] (nth m 1)]))))

(defn- find-tensor-type [invars var-types]
  (let [types (keep #(get var-types %) invars)]
    (last (sort-by (fn [t]
                     (if-let [[dims _] (parse-tensor-dims t)]
                       [(count dims) (reduce clojure.core/* 1 dims)]
                       [0 0]))
                   types))))

(defn infer-var-types
  "Infers StableHLO tensor types for each variable in the equation list."
  [invars eqns]
  (let [initial-types (into {} (map (fn [[v t]] [v (type->mlir-string t)]) invars))]
    (reduce (fn [acc {:keys [op invars outvars attrs call_target_name] :as eqn}]
              (let [in-vars (or invars [])
                    non-scalar-t (find-tensor-type in-vars acc)
                    first-in (first in-vars)
                    in-type (or non-scalar-t (when first-in (get acc first-in)))
                    target-name (or (:call_target_name attrs) (:call_target_name eqn) call_target_name)]
                (cond
                  (= op :stablehlo/constant)
                  (let [val (:value eqn)
                        out-t (if-let [t (or (:type eqn) (get-in eqn [:attrs :type]))]
                                (type->mlir-string t)
                                (if (vector? val) "tensor<128x128xf32>" "tensor<f32>"))]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/gather)
                  (let [[operand start-indices] in-vars
                        operand-t (get acc operand "tensor<50257x768xf32>")
                        indices-t (get acc start-indices "tensor<1x128xi32>")
                        [op-dims op-dtype] (or (parse-tensor-dims operand-t) [[50257 768] "f32"])
                        [idx-dims _] (or (parse-tensor-dims indices-t) [[1 128] "i32"])
                        hidden-dim (last op-dims)
                        base-idx-dims (cond
                                        (empty? idx-dims) []
                                        (and (= (count idx-dims) 1) (= (first idx-dims) 1)) []
                                        (and (> (count idx-dims) 1) (= (last idx-dims) 1)) (pop idx-dims)
                                        :else idx-dims)
                        out-dims (if (empty? base-idx-dims) [hidden-dim] (conj base-idx-dims hidden-dim))
                        out-t (str "tensor<" (str/join "x" out-dims) "x" op-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/convert)
                  (let [target-dtype (name (get attrs :target_dtype :f32))
                        [in-dims _] (or (parse-tensor-dims in-type) [[1] "i32"])
                        out-t (if (seq in-dims)
                                (str "tensor<" (str/join "x" in-dims) "x" target-dtype ">")
                                (str "tensor<" target-dtype ">"))]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/compare)
                  (let [in-var (first in-vars)
                        in-type (get acc in-var "tensor<1x128x768xf32>")
                        [in-dims _] (or (parse-tensor-dims in-type) [[1] "f32"])
                        out-type (if (seq in-dims)
                                   (str "tensor<" (str/join "x" in-dims) "xi1>")
                                   "tensor<i1>")]
                    (assoc acc (first outvars) out-type))

                  (= op :stablehlo/reshape)
                  (let [new-shape (get attrs :shape [1 128 1])
                        [_in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 128] "f32"])
                        out-t (if (seq new-shape)
                                (str "tensor<" (str/join "x" new-shape) "x" in-dtype ">")
                                (str "tensor<" in-dtype ">"))]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/transpose)
                  (let [perm (get attrs :permutation [0 2 1])
                        [in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 128 768] "f32"])
                        new-dims (mapv #(nth in-dims %) perm)
                        out-t (str "tensor<" (str/join "x" new-dims) "x" in-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/broadcast_in_dim)
                  (let [target-shape (or (get attrs :target_shape) (get attrs :shape) [1 3 3 128 64])
                        [_in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 3 1 128 64] "f32"])
                        out-t (if (seq target-shape)
                                (str "tensor<" (str/join "x" target-shape) "x" in-dtype ">")
                                (str "tensor<" in-dtype ">"))]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/custom_call)
                  (let [in-var (first in-vars)
                        in-t (get acc in-var "tensor<1xi32>")]
                    (assoc acc (first outvars) in-t))

                  (= op :stablehlo/dynamic_update_slice)
                  (let [[op-var _up-var] in-vars
                        op-t (get acc op-var "tensor<1x1x4096x256xbf16>")]
                    (assoc acc (first outvars) op-t))

                  (= op :stablehlo/dynamic_slice)
                  (let [slice-sizes (get attrs :slice_sizes [1 1 4096 256])
                        [_in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 1 4096 256] "f32"])
                        out-t (str "tensor<" (str/join "x" slice-sizes) "x" in-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/slice)
                  (let [starts (get attrs :start_indices [0 0 0])
                        limits (get attrs :limit_indices [1 128 768])
                        strides (get attrs :strides [1 1 1])
                        [in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 128 2304] "f32"])
                        rank (count in-dims)
                        out-dims (mapv (fn [i]
                                         (let [s (nth starts i 0)
                                               l (nth limits i (nth in-dims i))
                                               st (nth strides i 1)
                                               s-val (if (number? s) (long s) 0)
                                               l-val (if (number? l) (long l) (long (nth in-dims i)))
                                               st-val (if (number? st) (long st) 1)]
                                           (long (Math/ceil (/ (double (- l-val s-val)) (double st-val))))))
                                       (range rank))
                        out-t (str "tensor<" (str/join "x" out-dims) "x" in-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/concatenate)
                  (let [in-types (mapv #(get acc % "tensor<1x8x128x64xf32>") in-vars)
                        first-type (first in-types)
                        [dims dtype] (or (parse-tensor-dims first-type) [[1 8 128 64] "f32"])
                        dim (get attrs :dimension 3)
                        rank (count dims)
                        dim-idx (if (neg? dim) (clojure.core/+ rank dim) dim)
                        all-dim-sizes (map (fn [t]
                                             (let [[d _] (or (parse-tensor-dims t) [dims "f32"])]
                                               (nth d dim-idx)))
                                           in-types)
                        total-size (reduce clojure.core/+ all-dim-sizes)
                        out-dims (assoc dims dim-idx total-size)
                        out-t (str "tensor<" (str/join "x" out-dims) "x" dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/custom_call)
                  (if (= target-name "embed_lookup")
                    (assoc acc (first outvars) "tensor<1x128x768xf32>")
                    (assoc acc (first outvars) (or in-type "tensor<1x128x576xf32>")))

                  (or (= op :stablehlo/reduce_mean) (= op :stablehlo/reduce_sum) (= op :stablehlo/reduce_max) (= op :stablehlo/reduce_min))
                  (let [axes (get attrs :axes [2])
                        keep-dims? (get attrs :keep_dims true)
                        [in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 128 768] "f32"])
                        rank (count in-dims)
                        norm-axes (set (map #(if (neg? %) (+ rank %) %) axes))
                        out-dims (mapv (fn [idx dim]
                                         (if (contains? norm-axes idx)
                                           (if keep-dims? 1 nil)
                                           dim))
                                       (range rank)
                                       in-dims)
                        out-dims-clean (vec (remove nil? out-dims))
                        out-t (str "tensor<" (str/join "x" out-dims-clean) "x" in-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/dot_general)
                  (let [lhs-v (first in-vars)
                        rhs-v (second in-vars)
                        lhs-t (get acc lhs-v "tensor<1x128x768xf32>")
                        rhs-t (get acc rhs-v "tensor<768x50257xf32>")
                        [lhs-dims lhs-dtype] (or (parse-tensor-dims lhs-t) [[1 128 768] "f32"])
                        [rhs-dims _] (or (parse-tensor-dims rhs-t) [[768 50257] "f32"])
                        c-lhs-set (set (get-in attrs [:contracting_dims :lhs] [(dec (count lhs-dims))]))
                        c-rhs-set (set (get-in attrs [:contracting_dims :rhs] [0]))
                        b-lhs-set (set (get-in attrs [:batch_dims :lhs] []))
                        b-rhs-set (set (get-in attrs [:batch_dims :rhs] []))
                        batch-dims (mapv #(nth lhs-dims %) (get-in attrs [:batch_dims :lhs] []))
                        lhs-free (keep-indexed (fn [idx dim] (when-not (or (c-lhs-set idx) (b-lhs-set idx)) dim)) lhs-dims)
                        rhs-free (keep-indexed (fn [idx dim] (when-not (or (c-rhs-set idx) (b-rhs-set idx)) dim)) rhs-dims)
                        out-dims (vec (concat batch-dims lhs-free rhs-free))
                        out-t (str "tensor<" (str/join "x" out-dims) "x" lhs-dtype ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/iota)
                  (let [len (get attrs :len 128)
                        dt (name (get attrs :dtype :i32))
                        out-t (str "tensor<" len "x" dt ">")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/compare)
                  (let [in-t (get acc (first in-vars) "tensor<i32>")
                        [in-dims _] (parse-tensor-dims in-t)
                        out-t (if (seq in-dims)
                                (str "tensor<" (str/join "x" in-dims) "xi1>")
                                "tensor<i1>")]
                    (assoc acc (first outvars) out-t))

                  (or (= op :stablehlo/not) (= op :stablehlo/and) (= op :stablehlo/or) (= op :stablehlo/shift_right_logical))
                  (let [in-t (get acc (first in-vars) "tensor<i1>")
                        [in-dims in-dtype] (or (parse-tensor-dims in-t) [[] "i1"])
                        dtype-str (or in-dtype "i1")
                        out-t (if (seq in-dims)
                                (str "tensor<" (str/join "x" in-dims) "x" dtype-str ">")
                                (str "tensor<" dtype-str ">"))]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/argmax)
                  (let [in-t (get acc (first in-vars) "tensor<1x262144xf32>")
                        [in-dims _] (or (parse-tensor-dims in-t) [[1 262144] "f32"])
                        raw-axis (long (or (:axis attrs) (:dimension attrs) (dec (count in-dims))))
                        axis (if (neg? raw-axis) (+ (count in-dims) raw-axis) raw-axis)
                        out-dims (vec (concat (subvec (vec in-dims) 0 axis) (subvec (vec in-dims) (inc axis))))
                        out-t (if (seq out-dims)
                                (str "tensor<" (str/join "x" out-dims) "xi32>")
                                "tensor<i32>")]
                    (assoc acc (first outvars) out-t))

                  (= op :stablehlo/while)
                  (reduce (fn [a [inv outv]]
                            (if (and inv outv)
                              (assoc a outv (get a inv "tensor<1xi32>"))
                              a))
                          acc
                          (map vector in-vars outvars))

                  :else
                  (let [target-type (or in-type "tensor<1x128x768xf32>")]
                    (reduce (fn [a v]
                              (if (contains? initial-types v)
                                a
                                (assoc a v target-type)))
                            acc
                            outvars)))))
            initial-types
            eqns)))

(declare format-equation)

(defn format-while-region
  "Formats an SSA sub-graph as an MLIR basic block region for stablehlo.while."
  [graph var-types]
  (let [{:keys [invars outvars eqns carry-invars]} graph
        bb-invars (or carry-invars invars)
        sub-types (merge var-types (infer-var-types invars eqns))
        bb-args (str/join ", " (map (fn [[v t]] (str "%" (name v) ": " (type->mlir-string t))) bb-invars))
        eq-lines (map #(format-equation % sub-types) eqns)
        ret-vars (str/join ", " (map #(str "%" (name %)) outvars))
        ret-types (str/join ", " (map #(get sub-types % "tensor<1xi32>") outvars))]
    (str "    ^bb0(" bb-args "):\n"
         (if (seq (filter seq eq-lines))
           (str (str/join "\n" (filter seq eq-lines)) "\n")
           "")
         "      \"stablehlo.return\"(" ret-vars ") : (" ret-types ") -> ()")))

(defn format-equation
  "Formats an individual tensor equation into StableHLO MLIR instruction text."
  [eqn var-types]
  (let [{:keys [op invars outvars value attrs call_target_name]} (assoc eqn :invars (or (:invars eqn) []))
        mlir-op (format-op-name op)
        out-var (first outvars)
        target-name (or (:call_target_name attrs) (:call_target_name eqn) call_target_name "rope")]
    (cond
      (= op :stablehlo/constant)
      (let [out-type (get var-types out-var "tensor<f32>")
            [_ dtype-extracted] (parse-tensor-dims out-type)
            dtype-str (or dtype-extracted "f32")
            is-int? (boolean (re-find #"^(?:i32|i64|i8|i1|ui8|ui32|si32)$" dtype-str))]
        (if (vector? value)
          (let [flat-vals (flatten value)
                val-strs (if is-int?
                           (map #(if (number? %) (str (long %)) "0") flat-vals)
                           (map #(if (number? %) (format "%.6e" (double %)) "0.000000e+00") flat-vals))
                val-str (str/join ", " val-strs)
                n (count flat-vals)
                type-1d (str "tensor<" n "x" dtype-str ">")]
            (str "    %" (name out-var) "_1d = " mlir-op " dense<[" val-str "]> : " type-1d "\n"
                 "    %" (name out-var) " = stablehlo.reshape %" (name out-var) "_1d : (" type-1d ") -> " out-type))
          (let [num-str (cond
                          (boolean? value) (str value)
                          is-int? (str (long (or value 0)))
                          :else (format "%.6e" (double (or value 0.0))))]
            (str "    %" (name out-var) " = " mlir-op " dense<" num-str "> : " out-type))))

      (= op :stablehlo/convert)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x128x768xf32>")
            target-dtype (name (get attrs :target_dtype :f32))
            [in-dims _] (or (parse-tensor-dims in-type) [[1] "i32"])
            out-type (if (seq in-dims)
                       (str "tensor<" (str/join "x" in-dims) "x" target-dtype ">")
                       (str "tensor<" target-dtype ">"))]
        (str "    %" (name out-var) " = \"stablehlo.convert\"(%" (name in-var) ") : (" in-type ") -> " out-type))

      (= op :stablehlo/compare)
      (let [[in0 in1] invars
            in0-t (get var-types in0 "tensor<1x128x768xf32>")
            in1-t (get var-types in1 in0-t)
            out-type (get var-types out-var "tensor<1x128x768xi1>")
            dir (get attrs :comparison_direction "EQ")]
        (str "    %" (name out-var) " = \"stablehlo.compare\"(%" (name in0) ", %" (name in1) ") {comparison_direction = #stablehlo<comparison_direction " dir ">} : (" in0-t ", " in1-t ") -> " out-type))

      (= op :stablehlo/select)
      (let [[pred on-true on-false] invars
            pred-t (get var-types pred "tensor<1x128x768xi1>")
            true-t (get var-types on-true "tensor<1x128x768xf32>")
            false-t (get var-types on-false true-t)
            out-type (get var-types out-var true-t)]
        (str "    %" (name out-var) " = \"stablehlo.select\"(%" (name pred) ", %" (name on-true) ", %" (name on-false) ") : (" pred-t ", " true-t ", " false-t ") -> " out-type))

      (= op :stablehlo/not)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<i1>")
            out-type (get var-types out-var in-type)]
        (str "    %" (name out-var) " = \"stablehlo.not\"(%" (name in-var) ") : (" in-type ") -> " out-type))

      (= op :stablehlo/gather)
      (let [[operand start-indices] invars
            operand-t (get var-types operand "tensor<50257x768xf32>")
            indices-t (get var-types start-indices "tensor<1x128xi32>")
            out-type (get var-types out-var "tensor<1x128x768xf32>")
            offset-dims (get attrs :offset_dims [2])
            collapsed-slice-dims (get attrs :collapsed_slice_dims [0])
            start-idx-map (get attrs :start_index_map [0])
            idx-vec-dim (get attrs :index_vector_dim 2)
            slice-sizes (get attrs :slice_sizes [1 768])
            slice-str (str/join ", " slice-sizes)]
        (str "    %" (name out-var) " = \"stablehlo.gather\"(%" (name operand) ", %" (name start-indices) ") {"
             "dimension_numbers = #stablehlo.gather<"
             "offset_dims = [" (str/join ", " offset-dims) "], "
             "collapsed_slice_dims = [" (str/join ", " collapsed-slice-dims) "], "
             "start_index_map = [" (str/join ", " start-idx-map) "], "
             "index_vector_dim = " idx-vec-dim ">, "
             "slice_sizes = array<i64: " slice-str ">, "
             "indices_are_sorted = false} : (" operand-t ", " indices-t ") -> " out-type))

      (= op :stablehlo/reshape)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x128xf32>")
            out-type (get var-types out-var "tensor<1x128x1xf32>")]
        (str "    %" (name out-var) " = stablehlo.reshape %" (name in-var) " : (" in-type ") -> " out-type))

      (= op :stablehlo/transpose)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x128x768xf32>")
            out-type (get var-types out-var "tensor<1x768x128xf32>")
            perm (get attrs :permutation [0 2 1])
            perm-str (str/join ", " perm)]
        (str "    %" (name out-var) " = \"stablehlo.transpose\"(%" (name in-var) ") {permutation = array<i64: " perm-str ">} : (" in-type ") -> " out-type))

      (= op :stablehlo/broadcast_in_dim)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x3x1x128x64xf32>")
            out-type (get var-types out-var "tensor<1x3x3x128x64xf32>")
            bcast-dims (get attrs :broadcast_dimensions [0 1 2 3 4])
            bcast-attr (if (seq bcast-dims) (str "array<i64: " (str/join ", " bcast-dims) ">") "array<i64>")]
        (str "    %" (name out-var) " = \"stablehlo.broadcast_in_dim\"(%" (name in-var) ") {broadcast_dimensions = " bcast-attr "} : (" in-type ") -> " out-type))

      (= op :stablehlo/slice)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x128x2304xf32>")
            out-type (get var-types out-var "tensor<1x128x768xf32>")
            starts (get attrs :start_indices [0 0 0])
            limits (get attrs :limit_indices [1 128 768])
            strides (get attrs :strides [1 1 1])
            starts-str (str/join ", " (map #(if (number? %) (long %) 0) starts))
            limits-str (str/join ", " (map #(if (number? %) (long %) 1) limits))
            strides-str (str/join ", " (map #(if (number? %) (long %) 1) strides))]
        (str "    %" (name out-var) " = \"stablehlo.slice\"(%" (name in-var) ") {"
             "limit_indices = array<i64: " limits-str ">, "
             "start_indices = array<i64: " starts-str ">, "
             "strides = array<i64: " strides-str ">} : (" in-type ") -> " out-type))

      (= op :stablehlo/dynamic_update_slice)
      (let [[op-var up-var] invars
            op-type (get var-types op-var "tensor<1x4x128x256xf32>")
            up-type (get var-types up-var "tensor<1x4x1x256xf32>")
            out-type (get var-types out-var op-type)
            starts (get attrs :start_indices [0 0 0 0])
            [in-dims _in-dtype] (or (parse-tensor-dims op-type) [[1 4 128 256] "f32"])
            rank (count in-dims)
            prep-info (mapv (fn [i idx]
                              (let [c-var (str (name out-var) "_c" i)]
                                (cond
                                  (or (and (map? idx) (:id idx)) (keyword? idx) (symbol? idx))
                                  (let [v-id (if (map? idx) (:id idx) idx)
                                        v-type (get var-types v-id "tensor<1xi32>")
                                        [v-dims _] (or (parse-tensor-dims v-type) [[1] "i32"])]
                                    (if (seq v-dims)
                                      (let [v-1d (str "tensor<" (str/join "x" v-dims) "xi32>")]
                                        [(str "    %" c-var "_1d = \"stablehlo.convert\"(%" (name v-id) ") : (" v-type ") -> " v-1d "\n"
                                              "    %" c-var " = stablehlo.reshape %" c-var "_1d : (" v-1d ") -> tensor<i32>")
                                         (str "%" c-var)])
                                      [(str "    %" c-var " = \"stablehlo.convert\"(%" (name v-id) ") : (" v-type ") -> tensor<i32>")
                                       (str "%" c-var)]))

                                  :else
                                  [(str "    %" c-var " = stablehlo.constant dense<" (int idx) "> : tensor<i32>")
                                   (str "%" c-var)])))
                            (range rank) starts)
            const-lines (vec (remove nil? (map first prep-info)))
            idx-args (str/join ", " (map second prep-info))
            op-line (str "    %" (name out-var) " = \"stablehlo.dynamic_update_slice\"(%" (name op-var) ", %" (name up-var) ", " idx-args ") : ("
                         op-type ", " up-type ", " (str/join ", " (repeat rank "tensor<i32>")) ") -> " out-type)]
        (str/join "\n" (concat const-lines [op-line])))

      (= op :stablehlo/dynamic_slice)
      (let [[op-var _idx-var] invars
            op-type (get var-types op-var "tensor<1x1x12288x256xf32>")
            out-type (get var-types out-var op-type)
            slice-sizes (let [ss (get attrs :slice_sizes [1 1 12288 256])]
                          (if (coll? ss) (vec ss) [ss]))
            sizes-attr (str "array<i64: " (str/join ", " slice-sizes) ">")
            starts (let [s (get attrs :start_indices [0 0 0 0])]
                     (if (coll? s) (vec s) [s]))
            [in-dims _in-dtype] (or (parse-tensor-dims op-type) [[1 1 12288 256] "f32"])
            rank (count in-dims)
            prep-info (mapv (fn [i idx]
                              (let [c-var (str (name out-var) "_s" i)]
                                (cond
                                  (or (and (map? idx) (:id idx)) (keyword? idx) (symbol? idx))
                                  (let [v-id (if (map? idx) (:id idx) idx)
                                        v-type (get var-types v-id "tensor<1xi32>")
                                        [v-dims _] (or (parse-tensor-dims v-type) [[1] "i32"])]
                                    (if (seq v-dims)
                                      (let [v-1d (str "tensor<" (str/join "x" v-dims) "xi32>")]
                                        [(str "    %" c-var "_1d = \"stablehlo.convert\"(%" (name v-id) ") : (" v-type ") -> " v-1d "\n"
                                              "    %" c-var " = stablehlo.reshape %" c-var "_1d : (" v-1d ") -> tensor<i32>")
                                         (str "%" c-var)])
                                      [(str "    %" c-var " = \"stablehlo.convert\"(%" (name v-id) ") : (" v-type ") -> tensor<i32>")
                                       (str "%" c-var)]))

                                  :else
                                  [(str "    %" c-var " = stablehlo.constant dense<" (int idx) "> : tensor<i32>")
                                   (str "%" c-var)])))
                            (range rank) starts)
            const-lines (vec (remove nil? (map first prep-info)))
            idx-args (str/join ", " (map second prep-info))
            op-line (str "    %" (name out-var) " = \"stablehlo.dynamic_slice\"(%" (name op-var) ", " idx-args ") {slice_sizes = " sizes-attr "} : ("
                         op-type ", " (str/join ", " (repeat rank "tensor<i32>")) ") -> " out-type)]
        (str/join "\n" (concat const-lines [op-line])))

      (= op :stablehlo/iota)
      (let [len (get attrs :len 128)
            dt (name (get attrs :dtype :i32))
            out-type (str "tensor<" len "x" dt ">")
            dim (get attrs :iota_dimension 0)]
        (str "    %" (name out-var) " = \"stablehlo.iota\"() {iota_dimension = " dim " : i64} : () -> " out-type))

      (= op :stablehlo/concatenate)
      (let [in-args (str/join ", " (map #(str "%" (name %)) invars))
            in-types (str/join ", " (map #(get var-types % "tensor<1x8x128x64xf32>") invars))
            out-type (get var-types out-var "tensor<1x8x128x128xf32>")
            dim (get attrs :dimension 3)]
        (str "    %" (name out-var) " = \"stablehlo.concatenate\"(" in-args ") {dimension = " dim " : i64} : (" in-types ") -> " out-type))

      (or (= op :stablehlo/reduce_mean) (= op :stablehlo/reduce_sum) (= op :stablehlo/reduce_max) (= op :stablehlo/reduce_min))
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<f32>")
            out-type (get var-types out-var "tensor<1x128x1xf32>")
            [in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 12 128 128] "f32"])
            rank (count in-dims)
            axes (get attrs :axes [(dec rank)])
            norm-axes (mapv #(if (neg? %) (+ rank %) %) axes)
            axes-str (str/join ", " norm-axes)
            red-op-name (case op
                          :stablehlo/reduce_sum "stablehlo.add"
                          :stablehlo/reduce_max "stablehlo.maximum"
                          :stablehlo/reduce_min "stablehlo.minimum"
                          :stablehlo/reduce_mean "stablehlo.add")
            is-int-dtype? (boolean (re-find #"^(?:i32|i64|i8|i1|ui8|ui32|si32)$" (str in-dtype)))
            init-const (cond
                         (= op :stablehlo/reduce_max) (if is-int-dtype? "0" "-1.000000e+30")
                         (= op :stablehlo/reduce_min) (if is-int-dtype? "2147483647" "1.000000e+30")
                         :else (if is-int-dtype? "0" "0.000000e+00"))
            norm-axes-set (set norm-axes)
            reduced-dims (mapv #(nth in-dims %) (filter #(not (norm-axes-set %)) (range rank)))
            red-type (if (seq reduced-dims)
                       (str "tensor<" (str/join "x" reduced-dims) "x" in-dtype ">")
                       (str "tensor<" in-dtype ">"))]
        (if (= op :stablehlo/reduce_mean)
          (let [red-count (double (reduce clojure.core/* 1.0 (map #(nth in-dims %) norm-axes)))
                c-name (str (name out-var) "_count")
                unscaled-name (str (name out-var) "_unscaled")]
            (str "    %" (name out-var) "_c0 = stablehlo.constant dense<" init-const "> : tensor<" in-dtype ">\n"
                 "    %" (name out-var) "_red = \"stablehlo.reduce\"(%" (name in-var) ", %" (name out-var) "_c0) ({\n"
                 "    ^bb0(%arg_a: tensor<" in-dtype ">, %arg_b: tensor<" in-dtype ">):\n"
                 "      %arg_res = " red-op-name " %arg_a, %arg_b : tensor<" in-dtype ">\n"
                 "      \"stablehlo.return\"(%arg_res) : (tensor<" in-dtype ">) -> ()\n"
                 "    }) {dimensions = array<i64: " axes-str ">} : (" in-type ", tensor<" in-dtype ">) -> " red-type "\n"
                 "    %" unscaled-name " = stablehlo.reshape %" (name out-var) "_red : (" red-type ") -> " out-type "\n"
                 "    %" c-name " = stablehlo.constant dense<" (format "%.6e" red-count) "> : tensor<" in-dtype ">\n"
                 "    %" c-name "_bcast = \"stablehlo.broadcast_in_dim\"(%" c-name ") {broadcast_dimensions = array<i64>} : (tensor<" in-dtype ">) -> " out-type "\n"
                 "    %" (name out-var) " = stablehlo.divide %" unscaled-name ", %" c-name "_bcast : " out-type))
          (str "    %" (name out-var) "_c0 = stablehlo.constant dense<" init-const "> : tensor<" in-dtype ">\n"
               "    %" (name out-var) "_red = \"stablehlo.reduce\"(%" (name in-var) ", %" (name out-var) "_c0) ({\n"
               "    ^bb0(%arg_a: tensor<" in-dtype ">, %arg_b: tensor<" in-dtype ">):\n"
               "      %arg_res = " red-op-name " %arg_a, %arg_b : tensor<" in-dtype ">\n"
               "      \"stablehlo.return\"(%arg_res) : (tensor<" in-dtype ">) -> ()\n"
               "    }) {dimensions = array<i64: " axes-str ">} : (" in-type ", tensor<" in-dtype ">) -> " red-type "\n"
               "    %" (name out-var) " = stablehlo.reshape %" (name out-var) "_red : (" red-type ") -> " out-type)))

      (= op :stablehlo/argmax)
      (let [in-var (first invars)
            in-type (get var-types in-var "tensor<1x262144xf32>")
            [in-dims in-dtype] (or (parse-tensor-dims in-type) [[1 262144] "f32"])
            raw-axis (long (or (:axis attrs) (:dimension attrs) (dec (count in-dims))))
            axis-idx (if (neg? raw-axis) (+ (count in-dims) raw-axis) raw-axis)
            iota-type (str "tensor<" (str/join "x" in-dims) "xi32>")
            out-dims (vec (concat (subvec (vec in-dims) 0 axis-idx) (subvec (vec in-dims) (inc axis-idx))))
            out-val-type (if (seq out-dims) (str "tensor<" (str/join "x" out-dims) "x" in-dtype ">") (str "tensor<" in-dtype ">"))
            out-idx-type (if (seq out-dims) (str "tensor<" (str/join "x" out-dims) "xi32>") "tensor<i32>")
            iota-var (str (name out-var) "_iota")
            c-neg (str (name out-var) "_neg_inf")
            c-zero (str (name out-var) "_zero_i32")
            red-val (str (name out-var) "_max_val")
            red-idx (name out-var)]
        (str "    %" iota-var " = \"stablehlo.iota\"() {iota_dimension = " axis-idx " : i64} : () -> " iota-type "\n"
             "    %" c-neg " = stablehlo.constant dense<-1.000000e+30> : tensor<" in-dtype ">\n"
             "    %" c-zero " = stablehlo.constant dense<0> : tensor<i32>\n"
             "    %" red-val ", %" red-idx " = \"stablehlo.reduce\"(%" (name in-var) ", %" iota-var ", %" c-neg ", %" c-zero ") ({\n"
             "    ^bb0(%v1: tensor<" in-dtype ">, %i1: tensor<i32>, %v2: tensor<" in-dtype ">, %i2: tensor<i32>):\n"
             "      %cmp = \"stablehlo.compare\"(%v1, %v2) {comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<" in-dtype ">, tensor<" in-dtype ">) -> tensor<i1>\n"
             "      %rv = \"stablehlo.select\"(%cmp, %v1, %v2) : (tensor<i1>, tensor<" in-dtype ">, tensor<" in-dtype ">) -> tensor<" in-dtype ">\n"
             "      %ri = \"stablehlo.select\"(%cmp, %i1, %i2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>\n"
             "      \"stablehlo.return\"(%rv, %ri) : (tensor<" in-dtype ">, tensor<i32>) -> ()\n"
             "    }) {dimensions = array<i64: " axis-idx ">} : (" in-type ", " iota-type ", tensor<" in-dtype ">, tensor<i32>) -> (" out-val-type ", " out-idx-type ")"))

      (= op :stablehlo/custom_call)
      (let [[in0 in1] invars
            in0-t (get var-types in0 "tensor<1x128xi32>")
            in1-t (or (get var-types in1) "tensor<50257x768xf32>")
            out-type (get var-types out-var "tensor<1x128x768xf32>")]
        (if (= target-name "embed_lookup")
          (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in0) ", %" (name in1) ") {call_target_name = \"embed_lookup\"} : (" in0-t ", " in1-t ") -> " out-type)
          (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in0) ") {call_target_name = \"" target-name "\"} : (" in0-t ") -> " out-type)))

      (= op :stablehlo/dot_general)
      (let [[lhs rhs] invars
            lhs-type (get var-types lhs "tensor<1x128x768xf32>")
            rhs-type (get var-types rhs "tensor<768x50257xf32>")
            out-type (get var-types out-var "tensor<1x128x50257xf32>")
            c-dims (get attrs :contracting_dims)
            b-dims (get attrs :batch_dims)
            lhs-c (get c-dims :lhs (if (re-find #"tensor<\d+x\d+x\d+x?" lhs-type) [2] [1]))
            rhs-c (get c-dims :rhs [0])
            lhs-b (get b-dims :lhs nil)
            rhs-b (get b-dims :rhs nil)
            dot-attr (str "#stablehlo.dot<"
                          (when (seq lhs-b) (str "lhs_batching_dimensions = [" (str/join ", " lhs-b) "], "))
                          (when (seq rhs-b) (str "rhs_batching_dimensions = [" (str/join ", " rhs-b) "], "))
                          "lhs_contracting_dimensions = [" (str/join ", " lhs-c) "], "
                          "rhs_contracting_dimensions = [" (str/join ", " rhs-c) "]>")]
        (str "    %" (name out-var) " = \"stablehlo.dot_general\"(%" (name lhs) ", %" (name rhs) ") {dot_dimension_numbers = " dot-attr ", precision = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (" lhs-type ", " rhs-type ") -> " out-type))

      (= op :stablehlo/while)
      (let [in-args (str/join ", " (map #(str "%" (name %)) invars))
            in-types (mapv #(get var-types % "tensor<1xi32>") invars)
            in-types-str (str/join ", " in-types)
            out-types (mapv #(get var-types % "tensor<1xi32>") outvars)
            out-types-str (str/join ", " out-types)
            num-outs (count outvars)
            cond-body (or (get attrs :cond-mlir)
                          (when-let [cg (get attrs :cond-graph)]
                            (format-while-region cg var-types))
                          "      %cond = stablehlo.constant dense<false> : tensor<i1>\n      \"stablehlo.return\"(%cond) : (tensor<i1>) -> ()")
            body-body (or (get attrs :body-mlir)
                          (when-let [bg (get attrs :body-graph)]
                            (format-while-region bg var-types))
                          (str "      \"stablehlo.return\"(" in-args ") : (" in-types-str ") -> ()"))]
        (if (= 1 num-outs)
          (str "    %" (name (first outvars)) " = \"stablehlo.while\"(" in-args ") ({\n"
               cond-body "\n"
               "    }, {\n"
               body-body "\n"
               "    }) : (" in-types-str ") -> (" out-types-str ")")
          (let [base-var (name (first outvars))]
            (str "    %" base-var "_while:" num-outs " = \"stablehlo.while\"(" in-args ") ({\n"
                 cond-body "\n"
                 "    }, {\n"
                 body-body "\n"
                 "    }) : (" in-types-str ") -> (" out-types-str ")\n"
                 (str/join "\n" (map-indexed (fn [idx outv]
                                               (let [ot (nth out-types idx)]
                                                 (str "    %" (name outv) " = stablehlo.reshape %" base-var "_while#" idx " : (" ot ") -> " ot)))
                                             outvars))))))

      :else
      (let [in-types (map #(get var-types % "tensor<1x128x768xf32>") invars)
            out-type (get var-types out-var (or (find-tensor-type invars var-types) (get var-types (first invars)) "tensor<1x128x768xf32>"))
            [_ out-dtype] (or (parse-tensor-dims out-type) [[1] "f32"])
            add-line (fn [lines l]
                       (if (some #(= % l) lines) lines (conj lines l)))
            [in-vars-str prep-lines]
            (reduce (fn [[v-strs p-lines] [inv in-t-raw]]
                      (let [[in-dims in-dtype] (or (parse-tensor-dims in-t-raw) [[1] "f32"])
                            [inv-name in-t p-lines-curr]
                            (if (and (not= in-dtype out-dtype) (not= in-t-raw out-type))
                              (let [c-var (str (name inv) "_conv_" (name out-var))
                                    target-t (if (seq in-dims)
                                               (str "tensor<" (str/join "x" in-dims) "x" out-dtype ">")
                                               (str "tensor<" out-dtype ">"))
                                    cline (str "    %" c-var " = \"stablehlo.convert\"(%" (name inv) ") : (" in-t-raw ") -> " target-t)]
                                [c-var target-t (add-line p-lines cline)])
                              [(name inv) in-t-raw p-lines])]
                        (cond
                          (= in-t out-type)
                          [(conj v-strs (str "%" inv-name)) p-lines-curr]

                          (re-find #"^tensor<(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64>} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          (re-find #"^tensor<\d+x(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                out-dims-str (or (second (re-find #"tensor<([0-9x]+)x(?:bf16|f16|f32|f64|i8|i32|i64)>" out-type)) "")
                                out-rank (count (str/split out-dims-str #"x"))
                                bcast-dim (max 0 (dec out-rank))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64: " bcast-dim ">} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          (re-find #"^tensor<1x\d+x1x(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64: 0, 1, 2>} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          (re-find #"^tensor<1x\d+x\d+x\d+x(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          (re-find #"^tensor<1x1x\d+x\d+x(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64: 0, 1, 2, 3>} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          (re-find #"^tensor<\d+x\d+x(?:bf16|f16|f32|f64|i8|i32|i64)>$" in-t)
                          (let [bcast-var (str inv-name "_bcast_" (name out-var))
                                line (str "    %" bcast-var " = \"stablehlo.broadcast_in_dim\"(%" inv-name ") {broadcast_dimensions = array<i64: 2, 3>} : (" in-t ") -> " out-type)]
                            [(conj v-strs (str "%" bcast-var)) (add-line p-lines-curr line)])

                          :else
                          [(conj v-strs (str "%" inv-name)) p-lines-curr])))
                    [[] []]
                    (map vector invars in-types))
            loc-str (when-let [opn (get-in eqn [:metadata :op-name])]
                      (str " loc(\"" opn "\")"))
            raw-op-line (cond
                          (= op :stablehlo/custom_call)
                          (let [in-var (first invars)
                                target (get attrs :call_target_name "copy")
                                in-t (get var-types in-var "tensor<1xi32>")
                                out-type (get var-types out-var in-t)]
                            (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in-var) ") {call_target_name = \"" target "\"} : (" in-t ") -> " out-type))

                          (= op :debug/check-non-nan)
                          (let [in-var (first invars)
                                in-t (get var-types in-var "tensor<f32>")]
                            (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in-var) ") {call_target_name = \"check_non_nan\"} : (" in-t ") -> tensor<i1>"))

                          (= op :debug/check-non-inf)
                          (let [in-var (first invars)
                                in-t (get var-types in-var "tensor<f32>")]
                            (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in-var) ") {call_target_name = \"check_non_inf\"} : (" in-t ") -> tensor<i1>"))

                          (= op :debug/print)
                          (let [in-var (first invars)
                                in-t (get var-types in-var "tensor<f32>")]
                            (str "    %" (name out-var) " = \"stablehlo.custom_call\"(%" (name in-var) ") {call_target_name = \"debug_print\"} : (" in-t ") -> tensor<i1>"))

                          :else
                          (str "    %" (name out-var) " = " mlir-op " " (str/join ", " in-vars-str) " : " out-type))
            op-line (str raw-op-line (or loc-str ""))]
        (str/join "\n" (concat prep-lines [op-line]))))))

(defn graph->mlir-text
  "Converts a validated EDN SSA graph into a formatted StableHLO MLIR textual module."
  [graph]
  (validate-graph graph)
  (let [{:keys [name invars outvars eqns]} graph
        var-types (infer-var-types invars eqns)
        param-strs (map (fn [[v t]] (str "%" (clojure.core/name v) ": " (type->mlir-string t))) invars)
        fn-args-str (str/join ", " param-strs)
        return-types (map #(get var-types % "tensor<f32>") outvars)
        fn-ret-str (str "(" (str/join ", " return-types) ")")
        eqn-lines (map #(format-equation % var-types) eqns)
        return-vars-str (str/join ", " (map #(str "%" (clojure.core/name %)) outvars))
        return-types-str (str/join ", " return-types)]
    (str "module @" name " {\n"
         "  func.func @main(" fn-args-str ") -> " fn-ret-str " {\n"
         (str/join "\n" eqn-lines) "\n"
         "    return " return-vars-str " : " return-types-str "\n"
         "  }\n"
         "}\n")))
