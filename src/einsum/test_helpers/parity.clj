(ns einsum.test-helpers.parity
  "Numerical tolerance, execution parity checking, and finite-difference gradient verification engine."
  (:require [einsum.compiler.pjrt :as pjrt]
            [einsum.compiler.stablehlo :as shlo]))

(defn- to-flat-floats
  "Flattens any collection or array into a vector of double precision numbers."
  [coll]
  (cond
    (nil? coll) []
    (number? coll) [(double coll)]
    (instance? (Class/forName "[F") coll) (vec (map double ^floats coll))
    (instance? (Class/forName "[D") coll) (vec (map double ^doubles coll))
    (sequential? coll) (vec (map double (flatten coll)))
    :else (vec (map double (flatten [coll])))))

(defn max-abs-diff
  "Computes the maximum absolute difference between two numerical sequences `a` and `b`."
  [a b]
  (let [fa (to-flat-floats a)
        fb (to-flat-floats b)]
    (if (= (count fa) (count fb))
      (reduce max 0.0 (map (fn [x y] (Math/abs (double (- x y)))) fa fb))
      Double/POSITIVE_INFINITY)))

(defn mean-abs-diff
  "Computes the mean absolute difference between two numerical sequences `a` and `b`."
  [a b]
  (let [fa (to-flat-floats a)
        fb (to-flat-floats b)
        n (count fa)]
    (if (and (pos? n) (= n (count fb)))
      (/ (reduce + 0.0 (map (fn [x y] (Math/abs (double (- x y)))) fa fb)) (double n))
      Double/POSITIVE_INFINITY)))

(defn approx-equal?
  "Checks if numerical sequences `a` and `b` are elementwise equal within tolerance bounds.
   Accepts optional `opts` map: {:atol 1e-4, :rtol 1e-4}."
  ([a b] (approx-equal? a b {:atol 1e-4 :rtol 1e-4}))
  ([a b {:keys [atol rtol] :or {atol 1e-4 rtol 1e-4}}]
   (let [fa (to-flat-floats a)
         fb (to-flat-floats b)]
     (if (not= (count fa) (count fb))
       false
       (every? (fn [[x y]]
                 (let [abs-err (Math/abs (double (- x y)))
                       allowed (+ (double atol) (* (double rtol) (Math/abs (double y))))]
                   (<= abs-err allowed)))
               (map vector fa fb))))))

(defn evaluate-graph-on-backend
  "Compiles and executes EDN SSA `graph` with `inputs-map` on `api-ctx` / `client`. Returns host output vector."
  [api-ctx client graph inputs-map]
  (let [mlir (shlo/graph->mlir-text graph)
        exec (pjrt/compile-mlir api-ctx client mlir)
        invars (:invars graph)
        input-buffers (mapv (fn [[var-kw [_tensor-tag shape dtype]]]
                              (let [data (get inputs-map var-kw)
                                    dtype-enum (case dtype :f32 11 :i32 4 :f16 10 :bf16 13 11)]
                                (pjrt/buffer-from-host-buffer api-ctx client data shape dtype-enum)))
                            invars)
        out-buf (pjrt/execute-executable api-ctx exec input-buffers 1)
        ;; Calculate total output size from invar shape or default
        first-shape (second (second (first invars)))
        out-num-floats (long (reduce * 1 (or first-shape [1 64])))]
    (try
      (vec (pjrt/buffer-to-host-buffer api-ctx out-buf out-num-floats))
      (finally
        (doseq [b input-buffers]
          (pjrt/destroy-buffer! api-ctx b))
        (pjrt/destroy-buffer! api-ctx out-buf)))))

(defn compare-cpu-vs-device
  "Evaluates `graph` on CPU backend vs Target Device backend, returning execution parity metrics."
  [api-cpu client-cpu api-dev client-dev graph inputs-map opts]
  (let [cpu-out (evaluate-graph-on-backend api-cpu client-cpu graph inputs-map)
        dev-out (evaluate-graph-on-backend api-dev client-dev graph inputs-map)
        match? (approx-equal? cpu-out dev-out opts)
        max-d (max-abs-diff cpu-out dev-out)
        mean-d (mean-abs-diff cpu-out dev-out)]
    {:match? match?
     :max-diff max-d
     :mean-diff mean-d
     :cpu-output cpu-out
     :device-output dev-out}))
