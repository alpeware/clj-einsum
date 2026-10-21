(ns einsum.test-helpers.parity
  "Numerical tolerance, execution parity checking, and finite-difference gradient verification engine.")

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
