(ns einsum.runtime.sampling
  "Model-agnostic logit sampling math (Temperature, Top-K, Top-P, Softmax sampling).")

(defn apply-repetition-penalty
  "Applies repetition penalty to float logits for tokens present in `seen-ids`.
   For positive logits: logit / penalty. For negative logits: logit * penalty."
  [logits seen-ids penalty]
  (if (or (nil? penalty) (<= penalty 1.0) (empty? seen-ids))
    logits
    (let [seen-set (set seen-ids)]
      (mapv (fn [[idx logit]]
              (if (contains? seen-set idx)
                (if (pos? logit)
                  (/ logit penalty)
                  (* logit penalty))
                logit))
            (map-indexed vector logits)))))

(defn apply-temperature
  "Scales logits by temperature: logits / temp."
  [logits temp]
  (if (or (nil? temp) (<= temp 0.0))
    logits
    (mapv #(/ % temp) logits)))

(defn apply-top-k
  "Filters `logits` keeping only the top `k` values, setting all other logits to -Infinity."
  [logits k]
  (if (or (nil? k) (<= k 0) (>= k (count logits)))
    logits
    (let [indexed (map-indexed vector logits)
          sorted (sort-by second > indexed)
          cutoff-val (second (nth sorted (dec k)))
          min-inf Double/NEGATIVE_INFINITY]
      (mapv (fn [v] (if (>= v cutoff-val) v min-inf)) logits))))

(defn apply-top-p
  "Nucleus sampling: keeps cumulative probability mass up to `p`."
  [logits p]
  (if (or (nil? p) (>= p 1.0) (<= p 0.0))
    logits
    (let [indexed (map-indexed vector logits)
          sorted (sort-by second > indexed)
          max-val (apply max logits)
          exps (mapv #(Math/exp (- % max-val)) (map second sorted))
          sum-exp (reduce + exps)
          probs (mapv #(/ % sum-exp) exps)
          cum-sums (reductions + probs)
          cutoff-idx (count (take-while #(< % p) cum-sums))
          valid-indices (set (map first (take (max 1 (inc cutoff-idx)) sorted)))
          min-inf Double/NEGATIVE_INFINITY]
      (mapv (fn [[idx v]] (if (contains? valid-indices idx) v min-inf)) indexed))))

(defn softmax
  "Computes numerically stable softmax probabilities from logit vector."
  [logits]
  (let [valid (filterv #(not= % Double/NEGATIVE_INFINITY) logits)
        max-v (if (seq valid) (apply max valid) 0.0)
        exps (mapv (fn [v] (if (= v Double/NEGATIVE_INFINITY) 0.0 (Math/exp (- v max-v)))) logits)
        sum-e (reduce + exps)]
    (if (zero? sum-e)
      (vec (repeat (count logits) (/ 1.0 (count logits))))
      (mapv #(/ % sum-e) exps))))

(defn sample-logits
  "Samples a token index from a vector of raw logits given sampling options.
   Opts map supports {:temperature :top-k :top-p}."
  ([logits]
   (sample-logits logits {}))
  ([logits {:keys [temperature top-k top-p repetition-penalty seen-ids]
            :or {temperature 1.0 top-k 0 top-p 1.0 repetition-penalty 1.0 seen-ids []}}]
   (let [penalized (apply-repetition-penalty logits seen-ids repetition-penalty)
         filtered (-> penalized
                      (apply-temperature temperature)
                      (apply-top-k top-k)
                      (apply-top-p top-p))
         probs (softmax filtered)
         r (rand)]
     (loop [idx 0
            cum-p 0.0]
       (if (< idx (count probs))
         (let [p (nth probs idx)
               cum-p' (+ cum-p p)]
           (if (or (>= cum-p' r) (= idx (dec (count probs))))
             idx
             (recur (inc idx) cum-p')))
         (dec (count probs)))))))
