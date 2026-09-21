(ns einsum.quant.quip
  "QuIP# Incoherence Processing (FWHT / RHT) and E8 Lattice Codebooks for OpenXLA PJRT.")

;; --- Reference FWHT and RHT Implementations (Pure Sans-IO) ---

(defn fwht-ref
  "Pure reference implementation of Fast Walsh-Hadamard Transform (FWHT) for vectors of length 2^k."
  [v]
  (let [n (count v)
        k (long (/ (Math/log n) (Math/log 2)))
        _ (assert (= n (bit-shift-left 1 k)) (str "FWHT dimension must be a power of 2, got: " n))
        arr (double-array v)
        inv-sqrt2 (/ 1.0 (Math/sqrt 2.0))]
    (loop [s 0 stride 1]
      (if (= s k)
        (vec arr)
        (do
          (loop [i 0]
            (when (< i n)
              (dotimes [j stride]
                (let [idx1 (+ i j)
                      idx2 (+ idx1 stride)
                      u (aget arr idx1)
                      w (aget arr idx2)]
                  (aset arr idx1 (* (+ u w) inv-sqrt2))
                  (aset arr idx2 (* (- u w) inv-sqrt2))))
              (recur (+ i (* 2 stride)))))
          (recur (inc s) (* 2 stride)))))))

(defn rht-ref
  "Pure reference implementation of Randomized Hadamard Transform (RHT): FWHT(x * S)."
  [v signs]
  (let [scaled (mapv * v signs)]
    (fwht-ref scaled)))

(defn rht-inv-ref
  "Pure reference implementation of inverse RHT: FWHT(y) * S."
  [y signs]
  (let [h (fwht-ref y)]
    (mapv * h signs)))

;; --- Canonical E8P Lattice Codebook (arXiv:2402.04396 Appendix C.1) ---

(def ^:private e8p-norm12-padding
  "The 29 padding elements of |D8_hat| with norm squared 12 from QuIP# Appendix C.1."
  [[3 1 1 1 3 3 3 3]
   [1 3 1 1 3 3 3 3]
   [1 1 3 1 3 3 3 3]
   [1 1 1 3 3 3 3 3]
   [3 3 3 1 3 3 1 1]
   [3 3 3 1 3 1 3 1]
   [3 3 3 1 1 3 3 1]
   [3 3 3 1 3 1 1 3]
   [3 3 3 1 1 3 1 3]
   [3 3 3 1 1 1 3 3]
   [3 3 1 3 3 3 1 1]
   [3 3 1 3 3 1 3 1]
   [3 3 1 3 1 3 3 1]
   [3 3 1 3 3 1 1 3]
   [3 3 1 3 1 3 1 3]
   [3 3 1 3 1 1 3 3]
   [3 1 3 3 3 3 1 1]
   [3 1 3 3 3 1 3 1]
   [3 1 3 3 1 3 3 1]
   [3 1 3 3 3 1 1 3]
   [3 1 3 3 1 3 1 3]
   [1 3 3 3 1 1 3 3]
   [1 3 3 3 3 3 1 1]
   [1 3 3 3 3 1 3 1]
   [1 3 3 3 1 3 3 1]
   [1 3 3 3 3 1 1 3]
   [1 3 3 3 1 3 1 3]
   [1 1 3 3 1 3 3 3]
   [3 3 1 1 3 3 3 1]])

(defn generate-e8p-codebook
  "Constructs the canonical 256-entry 8-dimensional E8P lattice source codebook
   as defined in QuIP# (arXiv:2402.04396 Section 4.2 and Appendix C.1).
   Returns a vector of 256 8D vectors of floats."
  []
  (let [odd-positive-ints [1 3 5 7]
        base-elements (atom [])
        _ (letfn [(search [coord-idx current-sum current-vec]
                    (if (= coord-idx 8)
                      (when (<= current-sum 40)
                        (swap! base-elements conj (mapv #(/ (double %) 2.0) current-vec)))
                      (doseq [x odd-positive-ints]
                        (let [new-sum (+ current-sum (* x x))]
                          (when (<= (+ new-sum (* (- 7 coord-idx) 1)) 40)
                            (search (inc coord-idx) new-sum (conj current-vec x)))))))]
            (search 0 0 []))
        padding (mapv (fn [row] (mapv #(/ (double %) 2.0) row)) e8p-norm12-padding)]
    (vec (concat @base-elements padding))))

;; --- High-level Declarative Tensor Logic Ast Helpers ---

(defn quip-linear-ast
  "Constructs a Declarative Tensor Logic AST block for a QuIP# linear projection:
   y = RHT( ( RHT(x, SU) * dequant(codes, codebook, scales) ), SV ).
   All operations are pure relational tensor equations lowerable to StableHLO."
  ([out-head x-term su-term codes-term codebook-term scales-term sv-term]
   (quip-linear-ast out-head x-term su-term codes-term codebook-term scales-term sv-term {}))
  ([out-head x-term su-term codes-term codebook-term scales-term sv-term attrs]
   (let [b-name (or (:name attrs) :quip_linear)
         x-rot-head (into [(keyword (str (name (first out-head)) "_x_rot"))] (rest x-term))
         w-deq-head [(keyword (str (name (first out-head)) "_w_deq"))
                     (nth x-term (dec (count x-term)))
                     (nth out-head (dec (count out-head)))]
         y-rot-head (into [(keyword (str (name (first out-head)) "_y_rot"))] (rest out-head))]
     [:block {:name b-name}
      [:rht x-rot-head x-term su-term]
      [:quip-dequant w-deq-head codes-term codebook-term scales-term]
      [:= y-rot-head x-rot-head w-deq-head]
      [:rht out-head y-rot-head sv-term {:inverse? true}]])))
