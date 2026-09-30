(ns einsum.quant.turboquant
  "Fast-TurboQuant: Multiplier-Free Online Vector Quantization with Lloyd-Max and QJL Residual Correction.
   Implements 3-tier KV compression (arXiv:2606.21448 & arXiv:2504.19874) in pure Clojure with
   OpenXLA StableHLO tensor AST lowerings."
  (:import [java.util Random]))

;; ==============================================================================
;; 1. Mathematical Constants & Codebooks
;; ==============================================================================

;; Optimal 2-bit Lloyd-Max thresholds and reconstruction centroids for standard normal N(0, 1)
(def ^:const LLOYD_MAX_THRESHOLDS [-0.9816 0.0 0.9816])
(def ^:const LLOYD_MAX_CENTROIDS [-1.5104 -0.4528 0.4528 1.5104])

(def ^:const SQRT_PI_OVER_TWO (Math/sqrt (/ Math/PI 2.0)))

;; Cached deterministic signs and projection matrices
(defonce ^:private rademacher-cache (atom {}))
(defonce ^:private qjl-cache (atom {}))

;; ==============================================================================
;; 2. Deterministic Rademacher & QJL Projections
;; ==============================================================================

(defn rademacher-signs
  "Returns a deterministic pseudo-random sign vector in {-1.0, 1.0}^d generated with fixed seed."
  (^doubles [^long d] (rademacher-signs d 1337))
  (^doubles [^long d ^long seed]
   (let [k [d seed]]
     (or (get @rademacher-cache k)
         (let [rng (Random. seed)
               arr (double-array d)]
           (dotimes [i d]
             (aset-double arr i (if (.nextBoolean rng) 1.0 -1.0)))
           (swap! rademacher-cache assoc k arr)
           arr)))))

(defn qjl-projection-matrix
  "Returns a deterministic random sign matrix S in {-1.0, 1.0}^{m x d} with fixed seed.
   Returns an array of double-arrays (m rows of dimension d)."
  ([^long d ^long m] (qjl-projection-matrix d m 4242))
  ([^long d ^long m ^long seed]
   (let [k [d m seed]]
     (or (get @qjl-cache k)
         (let [rng (Random. seed)
               matrix (into-array (Class/forName "[D")
                                  (mapv (fn [_]
                                          (let [row (double-array d)]
                                            (dotimes [j d]
                                              (aset-double row j (if (.nextBoolean rng) 1.0 -1.0)))
                                            row))
                                        (range m)))]
           (swap! qjl-cache assoc k matrix)
           matrix)))))

;; ==============================================================================
;; 3. Multiplier-Free Fast Walsh-Hadamard Transform (FWHT)
;; ==============================================================================

(defn fwht-doubles!
  "In-place Fast Walsh-Hadamard Transform on double-array of length 2^k.
   The butterfly stages are strictly multiplier-free additions and subtractions.
   Optionally scales by 1/sqrt(d) when normalized? is true."
  (^doubles [^doubles arr] (fwht-doubles! arr false))
  (^doubles [^doubles arr normalized?]
   (let [n (alength arr)]
     (loop [stride 1]
       (when (< stride n)
         (loop [i 0]
           (when (< i n)
             (dotimes [j stride]
               (let [idx1 (+ i j)
                     idx2 (+ idx1 stride)
                     u (aget arr idx1)
                     w (aget arr idx2)]
                 (aset-double arr idx1 (+ u w))
                 (aset-double arr idx2 (- u w))))
             (recur (+ i (* 2 stride)))))
         (recur (* 2 stride))))
     (when normalized?
       (let [scale (/ 1.0 (Math/sqrt (double n)))]
         (dotimes [i n]
           (aset-double arr i (* (aget arr i) scale)))))
     arr)))

(defn fwht-vector
  "Pure host reference implementation of Fast Walsh-Hadamard Transform (FWHT) for vectors of length 2^k.
   The butterfly stages are strictly multiplier-free additions and subtractions.
   Options:
     :normalized? - if true, scales output by 1/sqrt(d) (default: false)."
  ([v] (fwht-vector v {}))
  ([v {:keys [normalized?] :or {normalized? false}}]
   (let [n (count v)
         k (long (/ (Math/log n) (Math/log 2)))
         _ (assert (= n (bit-shift-left 1 k)) (str "FWHT dimension must be a power of 2, got: " n))
         arr (double-array n)]
     (dotimes [i n]
       (aset-double arr i (double (nth v i))))
     (fwht-doubles! arr (boolean normalized?))
     (vec arr))))

;; ==============================================================================
;; 4. 2-Bit Lloyd-Max Quantizer & Packing
;; ==============================================================================

(defn encode-lloyd-max-coord
  "Quantizes normalized coordinate z against Gaussian standard deviation sigma into 2-bit code (0..3)."
  ^long [^double z ^double sigma]
  (if (<= sigma 1e-12)
    1
    (let [u (/ z sigma)]
      (cond
        (< u -0.9816) 0
        (< u 0.0) 1
        (< u 0.9816) 2
        :else 3))))

(defn decode-lloyd-max-coord
  "Reconstructs Gaussian coordinate from 2-bit code (0..3) scaled by sigma."
  ^double [^long code ^double sigma]
  (case (int (bit-and code 3))
    0 (* -1.5104 sigma)
    1 (* -0.4528 sigma)
    2 (* 0.4528 sigma)
    3 (* 1.5104 sigma)))

(defn pack-2bit-byte
  "Packs four 2-bit unsigned codes (0..3) into a single unsigned byte (0..255)."
  ^long [^long c0 ^long c1 ^long c2 ^long c3]
  (bit-or (bit-and c0 3)
          (bit-shift-left (bit-and c1 3) 2)
          (bit-shift-left (bit-and c2 3) 4)
          (bit-shift-left (bit-and c3 3) 6)))

(defn unpack-2bit-byte
  "Unpacks an unsigned byte into a 4-element vector of 2-bit codes [c0 c1 c2 c3]."
  [^long b]
  [(bit-and b 3)
   (bit-and (bit-shift-right b 2) 3)
   (bit-and (bit-shift-right b 4) 3)
   (bit-and (bit-shift-right b 6) 3)])

(defn pack-turboquant-codes
  "Packs a sequence or int-array of 2-bit codes into a byte-array."
  ^bytes [codes]
  (if (instance? (Class/forName "[I") codes)
    (let [arr ^ints codes
          n (alength arr)
          _ (assert (zero? (mod n 4)) (str "Code count must be multiple of 4, got: " n))
          num-bytes (quot n 4)
          out (byte-array num-bytes)]
      (dotimes [i num-bytes]
        (let [base (* i 4)
              c0 (aget arr (+ base 0))
              c1 (aget arr (+ base 1))
              c2 (aget arr (+ base 2))
              c3 (aget arr (+ base 3))
              b (pack-2bit-byte c0 c1 c2 c3)]
          (aset-byte out i (unchecked-byte b))))
      out)
    (let [n (count codes)
          _ (assert (zero? (mod n 4)) (str "Code count must be multiple of 4, got: " n))
          num-bytes (quot n 4)
          out (byte-array num-bytes)]
      (dotimes [i num-bytes]
        (let [base (* i 4)
              c0 (long (nth codes (+ base 0)))
              c1 (long (nth codes (+ base 1)))
              c2 (long (nth codes (+ base 2)))
              c3 (long (nth codes (+ base 3)))
              b (pack-2bit-byte c0 c1 c2 c3)]
          (aset-byte out i (unchecked-byte b))))
      out)))

(defn unpack-turboquant-codes
  "Unpacks a byte-array into a vector of 2-bit codes."
  [^bytes packed]
  (let [num-bytes (alength packed)
        out (long-array (* num-bytes 4))]
    (dotimes [i num-bytes]
      (let [b (bit-and (int (aget packed i)) 0xFF)
            base (* i 4)]
        (aset-long out (+ base 0) (bit-and b 3))
        (aset-long out (+ base 1) (bit-and (bit-shift-right b 2) 3))
        (aset-long out (+ base 2) (bit-and (bit-shift-right b 4) 3))
        (aset-long out (+ base 3) (bit-and (bit-shift-right b 6) 3))))
    (vec out)))

;; ==============================================================================
;; 5. Fast-TurboQuant Pack / Unpack / Inner-Product
;; ==============================================================================

(defn- to-double-array ^doubles [v ^long d]
  (if (instance? (Class/forName "[D") v)
    ^doubles v
    (let [arr (double-array d)]
      (dotimes [i d]
        (aset-double arr i (double (nth v i))))
      arr)))

(defn fast-turboquant-pack
  "Compresses activation/key vector x into Fast-TurboQuant packed format:
   - Rademacher rotation + FWHT
   - 2-bit Lloyd-Max quantization
   - 1-bit QJL residual sketch projection (m bits)
   Returns map with packed byte buffers and scalar norms."
  ([x] (fast-turboquant-pack x (count x) 32))
  ([x ^long d ^long m]
   (let [signs ^doubles (rademacher-signs d)
         inv-sqrt-d (/ 1.0 (Math/sqrt (double d)))
         ^doubles x-arr (to-double-array x d)
         ;; 1. Rademacher phase inversion: x_tilde = D * x
         x-tilde (double-array d)
         norm-x (loop [i 0 acc 0.0]
                  (if (< i d)
                    (let [val (* (aget x-arr i) (aget signs i))]
                      (aset-double x-tilde i val)
                      (recur (inc i) (+ acc (* val val))))
                    (Math/sqrt acc)))
         sigma (* norm-x inv-sqrt-d)]
     ;; 2. FWHT: y = FWHT(x_tilde)
     (fwht-doubles! x-tilde false)
     ;; 3 & 4. Lloyd-Max Quantization and reconstruction
     (let [codes (int-array d)
           z-hat (double-array d)]
       (dotimes [i d]
         (let [zi (* (aget x-tilde i) inv-sqrt-d)
               c (encode-lloyd-max-coord zi sigma)]
           (aset-int codes i c)
           (aset-double z-hat i (decode-lloyd-max-coord c sigma))))
       ;; 5. Inverse FWHT and Rademacher: x_hat = D * FWHT(z_hat) / sqrt(d)
       (fwht-doubles! z-hat false)
       (let [x-hat (double-array d)
             r (double-array d)
             norm-r (loop [i 0 acc 0.0]
                      (if (< i d)
                        (let [rec (* (aget z-hat i) (aget signs i) inv-sqrt-d)
                              orig (aget x-arr i)
                              diff (- orig rec)]
                          (aset-double x-hat i rec)
                          (aset-double r i diff)
                          (recur (inc i) (+ acc (* diff diff))))
                        (Math/sqrt acc)))
             ;; 6. 1-bit QJL residual sketch: sign(S * r)
             s-proj ^objects (qjl-projection-matrix d m)
             qjl-words (long-array (max 1 (quot (+ m 63) 64)))]
         (dotimes [j m]
           (let [^doubles row (aget s-proj (int j))
                 dot (loop [k 0 acc 0.0]
                       (if (< k d)
                         (recur (inc k) (+ acc (* (aget row k) (aget r k))))
                         acc))]
             (when (>= dot 0.0)
               (let [word-idx (quot j 64)
                     bit-idx (rem j 64)]
                 (aset-long qjl-words word-idx
                            (bit-or (aget qjl-words word-idx)
                                    (bit-shift-left 1 bit-idx)))))))
         {:dim d
          :m m
          :packed-codes (pack-turboquant-codes codes)
          :sigma (float sigma)
          :qjl-bits qjl-words
          :residual-norm (float norm-r)})))))

(defn fast-turboquant-unpack-doubles
  "Reconstructs primary vector x_hat from Fast-TurboQuant packed format as a primitive double-array."
  ^doubles [{:keys [dim packed-codes sigma]}]
  (let [d (long dim)
        signs ^doubles (rademacher-signs d)
        inv-sqrt-d (/ 1.0 (Math/sqrt (double d)))
        codes (unpack-turboquant-codes packed-codes)
        z-hat (double-array d)]
    (dotimes [i d]
      (aset-double z-hat i (decode-lloyd-max-coord (int (nth codes i)) (double sigma))))
    (fwht-doubles! z-hat false)
    (let [x-hat (double-array d)]
      (dotimes [i d]
        (aset-double x-hat i (* (aget z-hat i) (aget signs i) inv-sqrt-d)))
      x-hat)))

(defn fast-turboquant-unpack
  "Reconstructs primary vector x_hat from Fast-TurboQuant packed format."
  [packed-k]
  (vec (fast-turboquant-unpack-doubles packed-k)))

(defn turboquant-inner-product
  "Computes mathematically unbiased attention inner product between query vector q
   and compressed key packed-k using the QJL residual estimator."
  ^double [q {:keys [dim m qjl-bits residual-norm] :as packed-k}]
  (let [d (long dim)
        m (long m)
        ^doubles q-arr (to-double-array q d)
        k-hat ^doubles (fast-turboquant-unpack-doubles packed-k)
        primary-dot (loop [i 0 acc 0.0]
                      (if (< i d)
                        (recur (inc i) (+ acc (* (aget q-arr i) (aget k-hat i))))
                        acc))
        r-norm (double residual-norm)]
    (if (or (<= r-norm 1e-12) (zero? m))
      primary-dot
      ;; Unbiased QJL residual correction
      (let [s-proj ^objects (qjl-projection-matrix d m)
            ^longs qjl-arr qjl-bits
            qjl-sum (loop [j 0 acc 0.0]
                      (if (< j m)
                        (let [^doubles row (aget s-proj (int j))
                              sq-dot (loop [k 0 dot 0.0]
                                       (if (< k d)
                                         (recur (inc k) (+ dot (* (aget row k) (aget q-arr k))))
                                         dot))
                              word-idx (quot j 64)
                              bit-idx (rem j 64)
                              bit-val (bit-and (bit-shift-right (aget qjl-arr word-idx) bit-idx) 1)
                              sign-val (if (pos? bit-val) 1.0 -1.0)]
                          (recur (inc j) (+ acc (* sq-dot sign-val))))
                        acc))
            correction (* SQRT_PI_OVER_TWO (/ r-norm (double m)) qjl-sum)]
        (+ primary-dot correction)))))

(defn packed-byte-size
  "Calculates the total byte footprint of a packed Fast-TurboQuant vector."
  [{:keys [packed-codes qjl-bits]}]
  (let [code-bytes (if packed-codes (alength ^bytes packed-codes) 0)
        qjl-bytes (if qjl-bits (* 8 (alength ^longs qjl-bits)) 0)
        scalar-bytes 4]
    (+ code-bytes qjl-bytes scalar-bytes)))

;; ==============================================================================
;; 6. Declarative Tensor Logic AST Helpers
;; ==============================================================================

(defn turboquant-unpack-ast
  "Constructs Declarative Tensor Logic AST block for unpacking Fast-TurboQuant keys/values:
   out-head: [out_var b p h d]
   codes-term: [codes_var b p h d_quarter]
   scales-term: [scales_var b p h 1]
   signs-term: [signs_var d]"
  ([out-head codes-term scales-term signs-term]
   (turboquant-unpack-ast out-head codes-term scales-term signs-term {}))
  ([out-head codes-term scales-term signs-term attrs]
   (let [b-name (or (:name attrs) :turboquant_unpack)]
     [:block {:name b-name}
      [:turboquant-unpack out-head codes-term scales-term signs-term attrs]])))
