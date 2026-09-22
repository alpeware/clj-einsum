(ns einsum.quant.ternary
  "Ternary Quantization ({-1, 0, 1}, 1.58-bit / 2-bit packed representation).
   Reference implementations, packing/unpacking, scaling, and Declarative Tensor Logic AST lowerings."
  (:import [java.util.function IntConsumer]
           [java.util.stream IntStream]))

;; ==============================================================================
;; 1. Encoding / Decoding Constants (Biased Branchless Mapping)
;; ==============================================================================

;; Biased BitNet b1.58 mapping:
;; Code 0 (0b00) -> -1.0  [(0 - 1.0)]
;; Code 1 (0b01) ->  0.0  [(1 - 1.0)]
;; Code 2 (0b10) -> +1.0  [(2 - 1.0)]
;; Code 3 (0b11) ->  0.0  (reserved / zero padding)
;;
;; In OpenXLA StableHLO, unpacking lowers to:
;; val = (convert(shift_right_logical(packed, s) & 0x03) - 1.0) * gamma

(defn decode-ternary-2bit
  "Decodes a 2-bit unsigned code (0..3) into a ternary float value {-1.0, 0.0, 1.0}
   via biased formula: (code - 1.0)."
  ^double [^long code]
  (case (int (bit-and code 3))
    0 -1.0
    1 0.0
    2 1.0
    3 0.0))

(defn encode-ternary-2bit
  "Encodes a ternary value {-1, 0, 1} into a 2-bit unsigned biased code (0, 1, 2)."
  ^long [^double val]
  (cond
    (< val -0.5) 0
    (> val 0.5) 2
    :else 1))

;; ==============================================================================
;; 2. Byte Packing & Unpacking
;; ==============================================================================

(defn unpack-ternary-byte
  "Unpacks an unsigned byte (0..255) containing four 2-bit ternary values (low bits first)
   into a 4-element vector of doubles."
  [^long b]
  [(decode-ternary-2bit (bit-and b 3))
   (decode-ternary-2bit (bit-and (bit-shift-right b 2) 3))
   (decode-ternary-2bit (bit-and (bit-shift-right b 4) 3))
   (decode-ternary-2bit (bit-and (bit-shift-right b 6) 3))])

(defn pack-ternary-byte
  "Packs four ternary values {-1, 0, 1} into a single unsigned byte (0..255)."
  [^double v0 ^double v1 ^double v2 ^double v3]
  (bit-or (encode-ternary-2bit v0)
          (bit-shift-left (encode-ternary-2bit v1) 2)
          (bit-shift-left (encode-ternary-2bit v2) 4)
          (bit-shift-left (encode-ternary-2bit v3) 6)))

(defn unpack-ternary-2bit-ref
  "Pure reference unpacker for 2-bit packed ternary weights.
   packed: sequence or byte-array of packed bytes.
   scale: scalar float multiplier (default 1.0).
   Returns a flat vector of double weights scaled by scale."
  ([packed]
   (unpack-ternary-2bit-ref packed 1.0))
  ([packed scale]
   (let [scale (double scale)
         n (count packed)
         out (double-array (* n 4))]
     (dotimes [i n]
       (let [b (bit-and (int (nth packed i)) 0xFF)
             base (* i 4)]
         (aset-double out (+ base 0) (* (decode-ternary-2bit (bit-and b 3)) scale))
         (aset-double out (+ base 1) (* (decode-ternary-2bit (bit-and (bit-shift-right b 2) 3)) scale))
         (aset-double out (+ base 2) (* (decode-ternary-2bit (bit-and (bit-shift-right b 4) 3)) scale))
         (aset-double out (+ base 3) (* (decode-ternary-2bit (bit-and (bit-shift-right b 6) 3)) scale))))
     (vec out))))

(defn pack-ternary-2bit-ref
  "Pure reference packer for ternary weights {-1.0, 0.0, 1.0}.
   weights: sequence of values in {-1, 0, 1}. Count must be a multiple of 4.
   Returns a vector of packed bytes."
  [weights]
  (let [n (count weights)
        _ (assert (zero? (mod n 4)) (str "Ternary weight count must be divisible by 4, got: " n))
        num-bytes (quot n 4)
        out (byte-array num-bytes)]
    (dotimes [i num-bytes]
      (let [base (* i 4)
            v0 (double (nth weights (+ base 0)))
            v1 (double (nth weights (+ base 1)))
            v2 (double (nth weights (+ base 2)))
            v3 (double (nth weights (+ base 3)))
            b (pack-ternary-byte v0 v1 v2 v3)]
        (aset-byte out i (unchecked-byte b))))
    (vec out)))

;; ==============================================================================
;; 3. Scaling & Quantization (BitNet b1.58 AbsMean)
;; ==============================================================================

(defn compute-absmean-scale
  "Computes the BitNet absmean scale factor gamma = (1 / N) * sum(|W_i|)."
  [weights]
  (let [n (count weights)]
    (if (zero? n)
      1.0
      (let [sum-abs (reduce (fn ^double [^double acc ^double w] (+ acc (Math/abs w))) 0.0 weights)]
        (/ sum-abs (double n))))))

(defn quantize-weights-absmean-ternary
  "Quantizes continuous weights into 2-bit packed ternary weights using the BitNet b1.58 scheme:
   gamma = mean(|W|)
   W_ternary = RoundClip(W / gamma, -1, 1)
   Returns {:data ^bytes :scale double :shape [rows cols]}."
  [weights rows cols]
  (let [total (* rows cols)
        _ (assert (zero? (mod total 4)) (str "Total elements must be divisible by 4, got: " total))
        gamma (compute-absmean-scale weights)
        inv-gamma (if (zero? gamma) 1.0 (/ 1.0 gamma))
        packed-bytes (byte-array (quot total 4))
        w-seq (vec weights)]
    (dotimes [i (quot total 4)]
      (let [base (* i 4)
            q (fn [idx]
                (let [v (* (double (nth w-seq idx)) inv-gamma)]
                  (double (max -1 (min 1 (Math/round v))))))
            b (pack-ternary-byte (q (+ base 0)) (q (+ base 1)) (q (+ base 2)) (q (+ base 3)))]
        (aset-byte packed-bytes i (unchecked-byte b))))
    {:data packed-bytes
     :scale gamma
     :shape [rows cols]}))

(defn- float->bf16-short
  [val]
  (let [bits (Float/floatToRawIntBits (float val))]
    (unchecked-short (bit-shift-right bits 16))))

(defn- bf16-short->float
  [s]
  (let [bits (bit-shift-left (bit-and (int s) 0xffff) 16)]
    (Float/intBitsToFloat bits)))

(defn- quantize-floats-row!
  [^floats f-arr ^bytes data-bytes ^floats scales-floats ^shorts scales-shorts
   r cols quarter-cols]
  (let [r (long r)
        cols (long cols)
        quarter-cols (long quarter-cols)
        r-base (int (* r cols))
        out-base (int (* r quarter-cols))
        sum-abs (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [v (double (aget f-arr (+ r-base i)))]
                      (recur (unchecked-inc i) (+ acc (Math/abs v))))
                    acc))
        norm-sq (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [v (double (aget f-arr (+ r-base i)))]
                      (recur (unchecked-inc i) (+ acc (* v v))))
                    acc))
        g-init (if (pos? cols) (/ sum-abs (double cols)) 0.0)
        [g-final active-cnt]
        (loop [it (int 0) curr (double g-init)]
          (let [delta (* 0.5 curr)
                [active-sum cnt] (loop [i (int 0) acc 0.0 c (int 0)]
                                   (if (< i (int cols))
                                     (let [a (Math/abs (double (aget f-arr (+ r-base i))))]
                                       (if (>= a delta)
                                         (recur (unchecked-inc i) (+ acc a) (unchecked-inc c))
                                         (recur (unchecked-inc i) acc c)))
                                     [acc c]))]
            (if (< it (int 3))
              (if (pos? (int cnt))
                (recur (unchecked-inc it) (/ active-sum (double cnt)))
                [curr cnt])
              [(if (pos? (int cnt)) (/ active-sum (double cnt)) curr) cnt])))
        gamma (if (pos? (long active-cnt))
                (Math/sqrt (/ norm-sq (double active-cnt)))
                g-final)
        delta (* 0.5 gamma)
        inv-delta (double (if (pos? delta) (/ 1.0 delta) 1.0))]
    (when scales-floats
      (aset scales-floats (int r) (float gamma)))
    (when scales-shorts
      (aset scales-shorts (int r) (float->bf16-short (float gamma))))
    (loop [k (int 0)]
      (when (< k (int quarter-cols))
        (let [idx (unchecked-add r-base (unchecked-multiply k 4))
              v0 (* (double (aget f-arr idx)) inv-delta)
              v1 (* (double (aget f-arr (unchecked-inc idx))) inv-delta)
              v2 (* (double (aget f-arr (unchecked-add idx 2))) inv-delta)
              v3 (* (double (aget f-arr (unchecked-add idx 3))) inv-delta)
              c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
              c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
              c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
              c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
              b (unchecked-byte (bit-or c0
                                        (bit-shift-left c1 2)
                                        (bit-shift-left c2 4)
                                        (bit-shift-left c3 6)))]
          (aset data-bytes (unchecked-add out-base k) b)
          (recur (unchecked-inc k)))))))

(defn- quantize-shorts-row!
  [^shorts s-arr ^bytes data-bytes ^floats scales-floats ^shorts scales-shorts
   r cols quarter-cols]
  (let [r (long r)
        cols (long cols)
        quarter-cols (long quarter-cols)
        r-base (int (* r cols))
        out-base (int (* r quarter-cols))
        sum-abs (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [s (aget s-arr (+ r-base i))
                          v (double (bf16-short->float s))]
                      (recur (unchecked-inc i) (+ acc (Math/abs v))))
                    acc))
        norm-sq (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [s (aget s-arr (+ r-base i))
                          v (double (bf16-short->float s))]
                      (recur (unchecked-inc i) (+ acc (* v v))))
                    acc))
        g-init (if (pos? cols) (/ sum-abs (double cols)) 0.0)
        [g-final active-cnt]
        (loop [it (int 0) curr (double g-init)]
          (let [delta (* 0.5 curr)
                [active-sum cnt] (loop [i (int 0) acc 0.0 c (int 0)]
                                   (if (< i (int cols))
                                     (let [s (aget s-arr (+ r-base i))
                                           a (Math/abs (double (bf16-short->float s)))]
                                       (if (>= a delta)
                                         (recur (unchecked-inc i) (+ acc a) (unchecked-inc c))
                                         (recur (unchecked-inc i) acc c)))
                                     [acc c]))]
            (if (< it (int 3))
              (if (pos? (int cnt))
                (recur (unchecked-inc it) (/ active-sum (double cnt)))
                [curr cnt])
              [(if (pos? (int cnt)) (/ active-sum (double cnt)) curr) cnt])))
        gamma (if (pos? (long active-cnt))
                (Math/sqrt (/ norm-sq (double active-cnt)))
                g-final)
        delta (* 0.5 gamma)
        inv-delta (double (if (pos? delta) (/ 1.0 delta) 1.0))]
    (when scales-floats
      (aset scales-floats (int r) (float gamma)))
    (when scales-shorts
      (aset scales-shorts (int r) (float->bf16-short (float gamma))))
    (loop [k (int 0)]
      (when (< k (int quarter-cols))
        (let [idx (unchecked-add r-base (unchecked-multiply k 4))
              v0 (* (double (bf16-short->float (aget s-arr idx))) inv-delta)
              v1 (* (double (bf16-short->float (aget s-arr (unchecked-inc idx)))) inv-delta)
              v2 (* (double (bf16-short->float (aget s-arr (unchecked-add idx 2)))) inv-delta)
              v3 (* (double (bf16-short->float (aget s-arr (unchecked-add idx 3)))) inv-delta)
              c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
              c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
              c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
              c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
              b (unchecked-byte (bit-or c0
                                        (bit-shift-left c1 2)
                                        (bit-shift-left c2 4)
                                        (bit-shift-left c3 6)))]
          (aset data-bytes (unchecked-add out-base k) b)
          (recur (unchecked-inc k)))))))

(defn- quantize-doubles-row!
  [^doubles d-arr ^bytes data-bytes ^floats scales-floats ^shorts scales-shorts
   r cols quarter-cols]
  (let [r (long r)
        cols (long cols)
        quarter-cols (long quarter-cols)
        r-base (int (* r cols))
        out-base (int (* r quarter-cols))
        sum-abs (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [v (aget d-arr (+ r-base i))]
                      (recur (unchecked-inc i) (+ acc (Math/abs v))))
                    acc))
        norm-sq (loop [i (int 0) acc 0.0]
                  (if (< i (int cols))
                    (let [v (aget d-arr (+ r-base i))]
                      (recur (unchecked-inc i) (+ acc (* v v))))
                    acc))
        g-init (if (pos? cols) (/ sum-abs (double cols)) 0.0)
        [g-final active-cnt]
        (loop [it (int 0) curr (double g-init)]
          (let [delta (* 0.5 curr)
                [active-sum cnt] (loop [i (int 0) acc 0.0 c (int 0)]
                                   (if (< i (int cols))
                                     (let [a (Math/abs (aget d-arr (+ r-base i)))]
                                       (if (>= a delta)
                                         (recur (unchecked-inc i) (+ acc a) (unchecked-inc c))
                                         (recur (unchecked-inc i) acc c)))
                                     [acc c]))]
            (if (< it (int 3))
              (if (pos? (int cnt))
                (recur (unchecked-inc it) (/ active-sum (double cnt)))
                [curr cnt])
              [(if (pos? (int cnt)) (/ active-sum (double cnt)) curr) cnt])))
        gamma (if (pos? (long active-cnt))
                (Math/sqrt (/ norm-sq (double active-cnt)))
                g-final)
        delta (* 0.5 gamma)
        inv-delta (double (if (pos? delta) (/ 1.0 delta) 1.0))]
    (when scales-floats
      (aset scales-floats (int r) (float gamma)))
    (when scales-shorts
      (aset scales-shorts (int r) (float->bf16-short (float gamma))))
    (loop [k (int 0)]
      (when (< k (int quarter-cols))
        (let [idx (unchecked-add r-base (unchecked-multiply k 4))
              v0 (* (aget d-arr idx) inv-delta)
              v1 (* (aget d-arr (unchecked-inc idx)) inv-delta)
              v2 (* (aget d-arr (unchecked-add idx 2)) inv-delta)
              v3 (* (aget d-arr (unchecked-add idx 3)) inv-delta)
              c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
              c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
              c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
              c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
              b (unchecked-byte (bit-or c0
                                        (bit-shift-left c1 2)
                                        (bit-shift-left c2 4)
                                        (bit-shift-left c3 6)))]
          (aset data-bytes (unchecked-add out-base k) b)
          (recur (unchecked-inc k)))))))

(defn- quantize-floats-grouped-row!
  [^floats f-arr ^bytes data-bytes ^floats scales-floats ^shorts scales-shorts
   r cols quarter-cols group-size quarter-g num-groups]
  (let [r (long r)
        cols (long cols)
        quarter-cols (long quarter-cols)
        group-size (long group-size)
        quarter-g (long quarter-g)
        num-groups (long num-groups)
        r-base (int (* r cols))
        out-base (int (* r quarter-cols))
        s-base (int (* r num-groups))]
    (dotimes [g num-groups]
      (let [g-base (unchecked-add r-base (unchecked-multiply (int g) (int group-size)))
            g-out-base (unchecked-add out-base (unchecked-multiply (int g) (int quarter-g)))
            sum-abs (loop [i (int 0) acc 0.0]
                      (if (< i (int group-size))
                        (let [v (double (aget f-arr (+ g-base i)))]
                          (recur (unchecked-inc i) (+ acc (Math/abs v))))
                        acc))
            norm-sq (loop [i (int 0) acc 0.0]
                      (if (< i (int group-size))
                        (let [v (double (aget f-arr (+ g-base i)))]
                          (recur (unchecked-inc i) (+ acc (* v v))))
                        acc))
            g-init (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)
            [g-final active-cnt]
            (loop [it (int 0) curr (double g-init)]
              (let [delta (* 0.5 curr)
                    [active-sum cnt] (loop [i (int 0) acc 0.0 c (int 0)]
                                       (if (< i (int group-size))
                                         (let [a (Math/abs (double (aget f-arr (+ g-base i))))]
                                           (if (>= a delta)
                                             (recur (unchecked-inc i) (+ acc a) (unchecked-inc c))
                                             (recur (unchecked-inc i) acc c)))
                                         [acc c]))]
                (if (< it (int 3))
                  (if (pos? (int cnt))
                    (recur (unchecked-inc it) (/ active-sum (double cnt)))
                    [curr cnt])
                  [(if (pos? (int cnt)) (/ active-sum (double cnt)) curr) cnt])))
            gamma (if (pos? (long active-cnt))
                    (Math/sqrt (/ norm-sq (double active-cnt)))
                    g-final)
            delta (* 0.5 gamma)
            inv-delta (double (if (pos? delta) (/ 1.0 delta) 1.0))
            s-idx (unchecked-add s-base (int g))]
        (when scales-floats
          (aset scales-floats s-idx (float gamma)))
        (when scales-shorts
          (aset scales-shorts s-idx (float->bf16-short (float gamma))))
        (loop [k (int 0)]
          (when (< k (int quarter-g))
            (let [idx (unchecked-add g-base (unchecked-multiply k 4))
                  v0 (* (double (aget f-arr idx)) inv-delta)
                  v1 (* (double (aget f-arr (unchecked-inc idx))) inv-delta)
                  v2 (* (double (aget f-arr (unchecked-add idx 2))) inv-delta)
                  v3 (* (double (aget f-arr (unchecked-add idx 3))) inv-delta)
                  c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
                  c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
                  c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
                  c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
                  b (unchecked-byte (bit-or c0
                                            (bit-shift-left c1 2)
                                            (bit-shift-left c2 4)
                                            (bit-shift-left c3 6)))]
              (aset data-bytes (unchecked-add g-out-base k) b)
              (recur (unchecked-inc k)))))))))

(defn- quantize-shorts-grouped-row!
  [^shorts s-arr ^bytes data-bytes ^floats scales-floats ^shorts scales-shorts
   r cols quarter-cols group-size quarter-g num-groups]
  (let [r (long r)
        cols (long cols)
        quarter-cols (long quarter-cols)
        group-size (long group-size)
        quarter-g (long quarter-g)
        num-groups (long num-groups)
        r-base (int (* r cols))
        out-base (int (* r quarter-cols))
        s-base (int (* r num-groups))]
    (dotimes [g num-groups]
      (let [g-base (unchecked-add r-base (unchecked-multiply (int g) (int group-size)))
            g-out-base (unchecked-add out-base (unchecked-multiply (int g) (int quarter-g)))
            sum-abs (loop [i (int 0) acc 0.0]
                      (if (< i (int group-size))
                        (let [s (aget s-arr (+ g-base i))
                              v (double (bf16-short->float s))]
                          (recur (unchecked-inc i) (+ acc (Math/abs v))))
                        acc))
            norm-sq (loop [i (int 0) acc 0.0]
                      (if (< i (int group-size))
                        (let [s (aget s-arr (+ g-base i))
                              v (double (bf16-short->float s))]
                          (recur (unchecked-inc i) (+ acc (* v v))))
                        acc))
            g-init (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)
            [g-final active-cnt]
            (loop [it (int 0) curr (double g-init)]
              (let [delta (* 0.5 curr)
                    [active-sum cnt] (loop [i (int 0) acc 0.0 c (int 0)]
                                       (if (< i (int group-size))
                                         (let [s (aget s-arr (+ g-base i))
                                               a (Math/abs (double (bf16-short->float s)))]
                                           (if (>= a delta)
                                             (recur (unchecked-inc i) (+ acc a) (unchecked-inc c))
                                             (recur (unchecked-inc i) acc c)))
                                         [acc c]))]
                (if (< it (int 3))
                  (if (pos? (int cnt))
                    (recur (unchecked-inc it) (/ active-sum (double cnt)))
                    [curr cnt])
                  [(if (pos? (int cnt)) (/ active-sum (double cnt)) curr) cnt])))
            gamma (if (pos? (long active-cnt))
                    (Math/sqrt (/ norm-sq (double active-cnt)))
                    g-final)
            delta (* 0.5 gamma)
            inv-delta (double (if (pos? delta) (/ 1.0 delta) 1.0))
            s-idx (unchecked-add s-base (int g))]
        (when scales-floats
          (aset scales-floats s-idx (float gamma)))
        (when scales-shorts
          (aset scales-shorts s-idx (float->bf16-short (float gamma))))
        (loop [k (int 0)]
          (when (< k (int quarter-g))
            (let [idx (unchecked-add g-base (unchecked-multiply k 4))
                  v0 (* (double (bf16-short->float (aget s-arr idx))) inv-delta)
                  v1 (* (double (bf16-short->float (aget s-arr (unchecked-inc idx)))) inv-delta)
                  v2 (* (double (bf16-short->float (aget s-arr (unchecked-add idx 2)))) inv-delta)
                  v3 (* (double (bf16-short->float (aget s-arr (unchecked-add idx 3)))) inv-delta)
                  c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
                  c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
                  c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
                  c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
                  b (unchecked-byte (bit-or c0
                                            (bit-shift-left c1 2)
                                            (bit-shift-left c2 4)
                                            (bit-shift-left c3 6)))]
              (aset data-bytes (unchecked-add g-out-base k) b)
              (recur (unchecked-inc k)))))))))

(defn quantize-weights-per-row-ternary
  "Quantizes an array of floats/doubles/BF16-shorts of shape [rows cols] into packed ternary
   bytes of shape [rows (/ cols 4)] with per-row (or block-wise) BitNet b1.58 absmean scaling.
   Runs in parallel across rows. Each byte packs four 2-bit values.
   Returns {:data ^bytes :scales scales-arr :shape [rows quarter-cols] :scale-shape scale-shape}."
  ([w-arr rows cols]
   (quantize-weights-per-row-ternary w-arr rows cols {}))
  ([w-arr rows cols opts]
   (let [quarter-cols (quot (long cols) 4)
         _ (assert (zero? (mod (long cols) 4)) (str "cols must be divisible by 4, got: " cols))
         total-packed (* (long rows) quarter-cols)
         data-bytes (byte-array total-packed)
         target-scale-format (get opts :as :f32)
         raw-group-size (get opts :group-size)
         group-size (when (and raw-group-size (pos? (long raw-group-size)) (zero? (mod (long cols) (long raw-group-size))))
                      (long raw-group-size))
         num-groups (if group-size (quot (long cols) group-size) 1)
         total-scales (* (long rows) num-groups)
         scales-floats (when (= target-scale-format :f32) (float-array total-scales))
         scales-shorts (when (= target-scale-format :bf16) (short-array total-scales))
         is-floats? (instance? (Class/forName "[F") w-arr)
         is-shorts? (instance? (Class/forName "[S") w-arr)
         is-doubles? (instance? (Class/forName "[D") w-arr)]
     (cond
       (and is-floats? group-size)
       (let [f-arr ^floats w-arr
             quarter-g (quot group-size 4)]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-floats-grouped-row! f-arr data-bytes scales-floats scales-shorts
                                                         (long r) (long cols) (long quarter-cols)
                                                         (long group-size) (long quarter-g) (long num-groups)))))))

       (and is-shorts? group-size)
       (let [s-arr ^shorts w-arr
             quarter-g (quot group-size 4)]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-shorts-grouped-row! s-arr data-bytes scales-floats scales-shorts
                                                         (long r) (long cols) (long quarter-cols)
                                                         (long group-size) (long quarter-g) (long num-groups)))))))

       is-floats?
       (let [f-arr ^floats w-arr]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-floats-row! f-arr data-bytes scales-floats scales-shorts
                                                 (long r) (long cols) (long quarter-cols)))))))

       is-shorts?
       (let [s-arr ^shorts w-arr]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-shorts-row! s-arr data-bytes scales-floats scales-shorts
                                                 (long r) (long cols) (long quarter-cols)))))))

       is-doubles?
       (let [d-arr ^doubles w-arr]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-doubles-row! d-arr data-bytes scales-floats scales-shorts
                                                  (long r) (long cols) (long quarter-cols)))))))

       :else
       (let [f-arr (float-array (map float w-arr))]
         (-> (IntStream/range 0 (int rows))
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (quantize-floats-row! f-arr data-bytes scales-floats scales-shorts
                                                 (long r) (long cols) (long quarter-cols))))))))
     {:data data-bytes
      :scales (or scales-floats scales-shorts)
      :shape [rows quarter-cols]
      :scale-shape (if group-size [rows num-groups] [rows])})))

;; ==============================================================================
;; 4. High-level Declarative Tensor Logic AST Blocks
;; ==============================================================================

(defn ternary-linear-ast
  "Constructs a Declarative Tensor Logic AST block for a 1.58-bit ternary linear projection:
   y = x * (unpack_ternary(packed_w) * scale).
   out-head: [y_var batch_dim out_dim]
   x-term: [x_var batch_dim in_dim]
   packed-w-term: [w_var out_dim in_dim_quarter]
   scale-term: [scale_var out_dim] or scalar."
  ([out-head x-term packed-w-term scale-term]
   (ternary-linear-ast out-head x-term packed-w-term scale-term {}))
  ([out-head x-term packed-w-term scale-term attrs]
   (let [b-name (or (:name attrs) :ternary_linear)
         w-deq-head [(keyword (str (name (first out-head)) "_w_ternary"))
                     (nth packed-w-term 1)
                     (nth x-term (dec (count x-term)))]]
     [:block {:name b-name}
      [:ternary-unpack w-deq-head packed-w-term scale-term]
      [:= out-head x-term w-deq-head]])))
