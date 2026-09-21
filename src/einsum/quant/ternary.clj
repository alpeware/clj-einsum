(ns einsum.quant.ternary
  "Ternary Quantization ({-1, 0, 1}, 1.58-bit / 2-bit packed representation)
   Reference implementations, packing/unpacking, scaling, and Declarative Tensor Logic AST lowerings.")

;; --- Encoding / Decoding Constants ---

;; Standard BitNet / 2-bit 1.58b mapping:
;; 00 (0) ->  0.0
;; 01 (1) -> +1.0
;; 10 (2) -> -1.0
;; 11 (3) ->  0.0 (reserved / zero padding)

(def ^:private TERNARY-LOOKUP
  (double-array [0.0 1.0 -1.0 0.0]))

(defn decode-ternary-2bit
  "Decodes a 2-bit unsigned code (0..3) into a ternary float value {-1.0, 0.0, 1.0}."
  ^double [^long code]
  (aget ^doubles TERNARY-LOOKUP (bit-and code 3)))

(defn encode-ternary-2bit
  "Encodes a ternary value {-1, 0, 1} into a 2-bit unsigned code (0, 1, 2)."
  ^long [^double val]
  (cond
    (> val 0.5) 1
    (< val -0.5) 2
    :else 0))

;; --- Byte Packing & Unpacking ---

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

;; --- Scaling & Quantization (BitNet b1.58 AbsMean) ---

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

;; --- High-level Declarative Tensor Logic AST Blocks ---

(defn ternary-linear-ast
  "Constructs a Declarative Tensor Logic AST block for a 1.58-bit ternary linear projection:
   y = x * (unpack_ternary(packed_w) * scale).
   out-head: [y_var batch_dim out_dim]
   x-term: [x_var batch_dim in_dim]
   packed-w-term: [w_var (quot in_dim 4) out_dim]
   scale-term: [scale_var] or scalar."
  ([out-head x-term packed-w-term scale-term]
   (ternary-linear-ast out-head x-term packed-w-term scale-term {}))
  ([out-head x-term packed-w-term scale-term attrs]
   (let [b-name (or (:name attrs) :ternary_linear)
         w-deq-head [(keyword (str (name (first out-head)) "_w_ternary"))
                     (nth x-term (dec (count x-term)))
                     (nth out-head (dec (count out-head)))]]
     [:block {:name b-name}
      [:ternary-dequant w-deq-head packed-w-term scale-term]
      [:= out-head x-term w-deq-head]])))
