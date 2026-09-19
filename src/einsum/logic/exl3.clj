(ns einsum.logic.exl3
  "EXL3 Quantization (turboderp-org/exllamav3) Procedural Codebooks, Trellis Decoding, and OpenXLA PJRT Lowers."
  (:require [einsum.logic.quip :as quip])
  (:import [java.lang.foreign MemorySegment]
           [java.util.function IntConsumer]
           [java.util.stream IntStream]))

;; --- Permutation Tables (Tensor Core Layout <-> Row-Major 16x16) ---

(defn make-tc-perm
  "Returns the canonical 256-element permutation vector mapping EXL3 trellis decode order to 16x16 row-major order."
  []
  (let [p (int-array 256)]
    (dotimes [t 32]
      (let [r0 (* (mod t 4) 2)
            r1 (inc r0)
            r2 (+ r0 8)
            r3 (inc r2)
            c0 (quot t 4)
            c1 (+ c0 8)]
        (aset-int p (+ (* t 8) 0) (+ (* r0 16) c0))
        (aset-int p (+ (* t 8) 1) (+ (* r1 16) c0))
        (aset-int p (+ (* t 8) 2) (+ (* r2 16) c0))
        (aset-int p (+ (* t 8) 3) (+ (* r3 16) c0))
        (aset-int p (+ (* t 8) 4) (+ (* r0 16) c1))
        (aset-int p (+ (* t 8) 5) (+ (* r1 16) c1))
        (aset-int p (+ (* t 8) 6) (+ (* r2 16) c1))
        (aset-int p (+ (* t 8) 7) (+ (* r3 16) c1))))
    (vec p)))

(defn make-tc-perm-inv
  "Returns the inverse permutation vector mapping 16x16 row-major index (r*16 + c) to decode step t."
  []
  (let [p (make-tc-perm)
        inv (int-array 256)]
    (dotimes [i 256]
      (aset-int inv (nth p i) i))
    (vec inv)))

(def tc-perm (delay (make-tc-perm)))
(def tc-perm-inv (delay (make-tc-perm-inv)))

;; --- Procedural Codebooks (65536 Entries, Bit-Exact Parity) ---

(def ^:private MUL1-MULT (long 0x83DCD12D))
(def ^:private MCG-MULT  (long 0xCBAC1FED))

(defn generate-mul1-codebook
  "Generates the official 65536-entry EXL3 mul1 codebook. Bit-exact with decode_3inst<2>."
  []
  (let [cb (double-array 65536)
        k-inv (double (Float/float16ToFloat (short 0x1eee)))
        k-bias (double (Float/float16ToFloat (unchecked-short 0xc931)))]
    (dotimes [s 65536]
      (let [prod (bit-and (* (long s) MUL1-MULT) 0xFFFFFFFF)
            bsum (+ (bit-and prod 0xFF)
                    (bit-and (bit-shift-right prod 8) 0xFF)
                    (bit-and (bit-shift-right prod 16) 0xFF)
                    (bit-and (bit-shift-right prod 24) 0xFF))
            h (+ 1024.0 (double bsum))
            val (+ (* h k-inv) k-bias)]
        (aset-double cb s val)))
    (vec cb)))

(defn generate-mcg-codebook
  "Generates the official 65536-entry EXL3 mcg codebook. Bit-exact with decode_3inst<1>."
  []
  (let [cb (double-array 65536)]
    (dotimes [s 65536]
      (let [prod (bit-and (* (long s) MCG-MULT) 0xFFFFFFFF)
            lop (bit-xor 0x3b603b60 (bit-and prod 0x8fff8fff))
            h0 (Float/float16ToFloat (unchecked-short (bit-and lop 0xFFFF)))
            h1 (Float/float16ToFloat (unchecked-short (bit-shift-right lop 16)))
            val (+ (double h0) (double h1))]
        (aset-double cb s val)))
    (vec cb)))

;; --- Trellis Bitstream Decoding (Sans-IO Pure Logic) ---

(defn decode-state-scalar
  "Decodes a single 16-bit trellis state for element t in [0..255] from packed 16-bit words.
   packed is a sequential vector or array of uint16/int16 words (length 16*bits)."
  [packed t-offset bits]
  (let [words32 (quot (* (long bits) 256) 32)
        b0 (+ (* (long t-offset) (long bits)) (long bits) -16 (* 256 (long bits)))
        b1 (+ b0 16)
        shift (- (* (inc (quot (dec b1) 32)) 32) b1)
        load-u32 (fn [idx]
                   (let [i (* (long idx) 2)
                         lo (bit-and (long (nth packed i)) 0xFFFF)
                         hi (bit-and (long (nth packed (inc i))) 0xFFFF)]
                     (bit-or lo (bit-shift-left hi 16))))
        idx0 (mod (quot b0 32) words32)
        idx1 (mod (quot (dec b1) 32) words32)
        w-hi (load-u32 idx0)
        w-lo (load-u32 idx1)
        merged (bit-or (bit-shift-left (long w-hi) 32) (long w-lo))]
    (bit-and (bit-shift-right merged shift) 0xFFFF)))

(defn dequant-tile-ref
  "Pure reference dequantizer for a single 16x16 tile.
   Returns a 256-element vector of floats in row-major order."
  [packed-tile bits codebook]
  (let [perm @tc-perm
        tile (double-array 256)]
    (dotimes [t 256]
      (let [state (decode-state-scalar packed-tile t bits)
            val (nth codebook state)]
        (aset-double tile (nth perm t) (double val))))
    (vec tile)))

;; --- Fast In-place Block-128 Hadamard Transform ---

(defn fwht128-in-place!
  "In-place normalized Fast Walsh-Hadamard Transform on a double-array slice of length 128.
   Applies 7 butterfly stages with normalization factor 1/sqrt(128)."
  [^doubles arr ^long offset]
  (let [inv-sqrt2 (/ 1.0 (Math/sqrt 2.0))]
    (loop [s 0 stride 1]
      (when (< s 7)
        (loop [i 0]
          (when (< i 128)
            (dotimes [j stride]
              (let [idx1 (+ offset i j)
                    idx2 (+ idx1 stride)
                    u (aget arr idx1)
                    w (aget arr idx2)]
                (aset arr idx1 (* (+ u w) inv-sqrt2))
                (aset arr idx2 (* (- u w) inv-sqrt2))))
            (recur (+ i (* 2 stride)))))
        (recur (inc s) (* 2 stride))))))

(defn hadamard-block-128-ref
  "Applies Fast Walsh-Hadamard Transform in independent blocks of 128 elements.
   Scaled by 1 / sqrt(128) per block."
  [v]
  (let [n (count v)
        _ (assert (zero? (mod n 128)) (str "Length must be divisible by 128, got: " n))
        num-blocks (quot n 128)
        out (double-array n)]
    (dotimes [b num-blocks]
      (let [start (* b 128)
            block-v (subvec v start (+ start 128))
            block-fwht (quip/fwht-ref block-v)]
        (dotimes [i 128]
          (aset-double out (+ start i) (nth block-fwht i)))))
    (vec out)))

(defn float->bf16-short
  "Converts a float to bfloat16 bit-pattern short with round-to-nearest-even."
  [^double f]
  (let [bits (Float/floatToRawIntBits (float f))
        biased (+ bits 0x7fff (bit-and (bit-shift-right bits 16) 1))]
    (short (bit-shift-right biased 16))))

(defn- decode-state-fast
  ^long [^shorts packed ^long tile-base ^long t ^long bits]
  (let [words32 (quot (* bits 256) 32)
        b0 (+ (* t bits) bits -16 (* 256 bits))
        b1 (+ b0 16)
        shift (- (* (inc (quot (dec b1) 32)) 32) b1)
        idx0 (long (rem (quot b0 32) words32))
        idx1 (long (rem (quot (dec b1) 32) words32))
        i0 (+ tile-base (* idx0 2))
        lo0 (bit-and (int (aget packed i0)) 0xFFFF)
        hi0 (bit-and (int (aget packed (inc i0))) 0xFFFF)
        w-hi (bit-or lo0 (bit-shift-left hi0 16))
        i1 (+ tile-base (* idx1 2))
        lo1 (bit-and (int (aget packed i1)) 0xFFFF)
        hi1 (bit-and (int (aget packed (inc i1))) 0xFFFF)
        w-lo (bit-or lo1 (bit-shift-left hi1 16))
        merged (bit-or (bit-shift-left (long w-hi) 32) (long w-lo))]
    (bit-and (bit-shift-right merged shift) 0xFFFF)))

(defn- extract-f16-doubles
  "Extracts float values as double-array from a sequence, float-array, or MemorySegment of float16 values."
  ^doubles [source ^long n]
  (cond
    (instance? MemorySegment source)
    (let [seg ^MemorySegment source
          res (double-array n)
          shorts (short-array n)]
      (MemorySegment/copy seg 0 (MemorySegment/ofArray shorts) 0 (* n 2))
      (dotimes [i n]
        (let [s (aget shorts i)
              f (Float/float16ToFloat s)]
          (aset res i (double f))))
      res)

    (instance? (Class/forName "[F") source)
    (let [res (double-array n)]
      (dotimes [i n]
        (aset-double res i (double (aget ^floats source i))))
      res)

    (instance? (Class/forName "[D") source)
    (let [res (double-array n)]
      (dotimes [i n]
        (aset-double res i (aget ^doubles source i)))
      res)

    :else
    (let [res (double-array n)
          s-vec (vec source)]
      (dotimes [i n]
        (aset-double res i (double (nth s-vec i))))
      res)))

(defn dequant-exl3-matrix
  "Dequantizes a full EXL3 weight matrix to a resident host array or nested vector.
   trellis: MemorySegment, short-array, or nested/flat vector of shape [in-tiles out-tiles words-per-tile]
   in-features: long (divisible by 128)
   out-features: long (divisible by 128)
   bits: long (bitrate K)
   suh: MemorySegment, float-array, or vector of length in-features
   svh: MemorySegment, float-array, or vector of length out-features
   opts: {:codebook (default mul1), :as :bf16 | :f32 | :f64 | :nested-vectors}"
  ([trellis in-features out-features bits suh svh]
   (dequant-exl3-matrix trellis in-features out-features bits suh svh {}))
  ([trellis in-features out-features bits suh svh opts]
   (let [in-features (long in-features)
         out-features (long out-features)
         bits (long bits)
         words-per-tile (* 16 bits)
         in-tiles (quot in-features 16)
         out-tiles (quot out-features 16)
         cb-vec (or (:codebook opts) (generate-mul1-codebook))
         cb ^doubles (double-array cb-vec)
         perm ^ints (int-array @tc-perm)
         suh-arr ^doubles (extract-f16-doubles suh in-features)
         svh-arr ^doubles (extract-f16-doubles svh out-features)
         total-elements (* in-features out-features)
         w ^doubles (double-array total-elements)
         total-words (* in-tiles out-tiles words-per-tile)
         trellis-shorts ^shorts (cond
                                  (instance? MemorySegment trellis)
                                  (let [arr (short-array total-words)]
                                    (MemorySegment/copy ^MemorySegment trellis 0 (MemorySegment/ofArray arr) 0 (* total-words 2))
                                    arr)
                                  (instance? (Class/forName "[S") trellis)
                                  ^shorts trellis
                                  :else
                                  (short-array trellis))]

     ;; Step 1: Trellis decode (parallel across in-tiles)
     (-> (IntStream/range 0 in-tiles)
         (.parallel)
         (.forEach
          (reify IntConsumer
            (accept [_ tk]
              (let [r-base (* tk 16)]
                (dotimes [tn out-tiles]
                  (let [c-base (* tn 16)
                        tile-idx (+ (* tk out-tiles) tn)
                        tile-base (* tile-idx words-per-tile)]
                    (dotimes [t 256]
                      (let [state (decode-state-fast trellis-shorts tile-base t bits)
                            val (aget cb state)
                            p (aget perm t)
                            r (+ r-base (bit-shift-right p 4))
                            c (+ c-base (bit-and p 15))]
                        (aset w (+ (* r out-features) c) val))))))))))

     ;; Step 2: Col FWHT128 (parallel across out-features)
     (let [col-blocks (quot in-features 128)]
       (-> (IntStream/range 0 out-features)
           (.parallel)
           (.forEach
            (reify IntConsumer
              (accept [_ c]
                (let [tmp ^doubles (double-array 128)]
                  (dotimes [bk col-blocks]
                    (let [base (+ (* bk 128 out-features) c)]
                      (dotimes [i 128]
                        (aset tmp i (aget w (+ base (* i out-features)))))
                      (fwht128-in-place! tmp 0)
                      (dotimes [i 128]
                        (aset w (+ base (* i out-features)) (aget tmp i)))))))))))

     ;; Step 3: Multiply by suh (parallel across in-features)
     (-> (IntStream/range 0 in-features)
         (.parallel)
         (.forEach
          (reify IntConsumer
            (accept [_ r]
              (let [s (aget suh-arr r)
                    base (* r out-features)]
                (dotimes [c out-features]
                  (let [idx (+ base c)]
                    (aset w idx (* (aget w idx) s)))))))))

     ;; Step 4: Row FWHT128 (parallel across in-features)
     (let [row-blocks (quot out-features 128)]
       (-> (IntStream/range 0 in-features)
           (.parallel)
           (.forEach
            (reify IntConsumer
              (accept [_ r]
                (dotimes [bn row-blocks]
                  (fwht128-in-place! w (+ (* r out-features) (* bn 128)))))))))

     ;; Step 5: Multiply by svh (sequential row-major in parallel across in-features)
     (-> (IntStream/range 0 in-features)
         (.parallel)
         (.forEach
          (reify IntConsumer
            (accept [_ r]
              (let [base (* r out-features)]
                (dotimes [c out-features]
                  (let [idx (+ base c)]
                    (aset w idx (* (aget w idx) (aget svh-arr c))))))))))

     ;; Return requested format
     (let [transpose? (get opts :transpose? false)]
       (case (:as opts :bf16)
         :bf16
         (let [out-arr ^shorts (short-array total-elements)]
           (if transpose?
             (-> (IntStream/range 0 in-features)
                 (.parallel)
                 (.forEach
                  (reify IntConsumer
                    (accept [_ r]
                      (let [r-out (* r out-features)]
                        (dotimes [c out-features]
                          (let [val (aget w (+ r-out c))
                                bf (float->bf16-short val)]
                            (aset-short out-arr (+ (* c in-features) r) bf))))))))
             (-> (IntStream/range 0 in-features)
                 (.parallel)
                 (.forEach
                  (reify IntConsumer
                    (accept [_ r]
                      (let [r-out (* r out-features)]
                        (dotimes [c out-features]
                          (let [val (aget w (+ r-out c))
                                bf (float->bf16-short val)]
                            (aset-short out-arr (+ r-out c) bf)))))))))
           out-arr)

         :f32
         (let [out-arr ^floats (float-array total-elements)]
           (if transpose?
             (-> (IntStream/range 0 in-features)
                 (.parallel)
                 (.forEach
                  (reify IntConsumer
                    (accept [_ r]
                      (let [r-out (* r out-features)]
                        (dotimes [c out-features]
                          (let [val (float (aget w (+ r-out c)))]
                            (aset-float out-arr (+ (* c in-features) r) val))))))))
             (-> (IntStream/range 0 in-features)
                 (.parallel)
                 (.forEach
                  (reify IntConsumer
                    (accept [_ r]
                      (let [r-out (* r out-features)]
                        (dotimes [c out-features]
                          (let [val (float (aget w (+ r-out c)))]
                            (aset-float out-arr (+ r-out c) val)))))))))
           out-arr)

         :f64
         (if transpose?
           (let [out-arr ^doubles (double-array total-elements)]
             (-> (IntStream/range 0 in-features)
                 (.parallel)
                 (.forEach
                  (reify IntConsumer
                    (accept [_ r]
                      (let [r-out (* r out-features)]
                        (dotimes [c out-features]
                          (aset-double out-arr (+ (* c in-features) r) (aget w (+ r-out c)))))))))
             out-arr)
           w)

         :nested-vectors
         (if transpose?
           (vec (for [c (range out-features)]
                  (vec (for [r (range in-features)]
                         (aget w (+ (* r out-features) c))))))
           (vec (for [r (range in-features)]
                  (vec (for [c (range out-features)]
                         (aget w (+ (* r out-features) c))))))))))))

;; --- High-level Declarative Tensor Logic AST Blocks ---

(defn exl3-linear-ast
  "Constructs a Declarative Tensor Logic AST block for an EXL3 linear projection.
   Decomposes into block-128 Hadamard rotations on input/output and GEMM against inner weights."
  ([out-head x-term trellis-term suh-term svh-term]
   (exl3-linear-ast out-head x-term trellis-term suh-term svh-term {}))
  ([out-head x-term trellis-term suh-term svh-term attrs]
   (let [b-name (or (:name attrs) :exl3_linear)
         x-rot-head (into [(keyword (str (name (first out-head)) "_x_rot"))] (rest x-term))
         w-deq-head [(keyword (str (name (first out-head)) "_w_deq"))
                     (nth x-term (dec (count x-term)))
                     (nth out-head (dec (count out-head)))]
         y-mid-head (into [(keyword (str (name (first out-head)) "_y_mid"))] (rest out-head))]
     [:block {:name b-name}
      [:hadamard-block-128 x-rot-head x-term suh-term]
      [:exl3-dequant w-deq-head trellis-term]
      [:= y-mid-head x-rot-head w-deq-head]
      [:hadamard-block-128 out-head y-mid-head svh-term {:inverse? true}]])))

(defn quantize-weights-per-row-int8
  "Quantizes an array of floats/doubles/BF16-shorts of shape [rows cols] into signed INT8 bytes
   with per-row symmetric scaling. Runs in parallel across rows.
   Returns {:data ^bytes :scales ^floats} (or :scales as ^shorts if :as :bf16)."
  ([w-arr rows cols]
   (quantize-weights-per-row-int8 w-arr rows cols {}))
  ([w-arr rows cols opts]
   (let [total (* rows cols)
         data-bytes (byte-array total)
         target-scale-format (get opts :as :f32)
         scales-floats (when (= target-scale-format :f32) (float-array rows))
         scales-shorts (when (= target-scale-format :bf16) (short-array rows))
         is-floats? (instance? (Class/forName "[F") w-arr)
         is-doubles? (instance? (Class/forName "[D") w-arr)
         is-shorts? (instance? (Class/forName "[S") w-arr)
         f-arr (when is-floats? ^floats w-arr)
         d-arr (when is-doubles? ^doubles w-arr)
         s-arr (when is-shorts? ^shorts w-arr)]
     (-> (IntStream/range 0 rows)
         (.parallel)
         (.forEach
          (reify IntConsumer
            (accept [_ r]
              (let [r-base (* r cols)
                    ;; 1. Compute max absolute value for row
                    max-abs (loop [c 0 m (float 0.0)]
                              (if (>= c cols)
                                m
                                (let [idx (+ r-base c)
                                      v (cond
                                          is-floats? (Math/abs (aget f-arr idx))
                                          is-doubles? (float (Math/abs (aget d-arr idx)))
                                          is-shorts? (let [s (int (aget s-arr idx))
                                                           bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
                                                       (Math/abs (Float/intBitsToFloat bits)))
                                          :else (float (Math/abs (double (nth w-arr idx)))))]
                                  (recur (inc c) (max m v)))))
                    scale (if (zero? max-abs) (float 1.0) (float (/ (double max-abs) 127.0)))
                    inv-scale (float (/ 1.0 scale))]
                (if (= target-scale-format :bf16)
                  (aset-short scales-shorts r (float->bf16-short scale))
                  (aset-float scales-floats r scale))
                ;; 2. Quantize row elements to signed INT8
                (dotimes [c cols]
                  (let [idx (+ r-base c)
                        v (cond
                            is-floats? (aget f-arr idx)
                            is-doubles? (float (aget d-arr idx))
                            is-shorts? (let [s (int (aget s-arr idx))
                                             bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
                                         (Float/intBitsToFloat bits))
                            :else (float (nth w-arr idx)))
                        q (Math/round (* v inv-scale))
                        clamped (max -127 (min 127 q))]
                    (aset-byte data-bytes idx (byte clamped)))))))))
     {:data data-bytes
      :scales (if (= target-scale-format :bf16) scales-shorts scales-floats)})))

(defn quantize-weights-per-row-int4
  "Quantizes an array of floats/doubles/BF16-shorts of shape [rows cols] into packed INT4 bytes
   of shape [rows (/ cols 2)] with per-row symmetric scaling. Runs in parallel across rows.
   Each byte packs 2 adjacent values: low nibble = col 2k, high nibble = col 2k+1 (offset +8).
   Returns {:data ^bytes :scales ^floats} (or :scales as ^shorts if :as :bf16)."
  ([w-arr rows cols]
   (quantize-weights-per-row-int4 w-arr rows cols {}))
  ([w-arr rows cols opts]
   (let [half-cols (quot cols 2)
         total-packed (* rows half-cols)
         data-bytes (byte-array total-packed)
         target-scale-format (get opts :as :f32)
         raw-group-size (get opts :group-size)
         group-size (when (and raw-group-size (pos? (long raw-group-size)) (zero? (mod cols (long raw-group-size))))
                      (long raw-group-size))
         num-groups (if group-size (quot cols group-size) 1)
         total-scales (* rows num-groups)
         scales-floats (when (= target-scale-format :f32) (float-array total-scales))
         scales-shorts (when (= target-scale-format :bf16) (short-array total-scales))
         is-floats? (instance? (Class/forName "[F") w-arr)
         is-doubles? (instance? (Class/forName "[D") w-arr)
         is-shorts? (instance? (Class/forName "[S") w-arr)
         f-arr (when is-floats? ^floats w-arr)
         d-arr (when is-doubles? ^doubles w-arr)
         s-arr (when is-shorts? ^shorts w-arr)
         get-val (fn ^double [^long idx]
                   (cond
                     is-floats? (double (aget f-arr idx))
                     is-doubles? (aget d-arr idx)
                     is-shorts? (let [s (int (aget s-arr idx))
                                      bits (unchecked-int (bit-shift-left (long (bit-and s 0xffff)) 16))]
                                  (double (Float/intBitsToFloat bits)))
                     :else (double (nth w-arr idx))))]
     (if group-size
       ;; Block-wise INT4 quantization with per-block optimal MSE scale search
       (let [half-g (quot group-size 2)]
         (-> (IntStream/range 0 rows)
             (.parallel)
             (.forEach
              (reify IntConsumer
                (accept [_ r]
                  (let [r-base (* r cols)
                        out-base (* r half-cols)
                        s-base (* r num-groups)]
                    (dotimes [g num-groups]
                      (let [g-base (+ r-base (* g group-size))
                            g-out-base (+ out-base (* g half-g))
                            ;; 1. Max absolute value in block
                            max-abs (loop [i 0 m 0.0]
                                      (if (>= i group-size)
                                        m
                                        (let [v (Math/abs (get-val (+ g-base i)))]
                                          (recur (inc i) (max m v)))))
                            base-scale (/ max-abs 7.0)
                            ;; 2. 10-step optimal scale search (minimizing MSE against outliers)
                            opt-mult (if (zero? max-abs)
                                       1.0
                                       (loop [step 0 best-m 1.0 best-err Double/MAX_VALUE]
                                         (if (>= step 10)
                                           best-m
                                           (let [m (+ 0.65 (* step 0.035))
                                                 cand-s (* base-scale m)
                                                 inv-s (/ 1.0 cand-s)
                                                 err (loop [i 0 acc 0.0]
                                                       (if (>= i group-size)
                                                         acc
                                                         (let [v (get-val (+ g-base i))
                                                               q (+ (max -7 (min 7 (Math/round (* v inv-s)))) 8)
                                                               rec (* (double (- q 8)) cand-s)
                                                               diff (- v rec)]
                                                           (recur (inc i) (+ acc (* diff diff))))))]
                                             (if (< err best-err)
                                               (recur (inc step) m err)
                                               (recur (inc step) best-m best-err))))))
                            scale (if (zero? max-abs) (float 1.0) (float (* base-scale opt-mult)))
                            inv-scale (/ 1.0 (double scale))]
                        (if (= target-scale-format :bf16)
                          (aset-short scales-shorts (+ s-base g) (float->bf16-short scale))
                          (aset-float scales-floats (+ s-base g) scale))
                        ;; 3. Pack 4-bit pairs
                        (dotimes [hc half-g]
                          (let [idx0 (+ g-base (* hc 2))
                                idx1 (inc idx0)
                                v0 (get-val idx0)
                                v1 (get-val idx1)
                                q0 (+ (max -7 (min 7 (Math/round (* v0 inv-scale)))) 8)
                                q1 (+ (max -7 (min 7 (Math/round (* v1 inv-scale)))) 8)
                                packed (unchecked-byte (bit-or (bit-and (int q0) 0x0F)
                                                               (bit-shift-left (bit-and (int q1) 0x0F) 4)))]
                            (aset-byte data-bytes (+ g-out-base hc) packed)))))))))))
       ;; Legacy per-row INT4 quantization
       (-> (IntStream/range 0 rows)
           (.parallel)
           (.forEach
            (reify IntConsumer
              (accept [_ r]
                (let [r-base (* r cols)
                      out-base (* r half-cols)
                      max-abs (loop [c 0 m 0.0]
                                (if (>= c cols)
                                  m
                                  (let [v (Math/abs (get-val (+ r-base c)))]
                                    (recur (inc c) (max m v)))))
                      scale (if (zero? max-abs) (float 1.0) (float (/ max-abs 7.0)))
                      inv-scale (/ 1.0 (double scale))]
                  (if (= target-scale-format :bf16)
                    (aset-short scales-shorts r (float->bf16-short scale))
                    (aset-float scales-floats r scale))
                  (dotimes [hc half-cols]
                    (let [idx0 (+ r-base (* hc 2))
                          idx1 (inc idx0)
                          v0 (get-val idx0)
                          v1 (get-val idx1)
                          q0 (+ (max -7 (min 7 (Math/round (* v0 inv-scale)))) 8)
                          q1 (+ (max -7 (min 7 (Math/round (* v1 inv-scale)))) 8)
                          packed (unchecked-byte (bit-or (bit-and (int q0) 0x0F)
                                                         (bit-shift-left (bit-and (int q1) 0x0F) 4)))]
                      (aset-byte data-bytes (+ out-base hc) packed)))))))))
     {:data data-bytes
      :scales (if (= target-scale-format :bf16) scales-shorts scales-floats)
      :shape [rows half-cols]
      :scale-shape (if group-size [rows num-groups] [rows])})))
