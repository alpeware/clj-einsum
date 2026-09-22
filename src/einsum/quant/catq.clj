(ns einsum.quant.catq
  "CAT-Q: Cost-Efficient Accurate Ternary Quantization with Learnable Modulation
   and LoRA Refinement. Pure mathematical quantization primitives, calibration routines,
   and zero-copy Safetensors model export."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [einsum.runtime.safetensors :as st])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.nio ByteBuffer]
           [java.nio.channels FileChannel]
           [java.nio.file Path StandardOpenOption]
           [java.util Random]
           [java.util.function IntConsumer]
           [java.util.stream IntStream]))

(set! *warn-on-reflection* true)

;; ==============================================================================
;; 1. Learnable Modulation Mathematical Primitives (Pure, Sans-IO)
;; ==============================================================================

(defn sigmoid
  "Numerically stable standard logistic sigmoid: 1 / (1 + e^(-x))."
  ^double [^double x]
  (if (>= x 0.0)
    (let [z (Math/exp (- x))]
      (/ 1.0 (+ 1.0 z)))
    (let [z (Math/exp x)]
      (/ z (+ 1.0 z)))))

(defn logit
  "Numerically clamped logit function: ln(p / (1 - p))."
  ^double [^double p]
  (let [p-clamped (Math/max 1e-6 (Math/min (- 1.0 1e-6) p))]
    (Math/log (/ p-clamped (- 1.0 p-clamped)))))

(defn compute-modulated-scale
  "Computes CAT-Q learnable modulated scale: s_g = s_base * (2 * sigmoid(z_s)).
   When z_s = 0, scale matches base scale exactly."
  ^double [^double base-scale ^double zs]
  (* base-scale 2.0 (sigmoid zs)))

(defn compute-modulated-threshold
  "Computes CAT-Q learnable modulated threshold: Delta_g = 0.5 * s_mod * (2 * sigmoid(z_r)).
   When z_r = 0, threshold matches 0.5 * s_mod exactly."
  ^double [^double s-mod ^double zr]
  (* 0.5 s-mod 2.0 (sigmoid zr)))

;; ==============================================================================
;; 2. LoRA Effective Weights Computation (Pure, Sans-IO)
;; ==============================================================================

(defn compute-effective-weights
  "Computes LoRA effective weights W_eff = W + (alpha / rank) * (B * A).
   w: float array of shape [rows cols]
   lora-a: float array of shape [rank cols]
   lora-b: float array of shape [rows rank]
   rank: int
   alpha: double (default 1.0)
   Returns a float-array of size (* rows cols)."
  ([w rows cols lora-a lora-b rank]
   (compute-effective-weights w rows cols lora-a lora-b rank 1.0))
  ([w rows cols lora-a lora-b rank alpha]
   (let [rows (int rows)
         cols (int cols)
         rank (int rank)
         alpha (double alpha)
         scaling (float (/ alpha (double rank)))
         total (* rows cols)
         out ^floats (float-array total)
         w-arr ^floats (if (instance? (Class/forName "[F") w) w (float-array (map float w)))]
     (if (or (nil? lora-b) (nil? lora-a) (zero? rank))
       (do (System/arraycopy w-arr 0 out 0 total) out)
       (let [a-arr ^floats (if (instance? (Class/forName "[F") lora-a) lora-a (float-array (map float lora-a)))
             b-arr ^floats (if (instance? (Class/forName "[F") lora-b) lora-b (float-array (map float lora-b)))]
         (-> (IntStream/range 0 rows)
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (let [r (int r)
                                 ^floats w-arr w-arr
                                 ^floats a-arr a-arr
                                 ^floats b-arr b-arr
                                 ^floats out out
                                 r-base (* r cols)
                                 b-base (* r rank)]
                             (dotimes [c cols]
                               (let [sum (loop [k (int 0) acc (float 0.0)]
                                           (if (< k rank)
                                             (recur (unchecked-inc k)
                                                    (+ acc (* (aget b-arr (+ b-base k))
                                                              (aget a-arr (+ (* k cols) c)))))
                                             acc))]
                                 (aset-float out (+ r-base c) (+ (aget w-arr (+ r-base c)) (* scaling sum))))))))))
         out)))))

;; ==============================================================================
;; 3. Binary & Ternary Bit-Packing Helpers (Q2_0 Format)
;; ==============================================================================

(defn float->bf16-short
  [val]
  (let [bits (Float/floatToRawIntBits (float val))]
    (unchecked-short (bit-shift-right bits 16))))

(defn bf16-short->float
  [s]
  (let [bits (unchecked-int (bit-shift-left (bit-and (int s) 0xffff) 16))]
    (Float/intBitsToFloat bits)))

(defn decode-ternary-2bit
  "Decodes a 2-bit unsigned code (0..3) into ternary float {-1.0, 0.0, 1.0}."
  ^double [^long code]
  (case (int (bit-and code 3))
    0 -1.0
    1  0.0
    2  1.0
    3  0.0))

(defn quantize-matrix-catq
  "Quantizes weight matrix of shape [rows cols] into 2-bit packed bytes (4 weights per byte)
   and per-group BF16 scales using CAT-Q learnable modulation parameters.
   Opts:
     :group-size - block size (default 128)
     :zs - optional double array or sequence of scale modulation factors of size (* rows num-groups)
     :zr - optional double array or sequence of round modulation factors of size (* rows num-groups)
   Returns:
     {:data ^bytes :scales ^shorts :shape [rows quarter-cols] :scale-shape [rows num-groups]}."
  ([w rows cols]
   (quantize-matrix-catq w rows cols {}))
  ([w rows cols opts]
   (let [rows (int rows)
         cols (int cols)
         group-size (int (get opts :group-size 128))
         _ (assert (zero? (mod cols group-size)) (str "cols must be divisible by group-size: " cols " vs " group-size))
         num-groups (int (quot cols group-size))
         quarter-cols (int (quot cols 4))
         quarter-g (int (quot group-size 4))
         data ^bytes (byte-array (* rows quarter-cols))
         scales ^shorts (short-array (* rows num-groups))
         w-arr ^floats (if (instance? (Class/forName "[F") w) w (float-array (map float w)))
         zs-arr ^doubles (when-let [zs (:zs opts)]
                           (if (instance? (Class/forName "[D") zs) zs (double-array (map double zs))))
         zr-arr ^doubles (when-let [zr (:zr opts)]
                           (if (instance? (Class/forName "[D") zr) zr (double-array (map double zr))))]
     (-> (IntStream/range 0 rows)
         (.parallel)
         (.forEach (reify IntConsumer
                     (accept [_ r]
                       (let [r (int r)
                             ^floats w-arr w-arr
                             ^bytes data data
                             ^shorts scales scales
                             ^doubles zs-arr zs-arr
                             ^doubles zr-arr zr-arr
                             r-base (* r cols)
                             r-out-base (* r quarter-cols)
                             r-scale-base (* r num-groups)]
                         (dotimes [g num-groups]
                           (let [g-base (+ r-base (* g group-size))
                                 g-out-base (+ r-out-base (* g quarter-g))
                                 scale-idx (+ r-scale-base g)
                                 sum-abs (loop [i (int 0) acc 0.0]
                                           (if (< i group-size)
                                             (recur (unchecked-inc i) (+ acc (Math/abs (double (aget w-arr (+ g-base i))))))
                                             acc))
                                 base-scale (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)
                                 zs (if zs-arr (aget zs-arr scale-idx) 0.0)
                                 zr (if zr-arr (aget zr-arr scale-idx) 0.0)
                                 s-mod (compute-modulated-scale base-scale zs)
                                 thd-mod (compute-modulated-threshold s-mod zr)
                                 inv-thd (if (pos? thd-mod) (/ 1.0 thd-mod) 1.0)]
                             (aset-short scales (int scale-idx) (short (float->bf16-short (float s-mod))))
                             (dotimes [k quarter-g]
                               (let [idx (+ g-base (* k 4))
                                     v0 (* (double (aget w-arr idx)) inv-thd)
                                     v1 (* (double (aget w-arr (+ idx 1))) inv-thd)
                                     v2 (* (double (aget w-arr (+ idx 2))) inv-thd)
                                     v3 (* (double (aget w-arr (+ idx 3))) inv-thd)
                                     c0 (int (cond (>= v0 1.0) 2 (<= v0 -1.0) 0 :else 1))
                                     c1 (int (cond (>= v1 1.0) 2 (<= v1 -1.0) 0 :else 1))
                                     c2 (int (cond (>= v2 1.0) 2 (<= v2 -1.0) 0 :else 1))
                                     c3 (int (cond (>= v3 1.0) 2 (<= v3 -1.0) 0 :else 1))
                                     b (unchecked-byte (bit-or c0 (bit-shift-left c1 2) (bit-shift-left c2 4) (bit-shift-left c3 6)))]
                                 (aset-byte data (+ g-out-base k) b))))))))))
     {:data data
      :scales scales
      :shape [rows quarter-cols]
      :scale-shape [rows num-groups]})))

(defn dequantize-matrix-catq
  "Dequantizes 2-bit packed bytes into continuous weights scaled by per-group BF16/FP32 scales.
   Returns a float-array of size (* rows cols)."
  [data scales rows cols group-size]
  (let [rows (int rows)
        cols (int cols)
        group-size (int group-size)
        num-groups (int (quot cols group-size))
        quarter-cols (int (quot cols 4))
        out ^floats (float-array (* rows cols))
        data-bytes ^bytes data
        is-shorts? (instance? (Class/forName "[S") scales)
        scales-shorts ^shorts (when is-shorts? scales)
        scales-floats ^floats (when-not is-shorts? scales)]
    (-> (IntStream/range 0 rows)
        (.parallel)
        (.forEach (reify IntConsumer
                    (accept [_ r]
                      (let [r (int r)
                            ^floats out out
                            ^bytes data-bytes data-bytes
                            ^shorts scales-shorts scales-shorts
                            ^floats scales-floats scales-floats
                            r-base (* r cols)
                            r-data-base (* r quarter-cols)
                            r-scale-base (* r num-groups)]
                        (dotimes [c cols]
                          (let [g (quot c group-size)
                                s (if is-shorts?
                                    (bf16-short->float (aget scales-shorts (+ r-scale-base g)))
                                    (aget scales-floats (+ r-scale-base g)))
                                byte-idx (+ r-data-base (quot c 4))
                                shift (* (mod c 4) 2)
                                b (int (aget data-bytes byte-idx))
                                code (bit-and (bit-shift-right b shift) 3)
                                v (* (decode-ternary-2bit code) (double s))]
                            (aset-float out (+ r-base c) (float v)))))))))
    out))

(defn floats->bf16-shorts
  "Converts float array to short array of bfloat16 bit-patterns."
  ^shorts [^floats fa]
  (let [n (alength fa)
        sa (short-array n)]
    (dotimes [i n]
      (aset-short sa i (float->bf16-short (aget fa i))))
    sa))

;; ==============================================================================
;; 4. Optimal Brain Surgeon (OBS / GPTQ) Calibration Engine
;; ==============================================================================

(defn invert-spd-128
  "Inverts a positive definite matrix of size n x n (e.g. n=128) using Cholesky LDL^T."
  [^doubles A n]
  (let [n (int n)
        inv ^doubles (double-array (* n n))
        L ^doubles (double-array (* n n))]
    ;; Cholesky L L^T = A
    (dotimes [i n]
      (dotimes [j (inc i)]
        (let [sum (loop [k (int 0) acc 0.0]
                    (if (< k j)
                      (recur (unchecked-inc k) (+ acc (* (aget L (+ (* i n) k)) (aget L (+ (* j n) k)))))
                      acc))]
          (if (= i j)
            (let [v (- (aget A (+ (* i n) i)) sum)]
              (aset-double L (+ (* i n) j) (Math/sqrt (Math/max 1e-10 v))))
            (let [v (/ (- (aget A (+ (* i n) j)) sum) (aget L (+ (* j n) j)))]
              (aset-double L (+ (* i n) j) v))))))
    ;; Invert L: L_inv
    (let [Linv ^doubles (double-array (* n n))]
      (dotimes [i n]
        (aset-double Linv (+ (* i n) i) (/ 1.0 (aget L (+ (* i n) i))))
        (loop [j (int 0)]
          (when (< j i)
            (let [sum (loop [k j acc 0.0]
                        (if (< k i)
                          (recur (unchecked-inc k) (+ acc (* (aget L (+ (* i n) k)) (aget Linv (+ (* k n) j)))))
                          acc))]
              (aset-double Linv (+ (* i n) j) (/ (- sum) (aget L (+ (* i n) i)))))
            (recur (unchecked-inc j)))))
      ;; inv = Linv^T * Linv
      (dotimes [i n]
        (dotimes [j n]
          (let [sum (loop [k (int 0) acc 0.0]
                      (if (< k n)
                        (recur (unchecked-inc k) (+ acc (* (aget Linv (+ (* k n) i)) (aget Linv (+ (* k n) j)))))
                        acc))]
            (aset-double inv (+ (* i n) j) sum)))))
    inv))

(defn quantize-matrix-obs
  "Quantizes weight matrix [rows cols] using block-diagonal Hessian (group size 128)
   and Hessian-conditioned least-squares scale with learnable modulation factors.
   Parallelized across all CPU cores with zero-boxing primitive arrays.
   Returns:
     {:data ^bytes :scales ^shorts :w-deq ^floats :shape [rows quarter-cols]
      :scale-shape [rows num-groups] :zs ^doubles :zr ^doubles}."
  ([x-act w-floats num-tokens rows cols group-size]
   (quantize-matrix-obs x-act w-floats num-tokens rows cols group-size nil))
  ([x-act w-floats num-tokens rows cols group-size opts]
   (let [w-arr ^floats (if (instance? (Class/forName "[F") w-floats) w-floats (float-array (map float w-floats)))
         rows (int rows)
         cols (int cols)
         num-tokens (int (or num-tokens 0))
         group-size (int group-size)
         num-groups (int (quot cols group-size))
         quarter-cols (int (quot cols 4))
         quarter-g (int (quot group-size 4))
         data ^bytes (byte-array (* rows quarter-cols))
         scales ^shorts (short-array (* rows num-groups))
         w-deq ^floats (float-array (* rows cols))
         zs-arr ^doubles (double-array (* rows num-groups))
         zr-arr ^doubles (double-array (* rows num-groups))
         gamma-cands-raw (or (:gamma-candidates opts) [0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70])
         ^doubles gamma-cands (if (instance? (Class/forName "[D") gamma-cands-raw)
                                ^doubles gamma-cands-raw
                                (double-array (map double gamma-cands-raw)))
         num-cands (alength gamma-cands)
         has-acts? (and (some? x-act) (pos? num-tokens))
         ^floats x-arr (when has-acts?
                         (if (instance? (Class/forName "[F") x-act) x-act (float-array (map float x-act))))]

     (if-not has-acts?
       ;; 1. Weight-space analytic least-squares quantization (Parallelized across rows)
       (-> (IntStream/range 0 rows)
           (.parallel)
           (.forEach (reify IntConsumer
                       (accept [_ r]
                         (let [r (int r)
                               ^floats w-arr w-arr
                               ^doubles gamma-cands gamma-cands
                               r-base (* r cols)
                               r-data-base (* r quarter-cols)
                               r-scale-base (* r num-groups)
                               best-t ^doubles (double-array group-size)
                               t-cur ^doubles (double-array group-size)
                               t-ref ^doubles (double-array group-size)]
                           (dotimes [g num-groups]
                             (let [g (int g)
                                   g-base (+ r-base (* g group-size))
                                   g-out-base (+ r-data-base (* g quarter-g))
                                   s-idx (+ r-scale-base g)
                                   sum-abs (loop [i (int 0) acc 0.0]
                                             (if (< i group-size)
                                               (recur (unchecked-inc i) (+ acc (Math/abs (double (aget w-arr (+ g-base i))))))
                                               acc))
                                   s-base (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)]
                               (loop [cand-idx (int 0)
                                      best-gamma 0.5
                                      best-s (double s-base)
                                      best-score Double/NEGATIVE_INFINITY]
                                 (if (< cand-idx num-cands)
                                   (let [gamma (double (aget gamma-cands cand-idx))
                                         delta (* gamma s-base)]
                                     (dotimes [i group-size]
                                       (let [wi (double (aget w-arr (+ g-base i)))]
                                         (aset-double t-cur i (cond (>= wi delta) 1.0 (<= wi (- delta)) -1.0 :else 0.0))))
                                     (let [[wt-dot tt-dot]
                                           (loop [i (int 0) w-dot 0.0 t-dot 0.0]
                                             (if (< i group-size)
                                               (let [wi (double (aget w-arr (+ g-base i)))
                                                     ti (double (aget t-cur i))]
                                                 (recur (unchecked-inc i) (+ w-dot (* wi ti)) (+ t-dot (* ti ti))))
                                               [w-dot t-dot]))
                                           s-opt (double (if (> (double tt-dot) 1e-12)
                                                           (Math/max (* 0.5 s-base) (Math/min (* 2.0 s-base) (/ (double wt-dot) (double tt-dot))))
                                                           s-base))
                                           mid (* 0.5 s-opt)]
                                       (dotimes [i group-size]
                                         (let [wi (double (aget w-arr (+ g-base i)))]
                                           (aset-double t-ref i (cond (>= wi mid) 1.0 (<= wi (- mid)) -1.0 :else 0.0))))
                                       (let [[wt2 tt2]
                                             (loop [i (int 0) w-dot 0.0 t-dot 0.0]
                                               (if (< i group-size)
                                                 (let [wi (double (aget w-arr (+ g-base i)))
                                                       ti (double (aget t-ref i))]
                                                   (recur (unchecked-inc i) (+ w-dot (* wi ti)) (+ t-dot (* ti ti))))
                                                 [w-dot t-dot]))
                                             s2 (double (if (> (double tt2) 1e-12)
                                                          (Math/max (* 0.5 s-base) (Math/min (* 2.0 s-base) (/ (double wt2) (double tt2))))
                                                          s-opt))
                                             score (- (* 2.0 s2 (double wt2)) (* s2 s2 (double tt2)))]
                                         (if (> score best-score)
                                           (do
                                             (System/arraycopy t-ref 0 best-t 0 group-size)
                                             (recur (unchecked-inc cand-idx) gamma s2 score))
                                           (recur (unchecked-inc cand-idx) best-gamma best-s best-score)))))
                                   ;; Final write
                                   (let [s-short (short (float->bf16-short (float best-s)))
                                         deq-s (float (bf16-short->float s-short))
                                         final-gamma (double best-gamma)]
                                     (dotimes [i group-size]
                                       (aset-float w-deq (+ g-base i) (* (float (aget best-t i)) deq-s)))
                                     (aset-short scales s-idx s-short)
                                     (aset-double zs-arr s-idx (logit (/ (double deq-s) (* 2.0 (Math/max 1e-6 s-base)))))
                                     (aset-double zr-arr s-idx (logit final-gamma))
                                     (dotimes [k quarter-g]
                                       (let [idx (* (int k) 4)
                                             c0 (int (+ (aget best-t idx) 1.0))
                                             c1 (int (+ (aget best-t (+ idx 1)) 1.0))
                                             c2 (int (+ (aget best-t (+ idx 2)) 1.0))
                                             c3 (int (+ (aget best-t (+ idx 3)) 1.0))
                                             b (unchecked-byte (bit-or c0 (bit-shift-left c1 2) (bit-shift-left c2 4) (bit-shift-left c3 6)))]
                                         (aset-byte data (+ g-out-base (int k)) b)))))))))))))

       ;; 2. Hessian-conditioned quantization on input activations X (Parallelized across groups and rows)
       (let [H-mats ^"[[D" (make-array (Class/forName "[D") num-groups)]
         ;; 2a. Precompute H per group in parallel
         (-> (IntStream/range 0 num-groups)
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ g]
                           (let [g (int g)
                                 ^floats x-arr x-arr
                                 g-col-base (* g group-size)
                                 H ^doubles (double-array (* group-size group-size))]
                             (dotimes [i group-size]
                               (dotimes [j group-size]
                                 (let [ci (+ g-col-base (int i))
                                       cj (+ g-col-base (int j))
                                       sum (loop [t (int 0) acc 0.0]
                                             (if (< t num-tokens)
                                               (let [t-base (* t cols)]
                                                 (recur (unchecked-inc t)
                                                        (+ acc (* (double (aget x-arr (+ t-base ci)))
                                                                  (double (aget x-arr (+ t-base cj)))))))
                                               acc))]
                                   (aset-double H (+ (* (int i) group-size) (int j)) sum))))
                             (let [trace (loop [i (int 0) acc 0.0]
                                           (if (< i group-size)
                                             (recur (unchecked-inc i) (+ acc (aget H (+ (* (int i) group-size) (int i)))))
                                             acc))
                                   scale-H (/ (double group-size) (Math/max 1e-8 (double trace)))]
                               (dotimes [k (* group-size group-size)]
                                 (aset-double H (int k) (* (aget H (int k)) scale-H))))
                             (aset H-mats g H))))))

         ;; 2b. Quantize each row in parallel
         (-> (IntStream/range 0 rows)
             (.parallel)
             (.forEach (reify IntConsumer
                         (accept [_ r]
                           (let [r (int r)
                                 ^floats w-arr w-arr
                                 ^doubles gamma-cands gamma-cands
                                 ^"[[D" H-mats H-mats
                                 r-base (* r cols)
                                 r-data-base (* r quarter-cols)
                                 r-scale-base (* r num-groups)
                                 best-t ^doubles (double-array group-size)
                                 t-cur ^doubles (double-array group-size)
                                 t-ref ^doubles (double-array group-size)
                                 Ht ^doubles (double-array group-size)]
                             (dotimes [g num-groups]
                               (let [g (int g)
                                     g-base (+ r-base (* g group-size))
                                     g-out-base (+ r-data-base (* g quarter-g))
                                     s-idx (+ r-scale-base g)
                                     ^doubles H (aget H-mats g)
                                     sum-abs (loop [i (int 0) acc 0.0]
                                               (if (< i group-size)
                                                 (recur (unchecked-inc i) (+ acc (Math/abs (double (aget w-arr (+ g-base i))))))
                                                 acc))
                                     s-base (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)]
                                 (loop [cand-idx (int 0)
                                        best-gamma 0.5
                                        best-s (double s-base)
                                        best-score Double/NEGATIVE_INFINITY]
                                   (if (< cand-idx num-cands)
                                     (let [gamma (double (aget gamma-cands cand-idx))
                                           delta (* gamma s-base)]
                                       (dotimes [i group-size]
                                         (let [wi (double (aget w-arr (+ g-base i)))]
                                           (aset-double t-cur (int i) (cond (>= wi delta) 1.0 (<= wi (- delta)) -1.0 :else 0.0))))
                                       ;; H * t_cur
                                       (dotimes [i group-size]
                                         (let [row-idx (* (int i) group-size)
                                               sum (loop [j (int 0) acc 0.0]
                                                     (if (< j group-size)
                                                       (recur (unchecked-inc j) (+ acc (* (aget H (+ row-idx j)) (aget t-cur j))))
                                                       acc))]
                                           (aset-double Ht (int i) sum)))
                                       (let [wt-dot (double (loop [i (int 0) acc 0.0]
                                                              (if (< i group-size)
                                                                (recur (unchecked-inc i) (+ acc (* (double (aget w-arr (+ g-base i))) (aget Ht i))))
                                                                acc)))
                                             tt-dot (double (loop [i (int 0) acc 0.0]
                                                              (if (< i group-size)
                                                                (recur (unchecked-inc i) (+ acc (* (aget t-cur i) (aget Ht i))))
                                                                acc)))
                                             s-opt (double (if (> tt-dot 1e-12)
                                                             (Math/max (* 0.5 s-base) (Math/min (* 2.0 s-base) (/ wt-dot tt-dot)))
                                                             s-base))
                                             mid (* 0.5 s-opt)]
                                         ;; Refinement with midpoint threshold
                                         (dotimes [i group-size]
                                           (let [wi (double (aget w-arr (+ g-base i)))]
                                             (aset-double t-ref (int i) (cond (>= wi mid) 1.0 (<= wi (- mid)) -1.0 :else 0.0))))
                                         ;; H * t_ref
                                         (dotimes [i group-size]
                                           (let [row-idx (* (int i) group-size)
                                                 sum (loop [j (int 0) acc 0.0]
                                                       (if (< j group-size)
                                                         (recur (unchecked-inc j) (+ acc (* (aget H (+ row-idx j)) (aget t-ref j))))
                                                         acc))]
                                             (aset-double Ht (int i) sum)))
                                         (let [wt2 (double (loop [i (int 0) acc 0.0]
                                                             (if (< i group-size)
                                                               (recur (unchecked-inc i) (+ acc (* (double (aget w-arr (+ g-base i))) (aget Ht i))))
                                                               acc)))
                                               tt2 (double (loop [i (int 0) acc 0.0]
                                                             (if (< i group-size)
                                                               (recur (unchecked-inc i) (+ acc (* (aget t-ref i) (aget Ht i))))
                                                               acc)))
                                               s2 (double (if (> tt2 1e-12)
                                                            (Math/max (* 0.5 s-base) (Math/min (* 2.0 s-base) (/ wt2 tt2)))
                                                            s-opt))
                                               score (- (* 2.0 s2 wt2) (* s2 s2 tt2))]
                                           (if (> score best-score)
                                             (do
                                               (System/arraycopy t-ref 0 best-t 0 group-size)
                                               (recur (unchecked-inc cand-idx) gamma s2 score))
                                             (recur (unchecked-inc cand-idx) best-gamma best-s best-score)))))
                                     ;; Final write
                                     (let [s-short (short (float->bf16-short (float best-s)))
                                           deq-s (float (bf16-short->float s-short))
                                           final-gamma (double best-gamma)]
                                       (dotimes [i group-size]
                                         (aset-float w-deq (+ g-base i) (* (float (aget best-t i)) deq-s)))
                                       (aset-short scales s-idx s-short)
                                       (aset-double zs-arr s-idx (logit (/ (double deq-s) (* 2.0 (Math/max 1e-6 s-base)))))
                                       (aset-double zr-arr s-idx (logit final-gamma))
                                       (dotimes [k quarter-g]
                                         (let [idx (* (int k) 4)
                                               c0 (int (+ (aget best-t idx) 1.0))
                                               c1 (int (+ (aget best-t (+ idx 1)) 1.0))
                                               c2 (int (+ (aget best-t (+ idx 2)) 1.0))
                                               c3 (int (+ (aget best-t (+ idx 3)) 1.0))
                                               b (unchecked-byte (bit-or c0 (bit-shift-left c1 2) (bit-shift-left c2 4) (bit-shift-left c3 6)))]
                                           (aset-byte data (+ g-out-base (int k)) b)))))))))))))))

     {:data data
      :scales scales
      :w-deq w-deq
      :shape [rows quarter-cols]
      :scale-shape [rows num-groups]
      :zs zs-arr
      :zr zr-arr})))

;; ==============================================================================
;; 5. Projection Calibration Engine (Learnable Modulation + LoRA)
;; ==============================================================================

(defn compute-projection-mse
  "Computes projection MSE: (1 / (num-tokens * rows)) * || X * W^T - X * W_deq^T ||_F^2
   x: float array of shape [num-tokens cols]
   w: float array of shape [rows cols]
   w-deq: float array of shape [rows cols]"
  [x w w-deq num-tokens rows cols]
  (let [num-tokens (int num-tokens)
        eval-tokens (int (min 32 num-tokens))
        rows (int rows)
        cols (int cols)
        ^floats x-arr (if (instance? (Class/forName "[F") x) x (float-array (map float x)))
        ^floats w-arr (if (instance? (Class/forName "[F") w) w (float-array (map float w)))
        ^floats d-arr (if (instance? (Class/forName "[F") w-deq) w-deq (float-array (map float w-deq)))
        total-w (* rows cols)
        ^floats w-diff (float-array total-w)]
    (dotimes [i total-w]
      (aset-float w-diff i (- (aget w-arr i) (aget d-arr i))))
    (let [sum-sq (-> (IntStream/range 0 rows)
                     (.parallel)
                     (.mapToDouble (reify java.util.function.IntToDoubleFunction
                                     (applyAsDouble [_ r]
                                       (let [r (int r)
                                             r-base (* r cols)
                                             ^floats x-arr x-arr
                                             ^floats w-diff w-diff]
                                         (loop [t (int 0) acc-t 0.0]
                                           (if (< t eval-tokens)
                                             (let [t-base (* t cols)
                                                   diff (loop [c (int 0) acc-c 0.0]
                                                          (if (< c cols)
                                                            (recur (unchecked-inc c)
                                                                   (+ acc-c (* (double (aget x-arr (+ t-base c)))
                                                                               (double (aget w-diff (+ r-base c))))))
                                                            acc-c))]
                                               (recur (unchecked-inc t) (+ acc-t (* diff diff))))
                                             acc-t))))))
                     (.sum))]
      (/ sum-sq (double (* eval-tokens rows))))))

(defn calibrate-projection
  "Calibrates learnable modulation factors (z_s, z_r) and LoRA refinement (A, B)
   to minimize projection reconstruction MSE on input activations X."
  [x w num-tokens rows cols opts]
  (let [rows (int rows)
        cols (int cols)
        num-tokens (int num-tokens)
        group-size (int (get opts :group-size 128))
        num-groups (int (quot cols group-size))
        total-scales (* rows num-groups)
        rank (int (get opts :rank 16))
        epochs (int (get opts :epochs 15))
        alpha (double (get opts :alpha 1.0))
        w-arr ^floats (if (instance? (Class/forName "[F") w) w (float-array (map float w)))
        zs ^doubles (double-array total-scales 0.0)
        zr ^doubles (double-array total-scales 0.0)
        gamma-candidates ^doubles (double-array [0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70])
        num-cands (alength gamma-candidates)]
    ;; 1. Analytic least-squares scale and optimal threshold search per group (Parallelized across rows!)
    (-> (IntStream/range 0 rows)
        (.parallel)
        (.forEach (reify IntConsumer
                    (accept [_ r]
                      (let [r (int r)
                            ^floats w-arr w-arr
                            ^doubles zs zs
                            ^doubles zr zr
                            r-base (* r cols)
                            r-scale-base (* r num-groups)]
                        (dotimes [g num-groups]
                          (let [g-base (+ r-base (* g group-size))
                                s-idx (+ r-scale-base g)
                                sum-abs (loop [i (int 0) acc 0.0]
                                          (if (< i group-size)
                                            (recur (unchecked-inc i) (+ acc (Math/abs (double (aget w-arr (+ g-base i))))))
                                            acc))
                                base-scale (if (pos? group-size) (/ sum-abs (double group-size)) 0.0)
                                [best-gamma best-scale _best-err]
                                (loop [cand-idx (int 0)
                                       b-gamma 0.5
                                       b-scale base-scale
                                       b-err Double/MAX_VALUE]
                                  (if (< cand-idx num-cands)
                                    (let [gamma (aget gamma-candidates cand-idx)
                                          delta (* gamma base-scale)
                                          inv-thd (if (pos? delta) (/ 1.0 delta) 1.0)
                                          [w-dot-t t-dot-t]
                                          (loop [i (int 0) dot 0.0 cnt 0.0]
                                            (if (< i group-size)
                                              (let [wi (double (aget w-arr (+ g-base i)))
                                                    v (* wi inv-thd)
                                                    ti (cond (>= v 1.0) 1.0 (<= v -1.0) -1.0 :else 0.0)]
                                                (recur (unchecked-inc i) (+ dot (* wi ti)) (+ cnt (* ti ti))))
                                              [dot cnt]))
                                          opt-s (if (pos? (double t-dot-t))
                                                  (/ (double w-dot-t) (double t-dot-t))
                                                  base-scale)
                                          err (loop [i (int 0) acc 0.0]
                                                (if (< i group-size)
                                                  (let [wi (double (aget w-arr (+ g-base i)))
                                                        v (* wi inv-thd)
                                                        ti (cond (>= v 1.0) 1.0 (<= v -1.0) -1.0 :else 0.0)
                                                        diff (- wi (* opt-s ti))]
                                                    (recur (unchecked-inc i) (+ acc (* diff diff))))
                                                  acc))]
                                      (if (< (double err) (double b-err))
                                        (recur (unchecked-inc cand-idx) (double gamma) (double opt-s) (double err))
                                        (recur (unchecked-inc cand-idx) b-gamma b-scale b-err)))
                                    [b-gamma b-scale b-err]))]
                            (aset-double zs s-idx (logit (/ (double best-scale) (* 2.0 (Math/max 1e-6 (double base-scale))))))
                            (aset-double zr s-idx (logit (double best-gamma))))))))))

    ;; 2. LoRA Refinement via projection error minimization
    (let [lora-a ^floats (float-array (* rank cols) 0.0)
          lora-b ^floats (float-array (* rows rank) 0.0)
          rnd (Random. 42)]
      (dotimes [i (* rank cols)]
        (aset-float lora-a i (float (* 0.01 (.nextGaussian rnd)))))
      (let [best-state (atom {:mse Double/MAX_VALUE :rec nil})
            init-rec (quantize-matrix-catq w-arr rows cols {:group-size group-size :zs zs :zr zr})
            init-deq ^floats (dequantize-matrix-catq (:data init-rec) (:scales init-rec) rows cols group-size)
            init-mse (compute-projection-mse x w init-deq num-tokens rows cols)]
        (reset! best-state {:mse init-mse :rec init-rec :lora-a (aclone lora-a) :lora-b (aclone lora-b)})

        (loop [ep (int 0)]
          (when (< ep epochs)
            (let [w-eff ^floats (compute-effective-weights w-arr rows cols lora-a lora-b rank alpha)
                  cur-rec (quantize-matrix-catq w-eff rows cols {:group-size group-size :zs zs :zr zr})
                  cur-deq ^floats (dequantize-matrix-catq (:data cur-rec) (:scales cur-rec) rows cols group-size)
                  cur-mse (compute-projection-mse x w cur-deq num-tokens rows cols)]
              (when (< cur-mse (:mse @best-state))
                (reset! best-state {:mse cur-mse :rec cur-rec :lora-a (aclone lora-a) :lora-b (aclone lora-b)}))
              ;; Gradient step on B (Parallelized across rows!)
              (let [scaling (float (/ alpha (double rank)))
                    lr (float 0.005)]
                (-> (IntStream/range 0 rows)
                    (.parallel)
                    (.forEach (reify IntConsumer
                                (accept [_ r]
                                  (let [r (int r)
                                        ^floats w-arr w-arr
                                        ^floats cur-deq cur-deq
                                        ^floats lora-a lora-a
                                        ^floats lora-b lora-b
                                        r-base (* r cols)
                                        b-base (* r rank)]
                                    (dotimes [k rank]
                                      (let [b-val (aget lora-b (+ b-base k))
                                            grad (loop [c (int 0) acc (float 0.0)]
                                                   (if (< c cols)
                                                     (let [diff (- (aget w-arr (+ r-base c)) (aget cur-deq (+ r-base c)))]
                                                       (recur (unchecked-inc c) (+ acc (* diff (aget lora-a (+ (* k cols) c))))))
                                                     acc))]
                                        (aset-float lora-b (+ b-base k) (+ b-val (* lr scaling grad)))))))))))
              (recur (unchecked-inc ep)))))
        (assoc (:rec @best-state)
               :zs zs
               :zr zr
               :lora-a (:lora-a @best-state)
               :lora-b (:lora-b @best-state)
               :rank rank
               :alpha alpha
               :mse (:mse @best-state))))))

;; ==============================================================================
;; 5. Model-Level Calibration & Artifact Serialization (Boundary Shell)
;; ==============================================================================

(defn save-catq-model!
  "Persists both the calibration artifacts (`catq_factors.edn`) and the
   quantized model weights (`model.safetensors` with metadata) to disk."
  [src-model-dir out-dir factors-map quantized-specs]
  (let [src-dir (io/file src-model-dir)
        dst-dir (io/file out-dir)
        _ (.mkdirs dst-dir)
        arena (Arena/ofAuto)
        factors-file (io/file dst-dir "catq_factors.edn")
        out-safetensors (io/file dst-dir "model.safetensors")
        out-path (Path/of (.getAbsolutePath out-safetensors) (into-array String []))]

    ;; 1. Save catq_factors.edn
    (spit factors-file (pr-str factors-map))

    ;; 2. Build Safetensors layout
    (let [{:keys [specs _total-bytes]}
          (loop [items quantized-specs
                 curr-offset 0
                 accum []]
            (if (empty? items)
              {:specs accum :total-bytes curr-offset}
              (let [item (first items)]
                (if (:quantize? item)
                  (let [w-len (alength ^bytes (:data item))
                        scale-len (* (alength ^shorts (:scales item)) 2)
                        w-end (+ curr-offset w-len)
                        s-end (+ w-end scale-len)
                        spec (assoc item
                                    :data-offsets [curr-offset w-end]
                                    :scale-offsets [w-end s-end])]
                    (recur (rest items) s-end (conj accum spec)))
                  (let [slice ^MemorySegment (:segment item)
                        len (.byteSize slice)
                        end (+ curr-offset len)
                        spec (assoc item :data-offsets [curr-offset end])]
                    (recur (rest items) end (conj accum spec)))))))

          header-tensors
          (into {}
                (mapcat (fn [spec]
                          (if (:quantize? spec)
                            [[(:name spec) {"dtype" "I8"
                                            "shape" (:shape spec)
                                            "data_offsets" (:data-offsets spec)}]
                             [(:scale-name spec) {"dtype" "BF16"
                                                  "shape" (:scale-shape spec)
                                                  "data_offsets" (:scale-offsets spec)}]]
                            [[(:name spec) {"dtype" (:dtype spec)
                                            "shape" (:shape spec)
                                            "data_offsets" (:data-offsets spec)}]]))
                        specs))

          metadata {"quantization" "ternary"
                    "format" "clj-xla"
                    "producer" "einsum.quant.catq"
                    "group_size" "128"}
          full-header-map (assoc header-tensors "__metadata__" metadata)
          json-str (json/write-str full-header-map :escape-slash false)
          header-bytes (.getBytes json-str "UTF-8")
          header-size (count header-bytes)]

      ;; 3. Stream Safetensors to file
      (with-open [fc (FileChannel/open out-path (into-array [StandardOpenOption/CREATE
                                                             StandardOpenOption/WRITE
                                                             StandardOpenOption/TRUNCATE_EXISTING]))]
        ;; Write header size LE
        (let [size-buf (ByteBuffer/allocate 8)]
          (.putLong size-buf (Long/reverseBytes (long header-size)))
          (.flip size-buf)
          (.write fc size-buf))
        ;; Write JSON header
        (.write fc (ByteBuffer/wrap header-bytes))

        ;; Write payload
        (doseq [spec specs]
          (if (:quantize? spec)
            (let [^bytes data-bytes (:data spec)
                  ^shorts scale-shorts (:scales spec)
                  num-scales (alength scale-shorts)
                  scale-seg (.allocate arena (* (long num-scales) 2) (long 1))]
              (.write fc (ByteBuffer/wrap data-bytes))
              (MemorySegment/copy scale-shorts 0 scale-seg ValueLayout/JAVA_SHORT (long 0) num-scales)
              (.write fc (.asByteBuffer scale-seg)))
            (let [slice ^MemorySegment (:segment spec)]
              (st/write-segment-to-channel! fc slice)))))

      ;; 4. Copy configuration and tokenizer files
      (doseq [fname ["config.json" "generation_config.json" "tokenizer.json" "tokenizer_config.json"]]
        (let [src-f (io/file src-dir fname)
              dst-f (io/file dst-dir fname)]
          (when (.exists src-f)
            (java.nio.file.Files/copy (.toPath src-f)
                                      (.toPath dst-f)
                                      ^"[Ljava.nio.file.CopyOption;" (into-array java.nio.file.CopyOption
                                                                                 [java.nio.file.StandardCopyOption/REPLACE_EXISTING]))))))))
