(ns einsum.logic.interpret-simd
  "Panama Vector API (jdk.incubator.vector) accelerated SGEMM kernel for pure-JVM StableHLO.
   Requires --add-modules=jdk.incubator.vector at JVM launch."
  (:import [java.util Arrays]
           [java.util.function IntConsumer]
           [java.util.stream IntStream]
           [jdk.incubator.vector FloatVector VectorSpecies]))

;; ---------------------------------------------------------------------------
;; SIMD Configuration
;; ---------------------------------------------------------------------------

(def ^VectorSpecies ^:private simd-species FloatVector/SPECIES_PREFERRED)
(def ^:private simd-vl (long (.length simd-species)))
(def ^:private simd-jb (* 4 simd-vl))
(def ^:private simd-kb 128)

(def ^:dynamic *parallel-enabled?* true)

(defn- p-for-range
  "Runs (f i) for i in 0..(n-1) in parallel using ForkJoinPool when parallel? is true and n >= min-chunk."
  [^long n ^long min-chunk f]
  (if (or (not *parallel-enabled?*) (<= n 1) (< n min-chunk))
    (dotimes [i n] (f i))
    (.forEach (.parallel (IntStream/range 0 n))
              (reify IntConsumer
                (accept [_ i] (f (long i)))))))

;; ---------------------------------------------------------------------------
;; SIMD Micro-Kernels
;; ---------------------------------------------------------------------------

(defn- simd-kb-block!
  [^floats C coff ^floats A aoff ^floats B boff N kb kend jb]
  (let [coff (long coff) aoff (long aoff) boff (long boff)
        N (long N) kb (long kb) kend (long kend) jb (long jb)
        o0 jb
        o1 (+ jb simd-vl)
        o2 (+ jb (* 2 simd-vl))
        o3 (+ jb (* 3 simd-vl))]
    (loop [k kb
           ^FloatVector c0 (FloatVector/fromArray simd-species C (int (+ coff o0)))
           ^FloatVector c1 (FloatVector/fromArray simd-species C (int (+ coff o1)))
           ^FloatVector c2 (FloatVector/fromArray simd-species C (int (+ coff o2)))
           ^FloatVector c3 (FloatVector/fromArray simd-species C (int (+ coff o3)))]
      (if (< k kend)
        (let [^FloatVector av (FloatVector/broadcast simd-species (aget A (int (+ aoff k))))
              brow (int (+ boff (* k N)))]
          (recur (inc k)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o0))) c0)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o1))) c1)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o2))) c2)
                 (.fma av (FloatVector/fromArray simd-species B (int (+ brow o3))) c3)))
        (do
          (.intoArray c0 C (int (+ coff o0)))
          (.intoArray c1 C (int (+ coff o1)))
          (.intoArray c2 C (int (+ coff o2)))
          (.intoArray c3 C (int (+ coff o3))))))))

(defn- simd-cleanup!
  [^floats C coff ^floats A aoff ^floats B boff N kb kend j0 j1]
  (let [coff (long coff) aoff (long aoff) boff (long boff)
        N (long N) kb (long kb) kend (long kend) j0 (long j0) j1 (long j1)]
    (loop [k kb]
      (when (< k kend)
        (let [a (float (aget A (int (+ aoff k))))
              brow (int (+ boff (* k N)))]
          (loop [j j0]
            (when (< j j1)
              (let [idx (int (+ coff j))]
                (aset C idx (float (+ (aget C idx) (* a (aget B (int (+ brow j))))))))
              (recur (inc j)))))
        (recur (inc k))))))

;; ---------------------------------------------------------------------------
;; SIMD SGEMM Implementation
;; ---------------------------------------------------------------------------

(defn simd-sgemm-slice!
  "Computes a block of rows [m-start, m-end) for C = A @ B with offsets.
   Assumes C[coff + m-start*N .. coff + m-end*N] is pre-cleared."
  [^floats C coff ^floats A aoff ^floats B boff m-start m-end N K]
  (let [coff (long coff) aoff (long aoff) boff (long boff)
        m-start (long m-start) m-end (long m-end) N (long N) K (long K)
        jb (long simd-jb)
        kb (long simd-kb)]
    (loop [kbb (long 0)]
      (when (< kbb K)
        (let [kend (long (min K (+ kbb kb)))]
          (loop [jbb (long 0)]
            (if (<= (+ jbb jb) N)
              (do
                (loop [i m-start]
                  (when (< i m-end)
                    (simd-kb-block! C (+ coff (* i N)) A (+ aoff (* i K)) B boff N kbb kend jbb)
                    (recur (inc i))))
                (recur (+ jbb jb)))
              (when (< jbb N)
                (loop [i m-start]
                  (when (< i m-end)
                    (simd-cleanup! C (+ coff (* i N)) A (+ aoff (* i K)) B boff N kbb kend jbb N)
                    (recur (inc i)))))))
          (recur (+ kbb kb)))))))

(defn simd-sgemm-col-slice!
  "Computes a block of columns [j-start, j-end) for C = A @ B when M=1.
   Assumes C[coff + j-start .. coff + j-end] is pre-cleared."
  [^floats C coff ^floats A aoff ^floats B boff j-start j-end N K]
  (let [coff (long coff) aoff (long aoff) boff (long boff)
        j-start (long j-start) j-end (long j-end) N (long N) K (long K)
        jb (long simd-jb)
        kb (long simd-kb)]
    (loop [kbb (long 0)]
      (when (< kbb K)
        (let [kend (long (min K (+ kbb kb)))]
          (loop [jbb j-start]
            (if (<= (+ jbb jb) j-end)
              (do
                (simd-kb-block! C coff A aoff B boff N kbb kend jbb)
                (recur (+ jbb jb)))
              (when (< jbb j-end)
                (simd-cleanup! C coff A aoff B boff N kbb kend jbb j-end))))
          (recur (+ kbb kb)))))))

(defn simd-sgemm!
  "Computes C = A @ B, OVERWRITING C. A is MxK, B is KxN, C is MxN, row-major.
   Supports arbitrary buffer offsets and automatic parallelization across CPU cores."
  ([^floats C ^floats A ^floats B M N K]
   (simd-sgemm! C 0 A 0 B 0 M N K))
  ([^floats C coff ^floats A aoff ^floats B boff M N K]
   (let [coff (long coff) aoff (long aoff) boff (long boff)
         M (long M) N (long N) K (long K)
         total-elems (long (* M N))]
     ;; Clear destination region
     (Arrays/fill C (int coff) (int (+ coff total-elems)) (float 0))

     (cond
       ;; Strategy A: Parallelize across rows M when M >= 4 and total work is significant
       (and (>= M 4) (>= (* M N K) 32768))
       (let [chunk-size (long (max 1 (quot M (.availableProcessors (Runtime/getRuntime)))))]
         (p-for-range M chunk-size
                      (fn [i]
                        (simd-sgemm-slice! C coff A aoff B boff i (inc i) N K))))

       ;; Strategy B: Parallelize across column blocks N when M < 4, N >= 512, and K is significant
       (and (< M 4) (>= N 512) (>= (* M N K) 32768))
       (let [num-blocks (long (Math/ceil (/ (double N) simd-jb)))]
         (p-for-range num-blocks 2
                      (fn [b-idx]
                        (let [j-start (* b-idx simd-jb)
                              j-end (min N (+ j-start simd-jb))]
                          (dotimes [i M]
                            (simd-sgemm-col-slice! C (+ coff (* i N)) A (+ aoff (* i K)) B boff
                                                   j-start j-end N K))))))

       ;; Strategy C: Single-threaded SIMD execution
       :else
       (simd-sgemm-slice! C coff A aoff B boff 0 M N K)))))
