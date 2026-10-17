(ns einsum.uop.beam
  "Pure-Clojure implementation of tinygrad's Beam Search & Hardware-in-the-Loop Auto-Tuning.
   Modeled on tinygrad's `tinygrad/codegen/opt/search.py` and `tinygrad/engine/realize.py`.
   Generates valid `OptOps` candidate mutations, profiles candidate HIP kernels directly on device,
   and caches optimal kernel recipes in EDN disk format."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [einsum.uop.opt :as opt]
            [einsum.uop.render :as render]
            [einsum.uop.runtime :as runtime]))

;; -----------------------------------------------------------------------------
;; 1. Candidate Action Generator (OptOps Mutations)
;; -----------------------------------------------------------------------------

(defn candidate-actions
  "Generates all valid `Opt` mutations `{:op op :axis axis :amount amount}` for kernel entry `ke`.
   Ensures amounts are exact divisors of the respective axis dimension, and respects hardware limits."
  [ke]
  (let [axes (or (:axes ke)
                 (:out-shape (:rangeified ke))
                 [(:out-size (:rangeified ke) 1)])
        red-axes (or (:red-axes ke)
                     (:red-shape (:rangeified ke))
                     [])
        spatial-actions
        (for [axis (range (count axes))
              :let [dim (long (nth axes axis 1))]
              :when (pos? dim)
              divisor [2 4 8 16 32]
              :when (zero? (mod dim divisor))
              op [:local :upcast]]
          {:op op :axis axis :amount divisor})

        reduction-actions
        (if (empty? red-axes)
          []
          (for [axis (range (count red-axes))
                :let [dim (long (nth red-axes axis 1))]
                :when (pos? dim)
                divisor [2 4 8 16 32 64]
                :when (zero? (mod dim divisor))
                op [:group :unroll]]
            {:op op :axis axis :amount divisor}))

        all-actions (concat spatial-actions reduction-actions)]
    (vec (distinct all-actions))))

;; -----------------------------------------------------------------------------
;; 2. Catalog & Disk Cache Persistence
;; -----------------------------------------------------------------------------

(defn load-catalog
  "Loads the pre-populated RDNA3 beam catalog `resources/uop/rdna3_beam_catalog.edn`."
  []
  (if-let [res (io/resource "uop/rdna3_beam_catalog.edn")]
    (edn/read-string (slurp res))
    (let [f (io/file "resources/uop/rdna3_beam_catalog.edn")]
      (if (.exists f)
        (edn/read-string (slurp f))
        {}))))

(defn lookup-cache-entry
  "Looks up an optimization recipe for `key` in `cache-file`."
  [cache-file key]
  (let [f (io/file cache-file)]
    (when (.exists f)
      (try
        (let [m (edn/read-string (slurp f))]
          (get m key))
        (catch Throwable _ nil)))))

(defn save-cache-entry!
  "Persists an optimization recipe `recipe` for `key` into `cache-file`."
  [cache-file key recipe]
  (let [f (io/file cache-file)
        _ (when-let [p (.getParentFile f)] (.mkdirs p))
        existing (or (lookup-cache-entry f key) {})
        data (if (map? existing) (assoc existing key recipe) {key recipe})]
    (spit f (pr-str data))
    recipe))

;; -----------------------------------------------------------------------------
;; 3. Hardware-in-the-Loop Timing & Profiler
;; -----------------------------------------------------------------------------

(defn time-kernel-rocm
  "Measures median execution time of a candidate kernel `ke` on device in microseconds.
   Performs a warm-up launch followed by `cnt` timed runs."
  ([ctx ke]
   (time-kernel-rocm ctx ke 3))
  ([ctx ke cnt]
   (try
     (let [rendered (render/render-kernel ke {:target :rocm})
           compiled (runtime/compile-rocm-module! ctx [rendered])
           fn-handle (get (:fn-handles compiled) (:canonical-fn-name rendered))]
       (if-not fn-handle
         Double/POSITIVE_INFINITY
         (let [slots (:buffer-slots rendered)
               scratch-bufs (mapv (fn [slot]
                                    (let [b-meta (get-in ke [:schedule :buffers slot])]
                                      (runtime/allocate-device-buffer ctx (or (:shape b-meta) [256]) (or (:dtype b-meta) :f32))))
                                  slots)
               exec-arena (java.lang.foreign.Arena/ofShared)
               n-args (count slots)
               arg-vals-seg (.allocate exec-arena java.lang.foreign.ValueLayout/ADDRESS n-args)
               kernel-params-seg (.allocate exec-arena java.lang.foreign.ValueLayout/ADDRESS n-args)]
           (try
             (dotimes [i n-args]
               (.setAtIndex arg-vals-seg java.lang.foreign.ValueLayout/ADDRESS (long i) (:ptr (nth scratch-bufs i)))
               (.setAtIndex kernel-params-seg java.lang.foreign.ValueLayout/ADDRESS (long i)
                            (.asSlice arg-vals-seg (* i 8) 8)))
             (let [hip (or (:hip ctx) (runtime/init-hip-bindings!))
                   ^java.lang.invoke.MethodHandle launch-mh (:hipModuleLaunchKernel hip)
                   ^java.lang.invoke.MethodHandle sync-mh (:hipDeviceSynchronize hip)
                   [gx gy gz] (or (:global-size ke) [1 1 1])
                   [lx ly lz] (or (:local-size ke) [256 1 1])
                   launch-fn (fn []
                               (let [rc (int (.invokeWithArguments launch-mh
                                                                   (into-array Object [fn-handle
                                                                                       (int gx) (int gy) (int gz)
                                                                                       (int lx) (int ly) (int lz)
                                                                                       (int 0)
                                                                                       java.lang.foreign.MemorySegment/NULL
                                                                                       kernel-params-seg
                                                                                       java.lang.foreign.MemorySegment/NULL])))]
                                 (runtime/check-hip! rc "hipModuleLaunchKernel")
                                 (let [rc-sync (int (.invokeWithArguments sync-mh (into-array Object [])))]
                                   (runtime/check-hip! rc-sync "hipDeviceSynchronize"))))]
               ;; Warm-up run
               (launch-fn)
               ;; Timed runs
               (let [times (vec (for [_ (range cnt)]
                                  (let [t0 (System/nanoTime)]
                                    (launch-fn)
                                    (/ (double (- (System/nanoTime) t0)) 1000.0))))
                     sorted-times (sort times)
                     median-us (nth sorted-times (quot cnt 2))]
                 median-us))
             (finally
               (doseq [b scratch-bufs]
                 (runtime/destroy-buffer! ctx b))
               (when (.isAlive (.scope exec-arena))
                 (.close exec-arena)))))))
     (catch Throwable _
       Double/POSITIVE_INFINITY))))

;; -----------------------------------------------------------------------------
;; 4. Tinygrad Beam Search Engine
;; -----------------------------------------------------------------------------

(defn beam-search
  "Searches for the optimal `Opt` recipe using width-`k` beam search.
   Options:
     `:width`       - beam width k (default 4)
     `:depth`       - search depth (default 3)
     `:cost-fn`     - function `(fn [cand])` returning microsecond cost (default: hardware profiler)
     `:use-cache?`  - boolean, whether to use persistent EDN cache (default true)
     `:cache-file`  - path to cache file (default `~/.cache/clj_einsum/uop/beam_cache.edn`)"
  ([ke]
   (beam-search ke {}))
  ([ke opts]
   (let [width (long (or (:width opts) 4))
         depth (long (or (:depth opts) 3))
         cost-fn (or (:cost-fn opts) (fn [cand] (time-kernel-rocm (:ctx opts) cand)))
         use-cache? (get opts :use-cache? true)
         user-home (System/getProperty "user.home")
         cache-file (or (:cache-file opts)
                        (str user-home "/.cache/clj_einsum/uop/beam_cache.edn"))
         arch-key (str "gfx1100:" (pr-str (opt/kernel-signature ke)))
         cached (when use-cache?
                  (or (lookup-cache-entry cache-file arch-key)
                      (opt/lookup-catalog ke)))]
     (if (and cached (:opts cached))
       cached
       (let [init-cost (double (cost-fn ke))
             init-beam [{:kernel-entry ke
                         :opts (vec (or (:opts ke) []))
                         :cost init-cost}]
             final-beam
             (loop [step 0
                    current-beam init-beam]
               (if (>= step depth)
                 current-beam
                 (let [expanded
                       (reduce
                        (fn [acc cand]
                          (let [cur-ke (:kernel-entry cand)
                                actions (candidate-actions cur-ke)]
                            (into acc
                                  (keep (fn [act]
                                          (try
                                            (let [next-ke (opt/apply-opt cur-ke act)
                                                  cost (double (cost-fn next-ke))]
                                              (when-not (Double/isInfinite cost)
                                                {:kernel-entry next-ke
                                                 :opts (:opts next-ke)
                                                 :cost cost}))
                                            (catch Throwable _ nil)))
                                        actions))))
                        current-beam
                        current-beam)
                       deduped (vals (into {} (map (fn [c] [(:opts c) c]) expanded)))
                       sorted (sort-by :cost deduped)
                       next-beam (vec (take width sorted))]
                   (if (empty? next-beam)
                     current-beam
                     (recur (inc step) next-beam)))))
             best (first (sort-by :cost final-beam))
             result {:opts (:opts best)
                     :best-cost (:cost best)
                     :kernel-entry (:kernel-entry best)}]
         (when use-cache?
           (save-cache-entry! cache-file arch-key result))
         result)))))
