(ns einsum.uop.golden-test
  "Step 0: Golden Battery & Post-XLA Regression Oracle Tests.
   Validates:
   1. Completeness, structure, and greedy token parity of the golden regression oracle
      (gemma4_uop_oracle.edn and xla_vs_uop_archive.edn) for E2B, E4B, 31B INT4, and micro models.
   2. Deterministic micro-Gemma-4 BF16 and INT4 GPTQ execution via :uop-cpu asserting bit-level
      or tight (< 1e-4) logit tolerance and exact greedy continuation parity.
   3. Generative property verification of top-k ordering and argmax soundness across oracle entries."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.uop.runtime :as uop-rt]
            [einsum.models.gemma4.kernels :as kernels]
            [einsum.models.gemma4.runtime :as rt]))

;; -----------------------------------------------------------------------------
;; 1. Resource Loaders & Helpers
;; -----------------------------------------------------------------------------

(defn- load-golden-oracle []
  (let [res (io/resource "golden/gemma4_uop_oracle.edn")]
    (assert (some? res) "golden/gemma4_uop_oracle.edn resource not found on classpath!")
    (edn/read-string (slurp res))))

(defn- load-archive []
  (let [res (io/resource "golden/xla_vs_uop_archive.edn")]
    (assert (some? res) "golden/xla_vs_uop_archive.edn resource not found on classpath!")
    (edn/read-string (slurp res))))

(defn- allocate-deterministic-weights [invars seed]
  (let [rng (java.util.Random. (long seed))]
    (reduce (fn [acc [var-kw [_ sh dt]]]
              (if (contains? #{:x :pos :k_cache_in_0 :v_cache_in_0} var-kw)
                acc
                (let [size (long (reduce * 1 sh))
                      nm (name var-kw)
                      arr (cond
                            (re-find #"norm" nm)
                            (float-array (repeatedly size #(float (+ 0.8 (* 0.4 (.nextFloat rng))))))

                            (re-find #"scalar" nm)
                            (float-array (repeatedly size #(float 1.0)))

                            (re-find #"scale" nm)
                            (float-array (repeatedly size #(float (+ 0.01 (* 0.05 (.nextFloat rng))))))

                            (contains? #{:f16 :bf16 :f32 :f64} dt)
                            (float-array (repeatedly size #(float (- (* 0.2 (.nextFloat rng)) 0.1))))

                            (= dt :i8)
                            (byte-array (repeatedly size #(byte (unchecked-byte (- (.nextInt rng 256) 128)))))

                            :else
                            (byte-array size 0))]
                  (assoc acc var-kw arr))))
            {}
            invars)))

(defn- extract-logits-info [^floats arr top-n]
  (let [n (alength arr)
        first-32 (mapv (fn [i] (double (aget arr i))) (range (min 32 n)))
        pq (java.util.PriorityQueue. (int (inc top-n))
                                     (reify java.util.Comparator
                                       (compare [_ a b]
                                         (Double/compare (aget arr (int a)) (aget arr (int b))))))]
    (dotimes [i n]
      (.add pq (int i))
      (when (> (.size pq) top-n)
        (.poll pq)))
    (let [top-indices (loop [acc '()]
                        (if (.isEmpty pq)
                          (vec acc)
                          (recur (cons (.poll pq) acc))))
          top-vec (mapv (fn [i] {:token-id i :logit (double (aget arr (int i)))}) top-indices)]
      {:first-32-logits first-32
       :top-8 top-vec})))

(defn- run-micro-model [cfg seed]
  (let [ctx (uop-rt/init-uop-context :uop-cpu {:quiet? true})
        session {:ctx ctx :config cfg :opts {:quiet true}}
        prompt-ids [2 10 20 30]
        prompt-len (count prompt-ids)
        max-seq (:max-seq-len cfg)
        vocab-sz (:vocab-size cfg)
        prefill-exec (kernels/compile-gemma4-chunked-prefill-executable session max-seq prompt-len)
        kv-exec (kernels/compile-gemma4-kv-executable session max-seq)
        exec-step (requiring-resolve 'einsum.models.gemma4.runtime/execute-step-executable)
        prefill-invars (kernels/build-gemma4-kv-invars (assoc cfg :q-len prompt-len :chunk-size prompt-len) max-seq)
        kv-invars (kernels/build-gemma4-kv-invars cfg max-seq)
        weights (allocate-deterministic-weights prefill-invars seed)

        ;; Prefill Step 0
        k-cache-0 (float-array (* max-seq 16) 0.0)
        v-cache-0 (float-array (* max-seq 16) 0.0)
        prefill-inputs (mapv (fn [[var-kw _]]
                               (case var-kw
                                 :x (int-array prompt-ids)
                                 :pos (int-array [0])
                                 :k_cache_in_0 k-cache-0
                                 :v_cache_in_0 v-cache-0
                                 (get weights var-kw)))
                             prefill-invars)
        [logits-buf-0 k-buf-1 v-buf-1] (exec-step ctx prefill-exec prefill-inputs 3)
        logits-0 (uop-rt/device->host-floats ctx logits-buf-0 vocab-sz)
        logits-0-info (extract-logits-info logits-0 8)
        t1 (rt/argmax-host logits-0)

        ;; Decode Step 1
        kv-inputs-1 (mapv (fn [[var-kw _]]
                            (case var-kw
                              :x (int-array [t1])
                              :pos (int-array [prompt-len])
                              :k_cache_in_0 k-buf-1
                              :v_cache_in_0 v-buf-1
                              (get weights var-kw)))
                          kv-invars)
        [logits-buf-1 k-buf-2 v-buf-2] (exec-step ctx kv-exec kv-inputs-1 3)
        logits-1 (uop-rt/device->host-floats ctx logits-buf-1 vocab-sz)
        logits-1-info (extract-logits-info logits-1 8)
        t2 (rt/argmax-host logits-1)

        ;; Decode Step 2
        kv-inputs-2 (mapv (fn [[var-kw _]]
                            (case var-kw
                              :x (int-array [t2])
                              :pos (int-array [(inc prompt-len)])
                              :k_cache_in_0 k-buf-2
                              :v_cache_in_0 v-buf-2
                              (get weights var-kw)))
                          kv-invars)
        [logits-buf-2 k-buf-3 v-buf-3] (exec-step ctx kv-exec kv-inputs-2 3)
        logits-2 (uop-rt/device->host-floats ctx logits-buf-2 vocab-sz)
        t3 (rt/argmax-host logits-2)

        ;; Decode Step 3
        kv-inputs-3 (mapv (fn [[var-kw _]]
                            (case var-kw
                              :x (int-array [t3])
                              :pos (int-array [(+ prompt-len 2)])
                              :k_cache_in_0 k-buf-3
                              :v_cache_in_0 v-buf-3
                              (get weights var-kw)))
                          kv-invars)
        [logits-buf-3 _ _] (exec-step ctx kv-exec kv-inputs-3 3)
        logits-3 (uop-rt/device->host-floats ctx logits-buf-3 vocab-sz)
        t4 (rt/argmax-host logits-3)]

    {:config cfg
     :prompt-ids prompt-ids
     :generated-tokens [t1 t2 t3 t4]
     :step-logits {0 logits-0-info
                   1 logits-1-info}}))

;; -----------------------------------------------------------------------------
;; 2. Completeness & Parity Tests
;; -----------------------------------------------------------------------------

(deftest test-golden-oracle-completeness-and-parity
  (testing "Golden oracle contains all required models, prompts, M-sweeps, and archive metrics"
    (let [oracle (load-golden-oracle)
          archive (load-archive)]
      ;; 1. Oracle top-level structure
      (is (contains? oracle :gemma-4-e2b-it))
      (is (contains? oracle :gemma-4-e4b-it))
      (is (contains? oracle :gemma-4-31b-it-int4))
      (is (contains? oracle :micro-gemma4-oracle))

      ;; 2. E2B prompts
      (let [e2b-prompts (get-in oracle [:gemma-4-e2b-it :prompts])]
        (is (= 3 (count e2b-prompts)))
        (is (= [:prompt-1 :prompt-1b :prompt-2] (mapv :id e2b-prompts)))
        (doseq [p e2b-prompts]
          (is (pos? (count (:prompt-token-ids p))))
          (is (pos? (count (:generated-token-ids p))))
          (is (string? (:generated-text p))))
        ;; Prompt 1 step logits
        (let [p1 (first e2b-prompts)
              logits-0 (get-in p1 [:step-logits 0])]
          (is (= 32 (count (:first-32-logits logits-0))))
          (is (= 16 (count (:top-16 logits-0))))
          (is (= (first (:generated-token-ids p1)) (:token-id (first (:top-16 logits-0)))))))

      ;; 3. E4B prompts
      (let [e4b-prompts (get-in oracle [:gemma-4-e4b-it :prompts])]
        (is (= 3 (count e4b-prompts)))
        (is (= [:prompt-1 :prompt-1b :prompt-2] (mapv :id e4b-prompts)))
        (let [p1 (first e4b-prompts)
              logits-0 (get-in p1 [:step-logits 0])]
          (is (= 32 (count (:first-32-logits logits-0))))
          (is (= 16 (count (:top-16 logits-0))))
          (is (= (first (:generated-token-ids p1)) (:token-id (first (:top-16 logits-0)))))))

      ;; 4. 31B INT4 prompts & M-sweep
      (let [p31-prompts (get-in oracle [:gemma-4-31b-it-int4 :prompts])
            m-sweep (get-in oracle [:gemma-4-31b-it-int4 :m-sweep])]
        (is (= 3 (count p31-prompts)))
        (is (= [:prompt-1 :prompt-1b :prompt-2] (mapv :id p31-prompts)))
        (is (= [2 8 14 15 16 32] (mapv :m m-sweep)))
        (doseq [entry m-sweep]
          (is (= (:m entry) (count (:input-token-ids entry))))
          (is (= 236744 (:next-token-id entry))))
        ;; M=16 greedy continuation
        (let [m16 (first (filter #(== 16 (:m %)) m-sweep))]
          (is (= [236744 398 236770 569 236744 241790] (:greedy-continuation m16)))))

      ;; 5. Archive benchmark structure
      (is (contains? archive :metadata))
      (is (contains? archive :live-benchmarks))
      (is (contains? archive :historical-scaling-archive))
      (is (= 3 (count (get-in archive [:live-benchmarks :gemma-4-e2b-it]))))
      (is (= 3 (count (get-in archive [:live-benchmarks :gemma-4-e4b-it]))))
      (is (= 7 (count (get-in archive [:historical-scaling-archive :phase-6d-e2b-scaling :prefill-table])))))))

;; -----------------------------------------------------------------------------
;; 3. Deterministic Micro-Model Regression Test
;; -----------------------------------------------------------------------------

(deftest test-micro-gemma4-bf16-and-int4-uop-golden-regression
  (testing "Deterministic 1-layer BF16 and INT4 micro models reproduce exact golden token sequences and logits on :uop-cpu"
    (let [oracle (load-golden-oracle)
          bf16-ref (get-in oracle [:micro-gemma4-oracle :micro-bf16])
          int4-ref (get-in oracle [:micro-gemma4-oracle :micro-int4])

          ;; Re-run micro BF16 model on UOp CPU
          bf16-run (run-micro-model (:config bf16-ref) 42)
          ;; Re-run micro INT4 model on UOp CPU
          int4-run (run-micro-model (:config int4-ref) 42)]

      ;; Verify exact 4-token greedy sequence parity
      (is (= (:generated-tokens bf16-ref) (:generated-tokens bf16-run)))
      (is (= (:generated-tokens int4-ref) (:generated-tokens int4-run)))
      (is (= [30 30 30 30] (:generated-tokens bf16-run)))
      (is (= [30 30 30 30] (:generated-tokens int4-run)))

      ;; Verify tight float32 tolerance (< 1e-4) on Step 0 and Step 1 logits
      (doseq [step [0 1]]
        (let [ref-logits-bf16 (get-in bf16-ref [:step-logits step :first-32-logits])
              run-logits-bf16 (get-in bf16-run [:step-logits step :first-32-logits])
              ref-logits-int4 (get-in int4-ref [:step-logits step :first-32-logits])
              run-logits-int4 (get-in int4-run [:step-logits step :first-32-logits])
              diff-bf16 (apply max (map #(Math/abs (- (double %1) (double %2))) ref-logits-bf16 run-logits-bf16))
              diff-int4 (apply max (map #(Math/abs (- (double %1) (double %2))) ref-logits-int4 run-logits-int4))]
          (is (< diff-bf16 1e-4) (str "BF16 step " step " logits max error exceeded: " diff-bf16))
          (is (< diff-int4 1e-4) (str "INT4 step " step " logits max error exceeded: " diff-int4)))))))

;; -----------------------------------------------------------------------------
;; 4. Generative Top-K & Logit Soundness Invariants
;; -----------------------------------------------------------------------------

(defspec prop-golden-logit-top-k-invariants 25
  (let [oracle (load-golden-oracle)
        candidate-steps
        (vec (concat
              (for [p (get-in oracle [:gemma-4-e2b-it :prompts])
                    [_step info] (:step-logits p)]
                info)
              (for [p (get-in oracle [:gemma-4-e4b-it :prompts])
                    [_step info] (:step-logits p)]
                info)
              (for [p (get-in oracle [:gemma-4-31b-it-int4 :prompts])
                    [_step info] (:step-logits p)]
                info)
              (vals (get-in oracle [:micro-gemma4-oracle :micro-bf16 :step-logits]))
              (vals (get-in oracle [:micro-gemma4-oracle :micro-int4 :step-logits]))))]
    (prop/for-all [step-info (gen/elements candidate-steps)]
                  (let [top-list (or (:top-16 step-info) (:top-8 step-info))
                        logits (mapv :logit top-list)
                        first-32 (:first-32-logits step-info)]
                    (and
                     ;; Strictly descending order of top-k logits
                     (every? true? (map >= logits (rest logits)))
                     ;; All logits are non-NaN and finite
                     (every? #(and (number? %) (not (Double/isNaN (double %))) (Double/isFinite (double %))) logits)
                     ;; First 32 logits are populated and finite
                     (= 32 (count first-32))
                     (every? #(and (number? %) (not (Double/isNaN (double %))) (Double/isFinite (double %))) first-32))))))
