(ns tools.analyze-catq-checkpoint
  (:require [clojure.java.io :as io]
            [einsum.quant.catq :as catq]
            [einsum.runtime.safetensors :as st])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(set! *warn-on-reflection* true)

(defn snr-db [^floats orig ^floats deq]
  (let [n (alength orig)
        sig-power (loop [i (int 0) acc 0.0]
                    (if (< i n)
                      (let [v (double (aget orig i))]
                        (recur (unchecked-inc i) (+ acc (* v v))))
                      acc))
        noise-power (loop [i (int 0) acc 0.0]
                      (if (< i n)
                        (let [diff (- (double (aget orig i)) (double (aget deq i)))]
                          (recur (unchecked-inc i) (+ acc (* diff diff))))
                        acc))]
    (if (zero? noise-power)
      Double/POSITIVE_INFINITY
      (* 10.0 (Math/log10 (/ sig-power (max 1e-12 noise-power)))))))

(defn cos-sim-vec [^floats orig ^floats deq]
  (let [n (alength orig)
        no (Math/sqrt (loop [i (int 0) acc 0.0]
                        (if (< i n)
                          (let [v (double (aget orig i))] (recur (unchecked-inc i) (+ acc (* v v))))
                          acc)))
        nd (Math/sqrt (loop [i (int 0) acc 0.0]
                        (if (< i n)
                          (let [v (double (aget deq i))] (recur (unchecked-inc i) (+ acc (* v v))))
                          acc)))]
    (if (or (zero? no) (zero? nd)) 0.0
        (/ (loop [i (int 0) acc 0.0]
             (if (< i n)
               (recur (unchecked-inc i) (+ acc (* (double (aget orig i)) (double (aget deq i)))))
               acc))
           (* no nd)))))

(defn -main [& _args]
  (println "\n==========================================================================================================")
  (println "  CAT-Q Checkpoint Analysis: Weight Reconstruction SNR, Gamma Distribution & Scales")
  (println "==========================================================================================================")
  (let [arena (Arena/ofAuto)
        base-mmap (st/map-safetensors-weights ".models/gemma-4-E2B-it" arena)
        catq-mmap (st/map-safetensors-weights ".models/gemma-4-E2B-it-catq" arena)
        factors-file (io/file ".models/gemma-4-E2B-it-catq/catq_factors.edn")
        factors (when (.exists factors-file)
                  (println "Loading catq_factors.edn...")
                  (read-string (slurp factors-file)))
        projs-factors (:projections factors)
        projs ["q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj"]]

    (println "\nLayer | Projection | Shape         | SNR (dB) | Cos Sim | Mean Abs W | Scale Ratio | Mean Gamma | % Pegged (<=0.4 or >=0.6)")
    (println "------+------------+---------------+----------+---------+------------+-------------+------------+--------------------------")

    (dotimes [L 35]
      (doseq [p projs]
        (let [mid (if (or (= p "gate_proj") (= p "up_proj") (= p "down_proj")) ".mlp." ".self_attn.")
              tname (str "model.language_model.layers." L mid p ".weight")
              wb (st/get-tensor-floats base-mmap tname)]
          (when wb
            (let [shape (get-in (:header base-mmap) [tname "shape"])
                  rows (int (first shape))
                  cols (int (second shape))
                  ^MemorySegment data-seg (st/get-tensor-slice catq-mmap tname)
                  ^MemorySegment scale-seg (st/get-tensor-slice catq-mmap (str tname ".scales"))
                  wc-data (.toArray data-seg ValueLayout/JAVA_BYTE)
                  wc-scale (.toArray scale-seg ValueLayout/JAVA_SHORT)
                  deq (catq/dequantize-matrix-catq wc-data wc-scale rows cols 128)
                  snr (snr-db wb deq)
                  sim (cos-sim-vec wb deq)
                  mean-abs-orig (/ (areduce wb i s 0.0 (+ s (Math/abs (double (aget wb i))))) (double (alength wb)))
                  num-groups (quot cols 128)
                  total-scales (* rows num-groups)
                  scales-floats (float-array total-scales)
                  _ (dotimes [i total-scales]
                      (aset scales-floats i (catq/bf16-short->float (aget ^shorts wc-scale i))))
                  mean-scale (/ (areduce scales-floats i s 0.0 (+ s (double (aget scales-floats i)))) (double total-scales))
                  scale-ratio (/ mean-scale (max 1e-8 mean-abs-orig))

                  ;; Gamma stats from factors
                  factor-info (get projs-factors tname)
                  zr-vec (:zr factor-info)
                  gammas (when (seq zr-vec)
                           (mapv #(catq/sigmoid (double %)) zr-vec))
                  mean-gamma (if (seq gammas) (/ (reduce + 0.0 gammas) (double (count gammas))) 0.5)
                  pegged-count (if (seq gammas)
                                 (count (filter #(or (<= % 0.401) (>= % 0.599)) gammas))
                                 0)
                  pct-pegged (if (seq gammas) (* 100.0 (/ (double pegged-count) (double (count gammas)))) 0.0)]

              (println (format "L%02d   | %-10s | [%4d x %5d] | %7.2f  | %7.4f | %10.6f | %11.4f | %10.4f | %6.1f%%"
                               L p rows cols snr sim mean-abs-orig scale-ratio mean-gamma pct-pegged)))))))))
