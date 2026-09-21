(ns tools.poc-tl-transformer-block
  "Experiment E6 Driver: The Unified TL-Transformer Layer Block.
   Demonstrates an end-to-end hybrid forward pass synthesizing KG-masked self-attention (E2),
   cross-attention probing (E1), contrastive subspace projection (E3), and ephemeral fast-weight
   unbinding (E5) inside a single OpenXLA PJRT compiled layer block.
   Evaluates on AMD Radeon RX 7900 XTX via OpenXLA PJRT ROCm plugin."
  (:require [einsum.core :as xla]
            [einsum.models.tl-block :as tl]
            [clojure.string :as str])
  (:gen-class))

(def DEFAULT_OPTS
  {:backend :cpu
   :batch 1
   :seq-len 16
   :num-heads 4
   :head-dim 64
   :dim-mem 64
   :dff 1024
   :num-entities 16
   :num-cands 16
   :gamma 4.0
   :lambda-mem 0.25
   :tau 0.15
   :threshold 0.5})

(defn- normalize-args [args]
  (mapcat (fn [arg]
            (if (and (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args [args]
  (loop [remaining (vec (normalize-args args))
         opts DEFAULT_OPTS]
    (if (empty? remaining)
      opts
      (let [k (first remaining)
            v (second remaining)]
        (case k
          "--backend" (recur (subvec remaining 2) (assoc opts :backend (keyword v)))
          "--batch" (recur (subvec remaining 2) (assoc opts :batch (Long/parseLong v)))
          "--seq-len" (recur (subvec remaining 2) (assoc opts :seq-len (Long/parseLong v)))
          "--num-heads" (recur (subvec remaining 2) (assoc opts :num-heads (Long/parseLong v)))
          "--head-dim" (recur (subvec remaining 2) (assoc opts :head-dim (Long/parseLong v)))
          "--dim-mem" (recur (subvec remaining 2) (assoc opts :dim-mem (Long/parseLong v)))
          "--dff" (recur (subvec remaining 2) (assoc opts :dff (Long/parseLong v)))
          "--gamma" (recur (subvec remaining 2) (assoc opts :gamma (Double/parseDouble v)))
          "--lambda-mem" (recur (subvec remaining 2) (assoc opts :lambda-mem (Double/parseDouble v)))
          "--threshold" (recur (subvec remaining 2) (assoc opts :threshold (Double/parseDouble v)))
          (recur (subvec remaining 1) opts))))))

;; ==============================================================================
;; 1. Entities & Domain Setup
;; ==============================================================================

(def ENTITY_NAMES
  ["frontend-ui" "api-gateway" "auth-service" "billing-service"
   "database-primary" "database-replica" "redis-cache" "kafka-broker"
   "vault-secrets" "k8s-cluster-prod" "k8s-cluster-staging" "aws-us-east"
   "aws-us-west" "tls-cert-wildcard" "s3-bucket-logs" "role-admin"])

(defn- build-orthogonal-entity-table
  "Creates an orthonormal entity embedding table [N, D]."
  [n d]
  (let [arr (float-array (* n d))]
    (dotimes [i n]
      (dotimes [j d]
        (let [val (if (= (mod j n) i) 1.0 0.0)]
          (aset arr (+ (* i d) j) (float val)))))
    arr))

;; ==============================================================================
;; 2. Execution & Demonstration Routine
;; ==============================================================================

(defn -main [& args]
  (let [opts (parse-cli-args args)
        backend (:backend opts)
        b (long (:batch opts))
        l (long (:seq-len opts))
        h (long (:num-heads opts))
        dh (long (:head-dim opts))
        d (* h dh)
        dff (long (:dff opts))
        dm (long (:dim-mem opts))
        ne (long (:num-entities opts))
        nc (long (:num-cands opts))
        gamma (double (:gamma opts))
        lambda-mem (double (:lambda-mem opts))
        tau (double (:tau opts))
        threshold (double (:threshold opts))]

    (println "================================================================================")
    (println "     EXPERIMENT E6: THE UNIFIED TL-TRANSFORMER LAYER BLOCK (PJRT)")
    (println "================================================================================")
    (println (format "Backend:              [%s]" (name backend)))
    (println (format "Sequence / Batch:     L=%d tokens | B=%d" l b))
    (println (format "Attention Geometry:   H=%d heads x Dh=%d (Hidden Dim D=%d)" h dh d))
    (println (format "Feed-Forward (Dff):   %d (GeGLU MLP)" dff))
    (println (format "Memory Dimension:     Dm=%d (Relational Subspace)" dm))
    (println (format "Entity Universe:      %d entities | %d candidate targets" ne nc))
    (println (format "Gating Hyperparams:   gamma=%.1f (KG) | lambda=%.2f (Mem) | theta=%.2f"
                     gamma lambda-mem threshold))
    (println "================================================================================")

    ;; 1. Initialize OpenXLA PJRT Runtime
    (println "\n[1/5] Initializing OpenXLA PJRT Runtime...")
    (let [ctx (xla/init-backend! backend)
          _ (println (format "  ↳ Active PJRT Backend: [%s]" (name (:platform ctx))))

          ;; 2. Synthesize Model Parameters and In-Memory Facts
          _ (println "\n[2/5] Initializing TL-Transformer Parameters & Resident VRAM Cores...")
          e-cand (build-orthogonal-entity-table nc d)
          w-mem (float-array (* d dm))
          _ (dotimes [i (* d dm)]
              (aset w-mem i (* (float 0.1) (float (Math/cos (* 1.3 (double i)))))))

          ;; Self-attention projection weights
          w-q (float-array (* d h dh) (float 0.05))
          w-k (float-array (* d h dh) (float 0.05))
          w-v (float-array (* d h dh) (float 0.05))
          w-o (float-array (* h dh d) (float 0.05))

          ;; GeGLU MLP weights
          w-gate (float-array (* d dff) (float 0.05))
          w-up (float-array (* d dff) (float 0.05))
          w-down (float-array (* dff d) (float 0.05))

          ;; Sequence token mapping: token 1 = api-gateway (ent 1), token 5 = database-primary (ent 4)
          t-mat (float-array (* l ne) (float 0.0))
          _ (aset t-mat (+ (* 1 ne) 1) (float 1.0))
          _ (aset t-mat (+ (* 5 ne) 4) (float 1.0))

          ;; KG relational adjacency: api-gateway (1) -> database-primary (4)
          r-adj (float-array (* ne ne) (float 0.0))
          _ (aset r-adj (+ (* 1 ne) 4) (float 1.0))

          ;; Relational memory fast weights: inject fact api-gateway -> database-primary
          r-mem (float-array (* dm dm) (float 0.0))
          _ (dotimes [i dm]
              (let [phase (* 2.1 (double i))]
                (aset r-mem (+ (* (mod i dm) dm) (mod (+ i 3) dm))
                      (* (float 2.5) (float (Math/sin phase))))))

          ;; Input hidden states H [B, L, D]
          h-in (float-array (* b l d))
          _ (dotimes [i (* b l d)]
              (aset h-in i (* (float 0.1) (float (Math/sin (* 1.7 (double i)))))))

          thresh-arr (float-array 1 (float threshold))]

      ;; 3. Compile Unified OpenXLA PJRT Layer Block
      (println "\n[3/5] Compiling Unified TL-Transformer Layer Block (StableHLO Lowering)...")
      (let [t0-comp (System/nanoTime)
            exec (tl/compile-tl-transformer-block ctx b l h dh dff dm ne nc
                                                  {:gamma gamma
                                                   :lambda-mem lambda-mem
                                                   :tau tau})
            comp-ms (/ (double (- (System/nanoTime) t0-comp)) 1e6)]
        (println (format "  ↳ Compiled Full Block into StableHLO MLIR in %.2f ms" comp-ms))

        ;; 4. Warmup and Benchmark Forward Pass
        (println "\n[4/5] Benchmarking Forward Pass Execution...")
        (let [inputs {:H h-in
                      :Q w-q :K w-k :V w-v :W_o w-o
                      :T t-mat :R_adj r-adj
                      :W_mem w-mem :R_mem r-mem
                      :E_cand e-cand
                      :threshold thresh-arr
                      :W_gate w-gate :W_up w-up :W_down w-down}
              ;; Warmup run
              _ (tl/run-tl-transformer-block! exec inputs)
              iters 50
              t0-bench (System/nanoTime)
              out (loop [i 0 last-out nil]
                    (if (>= i iters)
                      last-out
                      (recur (inc i) (tl/run-tl-transformer-block! exec inputs))))
              bench-ms (/ (double (- (System/nanoTime) t0-bench)) (* 1e6 (double iters)))
              ^floats _h-out (:H_out out)
              ^floats cand-scores (:cand_scores out)
              ^floats clamped-scores (:clamped_scores out)
              ^floats attn-probs (:attn_probs out)]

          ;; 5. Empirical Results Analysis
          (println "\n================================================================================")
          (println "                  EXPERIMENT E6: EMPIRICAL OUTCOMES")
          (println "================================================================================")
          (println (format "Layer Execution Latency: %.3f ms per layer block" bench-ms))
          (println (format "Equivalent Full Model:   %.2f ms (32 layers @ %.3f ms)" (* 32 bench-ms) bench-ms))
          (println (format "Output Tensor Shape:     [B=%d, L=%d, D=%d] float32 = %.2f KB"
                           b l d (/ (* b l d 4.0) 1024.0)))

          ;; Check KG Attention Bias Effect: pos 5 attending back to pos 1
          (let [p-q 5, p-k 1 ;; database-primary attending to api-gateway
                head-0-attn (double (aget attn-probs (+ (* 0 l l) (* p-q l) p-k)))]
            (println (format "\nKG-Masked Attention Mass (token %d -> token %d): %.2f%% (Relational Prior Active)"
                             p-q p-k (* 100.0 head-0-attn))))

          ;; Check Deductive Gating
          (let [max-score (reduce max (map #(double (aget cand-scores %)) (range (* b l nc))))
                active-count (count (filter #(> (double %) 0.0) clamped-scores))]
            (println (format "Peak Relational Unbind Score: %.4f (Threshold = %.2f)" max-score threshold))
            (println (format "Deductively Grounded Tokens:  %d positions gated into residual stream"
                             active-count)))

          (println "================================================================================")
          (if (< bench-ms 15.0)
            (println ">>> SUCCESS: Unified TL-Transformer layer block executed within latency budget.")
            (println ">>> COMPLETED: Unified TL-Transformer layer block executed."))))))

  (System/exit 0))
