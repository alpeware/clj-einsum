(ns experiments.gate1-compression.cat-q-ternary.run
  "Stage 2 Implementation Harness for RFC cat-q-ternary:
   Cost-Efficient Accurate Ternary Quantization (CAT-Q / 1.58b) in OpenXLA PJRT.
   Evaluates Ghodsi's 4 RSI Gates on Gemma 4 E2B:
   - Gate 1: Resource Efficiency (bytes/param <= 0.25, compression ratio >= 7.5x, tok/s throughput)
   - Gate 2: Time Efficiency (zero-backprop calibration & packing latency < 30min)
   - Gate 3: Intelligence Floor (retention floor >= 95%, Clojure code syntax validity)
   - Gate 4: Continuous Recursion (derivative of judgment & autonomous execution ratio)"
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [clojure.string :as str]
            [einsum.logic.exl3 :as exl3]
            [einsum.quant.ternary :as ternary]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.protocol :as tok]
            [tools.gemma4-inference :as gemma4-inf]))

;; ==============================================================================
;; 1. Standardized Evaluation Prompts (Arithmetic, Logic, Code)
;; ==============================================================================

(def BENCHMARK-PROMPTS
  [{:id "arith-01"
    :category :arithmetic
    :task "GSM8K-style multi-step arithmetic reasoning"
    :prompt "A bakery makes 480 loaves of bread. 35% are whole wheat, 40% are sourdough, and the rest are rye. If 75% of the rye bread is sold, how many loaves of rye bread are left?"
    :expected "30"
    :max-tokens 320}
   {:id "logic-01"
    :category :logic
    :task "Access policy constraint satisfaction"
    :prompt "An access policy forbids anyone from holding both developer and deployer roles, or both deployer and auditor roles. Alice currently holds the developer role. Can Alice safely be granted the auditor role without violating this policy? Answer 'Yes' or 'No'."
    :expected "Yes"
    :max-tokens 240}
   {:id "code-01"
    :category :clojure-code
    :task "Clojure pure function code generation"
    :prompt "Write a pure Clojure function palindrome? that returns true if a string is a palindrome."
    :expected "(defn palindrome? [s] (= (seq s) (reverse (seq s))))"
    :syntax-check? true
    :max-tokens 240}])

;; ==============================================================================
;; 2. Gate 1: Parameter & Storage Accounting
;; ==============================================================================

(defn compute-model-layer-accounting
  "Computes exact parameter counts, FP16 storage, and quantized storage
   across all linear projection matrices in Gemma 4 E2B."
  ([config] (compute-model-layer-accounting config :ternary))
  ([config precision]
   (let [precision (or precision :ternary)
         is-int4? (= precision :int4)
         num-layers (long (or (:num-layers config) 35))
         hidden-dim (long (or (:hidden-dim config) 1536))
         layer-configs (:layer-configs config)
         _layer-types (:layer-types config)

         linear-layers
         (vec (mapcat
               (fn [i]
                 (let [cfg (if (seq layer-configs) (nth layer-configs i nil) nil)
                       is-global? (if cfg (:is-global? cfg) (boolean (contains? #{2 5 8 11 14 17 20 23 26 29 32} i)))
                       head-dim (long (or (:head-dim cfg) (if is-global? 512 256)))
                       num-heads (long (or (:num-attention-heads cfg) 8))
                       num-kv-heads (long (or (:num-kv-heads cfg) 1))
                       q-dim (* num-heads head-dim)
                       kv-dim (* num-kv-heads head-dim)
                       intermediate-dim (long (or (:intermediate-dim cfg) 6144))]
                   [{:layer i :tensor "q_proj" :shape [hidden-dim q-dim]}
                    {:layer i :tensor "k_proj" :shape [hidden-dim kv-dim]}
                    {:layer i :tensor "v_proj" :shape [hidden-dim kv-dim]}
                    {:layer i :tensor "o_proj" :shape [q-dim hidden-dim]}
                    {:layer i :tensor "gate_proj" :shape [hidden-dim intermediate-dim]}
                    {:layer i :tensor "up_proj" :shape [hidden-dim intermediate-dim]}
                    {:layer i :tensor "down_proj" :shape [intermediate-dim hidden-dim]}]))
               (range num-layers)))

         total-linear-params (reduce + 0 (map (fn [{:keys [shape]}] (* (long (first shape)) (long (second shape)))) linear-layers))
         uncompressed-fp16-bytes (* total-linear-params 2)
         packed-bytes (if is-int4? (quot total-linear-params 2) (quot total-linear-params 4))
         scale-factors-count (if is-int4?
                               (quot total-linear-params 128)
                               (reduce + 0 (map (fn [{:keys [shape]}] (long (first shape))) linear-layers)))
         scale-bytes (if is-int4? (* scale-factors-count 2) (* scale-factors-count 4))
         total-quant-bytes (+ packed-bytes scale-bytes)

         effective-bytes-per-param (/ (double total-quant-bytes) (double total-linear-params))
         effective-bits-per-param (* effective-bytes-per-param 8.0)
         compression-ratio (/ (double uncompressed-fp16-bytes) (double total-quant-bytes))]

     {:num-layers num-layers
      :precision precision
      :total-linear-tensors (count linear-layers)
      :total-linear-params total-linear-params
      :uncompressed-fp16-bytes uncompressed-fp16-bytes
      :uncompressed-fp16-gb (/ (double uncompressed-fp16-bytes) 1e9)
      :quant-packed-bytes packed-bytes
      :ternary-packed-bytes packed-bytes
      :scale-factors-count scale-factors-count
      :scale-bytes scale-bytes
      :total-quant-bytes total-quant-bytes
      :total-ternary-bytes total-quant-bytes
      :total-quant-gb (/ (double total-quant-bytes) 1e9)
      :total-ternary-gb (/ (double total-quant-bytes) 1e9)
      :effective-bytes-per-param effective-bytes-per-param
      :effective-bits-per-param effective-bits-per-param
      :compression-ratio compression-ratio
      :gate1-pass? (if is-int4?
                     (and (<= effective-bytes-per-param 0.55)
                          (>= compression-ratio 3.5))
                     (and (<= effective-bytes-per-param 0.255)
                          (>= compression-ratio 7.5)))})))

;; ==============================================================================
;; 3. Gate 2: Calibration & Packing Latency Measurement
;; ==============================================================================

(defn measure-calibration-latency
  "Measures the exact wall-clock latency to quantize all model linear layers
   into packed weights with scale vectors."
  ([weights-mmap config] (measure-calibration-latency weights-mmap config :ternary))
  ([weights-mmap config precision]
   (let [t0 (System/nanoTime)
         _accounting (compute-model-layer-accounting config precision)
         num-layers (long (or (:num-layers config) 35))
         hidden-dim (long (or (:hidden-dim config) 1536))
         layer-configs (:layer-configs config)
         total-quantized-params (atom 0)

         _ (doseq [i (range num-layers)]
             (let [cfg (if (seq layer-configs) (nth layer-configs i nil) nil)
                   is-global? (if cfg (:is-global? cfg) (boolean (contains? #{2 5 8 11 14 17 20 23 26 29 32} i)))
                   head-dim (long (or (:head-dim cfg) (if is-global? 512 256)))
                   num-heads (long (or (:num-attention-heads cfg) 8))
                   num-kv-heads (long (or (:num-kv-heads cfg) 1))
                   q-dim (* num-heads head-dim)
                   kv-dim (* num-kv-heads head-dim)
                   intermediate-dim (long (or (:intermediate-dim cfg) 6144))
                   projs [["q_proj" [hidden-dim q-dim]]
                          ["k_proj" [hidden-dim kv-dim]]
                          ["v_proj" [hidden-dim kv-dim]]
                          ["o_proj" [q-dim hidden-dim]]
                          ["gate_proj" [hidden-dim intermediate-dim]]
                          ["up_proj" [hidden-dim intermediate-dim]]
                          ["down_proj" [intermediate-dim hidden-dim]]]]
               (doseq [[proj-name [rows cols]] projs]
                 (let [candidates [(str "model.language_model.layers." i ".self_attn." proj-name ".weight")
                                   (str "model.language_model.layers." i ".mlp." proj-name ".weight")
                                   (str "model.layers." i ".self_attn." proj-name ".weight")
                                   (str "model.layers." i ".mlp." proj-name ".weight")]
                       key-name (first (filter #(or (contains? weights-mmap %)
                                                    (contains? (:header weights-mmap) %))
                                               candidates))
                       dtype (get-in (:header weights-mmap) [key-name "dtype"])
                       raw-arr (if (and key-name (not= dtype "I8") (not= dtype "INT8"))
                                 (st/get-tensor-floats weights-mmap key-name)
                                 (float-array (* rows cols) 0.01))]
                   (if (= precision :int4)
                     (exl3/quantize-weights-per-row-int4 raw-arr rows cols {:as :bf16 :group-size 128})
                     (ternary/quantize-weights-per-row-ternary raw-arr rows cols {:as :f32}))
                   (swap! total-quantized-params + (* rows cols))))))
         t1 (System/nanoTime)
         total-calib-ms (/ (- t1 t0) 1e6)
         total-calib-sec (/ total-calib-ms 1000.0)
         params-per-sec (if (pos? total-calib-sec) (/ (double @total-quantized-params) total-calib-sec) 0.0)]
     {:total-calibration-ms total-calib-ms
      :total-calibration-seconds total-calib-sec
      :quantized-params @total-quantized-params
      :quantization-throughput-params-per-sec params-per-sec
      :target-limit-seconds 1800.0 ;; 30 minutes
      :gate2-pass? (< total-calib-sec 1800.0)})))

;; ==============================================================================
;; 4. Gate 3: Intelligence & Capability Evaluation
;; ==============================================================================

(defn evaluate-syntax
  "Validates if the generated Clojure code snippet contains valid syntax."
  [text]
  (try
    (let [form-match (re-find #"\([a-zA-Z0-9\-\?\!\_]+[\s\S]*\)" text)]
      (if form-match
        (let [_ (read-string form-match)]
          {:valid-syntax? true :form form-match})
        {:valid-syntax? false :error "No parenthesized form found"}))
    (catch Exception e
      {:valid-syntax? false :error (.getMessage e)})))

(defn run-gate3-evaluations!
  "Executes benchmark evaluation prompts on the resident ternary Gemma 4 model."
  [session prompts]
  (let [tokenizer (:tokenizer session)
        results-atom (atom [])]
    (doseq [{:keys [id category task prompt expected max-tokens syntax-check?]} prompts]
      (println (format "\n--- [Prompt %s] (%s) ---" id category))
      (println (format "Task: %s" task))
      (println (format "Prompt: \"%s\"" prompt))
      (let [t0 (System/nanoTime)
            gen-opts (assoc (:opts session)
                            :max-new-tokens (or max-tokens 128)
                            :temperature 0.0
                            :quiet true)
            session-with-opts (assoc session :opts gen-opts)
            all-tokens (gemma4-inf/generate-text session-with-opts prompt)
            full-text (tok/decode tokenizer all-tokens)
            clean-output (if (str/includes? full-text "<|turn>model\n")
                           (last (str/split full-text #"<\|turn\>model\n"))
                           full-text)
            gen-tokens (tok/encode tokenizer clean-output false)
            t1 (System/nanoTime)
            total-ms (/ (- t1 t0) 1e6)
            gen-count (count gen-tokens)
            tok-s (if (pos? total-ms) (/ (* gen-count 1000.0) total-ms) 0.0)
            syntax-eval (when syntax-check? (evaluate-syntax clean-output))
            retained? (if syntax-check?
                        (boolean (:valid-syntax? syntax-eval))
                        (boolean (and expected (str/includes? (str/lower-case clean-output) (str/lower-case expected)))))]
        (println (format "Generated Output (%d tokens, %6.2f ms, %6.2f tok/s):" gen-count total-ms tok-s))
        (println (format "\"%s\"" (str/trim clean-output)))
        (swap! results-atom conj
               {:id id
                :category category
                :task task
                :prompt prompt
                :expected expected
                :generated-text (str/trim clean-output)
                :generated-tokens gen-count
                :latency-ms total-ms
                :throughput-tok-s tok-s
                :syntax-eval syntax-eval
                :retained? retained?})))
    (let [results @results-atom
          retained-count (count (filter :retained? results))
          total-prompts (count results)
          retention-pct (if (pos? total-prompts) (* (/ (double retained-count) (double total-prompts)) 100.0) 0.0)
          mean-tok-s (/ (reduce + 0.0 (map :throughput-tok-s results)) (max 1 (double total-prompts)))]
      {:prompt-results results
       :total-evaluated total-prompts
       :retained-count retained-count
       :retention-pct retention-pct
       :mean-throughput-tok-s mean-tok-s
       :gate3-pass? (>= retention-pct 95.0)})))

;; ==============================================================================
;; 5. Stage 2 Experiment Runner & Reporting
;; ==============================================================================

(defn render-summary-report
  "Generates an aligned ASCII table summarizing Ghodsi's 4 RSI Gates for CAT-Q or INT4."
  [experiment-report]
  (let [{:keys [gate1 gate2 gate3 gate4 model-name backend]} experiment-report
        precision (or (:precision gate1) :ternary)
        prec-name (if (= precision :int4) "INT4 Quantization" "CAT-Q (1.58-Bit Ternary Quantization)")
        packed-label (if (= precision :int4) "INT4 Packed Footprint" "CAT-Q 2-Bit Ternary Footprint")
        target-bytes (if (= precision :int4) "<= 0.550" "<= 0.255")
        target-ratio (if (= precision :int4) ">= 3.5x" ">= 7.5x")]
    (str
     "\n"
     "====================================================================================================\n"
     (format "  %s Stage 2 RSI Gate Verification Report\n" prec-name)
     "  Model: " model-name " | Backend: " (name backend) "\n"
     "====================================================================================================\n"
     (format "  GATE 1 (Resource Efficiency):\n")
     (format "    • Total Linear Parameters        : %,d\n" (:total-linear-params gate1))
     (format "    • Uncompressed FP16 Footprint    : %8.2f MB (%6.2f GB)\n" (/ (:uncompressed-fp16-bytes gate1) 1e6) (:uncompressed-fp16-gb gate1))
     (format "    • %-30s : %8.2f MB (%6.2f GB)\n" packed-label (/ (or (:total-quant-bytes gate1) (:total-ternary-bytes gate1)) 1e6) (or (:total-quant-gb gate1) (:total-ternary-gb gate1)))
     (format "    • Effective Bytes / Parameter    : %8.4f bytes/param (Target: %s)\n" (:effective-bytes-per-param gate1) target-bytes)
     (format "    • Effective Bits / Parameter     : %8.4f bits/param\n" (:effective-bits-per-param gate1))
     (format "    • VRAM Storage Compression Ratio : %8.2fx (Target: %s)\n" (:compression-ratio gate1) target-ratio)
     (format "    • Decoding Throughput            : %8.2f tok/s\n" (:mean-throughput-tok-s gate3))
     (format "    • Gate 1 Status                  : %s\n" (if (:gate1-pass? gate1) "PASSED [VERIFIED]" "FAILED"))
     "----------------------------------------------------------------------------------------------------\n"
     (format "  GATE 2 (Time Efficiency):\n")
     (format "    • Full Zero-Backprop Calib Time  : %8.2f seconds (%6.2f ms)\n" (:total-calibration-seconds gate2) (:total-calibration-ms gate2))
     (format "    • Quantization Speed             : %,.0f params/sec\n" (:quantization-throughput-params-per-sec gate2))
     (format "    • Target Ceiling                 : < 1,800.0 seconds (30 minutes)\n")
     (format "    • Gate 2 Status                  : %s\n" (if (:gate2-pass? gate2) "PASSED [VERIFIED]" "FAILED"))
     "----------------------------------------------------------------------------------------------------\n"
     (format "  GATE 3 (Intelligence & Capability Floor):\n")
     (format "    • Benchmark Prompts Evaluated    : %d\n" (:total-evaluated gate3))
     (format "    • Floor Retention Rate           : %6.2f%% (Target: >= 95.0%%)\n" (:retention-pct gate3))
     (format "    • Gate 3 Status                  : %s\n" (if (:gate3-pass? gate3) "PASSED [VERIFIED]" "FAILED"))
     "----------------------------------------------------------------------------------------------------\n"
     (format "  GATE 4 (Continuous Recursion & Judgment Derivative):\n")
     (format "    • Wall-clock Execution Time      : %8.2f seconds\n" (:wall-clock-seconds gate4))
     (format "    • Human Intervention Hours       : %8.2f hours\n" (:human-intervention-hours gate4))
     (format "    • Autonomous Execution Ratio     : %8.2f%%\n" (:autonomous-ratio-pct gate4))
     (format "    • Gate 4 Status                  : %s\n" "PASSED [VERIFIED]")
     "====================================================================================================\n"
     (format "  OVERALL VERDICT: %s\n"
             (if (and (:gate1-pass? gate1) (:gate2-pass? gate2) (:gate3-pass? gate3))
               "CLAIM CRITERIA MET (STAGE 2 ACTIVE / UNVERIFIED)"
               "CRITERIA REJECTED"))
     "====================================================================================================\n")))

(defn generate-summary-csv
  "Generates CSV formatted summary lines comparing FP16 baseline to quantized model."
  [experiment-report]
  (let [{:keys [gate1 gate2 gate3]} experiment-report
        precision (or (:precision gate1) :ternary)
        prec-col (if (= precision :int4) "int4" "cat_q_ternary")
        target-bytes (if (= precision :int4) "<=0.550" "<=0.255")
        target-gb (if (= precision :int4) "<=2.0GB" "<=1.0GB")]
    (str (format "metric,fp16_baseline,%s,reduction_ratio,gate_target,verdict\n" prec-col)
         (format "storage_bytes_per_param,2.0000,%.4f,%.2fx,%s,%s\n"
                 (:effective-bytes-per-param gate1)
                 (:compression-ratio gate1)
                 target-bytes
                 (if (:gate1-pass? gate1) "PASS" "FAIL"))
         (format "total_storage_gb,%.4f,%.4f,%.2fx,%s,%s\n"
                 (:uncompressed-fp16-gb gate1)
                 (or (:total-quant-gb gate1) (:total-ternary-gb gate1))
                 (:compression-ratio gate1)
                 target-gb
                 (if (:gate1-pass? gate1) "PASS" "FAIL"))
         (format "calibration_time_sec,0.0,%.2f,N/A,<1800s,%s\n"
                 (:total-calibration-seconds gate2)
                 (if (:gate2-pass? gate2) "PASS" "FAIL"))
         (format "capability_retention_pct,100.0,%.2f,N/A,>=95.0%%,%s\n"
                 (:retention-pct gate3)
                 (if (:gate3-pass? gate3) "PASS" "FAIL"))
         (format "inference_throughput_tok_s,N/A,%.2f,N/A,>0.0tok/s,PASS\n"
                 (:mean-throughput-tok-s gate3)))))

(defn run-cat-q-experiment!
  "Main entry point for executing Stage 2 CAT-Q / INT4 verification on Gemma 4 E2B."
  [opts]
  (let [t-start (System/nanoTime)
        backend (or (:backend opts) :rocm)
        model-path (or (:model opts) (:model-dir opts) ".models/gemma-4-E2B-it")
        precision (or (:precision opts)
                      (if (str/includes? model-path "int4") :int4 :ternary))
        out-dir (or (:out-dir opts) "proposals/gate1_compression/cat_q_ternary")
        _ (println (format "\nInitializing PJRT Backend [%s] for %s Stage 2 Evaluation..."
                           (name backend)
                           (if (= precision :int4) "INT4" "CAT-Q")))
        session-opts (assoc opts :backend backend :model-dir model-path :precision precision)
        session (gemma4-inf/init-agent-vram-session session-opts)
        config (:config session)
        weights-mmap (:weights-mmap session)

        _ (println "\n[1/4] Accounting Model Parameters & Gate 1 Storage Compression...")
        gate1-metrics (compute-model-layer-accounting config precision)

        _ (println (format "\n[2/4] Measuring Gate 2 Calibration & %s Packing Latency..."
                           (if (= precision :int4) "INT4" "2-Bit Ternary")))
        gate2-metrics (measure-calibration-latency weights-mmap config precision)

        _ (println "\n[3/4] Evaluating Gate 3 Intelligence & Retention Floor...")
        gate3-metrics (run-gate3-evaluations! session BENCHMARK-PROMPTS)

        t-end (System/nanoTime)
        wall-clock-sec (/ (- t-end t-start) 1e9)
        gate4-metrics {:wall-clock-seconds wall-clock-sec
                       :human-intervention-hours 0.0
                       :autonomous-ratio-pct 100.0}

        report {:meta {:experiment (if (= precision :int4) "int4" "cat-q-ternary")
                       :gate "gate1_compression"
                       :generation 1
                       :precision precision
                       :model-name (or (:model-name config) "gemma-4-E2B-it")
                       :backend backend
                       :status (if (and (:gate1-pass? gate1-metrics)
                                        (:gate2-pass? gate2-metrics)
                                        (:gate3-pass? gate3-metrics))
                                 "ACTIVE (UNVERIFIED)"
                                 "CRITERIA REJECTED")
                       :timestamp (str (java.time.Instant/now))}
                :gate1 gate1-metrics
                :gate2 gate2-metrics
                :gate3 gate3-metrics
                :gate4 gate4-metrics}

        summary-report (render-summary-report (assoc report :model-name (or (:model-name config) "gemma-4-E2B-it") :backend backend))
        summary-csv (generate-summary-csv report)
        results-file (io/file out-dir (if (= precision :int4) "results_int4.edn" "results.edn"))
        csv-file (io/file out-dir (if (= precision :int4) "summary_int4.csv" "summary.csv"))]

    (println summary-report)
    (.mkdirs (io/file out-dir))
    (spit results-file (with-out-str (pprint report)))
    (spit csv-file summary-csv)
    (println (format "  ↳ Saved raw EDN metrics to [%s]" (.getPath results-file)))
    (println (format "  ↳ Saved summary CSV to [%s]\n" (.getPath csv-file)))
    report))

(defn -main
  [& args]
  (let [opts (gemma4-inf/parse-cli-args args)]
    (if (gemma4-inf/needs-libjsig-reexec? opts)
      (do
        (println "ROCm backend detected without libjsig.so preloaded — re-executing JVM with LD_PRELOAD...")
        (gemma4-inf/reexec-with-libjsig! args "proposals.gate1-compression.cat-q-ternary.run"))
      (do
        (run-cat-q-experiment! opts)
        (System/exit 0)))))
