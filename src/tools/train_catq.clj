(ns tools.train-catq
  "Command-line entrypoint for training CAT-Q Learnable Modulation and LoRA Refinement
   directly on open-weights Gemma 4 models (E2B, E4B, 12B, etc.) using sequential PJRT
   activation-forwarded calibration and exporting resident 2-bit ternary Safetensors
   and catq_factors.edn artifacts."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [einsum.compiler.pjrt :as pjrt]
            [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [einsum.models.gemma4 :as gemma4]
            [einsum.quant.catq :as catq]
            [einsum.runtime.safetensors :as st]
            [einsum.runtime.tokenizer.protocol :as tok]
            [tools.gemma4-inference :as gemma4-inf]))

(set! *warn-on-reflection* true)

;; ==============================================================================
;; 1. Calibration Dataset (Domain-Diverse Reasoning, Code, Math, Logic)
;; ==============================================================================

(def DEFAULT-CALIBRATION-PROMPTS
  ["A bakery makes 480 loaves of bread. 35% are whole wheat, 40% are sourdough, and the rest are rye. If 75% of the rye bread is sold, how many loaves of rye bread are left?"
   "An access policy forbids anyone from holding both developer and deployer roles, or both deployer and auditor roles. Alice currently holds the developer role. Can Alice safely be granted the auditor role without violating this policy? Answer 'Yes' or 'No'."
   "Write a pure Clojure function palindrome? that returns true if a string is a palindrome."
   "Explain the differences between transient and persistent data structures in Clojure and why persistent structures guarantee thread safety without locks."
   "Calculate the derivative of f(x) = x^3 - 4x^2 + 2x - 7 at x = 3 and explain each differentiation rule used."
   "In a distributed system with Raft consensus, explain what happens when the leader network-partitions from the majority of followers."
   "Implement a pure tail-recursive function in Clojure to compute the greatest common divisor of two integers using the Euclidean algorithm."
   "Explain how Pedro Domingos' Declarative Tensor Logic unifies neural networks and relational knowledge bases into first-order logical queries."])

;; ==============================================================================
;; 2. Linear Projection Filtering & Dimension Discovery
;; ==============================================================================

(defn quantizable-weight?
  "Returns true if tensor is a linear projection weight eligible for CAT-Q ternary quantization."
  [tensor-name shape]
  (boolean
   (and (vector? shape)
        (= (count shape) 2)
        (str/ends-with? tensor-name ".weight")
        (or (re-find #"\.(?:q|k|v|o|gate|up|down)_proj\.weight$" tensor-name)
            (and (or (str/includes? tensor-name "self_attn")
                     (str/includes? tensor-name "mlp"))
                 (not (str/includes? tensor-name "norm"))
                 (not (str/includes? tensor-name "scale"))))
        (not (str/includes? tensor-name "embed"))
        (not (str/includes? tensor-name "norm"))
        (not (str/includes? tensor-name "scale"))
        (not (str/ends-with? tensor-name "bias"))
        (zero? (mod (long (second shape)) 128)))))

;; ==============================================================================
;; 3. Core Sequential Activation-Forwarded Model Calibration Engine
;; ==============================================================================

(defn calibrate-model-catq!
  "Calibrates CAT-Q modulation factors and OBS error compensation across all linear projections
   in `model-dir` using sequential activation-forwarded PJRT execution and streams the
   resulting 2-bit ternary model and artifacts to `output-dir`."
  [opts]
  (let [model-dir (or (:model opts) ".models/gemma-4-E2B-it")
        out-dir (or (:output-dir opts) (str model-dir "-catq"))
        backend (keyword (or (:backend opts) :rocm))
        max-seq-len (long (or (:max-seq-len opts) 128))
        group-size (long (or (:group-size opts) 128))
        skip-layers (set (or (:skip-layers opts) #{}))
        quiet? (boolean (:quiet opts))

        t0 (System/nanoTime)
        _ (when-not quiet?
            (println "\n====================================================================================================")
            (println "  CAT-Q Native Training & Quantization Pipeline (Sequential PJRT Activation Engine)")
            (println "  Source Model: " model-dir)
            (println "  Target Output:" out-dir)
            (println (format "  Backend: %s | Max Seq Len: %d | Group Size: %d"
                             (name backend) max-seq-len group-size))
            (println "====================================================================================================\n"))

        ;; 1. Initialize PJRT VRAM session
        session (gemma4-inf/init-agent-vram-session {:model model-dir :backend backend :quiet true :precision :bf16} max-seq-len)
        ctx (:ctx session)
        config (:config session)
        tok (:tokenizer session)
        num-layers (long (or (:num-layers config) 35))
        hidden-dim (long (or (:hidden-dim config) 1536))

        ;; 2. Format calibration prompts into individual chat sequences
        prompts (or (:prompts opts) DEFAULT-CALIBRATION-PROMPTS)
        calib-inputs
        (mapv (fn [p]
                (let [chat-str (str "<bos><|turn>user\n" p "<turn|>\n<|turn>model\n")
                      tokens (tok/encode tok chat-str false)
                      plen (min max-seq-len (count tokens))
                      arr (int-array max-seq-len)
                      _ (dotimes [i plen] (aset arr i (int (nth tokens i))))
                      buf (xla/buffer-from-host-buffer ctx (:client ctx) arr [1 max-seq-len] 4)]
                  {:buf buf :len plen}))
              prompts)

        ;; 3. Compile multi-layer forward activation graph targeting layer inputs
        cfg (assoc config :max-seq-len max-seq-len :last-token-only? false)
        targets (vec (mapcat (fn [i]
                               [(keyword (str "x_norm1_" i))
                                (keyword (str "ctx_flat_" i))
                                (keyword (str "x_norm2_" i))
                                (keyword (str "mlp_act_" i))])
                             (range num-layers)))
        target-sizes
        (into {}
              (mapcat (fn [i]
                        (let [layer-cfg (nth (:layer-configs config) i)
                              q-dim (long (:q-dim layer-cfg))
                              mlp-dim (long (:mlp-dim layer-cfg))]
                          [[(keyword (str "x_norm1_" i)) (* max-seq-len hidden-dim)]
                           [(keyword (str "ctx_flat_" i)) (* max-seq-len q-dim)]
                           [(keyword (str "x_norm2_" i)) (* max-seq-len hidden-dim)]
                           [(keyword (str "mlp_act_" i)) (* max-seq-len mlp-dim)]]))
                      (range num-layers)))

        invars (gemma4-inf/build-tensor-logic-invars cfg max-seq-len)
        ast (gemma4/gemma4-model-ast cfg)
        compiled (logic/compile-ast ctx "train_catq_seq_forward" invars ast targets)

        mmap (:weights-mmap session)
        header (:header mmap)
        tensor-names (sort (filter #(not= % "__metadata__") (keys header)))
        quant-specs (atom [])
        factors-map (atom {:model-dir model-dir
                           :hidden-dim hidden-dim
                           :num-layers num-layers
                           :group-size group-size
                           :projections {}})
        total-quantized-params (atom 0)

        device-weights (atom (vec (:device-weights session)))
        has-ple? (pos? (long (or (:total-pl-dim config) 0)))
        bufs-per-layer (if has-ple? 17 14)
        layer-base-fn (fn [L] (+ 4 (* L bufs-per-layer)))]

    (when-not quiet?
      (println (format "Starting Multi-Prompt Sequential OBS Calibration across %d Layers...\n" num-layers)))

    ;; 4. Sequential Layer-by-Layer Calibration Loop
    (dotimes [L num-layers]
      (let [layer-cfg (nth (:layer-configs config) L)
            q-dim (long (:q-dim layer-cfg))
            kv-dim (long (:kv-dim layer-cfg))
            mlp-dim (long (:mlp-dim layer-cfg))
            base-idx (layer-base-fn L)
            projs [["q_proj" (str "model.language_model.layers." L ".self_attn.q_proj.weight") :x-norm1 q-dim hidden-dim (+ base-idx 2) [q-dim hidden-dim]]
                   ["k_proj" (str "model.language_model.layers." L ".self_attn.k_proj.weight") :x-norm1 kv-dim hidden-dim (+ base-idx 3) [kv-dim hidden-dim]]
                   ["v_proj" (str "model.language_model.layers." L ".self_attn.v_proj.weight") :x-norm1 kv-dim hidden-dim (+ base-idx 4) [kv-dim hidden-dim]]
                   ["o_proj" (str "model.language_model.layers." L ".self_attn.o_proj.weight") :ctx-flat hidden-dim q-dim (+ base-idx 5) [hidden-dim q-dim]]
                   ["gate_proj" (str "model.language_model.layers." L ".mlp.gate_proj.weight") :x-norm2 mlp-dim hidden-dim (+ base-idx 11) [mlp-dim hidden-dim]]
                   ["up_proj" (str "model.language_model.layers." L ".mlp.up_proj.weight") :x-norm2 mlp-dim hidden-dim (+ base-idx 12) [mlp-dim hidden-dim]]
                   ["down_proj" (str "model.language_model.layers." L ".mlp.down_proj.weight") :mlp-act hidden-dim mlp-dim (+ base-idx 13) [hidden-dim mlp-dim]]]]
        (if (contains? skip-layers L)
          (do
            (when-not quiet?
              (println (format "  [%2d/%2d] Skipping Layer %2d (Preserving Uncompressed BF16)"
                               (inc L) num-layers L)))
            (doseq [[_pname tname _act-key _rows _cols _buf-idx shape] projs]
              (let [slice (st/get-tensor-slice mmap tname)
                    dtype (get-in header [tname "dtype"] "BF16")
                    spec {:name tname
                          :quantize? false
                          :shape shape
                          :dtype dtype
                          :segment slice}]
                (swap! quant-specs conj spec))))
          (let [t-layer-start (System/nanoTime)
                layer-metrics (atom [])
                ;; Forward pass across all calibration prompts with weights quantized up to L-1
                prompt-acts
                (mapv (fn [{:keys [buf len]}]
                        (let [outs (pjrt/execute-executable ctx (or (:handle compiled) compiled) (into [buf] @device-weights) (count targets))
                              get-act (fn [target-key]
                                        (let [idx (.indexOf ^java.util.List targets target-key)
                                              sz (long (get target-sizes target-key (* max-seq-len hidden-dim)))]
                                          (pjrt/buffer-to-host-buffer ctx (nth outs idx) sz :bf16)))
                              act-map {:len (long len)
                                       :x-norm1 (get-act (keyword (str "x_norm1_" L)))
                                       :ctx-flat (get-act (keyword (str "ctx_flat_" L)))
                                       :x-norm2 (get-act (keyword (str "x_norm2_" L)))
                                       :mlp-act (get-act (keyword (str "mlp_act_" L)))}]
                          (doseq [b outs]
                            (pjrt/destroy-buffer! ctx b))
                          act-map))
                      calib-inputs)]
            (doseq [[_pname tname act-key rows cols buf-idx shape] projs]
              (let [^floats w (st/get-tensor-floats mmap tname)
                    total-tokens (long (reduce + 0 (map :len prompt-acts)))
                    combined-act (float-array (* total-tokens (long cols)))
                    _ (loop [items prompt-acts dst-token-offset (long 0)]
                        (when-let [item (first items)]
                          (let [len (long (:len item))
                                ^floats src-act (get item act-key)]
                            (System/arraycopy src-act 0 combined-act (int (* dst-token-offset (long cols))) (int (* len (long cols))))
                            (recur (rest items) (+ dst-token-offset len)))))
                    res (catq/quantize-matrix-obs combined-act w total-tokens rows cols group-size)
                    ^floats deq (:w-deq res)
                    w-norm-sq (double (areduce w i s 0.0 (+ s (* (aget w i) (aget w i)))))
                    deq-norm-sq (double (areduce deq i s 0.0 (+ s (* (aget deq i) (aget deq i)))))
                    dot (double (areduce w i s 0.0 (+ s (* (aget w i) (aget deq i)))))
                    diff-sq (double (areduce w i s 0.0 (let [d (- (aget w i) (aget deq i))] (+ s (* d d)))))
                    snr (* 10.0 (Math/log10 (/ (Math/max 1e-12 w-norm-sq) (Math/max 1e-12 diff-sq))))
                    cos-sim (/ dot (Math/sqrt (* (Math/max 1e-12 w-norm-sq) (Math/max 1e-12 deq-norm-sq))))
                    _ (swap! layer-metrics conj {:snr snr :cos-sim cos-sim})
                    spec {:name tname
                          :quantize? true
                          :shape (:shape res)
                          :scale-name (str tname ".scales")
                          :scale-shape (:scale-shape res)
                          :data (:data res)
                          :scales (:scales res)}
                    w-shorts (catq/floats->bf16-shorts deq)
                    old-dev-buf (nth @device-weights buf-idx)
                    new-dev-buf (xla/buffer-from-host-buffer ctx (:client ctx) w-shorts shape 13)]
                (swap! quant-specs conj spec)
                (swap! total-quantized-params + (* (long rows) (long cols)))
                (swap! factors-map assoc-in [:projections tname]
                       {:rows rows :cols cols :zs (vec (:zs res)) :zr (vec (:zr res))})
                (swap! device-weights assoc buf-idx new-dev-buf)
                (when old-dev-buf
                  (pjrt/destroy-buffer! ctx old-dev-buf))))

            (when-not quiet?
              (let [t-layer-end (System/nanoTime)
                    n-m (count @layer-metrics)
                    avg-snr (if (pos? n-m) (/ (double (reduce + 0.0 (map :snr @layer-metrics))) (double n-m)) 0.0)
                    avg-cos (if (pos? n-m) (/ (double (reduce + 0.0 (map :cos-sim @layer-metrics))) (double n-m)) 0.0)]
                (println (format "  [%2d/%2d] Calibrated Layer %2d in %6.2f ms | Avg SNR: %5.2f dB, CosSim: %.4f | (Activations Forwarded to Layer %2d)"
                                 (inc L) num-layers L (/ (- t-layer-end t-layer-start) 1e6) avg-snr avg-cos (inc L)))))))))

    ;; 5. Collect non-quantizable weights (embeddings, norms, biases, PLE)
    (let [quantized-names (set (map :name @quant-specs))]
      (doseq [t-name tensor-names]
        (when-not (contains? quantized-names t-name)
          (let [shape (get-in header [t-name "shape"])
                slice (st/get-tensor-slice mmap t-name)
                spec {:name t-name
                      :quantize? false
                      :shape shape
                      :dtype (get-in header [t-name "dtype"])
                      :segment slice}]
            (swap! quant-specs conj spec)))))

    ;; 6. Persist Model and Factors to Disk
    (when-not quiet?
      (println (format "\nSerializing CAT-Q factors (catq_factors.edn) and 2-bit Safetensors to [%s]..." out-dir)))
    (catq/save-catq-model! model-dir out-dir @factors-map @quant-specs)
    (doseq [w @device-weights]
      (try (pjrt/destroy-buffer! ctx w) (catch Exception _ nil)))

    (let [t1 (System/nanoTime)
          total-sec (/ (- t1 t0) 1e9)
          total-params @total-quantized-params
          out-model-file (io/file out-dir "model.safetensors")
          out-size-gb (/ (double (.length out-model-file)) 1e9)
          speed-params-sec (/ (double total-params) (max 0.001 total-sec))]
      (when-not quiet?
        (println "\n====================================================================================================")
        (println "  CAT-Q Calibration & Quantization Complete!")
        (println (format "  Total Quantized Parameters: %,d" total-params))
        (println (format "  Output Model Size:          %.2f GB" out-size-gb))
        (println (format "  Elapsed Time:               %.2f seconds (%,.0f params/sec)" total-sec speed-params-sec))
        (println (format "  Artifacts Persisted:        %s/catq_factors.edn" out-dir))
        (println (format "                              %s/model.safetensors" out-dir))
        (println "====================================================================================================\n")
        (println "To run inference on the resident calibrated model:")
        (println (format "  ./tools/gemma4.sh --backend %s --model %s --prompt \"Explain monads in Clojure\"\n"
                         (name backend) out-dir)))

      {:total-params total-params
       :elapsed-seconds total-sec
       :output-dir out-dir
       :output-size-gb out-size-gb})))

;; ==============================================================================
;; 4. CLI Entrypoint
;; ==============================================================================

(defn parse-cli-args [args]
  (loop [remaining args
         opts {:model ".models/gemma-4-E2B-it"
               :output-dir ".models/gemma-4-E2B-it-catq"
               :backend :rocm
               :max-seq-len 128
               :group-size 128}]
    (if (empty? remaining)
      opts
      (let [flag (first remaining)]
        (cond
          (or (= flag "--model") (= flag "-m"))
          (recur (subvec remaining 2) (assoc opts :model (second remaining)))

          (or (= flag "--output-dir") (= flag "-o"))
          (recur (subvec remaining 2) (assoc opts :output-dir (second remaining)))

          (or (= flag "--backend") (= flag "-b"))
          (recur (subvec remaining 2) (assoc opts :backend (keyword (second remaining))))

          (or (= flag "--max-seq-len") (= flag "-s"))
          (recur (subvec remaining 2) (assoc opts :max-seq-len (Long/parseLong (second remaining))))

          (or (= flag "--group-size") (= flag "-g"))
          (recur (subvec remaining 2) (assoc opts :group-size (Long/parseLong (second remaining))))

          (= flag "--skip-layers")
          (let [layers (into #{} (map #(Long/parseLong (str/trim %)) (str/split (second remaining) #",")))]
            (recur (subvec remaining 2) (assoc opts :skip-layers layers)))

          (= flag "--quiet")
          (recur (subvec remaining 1) (assoc opts :quiet true))

          :else
          (recur (subvec remaining 1) opts))))))

(defn -main [& args]
  (let [opts (parse-cli-args (vec args))]
    (calibrate-model-catq! opts)))
