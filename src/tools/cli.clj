(ns tools.cli
  "Shared CLI argument parsing, model directory discovery, and input marshaling for clj-xla tools."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

;; ==============================================================================
;; 1. Standard Defaults & Model Candidates
;; ==============================================================================

(def STANDARD_MODEL_CANDIDATES
  {:gemma-2 [".models/gemma-2-2b-it" ".models/gemma-2b" ".models/gemma-2-2b" ".models/gemma"]
   :gemma-3 [".models/gemma-3-270m-it" ".models/gemma-3-270m" ".models/gemma-3" ".models/gemma"]
   :gemma-4 [".models/gemma-4-E2B-it" ".models/gemma-4-2b-it" ".models/gemma-4-E4B-it"
             ".models/gemma-4-4b-it" ".models/gemma-4-12B-it" ".models/gemma-4-12b-it"
             ".models/gemma-4-31B-it" ".models/gemma-4-31b-it" ".models/gemma4-2b"
             ".models/gemma-2b" ".models/gemma4"]
   :smollm  [".models/smollm-135m" ".models/smollm"]
   :gpt2    [".models/gpt2"]})

(def DEFAULT_INFERENCE_OPTS
  {:prompt "The capital of France is"
   :max-new-tokens 10
   :temperature 0.70
   :top-k 10
   :backend :cpu
   :precision :bf16
   :verbose false})

(def DEFAULT_AGENT_OPTS
  {:system "You are a helpful coding assistant with access to Clojure tools."
   :prompt "Inspect src/ and calculate total Clojure lines"
   :max-turns 8
   :max-consecutive-errors 3
   :sandbox :agent
   :max-new-tokens 256
   :temperature 0.70
   :top-k 40
   :repetition-penalty 1.10
   :backend :cpu
   :precision :bf16
   :thinking false
   :quiet false
   :verbose false})

;; ==============================================================================
;; 2. Model Directory Discovery
;; ==============================================================================

(defn model-dir?
  "Returns true if `dir-or-file` exists and is a valid model directory or weights file."
  [dir-or-file]
  (when dir-or-file
    (let [f (io/file dir-or-file)]
      (and (.exists f)
           (or (.isFile f)
               (.exists (io/file f "model.safetensors"))
               (.exists (io/file f "model-00001-of-00002.safetensors"))
               (.exists (io/file f "model-00001-of-00004.safetensors"))
               (.exists (io/file f "config.json"))
               (boolean (seq (filter (fn [^java.io.File sf]
                                       (and (.isFile sf) (str/ends-with? (.getName sf) ".safetensors")))
                                     (or (.listFiles f) [])))))))))

(defn find-model-dir
  "Resolves an existing model directory from `candidates-or-family` or `explicit-path`.
   `candidates-or-family` can be:
     - an explicit path string or File
     - a model family keyword (:gemma-2, :gemma-3, :gemma-4, :smollm, :gpt2)
     - a sequential collection of candidate directory paths.
   Throws ExceptionInfo listing searched candidate paths if none exist."
  ([candidates-or-family]
   (find-model-dir nil candidates-or-family))
  ([explicit-path candidates-or-family]
   (if (and explicit-path (not (str/blank? (str explicit-path))))
     (let [f (io/file explicit-path)]
       (if (.exists f)
         (if (.isFile f) (.getParent f) (.getPath f))
         (throw (ex-info (str "Specified model directory does not exist: " explicit-path)
                         {:model-dir explicit-path}))))
     (let [candidates (cond
                        (keyword? candidates-or-family)
                        (get STANDARD_MODEL_CANDIDATES candidates-or-family [])

                        (sequential? candidates-or-family)
                        candidates-or-family

                        (or (string? candidates-or-family) (instance? java.io.File candidates-or-family))
                        [candidates-or-family]

                        :else [])
           existing (first (filter model-dir? candidates))]
       (if existing
         (let [f (io/file existing)]
           (if (.isFile f) (.getParent f) (.getPath f)))
         (throw (ex-info (str "Model directory not found in candidates: " (vec candidates))
                         {:searched-dirs (vec candidates)
                          :family candidates-or-family})))))))

;; ==============================================================================
;; 3. CLI Argument Normalization & Parsing
;; ==============================================================================

(defn normalize-args
  "Splits '--key=val' arguments into separate '--key' and 'val' entries."
  [args]
  (mapcat (fn [arg]
            (if (and (string? arg) (str/starts-with? arg "--") (str/includes? arg "="))
              (str/split arg #"=" 2)
              [arg]))
          args))

(defn parse-cli-args
  "Parses command-line flags against `default-opts` (defaulting to DEFAULT_INFERENCE_OPTS).
   Supports optional `custom-handlers` map of {flag-str (fn [opts remaining-args]) -> [new-opts consumed-count]}.
   Supported backend targets include :cpu, :rocm, :sycl, :cuda12, and :interpreter (pure-JVM StableHLO)."
  ([args]
   (parse-cli-args args DEFAULT_INFERENCE_OPTS {}))
  ([args default-opts]
   (parse-cli-args args default-opts {}))
  ([args default-opts custom-handlers]
   (loop [remaining (vec (normalize-args args))
          opts (or default-opts {})]
     (if (empty? remaining)
       opts
       (let [arg (first remaining)
             val (second remaining)]
         (if-let [custom-fn (get custom-handlers arg)]
           (let [[new-opts consumed] (custom-fn opts remaining)]
             (recur (subvec remaining consumed) new-opts))
           (cond
             ;; String flags with values
             (= arg "--prompt")
             (recur (subvec remaining 2) (assoc opts :prompt val))

             (= arg "--prompt-file")
             (recur (subvec remaining 2) (assoc opts :prompt (slurp val)))

             (= arg "--system")
             (recur (subvec remaining 2) (assoc opts :system val))

             (= arg "--system-file")
             (recur (subvec remaining 2) (assoc opts :system (slurp val)))

             (or (= arg "--model-dir") (= arg "--model") (= arg "--model-name") (= arg "-m"))
             (let [dir (if (and (string? val) (not (str/starts-with? val ".")) (not (str/starts-with? val "/")))
                         (if (.exists (io/file val)) val (str ".models/" (last (str/split val #"/"))))
                         val)]
               (recur (subvec remaining 2) (assoc opts :model-dir dir :model dir)))

             (= arg "--out")
             (recur (subvec remaining 2) (assoc opts :out val))

             (= arg "--profile-out")
             (recur (subvec remaining 2) (assoc opts :profile-out val))

             (= arg "--chrome-trace-out")
             (recur (subvec remaining 2) (assoc opts :chrome-trace-out val))

             ;; Integer / Long flags
             (= arg "--max-new-tokens")
             (recur (subvec remaining 2) (assoc opts :max-new-tokens (Long/parseLong val)))

             (= arg "--max-seq-len")
             (recur (subvec remaining 2) (assoc opts :max-seq-len (Long/parseLong val)))

             (= arg "--top-k")
             (recur (subvec remaining 2) (assoc opts :top-k (Long/parseLong val)))

             (= arg "--max-turns")
             (recur (subvec remaining 2) (assoc opts :max-turns (Long/parseLong val)))

             (= arg "--max-consecutive-errors")
             (recur (subvec remaining 2) (assoc opts :max-consecutive-errors (Long/parseLong val)))

             (or (= arg "--group-size") (= arg "-g"))
             (recur (subvec remaining 2) (assoc opts :group-size (Long/parseLong val)))

             ;; Double / Float flags
             (or (= arg "--temperature") (= arg "--temp"))
             (recur (subvec remaining 2) (assoc opts :temperature (Double/parseDouble val)))

             (= arg "--repetition-penalty")
             (recur (subvec remaining 2) (assoc opts :repetition-penalty (Double/parseDouble val)))

             ;; Keyword flags (strip leading colons)
             (= arg "--backend")
             (let [b (keyword (str/replace val #"^:+" ""))]
               (recur (subvec remaining 2) (assoc opts :backend (if (= b :interp) :interpreter b))))

             (= arg "--precision")
             (recur (subvec remaining 2) (assoc opts :precision (keyword (str/replace val #"^:+" ""))))

             (= arg "--method")
             (recur (subvec remaining 2) (assoc opts :method (keyword (str/replace val #"^:+" ""))))

             (= arg "--mode")
             (recur (subvec remaining 2) (assoc opts :mode (keyword (str/replace val #"^:+" ""))))

             (= arg "--sandbox")
             (recur (subvec remaining 2) (assoc opts :sandbox (keyword (str/replace val #"^:+" ""))))

             ;; Boolean valued flags
             (= arg "--compare")
             (recur (subvec remaining 2) (assoc opts :compare (Boolean/parseBoolean val)))

             (= arg "--chat")
             (if (and val (not (str/starts-with? val "--")))
               (recur (subvec remaining 2) (assoc opts :chat (Boolean/parseBoolean val)))
               (recur (subvec remaining 1) (assoc opts :chat true)))

             ;; Boolean switch flags (no value)
             (= arg "--verbose")
             (recur (subvec remaining 1) (assoc opts :verbose true))

             (= arg "--quiet")
             (do (System/setProperty "clj-xla.quiet" "true")
                 (recur (subvec remaining 1) (assoc opts :quiet true)))

             (or (= arg "--thinking") (= arg "--think"))
             (if (and val (not (str/starts-with? val "--")))
               (recur (subvec remaining 2) (assoc opts :thinking (Boolean/parseBoolean val)))
               (recur (subvec remaining 1) (assoc opts :thinking true)))

             (or (= arg "--no-thinking") (= arg "--no-think"))
             (recur (subvec remaining 1) (assoc opts :thinking false))

             (= arg "--ternary")
             (recur (subvec remaining 1) (assoc opts :precision :ternary :is-ternary true))

             ;; Comma-separated set of integers
             (= arg "--skip-layers")
             (let [layers (into #{} (map #(Long/parseLong (str/trim %)) (str/split val #",")))]
               (recur (subvec remaining 2) (assoc opts :skip-layers layers)))

             :else
             (if (str/starts-with? arg "--")
               (if (and val (not (str/starts-with? val "--")))
                 (let [k (keyword (subs arg 2))]
                   (recur (subvec remaining 2) (assoc opts k val)))
                 (let [k (keyword (subs arg 2))]
                   (recur (subvec remaining 1) (assoc opts k true))))
               (recur (subvec remaining 1) (update opts :extra-args (fnil conj []) arg))))))))))

;; ==============================================================================
;; 4. Device Input Buffer Marshaling
;; ==============================================================================

(defn prepare-input-tensor
  "Pads token ID sequence `tokens` to `max-len` with zero padding, returning an int-array."
  [tokens max-len]
  (let [padded (take max-len (concat tokens (repeat 0)))]
    (int-array (vec padded))))
