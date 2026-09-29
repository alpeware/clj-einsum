(ns experiments.gate3-evals.prompt-tuning-v1.core
  "Core library for Prompt Tuning v1: Worked Agentic Trajectories (Gate 3 Evals).
   Provides prompt variant definitions (P0, P1, P2), token budget auditing,
   strict catalog disjointness checks, mechanical canary echoing detection,
   and pre-registered McNemar paired decision rules."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [einsum.runtime.tokenizer.protocol :as tok-proto]
            [sci.core :as sci])
  (:import [java.security MessageDigest]))

;; =============================================================================
;; 1. Candidate System Prompts (P0, P1, P2)
;; =============================================================================

(def PROMPT-RULES-BASE
  "You are an expert Clojure engineer with access to an interactive Clojure REPL sandbox.
Keep internal reasoning very concise (1-2 sentences): plan logic briefly, do not write full code in thought, and wrap your Clojure code in ```clojure ... ``` markdown fences to test against public examples.
Execution results return in ```clojure_result ... ```.
If tests fail, inspect the error, revise your code, and output revised code in ```clojure ... ``` to re-test.
Once all public tests pass, provide your final answer in plain text.")

(def WORKED-EXAMPLE-1
  "

Worked Example 1:
Goal: (sum-even-squares [nums]) returning sum of squares of even numbers.
Turn 1 thought: Filter even numbers, square them, and sum.
```clojure
(defn sum-even-squares [nums]
  (reduce + 0 (map #(* % %) (filter odd? nums))))
```
```clojure_result
FAIL: (sum-even-squares [1 2 3 4]) -> Expected 20, got 10
```
Turn 2 thought: Predicate used odd? instead of even?. Correcting.
```clojure
(defn sum-even-squares [nums]
  (reduce + 0 (map #(* % %) (filter even? nums))))
```
```clojure_result
PASS: (sum-even-squares [1 2 3 4]) -> 20
```
Turn 3: All tests pass. (sum-even-squares [nums]) filters evens first.")

(def WORKED-EXAMPLE-2
  "

Worked Example 2:
Goal: (word-lengths [s]) returning word lengths in s, or [] if blank.
Turn 1 thought: Split string s by whitespace and count words.
```clojure
(defn word-lengths [s]
  (mapv count (clojure.string/split s #\" \")))
```
```clojure_result
FAIL: (word-lengths \"\") -> Expected [], got [0]
```
Turn 2 thought: Splitting \"\" gives [\"\"]. Guard against blank input.
```clojure
(defn word-lengths [s]
  (if (clojure.string/blank? s) [] (mapv count (re-seq #\"\\S+\" s))))
```
```clojure_result
PASS: (word-lengths \"hello world\") -> [5 5]
PASS: (word-lengths \"\") -> []
```
Turn 3: All tests pass. (word-lengths [s]) guards against blank input.")

(def PROMPT-V0-ZERO-SHOT
  PROMPT-RULES-BASE)

(def PROMPT-V1-SINGLE
  (str PROMPT-RULES-BASE WORKED-EXAMPLE-1))

(def PROMPT-V2-DUAL
  (str PROMPT-RULES-BASE WORKED-EXAMPLE-1 WORKED-EXAMPLE-2))

(def PROMPT-VARIANTS
  {:p0 {:id :p0 :name "fenced-zero-shot" :system PROMPT-V0-ZERO-SHOT}
   :p1 {:id :p1 :name "fenced-worked-single" :system PROMPT-V1-SINGLE}
   :p2 {:id :p2 :name "fenced-worked-dual" :system PROMPT-V2-DUAL}})

(defn get-system-prompt
  "Retrieves the system prompt string for a prompt variant key (:p0, :p1, :p2)."
  [variant-key]
  (get-in PROMPT-VARIANTS [variant-key :system]))

(defn compute-prompt-sha
  "Computes a deterministic lowercase hexadecimal SHA-256 hash for a prompt variant key
   (:p0, :p1, :p2) or arbitrary prompt string."
  [variant-or-text]
  (let [text (if (keyword? variant-or-text)
               (or (get-system-prompt variant-or-text) (str variant-or-text))
               (str variant-or-text))
        digest (MessageDigest/getInstance "SHA-256")
        bytes (.digest digest (.getBytes (str text) "UTF-8"))]
    (apply str (map (fn [^Byte b] (format "%02x" (bit-and (int b) 0xff))) bytes))))

;; =============================================================================
;; 2. Token Budget & Disjointness Auditing (Gate 1)
;; =============================================================================

(defn measure-prompt-tokens
  "Tokenizes text using the provided Tokenizer instance and returns exact token count."
  [tokenizer text]
  (count (tok-proto/encode tokenizer text)))

(def SYNTHETIC-WORKED-SYMBOLS
  #{"sum-even-squares" "word-lengths"})

(defn check-disjointness
  "Verifies that synthetic worked symbols do not overlap with any task ID,
   function name, or title in the provided catalog task collection.
   Returns {:disjoint? boolean :conflicts set}."
  [tasks]
  (let [catalog-symbols (set (mapcat (fn [t]
                                       [(str (:fn-name t))
                                        (str (:id t))
                                        (str (:title t))])
                                     tasks))
        conflicts (set/intersection SYNTHETIC-WORKED-SYMBOLS catalog-symbols)]
    {:disjoint? (empty? conflicts)
     :conflicts conflicts}))

;; =============================================================================
;; 3. Mechanical Canary Echoing Detector
;; =============================================================================

(def ECHO-CANARY-SYMBOLS
  #{"sum-even-squares" "word-lengths"})

(def ECHO-CANARY-LITERALS
  #{"Expected 20, got 10"})

(def ^:private sci-reader-ctx (sci/init {}))

(defn extract-ast-symbol-names
  "Parses Clojure code text into s-expressions and extracts all symbol names."
  [code-str]
  (if (str/blank? code-str)
    #{}
    (try
      (let [reader (sci/source-reader code-str)
            symbols (atom #{})]
        (loop []
          (let [form (try (sci/parse-next sci-reader-ctx reader) (catch Throwable _ :eof))]
            (when (and (some? form) (not= form :eof) (not= form :sci.core/eof))
              (walk/postwalk (fn [x]
                               (when (symbol? x)
                                 (swap! symbols conj (name x)))
                               x)
                             form)
              (recur))))
        @symbols)
      (catch Throwable _
        ;; Fallback to regex word extraction if unparseable
        (set (re-seq #"[a-zA-Z0-9_\-\*\+\?]+" code-str))))))

(defn detect-canary-echo
  "Inspects candidate submission code for worked example canary leaks.
   Exempts tasks whose target function name matches the canary symbol itself.
   Returns {:leak? boolean :echo-detected? boolean :reasons [string] :matches [string]}."
  [candidate-code target-fn-name]
  (let [target-name-str (str target-fn-name)
        ast-symbols (extract-ast-symbol-names candidate-code)
        leaked-symbols (set/intersection (set (remove #(= % target-name-str) ECHO-CANARY-SYMBOLS))
                                         ast-symbols)
        leaked-literals (filterv #(str/includes? (or candidate-code "") %) ECHO-CANARY-LITERALS)
        reasons (cond-> []
                  (seq leaked-symbols) (conj (str "Leaked canary symbols: " (str/join ", " leaked-symbols)))
                  (seq leaked-literals) (conj (str "Leaked canary literals: " (str/join ", " leaked-literals))))
        has-leak? (boolean (seq reasons))]
    {:leak? has-leak?
     :echo-detected? has-leak?
     :reasons reasons
     :matches reasons}))

;; =============================================================================
;; 4. Paired McNemar Test & Decision Logic
;; =============================================================================

(defn factorial
  "Exact factorial for small integer n."
  [n]
  (reduce *' 1 (range 1 (inc n))))

(defn binomial-cdf
  "Computes cumulative probability P(X <= k) for X ~ Binomial(n, 0.5)."
  [k n]
  (if (< k 0)
    0.0
    (let [total-prob (reduce + (map (fn [i]
                                      (/ (double (factorial n))
                                         (* (double (factorial i))
                                            (double (factorial (- n i))))))
                                    (range 0 (inc k))))]
      (/ total-prob (Math/pow 2.0 (double n))))))

(def ^:private NORMAL-CDF-COEFFS
  [1.330274429 -1.821255978 1.781477937 -0.356563782 0.319381530])

(defn standard-normal-tail-p
  "Computes one-tailed standard normal tail probability P(Z >= z) for Z ~ Normal(0, 1)
   via Abramowitz & Stegun formula 26.2.17 (absolute error < 7.5e-8)."
  [^double z]
  (if (< z 0.0)
    (- 1.0 (standard-normal-tail-p (- z)))
    (let [p 0.2316419
          t (/ 1.0 (+ 1.0 (* p z)))
          poly (* t (reduce (fn [acc c] (+ (double c) (* t acc))) 0.0 NORMAL-CDF-COEFFS))
          inv-sqrt-2pi 0.3989422804014327]
      (* inv-sqrt-2pi (Math/exp (* -0.5 z z)) poly))))

(defn mcnemar-test
  "Computes paired McNemar test on discordant pairs (b, c) across N tasks.
   b = candidate passes, control fails (favorable flip)
   c = candidate fails, control passes (unfavorable flip)
   Returns {:b int :c int :delta double :chi2 double :p-value double :significant? boolean}."
  [b c n]
  (let [total-discordant (+ b c)
        delta (if (pos? n) (double (/ (- b c) n)) 0.0)]
    (if (zero? total-discordant)
      {:b b :c c :delta delta :chi2 0.0 :p-value 1.0 :significant? false}
      (if (< total-discordant 25)
        ;; Exact one-tailed binomial test for small sample
        (let [p-val (binomial-cdf (min b c) total-discordant)
              significant? (and (> b c) (< p-val 0.05))]
          {:b b
           :c c
           :delta delta
           :chi2 0.0
           :p-value p-val
           :significant? significant?
           :exact? true})
        ;; Edwards continuity-corrected chi-squared test (1 df)
        (let [diff (Math/abs (double (- b c)))
              num (Math/pow (max 0.0 (- diff 1.0)) 2.0)
              chi2 (/ num (double total-discordant))
              z (if (> b c) (Math/sqrt chi2) (- (Math/sqrt chi2)))
              p-val (standard-normal-tail-p z)
              significant? (and (> b c) (< p-val 0.05))]
          {:b b
           :c c
           :delta delta
           :chi2 chi2
           :p-value p-val
           :significant? significant?
           :exact? false})))))

(defn evaluate-precedence-decision
  "Evaluates the complete decision rules per Section 6 precedence hierarchy:
   1. KILLED triggers dominate.
   2. Otherwise ADOPT iff all conjoined criteria hold.
   3. Otherwise REJECT (exhaustive fallback).
   Guarantees zero dead zones and deterministic classification."
  [{:keys [delta-e4b
           mcnemar-significant?
           delta-31b
           sealed10-regressions
           canary-leaks
           prompt-tokens
           wall-clock-inflation
           phase2-early-stop?]}]
  (let [adverse-flips (long (or sealed10-regressions 0))
        leaks (long (or canary-leaks 0))
        tokens (long (or prompt-tokens 0))
        inflation (double (or wall-clock-inflation 0.0))
        d-e4b (double (or delta-e4b 0.0))
        d-31b (double (or delta-31b 0.0))
        sig? (boolean mcnemar-significant?)
        early-stop? (boolean phase2-early-stop?)]
    (cond
      ;; 1. KILLED Triggers Dominate
      (or (< d-e4b 0.0)
          (pos? adverse-flips)
          (pos? leaks)
          (> tokens 600)
          (> inflation 0.25))
      {:verdict :killed
       :rationale (str/join "; "
                            (cond-> []
                              (< d-e4b 0.0) (conj (format "Negative transfer on dev (delta = %.2f%%)" (* 100.0 d-e4b)))
                              (pos? adverse-flips) (conj (format "%d adverse regressions on sealed-10" adverse-flips))
                              (pos? leaks) (conj (format "%d canary echoing leaks detected" leaks))
                              (> tokens 600) (conj (format "Prompt token count %d exceeds 600 cap" tokens))
                              (> inflation 0.25) (conj (format "Wall-clock inflation %.1f%% exceeds 25%%" (* 100.0 inflation)))))}

      ;; 2. ADOPT Requires All Conjoined Criteria
      (and (not early-stop?)
           (>= d-e4b 0.05)
           sig?
           (>= d-31b 0.0)
           (zero? adverse-flips)
           (zero? leaks)
           (<= tokens 600)
           (<= inflation 0.25))
      {:verdict :adopt
       :rationale (format "All ADOPT criteria satisfied: delta-E4B = +%.2f%% (McNemar p < 0.05), delta-31B = +%.2f%%, 0 sealed regressions, prompt = %d tokens"
                          (* 100.0 d-e4b) (* 100.0 d-31b) tokens)}

      ;; 3. REJECT Exhaustive Fallback
      :else
      {:verdict :reject
       :rationale (if early-stop?
                    "Phase 2 pilot early-stop triggered (neither candidate beat P0 by >= 2 tasks)"
                    (format "Failed to meet all conjoined ADOPT criteria without fatal condition: delta-E4B = %.2f%%, significant? = %s, delta-31B = %.2f%%"
                            (* 100.0 d-e4b) sig? (* 100.0 d-31b)))})))
