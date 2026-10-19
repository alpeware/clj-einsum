(ns einsum.uop.render
  "Linearizer & CStyle Source Renderer for rangeified UOp kernels.
   Modeled on tinygrad's `CStyleLanguage` / `HIPRenderer` (`tinygrad/renderer/cstyle.py`)
   and geohot's ML compiler syllabus (Weeks 2, 6, 7).
   Renders optimized, rangeified UOp kernels into standalone C99 (for `:uop-cpu`)
   or HIP C++ (for `:uop-rocm` / `gfx1100`) source strings."
  (:require [clojure.string :as str]
            [einsum.uop.core :as uop]
            [einsum.uop.rangeify :as rangeify]))

;; -----------------------------------------------------------------------------
;; 1. C / HIP Type & Literal Formatting
;; -----------------------------------------------------------------------------

(defn- dtype->c-type [dtype]
  (case dtype
    (:bool :i1) "bool"
    :i8 "signed char"
    :u8 "unsigned char"
    :i16 "short"
    :u16 "unsigned short"
    :i32 "int"
    :u32 "unsigned int"
    :i64 "long long"
    :u64 "unsigned long long"
    :f16 "_Float16"
    :bf16 "unsigned short"
    :f32 "float"
    :f64 "double"
    "float"))

(defn- format-const-literal [dtype val]
  (cond
    (contains? #{:bool :i1} dtype)
    (if val "1" "0")

    (contains? #{:f16 :bf16 :f32} dtype)
    (let [d (double val)]
      (cond
        (Double/isNaN d) "(0.0f/0.0f)"
        (Double/isInfinite d) (if (pos? d) "(1.0f/0.0f)" "(-1.0f/0.0f)")
        :else (let [s (str (float d))]
                (if (or (str/includes? s ".") (str/includes? s "E") (str/includes? s "e"))
                  (str s "f")
                  (str s ".0f")))))

    (= dtype :f64)
    (str (double val))

    :else
    (str (long val))))

;; -----------------------------------------------------------------------------
;; 2. Buffer Slot & Constant Table Discovery
;; -----------------------------------------------------------------------------

(defn- collect-referenced-slots
  "Collects all referenced global/param `:uop/buffer` slots in deterministic topological DFS order,
   ensuring `out-slot` is always at index 0."
  [sink-uop out-slot out-dtype]
  (let [nodes (uop/toposort sink-uop)
        buf-nodes (filter #(= :uop/buffer (:op %)) nodes)]
    (reduce
     (fn [acc b]
       (let [slot (:slot (:arg b))]
         (if (or (= :vconst (:device (:arg b)))
                 (some #(= slot (:slot %)) acc))
           acc
           (conj acc {:slot slot :dtype (:dtype b)}))))
     [{:slot out-slot :dtype out-dtype}]
     buf-nodes)))

(defn- sha256-hex
  "Computes lower-case hex SHA-256 of string `s`."
  [^String s]
  (let [md (java.security.MessageDigest/getInstance "SHA-256")
        bytes (.digest md (.getBytes s "UTF-8"))
        sb (StringBuilder. 64)]
    (doseq [b bytes]
      (.append sb (format "%02x" (bit-and (int b) 0xff))))
    (.toString sb)))

;; -----------------------------------------------------------------------------
;; 3. Expression & Statement Code Generator
;; -----------------------------------------------------------------------------

(defn- sanitize-ident [s]
  (str/replace (str s) #"[^a-zA-Z0-9_]" "_"))

(defn- find-invariant-load-roots
  "Finds maximal sub-expressions within root that contain at least one :uop/load
   and are completely invariant to token ranges (i.e. do not contain any token ranges)."
  [root node-has-token-range?]
  (letfn [(helper [node]
            (if (not (node-has-token-range? node))
              (if (some #(= (:op %) :uop/load) (tree-seq #(seq (:src %)) :src node))
                [node]
                [])
              (mapcat helper (:src node))))]
    (if (not (node-has-token-range? root))
      []
      (helper root))))

(defn- find-rk-inv-tok-inv-roots
  "Finds maximal sub-expressions within root that do not depend on red-range-ids
   or token ranges, and contain at least one :uop/load or ALU/cast operation."
  [root node-has-red-range? node-has-token-range?]
  (letfn [(has-computable? [node]
            (some #(or (= (:op %) :uop/load)
                       (contains? uop/ALU_OPS (:op %))
                       (= (:op %) :uop/cast))
                  (tree-seq #(seq (:src %)) :src node)))
          (helper [node]
            (if (and (not (node-has-red-range? node))
                     (not (node-has-token-range? node)))
              (if (has-computable? node)
                [node]
                [])
              (mapcat helper (:src node))))]
    (if (and (not (node-has-red-range? root))
             (not (node-has-token-range? root)))
      (if (has-computable? root) [root] [])
      (helper root))))

(defn- find-rk-inv-tok-dep-roots
  "Finds maximal sub-expressions within root that do not depend on red-range-ids
   but DO depend on token ranges, and contain at least one :uop/load or ALU/cast operation."
  [root node-has-red-range? node-has-token-range?]
  (letfn [(has-computable? [node]
            (some #(or (= (:op %) :uop/load)
                       (contains? uop/ALU_OPS (:op %))
                       (= (:op %) :uop/cast))
                  (tree-seq #(seq (:src %)) :src node)))
          (helper [node]
            (if (not (node-has-red-range? node))
              (if (node-has-token-range? node)
                (if (has-computable? node)
                  [node]
                  [])
                [])
              (mapcat helper (:src node))))]
    (if (not (node-has-red-range? root))
      (if (and (node-has-token-range? root) (has-computable? root)) [root] [])
      (helper root))))

(defn- find-rk-invariant-roots
  "Finds maximal sub-expressions within root that do not depend on red-range-ids
   and contain at least one :uop/load or ALU/cast operation."
  [root node-has-red-range?]
  (letfn [(has-computable? [node]
            (some #(or (= (:op %) :uop/load)
                       (contains? uop/ALU_OPS (:op %))
                       (= (:op %) :uop/cast))
                  (tree-seq #(seq (:src %)) :src node)))
          (helper [node]
            (if (not (node-has-red-range? node))
              (if (has-computable? node)
                [node]
                [])
              (mapcat helper (:src node))))]
    (if (not (node-has-red-range? root))
      (if (has-computable? root) [root] [])
      (helper root))))

(defn- bind-token-ranges
  "Copies `scoped-map` and binds leading `out-ranges` to expressions using `tok-var`."
  ([^java.util.Map scoped-map out-ranges out-shape s-dim tok-var]
   (bind-token-ranges scoped-map out-ranges out-shape s-dim 1 tok-var))
  ([^java.util.Map scoped-map out-ranges out-shape s-dim batch-heads tok-var]
   (let [m (java.util.HashMap. scoped-map)
         n-ranges (count out-ranges)
         leading-shape (when (>= n-ranges 2) (subvec (vec out-shape) 0 (dec n-ranges)))
         non-one-leading-idxs (when (>= n-ranges 2)
                                (vec (keep-indexed (fn [idx d] (when (> (long d) 1) idx)) leading-shape)))]
     (cond
       (and (> (long (or batch-heads 1)) 1) (>= (count non-one-leading-idxs) 2))
       (let [bh (long batch-heads)
             q-range-idx (peek non-one-leading-idxs)
             head-shape (subvec (vec out-shape) 0 q-range-idx)
             head-ranges (subvec (vec out-ranges) 0 q-range-idx)
             head-strides (rangeify/row-major-strides head-shape)
             q-range (nth out-ranges q-range-idx)]
         (doseq [[r dim stride] (map vector head-ranges head-shape head-strides)]
           (let [d (long dim)
                 s (long stride)]
             (cond
               (== 1 d) (.put m (:id r) "0")
               (== 1 s) (.put m (:id r) (if (== d bh) "r_head" (format "(r_head %% %d)" d)))
               :else (.put m (:id r) (if (== (* d s) bh)
                                       (format "(r_head / %d)" s)
                                       (format "((r_head / %d) %% %d)" s d))))))
         (.put m (:id q-range) tok-var)
         (doseq [r (subvec (vec out-ranges) (inc q-range-idx) (dec n-ranges))]
           (.put m (:id r) "0"))
         m)

       (== n-ranges 2)
       (do
         (.put m (:id (nth out-ranges 0)) tok-var)
         m)

       (>= n-ranges 3)
       (let [leading-shape (subvec (vec out-shape) 0 (dec n-ranges))
             leading-ranges (subvec (vec out-ranges) 0 (dec n-ranges))
             strides (rangeify/row-major-strides leading-shape)
             total-rows (long (or s-dim (reduce * 1 leading-shape)))]
         (doseq [[r dim stride] (map vector leading-ranges leading-shape strides)]
           (let [d (long dim)
                 s (long stride)]
             (cond
               (== 1 d)
               (.put m (:id r) "0")

               (== 1 s)
               (.put m (:id r) (if (== d total-rows)
                                 tok-var
                                 (format "(%s %% %d)" tok-var d)))

               :else
               (.put m (:id r) (if (== (* d s) total-rows)
                                 (format "(%s / %d)" tok-var s)
                                 (format "((%s / %d) %% %d)" tok-var s d))))))
         m)

       :else m))))

(defn render-kernel
  "Renders an optimized kernel map (from `einsum.uop.opt/optimize-kernel` or `schedule-graph`)
   into a standalone C99 (`:target :cpu`) or HIP C++ (`:target :rocm`) program map:
   `{:fn-name string
     :canonical-fn-name string
     :kernel-hash string
     :target (:cpu | :rocm)
     :source string
     :canonical-source string
     :buffer-slots [slot0 slot1 ...]
     :buffer-dtypes [dt0 dt1 ...]
     :global-size [gx gy gz]
     :local-size [lx ly lz]}`."
  ([kernel-entry]
   (render-kernel kernel-entry {:target :rocm}))
  ([kernel-entry {:keys [target] :or {target :rocm}}]
   (let [rk (:rangeified kernel-entry)
         {:keys [sink val-expr out-ranges out-shape out-dtype out-size out-slot vconsts]} rk
         raw-fn-name (sanitize-ident (or (:name kernel-entry) (str "uop_kernel_" out-slot)))
         slot-infos (collect-referenced-slots sink out-slot out-dtype)
         buffer-slots (mapv :slot slot-infos)
         buffer-dtypes (mapv :dtype slot-infos)
         slot->param-name (into {} (map-indexed (fn [idx {:keys [slot]}] [slot (str "buf_" idx)]) slot-infos))
         out-lin-idx-uop (rangeify/coords->linear-index out-ranges out-shape)
         local-threads (long (first (or (:local-size kernel-entry)
                                        (if (= target :cpu) [1 1 1] [256 1 1]))))
         local-rows (long (or (:local-rows kernel-entry)
                              (second (or (:local-size kernel-entry) [1 1 1]))
                              1))
         group-sz (long (or (:group-size kernel-entry) 64))
         unroll-factor (long (or (:unroll-factor kernel-entry) 4))
         is-2d-gemm? (and (= target :rocm)
                          (or (true? (:gemm-2d? kernel-entry))
                              (and (= (:local-size kernel-entry) [16 16 1])
                                   (not (:has-group? kernel-entry)))))
         is-2d-prefill? (and (= target :rocm)
                             (not is-2d-gemm?)
                             (:has-group? kernel-entry)
                             (>= (count out-shape) 2)
                             (>= (long (last out-shape)) 16)
                             (let [n (long (last out-shape))
                                   s (long (quot (long out-size) n))]
                               (>= s 2)))
         s-dim (cond
                 is-2d-gemm? (long (or (:s-dim kernel-entry)
                                       (let [n (long (last out-shape))]
                                         (quot (long out-size) n))))
                 is-2d-prefill? (long (let [n (long (last out-shape))]
                                        (quot (long out-size) n)))
                 :else nil)
         n-dim (cond
                 (or is-2d-gemm? is-2d-prefill?) (long (last out-shape))
                 :else nil)
         leading-shape (when (>= (count out-shape) 2) (subvec (vec out-shape) 0 (dec (count out-shape))))
         non-one-leading-idxs (when (>= (count out-shape) 2)
                                (vec (keep-indexed (fn [idx d] (when (> (long d) 1) idx)) leading-shape)))
         q-range-idx (if (and is-2d-gemm? (>= (count non-one-leading-idxs) 2))
                       (peek non-one-leading-idxs)
                       (or (peek non-one-leading-idxs) 0))
         q-dim (if (and is-2d-gemm? (>= (count non-one-leading-idxs) 2))
                 (long (nth leading-shape q-range-idx))
                 (or s-dim 1))
         batch-heads (if (and is-2d-gemm? (>= (count non-one-leading-idxs) 2))
                       (long (quot (or s-dim 1) (max 1 q-dim)))
                       1)
         all-red-nodes (filterv #(= (:op %) :uop/reduce-axis)
                                (tree-seq #(seq (:src %)) :src val-expr))
         gemm-red-node (when (== 1 (count all-red-nodes))
                         (first all-red-nodes))
         tok-range-ids (when (and is-2d-gemm? (seq out-ranges))
                         #{(:id (nth out-ranges q-range-idx))})
         node-has-token-range? (fn [uop]
                                 (some (fn [node]
                                         (contains? tok-range-ids (:id node)))
                                       (tree-seq #(seq (:src %)) :src uop)))
         has-invariant-weight-load? (when (and is-2d-gemm? gemm-red-node)
                                      (let [body-uop (first (:src gemm-red-node))
                                            loads (filter #(= (:op %) :uop/load)
                                                          (tree-seq #(seq (:src %)) :src body-uop))]
                                        (boolean (some #(not (node-has-token-range? %)) loads))))
         raw-gemm-tile-m (if is-2d-gemm?
                           (long (or (:tile-m kernel-entry)
                                     (let [target-dim (if (> batch-heads 1) q-dim s-dim)]
                                       (if (and target-dim (>= target-dim 16))
                                         (min 8 (long (quot (+ target-dim 15) 16)))
                                         1))))
                           1)
         use-gemm-tile? (boolean (and (= target :rocm)
                                      is-2d-gemm?
                                      (> raw-gemm-tile-m 1)
                                      (some? gemm-red-node)
                                      has-invariant-weight-load?))
         gemm-tile-m (if use-gemm-tile? raw-gemm-tile-m 1)
         block-m (* 16 gemm-tile-m)
         chan-range-id (when (seq out-ranges) (:id (last out-ranges)))
         node-has-chan-range? (fn [uop] (boolean (and chan-range-id (some #(= chan-range-id (:id %)) (tree-seq #(seq (:src %)) :src uop)))))
         lds-gemm-factors (when (and use-gemm-tile?
                                     (= :add (:op (:arg gemm-red-node)))
                                     (== 1 (count (rest (:src gemm-red-node))))
                                     (let [b (first (:src (first (rest (:src gemm-red-node)))))]
                                       (and (= :uop/const (:op b)) (>= (long (:arg b)) 16)))
                                     (= :uop/mul (:op (first (:src gemm-red-node)))))
                            (let [[f0 f1] (:src (first (:src gemm-red-node)))]
                              (cond
                                (and (not (node-has-chan-range? f0))
                                     (not (node-has-token-range? f1)))
                                {:x-factor f0 :w-factor f1}

                                (and (not (node-has-chan-range? f1))
                                     (not (node-has-token-range? f0)))
                                {:x-factor f1 :w-factor f0}

                                :else nil)))
         use-lds-gemm? (some? lds-gemm-factors)
         grid-n (cond
                  is-2d-gemm? (long (quot (+ n-dim 16 -1) 16))
                  is-2d-prefill? (long (quot (+ n-dim local-rows -1) local-rows))
                  :else nil)
         grid-q (when is-2d-gemm?
                  (if (> batch-heads 1)
                    (max 1 (long (quot (+ q-dim block-m -1) block-m)))
                    1))
         grid-m (when is-2d-gemm?
                  (if (> batch-heads 1)
                    (* batch-heads grid-q)
                    (long (quot (+ s-dim block-m -1) block-m))))
         grid-blocks (if (= target :cpu)
                       1
                       (cond
                         is-2d-gemm? grid-n
                         is-2d-prefill? grid-n
                         (:has-group? kernel-entry) (long (quot (+ (long out-size) local-rows -1) local-rows))
                         :else (long (quot (+ (long out-size) local-threads -1) local-threads))))
         global-sz (cond
                     is-2d-gemm? [grid-n grid-m 1]
                     is-2d-prefill? [grid-n s-dim 1]
                     :else (or (:global-size kernel-entry) [grid-blocks 1 1]))
         local-sz (cond
                    is-2d-gemm? [16 16 1]
                    (:has-group? kernel-entry) [group-sz local-rows 1]
                    :else [local-threads 1 1])

         body-sb (StringBuilder.)
         var-names (java.util.HashMap.)
         vconst-names (into {} (map-indexed (fn [idx {:keys [slot]}] [slot (str "vc___FN___" idx)]) vconsts))
         ssa-counter (atom 0)
         rk-counter (atom 0)]

     ;; 1. Emit static vconst tables (e.g. RoPE cos/sin, causal mask) using __FN__ placeholder
     (doseq [[idx {:keys [values dtype size]}] (map-indexed vector vconsts)]
       (let [c-type (dtype->c-type dtype)
             qualifier (if (= target :rocm) "__constant__ " "static const ")
             lits (str/join ", " (map #(format-const-literal dtype %) values))]
         (.append body-sb (format "%s%s vc___FN___%d[%d] = {%s};\n" qualifier c-type (long idx) (long size) lits))))

     ;; 2. Emit Function Signature with __FN__ placeholder and positional buf_0, buf_1, ...
     (let [param-decls (str/join ", "
                                 (map-indexed
                                  (fn [idx {:keys [dtype]}]
                                    (if (zero? idx)
                                      (format "%s* __restrict__ buf_%d" (dtype->c-type dtype) (long idx))
                                      (format "const %s* __restrict__ buf_%d" (dtype->c-type dtype) (long idx))))
                                  slot-infos))]
       (if (= target :rocm)
         (.append body-sb (format "\nextern \"C\" __global__ void __FN__(%s) {\n" param-decls))
         (.append body-sb (format "\nstatic void __FN___impl(%s) {\n" param-decls))))

     ;; 3. Bind Output Loop Coordinates
     (if (= target :rocm)
       (cond
         is-2d-gemm?
         (if use-lds-gemm?
           (if (> batch-heads 1)
             (do
               (.append body-sb "  __shared__ float s_W[16][17];\n")
               (.append body-sb (format "  __shared__ float s_X[%d][16][17];\n" gemm-tile-m))
               (.append body-sb (format "  int r_head = (int)(blockIdx.y / %d);\n" grid-q))
               (.append body-sb (format "  int block_token_base = (int)((blockIdx.y %% %d) * %d);\n" grid-q block-m))
               (.append body-sb "  int block_channel_base = (int)(blockIdx.x * 16);\n")
               (.append body-sb (format "  if (r_head >= %d || block_token_base >= %d || block_channel_base >= %d) return;\n"
                                        batch-heads q-dim n-dim))
               (.append body-sb "  int raw_token_base = block_token_base + (int)threadIdx.y;\n")
               (.append body-sb "  int raw_channel = block_channel_base + (int)threadIdx.x;\n")
               (.append body-sb "  int raw_channel_w = block_channel_base + (int)threadIdx.y;\n")
               (.append body-sb (format "  int r_token_base = (raw_token_base < %d) ? raw_token_base : 0;\n" q-dim))
               (.append body-sb (format "  int r_channel = (raw_channel < %d) ? raw_channel : 0;\n" n-dim))
               (.append body-sb (format "  int r_channel_w = (raw_channel_w < %d) ? raw_channel_w : 0;\n" n-dim))
               (dotimes [tm gemm-tile-m]
                 (if (zero? tm)
                   (.append body-sb "  int r_tok_0 = r_token_base;\n")
                   (.append body-sb (format "  int r_tok_%d = (raw_token_base + %d < %d) ? (raw_token_base + %d) : r_token_base;\n"
                                            tm (* tm 16) q-dim (* tm 16))))))
             (do
               (.append body-sb "  __shared__ float s_W[16][17];\n")
               (.append body-sb (format "  __shared__ float s_X[%d][16][17];\n" gemm-tile-m))
               (.append body-sb (format "  int block_token_base = (int)(blockIdx.y * %d);\n" block-m))
               (.append body-sb "  int block_channel_base = (int)(blockIdx.x * 16);\n")
               (.append body-sb (format "  if (block_token_base >= %d || block_channel_base >= %d) return;\n"
                                        s-dim n-dim))
               (.append body-sb "  int raw_token_base = block_token_base + (int)threadIdx.y;\n")
               (.append body-sb "  int raw_channel = block_channel_base + (int)threadIdx.x;\n")
               (.append body-sb "  int raw_channel_w = block_channel_base + (int)threadIdx.y;\n")
               (.append body-sb (format "  int r_token_base = (raw_token_base < %d) ? raw_token_base : 0;\n" s-dim))
               (.append body-sb (format "  int r_channel = (raw_channel < %d) ? raw_channel : 0;\n" n-dim))
               (.append body-sb (format "  int r_channel_w = (raw_channel_w < %d) ? raw_channel_w : 0;\n" n-dim))
               (dotimes [tm gemm-tile-m]
                 (if (zero? tm)
                   (.append body-sb "  int r_tok_0 = r_token_base;\n")
                   (.append body-sb (format "  int r_tok_%d = (raw_token_base + %d < %d) ? (raw_token_base + %d) : r_token_base;\n"
                                            tm (* tm 16) s-dim (* tm 16)))))))
           (if (> batch-heads 1)
             (if use-gemm-tile?
               (do
                 (.append body-sb (format "  int r_head = (int)(blockIdx.y / %d);\n" grid-q))
                 (.append body-sb (format "  int r_token_base = (int)((blockIdx.y %% %d) * %d + threadIdx.y);\n" grid-q block-m))
                 (.append body-sb "  int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);\n")
                 (.append body-sb (format "  if (r_head >= %d || r_token_base >= %d || r_channel >= %d) return;\n" batch-heads q-dim n-dim))
                 (dotimes [tm gemm-tile-m]
                   (if (zero? tm)
                     (.append body-sb "  int r_tok_0 = r_token_base;\n")
                     (.append body-sb (format "  int r_tok_%d = (r_token_base + %d < %d) ? (r_token_base + %d) : r_token_base;\n"
                                              tm (* tm 16) q-dim (* tm 16))))))
               (do
                 (.append body-sb (format "  int r_head = (int)(blockIdx.y / %d);\n" grid-q))
                 (.append body-sb (format "  int r_token = (int)((blockIdx.y %% %d) * 16 + threadIdx.y);\n" grid-q))
                 (.append body-sb "  int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);\n")
                 (.append body-sb (format "  if (r_head >= %d || r_token >= %d || r_channel >= %d) return;\n" batch-heads q-dim n-dim))))
             (if use-gemm-tile?
               (do
                 (.append body-sb (format "  int r_token_base = (int)(blockIdx.y * %d + threadIdx.y);\n" block-m))
                 (.append body-sb "  int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);\n")
                 (.append body-sb (format "  if (r_token_base >= %d || r_channel >= %d) return;\n" s-dim n-dim))
                 (dotimes [tm gemm-tile-m]
                   (if (zero? tm)
                     (.append body-sb "  int r_tok_0 = r_token_base;\n")
                     (.append body-sb (format "  int r_tok_%d = (r_token_base + %d < %d) ? (r_token_base + %d) : r_token_base;\n"
                                              tm (* tm 16) s-dim (* tm 16))))))
               (do
                 (.append body-sb "  int r_token = (int)(blockIdx.y * 16 + threadIdx.y);\n")
                 (.append body-sb "  int r_channel = (int)(blockIdx.x * 16 + threadIdx.x);\n")
                 (.append body-sb (format "  if (r_token >= %d || r_channel >= %d) return;\n" s-dim n-dim))
                 (.append body-sb (format "  int gidx = r_token * %d + r_channel;\n" n-dim))))))

         is-2d-prefill?
         (let [lds-size (* local-rows group-sz)]
           (.append body-sb (format "  __shared__ float lds[%d];\n" (long lds-size)))
           (.append body-sb "  int r_token = (int)blockIdx.y;\n")
           (.append body-sb (format "  if (r_token >= %d) return;\n" s-dim))
           (.append body-sb (format "  int r_channel = (int)(blockIdx.x * %d + threadIdx.y);\n" local-rows))
           (.append body-sb (format "  bool valid_gidx = (r_channel < %d);\n" n-dim))
           (.append body-sb (format "  int lds_offset = (int)threadIdx.y * %d;\n" group-sz))
           (.append body-sb (format "  int gidx = r_token * %d + r_channel;\n" n-dim)))

         (:has-group? kernel-entry)
         (let [lds-size (* local-rows group-sz)]
           (.append body-sb (format "  __shared__ float lds[%d];\n" (long lds-size)))
           (if (> local-rows 1)
             (do
               (.append body-sb (format "  if ((int)(blockIdx.x * %d) >= %d) return;\n" local-rows (long out-size)))
               (.append body-sb (format "  int gidx = (int)(blockIdx.x * %d + threadIdx.y);\n" local-rows))
               (.append body-sb (format "  bool valid_gidx = (gidx < %d);\n" (long out-size)))
               (.append body-sb (format "  int lds_offset = (int)threadIdx.y * %d;\n" group-sz)))
             (do
               (.append body-sb "  int gidx = (int)blockIdx.x;\n")
               (.append body-sb (format "  if (gidx >= %d) return;\n" (long out-size)))
               (.append body-sb "  bool valid_gidx = true;\n")
               (.append body-sb "  int lds_offset = 0;\n"))))

         :else
         (do
           (.append body-sb (format "  int gidx = (int)(blockIdx.x * %d + threadIdx.x);\n" local-threads))
           (.append body-sb (format "  if (gidx >= %d) return;\n" (long out-size)))))
       (do
         (when (>= (long out-size) 256)
           (.append body-sb "  #pragma omp parallel for schedule(static)\n"))
         (.append body-sb (format "  for (int gidx = 0; gidx < %d; ++gidx) {\n" (long out-size)))))

     (let [indent (if (= target :rocm) "  " "    ")]
       (cond
         (or is-2d-gemm? is-2d-prefill?)
         (let [n-ranges (count out-ranges)]
           (cond
             (== n-ranges 2)
             (do
               (.put var-names (:id (nth out-ranges 0)) "r_token")
               (.put var-names (:id (nth out-ranges 1)) "r_channel"))

             (>= n-ranges 3)
             (if (and is-2d-gemm? (> batch-heads 1) (>= (count non-one-leading-idxs) 2))
               (let [bh (long batch-heads)
                     q-range-idx (peek non-one-leading-idxs)
                     head-shape (subvec (vec out-shape) 0 q-range-idx)
                     head-ranges (subvec (vec out-ranges) 0 q-range-idx)
                     head-strides (rangeify/row-major-strides head-shape)
                     q-range (nth out-ranges q-range-idx)]
                 (doseq [[r dim stride] (map vector head-ranges head-shape head-strides)]
                   (let [d (long dim)
                         s (long stride)]
                     (cond
                       (== 1 d)
                       (.put var-names (:id r) "0")

                       (== 1 s)
                       (.put var-names (:id r) (if (== d bh)
                                                 "r_head"
                                                 (format "(r_head %% %d)" d)))

                       :else
                       (.put var-names (:id r) (if (== (* d s) bh)
                                                 (format "(r_head / %d)" s)
                                                 (format "((r_head / %d) %% %d)" s d))))))
                 (.put var-names (:id q-range) "r_token")
                 (doseq [r (subvec (vec out-ranges) (inc q-range-idx) (dec n-ranges))]
                   (.put var-names (:id r) "0"))
                 (.put var-names (:id (last out-ranges)) "r_channel"))
               (let [leading-shape (subvec (vec out-shape) 0 (dec n-ranges))
                     leading-ranges (subvec (vec out-ranges) 0 (dec n-ranges))
                     strides (rangeify/row-major-strides leading-shape)
                     total-rows (long (or s-dim (reduce * 1 leading-shape)))]
                 (doseq [[r dim stride] (map vector leading-ranges leading-shape strides)]
                   (let [d (long dim)
                         s (long stride)]
                     (cond
                       (== 1 d)
                       (.put var-names (:id r) "0")

                       (== 1 s)
                       (.put var-names (:id r) (if (== d total-rows)
                                                 "r_token"
                                                 (format "(r_token %% %d)" d)))

                       :else
                       (.put var-names (:id r) (if (== (* d s) total-rows)
                                                 (format "(r_token / %d)" s)
                                                 (format "((r_token / %d) %% %d)" s d))))))
                 (.put var-names (:id (last out-ranges)) "r_channel")))))

         :else
         (let [strides (rangeify/row-major-strides out-shape)]
           (doseq [[idx [r dim stride]] (map-indexed vector (map vector out-ranges out-shape strides))]
             (when (= :uop/range (:op r))
               (let [r-var (str "r_" idx)
                     d (long dim)
                     s (long stride)]
                 (.put var-names (:id r) r-var)
                 (if (== 1 s)
                   (.append body-sb (format "%sint %s = (gidx %% %d);\n" indent r-var d))
                   (.append body-sb (format "%sint %s = ((gidx / %d) %% %d);\n" indent r-var s d))))))))

       ;; 4. Recursive SSA Emitter for `out-lin-idx-uop` and `val-expr` with lexical loop scoping
       (letfn [(emit-expr [node cur-indent ^java.util.HashMap scoped-vars]
                 (if-let [existing (.get scoped-vars (:id node))]
                   existing
                   (let [{:keys [op dtype src arg]} node
                         c-type (dtype->c-type dtype)]
                     (case op
                       :uop/const
                       (format-const-literal dtype arg)

                       :uop/load
                       (let [idx-uop (first src)
                             buf-uop (first (:src idx-uop))
                             lin-uop (second (:src idx-uop))
                             slot (:slot (:arg buf-uop))
                             buf-name (or (get vconst-names slot)
                                          (get slot->param-name slot)
                                          (str "buf_" slot))
                             idx-expr (emit-expr lin-uop cur-indent scoped-vars)
                             v-name (str "v_" (swap! ssa-counter inc))]
                         (.append body-sb (format "%s%s %s = %s[%s];\n" cur-indent c-type v-name buf-name idx-expr))
                         (.put scoped-vars (:id node) v-name)
                         v-name)                        :uop/reduce-axis
                       (let [body-uop (first src)
                             r-uops (rest src)
                             red-op (:op arg)
                             acc-name (str "acc_" (swap! ssa-counter inc))
                             init-lit (case red-op
                                        :max "(-1.0e30f)"
                                        :min "(1.0e30f)"
                                        (format-const-literal dtype 0))
                             inner-vars (java.util.HashMap. scoped-vars)
                             has-guard? (and (= target :rocm) (:has-group? kernel-entry) (> local-rows 1))
                             pre-loop-indent (if has-guard? (str cur-indent "  ") cur-indent)
                             red-range-ids (set (map :id r-uops))
                             node-has-red-range? (fn [uop]
                                                   (boolean (some #(contains? red-range-ids (:id %))
                                                                  (tree-seq #(seq (:src %)) :src uop))))
                             is-gemm-red? (boolean (and use-gemm-tile? (= node gemm-red-node)))
                             is-lds-gemm? (boolean (and use-lds-gemm? (= node gemm-red-node)))]
                         (if is-gemm-red?
                           (dotimes [tm gemm-tile-m]
                             (.append body-sb (format "%s%s %s_%d = %s;\n" cur-indent c-type acc-name tm init-lit)))
                           (.append body-sb (format "%s%s %s = %s;\n" cur-indent c-type acc-name init-lit)))
                         (when has-guard?
                           (.append body-sb (format "%sif (valid_gidx) {\n" cur-indent)))

                         (if is-lds-gemm?
                            ;; --- Cooperative LDS GEMM ---
                           (let [{:keys [x-factor w-factor]} lds-gemm-factors
                                 node-has-token-or-chan? #(or (node-has-token-range? %) (node-has-chan-range? %))]
                              ;; 1. Hoist body-uop roots invariant to rk, token, and channel into inner-vars
                             (doseq [inv-node (find-rk-inv-tok-inv-roots body-uop node-has-red-range? node-has-token-or-chan?)]
                               (emit-expr inv-node cur-indent inner-vars))

                              ;; 2. Hoist rk-invariant roots of w-factor into w-base-vars
                             (let [w-base-vars (doto (java.util.HashMap. inner-vars)
                                                 (.put chan-range-id "r_channel_w"))]
                               (doseq [inv-node (find-rk-invariant-roots w-factor node-has-red-range?)]
                                 (emit-expr inv-node cur-indent w-base-vars))

                                ;; 3. Construct tm-base-vars and hoist rk-invariant, token-dependent roots of x-factor
                               (let [tm-base-vars (mapv (fn [tm]
                                                          (bind-token-ranges inner-vars out-ranges out-shape s-dim batch-heads (format "r_tok_%d" tm)))
                                                        (range gemm-tile-m))
                                     tok-dep-nodes (find-rk-inv-tok-dep-roots x-factor node-has-red-range? node-has-token-range?)]
                                 (dotimes [tm gemm-tile-m]
                                   (let [vars (nth tm-base-vars tm)]
                                     (doseq [dep-node tok-dep-nodes]
                                       (emit-expr dep-node cur-indent vars))))

                                  ;; 4. Reduction tile loop (stepping by 16)
                                 (let [r-uop (first r-uops)
                                       bound-node (first (:src r-uop))
                                       bound-expr (emit-expr bound-node cur-indent inner-vars)
                                       k-const (when (= :uop/const (:op bound-node)) (long (:arg bound-node)))
                                       k-aligned? (and (number? k-const) (zero? (mod k-const 16)))
                                       tile-indent (str cur-indent "  ")
                                       r-var (str "rk_" (swap! rk-counter inc))]
                                   (.append body-sb (format "%sfor (int rk_tile = 0; rk_tile < %s; rk_tile += 16) {\n"
                                                            cur-indent bound-expr))
                                   (.append body-sb (format "%sint %s = rk_tile + (int)threadIdx.x;\n" tile-indent r-var))

                                    ;; Cooperative load into s_W and s_X
                                   (let [load-indent (if k-aligned? tile-indent (str tile-indent "  "))
                                         w-scoped-vars (doto (java.util.HashMap. w-base-vars)
                                                         (.put (:id r-uop) r-var))
                                         x-shared-vars (doto (java.util.HashMap. inner-vars)
                                                         (.put (:id r-uop) r-var))]
                                     (when-not k-aligned?
                                       (.append body-sb (format "%sif (%s < %s) {\n" tile-indent r-var bound-expr)))

                                      ;; Load w-factor
                                     (let [w-val (emit-expr w-factor load-indent w-scoped-vars)]
                                       (.append body-sb (format "%ss_W[threadIdx.y][threadIdx.x] = %s;\n" load-indent w-val)))

                                      ;; Hoist token-invariant load roots of x-factor into x-shared-vars
                                     (doseq [inv-node (find-invariant-load-roots x-factor node-has-token-range?)]
                                       (emit-expr inv-node load-indent x-shared-vars))

                                      ;; Load x-factor for each tm
                                     (dotimes [tm gemm-tile-m]
                                       (let [tm-vars (doto (java.util.HashMap. x-shared-vars)
                                                       (.putAll ^java.util.Map (nth tm-base-vars tm)))
                                             x-val (emit-expr x-factor load-indent tm-vars)]
                                         (.append body-sb (format "%ss_X[%d][threadIdx.y][threadIdx.x] = %s;\n"
                                                                  load-indent tm x-val))))

                                     (when-not k-aligned?
                                       (let [else-indent (str tile-indent "  ")]
                                         (.append body-sb (format "%s} else {\n" tile-indent))
                                         (.append body-sb (format "%ss_W[threadIdx.y][threadIdx.x] = 0.0f;\n" else-indent))
                                         (dotimes [tm gemm-tile-m]
                                           (.append body-sb (format "%ss_X[%d][threadIdx.y][threadIdx.x] = 0.0f;\n" else-indent tm)))
                                         (.append body-sb (format "%s}\n" tile-indent)))))

                                    ;; Synchronize before FMA
                                   (.append body-sb (format "%s__syncthreads();\n" tile-indent))

                                    ;; Unrolled 16-step FMA accumulation
                                   (.append body-sb (format "%s#pragma unroll 16\n" tile-indent))
                                   (.append body-sb (format "%sfor (int k_sub = 0; k_sub < 16; ++k_sub) {\n" tile-indent))
                                   (let [fma-indent (str tile-indent "  ")]
                                     (.append body-sb (format "%sfloat w_reg = s_W[threadIdx.x][k_sub];\n" fma-indent))
                                     (dotimes [tm gemm-tile-m]
                                       (.append body-sb (format "%s%s_%d += s_X[%d][threadIdx.y][k_sub] * w_reg;\n"
                                                                fma-indent acc-name tm tm)))
                                     (.append body-sb (format "%s}\n" tile-indent)))

                                    ;; Synchronize before next tile load to avoid race on shared memory
                                   (.append body-sb (format "%s__syncthreads();\n" tile-indent))
                                   (.append body-sb (format "%s}\n" cur-indent))))))

                            ;; --- Standard reduction loops (non-LDS GEMM tile and general reductions) ---
                           (let [tm-base-vars
                                 (when is-gemm-red?
                                  ;; 1. Hoist rk-invariant + token-invariant nodes to inner-vars
                                   (doseq [inv-node (find-rk-inv-tok-inv-roots body-uop node-has-red-range? node-has-token-range?)]
                                     (emit-expr inv-node pre-loop-indent inner-vars))
                                  ;; 2. Construct tm-base-vars with leading out-ranges bound to r_tok_tm
                                   (let [base-vars (mapv (fn [tm]
                                                           (bind-token-ranges inner-vars out-ranges out-shape s-dim batch-heads (str "r_tok_" tm)))
                                                         (range gemm-tile-m))
                                         tok-dep-nodes (find-rk-inv-tok-dep-roots body-uop node-has-red-range? node-has-token-range?)]
                                    ;; 3. Hoist rk-invariant + token-dependent nodes per tm
                                     (dotimes [tm gemm-tile-m]
                                       (let [vars (nth base-vars tm)]
                                         (doseq [dep-node tok-dep-nodes]
                                           (emit-expr dep-node pre-loop-indent vars))))
                                     base-vars))]
                             (when-not is-gemm-red?
                              ;; For non-GEMM tile, hoist all rk-invariant nodes to inner-vars
                               (doseq [inv-node (find-rk-invariant-roots body-uop node-has-red-range?)]
                                 (emit-expr inv-node pre-loop-indent inner-vars)))

                            ;; Open nested reduction loops
                             (let [vectorized-depths (atom {})]
                               (doseq [[depth r] (map-indexed vector r-uops)]
                                 (let [loop-indent (str (if has-guard? (str cur-indent "  ") cur-indent)
                                                        (apply str (repeat depth "  ")))
                                       r-var (str "rk_" (swap! rk-counter inc))
                                       bound-node (first (:src r))
                                       bound-expr (emit-expr bound-node loop-indent inner-vars)
                                       k-const (when (= :uop/const (:op bound-node)) (:arg bound-node))
                                       group-sz (long (or (:group-size kernel-entry) 64))
                                       vec-sz (when (and (number? k-const)
                                                         (= target :rocm)
                                                         (:has-group? kernel-entry)
                                                         (= depth (dec (count r-uops))))
                                                (let [k (long k-const)]
                                                  (cond
                                                    (and (>= k (* group-sz 8)) (zero? (mod k (* group-sz 8)))) 8
                                                    (and (>= k (* group-sz 4)) (zero? (mod k (* group-sz 4)))) 4
                                                    (and (>= k (* group-sz 2)) (zero? (mod k (* group-sz 2)))) 2
                                                    :else nil)))]
                                   (.put inner-vars (:id r) r-var)
                                   (when (and (pos? unroll-factor) (not vec-sz))
                                     (.append body-sb (format "%s#pragma unroll %d\n" loop-indent unroll-factor)))
                                   (cond
                                     vec-sz
                                     (let [step (* group-sz (long vec-sz))
                                           chunk-var (str r-var "_chunk")
                                           base-var (str r-var "_base")
                                           lane-var (str r-var "_lane")]
                                       (swap! vectorized-depths assoc depth vec-sz)
                                       (.append body-sb (format "%sfor (int %s = 0; %s < %s; %s += %d) {\n"
                                                                loop-indent chunk-var chunk-var bound-expr chunk-var step))
                                       (.append body-sb (format "%s  int %s = %s + (int)threadIdx.x * %d;\n"
                                                                loop-indent base-var chunk-var (long vec-sz)))
                                       (.append body-sb (format "%s  #pragma unroll\n" loop-indent))
                                       (.append body-sb (format "%s  for (int %s = 0; %s < %d; ++%s) {\n"
                                                                loop-indent lane-var lane-var (long vec-sz) lane-var))
                                       (.append body-sb (format "%s    int %s = %s + %s;\n"
                                                                loop-indent r-var base-var lane-var)))

                                     (and (= target :rocm) (:has-group? kernel-entry) (= depth (dec (count r-uops))))
                                     (.append body-sb (format "%sfor (int %s = (int)threadIdx.x; %s < %s; %s += %d) {\n"
                                                              loop-indent r-var r-var bound-expr r-var group-sz))

                                     :else
                                     (.append body-sb (format "%sfor (int %s = 0; %s < %s; ++%s) {\n"
                                                              loop-indent r-var r-var bound-expr r-var)))))
                              ;; Emit reduction body inside innermost loop using lexical `inner-vars`
                               (let [extra-indent (if (get @vectorized-depths (dec (count r-uops))) "    " "")
                                     inner-indent (str (if has-guard? (str cur-indent "  ") cur-indent)
                                                       (apply str (repeat (count r-uops) "  "))
                                                       extra-indent)]
                                 (if is-gemm-red?
                                   (do
                                    ;; Hoist invariant weight loads and their casts into inner-vars
                                     (doseq [inv-node (find-invariant-load-roots body-uop node-has-token-range?)]
                                       (emit-expr inv-node inner-indent inner-vars))
                                    ;; Unrolled branchless scoped token accumulation blocks
                                     (dotimes [tm gemm-tile-m]
                                       (let [tm-indent (str inner-indent "  ")
                                             tm-vars (doto (java.util.HashMap. inner-vars)
                                                       (.putAll ^java.util.Map (nth tm-base-vars tm)))]
                                         (.append body-sb (format "%s{\n" inner-indent))
                                         (.append body-sb (format "%sint r_token = r_tok_%d;\n" tm-indent tm))
                                         (let [body-val (emit-expr body-uop tm-indent tm-vars)]
                                           (case red-op
                                             :max (.append body-sb (format "%s%s_%d = (%s > %s_%d) ? %s : %s_%d;\n"
                                                                           tm-indent acc-name tm body-val acc-name tm body-val acc-name tm))
                                             :min (.append body-sb (format "%s%s_%d = (%s < %s_%d) ? %s : %s_%d;\n"
                                                                           tm-indent acc-name tm body-val acc-name tm body-val acc-name tm))
                                             (.append body-sb (format "%s%s_%d += %s;\n" tm-indent acc-name tm body-val))))
                                         (.append body-sb (format "%s}\n" inner-indent)))))
                                   (let [body-val (emit-expr body-uop inner-indent inner-vars)]
                                     (case red-op
                                       :max (.append body-sb (format "%s%s = (%s > %s) ? %s : %s;\n"
                                                                     inner-indent acc-name body-val acc-name body-val acc-name))
                                       :min (.append body-sb (format "%s%s = (%s < %s) ? %s : %s;\n"
                                                                     inner-indent acc-name body-val acc-name body-val acc-name))
                                       (.append body-sb (format "%s%s += %s;\n" inner-indent acc-name body-val))))))
                              ;; Close reduction loops
                               (doseq [depth (reverse (range (count r-uops)))]
                                 (let [loop-indent (str (if has-guard? (str cur-indent "  ") cur-indent)
                                                        (apply str (repeat depth "  ")))]
                                   (if (get @vectorized-depths depth)
                                     (do
                                       (.append body-sb (format "%s  }\n" loop-indent))
                                       (.append body-sb (format "%s}\n" loop-indent)))
                                     (.append body-sb (format "%s}\n" loop-indent))))))))
                         (when has-guard?
                           (.append body-sb (format "%s}\n" cur-indent)))
                         ;; If cooperative group reduction is enabled, perform tree reduction in LDS!
                         (when (and (= target :rocm) (:has-group? kernel-entry))
                           (let [group-sz (long (or (:group-size kernel-entry) 64))]
                             (.append body-sb (format "%slds[lds_offset + threadIdx.x] = %s;\n" cur-indent acc-name))
                             (.append body-sb (format "%s__syncthreads();\n" cur-indent))
                             (.append body-sb (format "%sfor (int s = %d; s > 0; s >>= 1) {\n" cur-indent (quot group-sz 2)))
                             (case red-op
                               :max (.append body-sb (format "%s  if (threadIdx.x < s) { float b = lds[lds_offset + threadIdx.x + s]; if (b > lds[lds_offset + threadIdx.x]) lds[lds_offset + threadIdx.x] = b; }\n" cur-indent))
                               :min (.append body-sb (format "%s  if (threadIdx.x < s) { float b = lds[lds_offset + threadIdx.x + s]; if (b < lds[lds_offset + threadIdx.x]) lds[lds_offset + threadIdx.x] = b; }\n" cur-indent))
                               (.append body-sb (format "%s  if (threadIdx.x < s) { lds[lds_offset + threadIdx.x] += lds[lds_offset + threadIdx.x + s]; }\n" cur-indent)))
                             (.append body-sb (format "%s  __syncthreads();\n" cur-indent))
                             (.append body-sb (format "%s}\n" cur-indent))
                             (.append body-sb (format "%s%s = lds[lds_offset];\n" cur-indent acc-name))))
                         (.put scoped-vars (:id node) acc-name)
                         acc-name)

                         ;; Elementwise Unary / Binary / Ternary ALU Ops
                       (let [child-exprs (mapv #(emit-expr % cur-indent scoped-vars) src)
                             [e0 e1 e2] child-exprs
                             rhs-code (case op
                                        :uop/neg (format "(-%s)" e0)
                                        :uop/recip (format "(1.0f / %s)" e0)
                                        :uop/exp2 (format "exp2f(%s)" e0)
                                        :uop/log2 (format "log2f(%s)" e0)
                                        :uop/sin (format "sinf(%s)" e0)
                                        :uop/cos (format "cosf(%s)" e0)
                                        :uop/sqrt (format "sqrtf(%s)" e0)
                                        :uop/trunc (format "truncf(%s)" e0)
                                        :uop/cast (let [in-dt (:dtype (first src))
                                                        out-dt arg]
                                                    (cond
                                                      (and (= in-dt :bf16) (= out-dt :f32))
                                                      (format "bf16_to_f32(%s)" e0)
                                                      (and (= in-dt :f32) (= out-dt :bf16))
                                                      (format "f32_to_bf16(%s)" e0)
                                                      :else
                                                      (format "((%s)(%s))" (dtype->c-type out-dt) e0)))
                                        :uop/add (format "(%s + %s)" e0 e1)
                                        :uop/sub (format "(%s - %s)" e0 e1)
                                        :uop/mul (format "(%s * %s)" e0 e1)
                                        :uop/div (format "(%s / %s)" e0 e1)
                                        :uop/idiv (format "(%s / %s)" e0 e1)
                                        :uop/mod (format "(%s %% %s)" e0 e1)
                                        :uop/max (format "((%s > %s) ? %s : %s)" e0 e1 e0 e1)
                                        :uop/min (format "((%s < %s) ? %s : %s)" e0 e1 e0 e1)
                                        :uop/shl (format "(%s << %s)" e0 e1)
                                        :uop/shr (format "((unsigned int)%s >> %s)" e0 e1)
                                        :uop/and (if (contains? #{:bool :i1} (:dtype (first src)))
                                                   (format "(%s && %s)" e0 e1)
                                                   (format "(%s & %s)" e0 e1))
                                        :uop/or (if (contains? #{:bool :i1} (:dtype (first src)))
                                                  (format "(%s || %s)" e0 e1)
                                                  (format "(%s | %s)" e0 e1))
                                        :uop/xor (format "(%s ^ %s)" e0 e1)
                                        :uop/cmplt (format "(%s < %s)" e0 e1)
                                        :uop/cmple (format "(%s <= %s)" e0 e1)
                                        :uop/cmpeq (format "(%s == %s)" e0 e1)
                                        :uop/cmpne (format "(%s != %s)" e0 e1)
                                        :uop/cmpgt (format "(%s > %s)" e0 e1)
                                        :uop/cmpge (format "(%s >= %s)" e0 e1)
                                        :uop/where (format "(%s ? %s : %s)" e0 e1 e2)
                                        (throw (ex-info "Unsupported UOp in CStyle renderer" {:op op :node node})))
                             v-name (str "v_" (swap! ssa-counter inc))]
                         (.append body-sb (format "%s%s %s = %s;\n" cur-indent c-type v-name rhs-code))
                         (.put scoped-vars (:id node) v-name)
                         v-name)))))]
         (if use-gemm-tile?
           (let [acc-name (emit-expr gemm-red-node indent var-names)
                 row-bound (if (> batch-heads 1) q-dim s-dim)]
             (dotimes [tm gemm-tile-m]
               (let [row-offset (* tm 16)
                     guard-str (if use-lds-gemm?
                                 (if (zero? row-offset)
                                   (format "raw_channel < %d && raw_token_base < %d" n-dim row-bound)
                                   (format "raw_channel < %d && raw_token_base + %d < %d" n-dim row-offset row-bound))
                                 (if (zero? row-offset)
                                   (format "r_token_base < %d" row-bound)
                                   (format "r_token_base + %d < %d" row-offset row-bound)))
                     tok-val (if (zero? row-offset)
                               (if use-lds-gemm? "raw_token_base" "r_token_base")
                               (if use-lds-gemm?
                                 (format "(raw_token_base + %d)" row-offset)
                                 (format "(r_token_base + %d)" row-offset)))
                     acc-var (format "%s_%d" acc-name tm)
                     store-indent (str indent "  ")
                     tm-store-vars (doto (java.util.HashMap. var-names)
                                     (.put (:id gemm-red-node) acc-var))]
                 (.append body-sb (format "%sif (%s) {\n" indent guard-str))
                 (.append body-sb (format "%sint r_token = %s;\n" store-indent tok-val))
                 (let [out-idx-var (emit-expr out-lin-idx-uop store-indent tm-store-vars)
                       final-val-var (if (= val-expr gemm-red-node)
                                       acc-var
                                       (emit-expr val-expr store-indent tm-store-vars))
                       val-to-write (if (and (= out-dtype :bf16) (= (:dtype val-expr) :f32))
                                      (format "f32_to_bf16(%s)" final-val-var)
                                      final-val-var)]
                   (.append body-sb (format "%sbuf_0[%s] = %s;\n"
                                            store-indent out-idx-var val-to-write))
                   (.append body-sb (format "%s}\n" indent))))))
           (let [out-idx-var (emit-expr out-lin-idx-uop indent var-names)
                 final-val-var (emit-expr val-expr indent var-names)
                 store-indent (if (and (= target :rocm) (:has-group? kernel-entry))
                                (str indent "  ")
                                indent)]
             (when (and (= target :rocm) (:has-group? kernel-entry))
               (.append body-sb (format "%sif (threadIdx.x == 0 && valid_gidx) {\n" indent)))
             (if (and (= out-dtype :bf16) (= (:dtype val-expr) :f32))
               (.append body-sb (format "%sbuf_0[%s] = f32_to_bf16(%s);\n" store-indent out-idx-var final-val-var))
               (.append body-sb (format "%sbuf_0[%s] = %s;\n" store-indent out-idx-var final-val-var)))
             (when (and (= target :rocm) (:has-group? kernel-entry))
               (.append body-sb (format "%s}\n" indent)))))))

     ;; 5. Close loop / function body and emit uniform void** entry wrapper for CPU
     (if (= target :rocm)
       (.append body-sb "}\n")
       (let [unpacked-args (str/join ", "
                                     (map-indexed
                                      (fn [idx {:keys [dtype]}]
                                        (if (zero? idx)
                                          (format "(%s*)buffers[%d]" (dtype->c-type dtype) idx)
                                          (format "(const %s*)buffers[%d]" (dtype->c-type dtype) idx)))
                                      slot-infos))]
         (.append body-sb "  }\n}\n\n")
         (.append body-sb (format "void __FN__(void** buffers) {\n  __FN___impl(%s);\n}\n"
                                  unpacked-args))))

     (let [template-body (.toString body-sb)
           k-hash (subs (sha256-hex (str (name target) ":" template-body)) 0 16)
           canonical-fn-name (str "uop_fn_" k-hash)
           header-str (if (= target :rocm)
                        (str "// Generated by clj-einsum UOp HIPRenderer (gfx1100)\n"
                             "#ifndef INFINITY\n#define INFINITY (1.0f/0.0f)\n#endif\n\n"
                             "typedef short __attribute__((ext_vector_type(16))) short16;\n"
                             "typedef float __attribute__((ext_vector_type(8))) float8;\n\n"
                             "static inline __device__ float8 rdna3_wmma_bf16_16x16x16(short16 a, short16 b, float8 c) {\n"
                             "  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);\n"
                             "}\n\n"
                             "static inline __device__ __host__ float bf16_to_f32(unsigned short x) {\n"
                             "  union { unsigned int u; float f; } v;\n"
                             "  v.u = ((unsigned int)x) << 16;\n"
                             "  return v.f;\n"
                             "}\n"
                             "static inline __device__ __host__ unsigned short f32_to_bf16(float x) {\n"
                             "  union { float f; unsigned int u; } v;\n"
                             "  v.f = x;\n"
                             "  return (unsigned short)((v.u + 0x7fffU + ((v.u >> 16) & 1U)) >> 16);\n"
                             "}\n\n")
                        (str "// Generated by clj-einsum UOp C99Renderer\n"
                             "#include <math.h>\n#include <stdint.h>\n\n"
                             "static inline float bf16_to_f32(unsigned short x) {\n"
                             "  union { unsigned int u; float f; } v;\n"
                             "  v.u = ((unsigned int)x) << 16;\n"
                             "  return v.f;\n"
                             "}\n"
                             "static inline unsigned short f32_to_bf16(float x) {\n"
                             "  union { float f; unsigned int u; } v;\n"
                             "  v.f = x;\n"
                             "  return (unsigned short)((v.u + 0x7fffU + ((v.u >> 16) & 1U)) >> 16);\n"
                             "}\n\n"))
           named-body (str/replace template-body "__FN__" raw-fn-name)
           canonical-body (str/replace template-body "__FN__" canonical-fn-name)]
       {:fn-name raw-fn-name
        :canonical-fn-name canonical-fn-name
        :kernel-hash k-hash
        :target target
        :source (str header-str named-body)
        :canonical-body canonical-body
        :buffer-slots buffer-slots
        :buffer-dtypes buffer-dtypes
        :global-size global-sz
        :local-size local-sz}))))
