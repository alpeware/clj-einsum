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
    (:bool :i1) "int"
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
         grid-blocks (if (= target :cpu)
                       1
                       (long (quot (+ (long out-size) local-threads -1) local-threads)))

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
       (do
         (when (:has-group? kernel-entry)
           (let [group-sz (long (or (:group-size kernel-entry) 64))]
             (.append body-sb (format "  __shared__ float lds[%d];\n" group-sz))))
         (.append body-sb (format "  int gidx = (int)(blockIdx.x * %d + threadIdx.x);\n" local-threads))
         (.append body-sb (format "  if (gidx >= %d) return;\n" (long out-size))))
       (do
         (when (>= (long out-size) 256)
           (.append body-sb "  #pragma omp parallel for schedule(static)\n"))
         (.append body-sb (format "  for (int gidx = 0; gidx < %d; ++gidx) {\n" (long out-size)))))

     (let [indent (if (= target :rocm) "  " "    ")
           strides (rangeify/row-major-strides out-shape)]
       (doseq [[idx [r dim stride]] (map-indexed vector (map vector out-ranges out-shape strides))]
         (when (= :uop/range (:op r))
           (let [r-var (str "r_" idx)
                 d (long dim)
                 s (long stride)]
             (.put var-names (:id r) r-var)
             (if (== 1 s)
               (.append body-sb (format "%sint %s = (gidx %% %d);\n" indent r-var d))
               (.append body-sb (format "%sint %s = ((gidx / %d) %% %d);\n" indent r-var s d))))))

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
                         v-name)

                       :uop/reduce-axis
                       (let [body-uop (first src)
                             r-uops (rest src)
                             red-op (:op arg)
                             acc-name (str "acc_" (swap! ssa-counter inc))
                             init-lit (case red-op
                                        :max "(-1.0e30f)"
                                        :min "(1.0e30f)"
                                        (format-const-literal dtype 0))
                             inner-vars (java.util.HashMap. scoped-vars)]
                         (.append body-sb (format "%s%s %s = %s;\n" cur-indent c-type acc-name init-lit))
                         ;; Open nested reduction loops
                         (doseq [[depth r] (map-indexed vector r-uops)]
                           (let [loop-indent (str cur-indent (apply str (repeat depth "  ")))
                                 r-var (str "rk_" (swap! rk-counter inc))
                                 bound-expr (emit-expr (first (:src r)) loop-indent inner-vars)]
                             (.put inner-vars (:id r) r-var)
                             (.append body-sb (format "%s#pragma unroll 4\n" loop-indent))
                             (.append body-sb (format "%sfor (int %s = 0; %s < %s; ++%s) {\n"
                                                      loop-indent r-var r-var bound-expr r-var))))
                         ;; Emit reduction body inside innermost loop using lexical `inner-vars`
                         (let [inner-indent (str cur-indent (apply str (repeat (count r-uops) "  ")))
                               body-val (emit-expr body-uop inner-indent inner-vars)]
                           (case red-op
                             :max (.append body-sb (format "%s%s = (%s > %s) ? %s : %s;\n"
                                                           inner-indent acc-name body-val acc-name body-val acc-name))
                             :min (.append body-sb (format "%s%s = (%s < %s) ? %s : %s;\n"
                                                           inner-indent acc-name body-val acc-name body-val acc-name))
                             (.append body-sb (format "%s%s += %s;\n" inner-indent acc-name body-val))))
                         ;; Close reduction loops
                         (doseq [depth (reverse (range (count r-uops)))]
                           (let [loop-indent (str cur-indent (apply str (repeat depth "  ")))]
                             (.append body-sb (format "%s}\n" loop-indent))))
                         ;; If cooperative group reduction is enabled, perform tree reduction in LDS!
                         (when (and (= target :rocm) (:has-group? kernel-entry))
                           (let [group-sz (long (or (:group-size kernel-entry) 64))]
                             (.append body-sb (format "%slds[threadIdx.x] = %s;\n" cur-indent acc-name))
                             (.append body-sb (format "%s__syncthreads();\n" cur-indent))
                             (.append body-sb (format "%sfor (int s = %d; s > 0; s >>= 1) {\n" cur-indent (quot group-sz 2)))
                             (.append body-sb (format "%s  if (threadIdx.x < s) { lds[threadIdx.x] += lds[threadIdx.x + s]; }\n" cur-indent))
                             (.append body-sb (format "%s  __syncthreads();\n" cur-indent))
                             (.append body-sb (format "%s}\n" cur-indent))
                             (.append body-sb (format "%s%s = lds[0];\n" cur-indent acc-name))))
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
         (let [out-idx-var (emit-expr out-lin-idx-uop indent var-names)
               final-val-var (emit-expr val-expr indent var-names)
               store-indent (if (and (= target :rocm) (:has-group? kernel-entry))
                              (str indent "  ")
                              indent)]
           (when (and (= target :rocm) (:has-group? kernel-entry))
             (.append body-sb (format "%sif (threadIdx.x == 0) {\n" indent)))
           (if (and (= out-dtype :bf16) (= (:dtype val-expr) :f32))
             (.append body-sb (format "%sbuf_0[%s] = f32_to_bf16(%s);\n" store-indent out-idx-var final-val-var))
             (.append body-sb (format "%sbuf_0[%s] = %s;\n" store-indent out-idx-var final-val-var)))
           (when (and (= target :rocm) (:has-group? kernel-entry))
             (.append body-sb (format "%s}\n" indent))))))

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
                             "#ifndef INFINITY\n#define INFINITY (1.0f/0.0f)\n#endif\n"
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
        :global-size [grid-blocks 1 1]
        :local-size [local-threads 1 1]}))))
