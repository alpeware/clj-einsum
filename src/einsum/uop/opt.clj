(ns einsum.uop.opt
  "Declarative Kernel Optimization (`OptOps`) and heuristic scheduler for UOp kernels.
   Modeled on tinygrad's `OptOps` (`tinygrad/codegen/opt/`), `shift-to` range-splitting,
   and `hand_coded_optimizations` (`tinygrad/codegen/opt/heuristic.py`).
   Transforms rangeified loop axes into GPU grid (`:global`), threadblock (`:local`),
   vectorized register (`:upcast`), and cooperative reduction (`:group`, `:unroll`) configurations."
  (:require [einsum.uop.core :as uop]))

;; -----------------------------------------------------------------------------
;; 1. Declarative Opt Recipe Representation & RDNA3 TensorCore Spec
;; -----------------------------------------------------------------------------

(def RDNA3_WMMA_SPEC
  "Declarative specification for AMD RDNA3 (`gfx1100` / RX 7900 XTX) Wave32 WMMA 16x16x16 Tensor Core,
   ported directly from tinygrad's `TensorCore.amd_rdna3`."
  {:arch :gfx1100
   :dims [16 16 16]
   :threads 32
   :dtypes #{[:f16 :f32] [:bf16 :f32] [:f16 :f16]}
   :intrinsic "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32"
   :bf16-intrinsic "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32"
   :opts [:l0 :l0 :l0 :l0 :l1 :u1 :u1 :u1]})

(defn has-reduce-axis?
  "Returns true if the rangeified kernel contains a `:uop/reduce-axis` node."
  [rk]
  (boolean (some #(= :uop/reduce-axis (:op %)) (uop/toposort (:sink rk)))))

;; -----------------------------------------------------------------------------
;; 2. Real OptOps Range-Splitting & Axis Manipulation (`apply-opt` / `shift-to`)
;; -----------------------------------------------------------------------------

(defn normalize-opt
  "Converts an `Opt` specification (map or vector) into a canonical map `{:op ... :axis ... :amount ...}`."
  [opt]
  (if (map? opt)
    opt
    {:op (first opt) :axis (second opt) :amount (nth opt 2 1)}))

(defn apply-opt
  "Applies an `Opt` transformation `{:op op :axis axis :amount amount}` to a scheduled kernel entry `ke`.
   Valid operations:
     `:local`  - splits a spatial axis into outer global and inner local (threadblock) axis.
     `:upcast` - unrolls a spatial axis into registers/unrolled operations.
     `:group`  - splits a reduction axis into outer sequential loop and inner cooperative workgroup reduction (`lds`).
     `:unroll` - sets the loop unrolling factor for a reduction loop."
  [ke raw-opt]
  (let [opt (normalize-opt raw-opt)
        {:keys [op axis amount]} opt
        amount (long (or amount 1))
        axis (long (or axis 0))
        opts (conj (vec (or (:opts ke) [])) opt)
        rk (:rangeified ke)
        axes (or (:axes ke) (:out-shape rk) [(:out-size rk 1)])
        local-sz (vec (or (:local-size ke) [1 1 1]))
        global-sz (vec (or (:global-size ke) [1 1 1]))]
    (case op
      :local
      (let [orig-dim (nth axes axis 1)
            outer-dim (max 1 (long (quot (+ orig-dim amount -1) amount)))
            ;; If 1D, local-sz[0] becomes amount, global-sz[0] becomes outer-dim
            ;; If multi-D, map axis 0 -> x (or 1 -> y if 2D)
            new-local (if (== axis 0)
                        (assoc local-sz 0 amount)
                        (assoc local-sz 1 amount))
            new-global (if (== axis 0)
                         (assoc global-sz 0 outer-dim)
                         (assoc global-sz axis outer-dim))]
        (assoc ke
               :opts opts
               :local-size new-local
               :global-size new-global))

      :upcast
      (let [orig-dim (nth axes axis 1)
            outer-dim (max 1 (long (quot orig-dim amount)))
            new-axes (assoc (vec axes) axis outer-dim)]
        (assoc ke
               :opts opts
               :axes (if (== (count axes) 1) [outer-dim amount] new-axes)
               :upcast-factor amount))

      :group
      (let [new-local (if (== (first local-sz) 1)
                        (assoc local-sz 0 amount)
                        local-sz)]
        (assoc ke
               :opts opts
               :has-group? true
               :group-size amount
               :local-size new-local))

      :unroll
      (assoc ke
             :opts opts
             :unroll-factor amount)

      ;; Default: record opt
      (assoc ke :opts opts))))

;; -----------------------------------------------------------------------------
;; 3. Heuristic Hand-Coded Optimizations (tinygrad heuristic.py port)
;; -----------------------------------------------------------------------------

(defn hand-coded-optimizations
  "Selects optimized `Opt` recipes and launch geometry based on hardware mechanical sympathy.
   Specializes for:
     1. Fused RMSNorm / Softmax reductions over last axis (cooperative LDS workgroup).
     2. GEMV / Matvec (M <= 4, large N, K: 64-thread cooperative reduction along K, 100% coalesced).
     3. 2D Matmul (M >= 16, N >= 16, K >= 16: 16x16 threadblock + 4x4 register upcasting).
     4. Default elementwise loop tiling."
  [ke {:keys [target] :or {target :rocm}}]
  (let [rk (or (:rangeified ke) ke)
        out-size (long (or (:out-size rk) (:out-size ke) 1))
        out-shape (or (:out-shape rk) (:axes ke) [out-size])
        red-shape (or (:red-shape rk) (:red-axes ke) [])
        has-red? (or (true? (:has-reduce? ke)) (seq red-shape) (has-reduce-axis? rk))]
    (cond
      (= target :cpu)
      (let [opts [[:omp-parallel 0 1]
                  (when has-red? [:unroll 0 4])]]
        (assoc ke
               :target :cpu
               :global-size [1 1 1]
               :local-size [1 1 1]
               :opts (vec (remove nil? opts))))

      ;; Pattern 1: 2D Matmul (M >= 16, N >= 16, K >= 16)
      (and has-red?
           (>= (count out-shape) 2)
           (>= (first out-shape) 16)
           (>= (last out-shape) 16)
           (>= (long (or (first red-shape) 0)) 16))
      (let [m (first out-shape)
            n (last out-shape)
            tile-m 64
            tile-n 64
            grid-x (long (quot (+ m tile-m -1) tile-m))
            grid-y (long (quot (+ n tile-n -1) tile-n))]
        (assoc ke
               :target :rocm
               :unroll-factor 4
               :global-size [grid-x grid-y 1]
               :local-size [16 16 1]
               :opts [{:op :local :axis 0 :amount 16}
                      {:op :local :axis 1 :amount 16}
                      {:op :upcast :axis 0 :amount 4}
                      {:op :upcast :axis 1 :amount 4}
                      {:op :unroll :axis 0 :amount 4}]))

      ;; Pattern 2: GEMV / Matvec (M <= 4, large N, large K)
      (and has-red?
           (>= (count out-shape) 2)
           (<= (first out-shape) 4)
           (>= (last out-shape) 16)
           (>= (long (or (first red-shape) 0)) 64))
      (let [m (first out-shape)
            n (last out-shape)
            group-sz 64
            cols-per-block 4
            grid-y (long (quot (+ n cols-per-block -1) cols-per-block))]
        (assoc ke
               :target :rocm
               :has-group? true
               :group-size group-sz
               :unroll-factor 4
               :global-size [m grid-y 1]
               :local-size [group-sz cols-per-block 1]
               :opts [{:op :group :axis 0 :amount group-sz}
                      {:op :local :axis 1 :amount cols-per-block}
                      {:op :unroll :axis 0 :amount 4}]))

      ;; Pattern 3: Fused RMSNorm / LayerNorm / Softmax (reduction over last axis D)
      (or (= (:reduction-kind rk) :norm)
          (and has-red?
               (== 1 (count red-shape))
               (or (<= (count out-shape) 1)
                   (== 1 (last out-shape)))))
      (let [k-dim (long (or (first red-shape) 256))
            group-sz (cond (>= k-dim 256) 256 (>= k-dim 64) 64 :else 32)
            grid-rows (long (quot (+ out-size (if (== 1 (last out-shape)) 1 1) -1) 1))]
        (assoc ke
               :target :rocm
               :has-group? true
               :group-size group-sz
               :unroll-factor 4
               :global-size [grid-rows 1 1]
               :local-size [group-sz 1 1]
               :opts [{:op :group :axis 0 :amount group-sz}
                      {:op :unroll :axis 0 :amount 4}]))

      ;; Pattern 4: General Reduction or Elementwise Fallback
      :else
      (let [local-threads (cond
                            (<= out-size 32) (max 1 out-size)
                            (<= out-size 64) 64
                            (<= out-size 128) 128
                            :else 256)
            grid-blocks (long (quot (+ out-size local-threads -1) local-threads))]
        (assoc ke
               :target :rocm
               :global-size [grid-blocks 1 1]
               :local-size [local-threads 1 1]
               :opts (if has-red?
                       [{:op :local :axis 0 :amount local-threads}
                        {:op :unroll :axis 0 :amount 4}]
                       [{:op :local :axis 0 :amount local-threads}]))))))

(defn optimize-kernel
  "Applies declarative `OptOps` and launch geometry to a scheduled kernel map `kernel-entry`."
  ([kernel-entry]
   (optimize-kernel kernel-entry {:target :rocm}))
  ([kernel-entry opts]
   (hand-coded-optimizations kernel-entry opts)))
