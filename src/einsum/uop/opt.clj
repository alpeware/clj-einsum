(ns einsum.uop.opt
  "Declarative Kernel Optimization (`OptOps`) and heuristic scheduler for UOp kernels.
   Modeled on tinygrad's `OptOps` (`tinygrad/codegen/opt/`), `shift-to` range-splitting,
   and `hand_coded_optimizations` (`tinygrad/codegen/opt/heuristic.py`).
   Transforms rangeified loop axes into GPU grid (`:global`), threadblock (`:local`),
   vectorized register (`:upcast`), and cooperative reduction (`:group`, `:unroll`) configurations."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [einsum.uop.core :as uop]))

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

(defn extract-reduction-ranges
  "Extracts the vector of reduction `:uop/range` nodes from the first `:uop/reduce-axis` node in `rk`."
  [rk]
  (let [sink (:sink rk)]
    (when-let [red-node (some #(when (= :uop/reduce-axis (:op %)) %) (uop/toposort sink))]
      (vec (rest (:src red-node))))))

(defn extract-reduction-shape
  "Extracts the reduction dimension bounds `[k0 k1 ...]` from `rk`."
  [rk]
  (if-let [r-uops (seq (extract-reduction-ranges rk))]
    (mapv (fn [r]
            (let [bound-node (first (:src r))]
              (long (or (:arg bound-node) 1))))
          r-uops)
    (vec (or (:red-shape rk) (:red-axes rk) []))))

;; -----------------------------------------------------------------------------
;; 2. RDNA3 Beam Catalog & Structural Signature
;; -----------------------------------------------------------------------------

(defn non-singleton-dims
  "Filters out unit dimensions of size 1 across all axes (squeezing any dimension equal to 1),
   returning the effective non-singleton dimensions `[dim1 dim2 ...]`.
   If all dimensions are 1 (scalar tensor), returns `[1]`."
  [shape]
  (let [non-ones (vec (remove #(= (long %) 1) shape))]
    (if (empty? non-ones)
      [1]
      non-ones)))

(defn kernel-signature
  "Returns a persistent structural signature for `ke` independent of ephemeral kernel name.
   Normalizes leading singleton batch dimensions so that [1 N] and [N] match, and [1 S N]
   exposes [S N]."
  [ke]
  (let [rk (or (:rangeified ke) ke)
        out-shape (vec (or (:out-shape rk) (:axes ke) [(:out-size rk 1)]))
        red-shape (vec (let [rs (or (:red-shape rk) (:red-axes ke))]
                         (if (seq rs) rs (extract-reduction-shape rk))))
        has-red? (or (true? (:has-reduce? ke)) (seq red-shape) (has-reduce-axis? rk))
        eff-out (non-singleton-dims out-shape)
        eff-red (non-singleton-dims red-shape)]
    {:has-reduce? (boolean has-red?)
     :out-shape eff-out
     :red-shape eff-red}))

(defn load-rdna3-catalog
  "Loads the pre-populated RDNA3 beam catalog `resources/uop/rdna3_beam_catalog.edn`."
  []
  (if-let [res (io/resource "uop/rdna3_beam_catalog.edn")]
    (edn/read-string (slurp res))
    (let [f (io/file "resources/uop/rdna3_beam_catalog.edn")]
      (if (.exists f)
        (edn/read-string (slurp f))
        {}))))

(defn lookup-catalog
  "Looks up an optimization recipe in `resources/uop/rdna3_beam_catalog.edn`
   first by exact structural signature `{:has-reduce? ... :out-shape ... :red-shape ...}`,
   then by generic category keyword (`:gemma4-gemv`, `:gemma4-prefill-gemm`, `:gemma4-gemm`, `:gemma4-rmsnorm`)."
  ([ke]
   (lookup-catalog (load-rdna3-catalog) ke))
  ([catalog ke]
   (when (map? catalog)
     (let [sig (if (and (map? ke) (or (contains? ke :rangeified) (contains? ke :axes) (contains? ke :out-shape)))
                 (kernel-signature ke)
                 ke)
           eff-out (if (map? sig) (:out-shape sig) (non-singleton-dims (or (:out-shape ke) [])))
           eff-red (if (map? sig) (:red-shape sig) (non-singleton-dims (or (:red-shape ke) [])))]
       (or (get catalog sig)
           (when (map? sig)
             (let [k-dim (long (or (first eff-red) 0))
                   num-out (count eff-out)
                   m-dim (if (> num-out 1) (first eff-out) 1)
                   n-dim (last eff-out)]
               (cond
                 ;; 2D GEMM (M >= 16, N >= 16)
                 (and (:has-reduce? sig)
                      (>= num-out 2)
                      (>= m-dim 16)
                      (>= n-dim 16))
                 (get catalog :gemma4-gemm)

                 ;; Short-batch Prefill GEMM (2 <= M < 16, N >= 16, K >= 64)
                 (and (:has-reduce? sig)
                      (>= num-out 2)
                      (>= m-dim 2)
                      (>= n-dim 16)
                      (>= k-dim 64))
                 (or (get catalog :gemma4-prefill-gemm)
                     (get catalog :gemma4-gemm))

                 ;; RMSNorm / 1-row reduction
                 (and (:has-reduce? sig)
                      (or (= [1] eff-out)
                          (== 1 (long (or (:out-size ke) (:out-size sig) 1))))
                      (>= k-dim 64))
                 (get catalog :gemma4-rmsnorm)

                 ;; GEMV (M == 1, N >= 64, K >= 64)
                 (and (:has-reduce? sig)
                      (or (= num-out 1) (== m-dim 1))
                      (>= k-dim 64))
                 (get catalog :gemma4-gemv)))))))))

;; -----------------------------------------------------------------------------
;; 3. Real OptOps Range-Splitting & Axis Manipulation (`apply-opt` / `shift-to`)
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
            has-grp? (true? (:has-group? ke))
            new-local (if has-grp?
                        (assoc local-sz 1 amount)
                        (if (== axis 0)
                          (assoc local-sz 0 amount)
                          (assoc local-sz 1 amount)))
            new-global (if has-grp?
                         (assoc global-sz 0 outer-dim)
                         (if (== axis 0)
                           (assoc global-sz 0 outer-dim)
                           (assoc global-sz axis outer-dim)))]
        (assoc ke
               :opts opts
               :local-rows (if has-grp? amount (get ke :local-rows 1))
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
      (let [new-local (assoc local-sz 0 amount)]
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
        red-shape (let [rs (or (:red-shape rk) (:red-axes ke))]
                    (if (seq rs) rs (extract-reduction-shape rk)))
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
      (let [m (long (first out-shape))
            n (long (last out-shape))
            grid-m (max 1 (long (quot (+ m 16 -1) 16)))
            grid-n (max 1 (long (quot (+ n 16 -1) 16)))]
        (assoc ke
               :target :rocm
               :gemm-2d? true
               :s-dim m
               :n-dim n
               :unroll-factor 4
               :global-size [grid-n grid-m 1]
               :local-size [16 16 1]
               :opts [{:op :local :axis 0 :amount 16}
                      {:op :local :axis 1 :amount 16}
                      {:op :upcast :axis 0 :amount 4}
                      {:op :upcast :axis 1 :amount 4}
                      {:op :unroll :axis 0 :amount 4}]))

      ;; Pattern 2: Cooperative Reduction (GEMV, Prefill GEMM, RMSNorm, Softmax) when has-red? and innermost K >= 64
      (and has-red?
           (>= (long (or (last red-shape) (first red-shape) 0)) 64))
      (let [k (long (or (last red-shape) (first red-shape) 64))
            eff-out (non-singleton-dims out-shape)
            single-row? (or (== out-size 1) (<= out-size 2) (= 1 (long (last eff-out))))
            group-size (if single-row?
                         (cond (>= k 256) 256 (>= k 128) 128 :else 64)
                         64)
            local-rows (if (and (not single-row?) (>= out-size 4)) 4 1)
            grid-blocks (max 1 (quot (+ out-size local-rows -1) local-rows))
            opts (cond-> [{:op :group :axis 0 :amount group-size}]
                   (> local-rows 1) (conj {:op :local :axis 0 :amount local-rows})
                   true (conj {:op :unroll :axis 0 :amount 4}))]
        (assoc ke
               :target :rocm
               :has-group? true
               :group-size group-size
               :local-rows local-rows
               :unroll-factor 4
               :global-size [grid-blocks 1 1]
               :local-size [group-size local-rows 1]
               :opts opts))

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
  "Applies declarative `OptOps` and launch geometry to a scheduled kernel map `kernel-entry`.
   Checks the persistent RDNA3 beam catalog before falling back to hand-coded heuristics."
  ([kernel-entry]
   (optimize-kernel kernel-entry {:target :rocm}))
  ([kernel-entry opts]
   (let [target (:target opts :rocm)]
     (if (and (= target :rocm) (not (:disable-catalog? opts)))
       (if-let [recipe (lookup-catalog kernel-entry)]
         (let [base-ke (assoc kernel-entry :target :rocm)]
           (if (and (:local-size recipe) (:opts recipe))
             (let [has-group? (boolean (some #(= (:op %) :group) (:opts recipe)))
                   group-sz (or (:amount (first (filter #(= (:op %) :group) (:opts recipe)))) 64)
                   local-rows (or (:amount (first (filter #(= (:op %) :local) (:opts recipe)))) 1)
                   unroll-factor (or (:amount (first (filter #(= (:op %) :unroll) (:opts recipe)))) 4)
                   out-size (long (or (:out-size base-ke) 1))
                   rk (or (:rangeified base-ke) base-ke)
                   out-shape (or (:out-shape rk) (:axes base-ke) [out-size])
                   eff-out (non-singleton-dims out-shape)
                   m (if (> (count eff-out) 1) (long (first eff-out)) 1)
                   n (long (last eff-out))
                   is-2d-gemm? (and (= (:local-size recipe) [16 16 1])
                                    (not has-group?))
                   grid-m (max 1 (long (quot (+ m 16 -1) 16)))
                   grid-n (max 1 (long (quot (+ n 16 -1) 16)))
                   grid-blocks (max 1 (quot (+ out-size local-rows -1) local-rows))]
               (assoc base-ke
                      :gemm-2d? is-2d-gemm?
                      :s-dim (when is-2d-gemm? m)
                      :n-dim (when is-2d-gemm? n)
                      :has-group? has-group?
                      :group-size group-sz
                      :local-rows local-rows
                      :unroll-factor unroll-factor
                      :local-size (:local-size recipe)
                      :global-size (if is-2d-gemm? [grid-n grid-m 1] [grid-blocks 1 1])
                      :opts (:opts recipe)))
             (hand-coded-optimizations kernel-entry opts)))
         (hand-coded-optimizations kernel-entry opts))
       (hand-coded-optimizations kernel-entry opts)))))
