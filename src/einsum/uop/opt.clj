(ns einsum.uop.opt
  "Declarative Kernel Optimization (`OptOps`) and heuristic scheduler for UOp kernels.
   Modeled on tinygrad's `OptOps` (`tinygrad/codegen/opt/`) and `hand_coded_optimizations`.
   Transforms rangeified loop axes into GPU grid (`:global`), threadblock (`:local`),
   vectorized register (`:upcast`), and unrolled reduction (`:unroll`) configurations."
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

(defn- has-reduce-axis?
  "Returns true if the rangeified kernel contains a `:uop/reduce-axis` node."
  [rk]
  (boolean (some #(= :uop/reduce-axis (:op %)) (uop/toposort (:sink rk)))))

(defn- select-hand-coded-opts
  "Selects a declarative optimization recipe (`Opt` vector) and GPU launch geometry
   `{:global-size [gx gy gz] :local-size [lx ly lz] :opts [...]}` for a rangeified kernel `rk`."
  [rk target]
  (let [out-size (long (or (:out-size rk) 1))
        has-red? (has-reduce-axis? rk)
        local-threads (cond
                        (= target :cpu) 1
                        (<= out-size 32) (max 1 out-size)
                        (<= out-size 64) 64
                        (<= out-size 128) 128
                        :else 256)
        grid-blocks (if (= target :cpu)
                      1
                      (long (quot (+ out-size local-threads -1) local-threads)))
        opts (cond
               (= target :cpu)
               [[:omp-parallel 0 1]
                (when has-red? [:unroll 0 4])]

               has-red?
               [[:global 0 grid-blocks]
                [:local 0 local-threads]
                [:unroll 0 4]]

               :else
               [[:global 0 grid-blocks]
                [:local 0 local-threads]
                [:upcast 0 4]])]
    {:target target
     :has-reduce? has-red?
     :out-size out-size
     :global-size [grid-blocks 1 1]
     :local-size [local-threads 1 1]
     :opts (vec (remove nil? opts))}))

(defn optimize-kernel
  "Applies declarative `OptOps` and launch geometry to a scheduled kernel map `kernel-entry`."
  ([kernel-entry]
   (optimize-kernel kernel-entry {:target :rocm}))
  ([kernel-entry {:keys [target] :or {target :rocm}}]
   (let [rk (:rangeified kernel-entry)
         opt-plan (select-hand-coded-opts rk target)]
     (assoc kernel-entry
            :opt-plan opt-plan
            :global-size (:global-size opt-plan)
            :local-size (:local-size opt-plan)
            :opts (:opts opt-plan)))))
