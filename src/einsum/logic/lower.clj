(ns einsum.logic.lower
  "Lowering engine from normalized Tensor Logic AST to StableHLO EDN SSA graph."
  (:require [einsum.logic.ast :as ast]
            [einsum.logic.dce :as dce]
            [einsum.logic.expand :as expand]
            [einsum.logic.index :as idx]
            [einsum.logic.shape :as shape]
            [einsum.compiler.stablehlo :as shlo]
            [einsum.runtime.weights :as weights]
            [clojure.set :as set]))

(defn- gen-id [prefix counter]
  (keyword (str prefix "_" (swap! counter inc))))

(defn- indices->dim-numbers [ordered-idxs target-idxs]
  (let [idx->pos (into {} (map-indexed (fn [pos k] [k pos]) ordered-idxs))]
    (mapv #(get idx->pos %) target-idxs)))

(defn- has-post-act? [attrs]
  (boolean (or (:scale attrs) (:act attrs) (:softcap attrs) (:clamp attrs))))

(defn- emit-post-activation! [eqns-atom counter in-var out-var attrs dtype]
  (let [scale (:scale attrs)
        act (:act attrs)
        softcap (:softcap attrs)
        clamp (:clamp attrs)]
    (cond
      scale
      (let [c-var (gen-id "c_scale" counter)
            c-eqn {:op :stablehlo/constant :value (double scale) :outvars [c-var]}
            mul-var (if (or act softcap clamp) (gen-id "t_scaled" counter) out-var)
            mul-eqn {:op :stablehlo/multiply :invars [in-var c-var] :outvars [mul-var]}]
        (swap! eqns-atom conj c-eqn mul-eqn)
        (if (or act softcap clamp)
          (emit-post-activation! eqns-atom counter mul-var out-var (dissoc attrs :scale) dtype)
          mul-var))

      ;; Clamp / ramp activation: clamp(x, low, high) = min(max(x, low), high)
      ;; Datalog continuous relaxation uses clamp(A + A x P, 0.0, 1.0)
      (or (:clamp attrs) (= act :clamp))
      (let [[low high] (cond
                         (vector? (:clamp attrs)) (:clamp attrs)
                         (sequential? (:clamp attrs)) (vec (:clamp attrs))
                         :else [0.0 1.0])
            c-low (gen-id "c_clamp_low" counter)
            low-eqn {:op :stablehlo/constant :value (double low) :outvars [c-low]}
            c-high (gen-id "c_clamp_high" counter)
            high-eqn {:op :stablehlo/constant :value (double high) :outvars [c-high]}
            t-max (gen-id "t_clamp_max" counter)
            max-eqn {:op :stablehlo/maximum :invars [in-var c-low] :outvars [t-max]}
            min-eqn {:op :stablehlo/minimum :invars [t-max c-high] :outvars [out-var]}]
        (swap! eqns-atom conj low-eqn high-eqn max-eqn min-eqn)
        out-var)

      (or (= act :sigmoid) (= act :logistic))
      (let [log-eqn {:op :stablehlo/logistic :invars [in-var] :outvars [out-var]}]
        (swap! eqns-atom conj log-eqn)
        out-var)

      (or (= act :silu) (= act :swish))
      (let [log-var (gen-id "t_logistic" counter)
            log-eqn {:op :stablehlo/logistic :invars [in-var] :outvars [log-var]}
            mul-eqn {:op :stablehlo/multiply :invars [in-var log-var] :outvars [out-var]}]
        (swap! eqns-atom conj log-eqn mul-eqn)
        out-var)

      (= act :tanh)
      (let [t-eqn {:op :stablehlo/tanh :invars [in-var] :outvars [out-var]}]
        (swap! eqns-atom conj t-eqn)
        out-var)

      (= act :gelu)
      (let [c-half (gen-id "c_half" counter)
            e-half {:op :stablehlo/constant :value 0.5 :outvars [c-half]}
            c-sqrt (gen-id "c_sqrt" counter)
            e-sqrt {:op :stablehlo/constant :value 0.7978845608 :outvars [c-sqrt]}
            c-poly (gen-id "c_poly" counter)
            e-poly {:op :stablehlo/constant :value 0.044715 :outvars [c-poly]}
            c-one (gen-id "c_one" counter)
            e-one {:op :stablehlo/constant :value 1.0 :outvars [c-one]}

            t-x2 (gen-id "t_x2" counter)
            e-x2 {:op :stablehlo/multiply :invars [in-var in-var] :outvars [t-x2]}
            t-x3 (gen-id "t_x3" counter)
            e-x3 {:op :stablehlo/multiply :invars [t-x2 in-var] :outvars [t-x3]}
            t-poly (gen-id "t_poly" counter)
            e-poly-mul {:op :stablehlo/multiply :invars [t-x3 c-poly] :outvars [t-poly]}
            t-inner (gen-id "t_inner" counter)
            e-inner {:op :stablehlo/add :invars [in-var t-poly] :outvars [t-inner]}
            t-scaled-inner (gen-id "t_scaled_inner" counter)
            e-sinner {:op :stablehlo/multiply :invars [t-inner c-sqrt] :outvars [t-scaled-inner]}
            t-tanh (gen-id "t_tanh" counter)
            e-tanh {:op :stablehlo/tanh :invars [t-scaled-inner] :outvars [t-tanh]}
            t-plus-one (gen-id "t_plus_one" counter)
            e-plus-one {:op :stablehlo/add :invars [t-tanh c-one] :outvars [t-plus-one]}
            t-half-x (gen-id "t_half_x" counter)
            e-half-x {:op :stablehlo/multiply :invars [in-var c-half] :outvars [t-half-x]}
            e-final {:op :stablehlo/multiply :invars [t-half-x t-plus-one] :outvars [out-var]}]
        (swap! eqns-atom conj e-half e-sqrt e-poly e-one e-x2 e-x3 e-poly-mul e-inner e-sinner e-tanh e-plus-one e-half-x e-final)
        out-var)

      softcap
      (let [cap (double softcap)
            c-cap (gen-id "c_cap" counter)
            c-cap-eqn {:op :stablehlo/constant :value cap :outvars [c-cap]}
            c-inv (gen-id "c_inv_cap" counter)
            c-inv-eqn {:op :stablehlo/constant :value (/ 1.0 cap) :outvars [c-inv]}
            scaled-var (gen-id "t_scaled_cap" counter)
            scale-eqn {:op :stablehlo/multiply :invars [in-var c-inv] :outvars [scaled-var]}
            tanh-var (gen-id "t_tanh_cap" counter)
            tanh-eqn {:op :stablehlo/tanh :invars [scaled-var] :outvars [tanh-var]}
            final-eqn {:op :stablehlo/multiply :invars [tanh-var c-cap] :outvars [out-var]}]
        (swap! eqns-atom conj c-cap-eqn c-inv-eqn scale-eqn tanh-eqn final-eqn)
        out-var)

      :else
      in-var)))

(defn- lower-binary-contraction!
  [eqns-atom counter head attrs lhs rhs known-shapes var-dtypes final-out-var]
  (let [head-idxs (vec (rest head))
        lhs-name (first lhs)
        lhs-idxs (vec (rest lhs))
        rhs-name (first rhs)
        rhs-idxs (vec (rest rhs))
        {:keys [contracting batch lhs-free rhs-free]}
        (idx/partition-indices lhs-idxs rhs-idxs head-idxs)]
    (if (and (empty? contracting)
             (or (= (set lhs-idxs) (set head-idxs))
                 (= (set rhs-idxs) (set head-idxs))))
      ;; Elementwise or Broadcast product
      (let [primary-is-lhs? (= (set lhs-idxs) (set head-idxs))
            primary-name (if primary-is-lhs? lhs-name rhs-name)
            primary-idxs (if primary-is-lhs? lhs-idxs rhs-idxs)
            secondary-name (if primary-is-lhs? rhs-name lhs-name)
            secondary-idxs (if primary-is-lhs? rhs-idxs lhs-idxs)
            secondary-v (cond
                          (= primary-idxs secondary-idxs)
                          secondary-name

                          (= (set primary-idxs) (set secondary-idxs))
                          (let [rhs-perm (indices->dim-numbers secondary-idxs primary-idxs)
                                out-r (gen-id "t_trans_r" counter)
                                trans-r {:op :stablehlo/transpose :invars [secondary-name] :outvars [out-r] :attrs {:permutation rhs-perm}}]
                            (swap! eqns-atom conj trans-r)
                            out-r)

                          (set/subset? (set secondary-idxs) (set primary-idxs))
                          (let [bcast-dims (indices->dim-numbers primary-idxs secondary-idxs)
                                target-shape (get known-shapes primary-name)
                                out-b (gen-id "t_bcast_r" counter)
                                bcast-r {:op :stablehlo/broadcast_in_dim :invars [secondary-name] :outvars [out-b]
                                         :attrs {:broadcast_dimensions bcast-dims :target_shape target-shape}}]
                            (swap! eqns-atom conj bcast-r)
                            out-b)

                          :else
                          secondary-name)
            needs-perm? (not= primary-idxs head-idxs)
            has-post-act? (has-post-act? attrs)
            mul-out-var (if (or needs-perm? has-post-act?)
                          (gen-id "t_mul" counter)
                          final-out-var)
            op-kw (if (or (= (:op attrs) :add) (= (:op attrs) :+) (:add? attrs))
                    :stablehlo/add
                    :stablehlo/multiply)
            mul-eqn {:op op-kw
                     :invars [primary-name secondary-v]
                     :outvars [mul-out-var]}]
        (swap! eqns-atom conj mul-eqn)
        (let [perm-out-var
              (if needs-perm?
                (let [perm (indices->dim-numbers primary-idxs head-idxs)
                      out-v (if has-post-act? (gen-id "t_trans" counter) final-out-var)
                      trans-eqn {:op :stablehlo/transpose
                                 :invars [mul-out-var]
                                 :outvars [out-v]
                                 :attrs {:permutation perm}}]
                  (swap! eqns-atom conj trans-eqn)
                  out-v)
                mul-out-var)]
          (when has-post-act?
            (emit-post-activation! eqns-atom counter perm-out-var final-out-var attrs (get var-dtypes primary-name :f32)))))
      ;; Semiring contraction (:min-plus / :tropical or :max-product / :viterbi)
      (let [semiring (:semiring attrs)]
        (if (or (= semiring :min-plus) (= semiring :tropical)
                (= semiring :max-product) (= semiring :viterbi))
          (let [all-idxs (vec (concat batch lhs-free contracting rhs-free))
                lhs-shape (get known-shapes lhs-name)
                rhs-shape (get known-shapes rhs-name)
                idx->dim (into (zipmap lhs-idxs lhs-shape) (zipmap rhs-idxs rhs-shape))
                target-shape (mapv idx->dim all-idxs)
                lhs-bcast-dims (indices->dim-numbers all-idxs lhs-idxs)
                lhs-v (if (= lhs-idxs all-idxs)
                        lhs-name
                        (let [out-lhs-b (gen-id "t_semiring_lhs_bcast" counter)
                              bcast-lhs {:op :stablehlo/broadcast_in_dim
                                         :invars [lhs-name]
                                         :outvars [out-lhs-b]
                                         :attrs {:broadcast_dimensions lhs-bcast-dims :target_shape target-shape}}]
                          (swap! eqns-atom conj bcast-lhs)
                          out-lhs-b))
                rhs-bcast-dims (indices->dim-numbers all-idxs rhs-idxs)
                rhs-v (if (= rhs-idxs all-idxs)
                        rhs-name
                        (let [out-rhs-b (gen-id "t_semiring_rhs_bcast" counter)
                              bcast-rhs {:op :stablehlo/broadcast_in_dim
                                         :invars [rhs-name]
                                         :outvars [out-rhs-b]
                                         :attrs {:broadcast_dimensions rhs-bcast-dims :target_shape target-shape}}]
                          (swap! eqns-atom conj bcast-rhs)
                          out-rhs-b))
                prod-op (if (or (= semiring :min-plus) (= semiring :tropical))
                          :stablehlo/add
                          :stablehlo/multiply)
                prod-out-var (gen-id "t_semiring_prod" counter)
                prod-eqn {:op prod-op :invars [lhs-v rhs-v] :outvars [prod-out-var]}
                _ (swap! eqns-atom conj prod-eqn)
                contracting-axes (indices->dim-numbers all-idxs contracting)
                red-op (if (or (= semiring :min-plus) (= semiring :tropical))
                         :stablehlo/reduce_min
                         :stablehlo/reduce_max)
                raw-reduced-idxs (vec (concat batch lhs-free rhs-free))
                needs-perm? (not= raw-reduced-idxs head-idxs)
                has-post-act? (has-post-act? attrs)
                red-out-var (if (or needs-perm? has-post-act?)
                              (gen-id "t_semiring_red" counter)
                              final-out-var)
                red-eqn {:op red-op :invars [prod-out-var] :outvars [red-out-var] :attrs {:axes contracting-axes :keep_dims false}}
                _ (swap! eqns-atom conj red-eqn)
                perm-out-var
                (if needs-perm?
                  (let [perm (indices->dim-numbers raw-reduced-idxs head-idxs)
                        out-v (if has-post-act? (gen-id "t_trans" counter) final-out-var)
                        trans-eqn {:op :stablehlo/transpose
                                   :invars [red-out-var]
                                   :outvars [out-v]
                                   :attrs {:permutation perm}}]
                    (swap! eqns-atom conj trans-eqn)
                    out-v)
                  red-out-var)]
            (when has-post-act?
              (emit-post-activation! eqns-atom counter perm-out-var final-out-var attrs (get var-dtypes lhs-name :f32)))
            nil)
          ;; Standard dot_general contraction
          (let [lhs-c-dims (indices->dim-numbers lhs-idxs contracting)
                rhs-c-dims (indices->dim-numbers rhs-idxs contracting)
                lhs-b-dims (indices->dim-numbers lhs-idxs batch)
                rhs-b-dims (indices->dim-numbers rhs-idxs batch)

                dot-attrs {:contracting_dims {:lhs lhs-c-dims :rhs rhs-c-dims}
                           :batch_dims {:lhs lhs-b-dims :rhs rhs-b-dims}}

                raw-dot-idxs (vec (concat batch lhs-free rhs-free))
                needs-perm? (not= raw-dot-idxs head-idxs)
                has-post-act? (has-post-act? attrs)

                dot-out-var (if (or needs-perm? has-post-act?)
                              (gen-id "t_dot" counter)
                              final-out-var)

                dot-eqn {:op :stablehlo/dot_general
                         :invars [lhs-name rhs-name]
                         :outvars [dot-out-var]
                         :attrs dot-attrs}]
            (swap! eqns-atom conj dot-eqn)
            (let [perm-out-var
                  (if needs-perm?
                    (let [perm (indices->dim-numbers raw-dot-idxs head-idxs)
                          out-v (if has-post-act? (gen-id "t_trans" counter) final-out-var)
                          trans-eqn {:op :stablehlo/transpose
                                     :invars [dot-out-var]
                                     :outvars [out-v]
                                     :attrs {:permutation perm}}]
                      (swap! eqns-atom conj trans-eqn)
                      out-v)
                    dot-out-var)
                  _post-out-var
                  (if has-post-act?
                    (emit-post-activation! eqns-atom counter perm-out-var final-out-var attrs (get var-dtypes lhs-name :f32))
                    perm-out-var)]
              nil)))))))

(defn- lower-unary-equation!
  [eqns-atom counter head attrs term known-shapes var-dtypes final-out-var]
  (let [head-idxs (vec (rest head))
        term-name (first term)
        term-idxs (vec (rest term))
        has-post-act? (has-post-act? attrs)]
    (cond
      ;; Direct copy or post-activation
      (= head-idxs term-idxs)
      (if has-post-act?
        (emit-post-activation! eqns-atom counter term-name final-out-var attrs (get var-dtypes term-name :f32))
        (let [target-shape (or (get known-shapes (first head))
                               (get known-shapes term-name))
              reshape-eqn {:op :stablehlo/reshape
                           :invars [term-name]
                           :outvars [final-out-var]
                           :attrs {:shape target-shape}}]
          (swap! eqns-atom conj reshape-eqn)))

      ;; Transpose / permutation
      (= (set head-idxs) (set term-idxs))
      (let [perm (indices->dim-numbers term-idxs head-idxs)
            trans-out (if has-post-act? (gen-id "t_trans" counter) final-out-var)
            trans-eqn {:op :stablehlo/transpose :invars [term-name] :outvars [trans-out] :attrs {:permutation perm}}]
        (swap! eqns-atom conj trans-eqn)
        (when has-post-act?
          (emit-post-activation! eqns-atom counter trans-out final-out-var attrs (get var-dtypes term-name :f32))))

      :else
      ;; Broadcast
      (let [target-shape (get known-shapes (first head))
            bcast-dims (indices->dim-numbers head-idxs term-idxs)
            bcast-out (if has-post-act? (gen-id "t_bcast" counter) final-out-var)
            bcast-eqn {:op :stablehlo/broadcast_in_dim
                       :invars [term-name]
                       :outvars [bcast-out]
                       :attrs {:broadcast_dimensions bcast-dims
                               :target_shape target-shape}}]
        (swap! eqns-atom conj bcast-eqn)
        (when has-post-act?
          (emit-post-activation! eqns-atom counter bcast-out final-out-var attrs (get var-dtypes term-name :f32)))))))

(defn- lower-gather! [eqns-atom counter _head table-term idx-term final-out-var known-shapes]
  (let [table-name (first table-term)
        idx-name (first idx-term)
        table-shape (get known-shapes table-name)
        hidden-dim (last table-shape)
        idx-shape (get known-shapes idx-name)
        expanded-idx-shape (conj (vec idx-shape) 1)
        reshaped-idx-var (gen-id "t_idx_reshape" counter)
        reshape-eqn {:op :stablehlo/reshape
                     :invars [idx-name]
                     :outvars [reshaped-idx-var]
                     :attrs {:shape expanded-idx-shape}}
        final-rank (count expanded-idx-shape)
        gather-eqn {:op :stablehlo/gather
                    :invars [table-name reshaped-idx-var]
                    :outvars [final-out-var]
                    :attrs {:offset_dims [(dec final-rank)]
                            :collapsed_slice_dims [0]
                            :start_index_map [0]
                            :index_vector_dim (dec final-rank)
                            :slice_sizes [1 hidden-dim]}}]
    (swap! eqns-atom conj reshape-eqn gather-eqn)))

(defn- lower-slice! [eqns-atom _head in-term attrs final-out-var]
  (let [in-name (first in-term)
        starts (or (:start_indices attrs) (:start attrs) [0 0 0])
        limits (or (:limit_indices attrs) (:limit attrs) [1 128 768])
        strides (or (:strides attrs) (vec (repeat (count starts) 1)))
        slice-eqn {:op :stablehlo/slice
                   :invars [in-name]
                   :outvars [final-out-var]
                   :attrs {:start_indices starts
                           :limit_indices limits
                           :strides strides}}]
    (swap! eqns-atom conj slice-eqn)))

(defn- lower-dynamic-slice! [eqns-atom _head in-term opt-starts attrs final-out-var]
  (let [in-name (first in-term)
        starts (or (when (vector? opt-starts) opt-starts)
                   (:start_indices attrs)
                   (:start-indices attrs)
                   [0 0 0])
        slice-sizes (or (:slice_sizes attrs) (:slice-sizes attrs) [1 1 1])
        slice-eqn {:op :stablehlo/dynamic_slice
                   :invars [in-name]
                   :outvars [final-out-var]
                   :attrs {:start_indices starts
                           :slice_sizes slice-sizes}}]
    (swap! eqns-atom conj slice-eqn)))

(defn- lower-dynamic-update-slice! [eqns-atom counter _head op-term up-term attrs final-out-var]
  (let [op-name (first op-term)
        up-name (first up-term)
        starts (or (:start_indices attrs) (:start-indices attrs) [0 0 0 0])
        window (:window attrs)
        actual-starts (if (and window (some keyword? starts))
                        (let [c-win (gen-id "c_win" counter)
                              c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                              c-win-1d (gen-id "c_win_1d" counter)
                              c-win-1d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-win] :outvars [c-win-1d]
                                            :attrs {:broadcast_dimensions [] :target_shape [1]}}
                              _ (swap! eqns-atom conj c-win-eqn c-win-1d-eqn)]
                          (mapv (fn [idx]
                                  (if (keyword? idx)
                                    (let [mod-var (gen-id "slot_mod" counter)
                                          mod-eqn {:op :stablehlo/remainder :invars [idx c-win-1d] :outvars [mod-var]}]
                                      (swap! eqns-atom conj mod-eqn)
                                      mod-var)
                                    idx))
                                starts))
                        starts)
        update-eqn {:op :stablehlo/dynamic_update_slice
                    :invars [op-name up-name]
                    :outvars [final-out-var]
                    :attrs {:start_indices actual-starts}}]
    (swap! eqns-atom conj update-eqn)))

(defn- lower-reshape! [eqns-atom head in-term attrs final-out-var]
  (let [in-name (first in-term)
        shape (or (:shape attrs) (vec (rest head)))
        reshape-eqn {:op :stablehlo/reshape
                     :invars [in-name]
                     :outvars [final-out-var]
                     :attrs {:shape shape}}]
    (swap! eqns-atom conj reshape-eqn)))

(defn- lower-layer-norm! [eqns-atom counter _head in-term gamma-term beta-term attrs final-out-var]
  (let [in-name (first in-term)
        gamma-name (first gamma-term)
        beta-name (first beta-term)
        eps (or (:eps attrs) 1e-5)
        mean-var (gen-id "ln_mean" counter)
        mean-eqn {:op :stablehlo/reduce_mean :invars [in-name] :outvars [mean-var] :attrs {:axes [-1] :keep_dims true}}
        diff-var (gen-id "ln_diff" counter)
        diff-eqn {:op :stablehlo/subtract :invars [in-name mean-var] :outvars [diff-var]}
        diff-sq-var (gen-id "ln_diff_sq" counter)
        diff-sq-eqn {:op :stablehlo/multiply :invars [diff-var diff-var] :outvars [diff-sq-var]}
        var-var (gen-id "ln_var" counter)
        var-eqn {:op :stablehlo/reduce_mean :invars [diff-sq-var] :outvars [var-var] :attrs {:axes [-1] :keep_dims true}}
        c-eps (gen-id "ln_eps" counter)
        c-eps-eqn {:op :stablehlo/constant :value (double eps) :outvars [c-eps]}
        var-eps-var (gen-id "ln_var_eps" counter)
        var-eps-eqn {:op :stablehlo/add :invars [var-var c-eps] :outvars [var-eps-var]}
        std-var (gen-id "ln_std" counter)
        std-eqn {:op :stablehlo/sqrt :invars [var-eps-var] :outvars [std-var]}
        xhat-var (gen-id "ln_xhat" counter)
        xhat-eqn {:op :stablehlo/divide :invars [diff-var std-var] :outvars [xhat-var]}
        scaled-var (gen-id "ln_scaled" counter)
        scaled-eqn {:op :stablehlo/multiply :invars [xhat-var gamma-name] :outvars [scaled-var]}
        final-eqn {:op :stablehlo/add :invars [scaled-var beta-name] :outvars [final-out-var]}]
    (swap! eqns-atom conj mean-eqn diff-eqn diff-sq-eqn var-eqn c-eps-eqn var-eps-eqn std-eqn xhat-eqn scaled-eqn final-eqn)))

(defn- lower-causal-softmax!
  ([eqns-atom counter in-term attrs final-out-var known-shapes]
   (lower-causal-softmax! eqns-atom counter in-term attrs final-out-var known-shapes :f32))
  ([eqns-atom counter in-term attrs final-out-var known-shapes dtype]
   (let [in-name (first in-term)
         orig-dtype (or dtype :f32)
         scores-shape (get known-shapes in-name [1 12 128 128])
         batch (nth scores-shape 0 1)
         num-heads (nth scores-shape 1 12)
         q-len (nth scores-shape 2 128)
         kv-len (nth scores-shape 3 128)
         window (or (:sliding-window attrs) (:window-size attrs))
         pos-var (:pos attrs)
         needs-f32? (not= orig-dtype :f32)
         f32-in-var (if needs-f32? (gen-id "s_f32" counter) in-name)
         conv-in-eqn (when needs-f32?
                       {:op :stablehlo/convert :invars [in-name] :outvars [f32-in-var] :attrs {:target_dtype :f32}})
         masked-var (gen-id "t_masked" counter)]
     (when needs-f32? (swap! eqns-atom conj conv-in-eqn))
     (let [cmp-final
           (if pos-var
             (let [pos-scalar (gen-id "pos_s" counter)
                   pos-s-eqn {:op :stablehlo/reshape :invars [pos-var] :outvars [pos-scalar] :attrs {:shape []}}
                   pos-4d (gen-id "pos_4d" counter)
                   pos-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [pos-scalar] :outvars [pos-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
                   iota-1d (gen-id "iota_1d" counter)
                   iota-1d-eqn {:op :stablehlo/iota :outvars [iota-1d] :attrs {:len kv-len :dtype :i32 :iota_dimension 0}}
                   iota-4d (gen-id "iota_4d" counter)
                   iota-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [iota-1d] :outvars [iota-4d] :attrs {:broadcast_dimensions [3] :target_shape [1 1 1 kv-len]}}
                   cmp-fut (gen-id "cmp_fut" counter)
                   cmp-fut-eqn {:op :stablehlo/compare :invars [iota-4d pos-4d] :outvars [cmp-fut] :attrs {:comparison_direction "GT"}}
                   _ (swap! eqns-atom conj pos-s-eqn pos-4d-eqn iota-1d-eqn iota-4d-eqn cmp-fut-eqn)]
               (if window
                 (let [c-win (gen-id "c_win" counter)
                       c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                       p-sub-win (gen-id "p_sub_w" counter)
                       p-sub-eqn {:op :stablehlo/subtract :invars [pos-scalar c-win] :outvars [p-sub-win]}
                       c-one (gen-id "c_one" counter)
                       c-one-eqn {:op :stablehlo/constant :value 1 :type [:tensor [] :i32] :outvars [c-one]}
                       min-p (gen-id "min_p" counter)
                       min-p-eqn {:op :stablehlo/add :invars [p-sub-win c-one] :outvars [min-p]}
                       min-p-4d (gen-id "min_p_4d" counter)
                       min-p-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [min-p] :outvars [min-p-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 1 kv-len]}}
                       cmp-old (gen-id "cmp_old" counter)
                       cmp-old-eqn {:op :stablehlo/compare :invars [iota-4d min-p-4d] :outvars [cmp-old] :attrs {:comparison_direction "LT"}}
                       cmp-comb (gen-id "cmp_comb" counter)
                       cmp-comb-eqn {:op :stablehlo/or :invars [cmp-fut cmp-old] :outvars [cmp-comb]}]
                   (swap! eqns-atom conj c-win-eqn p-sub-eqn c-one-eqn min-p-eqn min-p-4d-eqn cmp-old-eqn cmp-comb-eqn)
                   cmp-comb)
                 cmp-fut))
             ;; Dynamic 2D Causal Mask for Prefill (zero dense constants in MLIR)
             (let [iota-q-1d (gen-id "iota_q_1d" counter)
                   iota-q-1d-eqn {:op :stablehlo/iota :outvars [iota-q-1d] :attrs {:len q-len :dtype :i32 :iota_dimension 0}}
                   iota-q-4d (gen-id "iota_q_4d" counter)
                   iota-q-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [iota-q-1d] :outvars [iota-q-4d] :attrs {:broadcast_dimensions [2] :target_shape [1 1 q-len kv-len]}}
                   iota-kv-1d (gen-id "iota_kv_1d" counter)
                   iota-kv-1d-eqn {:op :stablehlo/iota :outvars [iota-kv-1d] :attrs {:len kv-len :dtype :i32 :iota_dimension 0}}
                   iota-kv-4d (gen-id "iota_kv_4d" counter)
                   iota-kv-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [iota-kv-1d] :outvars [iota-kv-4d] :attrs {:broadcast_dimensions [3] :target_shape [1 1 q-len kv-len]}}
                   cmp-fut (gen-id "cmp_fut" counter)
                   cmp-fut-eqn {:op :stablehlo/compare :invars [iota-kv-4d iota-q-4d] :outvars [cmp-fut] :attrs {:comparison_direction "GT"}}
                   _ (swap! eqns-atom conj iota-q-1d-eqn iota-q-4d-eqn iota-kv-1d-eqn iota-kv-4d-eqn cmp-fut-eqn)]
               (if window
                 (let [c-win (gen-id "c_win" counter)
                       c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                       c-win-4d (gen-id "c_win_4d" counter)
                       c-win-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-win] :outvars [c-win-4d] :attrs {:broadcast_dimensions [] :target_shape [1 1 q-len kv-len]}}
                       diff (gen-id "diff_qk" counter)
                       diff-eqn {:op :stablehlo/subtract :invars [iota-q-4d iota-kv-4d] :outvars [diff]}
                       cmp-old (gen-id "cmp_old" counter)
                       cmp-old-eqn {:op :stablehlo/compare :invars [diff c-win-4d] :outvars [cmp-old] :attrs {:comparison_direction "GE"}}
                       cmp-comb (gen-id "cmp_comb" counter)
                       cmp-comb-eqn {:op :stablehlo/or :invars [cmp-fut cmp-old] :outvars [cmp-comb]}]
                   (swap! eqns-atom conj c-win-eqn c-win-4d-eqn diff-eqn cmp-old-eqn cmp-comb-eqn)
                   cmp-comb)
                 cmp-fut)))
           mask-shape (if pos-var [1 1 1 kv-len] [1 1 q-len kv-len])
           c-neg (gen-id "c_neg" counter)
           c-neg-eqn {:op :stablehlo/constant :value -10000.0 :type [:tensor [] :f32] :outvars [c-neg]}
           c-neg-4d (gen-id "c_neg_4d" counter)
           c-neg-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-neg] :outvars [c-neg-4d] :attrs {:broadcast_dimensions [] :target_shape mask-shape}}
           c-zero (gen-id "c_zero" counter)
           c-zero-eqn {:op :stablehlo/constant :value 0.0 :type [:tensor [] :f32] :outvars [c-zero]}
           c-zero-4d (gen-id "c_zero_4d" counter)
           c-zero-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-zero] :outvars [c-zero-4d] :attrs {:broadcast_dimensions [] :target_shape mask-shape}}
           dyn-mask (gen-id "c_dyn_mask" counter)
           dyn-mask-eqn {:op :stablehlo/select :invars [cmp-final c-neg-4d c-zero-4d] :outvars [dyn-mask]}
           mask-broad (gen-id "c_mask_bcast" counter)
           mask-broad-eqn {:op :stablehlo/broadcast_in_dim :invars [dyn-mask] :outvars [mask-broad] :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch num-heads q-len kv-len]}}
           masked-eqn {:op :stablehlo/add :invars [f32-in-var mask-broad] :outvars [masked-var]}]
       (swap! eqns-atom conj c-neg-eqn c-neg-4d-eqn c-zero-eqn c-zero-4d-eqn dyn-mask-eqn mask-broad-eqn masked-eqn))
     (let [max-var (gen-id "t_smax" counter)
           max-eqn {:op :stablehlo/reduce_max :invars [masked-var] :outvars [max-var] :attrs {:axes [-1] :keep_dims true}}
           max-bcast-var (gen-id "t_smax_bcast" counter)
           max-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [max-var] :outvars [max-bcast-var]
                          :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch num-heads q-len kv-len]}}
           diff-var (gen-id "t_sdiff" counter)
           diff-eqn {:op :stablehlo/subtract :invars [masked-var max-bcast-var] :outvars [diff-var]}
           exp-var (gen-id "t_sexp" counter)
           exp-eqn {:op :stablehlo/exp :invars [diff-var] :outvars [exp-var]}
           sum-var (gen-id "t_ssum" counter)
           sum-eqn {:op :stablehlo/reduce_sum :invars [exp-var] :outvars [sum-var] :attrs {:axes [-1] :keep_dims true}}
           sum-bcast-var (gen-id "t_ssum_bcast" counter)
           sum-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [sum-var] :outvars [sum-bcast-var]
                          :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch num-heads q-len kv-len]}}
           f32-div-var (if needs-f32? (gen-id "t_sdiv" counter) final-out-var)
           div-eqn {:op :stablehlo/divide :invars [exp-var sum-bcast-var] :outvars [f32-div-var]}
           conv-out-eqn (when needs-f32?
                          {:op :stablehlo/convert :invars [f32-div-var] :outvars [final-out-var] :attrs {:target_dtype orig-dtype}})]
       (swap! eqns-atom conj max-eqn max-bcast-eqn diff-eqn exp-eqn sum-eqn sum-bcast-eqn div-eqn)
       (when needs-f32? (swap! eqns-atom conj conv-out-eqn))))))

(defn- lower-softmax!
  ([eqns-atom counter head in-term attrs final-out-var known-shapes]
   (lower-softmax! eqns-atom counter head in-term attrs final-out-var known-shapes :f32))
  ([eqns-atom counter head in-term attrs final-out-var known-shapes dtype]
   (let [in-name (first in-term)
         orig-dtype (or dtype :f32)
         in-shape (or (get known-shapes in-name)
                      (get known-shapes (first head)))
         rank (count in-shape)
         axes (or (:axes attrs)
                  (when-let [ax (:axis attrs)] [ax])
                  [(dec rank)])
         norm-axes (mapv #(if (neg? %) (+ rank %) %) axes)
         needs-f32? (not= orig-dtype :f32)
         f32-in-var (if needs-f32? (gen-id "s_f32" counter) in-name)
         conv-in-eqn (when needs-f32?
                       {:op :stablehlo/convert :invars [in-name] :outvars [f32-in-var] :attrs {:target_dtype :f32}})
         scale (:scale attrs)
         actual-in-var (if scale (gen-id "s_scaled" counter) f32-in-var)
         scale-eqn (when scale
                     (let [c-scale (gen-id "c_scale" counter)
                           c-scale-eqn {:op :stablehlo/constant :value (double scale) :outvars [c-scale]}]
                       (swap! eqns-atom conj c-scale-eqn)
                       {:op :stablehlo/multiply :invars [f32-in-var c-scale] :outvars [actual-in-var]}))
         max-var (gen-id "t_smax" counter)
         max-eqn {:op :stablehlo/reduce_max :invars [actual-in-var] :outvars [max-var] :attrs {:axes norm-axes :keep_dims true}}
         max-bcast-var (gen-id "t_smax_bcast" counter)
         max-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [max-var] :outvars [max-bcast-var]
                        :attrs {:broadcast_dimensions (vec (range rank)) :target_shape in-shape}}
         diff-var (gen-id "t_sdiff" counter)
         diff-eqn {:op :stablehlo/subtract :invars [actual-in-var max-bcast-var] :outvars [diff-var]}
         exp-var (gen-id "t_sexp" counter)
         exp-eqn {:op :stablehlo/exp :invars [diff-var] :outvars [exp-var]}
         sum-var (gen-id "t_ssum" counter)
         sum-eqn {:op :stablehlo/reduce_sum :invars [exp-var] :outvars [sum-var] :attrs {:axes norm-axes :keep_dims true}}
         sum-bcast-var (gen-id "t_ssum_bcast" counter)
         sum-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [sum-var] :outvars [sum-bcast-var]
                        :attrs {:broadcast_dimensions (vec (range rank)) :target_shape in-shape}}
         f32-div-var (if needs-f32? (gen-id "t_sdiv" counter) final-out-var)
         div-eqn {:op :stablehlo/divide :invars [exp-var sum-bcast-var] :outvars [f32-div-var]}
         conv-out-eqn (when needs-f32?
                        {:op :stablehlo/convert :invars [f32-div-var] :outvars [final-out-var] :attrs {:target_dtype orig-dtype}})]
     (when needs-f32? (swap! eqns-atom conj conv-in-eqn))
     (when scale-eqn (swap! eqns-atom conj scale-eqn))
     (swap! eqns-atom conj max-eqn max-bcast-eqn diff-eqn exp-eqn sum-eqn sum-bcast-eqn div-eqn)
     (when needs-f32? (swap! eqns-atom conj conv-out-eqn)))))

(defn- lower-chunked-attention!
  "Lowers chunked attention with online streaming softmax across tiled sequence chunks.
   Prevents exceeding the 64 KB Local Data Share (LDS) hardware limit on OpenXLA ROCm.
   Mathematically identical to causal softmax attention with O(1) shared memory scaling."
  [eqns-atom counter _head q-term k-term v-term attrs final-out-var known-shapes default-dtype]
  (let [q-name (first q-term)
        k-name (first k-term)
        v-name (first v-term)
        q-shape (get known-shapes q-name [1 1 8 256])
        k-shape (get known-shapes k-name [1 128 1 256])
        max-seq-len (long (or (:max-seq-len attrs) (nth k-shape 1 128)))
        head-dim (long (or (:head-dim attrs) (nth q-shape 3 256)))
        num-heads (long (or (:num-heads attrs) (nth q-shape 2 8)))
        num-kv-heads (long (or (:num-kv-heads attrs) (nth k-shape 2 1)))
        group-size (quot num-heads num-kv-heads)
        raw-chunk-size (long (or (:chunk-size attrs) 64))
        chunk-size (cond
                     (<= max-seq-len raw-chunk-size) max-seq-len
                     (zero? (mod max-seq-len raw-chunk-size)) raw-chunk-size
                     (zero? (mod max-seq-len 32)) 32
                     (zero? (mod max-seq-len 16)) 16
                     :else max-seq-len)
        num-chunks (quot max-seq-len chunk-size)
        window (or (:sliding-window attrs) (:window-size attrs))
        pos-var (:pos attrs)
        dtype (or default-dtype :f32)
        is-f32? (= dtype :f32)

        ;; 1. Reshape K, V to [1 num_chunks chunk_size num_kv_heads head_dim]
        k-chunked (gen-id "k_chunked" counter)
        v-chunked (gen-id "v_chunked" counter)
        k-chunk-eqn {:op :stablehlo/reshape :invars [k-name] :outvars [k-chunked]
                     :attrs {:shape [1 num-chunks chunk-size num-kv-heads head-dim]}}
        v-chunk-eqn {:op :stablehlo/reshape :invars [v-name] :outvars [v-chunked]
                     :attrs {:shape [1 num-chunks chunk-size num-kv-heads head-dim]}}
        _ (swap! eqns-atom conj k-chunk-eqn v-chunk-eqn)

        ;; 2. Broadcast KV heads if group-size > 1
        k-heads (if (> group-size 1)
                  (let [k-rep (gen-id "k_rep" counter)
                        kh (gen-id "k_heads" counter)
                        rep-eqn {:op :stablehlo/broadcast_in_dim :invars [k-chunked] :outvars [k-rep]
                                 :attrs {:broadcast_dimensions [0 1 2 3 5]
                                         :target_shape [1 num-chunks chunk-size num-kv-heads group-size head-dim]}}
                        rs-eqn {:op :stablehlo/reshape :invars [k-rep] :outvars [kh]
                                :attrs {:shape [1 num-chunks chunk-size num-heads head-dim]}}]
                    (swap! eqns-atom conj rep-eqn rs-eqn)
                    kh)
                  (let [kh (gen-id "k_heads" counter)
                        rs-eqn {:op :stablehlo/reshape :invars [k-chunked] :outvars [kh]
                                :attrs {:shape [1 num-chunks chunk-size num-heads head-dim]}}]
                    (swap! eqns-atom conj rs-eqn)
                    kh))

        v-heads (if (> group-size 1)
                  (let [v-rep (gen-id "v_rep" counter)
                        vh (gen-id "v_heads" counter)
                        rep-eqn {:op :stablehlo/broadcast_in_dim :invars [v-chunked] :outvars [v-rep]
                                 :attrs {:broadcast_dimensions [0 1 2 3 5]
                                         :target_shape [1 num-chunks chunk-size num-kv-heads group-size head-dim]}}
                        rs-eqn {:op :stablehlo/reshape :invars [v-rep] :outvars [vh]
                                :attrs {:shape [1 num-chunks chunk-size num-heads head-dim]}}]
                    (swap! eqns-atom conj rep-eqn rs-eqn)
                    vh)
                  (let [vh (gen-id "v_heads" counter)
                        rs-eqn {:op :stablehlo/reshape :invars [v-chunked] :outvars [vh]
                                :attrs {:shape [1 num-chunks chunk-size num-heads head-dim]}}]
                    (swap! eqns-atom conj rs-eqn)
                    vh))

        ;; 3. Chunked Q @ K^T:
        ;; q-name: [1 1 num-heads head-dim]
        ;; k-heads: [1 num-chunks chunk-size num-heads head-dim]
        ;; Output: [1 num-heads 1 num-chunks chunk-size]
        chunk-scores (gen-id "c_scores" counter)
        dot-qk-eqn {:op :stablehlo/dot_general
                    :invars [q-name k-heads]
                    :outvars [chunk-scores]
                    :attrs {:batch_dims {:lhs [0 2] :rhs [0 3]}
                            :contracting_dims {:lhs [3] :rhs [4]}}}
        scores-5d (gen-id "scores_5d" counter)
        rs-s-eqn {:op :stablehlo/reshape :invars [chunk-scores] :outvars [scores-5d]
                  :attrs {:shape [1 num-heads num-chunks 1 chunk-size]}}
        _ (swap! eqns-atom conj dot-qk-eqn rs-s-eqn)

        ;; 4. Convert scores to f32 if not f32
        f32-scores (if is-f32?
                     scores-5d
                     (let [s-f32 (gen-id "s_f32" counter)
                           conv-eqn {:op :stablehlo/convert :invars [scores-5d] :outvars [s-f32]
                                     :attrs {:target_dtype :f32}}]
                       (swap! eqns-atom conj conv-eqn)
                       s-f32))

        ;; 5. Causal Mask Generation in 5D [1 1 num-chunks 1 chunk-size]
        iota-seq (gen-id "iota_seq" counter)
        iota-eqn {:op :stablehlo/iota :outvars [iota-seq]
                  :attrs {:len max-seq-len :dtype :i32 :iota_dimension 0}}
        iota-5d (gen-id "iota_5d" counter)
        iota-rs-eqn {:op :stablehlo/reshape :invars [iota-seq] :outvars [iota-5d]
                     :attrs {:shape [1 1 num-chunks 1 chunk-size]}}
        pos-scalar (gen-id "pos_s" counter)
        pos-s-eqn {:op :stablehlo/reshape :invars [pos-var] :outvars [pos-scalar] :attrs {:shape []}}
        pos-5d (gen-id "pos_5d" counter)
        pos-5d-eqn {:op :stablehlo/broadcast_in_dim :invars [pos-scalar] :outvars [pos-5d]
                    :attrs {:broadcast_dimensions [] :target_shape [1 1 num-chunks 1 chunk-size]}}
        cmp-fut (gen-id "cmp_fut" counter)
        cmp-fut-eqn {:op :stablehlo/compare :invars [iota-5d pos-5d] :outvars [cmp-fut]
                     :attrs {:comparison_direction "GT"}}
        _ (swap! eqns-atom conj iota-eqn iota-rs-eqn pos-s-eqn pos-5d-eqn cmp-fut-eqn)

        ring-buffer? (:ring-buffer attrs)
        cmp-mask (cond
                   ring-buffer?
                   (let [c-win (gen-id "c_win" counter)
                         c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                         cmp-full (gen-id "cmp_full" counter)
                         cmp-full-eqn {:op :stablehlo/compare :invars [pos-scalar c-win] :outvars [cmp-full]
                                       :attrs {:comparison_direction "GE"}}
                         cmp-full-5d (gen-id "cmp_full_5d" counter)
                         cmp-full-5d-eqn {:op :stablehlo/broadcast_in_dim :invars [cmp-full] :outvars [cmp-full-5d]
                                          :attrs {:broadcast_dimensions [] :target_shape [1 1 num-chunks 1 chunk-size]}}
                         not-full-5d (gen-id "not_full_5d" counter)
                         not-full-eqn {:op :stablehlo/not :invars [cmp-full-5d] :outvars [not-full-5d]}
                         mask-cond (gen-id "mask_cond" counter)
                         mask-cond-eqn {:op :stablehlo/and :invars [not-full-5d cmp-fut] :outvars [mask-cond]}]
                     (swap! eqns-atom conj c-win-eqn cmp-full-eqn cmp-full-5d-eqn not-full-eqn mask-cond-eqn)
                     mask-cond)

                   window
                   (let [c-win (gen-id "c_win" counter)
                         c-win-eqn {:op :stablehlo/constant :value (int window) :type [:tensor [] :i32] :outvars [c-win]}
                         p-sub (gen-id "p_sub_w" counter)
                         p-sub-eqn {:op :stablehlo/subtract :invars [pos-scalar c-win] :outvars [p-sub]}
                         c-one (gen-id "c_one" counter)
                         c-one-eqn {:op :stablehlo/constant :value 1 :type [:tensor [] :i32] :outvars [c-one]}
                         min-p (gen-id "min_p" counter)
                         min-p-eqn {:op :stablehlo/add :invars [p-sub c-one] :outvars [min-p]}
                         min-p-5d (gen-id "min_p_5d" counter)
                         min-p-5d-eqn {:op :stablehlo/broadcast_in_dim :invars [min-p] :outvars [min-p-5d]
                                       :attrs {:broadcast_dimensions [] :target_shape [1 1 num-chunks 1 chunk-size]}}
                         cmp-old (gen-id "cmp_old" counter)
                         cmp-old-eqn {:op :stablehlo/compare :invars [iota-5d min-p-5d] :outvars [cmp-old]
                                      :attrs {:comparison_direction "LT"}}
                         cmp-comb (gen-id "cmp_comb" counter)
                         cmp-comb-eqn {:op :stablehlo/or :invars [cmp-fut cmp-old] :outvars [cmp-comb]}]
                     (swap! eqns-atom conj c-win-eqn p-sub-eqn c-one-eqn min-p-eqn min-p-5d-eqn cmp-old-eqn cmp-comb-eqn)
                     cmp-comb)

                   :else
                   cmp-fut)

        c-neg (gen-id "c_neg" counter)
        c-neg-eqn {:op :stablehlo/constant :value -10000.0 :type [:tensor [] :f32] :outvars [c-neg]}
        c-neg-5d (gen-id "c_neg_5d" counter)
        c-neg-5d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-neg] :outvars [c-neg-5d]
                      :attrs {:broadcast_dimensions [] :target_shape [1 1 num-chunks 1 chunk-size]}}
        c-zero (gen-id "c_zero" counter)
        c-zero-eqn {:op :stablehlo/constant :value 0.0 :type [:tensor [] :f32] :outvars [c-zero]}
        c-zero-5d (gen-id "c_zero_5d" counter)
        c-zero-5d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-zero] :outvars [c-zero-5d]
                       :attrs {:broadcast_dimensions [] :target_shape [1 1 num-chunks 1 chunk-size]}}
        mask-5d (gen-id "mask_5d" counter)
        mask-sel-eqn {:op :stablehlo/select :invars [cmp-mask c-neg-5d c-zero-5d] :outvars [mask-5d]}
        mask-full (gen-id "mask_full" counter)
        mask-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [mask-5d] :outvars [mask-full]
                        :attrs {:broadcast_dimensions [0 1 2 3 4] :target_shape [1 num-heads num-chunks 1 chunk-size]}}
        masked-scores (gen-id "masked_scores" counter)
        masked-s-eqn {:op :stablehlo/add :invars [f32-scores mask-full] :outvars [masked-scores]}
        _ (swap! eqns-atom conj c-neg-eqn c-neg-5d-eqn c-zero-eqn c-zero-5d-eqn mask-sel-eqn mask-bcast-eqn masked-s-eqn)

        ;; 6. Chunk & Global Max (Online Softmax)
        m-chunk (gen-id "m_chunk" counter)
        m-chunk-eqn {:op :stablehlo/reduce_max :invars [masked-scores] :outvars [m-chunk]
                     :attrs {:axes [4] :keep_dims true}}
        m-global (gen-id "m_global" counter)
        m-global-eqn {:op :stablehlo/reduce_max :invars [m-chunk] :outvars [m-global]
                      :attrs {:axes [2] :keep_dims true}}
        m-global-bcast (gen-id "m_gbcast" counter)
        m-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [m-global] :outvars [m-global-bcast]
                     :attrs {:broadcast_dimensions [0 1 2 3 4] :target_shape [1 num-heads num-chunks 1 chunk-size]}}
        m-diff (gen-id "m_diff" counter)
        m-diff-eqn {:op :stablehlo/subtract :invars [masked-scores m-global-bcast] :outvars [m-diff]}
        exp-scores (gen-id "exp_scores" counter)
        exp-eqn {:op :stablehlo/exp :invars [m-diff] :outvars [exp-scores]}
        _ (swap! eqns-atom conj m-chunk-eqn m-global-eqn m-bcast-eqn m-diff-eqn exp-eqn)

        ;; 7. Chunk & Global Sum of Exp
        l-chunk (gen-id "l_chunk" counter)
        l-chunk-eqn {:op :stablehlo/reduce_sum :invars [exp-scores] :outvars [l-chunk]
                     :attrs {:axes [4] :keep_dims true}}
        l-global (gen-id "l_global" counter)
        l-global-eqn {:op :stablehlo/reduce_sum :invars [l-chunk] :outvars [l-global]
                      :attrs {:axes [2] :keep_dims true}}
        l-global-bcast (gen-id "l_gbcast" counter)
        l-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [l-global] :outvars [l-global-bcast]
                     :attrs {:broadcast_dimensions [0 1 2 3 4] :target_shape [1 num-heads 1 1 head-dim]}}
        _ (swap! eqns-atom conj l-chunk-eqn l-global-eqn l-bcast-eqn)

        ;; 8. Chunk Context: exp_scores @ v_heads
        exp-dt (if is-f32?
                 exp-scores
                 (let [e-dt (gen-id "exp_dt" counter)
                       c-eqn {:op :stablehlo/convert :invars [exp-scores] :outvars [e-dt] :attrs {:target_dtype dtype}}]
                   (swap! eqns-atom conj c-eqn)
                   e-dt))
        o-chunk (gen-id "o_chunk" counter)
        o-chunk-eqn {:op :stablehlo/dot_general
                     :invars [exp-dt v-heads]
                     :outvars [o-chunk]
                     :attrs {:batch_dims {:lhs [0 1 2] :rhs [0 3 1]}
                             :contracting_dims {:lhs [4] :rhs [2]}}}
        _ (swap! eqns-atom conj o-chunk-eqn)
        o-in (if is-f32?
               o-chunk
               (let [o-f32 (gen-id "o_f32" counter)
                     c-eqn {:op :stablehlo/convert :invars [o-chunk] :outvars [o-f32] :attrs {:target_dtype :f32}}]
                 (swap! eqns-atom conj c-eqn)
                 o-f32))
        o-sum (gen-id "o_sum" counter)
        o-sum-eqn {:op :stablehlo/reduce_sum :invars [o-in] :outvars [o-sum]
                   :attrs {:axes [2] :keep_dims true}}
        ctx-norm-f32 (gen-id "ctx_norm_f32" counter)
        ctx-div-eqn {:op :stablehlo/divide :invars [o-sum l-global-bcast] :outvars [ctx-norm-f32]}
        _ (swap! eqns-atom conj o-sum-eqn ctx-div-eqn)

        ;; 9. Final type conversion & reshape to [1 1 num-heads head-dim]
        ctx-final-var (if is-f32?
                        ctx-norm-f32
                        (let [ctx-dt (gen-id "ctx_dt" counter)
                              c-eqn {:op :stablehlo/convert :invars [ctx-norm-f32] :outvars [ctx-dt] :attrs {:target_dtype dtype}}]
                          (swap! eqns-atom conj c-eqn)
                          ctx-dt))
        ctx-out-eqn {:op :stablehlo/reshape :invars [ctx-final-var] :outvars [final-out-var]
                     :attrs {:shape [1 1 num-heads head-dim]}}]
    (swap! eqns-atom conj ctx-out-eqn)))

(defn- lower-rms-norm!
  ([eqns-atom counter head in-term weight-term attrs final-out-var]
   (lower-rms-norm! eqns-atom counter head in-term weight-term attrs final-out-var nil))
  ([eqns-atom counter _head in-term weight-term attrs final-out-var known-shapes]
   (let [in-name (first in-term)
         weight-name (when weight-term (first weight-term))
         gemma? (:gemma? attrs)
         eps (or (:eps attrs) 1e-5)
         shape (when known-shapes (get known-shapes in-name))
         sq-var (gen-id "rms_sq" counter)
         sq-eqn {:op :stablehlo/multiply :invars [in-name in-name] :outvars [sq-var]}
         mean-var (gen-id "rms_mean" counter)
         mean-eqn {:op :stablehlo/reduce_mean :invars [sq-var] :outvars [mean-var] :attrs {:axes [-1] :keep_dims true}}
         c-eps (gen-id "rms_eps" counter)
         c-eps-eqn {:op :stablehlo/constant :value (double eps) :outvars [c-eps]}
         mean-eps-var (gen-id "rms_mean_eps" counter)
         mean-eps-eqn {:op :stablehlo/add :invars [mean-var c-eps] :outvars [mean-eps-var]}
         rsqrt-var (gen-id "rms_rsqrt" counter)
         rsqrt-eqn {:op :stablehlo/rsqrt :invars [mean-eps-var] :outvars [rsqrt-var]}
         bcast? (boolean (and shape (pos? (count shape))))
         rsqrt-bcast-var (if bcast? (gen-id "rms_rsqrt_bcast" counter) rsqrt-var)
         bcast-eqn (when bcast?
                     {:op :stablehlo/broadcast_in_dim :invars [rsqrt-var] :outvars [rsqrt-bcast-var]
                      :attrs {:broadcast_dimensions (vec (range (count shape))) :target_shape shape}})
         xhat-var (if weight-name (gen-id "rms_xhat" counter) final-out-var)
         xhat-eqn {:op :stablehlo/multiply :invars [in-name rsqrt-bcast-var] :outvars [xhat-var]}]
     (swap! eqns-atom conj sq-eqn mean-eqn c-eps-eqn mean-eps-eqn rsqrt-eqn)
     (when bcast? (swap! eqns-atom conj bcast-eqn))
     (swap! eqns-atom conj xhat-eqn)
     (when weight-name
       (let [w-var (if gemma?
                     (let [c-one (gen-id "rms_one" counter)
                           c-one-eqn {:op :stablehlo/constant :value 1.0 :outvars [c-one]}
                           w-plus-one (gen-id "rms_w1" counter)
                           add-one-eqn {:op :stablehlo/add :invars [weight-name c-one] :outvars [w-plus-one]}]
                       (swap! eqns-atom conj c-one-eqn add-one-eqn)
                       w-plus-one)
                     weight-name)
             scaled-eqn {:op :stablehlo/multiply :invars [xhat-var w-var] :outvars [final-out-var]}]
         (swap! eqns-atom conj scaled-eqn))))))

(defn- lower-rope!
  ([eqns-atom counter head in-term attrs final-out-var known-shapes]
   (lower-rope! eqns-atom counter head in-term attrs final-out-var known-shapes :f32))
  ([eqns-atom counter _head in-term attrs final-out-var known-shapes dtype]
   (let [in-name (first in-term)
         dtype (or dtype :f32)
         shape (get known-shapes in-name [1 128 576])
         batch (nth shape 0 1)
         seq-len (nth shape 1 128)
         total-dim (nth shape 2 576)
         head-dim (long (or (:head-dim attrs) 64))
         n-heads (quot total-dim head-dim)
         rope-prop (double (or (:rope-proportion attrs) (:partial-rotary-factor attrs) 1.0))
         half-dim (quot head-dim 2)
         rope-angles (long (* rope-prop half-dim))
         theta (double (or (:theta attrs) (:theta-base attrs) 10000.0))

         dynamic? (and (= seq-len 1) (:pos attrs))
         pos-var (when dynamic? (:pos attrs))

         ;; Frequencies: full head-dim as base exponent divisor.
         ;; Unrotated channels (i >= rope-angles) receive frequency 0.0 (cos=1.0, sin=0.0 -> identity pass).
         freqs (vec (for [i (range half-dim)]
                      (if (< i rope-angles)
                        (Math/pow theta (/ (* -2.0 i) (double head-dim)))
                        0.0)))
         r4d-var (gen-id "rope_r4d" counter)
         r4d-eqn {:op :stablehlo/reshape :invars [in-name] :outvars [r4d-var] :attrs {:shape [batch seq-len n-heads head-dim]}}
         trans-var (gen-id "rope_trans" counter)
         trans-eqn {:op :stablehlo/transpose :invars [r4d-var] :outvars [trans-var] :attrs {:permutation [0 2 1 3]}}
         x1-var (gen-id "rope_x1" counter)
         x1-eqn {:op :stablehlo/slice
                 :invars [trans-var]
                 :outvars [x1-var]
                 :attrs {:start_indices [0 0 0 0]
                         :limit_indices [batch n-heads seq-len half-dim]
                         :strides [1 1 1 1]}}
         x2-var (gen-id "rope_x2" counter)
         x2-eqn {:op :stablehlo/slice
                 :invars [trans-var]
                 :outvars [x2-var]
                 :attrs {:start_indices [0 0 0 half-dim]
                         :limit_indices [batch n-heads seq-len head-dim]
                         :strides [1 1 1 1]}}
         neg-x2-var (gen-id "rope_neg_x2" counter)
         neg-x2-eqn {:op :stablehlo/negate :invars [x2-var] :outvars [neg-x2-var]}
         rot-var (gen-id "rope_rot" counter)
         rot-eqn {:op :stablehlo/concatenate :invars [neg-x2-var x1-var] :outvars [rot-var] :attrs {:dimension 3}}
         _ (swap! eqns-atom conj r4d-eqn trans-eqn x1-eqn x2-eqn neg-x2-eqn rot-eqn)

         c-freqs (gen-id "rope_freqs" counter)
         c-freqs-eqn {:op :stablehlo/constant :value freqs :type [:tensor [half-dim] :f32] :outvars [c-freqs]}
         freqs-4d (gen-id "rope_freqs_4d" counter)
         freqs-4d-eqn {:op :stablehlo/broadcast_in_dim :invars [c-freqs] :outvars [freqs-4d]
                       :attrs {:broadcast_dimensions [3] :target_shape [1 1 seq-len half-dim]}}
         _ (swap! eqns-atom conj c-freqs-eqn freqs-4d-eqn)

         pos-4d (if dynamic?
                  (let [pos-s (gen-id "rope_pos_s" counter)
                        pos-s-eqn {:op :stablehlo/reshape :invars [pos-var] :outvars [pos-s] :attrs {:shape []}}
                        pos-f32 (gen-id "rope_pos_f32" counter)
                        pos-f32-eqn {:op :stablehlo/convert :invars [pos-s] :outvars [pos-f32] :attrs {:target_dtype :f32}}
                        p4d (gen-id "rope_pos_4d" counter)
                        p4d-eqn {:op :stablehlo/broadcast_in_dim :invars [pos-f32] :outvars [p4d]
                                 :attrs {:broadcast_dimensions [] :target_shape [1 1 1 half-dim]}}]
                    (swap! eqns-atom conj pos-s-eqn pos-f32-eqn p4d-eqn)
                    p4d)
                  (let [iota-1d (gen-id "rope_iota" counter)
                        iota-eqn {:op :stablehlo/iota :outvars [iota-1d] :attrs {:len seq-len :dtype :i32 :iota_dimension 0}}
                        iota-f32 (gen-id "rope_iota_f32" counter)
                        iota-f32-eqn {:op :stablehlo/convert :invars [iota-1d] :outvars [iota-f32] :attrs {:target_dtype :f32}}
                        p4d (gen-id "rope_pos_4d" counter)
                        p4d-eqn {:op :stablehlo/broadcast_in_dim :invars [iota-f32] :outvars [p4d]
                                 :attrs {:broadcast_dimensions [2] :target_shape [1 1 seq-len half-dim]}}]
                    (swap! eqns-atom conj iota-eqn iota-f32-eqn p4d-eqn)
                    p4d))

         ang-half (gen-id "rope_ang_half" counter)
         ang-half-eqn {:op :stablehlo/multiply :invars [pos-4d freqs-4d] :outvars [ang-half]}
         ang-full (gen-id "rope_ang_full" counter)
         ang-full-eqn {:op :stablehlo/concatenate :invars [ang-half ang-half] :outvars [ang-full] :attrs {:dimension 3}}
         cos-raw (gen-id "rope_cos_raw" counter)
         cos-raw-eqn {:op :stablehlo/cosine :invars [ang-full] :outvars [cos-raw]}
         sin-raw (gen-id "rope_sin_raw" counter)
         sin-raw-eqn {:op :stablehlo/sine :invars [ang-full] :outvars [sin-raw]}
         _ (swap! eqns-atom conj ang-half-eqn ang-full-eqn cos-raw-eqn sin-raw-eqn)

         cos-dt (if (= dtype :f32)
                  cos-raw
                  (let [c-dt (gen-id "rope_cos_dt" counter)
                        c-eqn {:op :stablehlo/convert :invars [cos-raw] :outvars [c-dt] :attrs {:target_dtype dtype}}]
                    (swap! eqns-atom conj c-eqn)
                    c-dt))
         sin-dt (if (= dtype :f32)
                  sin-raw
                  (let [s-dt (gen-id "rope_sin_dt" counter)
                        s-eqn {:op :stablehlo/convert :invars [sin-raw] :outvars [s-dt] :attrs {:target_dtype dtype}}]
                    (swap! eqns-atom conj s-eqn)
                    s-dt))

         c-cos-var (gen-id "rope_cos_bc" counter)
         c-cos-eqn {:op :stablehlo/broadcast_in_dim :invars [cos-dt] :outvars [c-cos-var]
                    :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch n-heads seq-len head-dim]}}
         c-sin-var (gen-id "rope_sin_bc" counter)
         c-sin-eqn {:op :stablehlo/broadcast_in_dim :invars [sin-dt] :outvars [c-sin-var]
                    :attrs {:broadcast_dimensions [0 1 2 3] :target_shape [batch n-heads seq-len head-dim]}}
         _ (swap! eqns-atom conj c-cos-eqn c-sin-eqn)

         x-cos-var (gen-id "rope_x_cos" counter)
         x-cos-eqn {:op :stablehlo/multiply :invars [trans-var c-cos-var] :outvars [x-cos-var]}

         x-sin-var (gen-id "rope_x_sin" counter)
         x-sin-eqn {:op :stablehlo/multiply :invars [rot-var c-sin-var] :outvars [x-sin-var]}

         add-var (gen-id "rope_res" counter)
         add-eqn {:op :stablehlo/add :invars [x-cos-var x-sin-var] :outvars [add-var]}

         back-trans-var (gen-id "rope_back_trans" counter)
         back-trans-eqn {:op :stablehlo/transpose :invars [add-var] :outvars [back-trans-var] :attrs {:permutation [0 2 1 3]}}

         final-eqn {:op :stablehlo/reshape :invars [back-trans-var] :outvars [final-out-var] :attrs {:shape [batch seq-len total-dim]}}]
     (swap! eqns-atom conj x-cos-eqn x-sin-eqn add-eqn back-trans-eqn final-eqn))))

(defn- lower-fwht! [eqns-atom counter _head in-term attrs final-out-var known-shapes]
  (let [in-name (first in-term)
        shape (get known-shapes in-name)
        rank (count shape)
        raw-axis (long (or (:axis attrs) (dec rank)))
        axis (if (neg? raw-axis) (+ rank raw-axis) raw-axis)
        d (nth shape axis)
        k (long (/ (Math/log d) (Math/log 2)))
        _ (assert (= d (bit-shift-left 1 k)) (str "FWHT dimension must be a power of 2, got: " d))
        leading-dims (subvec (vec shape) 0 axis)
        batch-size (long (reduce * 1 leading-dims))
        r-in (gen-id "t_fwht_in" counter)
        r-in-eqn {:op :stablehlo/reshape
                  :invars [in-name]
                  :outvars [r-in]
                  :attrs {:shape [batch-size d]}}
        _ (swap! eqns-atom conj r-in-eqn)
        unscaled-var
        (loop [s 0
               cur-var r-in]
          (if (= s k)
            cur-var
            (let [stride (bit-shift-left 1 s)
                  chunks (long (/ d (* 2 stride)))
                  r1-var (gen-id (str "t_fwht_s" s "_r1") counter)
                  r1-eqn {:op :stablehlo/reshape
                          :invars [cur-var]
                          :outvars [r1-var]
                          :attrs {:shape [batch-size chunks 2 stride]}}
                  a-var (gen-id (str "t_fwht_s" s "_a") counter)
                  a-eqn {:op :stablehlo/slice
                         :invars [r1-var]
                         :outvars [a-var]
                         :attrs {:start_indices [0 0 0 0]
                                 :limit_indices [batch-size chunks 1 stride]
                                 :strides [1 1 1 1]}}
                  b-var (gen-id (str "t_fwht_s" s "_b") counter)
                  b-eqn {:op :stablehlo/slice
                         :invars [r1-var]
                         :outvars [b-var]
                         :attrs {:start_indices [0 0 1 0]
                                 :limit_indices [batch-size chunks 2 stride]
                                 :strides [1 1 1 1]}}
                  sum-var (gen-id (str "t_fwht_s" s "_sum") counter)
                  sum-eqn {:op :stablehlo/add :invars [a-var b-var] :outvars [sum-var]}
                  diff-var (gen-id (str "t_fwht_s" s "_diff") counter)
                  diff-eqn {:op :stablehlo/subtract :invars [a-var b-var] :outvars [diff-var]}
                  cat-var (gen-id (str "t_fwht_s" s "_cat") counter)
                  cat-eqn {:op :stablehlo/concatenate :invars [sum-var diff-var] :outvars [cat-var]
                           :attrs {:dimension 2}}
                  r2-var (gen-id (str "t_fwht_s" s "_r2") counter)
                  r2-eqn {:op :stablehlo/reshape
                          :invars [cat-var]
                          :outvars [r2-var]
                          :attrs {:shape [batch-size d]}}]
              (swap! eqns-atom conj r1-eqn a-eqn b-eqn sum-eqn diff-eqn cat-eqn r2-eqn)
              (recur (inc s) r2-var))))
        normalized? (get attrs :normalized? true)
        scaled-var (if normalized?
                     (let [scale (double (or (:scale attrs) (/ 1.0 (Math/sqrt (double d)))))
                           c-scale (gen-id "c_fwht_scale" counter)
                           c-eqn {:op :stablehlo/constant :value scale :outvars [c-scale]}
                           mul-var (gen-id "t_fwht_scaled" counter)
                           mul-eqn {:op :stablehlo/multiply :invars [unscaled-var c-scale] :outvars [mul-var]}]
                       (swap! eqns-atom conj c-eqn mul-eqn)
                       mul-var)
                     unscaled-var)
        out-reshape-eqn {:op :stablehlo/reshape
                         :invars [scaled-var]
                         :outvars [final-out-var]
                         :attrs {:shape (vec shape)}}]
    (swap! eqns-atom conj out-reshape-eqn)))

(defn- lower-rht! [eqns-atom counter head in-term signs-term attrs final-out-var known-shapes]
  (let [in-name (first in-term)
        signs-name (first signs-term)
        inverse? (:inverse? attrs)
        shape (get known-shapes in-name)]
    (if inverse?
      (let [had-var (gen-id "t_rht_had" counter)
            _ (lower-fwht! eqns-atom counter head in-term attrs had-var known-shapes)
            mul-eqn {:op :stablehlo/multiply :invars [had-var signs-name] :outvars [final-out-var]}]
        (swap! eqns-atom conj mul-eqn))
      (let [scaled-var (gen-id "t_rht_scaled" counter)
            mul-eqn {:op :stablehlo/multiply :invars [in-name signs-name] :outvars [scaled-var]}
            _ (swap! eqns-atom conj mul-eqn)]
        (lower-fwht! eqns-atom counter head [scaled-var] attrs final-out-var (assoc known-shapes scaled-var shape))))))

(defn- lower-quip-dequant! [eqns-atom counter _head codes-term cb-term scales-term _attrs final-out-var known-shapes]
  (let [codes-name (first codes-term)
        cb-name (first cb-term)
        scales-name (when scales-term (first scales-term))
        codes-shape (get known-shapes codes-name)
        cb-shape (get known-shapes cb-name)
        k (first codes-shape)
        n8 (second codes-shape)
        dim8 (if (>= (count cb-shape) 2) (second cb-shape) 8)
        n (* n8 dim8)
        expanded-shape [k n8 1]
        r-idx (gen-id "t_quip_idx_r" counter)
        r-idx-eqn {:op :stablehlo/reshape :invars [codes-name] :outvars [r-idx] :attrs {:shape expanded-shape}}
        gathered (gen-id "t_quip_gathered" counter)
        gather-eqn {:op :stablehlo/gather :invars [cb-name r-idx] :outvars [gathered]
                    :attrs {:offset_dims [2] :collapsed_slice_dims [0] :start_index_map [0]
                            :index_vector_dim 2 :slice_sizes [1 dim8]}}
        r-w (if scales-name (gen-id "t_quip_w_flat" counter) final-out-var)
        r-w-eqn {:op :stablehlo/reshape :invars [gathered] :outvars [r-w] :attrs {:shape [k n]}}]
    (swap! eqns-atom conj r-idx-eqn gather-eqn r-w-eqn)
    (when scales-name
      (let [mul-eqn {:op :stablehlo/multiply :invars [r-w scales-name] :outvars [final-out-var]}]
        (swap! eqns-atom conj mul-eqn)))))

(defn- lower-hadamard-block-128! [eqns-atom counter _head in-term signs-term attrs final-out-var known-shapes]
  (let [in-name (first in-term)
        signs-name (when signs-term (first signs-term))
        inverse? (:inverse? attrs)
        shape (get known-shapes in-name)
        rank (count shape)
        raw-axis (long (or (:axis attrs) (dec rank)))
        axis (if (neg? raw-axis) (+ rank raw-axis) raw-axis)
        d (nth shape axis)
        _ (assert (zero? (mod d 128)) (str "Block-128 Hadamard dimension must be divisible by 128, got: " d))
        num-blocks (quot d 128)
        leading-dims (subvec (vec shape) 0 axis)
        batch-size (long (reduce * 1 leading-dims))
        effective-batch (* batch-size num-blocks)
        apply-had (fn [source-var dest-var]
                    (let [r-in (gen-id "t_had128_in" counter)
                          r-in-eqn {:op :stablehlo/reshape
                                    :invars [source-var]
                                    :outvars [r-in]
                                    :attrs {:shape [effective-batch 128]}}
                          _ (swap! eqns-atom conj r-in-eqn)
                          unscaled-var
                          (loop [s 0 cur-var r-in]
                            (if (= s 7)
                              cur-var
                              (let [stride (bit-shift-left 1 s)
                                    chunks (long (/ 128 (* 2 stride)))
                                    r1-var (gen-id (str "t_had128_s" s "_r1") counter)
                                    r1-eqn {:op :stablehlo/reshape
                                            :invars [cur-var]
                                            :outvars [r1-var]
                                            :attrs {:shape [effective-batch chunks 2 stride]}}
                                    a-var (gen-id (str "t_had128_s" s "_a") counter)
                                    a-eqn {:op :stablehlo/slice
                                           :invars [r1-var]
                                           :outvars [a-var]
                                           :attrs {:start_indices [0 0 0 0]
                                                   :limit_indices [effective-batch chunks 1 stride]
                                                   :strides [1 1 1 1]}}
                                    b-var (gen-id (str "t_had128_s" s "_b") counter)
                                    b-eqn {:op :stablehlo/slice
                                           :invars [r1-var]
                                           :outvars [b-var]
                                           :attrs {:start_indices [0 0 1 0]
                                                   :limit_indices [effective-batch chunks 2 stride]
                                                   :strides [1 1 1 1]}}
                                    sum-var (gen-id (str "t_had128_s" s "_sum") counter)
                                    sum-eqn {:op :stablehlo/add :invars [a-var b-var] :outvars [sum-var]}
                                    diff-var (gen-id (str "t_had128_s" s "_diff") counter)
                                    diff-eqn {:op :stablehlo/subtract :invars [a-var b-var] :outvars [diff-var]}
                                    cat-var (gen-id (str "t_had128_s" s "_cat") counter)
                                    cat-eqn {:op :stablehlo/concatenate :invars [sum-var diff-var] :outvars [cat-var]
                                             :attrs {:dimension 2}}
                                    r2-var (gen-id (str "t_had128_s" s "_r2") counter)
                                    r2-eqn {:op :stablehlo/reshape
                                            :invars [cat-var]
                                            :outvars [r2-var]
                                            :attrs {:shape [effective-batch 128]}}]
                                (swap! eqns-atom conj r1-eqn a-eqn b-eqn sum-eqn diff-eqn cat-eqn r2-eqn)
                                (recur (inc s) r2-var))))
                          c-scale (gen-id "c_had128_scale" counter)
                          scale-val (/ 1.0 (Math/sqrt 128.0))
                          c-eqn {:op :stablehlo/constant :value scale-val :outvars [c-scale]}
                          mul-var (gen-id "t_had128_scaled" counter)
                          mul-eqn {:op :stablehlo/multiply :invars [unscaled-var c-scale] :outvars [mul-var]}
                          r-out-eqn {:op :stablehlo/reshape
                                     :invars [mul-var]
                                     :outvars [dest-var]
                                     :attrs {:shape (vec shape)}}]
                      (swap! eqns-atom conj c-eqn mul-eqn r-out-eqn)))]
    (cond
      (and signs-name inverse?)
      (let [had-out (gen-id "t_had128_unscaled" counter)
            _ (apply-had in-name had-out)
            mul-eqn {:op :stablehlo/multiply :invars [had-out signs-name] :outvars [final-out-var]}]
        (swap! eqns-atom conj mul-eqn))

      signs-name
      (let [scaled-in (gen-id "t_had128_prescaled" counter)
            mul-eqn {:op :stablehlo/multiply :invars [in-name signs-name] :outvars [scaled-in]}
            _ (swap! eqns-atom conj mul-eqn)]
        (apply-had scaled-in final-out-var))

      :else
      (apply-had in-name final-out-var))))

(defn- lower-exl3-dequant! [eqns-atom _counter _head trellis-term _cb-term _attrs final-out-var known-shapes]
  (let [trellis-name (first trellis-term)
        out-shape (get known-shapes final-out-var)
        reshape-eqn {:op :stablehlo/reshape :invars [trellis-name] :outvars [final-out-var] :attrs {:shape out-shape}}]
    (swap! eqns-atom conj reshape-eqn)))

(defn- lower-int4-unpack! [eqns-atom counter _head packed-term scale-term _attrs final-out-var known-shapes default-dtype]
  (let [packed-name (first packed-term)
        scale-name (first scale-term)
        shape (get known-shapes packed-name)
        rows (first shape)
        half-cols (second shape)
        cols (* (long half-cols) 2)
        norm-dtype (or default-dtype :bf16)
        scale-shape (get known-shapes scale-name)
        block-wise? (and scale-shape (= (count scale-shape) 2) (> (second scale-shape) 1))
        num-groups (if block-wise? (second scale-shape) 1)
        group-size (if block-wise? (quot cols num-groups) cols)

        c-0f (gen-id "c_int4_0f" counter)
        c-4  (gen-id "c_int4_4" counter)
        c-8  (gen-id "c_int4_8" counter)

        c-0f-eqn {:op :stablehlo/constant :value (int 15) :type [:tensor [] :i8] :outvars [c-0f]}
        c-4-eqn  {:op :stablehlo/constant :value (int 4) :type [:tensor [] :i8] :outvars [c-4]}
        c-8-eqn  {:op :stablehlo/constant :value (float 8.0) :type [:tensor [] norm-dtype] :outvars [c-8]}

        lo-var (gen-id "t_int4_lo" counter)
        lo-eqn {:op :stablehlo/and :invars [packed-name c-0f] :outvars [lo-var]}

        hi-var (gen-id "t_int4_hi" counter)
        hi-eqn {:op :stablehlo/shift_right_logical :invars [packed-name c-4] :outvars [hi-var]}

        lo-bf (gen-id "t_int4_lo_bf" counter)
        lo-bf-eqn {:op :stablehlo/convert :invars [lo-var] :outvars [lo-bf] :attrs {:target-dtype norm-dtype}}

        hi-bf (gen-id "t_int4_hi_bf" counter)
        hi-bf-eqn {:op :stablehlo/convert :invars [hi-var] :outvars [hi-bf] :attrs {:target-dtype norm-dtype}}

        lo-c (gen-id "t_int4_lo_c" counter)
        lo-c-eqn {:op :stablehlo/subtract :invars [lo-bf c-8] :outvars [lo-c]}

        hi-c (gen-id "t_int4_hi_c" counter)
        hi-c-eqn {:op :stablehlo/subtract :invars [hi-bf c-8] :outvars [hi-c]}

        lo-r (gen-id "t_int4_lo_r" counter)
        lo-r-eqn {:op :stablehlo/reshape :invars [lo-c] :outvars [lo-r] :attrs {:shape [rows half-cols 1]}}

        hi-r (gen-id "t_int4_hi_r" counter)
        hi-r-eqn {:op :stablehlo/reshape :invars [hi-c] :outvars [hi-r] :attrs {:shape [rows half-cols 1]}}

        cat-var (gen-id "t_int4_cat" counter)
        cat-eqn {:op :stablehlo/concatenate :invars [lo-r hi-r] :outvars [cat-var] :attrs {:dimension 2}}]

    (if block-wise?
      (let [w-3d [rows num-groups group-size]
            unscaled-var (gen-id "t_int4_unscaled" counter)
            unscaled-eqn {:op :stablehlo/reshape :invars [cat-var] :outvars [unscaled-var] :attrs {:shape w-3d}}

            scale-bcast (gen-id "t_int4_scale_bcast" counter)
            scale-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [scale-name] :outvars [scale-bcast]
                             :attrs {:broadcast_dimensions [0 1] :target_shape w-3d :shape w-3d}}

            scaled-3d-var (gen-id "t_int4_scaled_3d" counter)
            mul-eqn {:op :stablehlo/multiply :invars [unscaled-var scale-bcast] :outvars [scaled-3d-var]}

            final-reshape-eqn {:op :stablehlo/reshape :invars [scaled-3d-var] :outvars [final-out-var] :attrs {:shape [rows cols]}}]
        (swap! eqns-atom conj
               c-0f-eqn c-4-eqn c-8-eqn
               lo-eqn hi-eqn
               lo-bf-eqn hi-bf-eqn
               lo-c-eqn hi-c-eqn
               lo-r-eqn hi-r-eqn
               cat-eqn unscaled-eqn
               scale-bcast-eqn mul-eqn final-reshape-eqn))

      (let [unscaled-var (gen-id "t_int4_unscaled" counter)
            unscaled-eqn {:op :stablehlo/reshape :invars [cat-var] :outvars [unscaled-var] :attrs {:shape [rows cols]}}

            scale-bcast (gen-id "t_int4_scale_bcast" counter)
            scale-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [scale-name] :outvars [scale-bcast]
                             :attrs {:broadcast_dimensions [0] :target_shape [rows cols] :shape [rows cols]}}

            mul-eqn {:op :stablehlo/multiply :invars [unscaled-var scale-bcast] :outvars [final-out-var]}]
        (swap! eqns-atom conj
               c-0f-eqn c-4-eqn c-8-eqn
               lo-eqn hi-eqn
               lo-bf-eqn hi-bf-eqn
               lo-c-eqn hi-c-eqn
               lo-r-eqn hi-r-eqn
               cat-eqn unscaled-eqn
               scale-bcast-eqn mul-eqn)))))

(defn- lower-ternary-unpack! [eqns-atom counter _head packed-term scale-term _attrs final-out-var known-shapes default-dtype]
  (let [packed-name (first packed-term)
        scale-name (when scale-term (first scale-term))
        shape (get known-shapes packed-name)
        rank (count shape)
        _ (assert (<= 1 rank 2) (str "ternary-unpack supports 1D or 2D tensors, got shape: " shape))
        rows (if (= rank 2) (first shape) 1)
        quarter-cols (if (= rank 2) (second shape) (first shape))
        cols (* (long quarter-cols) 4)
        out-shape (if (= rank 2) [rows cols] [cols])
        slice-shape (if (= rank 2) [rows quarter-cols 1] [quarter-cols 1])
        norm-dtype (or default-dtype :bf16)
        scale-shape (when scale-name (get known-shapes scale-name))
        block-wise? (and scale-shape (= (count scale-shape) 2) (> (second scale-shape) 1))
        num-groups (if block-wise? (second scale-shape) 1)
        group-size (if block-wise? (quot cols num-groups) cols)

        c-03 (gen-id "c_ternary_03" counter)
        c-2  (gen-id "c_ternary_2" counter)
        c-4  (gen-id "c_ternary_4" counter)
        c-6  (gen-id "c_ternary_6" counter)
        c-1f (gen-id "c_ternary_1f" counter)

        c-03-eqn {:op :stablehlo/constant :value (int 3) :type [:tensor [] :i8] :outvars [c-03]}
        c-2-eqn  {:op :stablehlo/constant :value (int 2) :type [:tensor [] :i8] :outvars [c-2]}
        c-4-eqn  {:op :stablehlo/constant :value (int 4) :type [:tensor [] :i8] :outvars [c-4]}
        c-6-eqn  {:op :stablehlo/constant :value (int 6) :type [:tensor [] :i8] :outvars [c-6]}
        c-1f-eqn {:op :stablehlo/constant :value (float 1.0) :type [:tensor [] norm-dtype] :outvars [c-1f]}

        ;; Slice 0: (packed & 0x03) - 1.0
        s0-var (gen-id "t_ternary_s0" counter)
        s0-eqn {:op :stablehlo/and :invars [packed-name c-03] :outvars [s0-var]}
        s0-flt (gen-id "t_ternary_s0_flt" counter)
        s0-flt-eqn {:op :stablehlo/convert :invars [s0-var] :outvars [s0-flt] :attrs {:target-dtype norm-dtype}}
        s0-c (gen-id "t_ternary_s0_c" counter)
        s0-c-eqn {:op :stablehlo/subtract :invars [s0-flt c-1f] :outvars [s0-c]}
        s0-r (gen-id "t_ternary_s0_r" counter)
        s0-r-eqn {:op :stablehlo/reshape :invars [s0-c] :outvars [s0-r] :attrs {:shape slice-shape}}

        ;; Slice 1: ((packed >> 2) & 0x03) - 1.0
        s1-shift (gen-id "t_ternary_s1_sh" counter)
        s1-sh-eqn {:op :stablehlo/shift_right_logical :invars [packed-name c-2] :outvars [s1-shift]}
        s1-var (gen-id "t_ternary_s1" counter)
        s1-eqn {:op :stablehlo/and :invars [s1-shift c-03] :outvars [s1-var]}
        s1-flt (gen-id "t_ternary_s1_flt" counter)
        s1-flt-eqn {:op :stablehlo/convert :invars [s1-var] :outvars [s1-flt] :attrs {:target-dtype norm-dtype}}
        s1-c (gen-id "t_ternary_s1_c" counter)
        s1-c-eqn {:op :stablehlo/subtract :invars [s1-flt c-1f] :outvars [s1-c]}
        s1-r (gen-id "t_ternary_s1_r" counter)
        s1-r-eqn {:op :stablehlo/reshape :invars [s1-c] :outvars [s1-r] :attrs {:shape slice-shape}}

        ;; Slice 2: ((packed >> 4) & 0x03) - 1.0
        s2-shift (gen-id "t_ternary_s2_sh" counter)
        s2-sh-eqn {:op :stablehlo/shift_right_logical :invars [packed-name c-4] :outvars [s2-shift]}
        s2-var (gen-id "t_ternary_s2" counter)
        s2-eqn {:op :stablehlo/and :invars [s2-shift c-03] :outvars [s2-var]}
        s2-flt (gen-id "t_ternary_s2_flt" counter)
        s2-flt-eqn {:op :stablehlo/convert :invars [s2-var] :outvars [s2-flt] :attrs {:target-dtype norm-dtype}}
        s2-c (gen-id "t_ternary_s2_c" counter)
        s2-c-eqn {:op :stablehlo/subtract :invars [s2-flt c-1f] :outvars [s2-c]}
        s2-r (gen-id "t_ternary_s2_r" counter)
        s2-r-eqn {:op :stablehlo/reshape :invars [s2-c] :outvars [s2-r] :attrs {:shape slice-shape}}

        ;; Slice 3: ((packed >> 6) & 0x03) - 1.0
        s3-shift (gen-id "t_ternary_s3_sh" counter)
        s3-sh-eqn {:op :stablehlo/shift_right_logical :invars [packed-name c-6] :outvars [s3-shift]}
        s3-var (gen-id "t_ternary_s3" counter)
        s3-eqn {:op :stablehlo/and :invars [s3-shift c-03] :outvars [s3-var]}
        s3-flt (gen-id "t_ternary_s3_flt" counter)
        s3-flt-eqn {:op :stablehlo/convert :invars [s3-var] :outvars [s3-flt] :attrs {:target-dtype norm-dtype}}
        s3-c (gen-id "t_ternary_s3_c" counter)
        s3-c-eqn {:op :stablehlo/subtract :invars [s3-flt c-1f] :outvars [s3-c]}
        s3-r (gen-id "t_ternary_s3_r" counter)
        s3-r-eqn {:op :stablehlo/reshape :invars [s3-c] :outvars [s3-r] :attrs {:shape slice-shape}}

        concat-dim (dec (count slice-shape))
        cat-var (gen-id "t_ternary_cat" counter)
        cat-eqn {:op :stablehlo/concatenate :invars [s0-r s1-r s2-r s3-r] :outvars [cat-var] :attrs {:dimension concat-dim}}

        common-eqns [c-03-eqn c-2-eqn c-4-eqn c-6-eqn c-1f-eqn
                     s0-eqn s0-flt-eqn s0-c-eqn s0-r-eqn
                     s1-sh-eqn s1-eqn s1-flt-eqn s1-c-eqn s1-r-eqn
                     s2-sh-eqn s2-eqn s2-flt-eqn s2-c-eqn s2-r-eqn
                     s3-sh-eqn s3-eqn s3-flt-eqn s3-c-eqn s3-r-eqn
                     cat-eqn]]
    (cond
      (nil? scale-name)
      (let [unscaled-eqn {:op :stablehlo/reshape :invars [cat-var] :outvars [final-out-var] :attrs {:shape out-shape}}]
        (swap! eqns-atom into (conj common-eqns unscaled-eqn)))

      block-wise?
      (let [w-3d [rows num-groups group-size]
            unscaled-var (gen-id "t_ternary_unscaled" counter)
            unscaled-eqn {:op :stablehlo/reshape :invars [cat-var] :outvars [unscaled-var] :attrs {:shape w-3d}}
            scale-bcast (gen-id "t_ternary_scale_bcast" counter)
            scale-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [scale-name] :outvars [scale-bcast]
                             :attrs {:broadcast_dimensions [0 1] :target_shape w-3d :shape w-3d}}
            scaled-3d-var (gen-id "t_ternary_scaled_3d" counter)
            mul-eqn {:op :stablehlo/multiply :invars [unscaled-var scale-bcast] :outvars [scaled-3d-var]}
            final-reshape-eqn {:op :stablehlo/reshape :invars [scaled-3d-var] :outvars [final-out-var] :attrs {:shape out-shape}}]
        (swap! eqns-atom into (concat common-eqns [unscaled-eqn scale-bcast-eqn mul-eqn final-reshape-eqn])))

      :else
      (let [unscaled-var (gen-id "t_ternary_unscaled" counter)
            unscaled-eqn {:op :stablehlo/reshape :invars [cat-var] :outvars [unscaled-var] :attrs {:shape out-shape}}
            scale-is-1d-scalar? (= scale-shape [1])
            s-scalar (when scale-is-1d-scalar? (gen-id "t_ternary_scale_scalar" counter))
            s-scalar-eqn (when scale-is-1d-scalar?
                           {:op :stablehlo/reshape :invars [scale-name] :outvars [s-scalar] :attrs {:shape []}})
            eff-scale (if scale-is-1d-scalar? s-scalar scale-name)
            eff-scale-shape (if scale-is-1d-scalar? [] (or scale-shape []))
            bcast-dims (cond
                         (empty? eff-scale-shape) []
                         (and (= rank 2) (= eff-scale-shape [rows])) [0]
                         (and (= rank 2) (= eff-scale-shape [cols])) [1]
                         (and (= rank 1) (= eff-scale-shape [cols])) [0]
                         (= eff-scale-shape out-shape) nil
                         :else [0])
            extra-eqns (if bcast-dims
                         (let [scale-bcast (gen-id "t_ternary_scale_bcast" counter)
                               scale-bcast-eqn {:op :stablehlo/broadcast_in_dim :invars [eff-scale] :outvars [scale-bcast]
                                                :attrs {:broadcast_dimensions bcast-dims :target_shape out-shape :shape out-shape}}
                               mul-eqn {:op :stablehlo/multiply :invars [unscaled-var scale-bcast] :outvars [final-out-var]}]
                           (cond-> [unscaled-eqn]
                             s-scalar-eqn (conj s-scalar-eqn)
                             :always (conj scale-bcast-eqn mul-eqn)))
                         (let [mul-eqn {:op :stablehlo/multiply :invars [unscaled-var eff-scale] :outvars [final-out-var]}]
                           (cond-> [unscaled-eqn]
                             s-scalar-eqn (conj s-scalar-eqn)
                             :always (conj mul-eqn))))]
        (swap! eqns-atom into (concat common-eqns extra-eqns))))))

(defn ast->graph
  "Compiles a Tensor Logic Hiccup AST into a validated EDN SSA graph for OpenXLA compilation.
   Pipeline: expand -> prune (DCE) -> unify shapes -> lower to dot_general & StableHLO ops.
   Supports:
     (ast->graph graph-name invars ast target-heads)
     (ast->graph graph-name ast opts) where opts specifies :targets and either :invars or (:inputs + :weights)."
  ([graph-name ast opts]
   (let [targets (or (:targets opts) (:target-heads opts))
         invars (or (:invars opts)
                    (weights/infer-invars ast (or (:inputs opts) []) (or (:weights opts) {})))]
     (ast->graph graph-name invars ast targets)))
  ([graph-name invars ast target-heads]
   (let [targets (if (set? target-heads) target-heads (set target-heads))
        ;; 1. Expand containers & decompose multi-term contractions
         expanded (expand/expand-ast {} ast)
        ;; 2. Backward-chaining DCE
         pruned (dce/prune-ast expanded targets)
        ;; 3. Shape unification
         in-shapes (into {} (map (fn [[v [_ shape _]]] [v (vec shape)]) invars))
         in-dtypes (into {} (map (fn [[v [_ _ dt]]] [v dt]) invars))
         default-dtype (or (get in-dtypes :embed_tokens) (get in-dtypes :final_norm_w) :f32)
         const-dtypes (into {} (keep (fn [eqn]
                                       (when (= (first eqn) :constant)
                                         (let [h (second eqn)
                                               h-name (if (vector? h) (first h) h)
                                               attrs (ast/attrs eqn)
                                               val (get attrs :value false)
                                               t (or (:type attrs) (if (boolean? val) [:tensor [] :i1] [:tensor [] :f32]))]
                                           [h-name (last t)])))
                                     pruned))
         known-dtypes (merge in-dtypes const-dtypes)
         known-shapes (shape/unify-shapes in-shapes pruned)

         eqns-atom (atom [])
         counter (atom 0)

        ;; 4. Track occurrences of heads for inline implicit accumulation
         head-total-counts (frequencies (map ast/head pruned))
         accum-state (atom {})]

    ;; Lower equations in topological order
     (doseq [eqn pruned]
       (let [op (first eqn)
             head (ast/head eqn)
             h-name (first head)
             attrs (ast/attrs eqn)
             body (ast/body-terms eqn)
             total (get head-total-counts head 1)
             seen (get-in @accum-state [head :seen] 0)
             idx (inc seen)
             is-accum? (> total 1)
             term-var (if is-accum?
                        (gen-id (str (name h-name) "_term" idx) counter)
                        h-name)
             final-var term-var]
        ;; Lower the underlying equation to write into final-var
         (cond
           (ast/eqn? eqn)
           (cond
             (= (count body) 2)
             (lower-binary-contraction! eqns-atom counter head attrs (first body) (second body)
                                        known-shapes in-dtypes final-var)

             (= (count body) 1)
             (lower-unary-equation! eqns-atom counter head attrs (first body)
                                    known-shapes in-dtypes final-var)

             :else
             (throw (ex-info "Unsupported contraction body arity after expansion" {:equation eqn})))

           (= op :gather)
           (lower-gather! eqns-atom counter head (first body) (second body) final-var known-shapes)

           (= op :slice)
           (lower-slice! eqns-atom head (first body) attrs final-var)

           (= op :dynamic-slice)
           (lower-dynamic-slice! eqns-atom head (first body) (second body) attrs final-var)

           (= op :dynamic-update-slice)
           (lower-dynamic-update-slice! eqns-atom counter head (first body) (second body) attrs final-var)

           (= op :reshape)
           (lower-reshape! eqns-atom head (first body) attrs final-var)

           (= op :layer-norm)
           (lower-layer-norm! eqns-atom counter head (first body) (second body) (nth body 2) attrs final-var)

           (= op :causal-softmax)
           (lower-causal-softmax! eqns-atom counter (first body) attrs final-var known-shapes default-dtype)

           (= op :softmax)
           (lower-softmax! eqns-atom counter head (first body) attrs final-var known-shapes default-dtype)

           (= op :chunked-attention)
           (lower-chunked-attention! eqns-atom counter head (first body) (second body) (nth body 2) attrs final-var known-shapes default-dtype)

           (or (= op :rms-norm) (= op :gemma-rms-norm))
           (lower-rms-norm! eqns-atom counter head (first body) (second body)
                            (if (= op :gemma-rms-norm) (assoc attrs :gemma? true) attrs)
                            final-var known-shapes)

           (= op :rope)
           (lower-rope! eqns-atom counter head (first body) attrs final-var known-shapes default-dtype)

           (= op :fwht)
           (lower-fwht! eqns-atom counter head (first body) attrs final-var known-shapes)

           (= op :rht)
           (lower-rht! eqns-atom counter head (first body) (second body) attrs final-var known-shapes)

           (= op :hadamard-block-128)
           (lower-hadamard-block-128! eqns-atom counter head (first body) (second body) attrs final-var known-shapes)

           (= op :quip-dequant)
           (lower-quip-dequant! eqns-atom counter head (first body) (second body) (nth body 2 nil) attrs final-var known-shapes)

           (= op :exl3-dequant)
           (lower-exl3-dequant! eqns-atom counter head (first body) (second body) attrs final-var known-shapes)

           (= op :int4-unpack)
           (lower-int4-unpack! eqns-atom counter head (first body) (second body) attrs final-var known-shapes default-dtype)

           (or (= op :ternary-unpack)
               (= op :ternary-dequant))
           (lower-ternary-unpack! eqns-atom counter head (first body) (second body) attrs final-var known-shapes default-dtype)

           (= op :while)
           (let [out-spec (second eqn)
                 in-spec (nth eqn 2)
                 out-vars (mapv #(if (vector? %) (first %) %) (if (vector? out-spec) out-spec [out-spec]))
                 in-vars (mapv #(if (vector? %) (first %) %) (if (vector? in-spec) in-spec [in-spec]))
                 cond-node (or (get attrs :cond)
                               (get attrs :cond-node)
                               (get attrs :cond-ast)
                               (first (filter #(and (vector? %) (= (first %) :cond)) (drop 3 eqn))))
                 cond-graph (when cond-node
                              (let [cond-head (second cond-node)
                                    cond-out (if (vector? cond-head) (first cond-head) cond-head)
                                    cond-children (filter vector?
                                                          (if (map? (nth cond-node 2 nil))
                                                            (drop 3 cond-node)
                                                            (drop 2 cond-node)))
                                    cond-arg-names (or (:cond-args attrs)
                                                       (:args (ast/attrs cond-node))
                                                       in-vars)
                                    cond-invars (mapv (fn [arg-name in-var]
                                                        (let [sh (or (get in-shapes in-var)
                                                                     (get known-shapes in-var)
                                                                     [])
                                                              dt (or (get known-dtypes in-var)
                                                                     (get in-dtypes in-var)
                                                                     (when (re-find #"(?:step|pos|count|max)" (name in-var)) :i32)
                                                                     (when (re-find #"(?:stopped|bool|flag|false|true)" (name in-var)) :i1)
                                                                     :i32)]
                                                          [arg-name [:tensor sh dt]]))
                                                      cond-arg-names
                                                      in-vars)
                                    cond-ast (vec (into [:block {}] cond-children))]
                                (ast->graph "cond" cond-invars cond-ast [cond-out])))
                 body-node (or (get attrs :body)
                               (get attrs :body-node)
                               (get attrs :body-ast)
                               (first (filter #(and (vector? %) (= (first %) :body)) (drop 3 eqn))))
                 body-graph (when body-node
                              (let [body-head (second body-node)
                                    body-outs (if (vector? body-head)
                                                (mapv #(if (vector? %) (first %) %) body-head)
                                                [body-head])
                                    body-children (filter vector?
                                                          (if (map? (nth body-node 2 nil))
                                                            (drop 3 body-node)
                                                            (drop 2 body-node)))
                                    body-arg-names (or (:body-args attrs)
                                                       (:args (ast/attrs body-node))
                                                       in-vars)
                                    carry-invars (mapv (fn [arg-name in-var]
                                                         (let [sh (or (get in-shapes in-var)
                                                                      (get known-shapes in-var)
                                                                      [])
                                                               dt (or (get known-dtypes in-var)
                                                                      (get in-dtypes in-var)
                                                                      (when (re-find #"(?:step|pos|count|max)" (name in-var)) :i32)
                                                                      (when (re-find #"(?:stopped|bool|flag|false|true)" (name in-var)) :i1)
                                                                      :i32)]
                                                           [arg-name [:tensor sh dt]]))
                                                       body-arg-names
                                                       in-vars)
                                    all-body-invars (into carry-invars invars)
                                    body-ast (vec (into [:block {}] body-children))
                                    bg (ast->graph "body" all-body-invars body-ast body-outs)]
                                (assoc bg :carry-invars carry-invars)))
                 attrs (cond-> attrs
                         cond-graph (assoc :cond-graph cond-graph)
                         body-graph (assoc :body-graph body-graph))]
             (swap! eqns-atom conj {:op :stablehlo/while
                                    :invars in-vars
                                    :outvars out-vars
                                    :attrs attrs}))

           (or (= op :+) (= op :add))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/add
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (or (= op :-) (= op :subtract))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/subtract
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (or (= op :*) (= op :multiply))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/multiply
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (or (= op :/) (= op :divide))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/divide
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (or (= op :min) (= op :minimum))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/minimum
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (or (= op :max) (= op :maximum))
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/maximum
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (= op :convert)
           (let [h-name (if (vector? head) (first head) head)
                 target-dtype (or (:target-dtype attrs) (:target_dtype attrs) :f32)]
             (swap! eqns-atom conj {:op :stablehlo/convert
                                    :invars [(first (first body))]
                                    :outvars [h-name]
                                    :attrs {:target_dtype target-dtype}}))

           (= op :argmax)
           (let [h-name (if (vector? head) (first head) head)
                 axis (or (:axis attrs) (:dimension attrs) 1)]
             (swap! eqns-atom conj {:op :stablehlo/argmax
                                    :invars [(first (first body))]
                                    :outvars [h-name]
                                    :attrs {:axis axis}}))

           (= op :compare)
           (let [h-name (if (vector? head) (first head) head)
                 in0 (first (first body))
                 in1 (first (second body))
                 in0-shape (get known-shapes in0)
                 in1-shape (get known-shapes in1)
                 dir (or (:direction attrs) (:comparison_direction attrs) "LT")
                 dir-str (if (keyword? dir) (name dir) (str dir))]
             (if (and in0-shape in1-shape (not= in0-shape in1-shape))
               (let [[target-shape bcast-var other-var other-is-first?]
                     (if (>= (count in0-shape) (count in1-shape))
                       [in0-shape in1 in0 true]
                       [in1-shape in0 in1 false])
                     bcast-shape (get known-shapes bcast-var [])
                     bcast-dims (if (empty? bcast-shape)
                                  []
                                  (vec (range (- (count target-shape) (count bcast-shape))
                                              (count target-shape))))
                     out-b (gen-id "t_cmp_bcast" counter)
                     bcast-eqn {:op :stablehlo/broadcast_in_dim
                                :invars [bcast-var]
                                :outvars [out-b]
                                :attrs {:broadcast_dimensions bcast-dims :target_shape target-shape}}
                     invars (if other-is-first? [other-var out-b] [out-b other-var])]
                 (swap! eqns-atom conj bcast-eqn {:op :stablehlo/compare
                                                  :invars invars
                                                  :outvars [h-name]
                                                  :attrs {:comparison_direction dir-str}}))
               (swap! eqns-atom conj {:op :stablehlo/compare
                                      :invars [in0 in1]
                                      :outvars [h-name]
                                      :attrs {:comparison_direction dir-str}})))

           (= op :not)
           (let [h-name (if (vector? head) (first head) head)]
             (swap! eqns-atom conj {:op :stablehlo/not
                                    :invars [(first (first body))]
                                    :outvars [h-name]}))

           (or (= op :and) (= op :or))
           (let [h-name (if (vector? head) (first head) head)
                 op-kw (if (= op :and) :stablehlo/and :stablehlo/or)]
             (swap! eqns-atom conj {:op op-kw
                                    :invars [(first (first body)) (first (second body))]
                                    :outvars [h-name]}))

           (= op :cond)
           (let [cond-children (filter vector?
                                       (if (map? (nth eqn 2 nil))
                                         (drop 3 eqn)
                                         (drop 2 eqn)))]
             (doseq [child cond-children]
               (let [c-op (first child)
                     c-head (second child)
                     c-body (ast/body-terms child)
                     c-attrs (ast/attrs child)]
                 (case c-op
                   :compare
                   (let [h (if (vector? c-head) (first c-head) c-head)
                         dir (or (:direction c-attrs) (:comparison_direction c-attrs) "LT")
                         dir-str (if (keyword? dir) (name dir) (str dir))]
                     (swap! eqns-atom conj {:op :stablehlo/compare
                                            :invars [(first (first c-body)) (first (second c-body))]
                                            :outvars [h]
                                            :attrs {:comparison_direction dir-str}}))
                   :not
                   (let [h (if (vector? c-head) (first c-head) c-head)]
                     (swap! eqns-atom conj {:op :stablehlo/not
                                            :invars [(first (first c-body))]
                                            :outvars [h]}))
                   :and
                   (let [h (if (vector? c-head) (first c-head) c-head)]
                     (swap! eqns-atom conj {:op :stablehlo/and
                                            :invars [(first (first c-body)) (first (second c-body))]
                                            :outvars [h]}))
                   :or
                   (let [h (if (vector? c-head) (first c-head) c-head)]
                     (swap! eqns-atom conj {:op :stablehlo/or
                                            :invars [(first (first c-body)) (first (second c-body))]
                                            :outvars [h]}))
                   nil))))

           (= op :constant)
           (let [h (second eqn)
                 h-name (if (vector? h) (first h) h)
                 val (get attrs :value false)
                 val-t (or (:type attrs) (if (boolean? val) [:tensor [] :i1] [:tensor [] :f32]))]
             (swap! eqns-atom conj {:op :stablehlo/constant
                                    :value val
                                    :type val-t
                                    :outvars [h-name]}))

           :else
           (throw (ex-info "Unknown AST equation or lowering hook" {:equation eqn :op op})))

        ;; If multi-term accumulation, emit inline addition immediately when idx > 1
         (when is-accum?
           (if (= idx 1)
             (swap! accum-state assoc head {:seen 1 :current-var term-var})
             (let [prev-var (get-in @accum-state [head :current-var])
                   is-last? (= idx total)
                   out-var (if is-last? h-name (gen-id (str (name h-name) "_acc" idx) counter))
                   add-eqn {:op :stablehlo/add :invars [prev-var term-var] :outvars [out-var]}]
               (swap! eqns-atom conj add-eqn)
               (swap! accum-state assoc head {:seen idx :current-var out-var}))))))

     (let [outvars (if (sequential? target-heads) (vec target-heads) (vec targets))
           graph {:name graph-name
                  :invars invars
                  :outvars outvars
                  :known-shapes known-shapes
                  :eqns @eqns-atom}]
       (shlo/validate-graph graph)))))
