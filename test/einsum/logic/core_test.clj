(ns einsum.logic.core-test
  "Generative property and unit tests for unified Tensor Logic core compilation API."
  (:require [einsum.core :as xla]
            [einsum.logic.core :as logic]
            [clojure.test :refer [deftest is]]))

(deftest test-compile-ast-end-to-end
  (let [invars [[:x [:tensor [2 4] :f32]]
                [:w [:tensor [4 2] :f32]]]
        ast [:= [:y :m :n] [:x :m :k] [:w :k :n]]
        compiled (logic/compile-ast "core_e2e_gemm" invars ast #{:y})
        x-data (float-array [1.0 2.0 3.0 4.0
                             5.0 6.0 7.0 8.0])
        w-data (float-array [1.0 0.0
                             0.0 1.0
                             1.0 0.0
                             0.0 1.0])
        out-buf (xla/execute compiled x-data w-data)
        res (xla/to-host-slice out-buf 0 4 4)]
    (is (= 4.0 (nth res 0)))
    (is (= 6.0 (nth res 1)))
    (is (= 12.0 (nth res 2)))
    (is (= 14.0 (nth res 3)))
    (xla/destroy-buffer! out-buf)))
