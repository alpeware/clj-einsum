(ns einsum.core-test
  "Unit and generative property tests for thin Declarative UOp entry point in einsum.core."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [einsum.core :as core]
            [einsum.uop.lower :as uop-lower]))

(deftest test-parse-spec
  (testing "parse-spec correctly parses explicit einsum subscript specifications"
    (let [p1 (core/parse-spec "ij,jk->ik")]
      (is (= [["i" "j"] ["j" "k"]] (:inputs p1)))
      (is (= ["i" "k"] (:output p1))))

    (let [p2 (core/parse-spec "i,j->ij")]
      (is (= [["i"] ["j"]] (:inputs p2)))
      (is (= ["i" "j"] (:output p2))))

    (let [p3 (core/parse-spec "ij->ji")]
      (is (= [["i" "j"]] (:inputs p3)))
      (is (= ["j" "i"] (:output p3))))

    (let [p4 (core/parse-spec "i,i->")]
      (is (= [["i"] ["i"]] (:inputs p4)))
      (is (= [] (:output p4)))))

  (testing "parse-spec derives implicit output from unique indices"
    (let [p (core/parse-spec "ij,jk")]
      (is (= [["i" "j"] ["j" "k"]] (:inputs p)))
      (is (= ["i" "k"] (:output p))))))

(defspec prop-parse-spec-structure
  50
  (prop/for-all [idx-a (gen/elements ["i" "a" "m"])
                 idx-b (gen/elements ["j" "b" "k"])
                 idx-c (gen/elements ["k" "c" "n"])]
                (let [spec-str (str idx-a idx-b "," idx-b idx-c "->" idx-a idx-c)
                      parsed (core/parse-spec spec-str)]
                  (and (map? parsed)
                       (= [[idx-a idx-b] [idx-b idx-c]] (:inputs parsed))
                       (= [idx-a idx-c] (:output parsed))))))

(deftest test-plan
  (testing "plan infers index dimensions and output shape from operand shapes"
    (let [pl1 (core/plan "ij,jk->ik" [[2 3] [3 4]])]
      (is (= [2 4] (:out-shape pl1)))
      (is (= {"i" 2 "j" 3 "k" 4} (:idx->dim pl1))))

    (let [pl2 (core/plan "i,j->ij" [[5] [7]])]
      (is (= [5 7] (:out-shape pl2)))
      (is (= {"i" 5 "j" 7} (:idx->dim pl2))))))

(deftest test-init-backend-and-context
  (testing "init-backend! with :interpreter initializes reference backend"
    (let [ctx (core/init-backend! :interpreter {:quiet? true})]
      (is (= :interpreter (:backend ctx)))
      (is (= :interpreter (:target ctx)))
      (is (= ctx (core/get-context)))))

  (testing "init-backend! with :uop-cpu initializes UOp CPU backend"
    (let [ctx (core/init-backend! :uop-cpu {:quiet? true})]
      (is (= :uop (:backend ctx)))
      (is (= :uop-cpu (:target ctx)))
      (is (= ctx (core/get-context))))))

(deftest test-with-device-arena
  (testing "with-device-arena manages scoped allocations cleanly"
    (let [ctx (core/init-backend! :uop-cpu {:quiet? true})]
      (core/with-device-arena [arena ctx]
        (let [arr (float-array [1.0 2.0 3.0])
              buf (core/device-buffer arena arr [3] :f32)]
          (is (some? buf))
          (is (some? (core/track! arena buf))))))))

(deftest test-compile-graph-and-execute-uop-cpu
  (testing "compile-graph and execute on UOp CPU backend"
    (let [ctx (core/init-backend! :uop-cpu {:quiet? true})
          ast [[:= [:z :d] [:x :d] [:y :d]]]
          invars [[:x [:tensor [2] :f32]]
                  [:y [:tensor [2] :f32]]]
          graph (uop-lower/ast->uop-graph "mul_core_test" invars ast [:z])
          exec (core/compile-graph ctx graph)
          x-arr (float-array [1.5 2.5])
          y-arr (float-array [3.5 4.5])
          res (core/execute exec [x-arr y-arr])]
      (is (some? res))
      (let [slice (core/to-host-slice res 0 2 2 :f32)]
        (is (= [5.25 11.25] (vec slice)))))))

(deftest test-compile-graph-and-execute-interpreter
  (testing "compile-graph and execute on interpreter backend"
    (let [ctx (core/init-backend! :interpreter {:quiet? true})
          graph {:name "mul_interp_test"
                 :invars [[:x [:tensor [2] :f32]]
                          [:y [:tensor [2] :f32]]]
                 :outvars [:z]
                 :known-shapes {}
                 :eqns [{:op :ir/multiply :invars [:x :y] :outvars [:z]}]}
          exec (core/compile-graph ctx graph)
          x-arr (float-array [10.0 20.0])
          y-arr (float-array [30.0 40.0])
          res (core/execute exec [x-arr y-arr])]
      (is (map? res))
      (let [slice (core/to-host-slice res 0 2 2 :f32)]
        (is (= [300.0 800.0] (vec slice)))))))
