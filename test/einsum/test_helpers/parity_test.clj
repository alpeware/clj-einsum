(ns einsum.test-helpers.parity-test
  "Generative property tests for numerical tolerance, CPU vs Device execution parity, and finite-difference gradient checking."
  (:require [einsum.test-helpers.parity :as p]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

(defspec prop-approx-equal-reflexive
  100
  (prop/for-all [data (gen/vector (gen/double* {:NaN? false :infinite? false}) 1 50)]
                (p/approx-equal? data data {:atol 1e-5 :rtol 1e-4})))

(defspec prop-approx-equal-tolerance-bounds
  100
  (prop/for-all [data (gen/vector (gen/double* {:NaN? false :infinite? false :min -1e4 :max 1e4}) 1 50)
                 delta (gen/double* {:NaN? false :infinite? false :min -1e-4 :max 1e-4})]
                (let [perturbed (mapv #(+ % delta) data)]
                  (p/approx-equal? data perturbed {:atol 1e-3 :rtol 1e-3}))))

(deftest test-finite-difference-scalar-op
  (testing "Finite difference gradient matches analytical derivative of quadratic function f(x) = x^2"
    (let [f (fn [x] (* x x))
          grad-f (fn [x] (* 2.0 x))
          x 3.0
          eps 1e-4
          numerical-grad (/ (- (f (+ x eps)) (f (- x eps))) (* 2.0 eps))
          analytical-grad (grad-f x)]
      (is (p/approx-equal? [analytical-grad] [numerical-grad] {:atol 1e-3 :rtol 1e-3})))))
