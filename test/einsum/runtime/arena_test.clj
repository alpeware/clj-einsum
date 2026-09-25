(ns einsum.runtime.arena-test
  "Generative property tests and unit tests for Scoped Device Arenas (einsum.runtime.arena)."
  (:require [einsum.runtime.arena :as arena]
            [clojure.test :refer [deftest is testing]]
            [clojure.test.check.clojure-test :refer [defspec]]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]))

;; Mock buffer record to verify destroy calls without requiring live GPU
(defrecord MockDeviceBuffer [id destroyed?])

(defn mock-buffer [id]
  (->MockDeviceBuffer id (atom false)))

(defn mock-destroy! [_ctx buf]
  (reset! (:destroyed? buf) true))

(deftest test-arena-creation-and-tracking
  (testing "create-arena initializes an open arena with empty tracking set"
    (let [ctx {:mock-destroy mock-destroy!}
          a (arena/create-arena ctx)]
      (is (arena/arena? a))
      (is (false? (arena/closed? a)))
      (is (empty? (arena/tracked-buffers a)))
      (let [b1 (mock-buffer 1)
            b2 (mock-buffer 2)]
        (arena/track! a b1)
        (arena/track! a b2)
        (is (= #{b1 b2} (arena/tracked-buffers a)))
        (arena/disown! a b1)
        (is (= #{b2} (arena/tracked-buffers a)))
        (arena/close! a)
        (is (true? (arena/closed? a)))
        (is (empty? (arena/tracked-buffers a)))))))

(defspec prop-arena-lifecycle-tracking
  50
  (prop/for-all [ids (gen/vector gen/nat 1 20)]
                (let [ctx {:destroy-fn (fn [_ b] (reset! (:destroyed? b) true))}
                      a (arena/create-arena ctx)
                      buffers (mapv mock-buffer ids)]
      ;; Track all
                  (doseq [b buffers]
                    (arena/track! a b))
                  (and (= (set buffers) (arena/tracked-buffers a))
                       (false? (arena/closed? a))
           ;; Close arena
                       (do (arena/close! a)
                           (and (true? (arena/closed? a))
                                (empty? (arena/tracked-buffers a))))))))

(defspec prop-promote-transfers-ownership
  50
  (prop/for-all [ids (gen/vector gen/nat 2 20)]
                (let [split-point (quot (count ids) 2)
                      parent (arena/create-arena {})
                      child (arena/create-arena {} parent)
                      buffers (mapv mock-buffer ids)
                      keep-in-child (subvec buffers 0 split-point)
                      promote-to-parent (subvec buffers split-point)]
                  (doseq [b buffers]
                    (arena/track! child b))
                  (arena/promote! child parent promote-to-parent)
                  (and (= (set keep-in-child) (arena/tracked-buffers child))
                       (= (set promote-to-parent) (arena/tracked-buffers parent))
                       (do
                         (arena/close! child)
                         (and (true? (arena/closed? child))
                              (= (set promote-to-parent) (arena/tracked-buffers parent))
                              (false? (arena/closed? parent))))))))

(deftest test-with-device-arena-lexical-scoping
  (testing "with-device-arena closes arena on normal completion"
    (let [destroyed-ids (atom #{})
          ctx {:destroy-fn (fn [_ b] (swap! destroyed-ids conj (:id b)))}
          captured-a (atom nil)
          b1 (mock-buffer 101)
          b2 (mock-buffer 102)]
      (arena/with-device-arena [a ctx]
        (reset! captured-a a)
        (arena/track! a b1)
        (arena/track! a b2)
        (is (= #{b1 b2} (arena/tracked-buffers a))))
      (is (true? (arena/closed? @captured-a)))
      (is (= #{101 102} @destroyed-ids))))

  (testing "with-device-arena closes arena on exception thrown"
    (let [destroyed-ids (atom #{})
          ctx {:destroy-fn (fn [_ b] (swap! destroyed-ids conj (:id b)))}
          b1 (mock-buffer 201)
          captured-arena (atom nil)]
      (try
        (arena/with-device-arena [a ctx]
          (reset! captured-arena a)
          (arena/track! a b1)
          (throw (ex-info "Simulated failure" {})))
        (catch Exception _ nil))
      (is (true? (arena/closed? @captured-arena)))
      (is (= #{201} @destroyed-ids))))

  (testing "with-device-arena supports promote! preserving long-lived buffers"
    (let [destroyed-ids (atom #{})
          ctx {:destroy-fn (fn [_ b] (swap! destroyed-ids conj (:id b)))}
          session-arena (arena/create-arena ctx)
          b-transient (mock-buffer 301)
          b-persistent (mock-buffer 302)]
      (arena/with-device-arena [step-arena session-arena]
        (arena/track! step-arena b-transient)
        (arena/track! step-arena b-persistent)
        (arena/promote! step-arena session-arena b-persistent))
      ;; Step arena is closed; b-transient destroyed, b-persistent alive in session-arena
      (is (= #{301} @destroyed-ids))
      (is (= #{b-persistent} (arena/tracked-buffers session-arena)))
      ;; Closing session-arena destroys b-persistent
      (arena/close! session-arena)
      (is (= #{301 302} @destroyed-ids)))))

(deftest test-track-closed-arena-throws
  (testing "Calling track! on a closed arena throws ExceptionInfo"
    (let [a (arena/create-arena {})
          b (mock-buffer 401)]
      (arena/close! a)
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Cannot track buffer in closed DeviceArena"
                            (arena/track! a b))))))

(defspec prop-track-closed-arena-throws
  30
  (prop/for-all [id gen/nat]
                (let [a (arena/create-arena {})
                      b (mock-buffer id)]
                  (arena/close! a)
                  (try
                    (arena/track! a b)
                    false
                    (catch clojure.lang.ExceptionInfo _
                      true)))))
