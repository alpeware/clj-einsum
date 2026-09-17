(ns clj-xla.logic.data.family-trees
  "Synthetic family tree generator for gradient-based predicate invention (Experiment E16/E17).
   Constructs a forest of discrete family trees with known ground-truth:
   - Parent: hidden relation (ground truth to be recovered by latent core)
   - Grandparent: observed target relation (two-hop composition of Parent)
   - Sibling: observed auxiliary relation (sharing common Parent)
   Pure Clojure (Sans-IO) conforming to Rule 2.")

(defn- seeded-random [seed]
  (java.util.Random. (long (or seed 42))))

(defn- seeded-shuffle [coll ^java.util.Random rnd]
  (let [al (java.util.ArrayList. ^java.util.Collection coll)]
    (java.util.Collections/shuffle al rnd)
    (vec al)))

(defn- generate-tree-nodes
  "Generates parent, grandparent, and sibling edges for a single family tree of size `tree-size`
   starting at entity id offset `offset`."
  [offset tree-size ^java.util.Random rnd]
  ;; Each tree needs at least 8 entities for 3 generations (founders, child+spouse, grandchildren)
  (let [t-size (max 8 (long tree-size))
        shuffled-entities (seeded-shuffle (range offset (+ offset t-size)) rnd)
        ;; Gen 0: 2 founders
        f1 (nth shuffled-entities 0)
        f2 (nth shuffled-entities 1)
        ;; Determine generation sizes
        remaining (- t-size 2)
        gen1-children-count (max 1 (min 3 (quot remaining 3)))
        gen1-spouse-count 1
        gen1-size (+ gen1-children-count gen1-spouse-count)
        _gen2-size (- remaining gen1-size)

        gen1-children (vec (take gen1-children-count (drop 2 shuffled-entities)))
        gen1-spouse (nth shuffled-entities (+ 2 gen1-children-count))
        gen2-children (vec (drop (+ 2 gen1-size) shuffled-entities))

        ;; Parent edges
        ;; Founders are parents to all Gen 1 children
        p-founders (for [f [f1 f2]
                         c gen1-children]
                     [f c])
        ;; First Gen 1 child mates with Gen 1 spouse to produce Gen 2 children
        mating-child (first gen1-children)
        p-gen1 (for [p [mating-child gen1-spouse]
                     c gen2-children]
                 [p c])
        parent-pairs (vec (concat p-founders p-gen1))

        ;; Grandparent edges:
        ;; gp is grandparent of gc iff exists m such that [gp m] in parent and [m gc] in parent.
        parent-set (set parent-pairs)
        all-entities shuffled-entities
        grandparent-pairs
        (vec (distinct (for [gp all-entities
                             gc all-entities
                             :when (and (not= gp gc)
                                        (some (fn [m]
                                                (and (contains? parent-set [gp m])
                                                     (contains? parent-set [m gc])))
                                              all-entities))]
                         [gp gc])))

        ;; Sibling edges:
        ;; s1, s2 are siblings iff s1 != s2 and exists p such that [p s1] in parent and [p s2] in parent.
        sibling-pairs
        (vec (distinct (for [s1 all-entities
                             s2 all-entities
                             :when (and (not= s1 s2)
                                        (some (fn [p]
                                                (and (contains? parent-set [p s1])
                                                     (contains? parent-set [p s2])))
                                              all-entities))]
                         [s1 s2])))]
    {:parent-pairs parent-pairs
     :grandparent-pairs grandparent-pairs
     :sibling-pairs sibling-pairs
     :mating-child mating-child
     :other-child (first (filter #(not= % mating-child) gen1-children))
     :founders [f1 f2]
     :spouse gen1-spouse
     :gen1-children gen1-children
     :gen2-children gen2-children}))

(defn- pairs->matrix
  "Converts a collection of [h t] index pairs into an N x N float matrix (nested vector)."
  [pairs n]
  (let [pair-set (set pairs)]
    (mapv (fn [i]
            (mapv (fn [j]
                    (if (contains? pair-set [i j]) 1.0 0.0))
                  (range n)))
          (range n))))

(defn split-pairs
  "Splits relation pairs into disjoint train and held-out sets according to `train-ratio`
   guaranteeing the transductive property: every entity appearing in held-out also
   appears at least once in train."
  [pairs train-ratio ^java.util.Random rnd]
  (let [shuffled (seeded-shuffle pairs rnd)
        total (count shuffled)
        target-train-count (min total (max 1 (long (Math/round (* (double train-ratio) (double total))))))
        entity-cover (loop [remaining shuffled
                            covered #{}
                            must-train []
                            leftover []]
                       (if (empty? remaining)
                         {:must-train must-train :leftover leftover}
                         (let [[h t :as p] (first remaining)]
                           (if (or (not (contains? covered h))
                                   (not (contains? covered t)))
                             (recur (rest remaining)
                                    (conj covered h t)
                                    (conj must-train p)
                                    leftover)
                             (recur (rest remaining)
                                    covered
                                    must-train
                                    (conj leftover p))))))
        must-train (:must-train entity-cover)
        leftover (:leftover entity-cover)
        train-entities (set (mapcat identity must-train))
        held-out-budget (- total target-train-count)
        split-res (loop [cands leftover
                         train-acc must-train
                         held-acc []]
                    (if (empty? cands)
                      {:train train-acc :held-out held-acc}
                      (let [[h t :as p] (first cands)]
                        (if (and (< (count held-acc) held-out-budget)
                                 (contains? train-entities h)
                                 (contains? train-entities t))
                          (recur (rest cands) train-acc (conj held-acc p))
                          (recur (rest cands) (conj train-acc p) held-acc)))))]
    {:train (:train split-res)
     :held-out (:held-out split-res)}))

(defn generate-family-forest
  "Generates a synthetic forest of family trees.
   Options:
   - :num-entities (default 64)
   - :num-trees (default 8)
   - :seed (default 42)
   - :train-ratio (default 0.7)

   Returns a map with:
   - :num-entities
   - :num-trees
   - :parent-pairs (hidden ground truth)
   - :grandparent-pairs (observed target)
   - :sibling-pairs (observed auxiliary)
   - :parent-matrix [N x N]
   - :grandparent-matrix [N x N]
   - :sibling-matrix [N x N]
   - :splits {:grandparent {:train ... :held-out ...}
              :sibling     {:train ... :held-out ...}}"
  ([] (generate-family-forest nil))
  ([opts]
   (let [n (long (or (:num-entities opts) 64))
         requested-trees (long (or (:num-trees opts) 8))
         ;; Each tree requires at least 8 entities
         num-trees (max 1 (min requested-trees (quot n 8)))
         seed (long (or (:seed opts) 42))
         train-ratio (double (or (:train-ratio opts) 0.7))
         rnd (seeded-random seed)

         tree-size (quot n num-trees)
         trees-data
         (mapv (fn [t-idx]
                 (let [offset (* t-idx tree-size)]
                   (generate-tree-nodes offset tree-size rnd)))
               (range num-trees))

         parent-pairs (vec (distinct (mapcat :parent-pairs trees-data)))
         gp-pairs (vec (distinct (mapcat :grandparent-pairs trees-data)))
         sib-pairs (vec (distinct (mapcat :sibling-pairs trees-data)))

         gp-splits (split-pairs gp-pairs train-ratio rnd)
         sib-splits (split-pairs sib-pairs train-ratio rnd)

         p-mat (pairs->matrix parent-pairs n)
         gp-mat (pairs->matrix gp-pairs n)
         sib-mat (pairs->matrix sib-pairs n)]

     {:num-entities n
      :num-trees num-trees
      :seed seed
      :trees trees-data
      :parent-pairs parent-pairs
      :grandparent-pairs gp-pairs
      :sibling-pairs sib-pairs
      :parent-matrix p-mat
      :grandparent-matrix gp-mat
      :sibling-matrix sib-mat
      :splits {:grandparent gp-splits
               :sibling sib-splits}})))
