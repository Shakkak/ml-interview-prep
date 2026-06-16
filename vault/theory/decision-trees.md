---
title: Decision Trees
tags: [decision-trees, classification, regression, non-parametric, splitting-criteria]
aliases: [decision tree, CART, Gini impurity, information gain, tree pruning, regression tree]
difficulty: 1
status: complete
depends_on: [entropy-mutual-info, statistical-inference-mle]
related: [ensemble-methods, entropy-mutual-info, bias-variance-double-descent, parametric-nonparametric, logistic-regression]
---

# Decision Trees

---

## Fundamental

**What problem does this solve?** We want a classifier that (1) is interpretable — a human can read the decision rules and understand why a prediction was made, (2) handles mixed feature types (continuous, categorical, ordinal) without preprocessing, (3) requires no feature scaling, and (4) can capture nonlinear patterns without feature engineering.

**The core question:** given a dataset with labels, how should we partition the feature space to best separate the classes? The decision tree answers this greedily: at each step, find the single binary question about one feature that best separates the classes in the current subset of data. Repeat until each region is pure (one class only) or stopping criteria are met.

**Definition.** A decision tree recursively partitions the feature space with axis-aligned
splits ("is feature $x_j \leq t$?"), forming a binary tree. Each internal node tests one
feature against a threshold; each leaf stores a prediction — the majority class
(classification) or the mean target value (regression) of the training examples that land
there.

**How a split is chosen — impurity measures.** At each node, the tree searches over every
feature and every candidate threshold, and picks the split that most reduces "impurity" —
how mixed the class labels are in a node. Two common measures for classification, both built
from the class proportions $p_k$ in a node:

$$\text{Gini}(S) = 1 - \sum_{k} p_k^2 \qquad\qquad H(S) = -\sum_k p_k \log_2 p_k \;\;\text{(entropy, see [[entropy-mutual-info]])}$$

where $S$ = the set of training examples at a node, $k$ = class index (e.g., 0 or 1 for binary), $p_k$ = fraction of examples in $S$ belonging to class $k$ (so $\sum_k p_k = 1$), and $p_k^2$ = squared probability (subtracting from 1 gives probability of misclassification if we randomly label with the class distribution).

> [!tip] Gini impurity vs. entropy — same shape, different cost
> Both are 0 for a pure node (one class only) and maximal for a perfectly mixed node. For two
> classes with proportions $(p, 1-p)$: Gini $= 2p(1-p)$ peaks at $0.5$ when $p=0.5$; entropy
> peaks at $1$ bit at the same point (see [[entropy-mutual-info]] for the full derivation of
> $H(p)$). Gini is cheaper (no logarithm) and is XGBoost/scikit-learn's default; entropy is
> more sensitive to class-probability changes near purity. In practice they almost always pick
> the same splits — the choice rarely matters for accuracy.

**Information gain** is the impurity reduction a split produces:
$$\text{IG} = \text{Impurity(parent)} - \sum_{\text{child } c} \frac{|c|}{|S|}\,\text{Impurity}(c)$$

where $\text{Impurity(parent)}$ = Gini or entropy of the node before splitting, $|c|$ = number of examples in child $c$, $|S|$ = total examples at this node, and $\frac{|c|}{|S|}$ = the weight for child $c$ (larger children contribute more to the average). Higher IG = better split.
— the weighted-average impurity of the children, subtracted from the parent's impurity. The
tree picks the (feature, threshold) pair that maximizes this at every node. This is exactly the
**mutual information** $I(X;Y)$ between a feature and the label, viewed locally at a node — see
[[entropy-mutual-info]] for the general definition and why high-MI features are informative for
splitting.

**Numerical example.** Node $S$ has 10 examples: 6 positive, 4 negative → $p_+=0.6$, Gini$(S)
= 1 - (0.6^2+0.4^2) = 0.48$. A candidate split sends 5 examples (4 positive, 1 negative) left
and 5 (2 positive, 3 negative) right:
- Left: Gini $= 1-(0.8^2+0.2^2)=0.32$;  Right: Gini $= 1-(0.4^2+0.6^2)=0.48$
- Weighted child impurity $= 0.5(0.32) + 0.5(0.48) = 0.40$
- Information gain $= 0.48 - 0.40 = 0.08$

The tree compares this gain against every other candidate split at this node and takes the
largest.

**Regression trees** replace impurity with variance (equivalently, MSE): a split is scored by
how much it reduces $\sum_i (y_i - \bar{y})^2$ within the resulting children. The leaf
prediction is the mean of the targets that reach it — so a regression tree's output is a
**piecewise-constant** function of the input.

**The CART algorithm** (Breiman et al., 1984; the basis of scikit-learn's `DecisionTree*`):
greedily and recursively (1) try every feature and threshold, (2) pick the split maximizing
impurity reduction, (3) recurse on each child, (4) stop when a node is pure, too small, or a
maximum depth is reached. It is greedy and **does not guarantee a globally optimal tree** —
finding the optimal tree is NP-hard, so every practical algorithm is a greedy heuristic.

---

## Intermediate

**Overfitting — the central weakness.** An unconstrained tree keeps splitting until every leaf
is pure, which means it can perfectly memorize the training set: zero bias, but very high
variance (a small change in the training data produces a very different tree — see
[[bias-variance-double-descent]]). This instability is the single most important fact about
trees — it is *why* ensembling them works so well (see Advanced).

**Pre-pruning (early stopping)** — constrain growth via hyperparameters:

| Hyperparameter | Effect |
|---|---|
| `max_depth` | Hard cap on tree depth — most direct variance control |
| `min_samples_leaf` / `min_samples_split` | Refuse splits that would create tiny leaves (noisy estimates) |
| `min_impurity_decrease` | Refuse splits whose information gain is below a threshold |
| `max_features` | Consider only a random subset of features per split (this *is* the Random Forest mechanism — see [[ensemble-methods]]) |

**Post-pruning (cost-complexity / weakest-link pruning)** — grow the full tree, then remove
subtrees that don't justify their complexity. Minimize:
$$R_\alpha(T) = R(T) + \alpha\,|T|$$

where $R(T)$ = tree's total training error (e.g., misclassification rate or MSE across all leaves), $|T|$ = number of leaf nodes (complexity of the tree), and $\alpha \geq 0$ = complexity parameter (larger $\alpha$ = more pruning, fewer leaves).
controls the complexity penalty (chosen by [[cross-validation]] — directly analogous to the
regularization strength in [[regularization-weight-decay|weight decay]]). As $\alpha$
increases, more subtrees collapse into single leaves. Post-pruning generally generalizes better
than pre-pruning because it sees the *fully-grown* tree's structure before deciding what to cut.

**Feature importance** — for each feature, sum the impurity decrease (weighted by the fraction
of samples reaching that node) over every split that uses it. This gives a fast, built-in
ranking of which features the tree relies on most — useful for interpretability, but biased
toward high-cardinality features (more candidate thresholds → more chances to find a
spuriously good split).

**Handling messy data.** Trees natively handle:
- **Categorical features** — split on subsets of categories (or, more commonly in practice,
  one-hot encode and let the tree pick individual categories one at a time).
- **Missing values** — *surrogate splits*: at each node, the tree also learns backup splits on
  other correlated features, used when the primary feature is missing for an example.
- **Mixed feature scales** — splits only compare a feature to *its own* thresholds, so trees
  are **invariant to monotonic transformations** of individual features (no need to standardize
  or normalize, unlike [[logistic-regression|linear models]] or distance-based methods).

**Decision boundary shape — the key structural limitation.** Because every split is
axis-aligned ("$x_j \leq t$"), a tree's decision regions are unions of axis-aligned hyper­
rectangles — a "staircase" boundary. A linear model ([[logistic-regression]]) draws a single
hyperplane; a tree can only approximate a diagonal boundary with many small steps, requiring
much more depth (and data) than a linear model would need for a linearly-separable problem.

---

## Advanced

**Why trees are the ideal base learner for bagging.** Random Forests ([[ensemble-methods]])
deliberately grow trees *deep and unpruned* — maximizing their individual variance — and then
average many of them. This only makes sense because of the instability noted above: trees are
**high-variance, low-bias** learners, exactly the profile that benefits most from variance-
reducing averaging (recall $\text{Var}(\bar y) = \sigma^2/N$ for independent learners). A
low-variance learner (e.g. a linear model) would gain little from bagging — there is little
variance left to cancel.

**Why shallow trees are the ideal base learner for boosting.** Gradient boosting
([[ensemble-methods]]) instead favors *shallow* trees (depth 3–6, sometimes single-split
"stumps"): high-bias, low-variance weak learners. Each one barely fits the data; boosting's
sequential correction mechanism supplies the bias reduction. Combined with a small learning
rate $\eta$ (shrinkage), this produces a smooth additive ensemble of piecewise-constant
functions that is far more accurate — and far more resistant to overfitting — than any single
deep tree could be. This is the precise reason the *same* base model (a tree) appears at
opposite ends of the bias-variance spectrum depending on the ensembling strategy wrapped
around it.

**CART vs. ID3 / C4.5.** The classic alternatives to CART (Quinlan's ID3 and its successor
C4.5) allow **multiway splits** on categorical features (one branch per category, rather than
CART's binary subset splits) and use **gain ratio** — information gain normalized by the
"intrinsic information" of the split — to counteract IG's bias toward high-cardinality
features. CART's insistence on binary splits makes it simpler, more uniform (works identically
for classification and regression), and is why it underlies essentially every modern
implementation (scikit-learn, XGBoost, LightGBM).

**Trees as piecewise-constant function approximators.** A regression tree of depth $d$
partitions the input space into at most $2^d$ axis-aligned regions, each predicting a constant.
This connects trees to the broader family of [[parametric-nonparametric|non-parametric
models]]: model complexity grows with the data (an unconstrained tree can have as many leaves
as training examples), rather than being fixed in advance as in a parametric model like
[[logistic-regression|logistic regression]]. It also explains a structural weakness:
approximating a smooth function (e.g. $y = x^2$) requires many small steps — trees are poor at
extrapolation and at representing smooth trends, which is part of why gradient-boosted trees
still need hundreds of shallow trees to match a smooth target well.

**Instability as a feature, not just a bug.** Two trees trained on [[bootstrap|bootstrap resamples]] of the
same dataset can disagree substantially — different top-level splits cascade into entirely
different downstream structure. This sensitivity (high variance) is usually framed as a
weakness, but it is the *exact property* that makes the ensemble methods built on top of trees
(Random Forest, gradient boosting, both covered in [[ensemble-methods]]) some of the strongest
off-the-shelf tabular-data models in practice — arguably more practically influential than the
single-tree model they're built from.

---

## Links

- [[entropy-mutual-info]] — Gini impurity and entropy are both impurity measures; information gain is the mutual information between a feature and the class label at a node
- [[statistical-inference-mle]] — regression trees minimize MSE; classification trees maximize likelihood under a categorical model
- [[ensemble-methods]] — Random Forests and gradient boosting are built on decision trees; the tree's high variance makes variance-reducing ensembles effective
- [[bias-variance-double-descent]] — deep unpruned trees have low bias and high variance; this is precisely why bagging helps
- [[regularization-weight-decay]] — cost-complexity pruning parameter $\alpha$ is analogous to the weight decay strength $\lambda$
- [[parametric-nonparametric]] — trees are non-parametric; complexity grows with the data rather than being fixed
- [[logistic-regression]] — logistic regression draws a single hyperplane; trees approximate any boundary with axis-aligned rectangles
- [[bootstrap]] — Random Forests use bootstrap resampling to decorrelate tree predictions
- [[cross-validation]] — used to choose `max_depth`, `min_samples_leaf`, and the pruning parameter $\alpha$
