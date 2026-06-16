---
title: Imbalanced Sampling Techniques
tags: [imbalanced-sampling, smote, adasyn, tomek-links, oversampling, undersampling]
aliases: [SMOTE, ADASYN, Tomek links, cluster centroids, NearMiss, oversampling, undersampling]
difficulty: 1
status: complete
depends_on: [feature-preprocessing, distributions-overview]
related: [class-imbalance, data-augmentation, feature-preprocessing, loss-focal, evaluation-metrics-guide]
---

# Imbalanced Sampling Techniques

---

## Fundamental

### Why Sampling?

Class imbalance causes models to ignore minority classes. One solution is to modify the training data distribution via **oversampling** (add minority examples) or **undersampling** (remove majority examples) before training.

**Intuition:** if the minority class has only 1% of examples, the model can achieve 99% accuracy by always predicting the majority class — never seeing a minority example worth learning from. Sampling restores the balance before training starts, making the minority class an equally-weighted learning signal.

Sampling works at the data level — the model sees a balanced distribution regardless of the algorithm used. Contrast with **algorithmic approaches** (focal loss, class weights) which work at the loss level.

---

## Intermediate

### Oversampling Techniques

**Random oversampling:** duplicate minority class examples at random until balance is achieved. Simple and fast. Risk: exact duplicates cause overfitting to specific training points — the model memorizes boundary examples.

**SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al., 2002):**
For each minority example $x_i$:
1. Find its $k$ nearest minority-class neighbors ($k = 5$ default)
2. Randomly select one neighbor $x_j$
3. Generate: $x_\text{new} = x_i + \lambda(x_j - x_i)$, $\lambda \sim U[0, 1]$

Synthetic examples lie on the line segment between real minority examples — filling the feature space of the minority class rather than exact copies.

**ADASYN (Adaptive Synthetic Sampling):** same as SMOTE but generates more synthetic samples for minority examples that are harder to learn (those surrounded by many majority-class neighbors):
$$n_i \propto \frac{\text{# majority neighbors of } x_i}{k}$$

Focuses synthetic data where the boundary is most confused.

**Borderline-SMOTE:** only oversample minority examples near the decision boundary (those with at least half majority neighbors) — avoids creating synthetic examples deep inside the minority region where the model already performs well.

### Undersampling Techniques

**Random undersampling:** randomly discard majority examples. Aggressive — can discard informative examples. Best combined with ensemble methods (each base learner sees a different random subset of the majority class).

**Tomek links:** a Tomek link is a pair $(x_i, x_j)$ of opposite-class examples that are each other's nearest neighbor. Remove the majority-class example in each Tomek link — this cleans the decision boundary by removing ambiguous majority examples near the boundary.

**Cluster centroids:** cluster majority class examples into $K$ clusters (using K-Means), replace each cluster with its centroid. Reduces majority class while preserving its distributional structure.

**NearMiss:** select majority examples that are closest to minority examples (NearMiss-1: average distance to 3 nearest minority; NearMiss-2: average distance to 3 farthest minority). Focuses the majority class on boundary regions.

### Combined Approaches

**SMOTE + Tomek links (SMOTETomek):** oversample minority with SMOTE, then clean both classes' noisy boundary samples with Tomek links removal. Both steps improve boundary quality.

**SMOTE + ENN (SMOTEENN):** use Edited Nearest Neighbors (remove examples whose majority of $k$ nearest neighbors belong to a different class) instead of Tomek links for a more aggressive cleaning step.

---

## Advanced

### When Sampling Helps vs Hurts

| Situation | Sampling likely helps | Sampling likely hurts |
|-----------|----------------------|----------------------|
| Low-dimensional tabular data | ✓ SMOTE well-defined | — |
| High-dimensional data (images) | ✗ Interpolation off-manifold | Use augmentation instead |
| Decision tree / gradient boosting | ✓ | — |
| Deep networks | Marginal; prefer loss weighting | SMOTE expensive |
| Extreme imbalance (1:1000) | ✓ Strong signal needed | Random OS may work |

**Images and text:** SMOTE-style interpolation in pixel or token space creates unrealistic examples. Use **augmentation-based oversampling** (random crops, flips, color jitter for images) instead.

### Evaluation Pitfall

Apply sampling **inside the CV fold** (to training data only). Applying it before splitting leaks future information into the training set and inflates validation performance (see [[data-leakage]]).

```python
# Correct: apply SMOTE inside the fold
for train_idx, val_idx in cv.split(X, y):
    X_train_resampled, y_train_resampled = smote.fit_resample(X[train_idx], y[train_idx])
    # train on X_train_resampled, evaluate on X[val_idx]
```

## Links

- [[feature-preprocessing]] — SMOTE interpolates in feature space; features should be scaled before SMOTE as it is distance-based
- [[distributions-overview]] — imbalanced sampling methods change the class marginal distribution $P(y)$; they do not change the class-conditional distribution $P(x|y)$
- [[class-imbalance]] — class-imbalance covers the problem definition and diagnostic tools; this file covers the sampling-based solutions
- [[data-augmentation]] — augmentation addresses imbalance by generating new examples from existing ones; SMOTE is a feature-space version of augmentation
- [[loss-focal]] — Focal Loss addresses class imbalance at the loss level without resampling; it down-weights easy majority-class examples
- [[data-leakage]] — oversampling must be done inside cross-validation folds to prevent synthetic minority examples from appearing in validation folds
