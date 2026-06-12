---
title: Gradient Boosting
tags: [gradient-boosting, xgboost, lightgbm, boosting, ensemble, trees]
aliases: [gradient boosted trees, GBM, gradient boosted machines, XGBoost, LightGBM, CatBoost]
difficulty: 2
status: complete
related: [decision-trees, ensemble-methods, loss-mse, loss-cross-entropy, regularization-weight-decay, bias-variance-double-descent]
---

# Gradient Boosting

---

## Fundamental

### Boosting Intuition

**Boosting** builds an ensemble by sequentially adding models that correct the mistakes of the previous ensemble. Unlike bagging (which trains models independently in parallel), boosting trains models sequentially — each new model focuses on the hard examples.

**The key insight:** instead of fitting a model to the original labels, each new model fits the **residuals** — how wrong the current ensemble is. Adding a model that reduces the residuals reduces the total error.

For regression with MSE loss, the optimal greedy strategy is:
1. Start with $F_0(\mathbf{x}) = \bar{y}$ (mean of targets)
2. At step $m$: compute residuals $r_i = y_i - F_{m-1}(\mathbf{x}_i)$
3. Fit a new model $h_m$ to the residuals
4. Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$ where $\eta$ is the learning rate

This is **gradient descent in function space** — residuals are the negative gradient of MSE.

### Gradient Boosting (General Form)

Friedman (2001) generalized this to arbitrary differentiable loss $\mathcal{L}$: fit each $h_m$ to the **negative gradient** of the loss:

$$r_i^{(m)} = -\left[\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

For MSE: $r_i = y_i - F_{m-1}(\mathbf{x}_i)$ (actual residuals).
For log-loss: $r_i = y_i - p_i$ where $p_i$ is the predicted probability (residuals in probability space).
For any loss: $r_i$ are the "pseudo-residuals" — the direction in prediction space that most reduces the loss.

The ensemble of $M$ trees is: $F_M(\mathbf{x}) = F_0 + \eta \sum_{m=1}^M h_m(\mathbf{x})$

---

## Intermediate

### Decision Trees as Base Learners

Each $h_m$ is typically a **shallow decision tree** (depth 3–6). Trees are:
- Non-linear: can capture interactions and non-smooth regions
- Piecewise constant: fit any function with enough trees
- Fast: $O(N \log N)$ per tree (sort features once)
- Naturally handle mixed types (continuous + categorical)

**Why not deep trees?** Deep trees overfit the residuals. Shallow trees act as weak learners — each reduces the bias a little, and the ensemble of many shallow trees achieves low bias + low variance.

**Shrinkage:** the learning rate $\eta \in (0.01, 0.1]$ is critical. Small $\eta$ requires more trees ($M$) but generalizes better. The optimal strategy is small $\eta$ + large $M$ with early stopping on a validation set.

### XGBoost: Second-Order Optimization

Standard gradient boosting uses first-order gradients. XGBoost (Chen & Guestrin, 2016) uses both first and second derivatives of the loss, approximating the loss as a second-order Taylor expansion:

$$\mathcal{L}^{(m)} \approx \sum_i [g_i f(\mathbf{x}_i) + \frac{1}{2}h_i f(\mathbf{x}_i)^2] + \Omega(f)$$

where $g_i = \partial \mathcal{L}/\partial \hat{y}_i$ (gradient) and $h_i = \partial^2 \mathcal{L}/\partial \hat{y}_i^2$ (Hessian) are computed from current predictions.

**Optimal leaf weights:** for a leaf containing instances $I_j$, the optimal score is:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**Optimal split gain:**
$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in L} g_i)^2}{\sum_{i \in L} h_i + \lambda} + \frac{(\sum_{i \in R} g_i)^2}{\sum_{i \in R} h_i + \lambda} - \frac{(\sum_i g_i)^2}{\sum_i h_i + \lambda}\right] - \gamma$$

where $\lambda$ regularizes leaf weights and $\gamma$ is the minimum gain to make a split. **XGBoost learns faster per tree** because it uses curvature information.

**Regularization in XGBoost:** $\Omega(f) = \gamma T + \frac{\lambda}{2}\sum_j w_j^2$ penalizes the number of leaves $T$ and leaf weight magnitude. This built-in regularization allows XGBoost to use deeper trees without overfitting.

### LightGBM: Speed at Scale

LightGBM (Ke et al., 2017) scales gradient boosting to datasets with millions of rows:

**Histogram-based splitting:** instead of sorting data by each feature ($O(N)$ per split), discretize features into bins (256 by default) and compute splits using histogram statistics — $O(B)$ per split where $B \ll N$. Speed: 10–20× faster than XGBoost's exact split algorithm.

**GOSS (Gradient-based One-Side Sampling):** keep all instances with large gradients (they contribute most to the gradient update) and randomly sample a fraction of small-gradient instances. Reduces training instances while maintaining accuracy.

**EFB (Exclusive Feature Bundling):** bundle mutually exclusive features (features that rarely take non-zero values simultaneously) into single features. Reduces the effective feature dimensionality in sparse datasets.

**Leaf-wise tree growth:** instead of growing trees level-by-level (depth-wise), LightGBM splits the leaf with maximum gain. This finds better splits faster but requires careful max-depth constraints to avoid overfitting.

---

## Advanced

### CatBoost and Ordered Boosting

Standard gradient boosting has a subtle bias: the gradient $r_i$ at step $m$ uses predictions $F_{m-1}(\mathbf{x}_i)$ computed on data that includes $(\mathbf{x}_i, y_i)$ itself — a form of target leakage. This makes the gradient estimates slightly biased, which compounds over many iterations.

**CatBoost (Prokhorenkova et al., 2018)** uses **ordered boosting**: for each training point, compute its gradient using a model trained on a random subset of earlier data (ordered by a random permutation). This eliminates the target leakage bias, often improving validation performance.

CatBoost also handles categorical features natively via **ordered target statistics** (computing mean-target per category using only preceding observations in the permutation order).

### Feature Importance

Gradient boosted trees provide multiple importance metrics:

**Split-based:** how many times a feature is used to split across all trees. Fast but biased toward high-cardinality features.

**Gain-based:** total gain (reduction in training loss) accumulated over all splits using the feature. More meaningful than split count.

**SHAP (Shapley Additive exPlanations):** the Shapley value of each feature — the average marginal contribution across all possible orderings of features. Theoretically grounded, consistent, and can attribute predictions to specific features for individual examples. XGBoost and LightGBM have native SHAP computation.

### When to Use Gradient Boosting vs Neural Networks

| Criterion | Gradient Boosting | Neural Networks |
|-----------|------------------|-----------------|
| Tabular structured data | Usually wins | Can compete with tuning |
| Time to good result | Fast (minutes) | Slow (hours–days) |
| Feature engineering | Less needed | Less needed |
| Interpretability | SHAP values | Less interpretable |
| High-dim inputs (images, text) | Not suitable | Excels |
| Very large datasets ($N > 10^7$) | LightGBM scales well | Scales with distributed training |
| Missing values | Handles natively | Requires imputation |

Empirically, gradient boosting (usually XGBoost or LightGBM) wins on tabular data competitions. Neural networks dominate for unstructured data.

---

*See also: [[decision-trees]] · [[ensemble-methods]] · [[bias-variance-double-descent]] · [[regularization-weight-decay]] · [[loss-mse]] · [[loss-cross-entropy]]*
