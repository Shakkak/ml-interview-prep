---
title: Ensemble Methods
tags: [ensemble, random-forest, boosting, bagging, bias-variance, statistics]
aliases: [ensembles, bagging, boosting, random forest, gradient boosting, model averaging]
difficulty: 2
status: complete
related: [bias-variance-double-descent, regularization-dropout, statistical-inference-mle]
---

# Ensemble Methods

---

## Fundamental

**Why ensembles work — variance reduction:**

For $N$ independent models, each with bias $b$ and variance $\sigma^2$, the ensemble prediction $\bar{y} = \frac{1}{N}\sum_i \hat{y}_i$ satisfies:

$$\mathbb{E}[\bar{y}] = \mu + b \quad \text{(bias unchanged)}$$
$$\text{Var}(\bar{y}) = \frac{\sigma^2}{N} \quad \text{(variance reduced by } N\text{)}$$

**Key insight:** ensembles reduce variance but not bias. They help most when individual models are high-variance (overfit). They cannot fix underfitting.

With partially correlated models (correlation $\rho$):

$$\text{Var}(\bar{y}) = \frac{\sigma^2}{N} + \frac{N-1}{N}\rho\sigma^2 \approx \rho\sigma^2 \text{ for large } N$$

**Consequence:** the benefit of ensembling vanishes when models are highly correlated. Diversity is essential.

**Bagging (Bootstrap Aggregating):** train $N$ models on different [[bootstrap|bootstrap samples]] (sample $n$ examples with replacement). Average predictions (regression) or vote (classification).

Why it reduces variance: each model sees a different training set → different errors → errors partially cancel.

**Random Forest = Bagging + feature randomness**, applied to [[decision-trees|decision trees]]:
1. Bootstrap sample for each tree
2. At each split, consider only $\sqrt{d}$ random features (classification) or $d/3$ (regression)
3. Grow each tree fully (no pruning) — high-variance/low-bias individual trees
4. Average predictions

Feature randomness decorrelates trees further, breaking the tendency for all trees to make the same first split on a dominant feature.

**Out-of-bag (OOB) error:** on average, 37% of training examples ($e^{-1} \approx 0.368$) are not in each bootstrap sample. These provide a free validation set without a separate hold-out.

**Boosting:** train models sequentially; each new model focuses on examples the previous models got wrong. Reduces bias by directly correcting mistakes.

**AdaBoost:**
1. Initialize sample weights $w_i = 1/n$
2. Train model $f_t$ on weighted data; compute weighted error $\epsilon_t$
3. Compute model weight $\alpha_t = \frac{1}{2}\ln\frac{1-\epsilon_t}{\epsilon_t}$
4. Increase weights of misclassified examples: $w_i \leftarrow w_i e^{\alpha_t \mathbf{1}[\text{wrong}]}$
5. Final: $F(x) = \text{sign}\left(\sum_t \alpha_t f_t(x)\right)$

---

## Intermediate

**Gradient Boosting (Friedman, 2001):** boosting in function space. At each step, fit a new model to the negative gradient (residuals) of the loss:

$$r_i = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{t-1}}$$

For MSE loss: $r_i = y_i - F_{t-1}(x_i)$ — just the residuals. Update: $F_t(x) = F_{t-1}(x) + \eta f_t(x)$ where $\eta$ is the learning rate (shrinkage).

Why boosting reduces bias: each step directly corrects the current model's mistakes. The ensemble can represent complex nonlinear functions even if each individual tree is shallow.

**XGBoost key innovations (Chen & Guestrin, 2016):**
- Second-order Taylor expansion of the loss (uses Hessian, not just gradient)
- Regularization terms on tree complexity in the objective
- Column subsampling (like Random Forest) for additional variance reduction
- Cache-aware computation for speed

**Bagging vs Boosting:**

| Property | Bagging | Boosting |
|----------|---------|---------|
| Training | Parallel | Sequential |
| Fixes | Variance | Bias |
| Overfitting risk | Low | Higher (without regularization) |
| Example | Random Forest | XGBoost, LightGBM |
| Best when | High-variance base models | High-bias base models |

**Stacking (Stacked Generalization):**
1. Split training data into $K$ folds
2. For each fold: train base models on other folds; predict held-out fold
3. Collect base model predictions (out-of-fold) as meta-features
4. Train a meta-learner (e.g., logistic regression) on these meta-features
5. At test time: base models predict; meta-learner combines

Advantage: can combine models of different types (CNN + GBM + SVM) and learn their strengths automatically.

**When ensembles don't help:**
1. Highly correlated models: averaging does not cancel errors
2. High-bias models: averaging underfit models gives a still-underfit ensemble
3. Speed or small model size required: ensembles are $N\times$ more expensive at inference
4. Large data: a single well-trained model often beats an ensemble on small data

**Diversity techniques:**

| Method | How |
|--------|-----|
| Bagging | Different training subsets |
| Random subspaces | Different feature subsets |
| Different seeds | Different random initialization |
| Different architectures | CNN + Transformer + GBM |
| Snapshot ensembles | Save checkpoints at different LR cycle phases |

---

## Advanced

**Connection to [[regularization-dropout|Dropout]]:** Dropout at inference time (MC Dropout) runs the same model $N$ times with dropout active. Averaging approximates a Bayesian model average over the posterior of weights — a cheap ensemble from a single model. The equivalence is approximate: dropout posterior is not the exact Bayesian posterior, but provides calibrated uncertainty estimates for free.

**Deep ensemble calibration (Lakshminarayanan et al., 2017):** train $M$ networks with random initialization, adversarial training on each. Ensemble predictions average the predictive distributions (not just logits). Empirically, ensembles of 5–10 models achieve far better calibration than temperature scaling alone, particularly for out-of-distribution inputs where the models disagree. The disagreement across ensemble members is a reliable uncertainty signal — high variance across ensemble members signals OOD.

**Gradient boosting and additive models:** gradient boosting with decision trees produces a piecewise constant additive function. The number of trees, tree depth, and learning rate interact in complex ways. Empirical finding: it is better to use more shallow trees with a small learning rate than fewer deep trees with a large learning rate — the shrinkage introduced by a small $\eta$ acts as explicit regularization, preventing individual trees from overfitting before the ensemble can self-correct.

**LightGBM vs XGBoost:** LightGBM (Ke et al., 2017) uses leaf-wise (best-first) tree growth rather than level-wise. Level-wise: grow all nodes at a depth before moving deeper. Leaf-wise: always split the node with the highest gain. Leaf-wise trees are deeper and more accurate for the same number of leaves, but require careful regularization (max depth, min data per leaf). LightGBM also uses Gradient-based One-Side Sampling (GOSS) — keep all instances with large gradients (poorly fitted), randomly sample small-gradient instances — achieving 20× speedup with minimal accuracy loss.

**The "no free lunch" theorem for ensembles:** the benefits of ensemble averaging hold in expectation over problems. For any specific problem, a perfectly tuned single model can match an ensemble. However, the ensemble is more robust — it is less likely to be completely wrong because it requires multiple models to simultaneously fail in the same way. This robustness is why ensembles are standard in competitive machine learning even when a single model achieves comparable average performance.

---

*See also: [[bias-variance-double-descent]] · [[regularization-dropout]] · [[statistical-inference-mle]] · [[bayesian-inference]] · [[decision-trees]] · [[bootstrap]]*
