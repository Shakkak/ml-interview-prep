---
title: Logistic Regression
tags: [statistics, classification, inference, linear-models]
aliases: [logistic regression, log-odds, odds ratio, sigmoid regression]
difficulty: 1
status: complete
related: [statistical-inference-mle, activation-sigmoid-tanh, loss-cross-entropy, regularization-weight-decay, kernel-methods]
---

# Logistic Regression

---

## Fundamental

Linear regression predicts a continuous output. For binary classification ($y \in \{0,1\}$), predicting probabilities directly with a linear model fails: outputs can exceed $[0,1]$, and the assumption that probability changes linearly with features is rarely realistic.

**Derivation from log-odds.** Model the log-odds (logit) as linear:

$$\log\frac{P(y=1|x)}{P(y=0|x)} = w^\top x + b \equiv z$$

The log-odds can take any real value, so a linear model is appropriate here. Solving for $P(y=1|x)$:

$$P(y=1|x) = \frac{e^z}{1+e^z} = \sigma(z) = \frac{1}{1+e^{-z}}$$

The [[activation-sigmoid-tanh|sigmoid]] is not an arbitrary output function — it is forced by the assumption that log-odds are linear in features.

**Odds ratios.** For a one-unit increase in feature $x_j$, the log-odds increases by $w_j$, so the odds are multiplied by $e^{w_j}$:

| $w_j$ | $e^{w_j}$ | Interpretation |
|---|---|---|
| $0.5$ | $1.65$ | Odds increase 65% per unit |
| $-1.0$ | $0.37$ | Odds decrease 63% per unit |
| $2.0$ | $7.39$ | Odds increase 7.4× per unit |

Unlike linear regression, logistic regression coefficients give exact multiplicative effects on odds, not additive effects on probability.

**Worked example.** $w = [1.0, -0.5]$, $b = 0.2$, $x = [2.0, 3.0]$.

$z = 1.0(2.0) + (-0.5)(3.0) + 0.2 = 0.7$. Log-odds = 0.7, odds = $e^{0.7} = 2.01$, probability = $\sigma(0.7) = 0.668$.

**Decision boundary.** Predict $y=1$ iff $\hat{p} > 0.5$, i.e., $z = w^\top x + b > 0$. This is a **hyperplane** in feature space — logistic regression is a linear classifier.

---

## Intermediate

### MLE = Cross-Entropy Minimization

The likelihood for $n$ binary observations is $\prod_i \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$. The log-likelihood:

$$\ell(w) = \sum_{i=1}^n \left[y_i \log\sigma(z_i) + (1-y_i)\log(1-\sigma(z_i))\right] = -\mathcal{L}_{CE}$$

Maximizing log-likelihood equals minimizing binary [[loss-cross-entropy|cross-entropy]]. The gradient is elegant:

$$\frac{\partial \ell}{\partial w} = \sum_{i=1}^n (y_i - \sigma(z_i))\, x_i = \sum_i (\text{label} - \text{prediction}) \cdot x_i$$

Same form as linear regression's OLS gradient. The error $(y_i - \hat{p}_i)$ scales each feature's update — simple and interpretable.

**One gradient step.** $n=2$ points: $(x^{(1)}=[1,0], y^{(1)}=1)$ and $(x^{(2)}=[0,1], y^{(2)}=0)$. Init $w=[0,0]$.

Both $\hat{p} = \sigma(0) = 0.5$. $\nabla w_1 = (1-0.5)(1) + (0-0.5)(0) = 0.5$. $\nabla w_2 = (1-0.5)(0) + (0-0.5)(1) = -0.5$.

With $\eta=0.1$: $w_1 \leftarrow 0.05$, $w_2 \leftarrow -0.05$. Feature 1 gets positive weight (predictive of $y=1$), feature 2 gets negative weight ✓.

### Regularization and Extensions

**[[regularization-weight-decay|L2 regularization]]** (ridge logistic regression): adds $\lambda\|w\|^2$ to the loss. Prevents extreme log-odds, improves calibration. In scikit-learn: `C = 1/λ` (note: inverse regularization strength).

**L1 regularization**: produces sparse weights — features with small coefficients are zeroed out. Natural for feature selection in high-dimensional problems.

**Multinomial ([[activation-softmax|softmax]]) extension.** For $K > 2$ classes:
$$P(y=k|x) = \frac{e^{w_k^\top x + b_k}}{\sum_{j=1}^K e^{w_j^\top x + b_j}}$$

The softmax is the multi-class generalization of the sigmoid. Loss: categorical cross-entropy. The $K$-class model has $K-1$ identifiable parameters (one class is the reference).

**Non-linear decision boundaries:** add polynomial features ($x_1^2, x_1 x_2, \ldots$) before logistic regression, or use a [[kernel-methods|kernel]] (kernelized logistic regression). Quadratic features give elliptical decision boundaries; RBF kernel gives arbitrary boundaries.

**When to use logistic regression:**

| Property | Logistic Regression |
|---|---|
| Decision boundary | Linear |
| Calibration | Well-calibrated by default |
| Interpretability | High (odds ratios) |
| Training speed | Very fast (no architecture search) |
| Non-linear patterns | Requires feature engineering |

---

## Advanced

### Connection to Exponential Family and GLMs

Logistic regression is a **Generalized Linear Model (GLM)** with Bernoulli distribution and logit link function. In the GLM framework, all linear models (linear regression, Poisson regression, logistic regression) are unified:
1. Response distribution is in the exponential family: $p(y|\eta) = h(y)\exp(\eta y - A(\eta))$
2. Natural parameter $\eta = w^\top x + b$ (linear predictor)
3. Mean function $\mu = A'(\eta)$ relates prediction to distribution mean

For Bernoulli: $A(\eta) = \log(1 + e^\eta)$, so $\mu = A'(\eta) = \sigma(\eta) = p$. The logit link is the **canonical link** for the Bernoulli family, making logistic regression the natural choice.

This unification means logistic regression inherits the asymptotic theory of GLMs: the [[statistical-inference-mle|MLE]] is consistent, asymptotically normal, and achieves the [[fisher-information|Cramér-Rao bound]].

### Perfect Separation and the Hauck-Donner Effect

When the training data is **linearly separable**, the MLE for logistic regression does not exist — the log-likelihood has no finite maximum. Weights diverge as the loss drives toward zero: $\|w\| \to \infty$ with the decision boundary becoming a hard step function.

In practice this manifests as extremely large coefficients and inflated standard errors (the **Hauck-Donner effect** — Wald test $p$-values become anti-conservative as effect size grows). Detection: check if any coefficient is $> 10\times$ the others or has very large standard error.

Fixes: (1) L2 regularization (ridge) adds a proper Bayesian prior, ensuring finite MLE; (2) Firth's penalized MLE uses Jeffreys prior to obtain finite estimates; (3) More data or feature reduction to break perfect separation.

### Logistic Regression vs Neural Networks

Logistic regression is a single-layer neural network with sigmoid activation and cross-entropy loss. Adding hidden layers makes it a neural network. The key insight: **feature representation matters more than the classifier**.

Feature engineering in logistic regression does the same job as hidden layers in neural networks — both map raw inputs to a space where the classes are linearly separable. Logistic regression with hand-crafted features was competitive with neural networks for NLP tasks until 2012–2015. The resurgence of neural networks is largely about **representation learning**: networks learn the feature transformation rather than having it hand-specified.

Modern use: **linear probing** (train logistic regression on frozen neural network embeddings) is the standard protocol for evaluating self-supervised representation quality (e.g., [[contrastive-learning|SimCLR]], DINO, MAE). The quality of the learned features is measured precisely by how well a logistic regression head can exploit them.

---

*See also: [[activation-sigmoid-tanh]] · [[loss-cross-entropy]] · [[regularization-weight-decay]] · [[statistical-inference-mle]] · [[kernel-methods]] · [[activation-softmax]] · [[fisher-information]] · [[contrastive-learning]]*
