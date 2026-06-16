---
title: Bias-Variance Tradeoff and Double Descent
tags: [statistics, generalization, overfitting, theory, machine-learning]
aliases: [bias-variance, double descent, interpolation threshold, implicit regularization]
difficulty: 2
status: complete
related: [bayesian-inference, backpropagation, normalization-layers]
depends_on: [statistical-inference-mle, regularization-weight-decay, generalization-bounds]
---

# Bias-Variance Tradeoff and Double Descent

---

## Fundamental

We train a model $\hat{f}$ on training set $\mathcal{D}$ drawn from distribution $P(X, Y)$. The **expected test error** for a new point $x_0$ with true relationship $Y = f(x_0) + \epsilon$:

$$\mathbb{E}_{\mathcal{D}, \epsilon}\left[(Y - \hat{f}(x_0))^2\right] = \underbrace{\left(f(x_0) - \mathbb{E}[\hat{f}(x_0)]\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)]\right)^2\right]}_{\text{Variance}} + \underbrace{\mathbb{E}[\epsilon^2]}_{\text{Irreducible Noise}}$$

where $\hat{f}$ = the model trained on dataset $\mathcal{D}$, $x_0$ = a test point, $f(x_0)$ = the true underlying function value at $x_0$, $\mathbb{E}[\hat{f}(x_0)]$ = expected prediction of $\hat{f}$ at $x_0$ averaged over all possible training sets $\mathcal{D}$, $\epsilon$ = irreducible observation noise (e.g., label noise), and $\mathbb{E}_{\mathcal{D}, \epsilon}$ = expectation over both the random training set and noise.

**Bias:** systematic error — how far the average prediction (over all possible training sets) is from the truth. A linear model fitted to a cubic relationship always misses the curvature regardless of how much data it sees. No amount of training data fixes structural model limitations.

**Variance:** how much the prediction changes across different training sets. A degree-100 polynomial passes through all training points, but a different training set gives a completely different polynomial.

**Classical tradeoff:**

For a family of models with tunable complexity (polynomial degree, KNN neighbors, tree depth), test error is U-shaped:

```
                Test Error
                    │
                    │      Overfit (high variance)
                    │     /
          Underfit  │    /
          (high    │   /  ← sweet spot
           bias)   │  /
                    │ /
                    │/___________
                    └──────────────── Model Complexity
```

**Intuition:** a 1-nearest-neighbor classifier has zero training error but high test error (memorizes noise); a constant classifier has high training error but is stable. The optimal model balances both.

**Geometric intuition for bias:** bias is the gap between the best model in the class and the truth. If the true function is sinusoidal but the model class is linear, bias is the best linear approximation's error — irreducible for this model class.

**Geometric intuition for variance:** variance is the spread of estimates around the class mean. Complex models with many parameters can fit any training set well, but different training sets produce wildly different fits — the model is sensitive to which $n$ examples happened to be drawn.

---

## Intermediate

**Full bias-variance derivation:**

Let $\bar{f} = \mathbb{E}_\mathcal{D}[\hat{f}(x_0)]$ be the expected prediction. Then:

$$\mathbb{E}\left[(f - \hat{f})^2\right] = \mathbb{E}\left[(f - \bar{f} + \bar{f} - \hat{f})^2\right] = (f - \bar{f})^2 + \mathbb{E}[(\hat{f} - \bar{f})^2]$$

The cross term $2(f-\bar{f})\mathbb{E}[\bar{f} - \hat{f}] = 0$ by definition of $\bar{f}$.

**Bias-variance for specific models:**

KNN: as $k$ increases, variance decreases (average more neighbors → stable prediction) and bias increases (farther neighbors may be wrong class). Optimal $k$ via [[cross-validation]].

Ridge regression: $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$ (where $X \in \mathbb{R}^{n \times p}$ = data matrix, $y \in \mathbb{R}^n$ = targets, $\lambda I$ = ridge penalty added to diagonal — makes the matrix invertible and shrinks $\hat{\beta}$). As $\lambda$ increases, variance decreases ($\hat{\beta}$ shrinks toward 0, less sensitivity to training set) but bias increases (solutions pulled away from truth). The $\lambda$ that minimizes test MSE is the bias-variance sweet spot.

**Practical implications:**

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| High train error + high test error | High bias (underfit) | More complex model, better features |
| Low train error + high test error | High variance (overfit) | Regularization, more data, simpler model |
| Both errors high | Possibly both | Try both |

Regularization (L2, [[regularization-dropout|dropout]], early stopping) reduces variance at the cost of slight bias increase — net benefit when variance dominates.

**Key insight often missed:** more data reduces variance but not bias. If the model class cannot represent the true function, more data does not help — the systematic error persists.

---

## Advanced

**Double descent (Belkin et al., 2019):** the U-curve is not the full picture. The true curve has **two descents**:

```
                Test Error
                    │
                    │         ╮ peak at
                    │         │ interpolation threshold
                    │     ╮   ╰──────────────
                    │     │         (second descent)
                    │     ╰ classical U-curve
                    └──────────────────────────── Model Complexity
                              ↑
                    interpolation threshold
```

**Interpolation threshold:** the point where the model has just enough capacity to fit the training data exactly (zero training loss). Classical statistics predicts this is the worst point.

**Double descent:** beyond the interpolation threshold, increasing model size further *decreases* test error. Very overparameterized models generalize well.

**Why it occurs — implicit regularization:** for overparameterized models, many parameter values fit the training data exactly. Gradient descent selects one specific solution — not randomly, but with an implicit bias toward a particular type of solution.

For linear models (analytically tractable), gradient descent from zero converges to the **minimum L2-norm solution** among all interpolating solutions:

$$\hat{\theta} = \arg\min \|\theta\|_2 \text{ s.t. } X\theta = y$$

This is well-regularized even though it fits all training points exactly. The minimum-norm interpolant in an overparameterized linear model equals the ridgeless limit of ridge regression ($\lambda \to 0^+$), connecting classical regularization to implicit regularization.

**Why the peak at the interpolation threshold:** at this threshold, the model can barely fit the training data — but only in a jagged, high-variance way. There is no "room" to find a smooth interpolating solution. Adding a bit more capacity allows finding the smooth minimum-norm interpolant.

**Epoch-wise double descent (Nakkiran et al., 2020):** double descent also occurs along the training axis. Test error can first decrease, then increase (classical overfitting), then decrease again as training continues. This happens because early in training the model is in the underparameterized regime for any given number of gradient steps; eventually it enters the overparameterized regime.

**[[neural-tangent-kernel|Neural Tangent Kernel]] connection:** in the infinite-width limit, neural networks trained with gradient flow are governed by the NTK. The NTK dynamics are linear, and the minimum-norm solution corresponds to minimum RKHS norm in the NTK's reproducing kernel Hilbert space. This provides a theoretical foundation for why overparameterized networks find smooth, generalizing solutions.

**Practical implications for practitioners:**

| Regime | Training Error | Test Error | Recommendation |
|--------|---------------|------------|----------------|
| Underfit | High | High | Increase capacity, train longer |
| Near interpolation threshold | Low | High | Regularize or increase capacity |
| Overparameterized | Zero | Decreasing | Regularize to improve further |

Rule of thumb: if you have enough compute, larger models almost always generalize better than smaller ones, given the same data. Don't be afraid of overparameterization — the second descent is real.

---

## Links

- [[statistical-inference-mle]] — the classical bias-variance tradeoff is derived from the MLE expected error decomposition: $E[\text{err}] = \text{Bias}^2 + \text{Variance} + \text{Noise}$
- [[regularization-weight-decay]] — regularization reduces variance at the cost of bias; the tradeoff minimum is found by cross-validation over the regularization strength
- [[generalization-bounds]] — double descent is explained by implicit regularization in overparameterized models; PAC-Bayes bounds and NTK theory formalize why interpolating models generalize
- [[spectral-bias]] — double descent occurs at the interpolation threshold; spectral bias explains why wide networks converge to minimum-norm solutions that generalize despite zero training error
- [[regularization-dropout]] — dropout is a variance reduction technique; too much dropout increases bias; the optimal rate trades off variance (model capacity) against bias
- [[neural-tangent-kernel]] — in the NTK regime, the test error of overparameterized models can be analyzed analytically; the NTK theory explains the second descent in the double-descent curve
- [[cross-validation]] — cross-validation estimates the bias-variance tradeoff empirically; k-fold CV selects the model complexity minimizing validation error
