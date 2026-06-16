---
title: Linear Regression
tags: [linear-regression, ols, ridge, lasso, regression, statistics]
aliases: [OLS, ordinary least squares, ridge regression, LASSO, linear model]
difficulty: 1
status: complete
related: [loss-mse, regularization-weight-decay, statistical-inference-mle, bias-variance-double-descent, numerical-methods, matrix-calculus, logistic-regression]
depends_on: [linear-algebra-fundamentals, loss-mse, statistical-inference-mle]
---

# Linear Regression

---

## Fundamental

**Linear regression** is the simplest prediction model: it estimates a continuous output as a weighted sum of input features. It matters far beyond its simplicity — it is the only model with a closed-form optimal solution, making it the ideal case study for understanding regularization (Ridge and LASSO), the bias-variance tradeoff, and what optimization is trying to do. Every more complex model is essentially doing what linear regression does, but in a feature space learned jointly with the prediction.

### Model and Objective

Linear regression models the relationship between inputs $\mathbf{x} \in \mathbb{R}^p$ and scalar output $y$:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + \cdots + w_p x_p + b$$

The objective is to minimize the mean squared error over $N$ training examples (see [[loss-mse]]):

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N}\|X\mathbf{w} - \mathbf{y}\|^2 = \frac{1}{N}\sum_{i=1}^N (y_i - \mathbf{w}^\top \mathbf{x}_i)^2$$

where $X \in \mathbb{R}^{N \times p}$ is the design matrix with bias absorbed (append 1 to each $\mathbf{x}_i$).

### The Closed-Form Solution

The MSE loss is convex and quadratic in $\mathbf{w}$. Setting the gradient to zero:

$$\nabla_\mathbf{w} \mathcal{L} = \frac{2}{N} X^\top(X\mathbf{w} - \mathbf{y}) = 0$$

Yields the **normal equations**: $X^\top X\mathbf{w} = X^\top \mathbf{y}$

**Ordinary Least Squares (OLS) solution:**
$$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

This requires $X^\top X$ to be invertible — which fails when $p \geq N$ or when features are perfectly collinear.

**Geometric interpretation:** $X\hat{\mathbf{w}}$ is the **orthogonal projection** of $\mathbf{y}$ onto the column space of $X$. The residual $\mathbf{y} - X\hat{\mathbf{w}}$ is orthogonal to all columns of $X$ — minimizing the distance from $\mathbf{y}$ to the space of achievable predictions.

### Probabilistic Interpretation (MLE)

Assuming $y_i = \mathbf{w}^\top \mathbf{x}_i + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, the log-likelihood is:

$$\log p(\mathbf{y}|X, \mathbf{w}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|X\mathbf{w} - \mathbf{y}\|^2$$

Maximizing this is equivalent to minimizing MSE. OLS is the MLE under Gaussian noise — connecting statistics and optimization (see [[statistical-inference-mle]]).

---

## Intermediate

### Ridge Regression (L2 Regularization)

When $X^\top X$ is ill-conditioned or $p \geq N$, add an L2 penalty:

$$\mathcal{L}_\text{ridge}(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$$

Closed-form: $\hat{\mathbf{w}}_\text{ridge} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$

The $\lambda I$ term ensures invertibility — the matrix is always positive definite. Numerically, ridge adds $\lambda$ to every singular value of $X^\top X$, preventing the small singular values from causing numerical instability.

**Bayesian interpretation (MAP):** ridge = MAP estimation with a zero-mean Gaussian prior $\mathbf{w} \sim \mathcal{N}(0, \sigma^2/\lambda \cdot I)$. The regularization strength $\lambda$ encodes prior belief that weights are small. See [[regularization-weight-decay]].

**Effect on SVD:** let $X = U\Sigma V^\top$. The OLS solution is $\hat{\mathbf{w}} = V\Sigma^{-1}U^\top \mathbf{y}$. Ridge: $\hat{\mathbf{w}}_\text{ridge} = V\text{diag}(\sigma_j / (\sigma_j^2 + \lambda))U^\top \mathbf{y}$. Ridge shrinks each component $\mathbf{v}_j$ by a factor $\sigma_j^2/(\sigma_j^2 + \lambda)$ — directions with small singular values (unstable in OLS) are shrunk most. This is the bias-variance tradeoff: accepting bias to reduce variance.

### LASSO (L1 Regularization)

$$\mathcal{L}_\text{lasso}(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|_1$$

No closed form — requires iterative optimization (coordinate descent, ISTA/FISTA).

**Sparsity:** L1 penalty induces exact zeros — LASSO performs automatic feature selection. Intuitively, the L1 ball has corners at the axes; the optimal solution often lands exactly at a corner, setting some weights to zero. L2 ball is smooth — no corners, weights approach but never reach zero.

**When to use which:**
- Ridge: many small effects (all features relevant), correlated features (ridge handles correlation, LASSO arbitrarily picks one)
- LASSO: few important features, feature selection needed, interpretability required
- Elastic Net: $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|^2$ — combines both, better than LASSO for correlated features

### Numerical Stability

The condition number of $X^\top X$ is $\kappa(X^\top X) = \kappa(X)^2$ — squaring the condition number makes the normal equations numerically unstable for ill-conditioned $X$ (see [[numerical-methods]]).

**Better approach:** QR decomposition. Write $X = QR$, then:
$$\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top \mathbf{y} = R^{-1}Q^\top \mathbf{y}$$

Solving $R\mathbf{w} = Q^\top \mathbf{y}$ by back-substitution has condition number $\kappa(X)$, not $\kappa(X)^2$. Always use `np.linalg.lstsq` (QR-based) rather than forming $X^\top X$ explicitly.

---

## Advanced

### Gauss-Markov Theorem

OLS is **BLUE** (Best Linear Unbiased Estimator): among all linear unbiased estimators of $\mathbf{w}$, OLS has the minimum variance. This requires:
1. $\mathbb{E}[\epsilon] = 0$ (zero-mean errors)
2. $\text{Var}(\epsilon) = \sigma^2 I$ (homoscedastic, uncorrelated errors)
3. No multicollinearity ($\text{rank}(X) = p$)

Violating assumption 2 (heteroscedastic or correlated errors) requires Generalized Least Squares (GLS): $\hat{\mathbf{w}}_\text{GLS} = (X^\top \Omega^{-1} X)^{-1} X^\top \Omega^{-1} \mathbf{y}$ where $\text{Var}(\boldsymbol{\epsilon}) = \sigma^2 \Omega$.

### Interpolation and Double Descent

In the overparameterized regime ($p \gg N$), OLS has infinitely many solutions that perfectly fit the training data. The minimum-norm solution (from pseudoinverse) is $\hat{\mathbf{w}} = X^\dagger \mathbf{y} = X^\top(XX^\top)^{-1}\mathbf{y}$.

Surprisingly, for sufficiently large $p$, the test error of the minimum-norm interpolating solution *decreases* as $p$ grows — the **double descent** phenomenon (see [[bias-variance-double-descent]]). The minimum-norm solution is "benign overfitting" in sufficiently high dimensions because the extra dimensions let it fit the noise in low-variance directions.

For linear regression, the double descent peak is exactly at $p = N$ (the phase transition between underdetermined and overdetermined regimes).

### Locally Weighted Regression

For non-linear relationships without explicit feature engineering: fit a weighted OLS at each test point $x_*$, with weights $w_i = \exp(-\|x_i - x_*\|^2 / 2\tau^2)$. Nearby training points matter most. LOESS (locally estimated scatterplot smoothing) uses this approach for non-parametric regression — a computationally expensive but highly flexible alternative to polynomial regression.

---

## Links

- [[linear-algebra-fundamentals]] — OLS solution is $\hat\beta = (X^\top X)^{-1}X^\top y$; the normal equations require matrix inversion, which is $O(d^3)$ and numerically unstable when $X^\top X$ is ill-conditioned
- [[loss-mse]] — linear regression minimizes MSE; the squared loss is convex and differentiable, leading to the closed-form OLS solution
- [[statistical-inference-mle]] — OLS is equivalent to MLE under Gaussian noise $y = X\beta + \epsilon$, $\epsilon \sim \mathcal{N}(0,\sigma^2)$; the MLE for $\beta$ equals the OLS estimator
- [[regularization-weight-decay]] — ridge regression adds $\lambda\|\beta\|^2$ (L2) to the MSE; LASSO adds $\lambda\|\beta\|_1$ (L1) for sparsity; both shrink the OLS estimate
- [[bias-variance-double-descent]] — OLS has zero bias but high variance for large $d$; ridge reduces variance at the cost of bias; the tradeoff is the classical bias-variance picture
- [[numerical-methods]] — computing $(X^\top X)^{-1}$ via Cholesky decomposition is preferred over direct inversion; iterative solvers (conjugate gradient) scale to large $d$
- [[logistic-regression]] — logistic regression replaces the MSE with cross-entropy and passes the linear predictor through a sigmoid; it is the classification analogue of linear regression
