---
title: Parametric vs Non-Parametric Models
tags: [statistics, machine-learning, parametric, non-parametric, gaussian-process, knn]
aliases: [parametric, non-parametric, KNN, Gaussian process, kernel density estimation]
difficulty: 2
status: complete
related: [statistical-inference-mle, distributions-gaussian, kernel-methods, bias-variance-double-descent, curse-of-dimensionality]
depends_on: [statistical-inference-mle, bayesian-inference, kernel-methods]
---

# Parametric vs Non-Parametric Models

---

## Fundamental

**Parametric model:** a fixed number of parameters regardless of training data size. The model's complexity is set before training.

**Non-parametric model:** complexity grows with the data. More data → more "parameters" (the training set itself is the model).

| Property | Parametric | Non-parametric |
|---|---|---|
| \# parameters | Fixed ($d$) | Grows with $n$ |
| Memory at inference | $O(d)$ | $O(n)$ or $O(n^2)$ |
| Inference speed | Fast ($O(d)$) | Slow ($O(n)$ or worse) |
| Inductive bias | Strong (functional form assumed) | Weak (few assumptions) |
| Data efficiency | High (uses all data in few params) | Low (needs many examples) |
| Expressiveness | Limited by chosen form | Unlimited with enough data |

**Parametric examples:** linear/logistic regression ($d+1$ parameters), neural networks (fixed architecture), naive Bayes (per-class mean/variance).

**Non-parametric examples:** KNN (training set is the model), kernel density estimation, [[decision-trees|decision trees]] with no depth limit, Gaussian processes.

### K-Nearest Neighbors (KNN)

**Classification:** find the $k$ nearest training points to query $x$ (by Euclidean distance); predict majority class.

**Properties:** no training phase; inference cost $O(nd)$ per query; decision boundary can be arbitrarily complex.

**[[bias-variance-double-descent|Bias-variance]]:** small $k$ → low bias, high variance. Large $k$ → high bias, low variance. Optimal $k$ balances these via [[cross-validation]].

**[[curse-of-dimensionality|Curse of dimensionality]]:** in high dimensions all pairwise distances concentrate — KNN becomes meaningless without dimensionality reduction first.

### Kernel Density Estimation (KDE)

$$\hat{p}(x) = \frac{1}{n}\sum_{i=1}^n K_h(x - x_i)$$

where $K_h$ = kernel function (e.g. Gaussian) with bandwidth $h$, $x_i$ = $i$-th training point, $n$ = number of training points, $\hat{p}(x)$ = density estimate at query point $x$.

Each training point contributes a "bump" of probability. **Silverman's rule of thumb** for optimal bandwidth: $h = 1.06\,\hat\sigma\, n^{-1/5}$ (1D Gaussian kernel).

---

## Intermediate

### Gaussian Processes — Non-parametric Bayesian Models

A GP places a prior over functions: $f \sim \mathcal{GP}(\mu, k)$. Any finite collection of function values is jointly [[distributions-gaussian|Gaussian]]: $(f(x_1), \ldots, f(x_n)) \sim \mathcal{N}(\mu(X), K)$ where $K_{ij} = k(x_i, x_j)$.

**Posterior after observing $y = f(X) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$:**

$$\mu^* = k(x^*, X)(K + \sigma^2 I)^{-1}y$$
$$\sigma^{*2} = k(x^*, x^*) - k(x^*, X)(K + \sigma^2 I)^{-1}k(X, x^*)$$

where $k(x^*, X)$ = vector of covariances between test point $x^*$ and all training points, $K$ = $n \times n$ kernel (covariance) matrix on training set, $\sigma^2$ = observation noise variance, $y$ = training targets, $\mu^*$ = posterior predictive mean, $\sigma^{*2}$ = posterior predictive variance (uncertainty at $x^*$).

**Key property:** GPs give calibrated uncertainty estimates $\sigma^{*2}$, not just point predictions. The posterior variance $\sigma^{*2}$ is large far from training points (model is uncertain) and small near training points.

**Computational cost:** solving $(K + \sigma^2 I)^{-1}y$ requires Cholesky decomposition — $O(n^3)$ training, $O(n)$ prediction, $O(n^2)$ storage. Impractical beyond $n \approx 10^4$.

**Sparse GPs:** use $m \ll n$ inducing points to approximate the full GP posterior. Methods: FITC (fully independent training conditional), VFE (variational free energy), SVGP. Cost: $O(nm^2 + m^3)$.

### The Parametric-Non-parametric Spectrum

Not a binary — there is a spectrum:

| Model | Position |
|---|---|
| Linear regression | Fully parametric |
| Neural network (fixed arch) | Parametric |
| [[kernel-methods\|SVM with RBF kernel]] | Semi-parametric (support vectors grow with $n$) |
| Gaussian process | Non-parametric |
| KNN | Non-parametric |

**SVMs:** the number of support vectors (model size) grows with $n$ in the worst case, making SVMs technically semi-parametric. In practice, well-separated datasets have few support vectors.

### Convergence Rates

**Parametric models** (correctly specified): mean squared error $\propto 1/n$ regardless of dimension $d$ — the parametric rate.

**Non-parametric models:** KDE achieves MSE $\propto n^{-4/(4+d)}$ — the convergence rate degrades exponentially with dimension (curse of dimensionality). For $d=10$, you need $n^{14/4} = n^{3.5}$ more samples than for $d=1$ to achieve the same accuracy.

**When to use each:**
- Parametric: domain knowledge about functional form, fast inference required, large datasets (neural networks)
- Non-parametric: no good model, need calibrated uncertainty (GPs for Bayesian optimization), small structured datasets

---

## Advanced

### Neural Tangent Kernel: Neural Networks as GPs

For infinitely wide neural networks, the limiting behavior under gradient descent is governed by the **[[neural-tangent-kernel|Neural Tangent Kernel (NTK)]]**:

$$K_\text{NTK}(x, x') = \mathbb{E}_{\theta\sim\text{init}}\left[\nabla_\theta f(x;\theta)^\top \nabla_\theta f(x';\theta)\right]$$

where $\theta \sim \text{init}$ = parameters sampled from the random initialization distribution, $\nabla_\theta f(x;\theta) \in \mathbb{R}^{|\theta|}$ = gradient vector of the network output w.r.t. parameters at $x$, inner product = similarity in gradient space.

In the infinite-width limit: (1) the NTK is constant during training (doesn't change as weights update); (2) training dynamics are linear; (3) the network converges to the minimum-norm interpolating solution in the NTK-induced RKHS.

This means **infinitely wide neural networks are exactly equivalent to kernel regression with the NTK kernel** — a non-parametric method. Finite-width networks deviate from this limit (feature learning), which is precisely what makes them better in practice.

**Implications:** the NTK analysis explains spectral bias (the NTK eigenspectrum determines which frequencies are learned first), connects neural network generalization to RKHS theory, and provides a theoretical explanation for why neural networks with more parameters can achieve lower test error (more expressive RKHS).

### Semiparametric Models and Partial Identification

**Semiparametric models** impose some structure to escape the curse of dimensionality while retaining flexibility:

**Additive models:** $f(x) = \sum_{j=1}^d f_j(x_j)$ — each feature has its own nonlinear component $f_j$, but interactions are excluded. Each $f_j$ can be estimated at the univariate non-parametric rate $n^{-4/5}$ regardless of $d$.

**Single-index models:** $f(x) = g(w^\top x)$ — a nonlinear function of a linear projection. The parametric part ($w$) is estimated at rate $\sqrt{n}$; the nonparametric part ($g$) at univariate rate.

**Partially linear models:** $y = x_1^\top \beta + g(x_2) + \epsilon$ — linear in the first variables, nonparametric in the rest. Used in econometrics and causal inference to estimate the linear effect of a treatment $\beta$ while flexibly controlling for confounders $x_2$.

### Bayesian Nonparametrics

**Dirichlet Process (DP):** a prior over probability distributions with infinite mixture components. The DP-mixture model: $p(x) = \sum_{k=1}^\infty \pi_k \mathcal{N}(x;\mu_k, \Sigma_k)$, where the number of components grows with data.

**Chinese Restaurant Process** (CRP): the de Finetti exchangeable representation — customer $n+1$ sits at existing table $k$ with probability $\propto n_k/(n+\alpha)$ or a new table with probability $\propto \alpha/(n+\alpha)$. Expected number of tables (components) grows as $\alpha\log n$.

**Gaussian process classification:** use GP as a latent function prior, apply sigmoid to get class probabilities, perform approximate inference (Laplace approximation or variational) since the likelihood is non-Gaussian. Provides calibrated uncertainty with no fixed model capacity — the complexity of the decision boundary adapts to the data.

---

## Links

- [[statistical-inference-mle]] — parametric models specify a family $p(x|\theta)$; MLE finds the $\hat\theta$ maximizing the likelihood; non-parametric models make no distributional assumption
- [[bayesian-inference]] — Bayesian non-parametric models (Gaussian processes, Dirichlet processes) have infinitely many parameters that grow with data size; the prior governs smoothness
- [[kernel-methods]] — SVMs and kernel regression are non-parametric: the model complexity (number of support vectors) grows with training data; the kernel encodes the inductive bias
- [[distributions-gaussian]] — Gaussian process regression is non-parametric: predictions are mean and variance of an infinite-dimensional Gaussian; the kernel defines the covariance structure
- [[bias-variance-double-descent]] — parametric models control complexity via parameter count (higher bias, lower variance for small models); non-parametric models grow in complexity with data
- [[curse-of-dimensionality]] — non-parametric methods (KNN, kernel density estimation) suffer most from the curse; sample complexity grows exponentially with dimension
- [[decision-trees]] — decision trees are non-parametric: a tree with $n$ leaves can fit any data with $n$ distinct input patterns; pruning (regularization) controls effective complexity
