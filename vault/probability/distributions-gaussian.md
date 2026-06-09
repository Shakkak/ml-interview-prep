---
title: The Gaussian Distribution
tags: [probability, distributions, statistics, gaussian]
aliases: [normal distribution, Gaussian, bell curve, multivariate Gaussian]
difficulty: 2
status: complete
related: [distributions-overview, maximum-entropy-principle, bayesian-inference, math-svd]
---

# The Gaussian Distribution

---

## Fundamental

### Univariate Gaussian

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- Mean: $\mu$, Variance: $\sigma^2$, Standard deviation: $\sigma$
- Entropy: $\frac{1}{2}\ln(2\pi e\sigma^2)$ nats
- The [[maximum-entropy-principle|maximum-entropy]] distribution given fixed mean and variance

### 68-95-99.7 Rule

| Range | Probability |
|-------|-------------|
| $\mu \pm 1\sigma$ | 68.27% |
| $\mu \pm 2\sigma$ | 95.45% |
| $\mu \pm 3\sigma$ | 99.73% |

Numerically: $P(|Z| \leq 1) = 2\Phi(1) - 1 = 2(0.8413) - 1 = 0.6827$ where $\Phi$ is the standard normal CDF. Standardization: $Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)$.

### Why the Gaussian Is Everywhere

1. **Central Limit Theorem:** sums of $n$ i.i.d. random variables with finite variance converge to Gaussian as $n \to \infty$, regardless of the underlying distribution — additive noise processes converge to Gaussian (see [[sampling-methods]] for how this motivates Gaussian proposals in MCMC).
2. **Maximum entropy:** under only a mean and variance constraint, the Gaussian is the least-informative distribution — the principled default.
3. **Closure:** sums, linear transforms, and conditioning all stay Gaussian — algebraically convenient.

### MLE for Gaussian

Log-likelihood for $n$ samples:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2$$

Setting $\partial\ell/\partial\mu = 0$: $\hat{\mu} = \bar{x}$

Setting $\partial\ell/\partial\sigma^2 = 0$: $\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum(x_i - \bar{x})^2$

Note: $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$ (biased). Unbiased: divide by $n-1$ (Bessel's correction).

**Worked example:** Data $x = \{1, 3, 5, 7, 9\}$ ($n=5$). $\hat{\mu} = 5$, $\hat{\sigma}^2_{MLE} = 8$, unbiased $s^2 = 10$. 95% range $\approx [5 \pm 5.66] = [-0.66, 10.66]$; all 5 points fall within.

---

## Intermediate

### Closure Properties

**Sum of independent Gaussians:**
$X \sim \mathcal{N}(\mu_1, \sigma_1^2)$, $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ independent: $X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$. This fails when $X$ and $Y$ are correlated (covariance term contributes).

**Linear transform:** $Y = aX + b \Rightarrow Y \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$.

**Product of Gaussian PDFs** (key for [[bayesian-inference|Bayesian]] updates):
$p_1(x) \propto \mathcal{N}(x|\mu_1, \sigma_1^2)$, $p_2(x) \propto \mathcal{N}(x|\mu_2, \sigma_2^2)$:

$$p_1(x)p_2(x) \propto \mathcal{N}(x | \mu^*, \sigma^{*2})$$

where $\frac{1}{\sigma^{*2}} = \frac{1}{\sigma_1^2} + \frac{1}{\sigma_2^2}$ and $\frac{\mu^*}{\sigma^{*2}} = \frac{\mu_1}{\sigma_1^2} + \frac{\mu_2}{\sigma_2^2}$ (precision-weighted average). This is the Bayesian posterior for a Gaussian prior times a Gaussian likelihood — the posterior precision is the sum of prior and likelihood precisions.

### Multivariate Gaussian

$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\!\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right)$$

- $\mu \in \mathbb{R}^d$: mean vector
- $\Sigma \in \mathbb{R}^{d \times d}$: positive semi-definite covariance matrix (see [[linear-algebra-fundamentals]] for Cholesky sampling: $x = \mu + Lz$, $z \sim \mathcal{N}(0,I)$, $\Sigma = LL^\top$)
- Entropy: $\frac{1}{2}\ln\left((2\pi e)^d |\Sigma|\right)$

> [!tip] Why $x = \mu + Lz$ samples from $\mathcal{N}(\mu, \Sigma)$ ([[linear-algebra-fundamentals]])
> Every positive semi-definite $\Sigma$ has a Cholesky factorisation $\Sigma = LL^\top$ with $L$ lower-triangular.
> For $z \sim \mathcal{N}(0, I)$:
> $\mathbb{E}[Lz] = L\,\mathbb{E}[z] = 0$, so the mean of $\mu + Lz$ is $\mu$.
> $\text{Var}(Lz) = L\,\text{Var}(z)\,L^\top = L I L^\top = LL^\top = \Sigma$.
> Since any affine transform of a Gaussian is Gaussian, $\mu + Lz \sim \mathcal{N}(\mu, \Sigma)$ exactly.
> In practice: `L = torch.linalg.cholesky(Sigma); x = mu + L @ torch.randn(d)`.

| $\Sigma$ shape | Meaning | Density contours |
|---------------|---------|-----------------|
| $\sigma^2 I$ | Isotropic (spherical) | Circles |
| Diagonal | Independent, different variances | Axis-aligned ellipses |
| Full | Correlated | Rotated ellipses |

### Mahalanobis Distance

$$D_M(x) = \sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}$$

Measures how far $x$ is from $\mu$ in units of the distribution's spread, accounting for correlations. **Example:** $\mu = [0,0]$, $\Sigma = \text{diag}(4, 1)$, $x = [2, 1]$. Euclidean: $\sqrt{5} \approx 2.24$. Mahalanobis: $\sqrt{1 + 1} = \sqrt{2} \approx 1.41$. The point is 1 standard deviation in each direction — Mahalanobis correctly shows this is moderate; Euclidean over-penalizes the low-variance axis.

### Marginal and Conditional Distributions

For joint Gaussian with $x = [x_1, x_2]^\top$, $\mu = [\mu_1, \mu_2]^\top$, $\Sigma = [[\Sigma_{11}, \Sigma_{12}],[\Sigma_{21}, \Sigma_{22}]]$:

**Marginal** of $x_1$: $x_1 \sim \mathcal{N}(\mu_1, \Sigma_{11})$ — take the relevant block.

**Conditional** $x_1 | x_2$:

$$x_1|x_2 \sim \mathcal{N}\!\left(\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2),\; \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)$$

The term $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is the **Schur complement** — the variance that remains after observing $x_2$.

**Example:** $\mu=[0,0]$, $\Sigma=[[1, 0.8],[0.8, 1]]$, observe $x_2 = 1$. $\mu_{1|2} = 0 + 0.8(1) = 0.8$, $\sigma^2_{1|2} = 1 - 0.64 = 0.36$. So $x_1|x_2=1 \sim \mathcal{N}(0.8, 0.36)$.

---

## Advanced

### Gaussian Processes

A GP is an infinite-dimensional Gaussian — a distribution over functions. Any finite collection of function values $[f(x_1),\ldots,f(x_n)]$ is jointly Gaussian. Formally: $f \sim \mathcal{GP}(m, k)$ where $m(x)$ is the mean function and $k(x, x')$ is the covariance (kernel) function.

**Posterior GP:** Given observations $\mathbf{y} = f(X) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_n^2 I)$:

$$f_* | X_*, X, \mathbf{y} \sim \mathcal{N}(K_{*,X}(K_{X,X} + \sigma_n^2 I)^{-1}\mathbf{y},\; K_{*,*} - K_{*,X}(K_{X,X}+\sigma_n^2 I)^{-1}K_{X,*})$$

Exact inference requires $O(n^3)$ for the [[linear-algebra-fundamentals|Cholesky]] of $K_{X,X}$. Sparse GP approximations (inducing points, Titsias 2009) reduce this to $O(nm^2)$ with $m$ inducing points.

### Information Geometry of the Gaussian Family

The Fisher information matrix for $\mathcal{N}(\mu, \sigma^2)$ with parameters $(\mu, \sigma^2)$ is:

$$F = \begin{bmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{bmatrix}$$

The KL divergence between two Gaussians has a closed form:

$$D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_2-\mu_1)^2}{\sigma_2^2} - 1 + \ln\frac{\sigma_2^2}{\sigma_1^2}\right]$$

The multivariate version: $D_{KL}(\mathcal{N}(\mu_1,\Sigma_1) \| \mathcal{N}(\mu_2,\Sigma_2)) = \frac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^\top\Sigma_2^{-1}(\mu_2-\mu_1) - d + \ln\frac{|\Sigma_2|}{|\Sigma_1|}\right]$.

This is the formula used in [[variational-autoencoders|VAE]] KL penalty terms.

### Stein's Paradox and Shrinkage

For $d \geq 3$, the MLE $\hat{\mu} = \bar{x}$ is inadmissible — the James-Stein estimator $\hat{\mu}_{JS} = \left(1 - \frac{(d-2)\sigma^2}{n\|\bar{x}\|^2}\right)\bar{x}$ has strictly lower expected MSE for all $\mu$. This counterintuitive result (Stein 1956) demonstrates that shrinking toward any fixed point improves total MSE in high dimensions, regardless of the true mean. It provides theoretical grounding for why L2 regularization works: biased estimators with lower variance can dominate unbiased ones.

### Non-Gaussian Generalizations

- **Student-t:** heavier tails, parameterized by degrees of freedom $\nu$; approaches Gaussian as $\nu \to \infty$. Robust likelihood for regression with outliers; used in t-SNE's low-dimensional kernel.
- **Skew-normal:** introduces asymmetry parameter $\alpha$; arises in truncated Gaussian mixtures.
- **Multivariate $t$-distribution:** all marginals and conditionals are also $t$-distributed. Used in robust Bayesian regression.

---

*See also: [[distributions-overview]] · [[maximum-entropy-principle]] · [[bayesian-inference]] · [[math-svd]] · [[linear-algebra-fundamentals]] · [[variational-autoencoders]] · [[lora-quantization]]*
