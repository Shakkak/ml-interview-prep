---
title: Generating Functions and Moment Theory
tags: [generating-functions, moment-generating-function, characteristic-function, cumulants, mgf]
aliases: [moment generating function, MGF, characteristic function, cumulant generating function, cumulants]
difficulty: 2
status: complete
depends_on: [distributions-overview, matrix-calculus]
related: [distributions-overview, distributions-gaussian, central-limit-theorem, fisher-information, exponential-family]
---

# Generating Functions and Moment Theory

---

## Fundamental

**Moments** summarize a distribution by its expected powers: the first moment is the mean, the second gives variance, the third measures skewness. A **moment generating function (MGF)** encodes all moments in a single function — differentiating it at zero gives any moment directly. MGFs are why the Central Limit Theorem works: the MGF of a sum of independent variables is the product of their individual MGFs, which converges to the Gaussian MGF as terms accumulate.

### Moments of a Distribution

The **$k$-th moment** of a random variable $X$ is $\mu_k = \mathbb{E}[X^k]$. Moments characterize the shape of a distribution:
- $\mu_1 = \mathbb{E}[X]$: mean
- $\mu_2 - \mu_1^2 = \text{Var}(X)$: variance (centered 2nd moment)
- Centered 3rd moment / $\sigma^3$: skewness
- Centered 4th moment / $\sigma^4 - 3$: excess kurtosis

### Moment Generating Function (MGF)

$$M_X(t) = \mathbb{E}[e^{tX}] = \sum_{k=0}^\infty \frac{\mu_k}{k!} t^k$$

The MGF encodes all moments: $\mu_k = M_X^{(k)}(0) = \left.\frac{d^k M_X}{dt^k}\right|_{t=0}$.

**Uniqueness:** if the MGF exists in a neighborhood of $t = 0$, it uniquely determines the distribution.

**Convolution:** for independent $X, Y$: $M_{X+Y}(t) = M_X(t) M_Y(t)$ — products of MGFs correspond to convolution of distributions.

| Distribution | MGF $M(t)$ |
|---|---|
| Normal$(\mu, \sigma^2)$ | $e^{\mu t + \sigma^2 t^2/2}$ |
| Exponential$(\lambda)$ | $\lambda/(\lambda - t)$, $t < \lambda$ |
| Bernoulli$(p)$ | $1 - p + pe^t$ |
| Poisson$(\lambda)$ | $e^{\lambda(e^t - 1)}$ |

---

## Intermediate

### Characteristic Function

$$\varphi_X(t) = \mathbb{E}[e^{itX}] = M_X(it)$$

where $i = \sqrt{-1}$. The characteristic function always exists (no convergence issues — $|e^{itX}| = 1$).

**Fourier relationship:** $\varphi_X(t)$ is the Fourier transform of the PDF $f_X$:
$$\varphi_X(t) = \int_{-\infty}^\infty f_X(x) e^{itx} \, dx$$

**Central Limit Theorem via characteristic functions:** for iid $X_i$ with mean $\mu$, variance $\sigma^2$:
$$\varphi_{\bar{X}_n}(t) = \left(\varphi_X\!\left(\frac{t}{\sigma\sqrt{n}}\right)\right)^n \to e^{-t^2/2}$$

where $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ = sample mean of $n$ iid copies, $\varphi_X$ = characteristic function of a single $X_i$, $\sigma$ = standard deviation of $X_i$, and $e^{-t^2/2}$ = characteristic function of $\mathcal{N}(0,1)$ — the CLT in a few lines.

### Cumulants and Cumulant Generating Function

**Cumulant generating function (CGF):**
$$K_X(t) = \log M_X(t) = \log \mathbb{E}[e^{tX}]$$

The **cumulants** $\kappa_k = K_X^{(k)}(0)$ are more natural than moments:

| Cumulant | Meaning |
|----------|---------|
| $\kappa_1$ | Mean |
| $\kappa_2$ | Variance |
| $\kappa_3$ | Skewness (unnormalized) |
| $\kappa_4$ | Excess kurtosis (unnormalized) |

**Key property:** for independent $X, Y$: $K_{X+Y}(t) = K_X(t) + K_Y(t)$ — **cumulants add under independence**. Moments don't have this property (cross-terms appear).

**Gaussian uniqueness:** the Gaussian is the unique distribution with all cumulants $\kappa_k = 0$ for $k \geq 3$. Deviations from Gaussianity are measured by $\kappa_3$ (skewness) and $\kappa_4$ (kurtosis).

---

## Advanced

### Connection to Exponential Families

For exponential families (see [[exponential-family]]), the log-partition function $A(\eta)$ is the CGF of the sufficient statistic $T(x)$:
$$A(\eta) = K_{T(X)}(\eta) = \log \mathbb{E}[e^{\eta T(x)}]$$

where $\eta$ = natural parameter of the exponential family, $T(x)$ = sufficient statistic (the function of data that captures all information about $\eta$), $A(\eta)$ = log-partition function (a normalizing constant that makes the distribution sum to 1 — also the CGF of $T(X)$ evaluated at $\eta$).

Therefore: $\nabla A(\eta) = \mathbb{E}[T(X)]$ and $\nabla^2 A(\eta) = \text{Cov}[T(X)]$ — the gradient and Hessian of the log-partition are the mean and variance of the sufficient statistic.

### Laplace Transform

The **Laplace transform** $\mathcal{L}_X(s) = \mathbb{E}[e^{-sX}] = M_X(-s)$ for non-negative $X$. Used in:
- Reliability theory: transform of survival time distributions
- Queueing theory: transforms of service time distributions
- Differential equations: converting ODEs to algebraic equations

## Links

- [[distributions-overview]] — the MGF $M_X(t) = E[e^{tX}]$ encodes all moments of a distribution; different distributions have different MGF forms
- [[matrix-calculus]] — moments are computed by differentiating the MGF: $E[X^n] = M_X^{(n)}(0)$; this is a calculus operation on the generating function
- [[distributions-gaussian]] — the Gaussian MGF is $e^{\mu t + \sigma^2 t^2/2}$; the normal distribution is uniquely characterized by its cumulants (mean and variance, all higher cumulants zero)
- [[central-limit-theorem]] — the CLT is proved using characteristic functions (Fourier transform of the density); convergence of CFs implies convergence in distribution
- [[exponential-family]] — exponential family distributions have MGFs of the form $e^{A(\theta+t) - A(\theta)}$; the log-MGF $A(\theta)$ is the cumulant generating function
- [[fisher-information]] — Fisher information is the second derivative of the log-likelihood (log-MGF) at the natural parameter; cumulant generating functions connect the two
