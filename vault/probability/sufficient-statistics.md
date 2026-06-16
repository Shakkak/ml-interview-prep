---
title: Sufficient Statistics
tags: [sufficient-statistics, fisher-neyman, minimal-sufficient, completeness, ancillary]
aliases: [sufficiency, sufficient statistic, minimal sufficient statistic, Fisher-Neyman factorization]
difficulty: 2
status: complete
related: [statistical-inference-mle, exponential-family, bayesian-inference, fisher-information, distributions-overview]
depends_on: [statistical-inference-mle, exponential-family, fisher-information]
---

# Sufficient Statistics

---

## Fundamental

When estimating a parameter from data, you usually don't need all the raw data — some compressed summary captures everything. A **sufficient statistic** is exactly that: a function of the data that retains all the information about the parameter. For example, to estimate the mean of a Gaussian you only need the sample mean, not all $n$ individual values. Sufficiency explains why MLE estimators in the exponential family have elegant closed forms.

### Definition

A statistic $T(X)$ is **sufficient** for parameter $\theta$ if the conditional distribution of $X$ given $T(X)$ does not depend on $\theta$:

$$p(X = x \mid T(X) = t, \theta) = p(X = x \mid T(X) = t) \quad \text{for all } \theta$$

**Intuition:** $T(X)$ captures everything the data tells you about $\theta$. Given $T$, you can throw away the raw data $X$ without losing any information for inference about $\theta$.

**Fisher-Neyman Factorization Theorem:** $T(X)$ is sufficient for $\theta$ if and only if the joint density factorizes as:

$$p(x; \theta) = g(T(x), \theta) \cdot h(x)$$

where $g$ depends on $x$ only through $T(x)$, and $h(x)$ doesn't depend on $\theta$.

### Examples

| Model | Sufficient statistic $T(X)$ |
|-------|---------------------------|
| Bernoulli($p$), $n$ samples | $\sum_{i=1}^n X_i$ (number of successes) |
| Normal($\mu$, $\sigma^2$ known), $n$ samples | $\bar{X} = \frac{1}{n}\sum X_i$ |
| Normal($\mu$, $\sigma^2$ unknown) | $(\bar{X}, S^2)$ — sample mean and variance |
| Poisson($\lambda$) | $\sum X_i$ |
| Uniform($0, \theta$) | $X_{(n)} = \max(X_1, \ldots, X_n)$ |

---

## Intermediate

### Minimal Sufficient Statistics

A sufficient statistic is **minimal sufficient** if it is a function of every other sufficient statistic — the maximum data compression that preserves information about $\theta$.

**Lehmann-Scheffé theorem:** $T(X)$ is minimal sufficient if, for two samples $x$ and $y$:
$$\frac{p(x; \theta)}{p(y; \theta)} \text{ is constant in } \theta \iff T(x) = T(y)$$

For exponential families (see [[exponential-family]]), the natural sufficient statistic $T(x) = \sum_i T(x_i)$ is minimal sufficient.

### Completeness and Unbiasedness

A sufficient statistic $T$ is **complete** if no non-constant function $g(T)$ has zero expectation for all $\theta$:
$$\mathbb{E}_\theta[g(T)] = 0 \text{ for all } \theta \implies g \equiv 0$$

**Rao-Blackwell Theorem:** given an unbiased estimator $\hat{\theta}$ and sufficient statistic $T$, the estimator $\mathbb{E}[\hat{\theta} | T]$ is:
- Unbiased for $\theta$
- Has variance $\leq \text{Var}(\hat{\theta})$ (always improves or ties)

**Lehmann-Scheffé Theorem:** if a complete sufficient statistic exists, the Rao-Blackwellized estimator using it is the **UMVUE** (Uniformly Minimum Variance Unbiased Estimator).

---

## Advanced

### Ancillary Statistics

An **ancillary statistic** $A(X)$ has a distribution that doesn't depend on $\theta$ at all — it contains no information about $\theta$.

**Conditionality principle:** inference should be conditional on the observed value of any ancillary statistic. The ancillary conveys information about "which experiment was effectively performed" but not about $\theta$.

**Example:** if sample size is random (say, a Poisson number of observations), the realized sample size $n$ is ancillary — condition on it for inference.

### Connection to Fisher Information

The Fisher information in a sufficient statistic equals the Fisher information in the full data:

$$I_T(\theta) = I_X(\theta)$$

This is equivalent to sufficiency — a sufficient statistic preserves all the Fisher information about $\theta$ in the data. Reducing data to $T(X)$ loses no statistical efficiency.

## Links

- [[statistical-inference-mle]] — the MLE depends only on the sufficient statistic $T(x)$; for exponential families, the MLE satisfies $E_\theta[T(x)] = T(x_{\text{obs}})$
- [[exponential-family]] — exponential families are exactly the distributions with finite-dimensional sufficient statistics (Pitman-Koopman-Darmois theorem); $T(x)$ is the natural sufficient statistic
- [[fisher-information]] — Fisher-Neyman factorization: $T(X)$ is sufficient iff $I(T(X);\theta) = I(X;\theta)$ — no Fisher information is lost by reducing to $T$
- [[bayesian-inference]] — in Bayesian inference with conjugate priors, the posterior depends on data only through the sufficient statistic; this makes conjugate updating tractable
- [[distributions-overview]] — each standard distribution has a natural sufficient statistic: Bernoulli → $\sum x_i$, Gaussian → $(\sum x_i, \sum x_i^2)$, Poisson → $\sum x_i$
