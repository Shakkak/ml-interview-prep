---
title: Sufficient Statistics
tags: [sufficient-statistics, fisher-neyman, minimal-sufficient, completeness, ancillary]
aliases: [sufficiency, sufficient statistic, minimal sufficient statistic, Fisher-Neyman factorization]
difficulty: 2
status: complete
related: [statistical-inference-mle, exponential-family, bayesian-inference, fisher-information, distributions-overview]
---

# Sufficient Statistics

---

## Fundamental

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

*See also: [[statistical-inference-mle]] · [[exponential-family]] · [[fisher-information]] · [[bayesian-inference]]*
