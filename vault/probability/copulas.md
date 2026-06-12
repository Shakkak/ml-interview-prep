---
title: Copulas
tags: [copulas, dependency-modeling, joint-distributions, gaussian-copula, tail-dependence]
aliases: [copula, Gaussian copula, Archimedean copula, Sklar's theorem, tail dependence]
difficulty: 2
status: complete
related: [distributions-overview, distributions-gaussian, monte-carlo-methods, bayesian-inference, change-of-variables]
---

# Copulas

---

## Fundamental

### Separating Marginals from Dependence

A joint distribution $F(x_1, x_2)$ mixes two things: the marginal distributions $F_1(x_1)$, $F_2(x_2)$, and the **dependence structure** between $x_1$ and $x_2$. Copulas separate these.

**Sklar's Theorem (1959):** every joint CDF $F$ with marginals $F_1, \ldots, F_d$ can be written as:

$$F(x_1, \ldots, x_d) = C(F_1(x_1), \ldots, F_d(x_d))$$

where $C: [0,1]^d \to [0,1]$ is a **copula** — a joint CDF on uniform marginals. $C$ is unique if the marginals are continuous.

**Inverse:** given any copula $C$ and any marginals $F_1, \ldots, F_d$, their combination is a valid joint distribution.

This means: any marginal distributions + any dependence structure (copula) → valid joint distribution.

---

## Intermediate

### Gaussian Copula

The **Gaussian copula** uses the dependence structure of a multivariate Gaussian with correlation matrix $\Sigma$:

$$C_\Sigma(u_1, \ldots, u_d) = \Phi_\Sigma(\Phi^{-1}(u_1), \ldots, \Phi^{-1}(u_d))$$

where $\Phi^{-1}$ is the standard normal quantile function and $\Phi_\Sigma$ is the multivariate normal CDF.

**Sampling:**
1. Sample $z \sim \mathcal{N}(0, \Sigma)$
2. Transform each $u_i = \Phi(z_i)$ — uniform marginals
3. Transform to desired marginals via $x_i = F_i^{-1}(u_i)$

The Gaussian copula captures elliptical (linear) dependence. It was notoriously used in CDO pricing in the 2008 financial crisis — underestimating tail dependence of mortgage defaults.

### Archimedean Copulas

**Archimedean copulas** are parameterized by a single generator function $\phi$:

$$C(u_1, \ldots, u_d) = \phi^{-1}(\phi(u_1) + \cdots + \phi(u_d))$$

**Common families:**

| Name | Generator $\phi(t)$ | Dependence |
|------|-------------------|-----------|
| Clayton | $t^{-\theta} - 1$ | Lower tail dependence |
| Gumbel | $(-\log t)^\theta$ | Upper tail dependence |
| Frank | $-\log\frac{e^{-\theta t}-1}{e^{-\theta}-1}$ | Symmetric, no tail dependence |

**Tail dependence:** the Clayton copula places more probability mass in the lower-left corner (both variables simultaneously very low). The Gumbel copula places mass in the upper-right corner. Gaussian copula has zero tail dependence — extreme co-movements are underestimated.

### Kendall's $\tau$ and Spearman's $\rho$

Copula-based dependence measures are invariant to monotone transformations of marginals:

**Kendall's $\tau$:** probability of concordant minus discordant pairs:
$$\tau = P[(X_1 - X_2)(Y_1 - Y_2) > 0] - P[(X_1 - X_2)(Y_1 - Y_2) < 0]$$

**Spearman's $\rho$:** correlation of the probability integral transforms (ranks).

Both depend only on the copula (dependence structure), not the marginals. Pearson correlation depends on both.

---

## Advanced

### Vine Copulas

High-dimensional copulas are hard to parameterize. **Vine (pair-copula) constructions** decompose the joint density into a product of bivariate copulas:

$$f(x_1, \ldots, x_d) = \prod_{k=1}^{d-1}\prod_{e \in \mathcal{E}_k} c_{j(e),k(e)|D(e)}$$

where each term is a bivariate copula density conditioning on a set $D(e)$. The vine structure (C-vine, D-vine, R-vine) specifies which pairs to model at each level. Allows flexible high-dimensional modeling by decomposing into tractable 2D pieces.

### Applications in ML

**Copula models for tabular data:** CTGAN and similar synthetic data generators can use copulas to model feature correlations with arbitrary marginals.

**Risk modeling:** joint distribution of asset returns — Gaussian marginals + Clayton copula captures crash dependence better than multivariate Gaussian.

**Multi-task learning:** copulas model task correlations when tasks have different output distributions.

*See also: [[distributions-overview]] · [[monte-carlo-methods]] · [[change-of-variables]] · [[bayesian-inference]]*
