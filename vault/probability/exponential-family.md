---
title: Exponential Family Distributions
tags: [exponential-family, sufficient-statistics, natural-parameters, conjugate-priors, information-geometry]
aliases: [exponential family, natural parameters, sufficient statistics, GLM, log-partition function]
difficulty: 2
status: complete
related: [distributions-overview, distributions-gaussian, statistical-inference-mle, bayesian-inference, fisher-information, maximum-entropy-principle]
---

# Exponential Family Distributions

---

## Fundamental

The **exponential family** is a large class of distributions — Gaussian, Bernoulli, Poisson, Gamma, Beta, and many more — that all share the same mathematical form. This shared structure means one unified theory covers all of them: MLE always reduces to matching sufficient statistics, conjugate priors always exist, and the natural gradient has a clean form. Understanding the exponential family is the key to understanding why so many ML algorithms have elegant closed-form solutions.

### Definition

A distribution $p(x \mid \eta)$ belongs to the **exponential family** if it can be written:

$$p(x \mid \eta) = h(x) \exp\!\left(\eta^\top T(x) - A(\eta)\right)$$

- $\eta \in \mathbb{R}^k$ — **natural parameters** (canonical parameters)
- $T(x) \in \mathbb{R}^k$ — **sufficient statistics**
- $h(x)$ — base measure (does not depend on $\eta$)
- $A(\eta) = \log \int h(x) \exp(\eta^\top T(x)) \, dx$ — **log-partition function** (normalizer)

### Member Distributions

| Distribution | $T(x)$ | $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Bernoulli($p$) | $x$ | $\log(p/(1-p))$ | $\log(1+e^\eta)$ |
| Gaussian($\mu, \sigma^2$) | $(x, x^2)$ | $(\mu/\sigma^2, -1/(2\sigma^2))$ | $-\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$ |
| Poisson($\lambda$) | $x$ | $\log\lambda$ | $e^\eta$ |
| Categorical($\pi$) | one-hot$(x)$ | $\log\pi$ | $\log\sum_k e^{\eta_k}$ |
| Beta($\alpha,\beta$) | $(\log x, \log(1-x))$ | $(\alpha-1, \beta-1)$ | $\log B(\alpha,\beta)$ |
| Dirichlet($\alpha$) | $\log x_k$ | $\alpha_k - 1$ | $\log B(\alpha)$ |

Almost every named distribution used in ML belongs to the exponential family.

---

## Intermediate

### Key Properties

**Sufficiency:** $T(x)$ is a sufficient statistic — $p(x \mid T(x), \eta)$ does not depend on $\eta$. Knowing $T(x)$ captures everything the data tells us about $\eta$. For $N$ i.i.d. samples, the sufficient statistic is $\sum_i T(x_i)$ — a fixed-dimensional summary regardless of $N$.

**Moments from $A(\eta)$:**
$$\mathbb{E}[T(x)] = \nabla_\eta A(\eta), \qquad \text{Cov}[T(x)] = \nabla_\eta^2 A(\eta)$$

The log-partition function is convex, and its gradient/Hessian give mean and covariance of the sufficient statistics directly.

**MLE for exponential families:** the MLE $\hat{\eta}$ satisfies the **moment matching** condition:

$$\nabla_\eta A(\hat{\eta}) = \frac{1}{N}\sum_i T(x_i)$$

Empirical means of sufficient statistics equal their model expectations — a clean closed-form condition.

### Conjugate Priors

For any exponential family likelihood, there exists a **conjugate prior** (also in the exponential family) that yields a posterior in the same family:

| Likelihood | Conjugate Prior |
|------------|----------------|
| Bernoulli / Binomial | Beta |
| Multinomial | Dirichlet |
| Gaussian (known $\sigma^2$) | Gaussian |
| Gaussian (unknown $\mu, \sigma^2$) | Normal-Inverse-Gamma |
| Poisson | Gamma |

**Bayesian update is just adding hyperparameters:** for a Dirichlet-Multinomial model with prior $\text{Dir}(\alpha)$, observing counts $n_k$ gives posterior $\text{Dir}(\alpha + n)$. The sufficient statistic accumulates.

---

## Advanced

### Connection to Maximum Entropy

The exponential family arises naturally from the **maximum entropy principle** (see [[maximum-entropy-principle]]): the distribution with maximum entropy subject to fixed expectation constraints $\mathbb{E}[T_j(x)] = \mu_j$ is an exponential family with $T(x)$ as sufficient statistics. Each constraint adds one natural parameter $\eta_j$.

This gives a variational characterization: exponential families are the "least informative" distributions consistent with given moment constraints.

### Generalized Linear Models

**Generalized linear models (GLMs)** extend linear regression to exponential family outputs:
1. Linear predictor: $\eta = \mathbf{w}^\top \mathbf{x}$
2. Mean via inverse link: $\mu = \nabla A(\eta)$
3. Likelihood: $p(y \mid \mathbf{x}, \mathbf{w})$ from the exponential family

| Link function | Output distribution | Model |
|---|---|---|
| Logistic $\sigma(\eta)$ | Bernoulli | Logistic regression |
| Identity | Gaussian | Linear regression |
| Exponential $e^\eta$ | Poisson | Poisson regression |
| Softmax | Multinomial | Softmax regression |

### Natural Gradient

The **natural gradient** uses the Fisher information matrix $F(\eta) = \nabla^2 A(\eta)$ as the Riemannian metric. For exponential families, $F = \text{Cov}[T(x)]$. The natural gradient update $\tilde{\nabla} = F^{-1} \nabla$ is invariant to reparameterization and converges faster than ordinary gradient descent in parameter space (see [[fisher-information]]).

*See also: [[distributions-overview]] · [[bayesian-inference]] · [[statistical-inference-mle]] · [[maximum-entropy-principle]] · [[fisher-information]]*
