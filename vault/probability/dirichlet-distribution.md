---
title: Dirichlet and Beta Distributions
tags: [dirichlet, beta-distribution, conjugate-prior, bayesian, concentration-parameter]
aliases: [Dirichlet distribution, Beta distribution, Dirichlet process, concentration parameter]
difficulty: 1
status: complete
related: [bayesian-inference, distributions-overview, exponential-family, gaussian-mixture-models, variational-inference]
depends_on: [distributions-overview, bayesian-inference, exponential-family]
---

# Dirichlet and Beta Distributions

---

## Fundamental

### Beta Distribution

The **Beta distribution** $\text{Beta}(\alpha, \beta)$ is a distribution over $[0, 1]$ — the natural prior for probabilities:

$$p(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

**Parameters:**
- $\alpha, \beta > 0$ — **pseudo-counts** (shape parameters)
- $\alpha = \beta = 1$: uniform on $[0,1]$
- $\alpha = \beta > 1$: peaked at 0.5 (certainty increases with $\alpha$)
- $\alpha > \beta$: skewed toward 1 (prior belief in higher probability)

**Mean:** $\mu = \alpha/(\alpha + \beta)$, **Mode:** $(\alpha-1)/(\alpha+\beta-2)$ for $\alpha, \beta > 1$

### Dirichlet Distribution

The **Dirichlet distribution** $\text{Dir}(\alpha)$ is a distribution over probability simplices — vectors $\pi \in \mathbb{R}^K$ with $\pi_k \geq 0$, $\sum_k \pi_k = 1$:

$$p(\pi \mid \alpha) = \frac{1}{B(\alpha)} \prod_{k=1}^K \pi_k^{\alpha_k - 1}$$

It is the multivariate generalization of the Beta: $\text{Beta}(\alpha, \beta) = \text{Dir}(\alpha, \beta)$ for $K=2$.

**Concentration parameter:** $\alpha_0 = \sum_k \alpha_k$ controls the spread:
- Small $\alpha_0$ (< 1): sparse, concentrated on corners of simplex (one $\pi_k \approx 1$)
- $\alpha_0 = K$: roughly uniform
- Large $\alpha_0$: concentrated near $\alpha/\alpha_0$ — low entropy, near-deterministic

---

## Intermediate

### Conjugate Prior for Categorical / Multinomial

**Bayesian update:** prior $\text{Dir}(\alpha)$ + multinomial likelihood with counts $n_k$ → posterior $\text{Dir}(\alpha + n)$. Just add observations to the hyperparameters.

**Predictive distribution:** the expected probability of category $k$ under the posterior is:
$$p(x = k) = \frac{\alpha_k + n_k}{\alpha_0 + N}$$

where $\alpha_k$ = prior pseudo-count for category $k$, $n_k$ = observed count of category $k$, $\alpha_0 = \sum_k \alpha_k$ = total prior strength, and $N = \sum_k n_k$ = total observations.

This is the **Laplace smoothing** formula when $\alpha_k = 1$. Dirichlet-Multinomial is the Bayesian foundation for smoothed language models.

**Beta-Bernoulli (coin flip):** observe $h$ heads, $t$ tails with prior $\text{Beta}(\alpha, \beta)$:
$$\text{posterior} = \text{Beta}(\alpha + h, \beta + t), \quad \mathbb{E}[p] = \frac{\alpha + h}{\alpha + h + \beta + t}$$

### Applications in ML

**Latent Dirichlet Allocation (LDA):** documents are mixtures of topics ($\pi_d \sim \text{Dir}(\alpha)$), topics are distributions over words ($\phi_k \sim \text{Dir}(\beta)$). The Dirichlet prior encourages sparse topic mixtures.

**Gaussian Mixture Models:** using a Dirichlet prior on mixture weights $\pi_k$ enables automatic component pruning in Variational Bayesian GMM — irrelevant components have $\pi_k \to 0$ under the posterior.

**Reinforcement learning:** Dirichlet prior on transition probabilities for Bayesian RL; posterior uncertainty can be used for exploration.

---

## Advanced

### Dirichlet Process

The **Dirichlet Process (DP)** $\text{DP}(\alpha_0, H)$ is a nonparametric extension: a distribution over countably infinite discrete distributions. A sample from a DP is itself a probability distribution.

**Stick-breaking construction:** imagine a stick of length 1. Break off a fraction $\beta_1 \sim \text{Beta}(1, \alpha_0)$ — assign this length to component 1. From the remaining stick, break off fraction $\beta_2$, assign to component 2, and so on:
$$\pi_k = \beta_k \prod_{j<k}(1 - \beta_j), \quad \beta_k \sim \text{Beta}(1, \alpha_0), \quad \theta_k \sim H$$

where $\pi_k$ = weight of component $k$ (the fraction of the stick it receives), $\beta_k$ = proportion broken at step $k$, $\prod_{j<k}(1-\beta_j)$ = remaining stick length before step $k$, $H$ = base measure (prior distribution from which component parameters $\theta_k$ are drawn), and $\alpha_0$ = concentration parameter (small: breaks stick quickly, few components dominate; large: many components get similar weight). As $\alpha_0 \to 0$, a single component dominates; as $\alpha_0 \to \infty$, components become equiprobable.

**Chinese Restaurant Process:** the generative story for DP mixture models — customer $n$ sits at an existing table $k$ with probability proportional to the number already there, or opens a new table with probability $\alpha_0 / (\alpha_0 + n - 1)$. Implements rich-get-richer dynamics.

Used in **Dirichlet Process Mixture Models** as an alternative to GMMs when the number of clusters is unknown — the DP automatically infers the number of clusters from data.

## Links

- [[bayesian-inference]] — the Dirichlet is the conjugate prior for the categorical/multinomial; the posterior is Dirichlet with updated concentration parameters $\alpha_k + n_k$
- [[distributions-overview]] — the Beta distribution is the 2-class special case of the Dirichlet; understanding common distributions motivates why Dirichlet is the natural prior over probability vectors
- [[exponential-family]] — the Dirichlet is an exponential family distribution with sufficient statistic $\log p_k$; natural parameters are $\alpha_k - 1$
- [[gaussian-mixture-models]] — the Dirichlet prior over mixing weights in a GMM enables Bayesian inference over cluster proportions; concentration $\alpha < 1$ encourages sparse mixtures
- [[variational-inference]] — mean-field VI for Dirichlet-Multinomial models uses the digamma function to compute expected sufficient statistics; the ELBO is tractable in closed form
