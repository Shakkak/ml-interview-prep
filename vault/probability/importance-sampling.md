---
title: Importance Sampling
tags: [importance-sampling, monte-carlo, variance-reduction, sampling, expectation]
aliases: [IS, importance weights, self-normalized IS, effective sample size]
difficulty: 2
status: complete
related: [sampling-methods, bayesian-inference, variational-inference, distributions-overview, statistical-inference-mle]
depends_on: [distributions-overview, bayesian-inference, monte-carlo-methods]
---

# Importance Sampling

---

## Fundamental

### The Problem

We want to compute $\mathbb{E}_{p}[f(x)] = \int f(x) p(x) \, dx$ but we cannot sample from $p$ directly (or $p$ is hard to evaluate, or samples would be wasted in regions where $f$ is small).

**Importance sampling (IS)** rewrites the expectation under a different distribution $q$ (the **proposal**):

$$\mathbb{E}_{p}[f(x)] = \int f(x) \frac{p(x)}{q(x)} q(x) \, dx = \mathbb{E}_{q}\left[f(x) \frac{p(x)}{q(x)}\right]$$

The ratio $w(x) = p(x)/q(x)$ is the **importance weight**. The estimator is:

$$\hat{\mu}_\text{IS} = \frac{1}{N} \sum_{i=1}^N f(x^{(i)}) w(x^{(i)}), \quad x^{(i)} \sim q$$

This is unbiased as long as $q(x) > 0$ wherever $p(x)f(x) \neq 0$.

### When IS Helps

IS is most useful when:
- Direct sampling from $p$ is impossible (e.g., unnormalized $p$)
- $p$ concentrates in a small region that naive Monte Carlo would rarely hit (rare events)
- We have samples from $q$ already (e.g., old policy in RL, off-policy evaluation)

---

## Intermediate

### Self-Normalized IS

In practice $p$ is often known only up to a normalizing constant: $p(x) = \tilde{p}(x)/Z$. We can't compute $w(x) = \tilde{p}(x)/(Zq(x))$ because $Z$ is unknown.

**Self-normalized IS (SNIS):**

$$\hat{\mu}_\text{SNIS} = \frac{\sum_i \tilde{w}(x^{(i)}) f(x^{(i)})}{\sum_i \tilde{w}(x^{(i)})}, \quad \tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}$$

SNIS is biased (ratio of two random variables) but consistent. The bias is $O(1/N)$ and usually negligible for large $N$.

### Effective Sample Size

Not all $N$ samples contribute equally — if a few have very large weights, the estimate is driven by those few. The **effective sample size (ESS)** quantifies this:

$$\text{ESS} = \frac{(\sum_i \tilde{w}_i)^2}{\sum_i \tilde{w}_i^2}$$

ESS ranges from 1 (one sample dominates) to $N$ (all weights equal). ESS $\ll N$ signals a poor proposal. Rule of thumb: ESS $< N/10$ means the IS estimate is unreliable.

**Weight collapse:** if $q$ has lighter tails than $p$, a few samples in the tails of $q$ receive exponentially large weights. The estimator has high variance or infinite variance. IS fails in high dimensions for this reason (curse of dimensionality).

### Choosing a Good Proposal

The **optimal proposal** that minimizes estimator variance for a given function $f$ is:

$$q^*(x) = \frac{|f(x)| p(x)}{\int |f(x')| p(x') \, dx'}$$

This is impractical (requires the answer), but it motivates designing $q$ to put mass where $|f(x)|p(x)$ is large.

**Practical guidelines:**
- $q$ should have heavier tails than $p$ (to avoid weight collapse)
- $q$ should be easy to sample from and evaluate
- For rare events: $q$ should be tilted toward the rare region

---

## Advanced

### IS in Machine Learning

**Off-policy RL:** the policy gradient theorem requires on-policy samples, but collected data came from an older policy $\mu$. IS corrects: $\nabla J(\pi) = \mathbb{E}_{\mu}[\frac{\pi(a|s)}{\mu(a|s)} \nabla \log \pi(a|s) Q^\pi(s,a)]$. Clipping the IS ratio (PPO) or bounding it (TRPO) prevents variance explosion.

**Variational inference:** the ELBO can be seen as an IS lower bound. IWAE (importance-weighted autoencoders) uses $K$ IS samples to tighten the bound: $\mathcal{L}_K = \mathbb{E}_{z^{(1:K)} \sim q}\left[\log \frac{1}{K}\sum_k \frac{p(x,z^{(k)})}{q(z^{(k)}|x)}\right] \geq \mathcal{L}_\text{ELBO}$. As $K \to \infty$, this recovers $\log p(x)$.

**Sequential IS (particle filter):** for time-series state estimation, propagate $N$ particles forward and reweight them by the likelihood of each new observation. Resample when ESS drops below a threshold.

### Annealed IS

Annealed importance sampling (AIS, Neal 2001) bridges $q$ to $p$ via a sequence of intermediate distributions $p_0 = q, p_1, \ldots, p_T = p$, running MCMC steps between them. Produces unbiased estimates of the normalizing constant $Z$ that are impossible with standard MCMC.

## Links

- [[distributions-overview]] — importance sampling reweights samples from a proposal $q$ to estimate expectations under a target $p$; the ratio $p(x)/q(x)$ corrects for the distributional mismatch
- [[bayesian-inference]] — IS is used for Bayesian model comparison (marginal likelihood estimation) and sequential Monte Carlo (particle filters) for online Bayesian inference
- [[monte-carlo-methods]] — importance sampling reduces Monte Carlo variance when the proposal $q$ concentrates on high-probability regions of $p$; optimal $q \propto |f(x)|p(x)$
- [[variational-inference]] — IS estimates the partition function for evidence lower bounds; self-normalized IS is an alternative to VI that is asymptotically unbiased
- [[sampling-methods]] — IS is a variance reduction technique for Monte Carlo; MCMC draws from $p$ directly but requires burn-in; IS reuses samples from a fixed proposal
