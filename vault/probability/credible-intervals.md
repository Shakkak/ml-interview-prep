---
title: Credible Intervals and Confidence Intervals
tags: [credible-intervals, confidence-intervals, bayesian-inference, hpd, frequentist]
aliases: [credible interval, HPD interval, highest posterior density, Bayesian credible region, confidence interval]
difficulty: 1
status: complete
related: [bayesian-inference, hypothesis-testing, bootstrap, statistical-inference-mle, distributions-gaussian]
---

# Credible Intervals and Confidence Intervals

---

## Fundamental

### Two Different Philosophies

**Confidence interval (frequentist):** if we repeat the experiment many times, 95% of the intervals constructed this way will contain the true parameter. The parameter is fixed; the interval is random.

**Credible interval (Bayesian):** given the observed data, there is a 95% probability that the true parameter lies in this interval. The parameter is treated as random; the interval is a statement about posterior probability.

The 95% frequentist CI does **not** mean "95% probability the true value is in this interval" (a common misconception). The Bayesian credible interval does mean exactly that.

---

## Intermediate

### Computing Credible Intervals

Given posterior $p(\theta | x)$, a 95% credible interval is any interval $[a, b]$ such that $\int_a^b p(\theta | x) d\theta = 0.95$.

**Equal-tails interval:** place 2.5% probability in each tail:
$$a = F_\theta^{-1}(0.025), \quad b = F_\theta^{-1}(0.975)$$
where $F_\theta$ is the posterior CDF. Simple but not shortest possible.

**Highest Posterior Density (HPD) interval:** the shortest interval containing 95% probability — always includes the region of highest posterior density. For unimodal symmetric posteriors, equal-tails and HPD coincide. For skewed or multimodal posteriors, HPD is shorter.

**For conjugate models:** analytic. For example, Beta posterior → Beta quantiles. Normal posterior → Normal quantiles.

**For MCMC samples:** sort posterior samples, find the $(0.025N, 0.975N)$ quantile pair (equal-tails), or find the shortest window containing 95% of samples (HPD).

### When They Differ

For large samples, Bayesian credible intervals ≈ frequentist confidence intervals by the **Bernstein-von Mises theorem**: as $n \to \infty$, the posterior concentrates at the true parameter value and becomes approximately Gaussian — the credible interval and confidence interval converge.

They differ for:
- Small samples (prior matters)
- Nuisance parameter marginalization (Bayesian naturally marginalizes; frequentist profile likelihood is needed)
- Bounded parameters (e.g., probability $\in [0,1]$; normal approximation CIs can exceed bounds)

---

## Advanced

### Jeffreys Prior and Coverage

For a uniform prior, Bayesian credible intervals have exact frequentist coverage only in special cases. **Jeffreys prior** $p(\theta) \propto \sqrt{|I(\theta)|}$ (Fisher information) yields credible intervals with approximate frequentist coverage to second order — a useful non-informative prior.

### Prediction Intervals

Both frameworks distinguish **parameter intervals** from **prediction intervals** (for a new observation $x_\text{new}$):

**Frequentist prediction interval:** accounts for both estimation uncertainty and future sampling variability:
$$\hat{x} \pm t_{\alpha/2} \hat{\sigma}\sqrt{1 + 1/n}$$
The extra $+1$ captures future sampling variability.

**Bayesian predictive interval:** integrate over the posterior on parameters:
$$p(x_\text{new} | x_{1:n}) = \int p(x_\text{new} | \theta) p(\theta | x_{1:n}) d\theta$$
The credible interval on this predictive distribution naturally captures all sources of uncertainty.

*See also: [[bayesian-inference]] · [[hypothesis-testing]] · [[bootstrap]] · [[statistical-inference-mle]]*
