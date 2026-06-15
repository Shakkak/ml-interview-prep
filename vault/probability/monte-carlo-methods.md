---
title: Monte Carlo Methods
tags: [monte-carlo, mcmc, variance-reduction, integration, stochastic-estimation]
aliases: [Monte Carlo integration, MCMC, Markov chain Monte Carlo, variance reduction, control variates]
difficulty: 2
status: complete
related: [sampling-methods, importance-sampling, distributions-overview, bayesian-inference, variational-inference]
---

# Monte Carlo Methods

---

## Fundamental

Most integrals in probability and machine learning — computing expectations, normalizing constants, or marginal likelihoods — cannot be solved analytically. **Monte Carlo methods** sidestep this by using random samples: draw many samples from a distribution, average the function over those samples, and use the average as the estimate. With enough samples, this converges to the true answer regardless of the dimensionality of the problem — making Monte Carlo the workhorse of Bayesian inference and RL.

### The Monte Carlo Estimator

For any expectation $\mu = \mathbb{E}_{p}[f(X)] = \int f(x) p(x) dx$ that is intractable analytically, the **Monte Carlo estimator** approximates it with a sample average:

$$\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^N f(x^{(i)}), \quad x^{(i)} \overset{\text{iid}}{\sim} p$$

**Correctness:**
- **Unbiased:** $\mathbb{E}[\hat{\mu}_N] = \mu$
- **Consistent:** $\hat{\mu}_N \to \mu$ a.s. as $N \to \infty$ (Law of Large Numbers)
- **Convergence rate:** $\text{Var}[\hat{\mu}_N] = \text{Var}[f(X)] / N$ → standard error $\propto 1/\sqrt{N}$

The $O(1/\sqrt{N})$ rate is dimension-independent — unlike deterministic quadrature ($O(N^{-k/d})$ for $k$-th order rule in $d$ dimensions). Monte Carlo wins in high dimensions.

---

## Intermediate

### Variance Reduction

The $1/\sqrt{N}$ rate can be improved by reducing $\text{Var}[f(X)]$ via:

**Control variates:** find $g(x)$ with $\mathbb{E}[g(X)] = \mu_g$ known analytically and $g$ correlated with $f$. Use:
$$\hat{\mu} = \frac{1}{N}\sum_i [f(x^{(i)}) - c(g(x^{(i)}) - \mu_g)]$$
Optimal $c^* = \text{Cov}(f, g) / \text{Var}(g)$; reduces variance by $1 - \rho_{fg}^2$.

**Antithetic variates:** if $x \sim p$ implies $\tilde{x} = $ "opposite" $\sim p$ (e.g., $U(0,1)$ samples $u, 1-u$), use $(f(x) + f(\tilde{x}))/2$. If $f$ is monotone, these are negatively correlated → variance reduction.

**Stratified sampling:** partition the domain into $K$ strata, sample $N/K$ points per stratum. Eliminates variance from inter-stratum variation; useful when $f$ varies systematically across the domain.

**Importance sampling** (see [[importance-sampling]]): sample from a proposal $q$ that puts mass where $|f(x)|p(x)$ is large.

### Quasi-Monte Carlo

**Quasi-Monte Carlo (QMC)** replaces random samples with **low-discrepancy sequences** (Sobol, Halton) that cover the space more uniformly than random. Convergence rate improves to $O((\log N)^d / N)$ — faster than Monte Carlo for moderate $d$, but the constant grows with $d$.

Used in: numerical integration, financial option pricing, rendering, variational inference with reparameterization.

---

## Advanced

### Markov Chain Monte Carlo (MCMC)

When we can evaluate $p(x)$ up to a normalizing constant but cannot sample directly, **MCMC** constructs a Markov chain with stationary distribution $p$:

**Metropolis-Hastings:** propose $x' \sim q(x' | x)$, accept with probability $\min(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)})$. The acceptance ratio cancels the unknown normalizer.

**Gibbs sampling:** for multivariate $p(x_1, \ldots, x_d)$, cycle through each variable and sample $x_i \sim p(x_i | x_{-i})$. Requires tractable conditionals — natural for conjugate models.

**Hamiltonian Monte Carlo (HMC):** introduces auxiliary momentum variables and simulates Hamiltonian dynamics to propose distant moves. Exponentially lower autocorrelation than random-walk MH in high dimensions. NUTS (No-U-Turn Sampler) automates the integration length.

**MCMC diagnostics:**
- $\hat{R}$ (R-hat): compares between-chain and within-chain variance; $\hat{R} > 1.1$ signals non-convergence
- Effective sample size (ESS): accounts for autocorrelation; ESS $= N / (1 + 2\sum_k \rho_k)$
- Trace plots: visual check for stationarity

### Monte Carlo in Deep Learning

**Reparameterization trick (VAE):** to backpropagate through $\mathbb{E}_{z \sim q_\phi(z|x)}[f(z)]$, write $z = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$. Then $\nabla_\phi \mathbb{E}[f] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu + \sigma\epsilon)]$ — a low-variance Monte Carlo gradient.

**Dropout as MC sampling:** at test time, running multiple stochastic forward passes with dropout gives an approximate Bayesian posterior (MC Dropout, Gal & Ghahramani 2016) — uncertainty estimation for free.

*See also: [[sampling-methods]] · [[importance-sampling]] · [[bayesian-inference]] · [[variational-inference]] · [[distributions-overview]]*
