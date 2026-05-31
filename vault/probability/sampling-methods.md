---
title: Sampling Methods — Monte Carlo, MCMC, and Importance Sampling
tags: [monte-carlo, mcmc, importance-sampling, metropolis-hastings, ancestral-sampling, rejection-sampling, gibbs-sampling]
aliases: [Monte Carlo, MCMC, importance sampling, Metropolis-Hastings, ancestral sampling, rejection sampling, Gibbs sampling]
difficulty: 3
status: complete
related: [bayesian-inference, markov-chains, diffusion-models, variational-autoencoders, distributions-gaussian, entropy-mutual-info]
---

# Sampling Methods — Monte Carlo, MCMC, and Importance Sampling

---

## Fundamental

### Why Sampling?

Many intractable integrals in ML can be estimated by sampling:

$$\mathbb{E}_{p(x)}[f(x)] = \int f(x)\, p(x)\, dx \approx \frac{1}{N} \sum_{i=1}^N f(x^{(i)}), \quad x^{(i)} \sim p(x)$$

Key examples:
- **Posterior inference:** $p(\theta \mid D) \propto p(D \mid \theta) p(\theta)$ — normalizing constant $p(D)$ is intractable
- **Marginal likelihood:** $p(x) = \int p(x \mid z) p(z)\, dz$ — requires integrating over latent states
- **Expected loss:** $\mathbb{E}_{x \sim p_{\text{data}}}[\mathcal{L}(\theta; x)]$ — training data is a finite sample

### Monte Carlo Estimation

Estimate $\mathbb{E}_p[f(x)]$ by drawing $N$ i.i.d. samples:

$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^N f(x^{(i)})$$

By the Law of Large Numbers, $\hat{\mu} \to \mathbb{E}_p[f(x)]$ as $N \to \infty$.

**Variance:** $\text{Var}(\hat{\mu}) = \text{Var}_p[f(x)] / N$. Standard error decreases as $1/\sqrt{N}$ — slow but universal. The $1/\sqrt{N}$ rate does not depend on dimension — this is what makes Monte Carlo dimension-independent (unlike deterministic quadrature, which degrades exponentially).

**Problem:** requires sampling from $p(x)$ directly, which is impossible when $p$ is an unnormalized posterior.

### Rejection Sampling

Sample from $p(x)$ using a simpler proposal $q(x)$ where $p(x) \leq M q(x)$ for some $M$:

1. Sample $x \sim q(x)$
2. Sample $u \sim \text{Uniform}[0, 1]$
3. Accept if $u \leq p(x) / (Mq(x))$; else reject and repeat

Acceptance probability: $1/M$ on average. Fails in high dimensions because $p/q$ has extremely high variance — nearly all samples are rejected.

---

## Intermediate

### Importance Sampling

Estimate $\mathbb{E}_p[f(x)]$ when sampling from $p$ is hard but proposal $q$ is easy:

$$\mathbb{E}_p[f(x)] = \mathbb{E}_q\!\left[f(x)\frac{p(x)}{q(x)}\right] \approx \frac{1}{N}\sum_{i=1}^N f(x^{(i)}) w(x^{(i)}), \quad w(x) = \frac{p(x)}{q(x)}$$

**Self-normalized IS (SNIS)** when $p$ is known only up to $\tilde{p}(x) = Zp(x)$:

$$\hat{\mu}_{SNIS} = \frac{\sum_i f(x^{(i)}) \tilde{w}(x^{(i)})}{\sum_i \tilde{w}(x^{(i)})}, \quad \tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}$$

**Effective sample size (ESS):** $N_{\text{eff}} = (\sum_i w_i)^2 / \sum_i w_i^2$. ESS $\ll N$ indicates weight degeneracy — a few samples dominate. Low when $p/q$ has heavy tails.

**Optimal proposal:** $q^*(x) \propto |f(x)| p(x)$ — requires knowing the answer. In practice, choose $q$ that covers the high-probability regions of $p$ well.

### Markov Chain Monte Carlo (MCMC)

Construct a Markov chain with stationary distribution $p(x)$. After burn-in, samples approximate i.i.d. from $p$.

**Metropolis-Hastings:**
1. At state $x$, propose $x' \sim q(x' \mid x)$
2. Compute $\alpha = \min\!\left(1, \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)}\right)$
3. Move to $x'$ with probability $\alpha$; stay at $x$ otherwise

Detailed balance is satisfied, so the stationary distribution is $p$. The normalizing constant cancels in the ratio.

**Tuning Metropolis-Hastings:** proposal variance $\sigma^2$ controls the tradeoff:
- Too small: high acceptance, slow exploration (random walk with tiny steps)
- Too large: low acceptance, chain gets stuck
- Optimal: ~23% acceptance rate for Gaussian proposals in high dimensions (Roberts-Gelman-Gilks, 1997)

**Gibbs sampling:** for multivariate $p(x_1,\ldots,x_d)$, iteratively sample each variable from its full conditional $p(x_i|x_{-i})$. Always accepts. Slow when variables are highly correlated.

### MCMC Convergence Diagnostics

MCMC samples are correlated and may not have reached stationarity.

**Trace plots:** plot $x^{(t)}$ vs $t$. Mixing chain looks like white noise. Non-mixing shows trends or stuck regions.

**$\hat{R}$ (Gelman-Rubin):** run multiple chains from different starts. $\hat{R} \approx 1$ means chains converged to the same distribution. $\hat{R} > 1.1$ indicates poor mixing.

**Effective sample size:** $N_{\text{eff}} = N / (1 + 2\sum_k \rho_k)$ where $\rho_k$ is lag-$k$ autocorrelation. A chain of 1000 samples with ESS = 50 is equivalent to 50 i.i.d. samples.

### Ancestral Sampling in Generative Models

**VAE:** (1) sample $z \sim \mathcal{N}(0, I)$; (2) sample $x \sim p_\theta(x \mid z)$.

**Autoregressive models (GPT):** sample $x_1 \sim p(x_1)$, then $x_t \sim p(x_t \mid x_{<t})$ autoregressively. Exact ancestral sampling — no MCMC needed.

**Diffusion models:** the DDPM sampling procedure is a Markov chain: $x_T \sim \mathcal{N}(0, I)$, then $x_{t-1} \sim p_\theta(x_{t-1} \mid x_t)$ for $t = T, \ldots, 1$.

---

## Advanced

### Hamiltonian Monte Carlo (HMC)

Standard MH with a Gaussian random walk proposal mixes poorly in high dimensions. HMC (Duane et al., 1987) introduces auxiliary momentum $p$ and simulates Hamiltonian dynamics:

$$H(x, p) = -\log \pi(x) + \frac{1}{2}p^\top M^{-1}p$$

Leapfrog integration discretizes the Hamiltonian dynamics: $p \leftarrow p - \frac{\epsilon}{2}\nabla_x U(x)$, $x \leftarrow x + \epsilon M^{-1}p$, $p \leftarrow p - \frac{\epsilon}{2}\nabla_x U(x)$.

A Metropolis correction accepts/rejects the proposed $(x', p')$ to correct for numerical integration error. The No-U-Turn Sampler (NUTS, Hoffman & Gelman 2014) automatically tunes the trajectory length, making HMC essentially tuning-free.

**Practical insight that surprises:** HMC's advantage over random walk MH grows as dimension increases. In $d$ dimensions, random walk MH needs $O(d^2)$ steps to traverse the typical set; HMC needs $O(d^{5/4})$ (Neal 2011). For $d = 100$, HMC is $\approx 100\times$ more efficient.

### Sequential Monte Carlo (SMC) / Particle Filters

SMC propagates a weighted set of particles $\{(x^{(i)}, w^{(i)})\}$ through a sequence of distributions $\pi_1, \ldots, \pi_T$ (annealing path from prior to posterior):

1. **Reweight:** $\tilde{w}^{(i)}_t \propto w^{(i)}_{t-1} \cdot \frac{\pi_t(x^{(i)})}{\pi_{t-1}(x^{(i)})}$
2. **Resample:** replace particles with low weights by duplicating high-weight particles
3. **Move:** apply MCMC kernel to rejuvenate particle diversity

SMC is embarrassingly parallel and handles multimodal distributions better than single-chain MCMC by maintaining a diverse particle population. The marginal likelihood estimate $\hat{p}(D) = \prod_t \bar{w}_t$ is unbiased — SMC is the only practical method for computing normalizing constants for complex models.

### Stochastic Gradient MCMC: SGLD and SGHMC

For large datasets, evaluating the full log-posterior gradient is expensive. Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011) uses minibatch gradients with injected noise:

$$\theta_{t+1} = \theta_t + \frac{\epsilon_t}{2}\nabla\log\tilde{p}(\theta_t|D_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t I)$$

where $D_t$ is a minibatch. As $\epsilon_t \to 0$ following a specific schedule ($\sum \epsilon_t = \infty$, $\sum \epsilon_t^2 < \infty$), the iterates converge to samples from the full posterior despite using stochastic gradients.

**Practical issue:** the Metropolis correction for SGLD is intractable (would require full gradient to verify acceptance). SGLD is therefore biased for any fixed $\epsilon_t > 0$ — it samples from a slightly perturbed posterior. Stochastic Gradient Hamiltonian Monte Carlo (SGHMC, Chen et al. 2014) adds a friction term to correct for gradient noise, reducing this bias.

### Stein Variational Gradient Descent (SVGD)

SVGD (Liu & Wang, 2016) maintains a set of particles $\{x_i\}$ and evolves them via:

$$x_i \leftarrow x_i + \epsilon \phi^*(x_i), \quad \phi^*(x) = \frac{1}{n}\sum_j [k(x_j, x)\nabla_{x_j}\log p(x_j) + \nabla_{x_j}k(x_j, x)]$$

The optimal perturbation $\phi^*$ in the RKHS minimizes the KL divergence between the particle distribution and the target. Unlike MCMC, SVGD is deterministic and leverages repulsion (from the kernel gradient term) to maintain particle diversity. It bridges variational inference (deterministic) and particle-based sampling.

---

*See also: [[bayesian-inference]] · [[markov-chains]] · [[diffusion-models]] · [[variational-autoencoders]] · [[distributions-gaussian]]*
