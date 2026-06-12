---
title: Variational Inference
tags: [variational-inference, bayesian, elbo, mean-field, amortized]
aliases: [VI, variational Bayes, ELBO, mean-field approximation, amortized inference]
difficulty: 3
status: complete
related: [bayesian-inference, loss-kl-divergence, variational-autoencoders, normalizing-flows, sampling-methods, fisher-information, change-of-variables]
---

# Variational Inference

---

## Fundamental

### The Problem: Intractable Posteriors

Given data $\mathcal{D}$ and a model with latent variables $\mathbf{z}$, Bayes' theorem gives the posterior:

$$p(\mathbf{z}|\mathcal{D}) = \frac{p(\mathcal{D}|\mathbf{z})\, p(\mathbf{z})}{p(\mathcal{D})}$$

The denominator $p(\mathcal{D}) = \int p(\mathcal{D}|\mathbf{z}) p(\mathbf{z})\, d\mathbf{z}$ is a high-dimensional integral — analytically intractable for almost all interesting models. Even MCMC (see [[sampling-methods]]) becomes too slow for modern neural network–scale models.

**Variational inference** replaces exact posterior computation with an optimization problem: find the distribution $q_\phi(\mathbf{z})$ within a tractable family $\mathcal{Q}$ that is closest to the true posterior $p(\mathbf{z}|\mathcal{D})$, where "closest" means minimizing the KL divergence (see [[loss-kl-divergence]]):

$$q^*(\mathbf{z}) = \arg\min_{q \in \mathcal{Q}} D_{KL}(q(\mathbf{z}) \| p(\mathbf{z}|\mathcal{D}))$$

### The ELBO

Minimizing $D_{KL}(q \| p(\mathbf{z}|\mathcal{D}))$ is still intractable because it requires computing $\log p(\mathcal{D})$. But expanding the KL:

$$D_{KL}(q(\mathbf{z}) \| p(\mathbf{z}|\mathcal{D})) = \mathbb{E}_q[\log q(\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{z}|\mathcal{D})]$$
$$= \mathbb{E}_q[\log q(\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{z}, \mathcal{D})] + \log p(\mathcal{D})$$

Rearranging: $\log p(\mathcal{D}) = \underbrace{\mathbb{E}_q[\log p(\mathbf{z}, \mathcal{D})] - \mathbb{E}_q[\log q(\mathbf{z})]}_{\text{ELBO}} + D_{KL}(q \| p(\mathbf{z}|\mathcal{D}))$

Since $\log p(\mathcal{D})$ is constant and $D_{KL} \geq 0$, **maximizing the ELBO is equivalent to minimizing KL**:

$$\text{ELBO}(q) = \mathbb{E}_q[\log p(\mathbf{z}, \mathcal{D})] - \mathbb{E}_q[\log q(\mathbf{z})] = \mathbb{E}_q\!\left[\log \frac{p(\mathbf{z}, \mathcal{D})}{q(\mathbf{z})}\right]$$

The ELBO lower-bounds $\log p(\mathcal{D})$ — the "evidence lower bound." Maximizing it tightens this bound while making $q$ closer to the posterior.

**Decomposition:** $\text{ELBO} = \mathbb{E}_q[\log p(\mathcal{D}|\mathbf{z})] - D_{KL}(q(\mathbf{z}) \| p(\mathbf{z}))$

- First term: reconstruction quality (how well $\mathbf{z}$ explains the data)
- Second term: regularization toward the prior (keep $q$ close to $p(\mathbf{z})$)

---

## Intermediate

### Mean-Field Variational Inference

The simplest tractable family: **mean-field** assumes complete factorization over latent variables:

$$q(\mathbf{z}) = \prod_{j=1}^d q_j(z_j)$$

Each factor $q_j$ is independent — no correlations are captured between latent variables.

**Coordinate ascent variational inference (CAVI):** optimize one factor at a time, holding others fixed. The optimal update for $q_j$ is:

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{z}, \mathcal{D})] + \text{const}$$

where $q_{-j}$ denotes the product over all factors except $j$. This has the form of an exponential family distribution for conjugate models.

**CAVI convergence:** each update is guaranteed to increase (or maintain) the ELBO. The algorithm converges to a local optimum — global optimality is not guaranteed.

**Limitation of mean-field:** factorization forces $q$ to underestimate posterior variance (a well-known pathology). For a bimodal posterior, mean-field $q$ concentrates on one mode. Mean-field VI is "zero-forcing" (tends to miss modes) compared to maximum likelihood estimation's "mass-covering" behavior — a direct consequence of optimizing $D_{KL}(q \| p)$ rather than $D_{KL}(p \| q)$.

### Black-Box Variational Inference

For models without conjugate structure, we can still maximize the ELBO using gradient ascent. The gradient is:

$$\nabla_\phi \text{ELBO} = \nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log \frac{p(\mathbf{z}, \mathcal{D})}{q_\phi(\mathbf{z})}\right]$$

**Score function estimator (REINFORCE):**
$$\nabla_\phi \text{ELBO} = \mathbb{E}_{q_\phi}\!\left[\nabla_\phi \log q_\phi(\mathbf{z}) \cdot \left(\log \frac{p(\mathbf{z}, \mathcal{D})}{q_\phi(\mathbf{z})}\right)\right]$$

Unbiased but high variance. Variance reduction techniques (control variates, Rao-Blackwellization) are needed.

**Reparameterization gradient** (see [[change-of-variables]]): when $q_\phi(\mathbf{z}) = q(\boldsymbol{\epsilon})$ with $\mathbf{z} = g_\phi(\boldsymbol{\epsilon})$:
$$\nabla_\phi \text{ELBO} = \mathbb{E}_{p(\boldsymbol{\epsilon})}\!\left[\nabla_\phi \log p(g_\phi(\boldsymbol{\epsilon}), \mathcal{D}) - \nabla_\phi \log q_\phi(g_\phi(\boldsymbol{\epsilon}))\right]$$

Much lower variance than score function estimator. Requires $g_\phi$ to be differentiable.

### Normalizing Flows as Variational Families

Mean-field $q$ is too simple for complex posteriors. **Normalizing flows** (see [[normalizing-flows]]) build richer variational families: start with a simple base $q_0(\mathbf{z}_0)$, apply a sequence of invertible transformations $f_1, \ldots, f_K$:

$$q_K(\mathbf{z}_K) = q_0(\mathbf{z}_0) \prod_{k=1}^K \left|\det J_{f_k}\right|^{-1}$$

The ELBO with flow-based $q$ is maximized jointly over base distribution parameters and transformation parameters. This trades the bias of mean-field for computational cost ($K$ flow steps per sample).

---

## Advanced

### Amortized Inference

**Standard VI:** for $N$ data points, optimize $q_\phi(\mathbf{z}_n)$ separately for each $n$ — $O(N)$ parameters, $O(N)$ optimization problems.

**Amortized VI:** learn a shared **inference network** (encoder) $q_\phi(\mathbf{z}|\mathbf{x})$ that maps any input $\mathbf{x}$ directly to posterior parameters $(\mu_\phi(\mathbf{x}), \sigma_\phi(\mathbf{x}))$. This is the encoder in a Variational Autoencoder (see [[variational-autoencoders]]).

**The amortization gap:** the encoder's approximation $q_\phi(\mathbf{z}|\mathbf{x})$ is never as tight as optimizing $q$ per data point. This gap can be reduced by fine-tuning the variational parameters at test time (semi-amortized VI) or using iterative inference.

### Expectation Propagation

EP (Minka, 2001) minimizes $D_{KL}(p \| q)$ rather than $D_{KL}(q \| p)$ — the reverse direction. This gives a "mass-covering" approximation (tends to be overdispersed rather than mode-seeking). EP alternates projection steps in exponential family form and is exact for Gaussian posteriors. Used in Bayesian neural network approximations and GP classification (see [[gaussian-processes]]).

### Natural Gradient Variational Inference

Gradient descent in the parameter space of $q_\phi$ is geometry-blind — the same step size causes different movements in probability space depending on the local curvature of the distribution manifold.

**Natural gradient:** multiply the gradient by the inverse Fisher information matrix $I(\phi)^{-1}$ to account for the Riemannian geometry of the distribution family:

$$\phi \leftarrow \phi + \eta I(\phi)^{-1} \nabla_\phi \text{ELBO}$$

For exponential families, the natural gradient has closed form. **Stochastic natural gradient VI** uses mini-batches and achieves much faster convergence per iteration than standard gradient VI — used in large-scale probabilistic models (deep GPs, hierarchical Bayesian models).

---

*See also: [[bayesian-inference]] · [[loss-kl-divergence]] · [[variational-autoencoders]] · [[normalizing-flows]] · [[sampling-methods]] · [[change-of-variables]] · [[expectation-maximization]]*
