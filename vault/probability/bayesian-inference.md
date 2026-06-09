---
title: Bayesian Inference
tags: [probability, bayesian, statistics, fundamentals]
aliases: [Bayes theorem, posterior, prior, likelihood]
difficulty: 1
status: complete
related: [variational-autoencoders, diffusion-models, backpropagation-advanced]
---

# Bayesian Inference

---

## Fundamental

### Probability Review

For two random variables $X$ and $Y$:

- **Joint:** $p(X, Y)$ — probability of $X$ and $Y$ occurring together
- **Conditional:** $p(X|Y) = p(X,Y)/p(Y)$ — probability of $X$ given $Y$
- **Marginal:** $p(X) = \sum_Y p(X, Y)$ — probability of $X$ after summing out $Y$

**Product rule:** $p(X, Y) = p(X|Y) p(Y) = p(Y|X) p(X)$

### Bayes' Theorem

Rearranging the product rule:

$$\boxed{p(\theta | D) = \frac{p(D|\theta)\, p(\theta)}{p(D)}}$$

| Term | Name | Meaning |
|------|------|---------|
| $p(\theta \| D)$ | **Posterior** | Belief about $\theta$ after observing data |
| $p(D \| \theta)$ | **Likelihood** | How probable is the data given parameters $\theta$ |
| $p(\theta)$ | **Prior** | Belief about $\theta$ before seeing data |
| $p(D)$ | **Evidence / Marginal Likelihood** | Normalizing constant: $\int p(D\|\theta) p(\theta)\, d\theta$ |

### Frequentist vs Bayesian

**Frequentist:** Parameters $\theta$ are fixed but unknown constants. Probability = long-run frequency. [[statistical-inference-mle|MLE]]: $\hat{\theta} = \arg\max_\theta p(D|\theta)$.

**Bayesian:** Parameters $\theta$ are random variables. Probability = degree of belief. Start with prior $p(\theta)$, update with data to get posterior $p(\theta|D)$. Credible interval: "there is 95% probability that $\theta$ lies in this interval."

Neither is universally correct — they answer different questions. Bayesian inference is natural when you have prior knowledge or care about uncertainty quantification.

### Maximum Likelihood Estimation (MLE)

$$\hat{\theta}_{MLE} = \arg\max_\theta p(D|\theta) = \arg\max_\theta \sum_{i=1}^N \log p(x_i|\theta)$$

**Example — Gaussian:** For $\{x_1,\dots,x_N\}$ from $\mathcal{N}(\mu, \sigma^2)$, setting derivatives to zero gives $\hat{\mu}_{MLE} = \frac{1}{N}\sum x_i$ and $\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum(x_i-\hat{\mu})^2$ (biased).

### Conjugate Priors

For certain likelihood-prior pairs, the posterior has the same distributional form as the prior — making Bayesian updates analytically tractable.

| Likelihood | Conjugate Prior | Posterior |
|-----------|----------------|-----------|
| Binomial | Beta | Beta |
| Gaussian (known $\sigma$) | Gaussian | Gaussian |
| Poisson | Gamma | Gamma |
| Multinomial | Dirichlet | Dirichlet |

**Beta-Binomial example:** Prior $\text{Beta}(2,2)$, observe 7 heads in 10 flips: Posterior $= \text{Beta}(9, 5)$. Mean $= 9/14 \approx 0.643$. MAP $= 8/12 \approx 0.667$. MLE $= 7/10 = 0.700$. The prior pulls all estimates below the MLE toward 0.5. Prior acts as **pseudo-counts**.

---

## Intermediate

### Maximum A Posteriori (MAP)

$$\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|D) = \arg\max_\theta \log p(D|\theta) + \log p(\theta)$$

MAP adds a log-prior term to MLE, equivalent to **regularized MLE**:

| Prior | Log-prior | Regularization |
|-------|-----------|----------------|
| Gaussian: $p(\theta) \propto e^{-\lambda\|\theta\|^2}$ | $-\lambda\|\theta\|^2$ | L2 (weight decay) |
| Laplace: $p(\theta) \propto e^{-\lambda\|\theta\|_1}$ | $-\lambda\|\theta\|_1$ | L1 (Lasso) |

If we assume $\theta_i \sim \mathcal{N}(0, 1/\lambda)$, then $\hat{\theta}_{MAP} = \arg\max_\theta \log p(D|\theta) - \lambda\|\theta\|^2$ — exactly the L2-regularized loss. MAP provides a Bayesian interpretation for a widely used engineering practice.

### Full Bayesian Inference and the Intractability Problem

MLE and MAP return a single point estimate. Full Bayesian inference maintains the entire posterior distribution $p(\theta|D)$.

**Prediction by Bayesian model averaging:**

$$p(x^* | D) = \int p(x^*|\theta) p(\theta|D)\, d\theta$$

This quantifies **epistemic uncertainty** — uncertainty from not knowing which $\theta$ is correct.

**The problem:** $p(D) = \int p(D|\theta) p(\theta)\, d\theta$ requires integrating over all possible parameters — intractable for neural networks with millions of parameters.

### Aleatoric vs Epistemic Uncertainty

| Type | Source | Reducible? |
|------|--------|-----------|
| **Aleatoric** | Inherent randomness in the data (sensor noise, label noise) | No — more data doesn't help |
| **Epistemic** | Limited knowledge / insufficient data | Yes — more data reduces it |

- Aleatoric: captured by $p(y|x, \theta)$ — prediction uncertainty given known parameters
- Epistemic: captured by $p(\theta|D)$ — uncertainty about which $\theta$ is correct

Deep neural networks typically only model aleatoric uncertainty (via softmax). Bayesian NNs or ensembles attempt to capture epistemic uncertainty.

### Approximate Inference: Variational Inference

Replace the intractable posterior $p(\theta|D)$ with a simpler approximate distribution $q_\phi(\theta)$ from a tractable family (e.g., factored Gaussian). Minimize [[loss-kl-divergence|KL divergence]]:

$$\phi^* = \arg\min_\phi D_{KL}(q_\phi(\theta) \,||\, p(\theta|D))$$

This is equivalent to maximizing the ELBO. **[[variational-autoencoders|VAEs]] are a deep learning application of variational inference** — the encoder learns $q_\phi(z|x)$ and the decoder implements $p_\theta(x|z)$.

### Approximate Inference: MCMC

The [[sampling-methods|Metropolis-Hastings]] algorithm samples from the posterior without computing $p(D)$:

1. Propose a new $\theta'$ near current $\theta$
2. Accept with probability $\min\left(1, \frac{p(D|\theta')p(\theta')}{p(D|\theta)p(\theta)}\right)$
3. The chain converges to samples from $p(\theta|D)$

The ratio cancels $p(D)$ — MCMC does not need the normalizing constant. Correct but slow; requires burn-in and mixing diagnostics.

---

## Advanced

### Dropout as Approximate Bayesian Inference

Gal & Ghahramani (2016) showed that a neural network with dropout applied at **test time** can be interpreted as approximate Bayesian inference. Running the same input through the network $T$ times with different dropout masks gives $T$ samples from an approximate posterior. The variance across outputs estimates epistemic uncertainty. This connection is elegant but the approximation quality is debated — the implicit variational family is very restricted.

### Laplace Approximation

A classical deterministic approximation: fit a Gaussian centered at the MAP estimate $\hat{\theta}_{MAP}$ using the observed [[fisher-information|Fisher information matrix]] as the precision:

$$p(\theta|D) \approx \mathcal{N}(\hat{\theta}_{MAP},\; [-\nabla^2 \log p(\theta|D)|_{\hat{\theta}}]^{-1})$$

Requires computing the Hessian of the log-posterior — $O(d^2)$ storage, $O(d^3)$ inversion. Practical with low-rank approximations (Ritter et al., 2018 — *A Scalable Laplace Approximation for Neural Networks*). Used in Bayesian deep learning to estimate predictive uncertainty post-hoc without retraining.

### ELBO Geometry and Tightness

The ELBO $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(D|\theta)] - D_{KL}(q_\phi \| p(\theta))$ lower-bounds $\log p(D)$:

$$\log p(D) = \mathcal{L}(\phi) + D_{KL}(q_\phi(\theta) \| p(\theta|D)) \geq \mathcal{L}(\phi)$$

The gap is exactly $D_{KL}(q_\phi \| p(\theta|D))$, which is zero when $q_\phi = p(\theta|D)$. Mean-field VI (fully factored $q$) achieves tight bounds only when posterior dimensions are independent — underestimates uncertainty for correlated posteriors. Normalizing flows as $q_\phi$ can tighten the bound arbitrarily (Rezende & Mohamed, 2015).

### Bayesian Nonparametrics

Standard Bayesian inference fixes the model structure; Bayesian nonparametrics lets the model complexity grow with data:

- **Gaussian Process:** prior over functions; posterior is a GP conditioned on observations. Exact inference $O(n^3)$; sparse GP approximations $O(nm^2)$ where $m \ll n$ inducing points (Titsias, 2009).
- **Dirichlet Process:** infinite mixture model — number of mixture components inferred from data. Chinese Restaurant Process and stick-breaking representations provide constructive priors.

These frameworks are theoretically principled but computationally demanding at scale.

---

*See also: [[variational-autoencoders]] · [[diffusion-models]] · [[backpropagation-advanced]] · [[statistical-inference-mle]] · [[sampling-methods]] · [[fisher-information]] · [[loss-kl-divergence]]*
