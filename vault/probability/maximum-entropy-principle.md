---
title: Maximum Entropy Principle
tags: [probability, entropy, information-theory, statistics]
aliases: [maximum entropy, MaxEnt, principle of maximum entropy]
difficulty: 3
status: complete
related: [distributions-gaussian, distributions-overview, entropy-mutual-info, math-convexity-jensen, bayesian-inference]
---

# Maximum Entropy Principle

---

## Fundamental

### The Core Idea

Given partial knowledge about a distribution (constraints), choose the distribution that is **maximally uncertain** (highest entropy) consistent with those constraints.

**Justification:** any distribution with lower entropy than the maximum makes implicit claims beyond what the constraints specify — it encodes information we don't actually have. MaxEnt is the least-biased choice.

E.T. Jaynes (1957): MaxEnt is a form of epistemological honesty — commit to no more structure than the data forces you to.

### Key Results Table

| Constraints | Support | MaxEnt Distribution | Entropy |
|------------|---------|---------------------|---------|
| None | $[a,b]$ | Uniform | $\ln(b-a)$ |
| Fixed mean $\mu$ | $[0,\infty)$ | Exponential($1/\mu$) | $1 + \ln\mu$ |
| Fixed mean $\mu$ and variance $\sigma^2$ | $\mathbb{R}$ | Gaussian($\mu, \sigma^2$) | $\frac{1}{2}\ln(2\pi e\sigma^2)$ |
| None | $\{1,\ldots,K\}$ | Uniform | $\ln K$ |
| Fixed mean on $\{0,1,2,\ldots\}$ | $\mathbb{N}_0$ | Geometric | — |
| Fixed covariance $\Sigma$ | $\mathbb{R}^d$ | $\mathcal{N}(\mu, \Sigma)$ | $\frac{1}{2}\ln((2\pi e)^d|\Sigma|)$ |

**Practical meaning:** when you assume [[distributions-gaussian|Gaussian]] noise, you're making the weakest possible assumption given that you know the variance — a principled default.

---

## Intermediate

### General Derivation via Lagrange Multipliers

Maximize $H(p) = -\int p(x)\ln p(x)\,dx$ subject to normalization and constraint functions $\int f_k(x)p(x)\,dx = \mu_k$:

$$\frac{\delta}{\delta p(x)}\left[-p\ln p - \lambda_0 p - \sum_k \lambda_k f_k(x)p\right] = 0$$

$$-\ln p - 1 - \lambda_0 - \sum_k \lambda_k f_k(x) = 0$$

$$p^*(x) = \exp\!\left(-1 - \lambda_0 - \sum_k \lambda_k f_k(x)\right)$$

This is always an **[[distributions-overview|exponential family]] distribution** — the Lagrange multipliers are the natural parameters. MaxEnt under moment constraints always yields an exponential family member.

**Uniqueness:** entropy is strictly concave, so the optimum is unique whenever the constraint set has non-empty interior (see [[math-convexity-jensen]]).

### Worked Derivation: Fixed Mean → Exponential

Constraints: $\int_0^\infty p(x)\,dx = 1$ and $\int_0^\infty x\, p(x)\,dx = \mu$.

Lagrangian: $p^*(x) \propto e^{-\lambda x}$. Normalization requires $\lambda > 0$. Mean constraint: $\int_0^\infty x \lambda e^{-\lambda x}\,dx = 1/\lambda = \mu$, so $\lambda = 1/\mu$.

Result: Exponential($1/\mu$). **Verification:** Exp(1): $H = 1 - \ln(1) = 1$ nat ✓.

### Numerical Comparison: Same Variance, Different Constraints

$\sigma^2 = 1$, $\mu = 0$, support $= \mathbb{R}$:

| Distribution | Constraint | Entropy |
|-------------|-----------|---------|
| Gaussian $\mathcal{N}(0,1)$ | Fixed $E[X^2]$ | $\frac{1}{2}\ln(2\pi e) \approx 1.42$ nats |
| Laplace $\text{Lap}(0, 1/\sqrt{2})$ | Fixed $E[|X|]$ | $1 + \ln(\sqrt{2}) \approx 1.35$ nats |

Gaussian has higher entropy than Laplace for the same variance — but Laplace maximizes entropy subject to fixed mean absolute deviation (not variance). They are each MaxEnt under different constraints.

### Why MaxEnt Matters in ML

1. **Regularization:** L2 regularization = Gaussian prior = MaxEnt prior given bounded expected squared weight.
2. **[[activation-softmax|Softmax]]:** the softmax output with no additional information is a uniform distribution — MaxEnt over $K$ classes. Training updates logits to encode class evidence.
3. **Energy-based models:** the Boltzmann distribution $p(x) \propto e^{-E(x)/T}$ is the MaxEnt distribution given fixed expected energy $E[E(x)]$.
4. **Feature matching:** GANs with Wasserstein loss, MMD, and MaxEnt models relate to matching moments (constraints) while being maximally uncertain otherwise.
5. **MaxEnt language models (pre-neural):** estimated $P(\text{word}|\text{context})$ by maximizing entropy subject to observed $n$-gram frequency constraints.

---

## Advanced

### MaxEnt and the Exponential Family: Full Connection

The canonical exponential family form $p(x|\eta) = h(x)\exp(\eta^\top T(x) - A(\eta))$ is equivalent to: the MaxEnt distribution subject to the constraint $E[T(x)] = \mu$. The natural parameters $\eta$ are the Lagrange multipliers. The log-partition function $A(\eta) = \log \int h(x)e^{\eta^\top T(x)}dx$ is the cumulant generating function:

$$\nabla_\eta A(\eta) = E_\eta[T(x)] = \mu, \quad \nabla^2_\eta A(\eta) = \text{Cov}_\eta[T(x)]$$

The Hessian of $A$ equals the Fisher information matrix restricted to this family. MLE in an exponential family is **moment matching**: set $E_\eta[T(x)] = \frac{1}{n}\sum T(x_i)$.

### Duality: MaxEnt and Maximum Likelihood

MaxEnt (primal): $\max_p H(p)$ subject to $E_p[f_k(x)] = \hat{\mu}_k$.

Maximum Likelihood in exponential family (dual): $\max_\eta \sum_i [\eta^\top T(x_i)] - A(\eta)$.

These two problems are dual (Jaynes-Darmois duality). The optimal $\eta^*$ in the dual corresponds to the optimal $p^*$ in the primal via $p^*(x) = h(x)\exp(\eta^{*\top} T(x) - A(\eta^*))$. The duality means that MLE for exponential families is exactly the same as MaxEnt with empirical moment constraints — a deep unification of frequentist MLE and Bayesian MaxEnt.

### Relative Entropy Minimization (MinKL)

MaxEnt is a special case of the broader principle: given a prior $q$, the "most honest" update given constraints is to minimize [[loss-kl-divergence|KL divergence]] $D_{KL}(p \| q)$ subject to the constraints (Kullback 1959). This reduces to MaxEnt when $q$ is uniform.

$$\min_p D_{KL}(p \| q) \quad \text{s.t. } E_p[f_k(x)] = \mu_k$$

Solution: $p^*(x) \propto q(x)\exp(-\sum_k \lambda_k f_k(x))$. This is **[[bayesian-inference|Bayesian]] updating as KL minimization** — the posterior $p(\theta|D) = \arg\min_p D_{KL}(p \| p_0)$ subject to $E_p[\log p(D|\theta)] = \ell^*$ (expected log-likelihood matching). Variational Bayes and power posteriors ($p(\theta|D) \propto p(\theta)p(D|\theta)^\beta$, used in cold posterior Bayesian deep learning) are instances of this framework.

### MaxEnt and the Gibbs Distribution in Statistical Mechanics

The Boltzmann distribution $p(x) \propto e^{-E(x)/T}$ solves:

$$\max_p H(p) \quad \text{s.t.} \quad E_p[E(x)] = U$$

where $U$ is the fixed expected energy and $T$ (temperature) is the Lagrange multiplier. As $T \to 0$: the distribution concentrates at the minimum energy state. As $T \to \infty$: the distribution becomes uniform.

This maps directly to score-based generative models, where $T$ controls the noise level during forward diffusion, and the denoising score function estimates $\nabla_x \log p(x)$ — the negative gradient of the energy.

---

*See also: [[distributions-gaussian]] · [[distributions-overview]] · [[entropy-mutual-info]] · [[math-convexity-jensen]] · [[bayesian-inference]] · [[activation-softmax]] · [[loss-kl-divergence]]*
