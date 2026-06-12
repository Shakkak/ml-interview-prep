---
title: Central Limit Theorem & Law of Large Numbers
tags: [probability, statistics, convergence, clt, lln]
aliases: [CLT, law of large numbers, LLN, convergence in distribution]
difficulty: 2
status: complete
related: [distributions-gaussian, statistical-inference-mle, bootstrap, hypothesis-testing, bayesian-inference, fisher-information]
---

# Central Limit Theorem & Law of Large Numbers

---

## Fundamental

### Law of Large Numbers (LLN)

Let $X_1, X_2, \ldots$ be i.i.d. random variables with mean $\mu = \mathbb{E}[X_i]$ and finite variance. The **Weak LLN** states:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

The sample mean converges in probability to the population mean — no matter how the distribution looks, averaging out enough independent samples kills the noise.

**Strong LLN** (Kolmogorov): the convergence is almost sure:

$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

**Intuition:** Averaging is noise cancellation. Independent errors are equally likely to be positive or negative; they cancel rather than reinforce. The variance of $\bar{X}_n$ is $\sigma^2/n$, so the standard deviation of the mean shrinks as $1/\sqrt{n}$.

**Why ML relies on LLN constantly:**
- Mini-batch gradient is an estimate of the true gradient; LLN says the average over enough samples converges to the true gradient
- Empirical risk (training loss) converges to expected risk (generalization loss) as dataset grows
- Monte Carlo estimates of intractable expectations $\mathbb{E}[f(X)]$ converge by LLN
- A/B test significance relies on sample mean converging to true effect size

### Central Limit Theorem (CLT)

Given i.i.d. $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2 < \infty$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

Equivalently: $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$.

**The remarkable fact:** the shape of $X_i$'s distribution does not matter — uniform, exponential, Bernoulli — as long as $\sigma^2 < \infty$, averages of i.i.d. variables become Gaussian. The Gaussian is the universal attractor for sums of independent noise.

**Practical rule of thumb:** CLT applies well when $n \geq 30$ for most distributions; for heavy-tailed or strongly skewed distributions, $n \geq 100$–$1000$ may be needed.

**Worked example:** coin flips ($p = 0.5$), $\mu = 0.5$, $\sigma^2 = 0.25$, $n = 100$:
$$\bar{X}_{100} \approx \mathcal{N}(0.5, 0.25/100) = \mathcal{N}(0.5, 0.0025)$$
Standard error $= 0.05$. An empirical proportion of 0.6 is $(0.6 - 0.5)/0.05 = 2\sigma$ away — a p-value of about 0.046.

---

## Intermediate

### Convergence Rates and Berry–Esseen Theorem

The CLT says *eventually* Gaussian, but not how fast. The **Berry–Esseen theorem** gives a finite-$n$ bound:

$$\sup_x \left|P\!\left(\frac{\sqrt{n}(\bar{X}-\mu)}{\sigma} \leq x\right) - \Phi(x)\right| \leq \frac{C \rho}{\sigma^3 \sqrt{n}}$$

where $\rho = \mathbb{E}[|X - \mu|^3]$ is the third absolute moment and $C \approx 0.4748$ (best known constant). The approximation error is $O(1/\sqrt{n})$ — adding 100x more data buys only 10x better accuracy of the Gaussian approximation.

**Implication for hypothesis testing:** for small datasets ($n < 30$) or skewed distributions, the CLT-based Gaussian approximation can be off by enough to invalidate p-values. This is why the t-distribution (with heavier tails) is used for small samples, and bootstrap methods are preferred for small $n$ with unknown distributions.

### Multivariate CLT

For i.i.d. random vectors $\mathbf{X}_i \in \mathbb{R}^d$ with mean $\boldsymbol{\mu}$ and covariance matrix $\Sigma$:

$$\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \Sigma)$$

**Application — gradient noise in SGD:** each mini-batch gradient estimate is $\hat{g} = \frac{1}{B}\sum_{i \in \text{batch}} g_i$. By the multivariate CLT, for large enough batch size $B$:

$$\hat{g} \approx \mathcal{N}\!\left(g, \frac{\Sigma_g}{B}\right)$$

where $\Sigma_g = \text{Cov}(g_i)$ is the gradient covariance matrix. This is the theoretical justification for treating SGD as noisy gradient descent with Gaussian noise — the noise covariance scales as $1/B$, explaining why doubling batch size halves noise variance (but also halves the number of updates per epoch, so the optimal learning rate scales as $\sqrt{B}$, the linear scaling rule).

### The Delta Method

The delta method is a generalization of the CLT to nonlinear functions of sample statistics. If $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$ and $g$ is differentiable at $\mu$:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2)$$

**Intuition:** linearize $g$ around $\mu$ via first-order Taylor expansion: $g(x) \approx g(\mu) + g'(\mu)(x - \mu)$. Then $g(\bar{X}) - g(\mu) \approx g'(\mu)(\bar{X} - \mu)$, and the variance scales by $[g'(\mu)]^2$.

**ML application — confidence intervals for model performance:** if $\hat{p}$ is an empirically measured accuracy (proportion of correct predictions), its standard error is $\sqrt{\hat{p}(1-\hat{p})/n}$ by CLT. For a derived metric like $F_1 = 2PR/(P+R)$, the delta method gives the variance in terms of the Jacobian of $F_1$ with respect to $(P, R)$. This is how confidence intervals for non-linear evaluation metrics are computed.

### CLT for Dependent Sequences (Mixing)

The standard CLT requires independence. In practice, sequential data is correlated. The CLT extends to **stationary mixing processes**: if $X_t$ is stationary and $\phi$-mixing (correlation decays sufficiently fast with lag), then:

$$\frac{1}{\sqrt{n}}\sum_{t=1}^n (X_t - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma_\infty^2)$$

where $\sigma_\infty^2 = \sum_{k=-\infty}^\infty \text{Cov}(X_0, X_k)$ is the **long-run variance** — the sum of all autocovariances. This is larger than $\sigma^2$ when series are positively correlated.

**Why this matters for time-series ML:** evaluation metrics computed on a sequential test set are not i.i.d.; the effective sample size is $n_\text{eff} = n \cdot \sigma^2 / \sigma_\infty^2 < n$. Ignoring serial correlation overstates statistical power — a common error in time-series model evaluation.

---

## Advanced

### The CLT and Maximum Likelihood Estimation

For a regular parametric model with log-likelihood $\ell(\theta)$, the MLE $\hat{\theta}_n$ satisfies:

$$\sqrt{n}(\hat{\theta}_n - \theta^*) \xrightarrow{d} \mathcal{N}\!\left(0, I(\theta^*)^{-1}\right)$$

where $I(\theta^*) = \mathbb{E}[-\partial^2 \ell / \partial\theta^2]$ is the [[fisher-information|Fisher information matrix]]. This follows from the multivariate CLT applied to the score function $\nabla_\theta \log p(X|\theta)$ (which has mean 0 and covariance $I(\theta^*)$) combined with a Taylor expansion of the score around $\theta^*$.

**Implication:** with $n$ samples, the MLE has variance $1/(nI)$. This is also the Cramér-Rao lower bound — no unbiased estimator can do better. The MLE is **asymptotically efficient**: it achieves the minimum possible variance as $n \to \infty$.

**In neural network terms:** training with cross-entropy loss is MLE. The network's parameter uncertainty after training on $n$ samples scales as $1/\sqrt{n}$ in each direction of parameter space, with the curvature (Fisher matrix) determining relative uncertainty along different directions. This is the theoretical foundation for Laplace approximations and Fisher-based uncertainty quantification.

### CLT, Batch Size, and the Linear Scaling Rule

When training with SGD, the gradient estimate variance is $\text{Var}(\hat{g}) = \Sigma_g / B$. As batch size $B$ doubles, gradient noise halves. To maintain the same noise-to-signal ratio with doubled batch size, the learning rate should also double — the **linear scaling rule** (Goyal et al., 2017, Facebook). 

However, this rule breaks down at large batch sizes because:
1. Gradient variance can't go below the true gradient's squared norm, which doesn't scale with $B$
2. The noise is actually beneficial for SGD (implicit regularization via Langevin dynamics), so reducing it too much hurts generalization
3. CLT requires i.i.d. samples — large batches exhaust the dataset faster, breaking effective i.i.d. assumption

The empirical threshold ("critical batch size") where linear scaling stops helping is approximately $B^* \approx \sigma_g^2 / \|g\|^2$ — the ratio of gradient variance to gradient magnitude squared.

### Non-CLT Behavior: Heavy Tails and Lévy Processes

When $\sigma^2 = \infty$ (e.g., Pareto distribution with exponent $\alpha \leq 2$), the CLT fails. Instead, sums of i.i.d. heavy-tailed variables converge to **stable distributions** (Lévy processes) rather than Gaussians. The characteristic function is $\exp(-c|t|^\alpha)$ instead of $\exp(-\sigma^2 t^2/2)$.

**Relevance to deep learning:** stochastic gradient noise in large language model training has been empirically shown to be heavy-tailed in some settings (Simsekli et al., 2019). Heavy-tailed noise changes the implicit regularization of SGD from favoring flat minima (as predicted by the Gaussian/Langevin theory) to behaviors more consistent with $\alpha$-stable processes — connected to the observation that larger models trained with SGD reach better solutions than smaller models with the same noise scale.

---

*See also: [[distributions-gaussian]] · [[statistical-inference-mle]] · [[bootstrap]] · [[hypothesis-testing]] · [[fisher-information]] · [[bayesian-inference]]*
