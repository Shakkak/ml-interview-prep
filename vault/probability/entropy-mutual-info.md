---
title: Entropy and Mutual Information
tags: [information-theory, probability, loss]
aliases: [entropy, mutual information, information gain, conditional entropy, joint entropy]
difficulty: 2
status: complete
related: [loss-cross-entropy, loss-kl-divergence, loss-nt-xent, bayesian-inference]
---

# Entropy and Mutual Information

---

## Fundamental

### Shannon Entropy

The entropy of a discrete random variable $X$ with distribution $p$:

$$H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}_p[-\log p(X)]$$

Measures **uncertainty** or **information content** of a distribution.

**Units:** bits (log base 2), nats (log base $e$). Conversion: 1 nat = 1.44 bits.

**Key examples:**

- Fair coin $p = [0.5, 0.5]$: $H = 1$ bit — maximum for binary
- Biased coin $p = [0.9, 0.1]$: $H = 0.469$ bits — less uncertainty
- Uniform over $K$ outcomes: $H = \log_2 K$ — maximum for $K$ outcomes
- Deterministic (one outcome = 1): $H = 0$ — no uncertainty
- 4-sided fair die: $H = \log_2 4 = 2$ bits

### Cross-Entropy

$$H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)]$$

The expected code length when using distribution $q$ to encode messages drawn from $p$.

$$H(p, q) = H(p) + D_{KL}(p \,\|\, q)$$

Minimizing cross-entropy (ML training objective) is equivalent to minimizing KL divergence from true distribution $p$ to model $q$.

### Conditional Entropy

$$H(X|Y) = -\sum_{x,y} p(x,y)\log p(x|y) = \mathbb{E}_{p(x,y)}[-\log p(X|Y)]$$

Average remaining uncertainty in $X$ after observing $Y$.

$H(X|Y) \leq H(X)$ always — conditioning can only reduce entropy.

### Mutual Information

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

How much knowing $Y$ reduces uncertainty about $X$.

Also expressible as KL divergence between joint and product of marginals:

$$I(X; Y) = D_{KL}(p(x,y) \,\|\, p(x)p(y))$$

If $X \perp Y$: $I(X;Y) = 0$. Chain rule: $H(X, Y) = H(X) + H(Y|X)$ and $I(X; Y) = H(X) + H(Y) - H(X, Y)$.

---

## Intermediate

### Worked Numerical Example

$X$ = weather (sunny/rainy): $P(S) = 0.7$, $P(R) = 0.3$.
$Y$ = umbrella (yes/no): $P(Y|S) = 0.1$, $P(Y|R) = 0.9$.

| | $Y=1$ | $Y=0$ |
|---|---|---|
| $S$ | $0.07$ | $0.63$ |
| $R$ | $0.27$ | $0.03$ |

$H(X) = 0.882$ nats. $H(X|Y=1) = 0.504$ nats, $H(X|Y=0) = 0.181$ nats.

$H(X|Y) = 0.34 \times 0.504 + 0.66 \times 0.181 = 0.291$ nats.

$I(X; Y) = 0.882 - 0.291 = 0.591$ nats — knowing whether someone carries an umbrella reduces weather uncertainty by 67%.

### MI in Machine Learning

**Contrastive learning (NT-Xent):** NT-Xent loss is a lower bound on $I(z_i; z_j)$ — mutual information between two views. Maximizing this bound trains representations that capture shared information across views (see [[loss-nt-xent]]).

**ICA (Independent Component Analysis):** minimize $I(s_1, s_2, \ldots, s_n)$ — find components that are maximally independent.

**Feature selection (information gain):** $I(X; Y)$ where $Y$ is the target label. Used in decision trees. Features with high MI with the label are informative.

**VAE / ELBO:** the ELBO contains $I(x; z)$ implicitly — the reconstruction term encourages the latent code to preserve information about the input.

### Differential Entropy

For continuous distributions, $H(X) = -\int p(x)\log p(x)\,dx$. Unlike discrete entropy:

- Can be negative (e.g., $\mathcal{N}(0, \sigma^2)$ when $\sigma^2 < 1/(2\pi e)$)
- Not invariant to change of variables (depends on parameterization)
- Gaussian entropy: $\frac{1}{2}\ln(2\pi e\sigma^2)$ — depends only on $\sigma^2$, not $\mu$

KL divergence between Gaussians has a closed form:

$$D_{KL}(\mathcal{N}(\mu_1,\sigma_1^2) \| \mathcal{N}(\mu_2,\sigma_2^2)) = \frac{1}{2}\left[\frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_2-\mu_1)^2}{\sigma_2^2} - 1 + \ln\frac{\sigma_2^2}{\sigma_1^2}\right]$$

---

## Advanced

### Information Bottleneck

A theory of deep learning representation: $Z = f(X)$ should be:
1. **Sufficient** for predicting $Y$: maximize $I(Z; Y)$
2. **Compressed** representation of $X$: minimize $I(Z; X)$

Optimal trade-off: $\max I(Z; Y) - \beta \cdot I(Z; X)$

Tishby & Schwartz-Ziv (2017) claimed deep networks undergo distinct compression and fitting phases in the $I(Z;X)$ vs $I(Z;Y)$ plane during training. This sparked controversy — Saxe et al. (2018) showed the claimed compression phase only appears with saturating activations (tanh), not ReLU. The IB framework remains theoretically interesting as a lens on generalization but its empirical claim is disputed.

### Data Processing Inequality

For a Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(X; Y)$$

Processing data can only reduce mutual information — you cannot create new information from a transformation. Consequence: any deterministic function $Z = g(Y)$ satisfies $I(X; Z) \leq I(X; Y)$. This means no encoding step in a neural network can increase the mutual information between input and a deeper representation — it can only preserve or reduce it.

### Entropy Rate of Stochastic Processes

For a stationary stochastic process $\{X_t\}$, the entropy rate is:

$$\bar{H} = \lim_{n\to\infty} \frac{1}{n} H(X_1, \ldots, X_n) = \lim_{n\to\infty} H(X_n | X_1,\ldots,X_{n-1})$$

For an ergodic Markov chain: $\bar{H} = -\sum_{ij} \pi_i P_{ij} \log P_{ij}$. Language models minimize the negative log-probability per token — which is an estimate of the entropy rate of language. GPT-4 achieving ~2.3 bits/token means natural language has about 2.3 bits of entropy per token conditional on context.

### Rényi Entropy and Connections to Distributions

The Rényi entropy of order $\alpha$:

$$H_\alpha(X) = \frac{1}{1-\alpha}\log\sum_x p(x)^\alpha$$

- $\alpha \to 1$: recovers Shannon entropy
- $\alpha = 2$: collision entropy — related to birthday problem
- $\alpha \to \infty$: min-entropy — relevant in cryptography and differential privacy

The Rényi-2 divergence connects to Gaussian kernel density estimators and maximum mean discrepancy (MMD), used in GAN training variants.

---

*See also: [[loss-cross-entropy]] · [[loss-kl-divergence]] · [[loss-nt-xent]] · [[bayesian-inference]] · [[maximum-entropy-principle]]*
