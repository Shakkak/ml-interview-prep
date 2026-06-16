---
title: Probability Distributions Overview
tags: [probability, distributions, statistics, gaussian, binomial, poisson]
aliases: [distributions, probability distributions, common distributions]
difficulty: 1
status: complete
related: [distributions-gaussian, maximum-entropy-principle, bayesian-inference, entropy-mutual-info, loss-cross-entropy]
depends_on: []

---

# Probability Distributions Overview

---

## Fundamental

**Why different distributions?** Every probabilistic model requires you to describe how the data is generated. A distribution is your claim about which values are likely and how spread out they are. The *wrong* distribution leads to a model that cannot express the true uncertainty in your data — for example, using a Gaussian to model count data forces negative values to have non-zero probability, which is nonsensical. Choosing the right distribution is the first design decision in any probabilistic model.

The key factors: **support** (what values can the variable take?), **generating process** (is this a sum of effects, a count, a probability?), and **tail behavior** (rare events likely or not?).

### Choosing a Distribution: Decision Framework

| Data type | Support | Generating process | Distribution |
|-----------|---------|-------------------|--------------|
| Binary outcome | $\{0,1\}$ | One trial | Bernoulli($p$) |
| Count of successes | $\{0,...,n\}$ | $n$ independent binary trials | Binomial($n,p$) |
| Count of events | $\{0,1,2,...\}$ | Rare events at constant rate | Poisson($\lambda$) |
| Time between events | $[0,\infty)$ | Memoryless process | Exponential($\lambda$) |
| Continuous, additive noise | $(-\infty, \infty)$ | Sum of many small effects | Gaussian($\mu, \sigma^2$) |
| Positive, multiplicative | $(0, \infty)$ | Product of many small effects | Log-Normal |
| Probability of success | $[0,1]$ | Prior over $p$ | Beta($\alpha, \beta$) |
| Categorical probabilities | simplex | Prior over multinomial | Dirichlet($\alpha$) |

### Key Discrete Distributions

**Bernoulli:** PMF: $P(X=1) = p$, $P(X=0) = 1-p$. Mean: $p$, Variance: $p(1-p)$. [[entropy-mutual-info|Entropy]]: $-p\log p - (1-p)\log(1-p)$, maximized at $p=0.5$.

**Binomial:** PMF: $\binom{n}{k}p^k(1-p)^{n-k}$. Mean: $np$, Variance: $np(1-p)$. For large $n$, small $p$ with $np$ moderate → Poisson($\lambda = np$).

**Numerical check:** $n=10$, $p=0.3$, $P(X=3) = \binom{10}{3}(0.3)^3(0.7)^7 = 120 \times 0.027 \times 0.0824 = 0.267$. Poisson($\lambda=3$) approximation: $3^3 e^{-3}/3! \approx 0.224$ — decent but not exact (Poisson works better for small $p$, large $n$).

**Poisson:** PMF: $P(X=k) = \lambda^k e^{-\lambda}/k!$. Mean $= \lambda$, Variance $= \lambda$ (mean = variance — key property). Over-dispersion: real count data often has Var > Mean → use Negative Binomial.

### Key Continuous Distributions

**Uniform:** PDF: $1/(b-a)$ on $[a,b]$. Entropy: $\ln(b-a)$ — [[maximum-entropy-principle|maximum entropy]] for bounded support.

**Exponential:** PDF: $\lambda e^{-\lambda x}$, $x \geq 0$. Mean: $1/\lambda$, Variance: $1/\lambda^2$. Memoryless: $P(X > s+t | X > s) = P(X > t)$. Maximum entropy given fixed mean on $[0,\infty)$.

**Gaussian:** Full treatment in [[distributions-gaussian]]. CLT: normalized sums converge to Gaussian. Maximum entropy given fixed mean and variance. Closed under linear transforms, sums, conditioning.

**Log-Normal:** $X = e^Y$ where $Y \sim \mathcal{N}(\mu, \sigma^2)$. Mean: $e^{\mu + \sigma^2/2}$, Median: $e^\mu$ (mean > median — right skew). Arises from multiplicative processes.

### Entropy Summary

| Distribution | Entropy |
|-------------|---------|
| Bernoulli($p$) | $-p\log p - (1-p)\log(1-p)$ |
| Uniform($[0,b]$) | $\ln b$ |
| Exponential($\lambda$) | $1 - \ln\lambda$ |
| Gaussian($\mu, \sigma^2$) | $\frac{1}{2}\ln(2\pi e \sigma^2)$ |
| Multivariate Gaussian | $\frac{1}{2}\ln((2\pi e)^d \vert \Sigma\vert )$ |

---

## Intermediate

### Conjugate Priors

| Likelihood | Conjugate Prior | Posterior |
|------------|----------------|-----------|
| Bernoulli($p$) | Beta($\alpha, \beta$) | Beta($\alpha+k, \beta+n-k$) |
| Poisson($\lambda$) | Gamma($\alpha, \beta$) | Gamma($\alpha+\sum x_i, \beta+n$) |
| Categorical($\pi$) | Dirichlet($\alpha$) | Dirichlet($\alpha + n_k$) |
| Gaussian($\mu$, known $\sigma^2$) | Gaussian($\mu_0, \tau^2$) | Gaussian (weighted average) |

Conjugacy means the posterior is the same family as the prior — closed-form [[bayesian-inference|Bayesian]] update.

### Beta and Dirichlet Distributions

**Beta($\alpha$, $\beta$):** PDF $\propto \theta^{\alpha-1}(1-\theta)^{{\beta}-1}$ on $[0,1]$. Mean $= \alpha/(\alpha+\beta)$. Conjugate prior for Bernoulli/Binomial. $\alpha, \beta$ act as pseudo-counts: Beta(1,1) = Uniform; Beta(2,2) slightly prefers $\theta = 0.5$.

**Dirichlet($\alpha_1,\ldots,\alpha_K$):** PDF $\propto \prod_k \pi_k^{\alpha_k-1}$ on the probability simplex. Generalization of Beta to $K$ outcomes. Conjugate prior for Categorical/Multinomial. Concentration parameter $\alpha_0 = \sum_k \alpha_k$ controls sparsity: small $\alpha_0$ → sparse (few active categories); large $\alpha_0$ → dense (near-uniform). Used in Latent Dirichlet Allocation (LDA).

### Exponential Family

A distribution is in the exponential family if $p(x|\eta) = h(x)\exp(\eta^\top T(x) - A(\eta))$ where $\eta$ = natural parameters, $T(x)$ = sufficient statistics, $A(\eta)$ = log-partition function. Members: Gaussian, Bernoulli, Binomial, Poisson, Exponential, Gamma, Beta, Dirichlet — nearly all common distributions.

Key properties: (1) Sufficient statistics $T(x)$ capture all information about $\eta$. (2) MLE = moment matching: $\nabla A(\eta) = \frac{1}{n}\sum T(x_i)$. (3) Conjugate priors always exist. (4) GLMs use exponential family for the response distribution.

### Heavy-Tailed Distributions

Heavy-tailed distributions have tails decaying slower than exponential: $P(X > x) \sim x^{-\alpha}$ (power law).

**Student-t with $\nu$ d.f.:** heavier tails than Gaussian; variance $\nu/(\nu-2)$ for $\nu>2$; approaches Gaussian as $\nu \to \infty$. Used in robust regression (outliers have non-negligible probability) and t-SNE's low-dimensional kernel ($q_{ij} \propto (1+\|y_i-y_j\|^2)^{-1}$ prevents crowding).

**Chi-squared:** $\chi^2_k = \sum_{i=1}^k Z_i^2$ for $Z_i \sim \mathcal{N}(0,1)$ i.i.d. Mean $k$, Variance $2k$. Arises in goodness-of-fit tests and in the distribution of sample variance.

**F-distribution:** $F_{d_1,d_2} = (\chi^2_{d_1}/d_1)/(\chi^2_{d_2}/d_2)$ — ratio of two scaled chi-squared. Used in ANOVA and regression F-tests.

**Negative Binomial:** Poisson with Gamma-distributed rate, allowing Var > Mean (overdispersion). PMF: $P(X=k) = \binom{k+r-1}{k}p^k(1-p)^r$. Real count data almost always shows overdispersion — NB is standard for RNA-seq analysis (DESeq2, edgeR) and NLP word counts.

---

## Advanced

### Gaussian Mixture Models

$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$ with mixing weights $\pi_k \geq 0$, $\sum \pi_k = 1$. Parameters estimated via EM:

- **E-step:** $r_{ik} = \frac{\pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}$ (soft cluster responsibilities)
- **M-step:** $\pi_k = \frac{1}{n}\sum_i r_{ik}$, $\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}$, $\Sigma_k = \frac{\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top}{\sum_i r_{ik}}$

EM guarantees monotone log-likelihood increase but finds only local optima. K-Means is hard-assignment EM with spherical equal-variance Gaussians — GMM strictly generalizes it.

### Change of Variables and Normalizing Flows

If $X$ has PDF $p_X(x)$ and $Y = g(X)$ is a differentiable bijection:

$$p_Y(y) = p_X(g^{-1}(y))\,|\det J_{g^{-1}}(y)|$$

The Jacobian determinant accounts for volume stretching. **[[normalizing-flows|Normalizing flows]]** (Rezende & Mohamed, 2015) compose many such transforms to convert a simple Gaussian into a complex distribution, with exact likelihood via the Jacobian: $\log p_X(x) = \log p_Z(f(x)) + \log|\det J_f(x)|$.

Affine coupling layers (RealNVP, Dinh et al. 2017) achieve $O(d)$ Jacobian computation by partitioning dimensions. Autoregressive flows (MAF/IAF, Papamakarios et al. 2017) have tractable density but slow sampling or vice versa — the architecture determines which direction is cheap.

### Sufficient Statistics and Fisher–Koopman–Darmois Theorem

By the FKD theorem, only exponential family distributions have finite-dimensional sufficient statistics for all sample sizes. For $X_i \sim \mathcal{N}(\mu, \sigma^2)$, $(\bar{X}, S^2)$ is sufficient — you can discard individual data points once you have these two numbers.

This has direct implications for online learning (update sufficient statistics incrementally) and for understanding what information is lost by summary statistics. Notably, the sample median is **not** sufficient for the Gaussian mean — it ignores information in the tails.

### Extreme Value Theory

The distribution of the maximum of $n$ i.i.d. random variables converges to one of three types: Gumbel (thin tails), Fréchet (heavy tails), or Weibull (bounded support). This is the Fisher-Tippett-Gnedenko theorem — an analog of the CLT for extremes. Relevant in ML for: modeling worst-case performance, anomaly detection, reliability of large distributed systems.

---

## Links

- [[distributions-gaussian]] — the most important continuous distribution; full treatment with multivariate, conditional, and GP forms
- [[maximum-entropy-principle]] — MaxEnt selects the distribution with least information given only the constraints you have (mean, support, etc.)
- [[bayesian-inference]] — conjugate prior tables connect directly to the likelihood families catalogued here
- [[entropy-mutual-info]] — Shannon entropy of each distribution measures its average information content
- [[normalizing-flows]] — change-of-variables formula for transforming simple distributions into complex ones with exact likelihood
