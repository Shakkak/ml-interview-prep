---
title: Statistical Inference and MLE
tags: [statistics, inference, mle, map, hypothesis-testing]
aliases: [MLE, maximum likelihood, MAP, statistical testing, confidence interval]
difficulty: 2
status: complete
related: [distributions-overview, distributions-gaussian, bayesian-inference, loss-cross-entropy, regularization-weight-decay]
---

# Statistical Inference and MLE

---

## Fundamental

### Maximum Likelihood Estimation

MLE finds the parameter that maximizes the probability of the observed data:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^n p(x_i|\theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i|\theta)$$

Log is used because products become sums, and the log is monotone so it preserves the argmax.

**Gaussian MLE:** $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$ (biased by factor $(n-1)/n$). Unbiased: divide by $n-1$ (Bessel's correction).

**Bernoulli MLE:** $\hat{p} = k/n$ (fraction of successes). Derivation: $d/dp[k\log p + (n-k)\log(1-p)] = k/p - (n-k)/(1-p) = 0$.

**Poisson MLE:** $\hat{\lambda} = \bar{x}$.

**Pattern for exponential family:** MLE = moment matching — set the expected sufficient statistic equal to the observed sufficient statistic. For Gaussian: match $\bar{x}$ and $\overline{x^2}$.

### Confidence Intervals

95% CI for Gaussian mean with known $\sigma^2$: $\bar{x} \pm 1.96\,\sigma/\sqrt{n}$.

Unknown $\sigma^2$: $\bar{x} \pm t_{\alpha/2, n-1}\, s/\sqrt{n}$ (t-distribution, slightly wider).

**Example:** $n=25$, $\bar{x}=12$, $\sigma=5$. CI $= 12 \pm 1.96 \times 1 = [10.04, 13.96]$.

**Width $\propto 1/\sqrt{n}$:** to halve the CI width, quadruple the sample size.

### Hypothesis Testing: Step-by-Step

**Setup:** Model A: 85.2%, Model B: 86.1% on $n=1000$ test examples. $H_0$: $p_A = p_B$.

Z-test for proportions: $\hat{p} = 0.857$, $z = (0.852 - 0.861)/\sqrt{0.857 \times 0.143 \times 0.002} = -0.574$.

$p$-value $= 0.566$. Fail to reject $H_0$ — the 0.9% difference is not significant. Need ~3.1% difference to be significant at $n=1000$.

---

## Intermediate

### MAP Estimation

$$\hat{\theta}_{MAP} = \arg\max_\theta [\log p(x_1,\ldots,x_n|\theta) + \log p(\theta)]$$

MAP adds a log-prior term to MLE, equivalent to regularized MLE:

| Prior | Regularization |
|-------|---------------|
| Gaussian $\mathcal{N}(0, 1/(2\lambda))$ | L2 ($\lambda\|\theta\|^2$) |
| Laplace $\propto e^{-\lambda|\theta|}$ | L1 ($\lambda\|\theta\|_1$) |
| Uniform | None (MAP = MLE) |

**Example:** $n=5$, observe $k=4$ successes. MLE: $\hat{p} = 0.8$.

- Beta(2,2) prior: MAP $= 5/7 = 0.714$ — shrunk toward 0.5
- Beta(10,10) prior: MAP $= 13/23 = 0.565$ — strongly shrunk
- $n=100$, $k=80$ with Beta(2,2): MAP $= 81/103 = 0.786 \approx$ MLE — data overwhelms prior

### Bootstrap CI

When no closed-form formula exists (e.g., AUC):

1. Draw $B = 10{,}000$ bootstrap samples (resample with replacement)
2. Compute statistic on each
3. 95% CI = 2.5th and 97.5th percentile

```python
aucs = []
for _ in range(10000):
    idx = np.random.choice(len(y_true), len(y_true), replace=True)
    aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
ci = np.percentile(aucs, [2.5, 97.5])
```

### Type I vs Type II Errors

| | $H_0$ true | $H_0$ false |
|--|-----------|------------|
| **Reject $H_0$** | Type I ($\alpha$) | Correct (power $1-\beta$) |
| **Fail to reject** | Correct | Type II ($\beta$) |

**Power analysis:** with $n=100$ per group, Cohen's $d = 0.5$:

$$n = \frac{(z_\alpha + z_\beta)^2}{d^2/2} = \frac{(1.96 + 0.84)^2}{0.125} = 63 \text{ per group}$$

100 per group gives power > 80% ✓.

### Fisher Information and Asymptotic Efficiency

The MLE is asymptotically efficient — it achieves the Cramér-Rao lower bound:

$$\sqrt{n}(\hat{\theta}_{MLE} - \theta^*) \xrightarrow{d} \mathcal{N}(0, I(\theta^*)^{-1})$$

For Bernoulli($p$): $I(p) = 1/(p(1-p))$. Cramér-Rao: $\text{Var}(\hat{p}) \geq p(1-p)/n$. The MLE $\hat{p} = k/n$ achieves this exactly — it is the minimum variance unbiased estimator (MVUE).

---

## Advanced

### Asymptotic Theory of MLE

Under regularity conditions (Cramér regularity):

1. **Consistency:** $\hat{\theta}_{MLE} \xrightarrow{p} \theta^*$ as $n \to \infty$
2. **Asymptotic normality:** $\sqrt{n}(\hat{\theta}_{MLE} - \theta^*) \to \mathcal{N}(0, F(\theta^*)^{-1})$
3. **Asymptotic efficiency:** no consistent asymptotically normal estimator has smaller asymptotic variance

The Wald test, score (Lagrange multiplier) test, and likelihood ratio test are three asymptotically equivalent tests derived from MLE theory, all with $\chi^2$ asymptotic distributions under $H_0$.

**Likelihood ratio test:** $\Lambda = 2[\ell(\hat{\theta}) - \ell(\theta_0)] \xrightarrow{d} \chi^2_k$ under $H_0$, where $k$ is the number of constrained parameters. Most powerful among tests with fixed size (Neyman-Pearson lemma).

### M-Estimators and Robustness

MLE is a special case of M-estimators: $\hat{\theta} = \arg\min_\theta \sum_i \rho(x_i, \theta)$ where $\rho = -\log p$ for MLE.

**Robust alternatives:**
- **Huber loss:** $\rho(r) = r^2/2$ for $|r| \leq k$, $k|r| - k^2/2$ for $|r| > k$. Less sensitive to outliers than squared loss.
- **Tukey's bisquare:** complete downweighting of outliers beyond a threshold.

The influence function $\text{IF}(x; T, F) = \lim_{\epsilon \to 0} [T(F_\epsilon) - T(F)]/\epsilon$ measures the sensitivity of estimator $T$ to a single contaminating observation. MLE has unbounded influence function for heavy-tailed models; robust M-estimators bound it.

### Information Criteria for Model Selection

AIC and BIC balance fit against complexity:

$$\text{AIC} = 2k - 2\ln\hat{L}, \quad \text{BIC} = k\ln n - 2\ln\hat{L}$$

where $k$ = number of parameters, $\hat{L}$ = maximum likelihood, $n$ = sample size.

**Derivation of AIC:** Akaike (1974) derived AIC as an approximately unbiased estimator of the expected KL divergence from the fitted model to the true distribution. The $2k$ penalty corrects for the optimistic bias of in-sample log-likelihood.

**Consistency of BIC:** BIC is consistent — as $n \to \infty$, it selects the true model if it is in the candidate set. AIC is not consistent but minimizes asymptotic prediction error. Use AIC for prediction, BIC for structure discovery.

**AICc (corrected AIC):** $\text{AICc} = \text{AIC} + 2k(k+1)/(n-k-1)$ — important correction for small $n/k$ ratios; AICc converges to AIC as $n \to \infty$.

### Expectation-Maximization (EM) Algorithm

For models with latent variables $Z$, direct MLE of $\theta$ requires marginalizing $p(X|\theta) = \sum_Z p(X,Z|\theta)$ — often intractable. EM maximizes the ELBO instead:

**E-step:** compute $Q(\theta|\theta^{(t)}) = E_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$

**M-step:** $\theta^{(t+1)} = \arg\max_\theta Q(\theta|\theta^{(t)})$

EM monotonically increases $\log p(X|\theta)$ and converges to a local maximum. It is the de facto algorithm for Gaussian mixture models, HMMs (Baum-Welch), and many latent variable models. The gap between EM and direct gradient ascent: EM has exact M-steps for exponential family models (linear systems), while gradient ascent requires step-size tuning and handles any differentiable likelihood.

---

*See also: [[distributions-gaussian]] · [[distributions-overview]] · [[bayesian-inference]] · [[regularization-weight-decay]] · [[loss-cross-entropy]] · [[fisher-information]]*
