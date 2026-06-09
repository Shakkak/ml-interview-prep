---
title: Hypothesis Testing and Statistical Inference
tags: [hypothesis-testing, p-value, t-test, type-1-error, type-2-error, confidence-intervals, statistical-power]
aliases: [hypothesis testing, p-value, t-test, confidence interval, null hypothesis, Type I error, Type II error, statistical significance]
difficulty: 2
status: complete
related: [statistical-inference-mle, distributions-gaussian, model-calibration, bayesian-inference, fisher-information]
---

# Hypothesis Testing and Statistical Inference

---

## Fundamental

### The Framework

Hypothesis testing answers: **is an observed effect real, or could it be explained by chance?**

**Null hypothesis $H_0$:** the default assumption — no effect, no difference.

**Alternative hypothesis $H_1$:** what you want to detect.

The test asks: assuming $H_0$ is true, how likely is it to observe data at least as extreme as what we saw? If this probability is small enough, we reject $H_0$.

### Type I and Type II Errors

|  | $H_0$ true | $H_0$ false |
|--|-----------|------------|
| **Reject $H_0$** | Type I error (false positive) rate = $\alpha$ | Correct (true positive) rate = power = $1-\beta$ |
| **Fail to reject $H_0$** | Correct (true negative) rate = $1-\alpha$ | Type II error (false negative) rate = $\beta$ |

- **Type I ($\alpha$):** concluding an effect exists when it doesn't. Typically $\alpha = 0.05$.
- **Type II ($\beta$):** missing a real effect. Typical target: $\beta = 0.20$ (power = 0.80).
- **The tradeoff:** decreasing $\alpha$ reduces Type I errors but increases Type II. Increasing sample size reduces both simultaneously — the only free lunch.

### The p-value

**Definition:** $p = P(T \geq t_{\text{obs}} \mid H_0)$ — probability of a test statistic at least as extreme as observed, assuming $H_0$.

**Decision rule:** reject $H_0$ if $p < \alpha$.

**Common misinterpretations:**
- ✗ "p = 0.03 means there is a 3% chance $H_0$ is true." (Requires a prior — it's [[bayesian-inference|Bayesian]].)
- ✗ "p = 0.03 means a 3% chance the result is due to chance." (p is computed *assuming* chance.)
- ✗ "p < 0.05 means the effect is large." (Statistical significance ≠ practical significance.)
- ✓ "p = 0.03 means: if $H_0$ were true, data this extreme would occur only 3% of the time."

A tiny effect in a large dataset can have p = 0.001. Always report effect size alongside p-value.

### Confidence Intervals

A 95% CI $[L, U]$ for the mean: "95% of intervals constructed this way across repeated experiments contain the true mean." For unknown $\sigma$:

$$\bar{x} \pm t_{n-1,\, 0.025} \cdot \frac{s}{\sqrt{n}}$$

**Correct interpretation:** the true mean is fixed; any specific interval either contains it or doesn't. The 95% is a property of the *procedure*, not of any specific interval.

**Duality with hypothesis testing:** rejecting $H_0: \mu = \mu_0$ at level $\alpha$ is equivalent to $\mu_0$ falling outside the $(1-\alpha)$ CI.

---

## Intermediate

### The t-Test

Used when population variance is unknown (estimated from sample).

**One-sample t-test:** $t = (\bar{x} - \mu_0)/(s/\sqrt{n})$. Under $H_0$: $t \sim t_{n-1}$.

**Two-sample independent t-test:** $t = (\bar{x}_1 - \bar{x}_2)/\sqrt{s_p^2(1/n_1 + 1/n_2)}$ where $s_p^2$ is the pooled variance. d.f. $= n_1 + n_2 - 2$.

**Paired t-test:** compute differences $d_i = x_{1i} - x_{2i}$, then one-sample t-test on $d_i$. More powerful than independent t-test because pairing removes between-subject variance.

**Why t, not z?** If $\sigma^2$ is known, use $z = (\bar{x} - \mu_0)/(\sigma/\sqrt{n})$. Using estimated $s$ introduces extra uncertainty, captured by heavier tails of the t-distribution. As $n \to \infty$, t-distribution converges to normal.

### Statistical Power

Power $= P(\text{reject } H_0 \mid H_1 \text{ true}) = 1 - \beta$. Depends on:

- **Effect size** $d = |\mu_1 - \mu_2|/\sigma$: larger effect → higher power
- **Sample size** $n$: power increases as $\sqrt{n}$
- **Significance level** $\alpha$: higher $\alpha$ → higher power (but more Type I errors)

**Power analysis:** before running an experiment, calculate minimum $n$ for target power (typically 0.80) at expected effect size. In ML: "how many test examples do I need to detect a 1% accuracy improvement with 80% power?"

### Multiple Comparisons

Testing $m$ hypotheses simultaneously inflates the false positive rate. For $m = 20$ independent tests at $\alpha = 0.05$: $1 - 0.95^{20} \approx 0.64$ chance of at least one false positive.

**Bonferroni correction:** use $\alpha/m$ per test. Guarantees family-wise error rate ≤ $\alpha$. Conservative when tests are correlated.

**Benjamini-Hochberg (FDR):** controls the *expected fraction* of rejections that are false positives. Less conservative. Preferred in high-dimensional settings (genomics, many-model comparisons).

**ML relevance:** running ablations over $m$ hyperparameter choices, comparing $m$ models, multiple evaluation metrics — all are multiple comparison problems.

### Worked Example — Comparing Two Models

Model A accuracy 82.3%, Model B 83.1% on $n=1000$ test examples:

$$z = \frac{0.831 - 0.823}{\sqrt{0.827 \times 0.173 \times 0.002}} \approx 0.47, \quad p\text{-value} \approx 0.64$$

Not significant — the 0.8% difference is within noise for $n = 1000$. You'd need $\approx n = 8000$ examples to detect a 0.8% difference at 80% power.

---

## Advanced

### Non-Parametric Tests

When the [[distributions-gaussian|Gaussian]] assumption fails:

- **Mann-Whitney U / Wilcoxon rank-sum:** tests whether two distributions have the same median without assuming normality. Based on ranks. Less powerful than t-test when normality holds, but robust.
- **McNemar's test:** paired test for classification accuracy — tests whether models make different errors on the same examples, not just whether overall accuracy differs. Exact for small samples.
- **Permutation test:** compute the test statistic, then randomly permute labels $B$ times to build the null distribution empirically. Exact finite-sample guarantee regardless of distribution.

### Bayesian Hypothesis Testing

Frequentist p-values answer "how surprising is the data under $H_0$?" Bayesian alternative directly computes the probability that $H_0$ is true:

$$P(H_0 \mid \text{data}) = \frac{P(\text{data} \mid H_0) P(H_0)}{P(\text{data})}$$

**Bayes factor:** $B = P(\text{data} \mid H_1) / P(\text{data} \mid H_0)$. Quantifies evidence in favor of $H_1$ without requiring a threshold. Jeffreys (1961) scale: $B > 3$ = moderate, $B > 10$ = strong, $B > 30$ = very strong evidence.

Bayesian testing avoids arbitrary thresholds but requires specifying a prior over both $H_0$ and $H_1$, including the distribution of effect sizes under $H_1$.

### Sequential Hypothesis Testing and A/B Tests

Standard tests assume the sample size is fixed before looking at data. Running a test and stopping as soon as $p < 0.05$ inflates Type I error to $\approx 30\%$ for sequential looks.

**Sequential probability ratio test (SPRT):** Wald (1945) showed that the optimal sequential test uses likelihood ratios:

$$\Lambda_n = \prod_{i=1}^n \frac{p_1(x_i)}{p_0(x_i)}$$

Stop and reject $H_0$ if $\Lambda_n > B$, stop and accept if $\Lambda_n < A$, otherwise continue. This achieves the minimum expected sample size for given $(\alpha, \beta)$.

**In practice (online A/B testing):** always-valid $p$-values (Johari et al., 2017, Optimizely) use the e-values framework — martingale-based statistics that remain valid under optional stopping. These are now standard in production A/B testing systems.

### Chi-Squared Test

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

Under $H_0$: $\chi^2 \sim \chi^2_{k-1}$ for goodness-of-fit with $k$ categories. For an independence test in an $r \times c$ contingency table: d.f. $= (r-1)(c-1)$.

Requirement: expected counts $E_i \geq 5$ in each cell for the asymptotic approximation to hold. For small samples, use Fisher's exact test instead.

---

*See also: [[statistical-inference-mle]] · [[distributions-gaussian]] · [[bayesian-inference]] · [[fisher-information]] · [[model-calibration]]*
