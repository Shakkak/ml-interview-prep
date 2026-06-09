---
title: Bootstrap (Resampling)
tags: [statistics, resampling, bootstrap, confidence-intervals, bagging]
aliases: [bootstrapping, bootstrap resampling, bootstrap sample, bootstrap CI, percentile bootstrap, bagging]
difficulty: 1
status: complete
related: [statistical-inference-mle, ensemble-methods, cross-validation, bias-variance-double-descent, hypothesis-testing]
---

# Bootstrap (Resampling)

---

## Fundamental

**Definition.** The bootstrap (Efron, 1979) is a resampling technique for approximating the
**sampling distribution** of a statistic — its standard error, bias, or confidence interval —
without deriving a closed-form formula or assuming a parametric model. The core move is the
**plug-in principle**: since we can't draw more samples from the true (unknown) population, we
treat the *observed sample* as a stand-in for it, and approximate "redraw from the population"
with "redraw from the sample, with replacement."

**The algorithm**, given an observed sample $x_1,\ldots,x_n$ and a statistic $\hat\theta =
T(x_1,\ldots,x_n)$ (the mean, median, AUC, a regression coefficient — anything computable from
data):
1. Repeat $B$ times (typically $B = 1{,}000$–$10{,}000$): draw $n$ points from
   $\{x_1,\ldots,x_n\}$ **with replacement**, forming a *bootstrap sample*
   $x_1^*,\ldots,x_n^*$ of the same size; compute $\hat\theta^*_b = T(x_1^*,\ldots,x_n^*)$
2. The empirical distribution of $\hat\theta^*_1,\ldots,\hat\theta^*_B$ **is** the bootstrap's
   estimate of $\hat\theta$'s sampling distribution — its spread approximates the standard
   error, its quantiles give a confidence interval (see Intermediate)

```
function bootstrap(data, statistic, B):
    estimates = []
    for b in 1..B:
        resample = sample(data, size=len(data), replace=True)
        estimates.append(statistic(resample))
    return estimates   # the empirical sampling distribution of `statistic`
```

**Worked example.** Sample $\{2, 4, 6, 8, 10\}$, $\bar x = 6$. Four bootstrap resamples (drawn
with replacement, same size $n=5$) and their means:

| Resample | $\hat\theta^* = $ mean |
|---|---|
| $\{4, 4, 8, 2, 10\}$ | $5.6$ |
| $\{6, 10, 6, 2, 8\}$ | $6.4$ |
| $\{2, 2, 4, 10, 10\}$ | $5.6$ |
| $\{8, 8, 6, 4, 6\}$ | $6.4$ |

The spread of $\{5.6, 6.4, 5.6, 6.4\}$ is a (very crude, $B=4$) estimate of how much $\bar x$
would vary across different samples from the population — with a real $B = 10{,}000$ this
becomes a smooth empirical distribution whose standard deviation **is** the bootstrap standard
error estimate $\widehat{SE}(\hat\theta) = \sqrt{\frac{1}{B-1}\sum_b(\hat\theta^*_b -
\bar{\hat\theta}^*)^2}$.

> [!tip] Why sampling *with* replacement is the whole trick
> Resampling *without* replacement, $n$ items from $n$, just reproduces the original sample
> every time — zero variability, nothing to measure. *With* replacement, each resample
> randomly omits some original points and duplicates others, and it is exactly this
> resample-to-resample variation that mimics the sample-to-sample variation you'd see if you
> could actually draw fresh samples from the population. On average about 37% of the original
> points are left out of any given resample ($e^{-1} \approx 0.368$ as $n\to\infty$) — the
> **out-of-bag** points exploited by bagging (see Intermediate).

---

## Intermediate

**Percentile bootstrap confidence interval.** The most common application: build a CI for a
statistic with no tractable formula (e.g. AUC, a correlation, a difference of medians) by
taking the empirical quantiles of the bootstrap distribution directly —
$$\text{95\% CI} = \big[\hat\theta^*_{(2.5\%)},\ \hat\theta^*_{(97.5\%)}\big]$$
i.e. the 2.5th and 97.5th percentiles of $\{\hat\theta^*_1,\ldots,\hat\theta^*_B\}$. See
[[statistical-inference-mle]] for the worked AUC version of exactly this procedure — the
mechanism is identical regardless of which statistic $T$ computes.

**Bagging — bootstrap put to work for variance reduction.** [[ensemble-methods|Bootstrap
Aggregating ("bagging")]] reuses the *same resampling step* for a completely different purpose:
instead of studying how much a statistic varies, it trains $N$ independent models — one per
bootstrap sample — and averages their predictions. The bootstrap sample is no longer a tool for
measuring uncertainty; it's a tool for manufacturing the *diversity* that makes averaging
reduce variance (recall $\text{Var}(\bar y) = \sigma^2/N$ for independent learners — see
[[bias-variance-double-descent]]). Random Forests are the canonical example.

**Out-of-bag (OOB) error — a free validation set.** Because each bootstrap sample omits ~37%
of the original points, those held-out points can evaluate the model trained on that sample —
yielding an honest validation estimate without sacrificing any training data to a separate
hold-out split. This is the bootstrap's main competitor to [[cross-validation]] for
model evaluation when the model is itself an ensemble of bootstrap-trained learners.

**The jackknife — bootstrap's older sibling.** Leave-one-out resampling: compute $\hat\theta$
on each of the $n$ subsamples of size $n-1$ (deleting one point at a time). It's
deterministic (no randomness, $n$ resamples total) and was the dominant resampling method
before the bootstrap, but its estimates are coarser — it can only "see" $n$ leave-one-out
configurations, whereas the bootstrap can generate effectively unlimited resamples and capture
richer features of the sampling distribution (e.g. its skew).

---

## Advanced

**Why it works — asymptotic justification.** Efron's key result is that, under mild regularity
conditions, the distribution of $\hat\theta^* - \hat\theta$ (resample statistic minus
observed statistic) converges to the same limiting distribution as $\hat\theta - \theta$
(observed statistic minus true population value) as $n \to \infty$. In other words, the
fluctuation of the resample around the *sample* mirrors the fluctuation of the sample around
the *population* — which is precisely the substitution the plug-in principle assumes. This is
what licenses using the bootstrap distribution's spread as a stand-in for the true sampling
distribution's spread.

**When the basic (percentile) bootstrap breaks down:**
- **Non-smooth statistics** — e.g. the sample maximum or minimum. Their bootstrap distribution
  doesn't converge to a normal shape and the percentile interval can be badly biased; smoother
  statistics (means, medians of large samples) are well-behaved.
- **Small samples / skewed statistics** — the plain percentile interval can have poor coverage
  (the true value falls outside the claimed 95% CI more than 5% of the time). The
  **bias-corrected and accelerated (BCa)** interval adjusts the percentiles using estimates of
  the bootstrap distribution's bias and skewness, and is the standard refinement in practice.
- **Dependent data** — the i.i.d. resampling assumption is violated for time series or
  spatially correlated data, where resampling individual points destroys the dependence
  structure that the statistic relies on. The **block bootstrap** resamples contiguous *blocks*
  of consecutive observations instead of individual points, preserving local dependence.

**Parametric vs. nonparametric bootstrap.** The procedure above is the *nonparametric*
bootstrap — it resamples the data directly, assuming nothing about its distribution. The
*parametric* bootstrap instead (1) fits a parametric model $\hat{F}_\theta$ to the data via
[[statistical-inference-mle|MLE]], then (2) draws synthetic samples from $\hat{F}_\theta$
itself rather than from the empirical data. This can be more efficient when the parametric
assumption is correct (it uses the fitted model's smoothness rather than the lumpy empirical
distribution), but introduces model-misspecification risk that the nonparametric version
sidesteps entirely.

**Computational cost and practical $B$.** Each bootstrap replicate requires recomputing the
statistic from scratch — for a statistic that is itself expensive (e.g. refitting a model),
$B = 10{,}000$ can be prohibitive. In practice $B$ in the low thousands is usually enough for
stable percentile estimates; standard-error estimates stabilize faster (a few hundred) than
extreme-tail quantiles (which need more replicates to populate the tails reliably).

---

*See also: [[statistical-inference-mle]] · [[ensemble-methods]] · [[cross-validation]] · [[bias-variance-double-descent]] · [[hypothesis-testing]]*
