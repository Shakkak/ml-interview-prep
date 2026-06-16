---
title: Arithmetic, Geometric, and Harmonic Means
tags: [math, means, averages, harmonic-mean, geometric-mean]
aliases: [harmonic mean, geometric mean, arithmetic mean, AM-GM-HM]
difficulty: 1
status: complete
related: [evaluation-metrics-guide, loss-dice, math-convexity-jensen]
depends_on: [math-convexity-jensen]
---

# Arithmetic, Geometric, and Harmonic Means

---

## Fundamental

For two positive values $a$ and $b$:

| Mean | Formula | Sensitive to |
|------|---------|-------------|
| Arithmetic (AM) | $\frac{a+b}{2}$ | Large values |
| Geometric (GM) | $\sqrt{ab}$ | Ratios / multiplicative scale |
| Harmonic (HM) | $\frac{2}{\frac{1}{a}+\frac{1}{b}} = \frac{2ab}{a+b}$ | Near-zero values |

**Key inequality:** $\text{HM} \leq \text{GM} \leq \text{AM}$, with equality iff $a = b$.

The harmonic mean is always pulled toward the smaller of the two values:

$$\text{HM}(a, b) \leq \min(a, b)$$

**Why:** HM takes the average of *reciprocals*, so a value near zero makes $1/a$ very large, dragging the reciprocal average up and the harmonic mean down.

### Worked Example — F1 Score

Precision $P = 1.0$, Recall $R = 0.01$ (a model that is perfect when it predicts but almost never predicts):

- AM $= (1.0 + 0.01)/2 = \mathbf{0.505}$ — falsely optimistic, hides the near-zero recall
- HM $= 2/(1/1.0 + 1/0.01) = 2/101 \approx \mathbf{0.020}$ — correctly near zero

[[evaluation-metrics-guide|F1]] uses HM precisely because a high AM can mask a near-zero component. F1 stays near zero unless *both* P and R are high.

---

## Intermediate

### $n$-value Harmonic Mean

For $n$ positive values $x_1, \ldots, x_n$:

$$\text{HM} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}$$

**Usage:** harmonic mean of rates when the denominator is constant. Example: average speed over equal *distances* (not equal times):

$$\bar{v}_{\text{harmonic}} = \frac{2 \cdot v_1 \cdot v_2}{v_1 + v_2}$$

### Geometric Mean for Multiplicative Quantities

$$\text{GM}(x_1, \ldots, x_n) = \left(\prod_{i=1}^n x_i\right)^{1/n} = \exp\!\left(\frac{1}{n}\sum_i \ln x_i\right)$$

**Usage:** compounding growth rates, normalizing scale-free quantities. Example: if an investment grows by factors $1.1$, $1.2$, $0.9$ over three years, the average annual factor is $\text{GM} = (1.1 \times 1.2 \times 0.9)^{1/3} \approx 1.062$.

### AM-GM Inequality — Connection to Jensen's Inequality

AM $\geq$ GM follows directly from [[math-convexity-jensen]]: $-\ln$ is convex, so by Jensen:

$$-\ln\!\left(\frac{a+b}{2}\right) \leq \frac{-\ln a - \ln b}{2} = -\ln\sqrt{ab}$$

Taking exponentials: $\frac{a+b}{2} \geq \sqrt{ab}$.

---

## Advanced

### When Each Mean Is the Right Choice

| Quantity | Right mean | Reason |
|----------|-----------|--------|
| Precision & Recall → F1 | HM | Near-zero component must dominate |
| Per-pixel IoU across classes | HM or macro-avg | Class imbalance; small classes matter |
| [[loss-dice\|Dice loss]] (segmentation) | HM | Dice = 2·\|P∩G\|/(|P|+|G|) = F1 for binary masks |
| Growth rates, BLEU | GM | Multiplicative structure |
| Standard average | AM | Additive, equal weights |

### Generalized Mean (Power Mean)

$$M_p(a, b) = \left(\frac{a^p + b^p}{2}\right)^{1/p}$$

| $p$ | Result |
|-----|--------|
| $-1$ | HM |
| $0$ (limit) | GM |
| $1$ | AM |
| $2$ | QM (root mean square) |
| $\to -\infty$ | $\min(a, b)$ |
| $\to +\infty$ | $\max(a, b)$ |

This unifies all classical means and shows HM and min are extremes of the same family.

### Harmonic Mean of p-values (Combined Testing)

When combining $K$ independent p-values $p_1, \ldots, p_K$, the harmonic mean test statistic:

$$T = \frac{K}{\sum_k 1/p_k}$$

is robust to dependence structure among tests, unlike Fisher's method which assumes independence. Used in [[hypothesis-testing]] meta-analyses.

---

## Links

- [[math-convexity-jensen]] — the AM-GM inequality is a special case of Jensen's inequality: $\log$ is concave, so $\frac{1}{n}\sum \log x_i \leq \log(\frac{1}{n}\sum x_i)$, i.e., $GM \leq AM$
- [[evaluation-metrics-guide]] — the harmonic mean underlies the F1 score: $F1 = 2/(1/P + 1/R)$; harmonic mean penalizes imbalances between precision and recall more than arithmetic mean
- [[loss-dice]] — Dice coefficient is equivalent to the F1 score, which uses harmonic mean; understanding mean types clarifies why Dice is preferred for imbalanced segmentation
- [[hypothesis-testing]] — test statistics often involve comparing sample means; the choice of mean (arithmetic vs. geometric) affects the test's sensitivity to outliers
