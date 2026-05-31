---
title: Data Visualization Guide
tags: [visualization, statistics, data-analysis, plots, charts, tsne, umap, calibration]
aliases: [charts, plots, visualization guide, chart selection, Q-Q plot, reliability diagram, drift detection]
difficulty: 1
status: complete
related: [statistical-inference-mle, distributions-overview, evaluation-metrics-guide, model-calibration]
---

# Data Visualization Guide

---

## Fundamental

### Chart Selection Decision Tree

```
What are you showing?
├── One variable
│   ├── Continuous → Histogram + KDE overlay
│   ├── Categorical → Bar chart (sorted by frequency)
│   └── Time series → Line plot
├── Two variables
│   ├── Both continuous → Scatter (< 10k pts) or Hexbin (> 10k)
│   ├── Continuous + Categorical → Box plot (many groups) or Violin (shape matters)
│   └── Both categorical → Heatmap / mosaic plot
├── Many variables
│   ├── ≤ 5 features → Pair plot (scatter matrix)
│   └── Many → Correlation heatmap
└── Model evaluation
    ├── Binary → ROC + PR curve (on same figure)
    ├── Calibration → Reliability diagram
    ├── Training → Loss/accuracy curves (log scale for loss)
    └── Error distribution → Residual histogram + Q-Q plot
```

### Distribution Visualization

**Histogram vs KDE:**
- Histogram: exact counts, easy to interpret, bin width choice matters.
- KDE: smooth, good for comparing multiple distributions, but can assign density to impossible values (negative counts, out-of-range ages).
- Rule of thumb for bin count: Sturges $\lceil\log_2 n + 1\rceil$; Scott $3.5\hat\sigma n^{-1/3}$ (better for large $n$).

**Box plot vs Violin:**
- Box plot: shows median, quartiles, whiskers, outliers. Hides bimodality.
- Violin: shows full estimated density shape. Reveals bimodality, asymmetry.
- Use box plot when comparing 8+ groups. Use violin when the shape of the distribution matters.

### Log Scale: When and Why

Use log scale when:
1. Values span multiple orders of magnitude (loss from 10 → 0.001).
2. Percentage changes are what matter (learning rates, stock returns).
3. Underlying relationship is a power law ($y = cx^\alpha$ → straight line on log-log).
4. Exponential decay ($y = ae^{-bt}$ → straight line on log-linear).

**Never:** log scale on bar charts (bars start at 1 not 0 — areas are misleading). Log scale cannot show zero or negative values.

---

## Intermediate

### Q-Q Plot Interpretation

A Q-Q plot plots sample quantiles vs theoretical (Gaussian) quantiles. Diagonal = perfect Gaussian.

| Pattern | Meaning |
|---|---|
| Points curve above right, below left | Right skew (positive skew, long right tail) |
| Points curve below right, above left | Left skew |
| S-curve (above at both extremes) | Heavy tails (leptokurtic, excess kurtosis) |
| Inverted S-curve (below at both extremes) | Light tails (platykurtic) |
| Single far-above point at top right | Large outlier |

**Worked example:** $n=5$ data points $\{1, 2, 3, 5, 20\}$. Expected Gaussian quantiles: $z_{0.1}, z_{0.3}, z_{0.5}, z_{0.7}, z_{0.9} = -1.28, -0.52, 0, 0.52, 1.28$. The point $(1.28, 20)$ lies far above the line — the outlier 20 deviates dramatically from what a Gaussian would predict at the 90th percentile.

### Error Bars: Choosing the Right One

For 5 CV fold scores $[0.82, 0.85, 0.83, 0.87, 0.84]$: $\bar{x}=0.842$, $s=0.018$.

| Error bar | Formula | Value | Use when |
|---|---|:-:|---|
| SD | $s$ | ±0.018 | Showing variability of individual measurements |
| SEM | $s/\sqrt{n}$ | ±0.0081 | Uncertainty about the mean (narrower) |
| 95% CI | $t_{.025,4} \cdot s/\sqrt{n} = 2.78 \times 0.0081$ | ±0.022 | Comparing models; statistical significance |

For model comparison, always report 95% CI — it directly communicates whether the improvement is statistically meaningful. SD is the right choice when you want to show how much individual fold scores vary.

### Heatmap Best Practices

**Colormap selection:**
- Correlation (−1 to 1): diverging (RdBu, coolwarm), centered at 0.
- Probabilities / counts (0 to max): sequential (Blues, viridis).
- Signed values (can be negative): diverging.

**Confusion matrix:** annotate raw counts in cells; use row-normalized values (recall per class) as the color scale. Raw counts alone hide class imbalance; row-normalization alone hides absolute scale.

### t-SNE and UMAP: What They Preserve

| | t-SNE | UMAP |
|--|---|---|
| Local structure | ✓ Excellent | ✓ Excellent |
| Global structure | ✗ Poor | ✓ Better |
| Speed | $O(n^2)$ naive, $O(n\log n)$ with BH-tree | Fast ($O(n)$ approx) |
| Key hyperparameter | Perplexity (5–50) | n_neighbors, min_dist |

**Valid conclusions:** clusters are separated → features discriminate; two sub-clusters within a class → possible subpopulations. **Invalid conclusions:** relative distances between clusters are not meaningful in t-SNE. Cluster size is an artifact of the algorithm, not a real data property.

---

## Advanced

### Monitoring Distribution Drift

**Feature drift detection:** for each feature, overlay train vs production distributions. Quantify with the **Kolmogorov-Smirnov (KS) statistic** — the maximum difference between the two empirical CDFs. Plot KS statistics sorted descending; investigate the top-10 most drifted features.

```python
from scipy.stats import ks_2samp
ks_stat, p_val = ks_2samp(train[feature], prod[feature])
```

**Score distribution drift:** plot a heatmap of model output score distributions over time (x-axis = day, y-axis = score bin, color = frequency). A shift in the predicted positive rate often precedes a drop in model accuracy by several weeks — this is an early-warning signal.

**Population Stability Index (PSI):** industry-standard metric for monitoring drift:
$$\text{PSI} = \sum_b (A_b - E_b)\ln(A_b/E_b)$$
where $A_b$ = actual (production) fraction in bin $b$, $E_b$ = expected (training) fraction. PSI < 0.1: stable; 0.1–0.25: slight drift, monitor; > 0.25: significant shift, retrain.

### Calibration Visualization

Reliability diagram: bin predictions by confidence $[0,0.1), [0.1,0.2), \ldots, [0.9,1.0]$; for each bin plot (mean confidence, fraction correct).

```
Perfect:     Overconfident:    Underconfident:
 ╱              ╱                    ╱╲
╱           ╱╲╱             ╲╱╲╱
```

**Interpreting over/underconfidence:**
- Points below diagonal: model says 0.8, actually 0.6 — overconfident, apply temperature $T > 1$.
- Points above diagonal: model says 0.4, actually 0.6 — underconfident, apply $T < 1$.
- Non-monotone pattern: the model's probability estimates are not well-ordered — requires more than temperature scaling (e.g., isotonic regression calibration).

### Pair Plots for Feature Exploration

A scatter matrix plots all pairwise relationships between $d$ features, with histograms (or KDEs) on the diagonal. At $d=10$, you get 45 scatter plots — still readable. At $d=100$, use correlation heatmap instead.

Color by target class to see separability. Look for: (1) clear linear separation → linear model may suffice; (2) non-linear structure → tree or kernel methods; (3) correlated features → consider PCA or regularization; (4) outliers in scatterplots → may need robust preprocessing.

---

*See also: [[evaluation-metrics-guide]] · [[statistical-inference-mle]] · [[distributions-overview]] · [[model-calibration]]*
