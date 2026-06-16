---
title: Outlier and Anomaly Detection
tags: [outlier-detection, anomaly-detection, isolation-forest, lof, statistical-outliers]
aliases: [outlier detection, anomaly detection, novelty detection, IQR method, isolation forest, LOF]
difficulty: 1
status: complete
depends_on: [distributions-overview, statistical-inference-mle]
related: [anomaly-detection, feature-preprocessing, distributions-overview, clustering, hypothesis-testing]
---

# Outlier and Anomaly Detection

---

## Fundamental

### Types of Outliers

- **Point outliers:** individual values that deviate from the bulk of the data
- **Contextual outliers:** values that are normal globally but anomalous in context (e.g., temperature of 25°C is normal in summer, anomalous in winter)
- **Collective outliers:** a sequence of values that is anomalous as a group but each individually appears normal

**Applications:** fraud detection, network intrusion, sensor fault detection, medical diagnostics, data cleaning before ML training.

### Univariate Methods

**Z-score:** standardize values and threshold by standard deviations from the mean:
$$z_i = \frac{x_i - \mu}{\sigma}, \quad \text{outlier if } |z_i| > 3$$

where $x_i$ = the observed value, $\mu$ = sample mean, $\sigma$ = sample standard deviation, and $z_i$ = number of standard deviations from the mean. Assumes Gaussian distribution. Sensitive to the outliers themselves (they inflate $\mu$ and $\sigma$).

**Modified Z-score (robust):** use median and MAD (Median Absolute Deviation):
$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}, \quad \text{MAD} = \text{median}(|x_i - \tilde{x}|)$$

where $\tilde{x}$ = sample median, $\text{MAD}$ = Median Absolute Deviation (the median of absolute deviations from the median — a robust spread measure that ignores outlier inflation), and $0.6745$ = a scaling constant so that $M_i \approx z_i$ for Gaussian data (since $\text{MAD} \approx 0.6745\sigma$ for a Gaussian). Threshold: $|M_i| > 3.5$. MAD is robust to outliers, making this method self-consistent.

**IQR method:** outlier if $x < Q_1 - 1.5 \cdot \text{IQR}$ or $x > Q_3 + 1.5 \cdot \text{IQR}$ (Tukey's fences). Box plots visualize this. Multiplier 3.0 for "far outliers."

---

## Intermediate

### Isolation Forest

**Isolation Forest (Liu et al., 2008):** exploits the intuition that outliers are few and different — they are **easier to isolate** (require fewer splits).

**Algorithm:**
1. Randomly select a feature, randomly select a split value between min and max
2. Partition the data by this split
3. Repeat recursively until each point is isolated (in its own leaf) or a depth limit is reached
4. **Anomaly score:** mean depth across many random trees; outliers have short isolation paths

Score: $s(x) = 2^{-E[h(x)] / c(n)}$ where $h(x)$ = isolation depth of point $x$ in a single random tree (number of splits needed to isolate it), $E[h(x)]$ = expected depth averaged across all trees in the forest, $c(n) = 2\ln(n-1) + 0.5772 - 2(n-1)/n$ = expected depth of an unsuccessful binary search tree of size $n$ (the normalization constant ensuring scores are in $(0, 1)$), and $n$ = number of training points. Scores near 1 = outlier (short isolation path); near 0.5 = normal (average depth); near 0 = certain inlier.

**Properties:**
- Linear $O(n \log n)$ complexity
- Works in high dimensions (random features automatically handle many dimensions)
- No distance computations — effective when Euclidean distance is not meaningful
- Handles high-dimensional sparse data well

### Local Outlier Factor (LOF)

**LOF (Breunig et al., 2000):** measures how isolated a point is relative to its local neighborhood. Outliers are in low-density regions compared to their neighbors.

For point $p$ with $k$ nearest neighbors:
1. Compute reachability distance: $\text{reach-dist}_k(p, o) = \max(d_k(o), d(p, o))$ (at least $k$-nn distance of $o$)
2. Local reachability density: $\text{lrd}_k(p) = 1 / \text{avg reach-dist}$ from $p$ to its neighbors
3. LOF score: $\text{LOF}_k(p) = \text{avg}_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}$

LOF ≈ 1: similar density to neighbors (normal). LOF >> 1: much lower density than neighbors (outlier).

**Advantage over global methods:** LOF detects local outliers that are anomalous relative to their neighborhood, not just globally.

---

## Advanced

### Deep Learning Approaches

**Autoencoder-based:** train autoencoder on normal data; anomalies have high reconstruction error. Limitation: autoencoders sometimes reconstruct anomalies well if they share features with normal data.

**Deep SVDD:** learn a neural mapping such that normal data maps into a small hypersphere; anomalies fall outside.

**Flow-based:** train normalizing flow on normal data; low-likelihood samples under the learned distribution are anomalies. Theoretically principled but sensitive to flow architecture choice.

### Threshold Selection

Outlier detectors produce **scores**, not binary labels. Setting the threshold requires:
- Domain knowledge (acceptable FPR for the application)
- Labeled anomaly examples (if available — rare in practice)
- Statistics of the score distribution (e.g., top 1% as anomalies)

**Operational consideration:** in fraud detection, the cost of a false negative (missed fraud) is far higher than a false positive (blocked legitimate transaction). Set threshold based on the cost matrix, not accuracy.

## Links

- [[distributions-overview]] — statistical outlier detection (IQR, Z-score) assumes a known distribution; the thresholds derive from quantiles of that distribution
- [[statistical-inference-mle]] — parametric outlier detection fits a distribution via MLE and flags examples with low log-likelihood; the threshold is a chi-squared quantile
- [[anomaly-detection]] — anomaly detection is the generative/probabilistic counterpart; outlier detection covers simpler rule-based and proximity-based methods
- [[clustering]] — LOF (Local Outlier Factor) measures density relative to neighbors; it is a clustering-proximity method without an explicit generative model
- [[feature-preprocessing]] — outlier detection is sensitive to feature scale; standardize before computing distance-based scores (LOF, k-NN distance)
- [[hypothesis-testing]] — Grubbs test and Dixon Q-test formalize outlier detection as hypothesis tests with $p$-values; they require distributional assumptions
