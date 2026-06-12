---
title: Rademacher Complexity
tags: [rademacher-complexity, generalization-bounds, uniform-convergence, data-dependent-bounds]
aliases: [Rademacher complexity, empirical Rademacher complexity, uniform convergence bounds]
difficulty: 3
status: complete
related: [pac-learning, generalization-bounds, bias-variance-double-descent, neural-tangent-kernel, statistical-inference-mle]
---

# Rademacher Complexity

---

## Fundamental

### Definition

**Rademacher complexity** measures how well a hypothesis class $\mathcal{H}$ can fit random noise — the more it can fit noise, the more complex it is.

**Empirical Rademacher complexity** given training sample $S = \{x_1, \ldots, x_m\}$:

$$\hat{\mathcal{R}}_S(\mathcal{H}) = \mathbb{E}_{\sigma}\!\left[\sup_{h \in \mathcal{H}} \frac{1}{m} \sum_{i=1}^m \sigma_i h(x_i)\right]$$

where $\sigma_i \in \{-1, +1\}$ are i.i.d. **Rademacher random variables** (uniform random signs). The expectation is over the random $\sigma$.

**Intuition:** try to align each $h \in \mathcal{H}$ with a random sign pattern. If $\mathcal{H}$ is rich, some $h$ will correlate well with any sign pattern — high $\hat{\mathcal{R}}_S$.

**Population Rademacher complexity:** $\mathcal{R}_m(\mathcal{H}) = \mathbb{E}_S[\hat{\mathcal{R}}_S(\mathcal{H})]$ — expectation also over the random sample $S$.

---

## Intermediate

### Generalization Bound

The key theorem: for any $h \in \mathcal{H}$ and $\delta > 0$, with probability $\geq 1-\delta$ over training data:

$$\text{err}(h) \leq \hat{\text{err}}(h) + 2\hat{\mathcal{R}}_S(\mathcal{H}) + 3\sqrt{\frac{\ln(2/\delta)}{2m}}$$

**Two important properties:**
1. **Data-dependent:** $\hat{\mathcal{R}}_S$ is computed on the actual training data, not worst-case
2. **Algorithm-independent:** the bound holds uniformly over all $h \in \mathcal{H}$

The bound improves on VC-dimension bounds because Rademacher complexity can be small even when the class is rich — if the data distribution doesn't allow $\mathcal{H}$ to fit noise, the bound is tight.

### Rademacher Complexity of Common Classes

**Linear functions** in $\mathbb{R}^d$ with bounded norm $\|w\|_2 \leq B$, bounded inputs $\|x\|_2 \leq C$:
$$\mathcal{R}_m \leq \frac{BC}{\sqrt{m}}$$

**Neural networks:** for a depth-$L$ network with spectral norms $\sigma_i$ per layer:
$$\mathcal{R}_m \leq \frac{(\sqrt{2L} \prod_{i=1}^L \sigma_i) \|x\|_2}{\sqrt{m}}$$

The spectral norm product acts as the relevant complexity. This motivates spectral normalization as a regularization technique.

### Connection to PAC-Bayes

**PAC-Bayes bounds** offer another approach for data-dependent generalization: for a prior $P$ (before training) and posterior $Q$ (after training):

$$\mathbb{E}_{h \sim Q}[\text{err}(h)] \leq \mathbb{E}_{h \sim Q}[\hat{\text{err}}(h)] + \sqrt{\frac{\text{KL}(Q\|P) + \ln(m/\delta)}{2(m-1)}}$$

The KL divergence between posterior and prior quantifies how much training "moved" the distribution — models that only move the prior a little generalize well.

---

## Advanced

### Rademacher Complexity of Deep Networks in Practice

For trained deep networks, empirical $\hat{\mathcal{R}}_S$ is hard to compute exactly (requires maximization over $\mathcal{H}$). Approximations:
- **Margin-based:** use the achieved margin on training data as a proxy for complexity
- **Compression:** if the trained network can be compressed to $k$ bits, generalization is $O(\sqrt{k/m})$
- **Sharpness:** flat minima (small Hessian eigenvalues) correlate with smaller effective Rademacher complexity

**Empirical finding:** despite having VC dimension $\gg m$, trained networks in practice have small effective Rademacher complexity — explaining why they generalize despite overparameterization.

### Uniform Convergence and Its Limits

Rademacher complexity bounds are **uniform convergence** bounds: they show that empirical risk is close to true risk for all $h \in \mathcal{H}$ simultaneously. Recent work (Nagarajan & Kolter, 2019) shows that uniform convergence bounds cannot explain interpolating classifiers in the overparameterized regime — they must be vacuous there by definition. Algorithm-specific analyses (stability, implicit bias of SGD) are needed.

*See also: [[pac-learning]] · [[generalization-bounds]] · [[bias-variance-double-descent]] · [[neural-tangent-kernel]]*
