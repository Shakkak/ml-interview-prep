---
title: Generalization Bounds — PAC Learning, VC Dimension, Rademacher Complexity
tags: [generalization, pac-learning, vc-dimension, rademacher-complexity, bias-variance, learning-theory]
aliases: [PAC learning, VC dimension, Rademacher complexity, generalization bound, sample complexity, growth function]
difficulty: 3
status: complete
related: [bias-variance-double-descent, regularization-weight-decay, regularization-dropout, kernel-methods, spectral-bias]
---

# Generalization Bounds — PAC Learning, VC Dimension, Rademacher Complexity

---

## Fundamental

A model's empirical risk (training loss) can be made arbitrarily small by memorization. What matters is **generalization** — performance on unseen data.

**Generalization gap:** $R[f] - \hat{R}[f]$, where $R$ = true risk, $\hat{R}$ = empirical risk.

**PAC Learning framework (Valiant, 1984):** Probably Approximately Correct. A hypothesis class $\mathcal{H}$ is PAC-learnable if there exists an algorithm $A$ such that: for any $\epsilon > 0$ (accuracy), $\delta > 0$ (confidence), and any distribution $\mathcal{D}$, given $m \geq m(\epsilon, \delta)$ i.i.d. samples, $A$ outputs $h$ satisfying:

$$P\left[R[h] \leq \min_{h^* \in \mathcal{H}} R[h^*] + \epsilon\right] \geq 1 - \delta$$

**Sample complexity for finite hypothesis classes:** how many training examples needed to learn within $\epsilon$ accuracy with probability $\geq 1-\delta$:

$$m(\epsilon, \delta) \geq \frac{1}{\epsilon}\left(\log|\mathcal{H}| + \log\frac{1}{\delta}\right)$$

For infinite hypothesis classes, need a notion of "effective size."

**VC Dimension:** shattering means $\mathcal{H}$ can produce every possible binary labeling of a set $S$. VC dimension = the size of the largest set $\mathcal{H}$ can shatter.

Examples:
- Linear classifiers in $\mathbb{R}^d$: $\text{VC} = d + 1$
- Axis-aligned rectangles in $\mathbb{R}^2$: $\text{VC} = 4$
- A single threshold on $\mathbb{R}$: $\text{VC} = 1$

**VC generalization bound:**

$$R[h] \leq \hat{R}[h] + \sqrt{\frac{d \log(m/d) + \log(1/\delta)}{m}}$$

Interpretation: empirical risk (how well the model fits training data) plus a complexity penalty (how expressive the model class is). Larger $\mathcal{H}$ (higher $d$) → tighter fit to training data but larger penalty.

---

## Intermediate

**Sauer's Lemma:** the growth function $\Pi_\mathcal{H}(m)$ (maximum number of distinct labelings $\mathcal{H}$ can produce on any $m$ points) satisfies:

$$\Pi_\mathcal{H}(m) \leq \sum_{i=0}^{d} \binom{m}{i}$$

For $m > d$: $\Pi_\mathcal{H}(m) \leq (em/d)^d$ — polynomial in $m$, not exponential. This polynomial growth is what makes learning possible from finite samples.

**Rademacher Complexity:** VC dimension is a combinatorial measure that ignores the data distribution. Rademacher complexity gives tighter, data-dependent bounds.

Empirical Rademacher complexity:
$$\hat{\mathfrak{R}}_m(\mathcal{H}) = \mathbb{E}_\sigma\!\left[\sup_{h \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^m \sigma_i h(x_i)\right]$$

where $\sigma_i \in \{-1, +1\}$ are i.i.d. uniform (Rademacher variables).

**Interpretation:** how well the best hypothesis in $\mathcal{H}$ can fit random ±1 labels on the training data. High complexity = can fit noise = likely to overfit.

**Rademacher bound:**

$$R[h] \leq \hat{R}[h] + 2\hat{\mathfrak{R}}_m(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2m}}$$

Why better than VC: Rademacher complexity depends on both $\mathcal{H}$ **and** the actual training data distribution. For data concentrated in a low-dimensional region, even a high-VC-dimension class may have small Rademacher complexity on that data.

**Bias-Variance-Complexity tradeoff:**

$$R[h] = \underbrace{\text{Bias}^2}_{\min_{h \in \mathcal{H}} R[h]} + \underbrace{\text{Variance}}_{R[h] - \min_{h \in \mathcal{H}} R[h]}$$

Bias decreases as $\mathcal{H}$ grows richer (better approximation). Variance increases as $\mathcal{H}$ grows richer (harder to estimate from finite data). Classical view: optimal complexity at the U-curve minimum.

---

## Advanced

**Why these bounds fail for neural networks:** neural networks often have VC dimension $\gg m$ (more expressive than training data). Classical bounds predict terrible generalization. Yet neural nets generalize well in practice.

What VC bounds miss:
- They are worst-case over all classifiers in $\mathcal{H}$ — SGD finds specific, well-generalizing solutions
- Implicit regularization: SGD with small learning rate biases toward low-norm, simple solutions
- The bound is on the entire class; the actual solution uses far less capacity

**Norm-based bounds:** bounds using weight norms are tighter for neural nets:

$$R[h] \leq \hat{R}[h] + O\!\left(\frac{B \cdot \prod_l \|W_l\|_\sigma}{\sqrt{m}}\right)$$

where $\|W_l\|_\sigma$ is the spectral norm of the $l$-th weight matrix and $B$ is the input norm. These bounds explain why weight decay and spectral normalization improve generalization — they explicitly constrain the product of spectral norms, tightening the bound.

**Margin bounds:** for classifiers achieving margin $\gamma$ on training data (all training points correctly classified with confidence $\gamma$):

$$R[h] \leq \frac{\hat{R}_\gamma[h]}{1} + O\!\left(\frac{\|W\|_F}{\gamma\sqrt{m}}\right)$$

where $\hat{R}_\gamma$ is the fraction of training points with margin less than $\gamma$. The bound improves with larger margin and smaller Frobenius norm. This provides theoretical justification for large-margin training (SVM objective) and weight decay.

**Double descent from learning theory perspective (Belkin et al., 2019):** in overparameterized regimes ($d \gg m$), the VC bound is vacuously large. But the actual solution found by SGD has norm-based complexity far smaller than the VC bound suggests. The minimum-norm interpolant has Rademacher complexity $O(\|w^*\|_2/\sqrt{m})$ where $\|w^*\|_2$ decreases as $d$ increases (the minimum-norm solution spreads weight across more parameters). This explains why adding parameters can reduce generalization error.

**PAC-Bayes bounds:** instead of worst-case bounds over $\mathcal{H}$, PAC-Bayes considers a prior distribution $P$ over hypotheses and a posterior $Q$ after seeing data. The McAllester bound:

$$\mathbb{E}_{h \sim Q}[R[h]] \leq \mathbb{E}_{h \sim Q}[\hat{R}[h]] + \sqrt{\frac{KL(Q\|P) + \log(m/\delta)}{2m-1}}$$

PAC-Bayes bounds are some of the tightest available for neural networks because the posterior $Q$ concentrated on the found solution is a very informative distribution. The KL penalty connects to Bayesian model comparison.

---

*See also: [[bias-variance-double-descent]] · [[regularization-weight-decay]] · [[kernel-methods]] · [[spectral-bias]]*
