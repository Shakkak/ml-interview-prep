---
title: PAC Learning
tags: [pac-learning, sample-complexity, vc-dimension, generalization, computational-learning-theory]
aliases: [PAC, probably approximately correct, VC dimension, sample complexity, consistent learner]
difficulty: 3
status: complete
related: [generalization-bounds, bias-variance-double-descent, rademacher-complexity, cross-validation, statistical-inference-mle]
depends_on: [statistical-inference-mle, generalization-bounds]
---

# PAC Learning

---

## Fundamental

### The PAC Framework

**Probably Approximately Correct (PAC) learning** (Valiant, 1984) asks: for a learning algorithm to be reliable, how many training examples does it need?

**Setting:**
- Data $(x, y)$ drawn i.i.d. from unknown distribution $\mathcal{D}$
- Hypothesis class $\mathcal{H}$ (e.g., all decision trees of depth ≤ $d$)
- A concept $c^* \in \mathcal{H}$ generates the labels (realizable case)

**PAC learnability:** class $\mathcal{H}$ is PAC-learnable if there exists an algorithm $A$ such that for all $\epsilon, \delta > 0$, given:

$$m \geq m_\mathcal{H}(\epsilon, \delta)$$

samples, with probability $\geq 1 - \delta$, $A$ outputs $h$ with $\text{err}(h) \leq \epsilon$.

Interpretation: with probability at least $1-\delta$ (probably), the error is at most $\epsilon$ (approximately correct).

### Finite Hypothesis Classes

For a finite class $|\mathcal{H}| < \infty$, a consistent learner (outputs $h$ with zero training error) needs:

$$m \geq \frac{1}{\epsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)$$

where $m$ = number of training samples needed, $\epsilon$ = error tolerance (we want $\text{err}(h) \leq \epsilon$), $\delta$ = failure probability (the bound holds with probability $\geq 1-\delta$), $|\mathcal{H}|$ = number of hypotheses in the class.

**Key insight:** complexity is $\log|\mathcal{H}|$ — not $|\mathcal{H}|$. A class of $2^{100}$ hypotheses (exponentially many) only requires ~100 samples to learn. The description length of the best hypothesis governs sample complexity.

---

## Intermediate

### VC Dimension

For infinite hypothesis classes (e.g., linear classifiers in $\mathbb{R}^d$), $\log|\mathcal{H}| = \infty$. The relevant complexity measure is the **VC (Vapnik-Chervonenkis) dimension**.

**Shattering:** $\mathcal{H}$ shatters a set $S = \{x_1, \ldots, x_k\}$ if for every labeling $y \in \{0,1\}^k$, some $h \in \mathcal{H}$ achieves that labeling on $S$.

**VC dimension** $d_\mathcal{H}$: the largest $k$ such that $\mathcal{H}$ shatters some set of size $k$.

**Examples:**
- Halfspaces in $\mathbb{R}^d$: $d_\text{VC} = d + 1$
- Decision trees with $k$ leaves: $d_\text{VC} = O(k \log k)$
- Neural networks with $W$ weights: $d_\text{VC} = O(W \log W)$
- Fully-connected depth-$L$ network: $d_\text{VC} = O(WL \log W)$

**PAC bound with VC dimension (Fundamental Theorem of Statistical Learning):**

$$m_\mathcal{H}(\epsilon, \delta) = O\!\left(\frac{d_\mathcal{H} + \log(1/\delta)}{\epsilon^2}\right)$$

where $d_\mathcal{H}$ = VC dimension of the hypothesis class, $\epsilon$ = error tolerance, $\delta$ = failure probability. The $\epsilon^2$ denominator (vs $\epsilon$ in the realizable case) reflects the harder agnostic setting.

The class is PAC-learnable if and only if $d_\mathcal{H} < \infty$.

### Agnostic PAC Learning

In the realizable case, $c^* \in \mathcal{H}$. In practice, the true concept may not be in $\mathcal{H}$ — the **agnostic** case.

**Agnostic goal:** find $h$ minimizing $\text{err}(h) \leq \min_{h' \in \mathcal{H}} \text{err}(h') + \epsilon$

Sample complexity scales as $O(1/\epsilon^2)$ vs $O(1/\epsilon)$ in the realizable case — slower, reflecting harder problem. The agnostic bound uses the **uniform convergence** property.

---

## Advanced

### Connection to Rademacher Complexity

VC dimension is combinatorial and worst-case. **Rademacher complexity** (see [[rademacher-complexity]]) provides data-dependent bounds that can be tighter:

$$\text{GenBound} = \hat{\mathcal{R}}_m(\mathcal{H}) + O\!\left(\sqrt{\frac{\log(1/\delta)}{m}}\right)$$

Unlike VC bounds (which depend only on the class), Rademacher bounds depend on the actual training data distribution — they can show that certain functions that are theoretically complex are empirically simple.

### PAC Bounds for Deep Learning

Deep networks have large VC dimension ($\sim WL$) — PAC bounds give vacuous (useless) generalization guarantees for modern networks. A ResNet-50 with $W = 25M$ has VC dim $\sim 10^8$, but it generalizes well on $\sim 10^6$ training samples.

This **generalization puzzle** motivates newer complexity measures: sharpness, PAC-Bayes bounds, compressibility bounds, and algorithmic stability — which give tighter bounds matching observed behavior.

## Links

- [[statistical-inference-mle]] — PAC learning bounds the generalization error of the empirical risk minimizer; MLE is ERM under log-loss
- [[generalization-bounds]] — PAC bounds are the foundational generalization bounds; Rademacher complexity and PAC-Bayes bounds are tighter extensions
- [[rademacher-complexity]] — Rademacher complexity provides data-dependent PAC bounds; it measures how well the hypothesis class can fit random labels on the training set
- [[bias-variance-double-descent]] — classical PAC theory predicts generalization improves with fewer parameters; double descent violates this prediction for overparameterized models
- [[cross-validation]] — CV is the empirical counterpart of PAC bounds: it estimates the generalization error without making distributional assumptions about the hypothesis class
