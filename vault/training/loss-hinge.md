---
title: Hinge Loss
tags: [hinge-loss, svm, margin, classification, structured-prediction]
aliases: [hinge loss, squared hinge, max-margin loss, SVM loss]
difficulty: 1
status: complete
related: [support-vector-machines, loss-cross-entropy, loss-mse, logistic-regression, generalization-bounds]
depends_on: [statistical-inference-mle, loss-cross-entropy, support-vector-machines]
---

# Hinge Loss

---

## Fundamental

### Definition

For binary classification with label $y \in \{-1, +1\}$ and score $f(x) \in \mathbb{R}$, the **hinge loss** is:

$$\ell_\text{hinge}(y, f) = \max(0, 1 - y \cdot f)$$

where $y \in \{-1,+1\}$ = true label, $f = f(x) \in \mathbb{R}$ = raw score (not probability), $y \cdot f$ = margin (positive if correct, negative if wrong), $1 - y \cdot f$ = margin violation.

Also written as $(1 - yf)_+$ where $(z)_+ = \max(0, z)$.

**Interpretation:**
- If the prediction $f$ has the correct sign and magnitude $|f| \geq 1$: loss = 0 (correctly classified with sufficient margin)
- If correct but $|f| < 1$: loss = $1 - |f|$ (inside the margin)
- If wrong sign: loss = $1 + |f|$ (misclassified, penalized proportionally)

The "1" defines a **margin** — the model must be confidently correct (by at least 1 unit) to incur zero loss.

### Hinge vs Cross-Entropy

| Property | Hinge | Cross-Entropy |
|----------|-------|---------------|
| Zero region | Yes — once margin satisfied | No — always positive |
| Gradient | Sparse (zero for correct) | Dense (always nonzero) |
| Probabilities | Not naturally calibrated | Directly estimates $p(y\|x)$ |
| Outlier sensitivity | Linear growth | Logarithmic growth |

Hinge loss produces **sparse solutions** — only support vectors (margin violators) contribute gradients. Cross-entropy continues updating even for perfectly classified points.

---

## Intermediate

### Connection to SVMs

Minimizing $\frac{1}{2}\|w\|^2 + C \sum_i \max(0, 1 - y_i f(x_i))$ is exactly the soft-margin SVM primal (see [[support-vector-machines]]). The $C$ trades off margin width and constraint violation.

The subgradient of the hinge loss:
$$\frac{\partial}{\partial f} \ell_\text{hinge} = \begin{cases} 0 & \text{if } yf \geq 1 \\ -y & \text{if } yf < 1 \end{cases}$$

Note: at $yf = 1$ the loss is not differentiable (a kink). Subgradient methods handle this by choosing any value in $[-y, 0]$.

### Squared Hinge Loss

$$\ell_\text{sq-hinge}(y, f) = \max(0, 1 - yf)^2$$

where $\max(0, 1-yf)$ = margin violation (0 for correctly classified points beyond margin), squared = more severe penalty for large violations, fully differentiable unlike hinge.

Properties:
- Differentiable everywhere (the kink is smoothed out)
- More strongly penalizes large margin violations (quadratic vs linear)
- Often performs similarly to hinge in practice but easier to optimize with gradient methods

### Multiclass Hinge (Weston-Watkins)

For $K$ classes with score vector $f(x) \in \mathbb{R}^K$ and true class $y$:

$$\ell = \sum_{k \neq y} \max(0, 1 + f_k - f_y)$$

where sum is over all wrong classes $k \neq y$, $f_y$ = score for the true class, $f_k$ = score for wrong class $k$, $1 + f_k - f_y > 0$ means class $k$ is within margin 1 of the true class.

Penalizes every incorrect class that scores within 1 unit of the correct class. Used in multiclass SVM and structured prediction.

---

## Advanced

### Structured Hinge Loss

For structured outputs (sequences, parse trees, bounding boxes), the **structured hinge loss** is:

$$\ell = \max_{y'} [\Delta(y, y') + f(x, y')] - f(x, y)$$

where $\Delta(y, y')$ is the task loss (e.g., Hamming distance, IoU complement). This is the foundation of structural SVMs — the loss is upper bounded by the hinge over all structured outputs.

The argmax over $y'$ is the **loss-augmented inference** problem — finding the output that is simultaneously likely under the model and dissimilar to the ground truth.

## Links

- [[support-vector-machines]] — hinge loss is the SVM objective: $\sum_i \max(0, 1 - y_i w^\top x_i)$; the max-margin hyperplane minimizes hinge loss + $\lambda\|w\|^2$
- [[loss-cross-entropy]] — cross-entropy goes to $-\infty$ for confident correct predictions; hinge loss has zero loss past margin 1, so correctly classified examples don't improve the objective
- [[statistical-inference-mle]] — hinge loss has no probabilistic interpretation (unlike cross-entropy); it optimizes a margin metric, not a likelihood
- [[logistic-regression]] — logistic regression uses log-loss (cross-entropy); SVM uses hinge loss; for linearly separable data, SVM gives a unique max-margin solution while logistic regression converges to infinity
- [[generalization-bounds]] — the SVM's PAC generalization bound is controlled by the margin $\rho$ and data norm $R$: $\epsilon \leq O(R^2/\rho^2 n)$; larger margin = better generalization bound
