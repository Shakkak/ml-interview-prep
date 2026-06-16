---
title: Support Vector Machines
tags: [svm, support-vector-machine, kernel, margin, classification]
aliases: [SVM, support vector machine, max-margin classifier, hard-margin SVM, soft-margin SVM]
difficulty: 2
status: complete
related: [kernel-methods, lagrangian-optimization, logistic-regression, linear-regression, parametric-nonparametric, loss-mse]
depends_on: [lagrangian-optimization, kernel-methods, linear-algebra-fundamentals]
---

# Support Vector Machines

---

## Fundamental

### The Maximum Margin Principle

For binary classification with labels $y_i \in \{-1, +1\}$ and a linear classifier $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$, the **decision boundary** is where $f = 0$. Many hyperplanes separate linearly separable data — which one should we choose?

**Margin:** the distance from the decision boundary to the nearest training point. For a unit vector $\hat{\mathbf{w}}$, the margin is $2 / \|\mathbf{w}\|$.

**The SVM principle:** choose the hyperplane that maximizes the margin. Intuitively: a larger margin means the classifier is more confident, generalizes better to small perturbations, and has a smaller Vapnik-Chervonenkis dimension — tighter generalization bounds.

### Hard-Margin SVM

Assume data is linearly separable. The max-margin hyperplane is found by solving:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \quad \forall i$$

where $\mathbf{w} \in \mathbb{R}^d$ = weight vector (normal to the decision boundary), $b$ = bias term, $\|\mathbf{w}\|^2$ = squared norm (minimizing this maximizes the margin $2/\|\mathbf{w}\|$), $y_i \in \{-1, +1\}$ = class label of training point $i$, and the constraint $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ requires each point to be correctly classified and at least 1 unit away from the boundary in the scaled coordinate system.

The constraints ensure all points are on the correct side with margin $\geq 1/\|\mathbf{w}\|$. Points where $y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$ are called **support vectors** — they are exactly on the margin boundary and are the only training points that determine the solution.

**Key insight:** the optimal hyperplane depends only on the support vectors (typically $\ll N$ total points). Remove any non-support-vector training point and the solution is unchanged. This makes SVMs robust to outliers that are not at the margin.

### Soft-Margin SVM

Real data is rarely linearly separable. The soft-margin SVM allows misclassifications via **slack variables** $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

where $\xi_i \geq 0$ = slack variable for point $i$ (amount by which constraint $i$ is violated; $\xi_i = 0$ means correctly classified outside the margin, $\xi_i = 1$ means exactly on the boundary, $\xi_i > 1$ means misclassified) and $C > 0$ = regularization hyperparameter (penalty per unit of slack).

$C > 0$ is the **regularization hyperparameter:**
- Large $C$: small tolerance for misclassification → narrow margin, may overfit
- Small $C$: large tolerance → wide margin, more regularized

**Hinge loss connection:** the soft-margin SVM minimizes $\frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, 1 - y_i f(\mathbf{x}_i))$. The $\max(0, 1 - yf)$ term is the **hinge loss** — zero when the point is correctly classified with sufficient margin, linear otherwise.

---

## Intermediate

### Dual Formulation and Support Vectors

Applying Lagrangian duality (see [[lagrangian-optimization]]) to the primal SVM yields the dual:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$
$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

where $\alpha_i \geq 0$ = dual variable (Lagrange multiplier) for training point $i$, nonzero only for support vectors; $y_i \in \{-1,+1\}$ = label; $\mathbf{x}_i^\top \mathbf{x}_j$ = dot product (replaced by $k(\mathbf{x}_i, \mathbf{x}_j)$ for kernel SVM); $C$ = box constraint from soft-margin.

The solution: $\hat{\mathbf{w}} = \sum_i \alpha_i y_i \mathbf{x}_i$ — a linear combination of training examples. By KKT conditions:
- $\alpha_i = 0$: point is outside the margin (correctly classified, irrelevant)
- $0 < \alpha_i < C$: point is exactly on the margin (a support vector)
- $\alpha_i = C$: point is inside the margin or misclassified (a margin violation)

**Predictions** only require dot products with support vectors: $f(\mathbf{x}) = \sum_{i:\alpha_i > 0} \alpha_i y_i \mathbf{x}_i^\top \mathbf{x} + b$

### The Kernel Trick

The dual depends on data only through **dot products** $\mathbf{x}_i^\top \mathbf{x}_j$. Replacing these with a **kernel function** $k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ implicitly maps to a high- (or infinite-) dimensional feature space without computing $\phi$ explicitly:

$$f(\mathbf{x}) = \sum_{i:\alpha_i > 0} \alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b$$

where sum runs only over support vectors ($\alpha_i > 0$), $k(\mathbf{x}_i, \mathbf{x})$ = kernel similarity between support vector $\mathbf{x}_i$ and query $\mathbf{x}$, $b$ = bias.

**Common kernels:**
- Polynomial: $k(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^d$
- Radial Basis Function (RBF/Gaussian): $k(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\|\mathbf{x} - \mathbf{x}'\|^2)$ — infinite-dimensional feature space
- Sigmoid: $k(\mathbf{x}, \mathbf{x}') = \tanh(\gamma\mathbf{x}^\top\mathbf{x}' + r)$

See [[kernel-methods]] for full kernel theory and Mercer's theorem.

### SVM vs Logistic Regression

| Property | SVM | Logistic Regression |
|----------|-----|---------------------|
| Loss | Hinge (sparse, 0 for well-classified) | Log-loss (dense, never exactly 0) |
| Outputs | Raw scores; no calibrated probabilities | Calibrated probabilities |
| Regularization | Implicit in margin objective | Explicit L2/L1 penalty |
| Solution | Sparse (support vectors only) | Dense (all training points) |
| Kernel | Natural via dual formulation | Less natural (approximations) |
| Speed at prediction | $O(|\text{SVs}| \cdot d)$ | $O(d)$ |

Logistic regression wins when well-calibrated probabilities are needed. SVMs win when the decision boundary is the only concern and kernel methods are required.

---

## Advanced

### Solving SVMs: SMO

The SVM dual is a quadratic program (QP). For large datasets, general QP solvers are $O(N^3)$. **Sequential Minimal Optimization (SMO, Platt 1998):** at each step, select the two most violating $\alpha_i$ pairs and update them analytically. This reduces the $N$-dimensional QP to a sequence of 2D QPs with closed-form solutions.

The heuristic for selecting the pair: choose $\alpha_i$ that most violates the KKT conditions, then choose $\alpha_j$ that maximizes the step size. The algorithm converges in $O(N)$ to $O(N^2)$ iterations in practice.

### Structural SVMs and Multi-Class

Multi-class SVMs extend the binary formulation:
- **One-vs-All (OvA):** train $K$ binary SVMs, predict the class with highest score. Fast but doesn't account for class interactions.
- **One-vs-One (OvO):** train $\binom{K}{2}$ SVMs, predict by majority vote. Better calibrated for small $K$.
- **Crammer & Singer multi-class SVM:** single optimization that penalizes the margin between the true class and the best incorrect class — theoretically cleaner.

**Structural SVMs** (Taskar et al., 2004; Tsochantaridis et al., 2005) extend to structured outputs (sequences, trees, graphs): find a weight vector that maximizes the margin between the true structure and all other valid structures simultaneously.

### SVMs in the Neural Network Era

SVMs dominated from ~1995–2012 for many tasks. Deep learning surpassed SVMs because:
1. Deep networks learn better features, eliminating the need for manual kernel design
2. SVMs scale as $O(N^2)$–$O(N^3)$ in training (kernel matrix); neural networks scale linearly via SGD
3. Kernel approximations (random Fourier features, see [[taylor-series]]) are needed for large-scale kernels

SVMs remain competitive for: small datasets ($N < 10^4$), high-dimensional sparse inputs (text), structured prediction, and problems where interpretability of the decision boundary matters.

---

## Links

- [[lagrangian-optimization]] — SVM training is a quadratic programming problem solved via Lagrangian duality; the KKT conditions yield the support vectors as the active constraints
- [[kernel-methods]] — kernel SVMs implicitly map inputs to a high-dimensional feature space via the kernel trick; the dual formulation only requires inner products $k(x_i, x_j)$, never the explicit feature map
- [[linear-algebra-fundamentals]] — the maximum-margin hyperplane is $w^\top x + b = 0$; the margin is $2/\|w\|$; maximizing margin = minimizing $\|w\|^2$, a quadratic objective in $w$
- [[logistic-regression]] — LR minimizes log-loss; SVM minimizes hinge loss; for separable data, SVM gives a unique maximum-margin solution while LR's solution diverges
- [[generalization-bounds]] — SVM generalization is governed by the margin: larger margin = tighter VC dimension bound; this theoretical result motivated max-margin classification
- [[parametric-nonparametric]] — SVMs are semi-parametric: the decision boundary has $s$ parameters (support vectors), growing with training data; kernel SVMs are non-parametric in the limit
