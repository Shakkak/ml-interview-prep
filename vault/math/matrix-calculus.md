---
title: Matrix Calculus
tags: [linear-algebra, calculus, backpropagation, optimization, jacobian, hessian]
aliases: [matrix calculus, vector derivatives, Jacobian, Hessian, gradient of matrix, denominator layout]
difficulty: 3
status: complete
related: [linear-algebra-fundamentals, eigenvalues-pca, backpropagation, backpropagation-advanced, lagrangian-optimization]
---

# Matrix Calculus

---

## Fundamental

We use **denominator layout** (Jacobian layout): the gradient $\nabla_x f$ has the same shape as $x$. For scalar $f$ and vector $x \in \mathbb{R}^n$: $\nabla_x f \in \mathbb{R}^n$.

### Scalar-by-Vector Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$: $(\nabla_x f)_i = \partial f / \partial x_i$.

**Key identities (memorize these):**

| $f$ | $\nabla_x f$ |
|---|---|
| $a^\top x$ | $a$ |
| $x^\top a$ | $a$ |
| $x^\top x = \|x\|^2$ | $2x$ |
| $x^\top A x$ (symmetric $A$) | $2Ax$ |
| $\|Ax - b\|^2$ | $2A^\top(Ax-b)$ |
| $\ln(x^\top a)$ | $a/(x^\top a)$ |

**Derivation of $\nabla_x(x^\top Ax)$:** write $f = \sum_{i,j} A_{ij} x_i x_j$. Differentiate w.r.t. $x_k$:
$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = (Ax)_k + (A^\top x)_k = ((A+A^\top)x)_k$$

For symmetric $A$: $(A+A^\top) = 2A$, so $\nabla_x(x^\top Ax) = 2Ax$.

**Derivation of $\nabla_x\|Ax - b\|^2$:** let $r = Ax - b$. $f = \|r\|^2 = \sum_i r_i^2$.

$\partial r_i / \partial x_k = A_{ik}$, so $\partial f / \partial x_k = 2\sum_i r_i A_{ik} = 2(A^\top r)_k$.

$\nabla_x\|Ax-b\|^2 = 2A^\top(Ax-b)$.

Setting to zero: $A^\top Ax = A^\top b$ — the **normal equations** for least squares.

### Chain Rule

For $L(x) = f(g(x))$: $\nabla_x L = J_g^\top \nabla_{g} f$ where $J_g$ is the Jacobian of $g$.

In scalar notation: $\partial L/\partial x_i = \sum_j (\partial L/\partial g_j)(\partial g_j/\partial x_i)$.

---

## Intermediate

### Vector-by-Vector Derivatives (Jacobian)

For $f: \mathbb{R}^n \to \mathbb{R}^m$: $J \in \mathbb{R}^{m \times n}$, $J_{ij} = \partial f_i / \partial x_j$.

**Key Jacobians:**

| $f(x)$ | $J = \partial f/\partial x$ |
|---|---|
| $Ax$ (linear) | $A$ |
| $\sigma(x)$ (elementwise [[activation-sigmoid-tanh\|sigmoid]]) | $\text{diag}(\sigma(x)(1-\sigma(x)))$ |
| [[activation-relu-variants\|$\text{ReLU}(x)$]] | $\text{diag}(\mathbf{1}[x > 0])$ |
| [[activation-softmax\|$\text{softmax}(x)$]] | $\text{diag}(p) - pp^\top$ where $p = \text{softmax}(x)$ |

**Softmax Jacobian derivation:** $p_i = e^{x_i}/\sum_k e^{x_k}$.

$\partial p_i/\partial x_j$: for $i = j$: $p_i(1-p_i)$. For $i \neq j$: $-p_i p_j$.

So $J_{ij} = p_i(\delta_{ij} - p_j)$, i.e., $J = \text{diag}(p) - pp^\top$.

**Worked example.** $x = [2, 1, 0]$, $p \approx [0.665, 0.245, 0.090]$.

Diagonal: $J_{11} = 0.665(1-0.665) = 0.223$, $J_{22} = 0.245(0.755) = 0.185$, $J_{33} = 0.090(0.910) = 0.082$.

Off-diagonal: $J_{12} = -0.665 \cdot 0.245 = -0.163$. Increasing $x_1$ pushes probability mass from all other classes to class 1 — the off-diagonal terms represent this competition.

### Cross-Entropy + Softmax Gradient

$L = -\log p_y$ where $y$ is the true class label, $p = \text{softmax}(x)$.

$$\frac{\partial L}{\partial x_i} = -\frac{1}{p_y}\frac{\partial p_y}{\partial x_i} = -\frac{1}{p_y} \cdot p_y(\delta_{iy} - p_i) = p_i - \delta_{iy}$$

**Result:** $\nabla_x L = p - e_y$ (predicted probabilities minus one-hot target). This famous result comes directly from combining the softmax Jacobian with the chain rule.

### Linear Layer Backpropagation

**Forward pass:** $Y = XW + b$, $X \in \mathbb{R}^{n\times d}$, $W \in \mathbb{R}^{d\times h}$, $Y \in \mathbb{R}^{n\times h}$.

**Upstream gradient:** $\delta = \partial L/\partial Y \in \mathbb{R}^{n\times h}$.

**Gradients:**
$$\nabla_W L = X^\top \delta \in \mathbb{R}^{d \times h}$$
$$\nabla_X L = \delta W^\top \in \mathbb{R}^{n \times d}$$
$$\nabla_b L = \mathbf{1}^\top \delta \in \mathbb{R}^{1 \times h} \quad\text{(sum over batch)}$$

**Dimension verification:**
- $X^\top \in \mathbb{R}^{d\times n}$, $\delta \in \mathbb{R}^{n\times h}$ → $X^\top\delta \in \mathbb{R}^{d\times h}$ = shape of $W$ ✓
- $\delta \in \mathbb{R}^{n\times h}$, $W^\top \in \mathbb{R}^{h\times d}$ → $\delta W^\top \in \mathbb{R}^{n\times d}$ = shape of $X$ ✓

**Numerical verification.** $n=2$, $d=2$, $h=2$, $W = I$, $X = [[1,2],[3,4]]$.

$Y = X$. $L = \|Y\|_F^2/2$, so $\delta = Y = [[1,2],[3,4]]$.

$\nabla_W = X^\top\delta = [[1,3],[2,4]][[1,2],[3,4]] = [[10,14],[14,20]]$.

$\nabla_X = \delta W^\top = [[1,2],[3,4]] \cdot I = [[1,2],[3,4]]$.

---

## Advanced

### Scalar-by-Matrix Derivatives

For $f: \mathbb{R}^{m \times n} \to \mathbb{R}$: $(\nabla_A f)_{ij} = \partial f/\partial A_{ij}$.

**Key identities:**

| $f$ | $\nabla_A f$ |
|---|---|
| $\text{tr}(A)$ | $I$ |
| $\text{tr}(AB)$ | $B^\top$ |
| $\text{tr}(A^\top B)$ | $B$ |
| $\|A\|_F^2 = \text{tr}(A^\top A)$ | $2A$ |
| $\ln\det(A)$ | $A^{-\top} = (A^{-1})^\top$ |
| $a^\top Ab$ | $ab^\top$ |

**Derivation of $\nabla_A\text{tr}(AB)$:** $\text{tr}(AB) = \sum_{ij} A_{ij} B_{ji}$. So $\partial/\partial A_{ij} = B_{ji} = (B^\top)_{ij}$. Hence $\nabla_A\text{tr}(AB) = B^\top$.

**Derivation of $\nabla_A\ln\det(A)$:** using cofactor expansion and the matrix [[linear-algebra-fundamentals|determinant]] lemma:
$$\frac{\partial\ln\det(A)}{\partial A_{ij}} = (A^{-\top})_{ij}$$

For symmetric $A$: $\nabla_A\ln\det(A) = A^{-1}$.

**ML application:** Gaussian log-likelihood $\mathcal{L} = -\frac{n}{2}\ln\det\Sigma - \frac{1}{2}\text{tr}(\Sigma^{-1}S)$ where $S = \frac{1}{n}\sum_i (x_i-\mu)(x_i-\mu)^\top$. Setting $\nabla_\Sigma \mathcal{L} = 0$:
$$-\frac{n}{2}\Sigma^{-1} + \frac{1}{2}\Sigma^{-1}S\Sigma^{-1} = 0 \Rightarrow \hat\Sigma = S$$

The MLE for a Gaussian covariance is the sample covariance — derived in one line from matrix calculus identities.

### The Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$: $H \in \mathbb{R}^{n\times n}$, $H_{ij} = \partial^2 f/\partial x_i \partial x_j$.

$H$ is symmetric (Schwarz's theorem). **Taylor expansion:** $f(x+\delta) \approx f(x) + \nabla f^\top\delta + \frac{1}{2}\delta^\top H\delta$.

**Newton's method:** minimize by iterating $x \leftarrow x - H^{-1}\nabla f$. On a quadratic $f = \frac{1}{2}x^\top Ax - b^\top x$ with $H = A$: one Newton step reaches the exact optimum ($x^* = A^{-1}b$). Convergence is quadratic near the optimum for non-quadratics.

Cost: $O(n^2)$ to store $H$, $O(n^3)$ to invert — impractical for neural networks ($n \sim 10^8$ parameters). Approximations:
- **Diagonal Hessian:** assume $H \approx \text{diag}(H)$. Cost $O(n)$. Used in Adagrad, Adam (with gradient squared as proxy for diagonal Hessian).
- **K-FAC (Kronecker-Factored Approximate Curvature):** $H \approx A \otimes B$ (Kronecker product of two small matrices). Cost $O(d^3)$ per layer. Second-order optimizer used in some large-scale training.

### Gauss-Newton and Fisher Information

For least-squares $L = \frac{1}{2}\|r(\theta)\|^2$, $r: \mathbb{R}^d \to \mathbb{R}^m$, the Hessian is:
$$H = J^\top J + \sum_i r_i H_i$$

**Gauss-Newton approximation:** drop the second-order term (valid near the optimum where $r_i \approx 0$): $H_{GN} = J^\top J \succeq 0$ (always PSD). Update: $\Delta\theta = -(J^\top J)^{-1}J^\top r$.

**Connection to [[fisher-information|Fisher Information]]:** for a probabilistic model $p_\theta(x)$, the **Fisher information matrix**:
$$F = \mathbb{E}_x\!\left[\nabla_\theta \log p_\theta(x) \nabla_\theta \log p_\theta(x)^\top\right]$$

For a model with Gaussian output, $F = J^\top J$ (the Gauss-Newton matrix). **Natural gradient descent** (Amari, 1998) uses $F^{-1}$ instead of $H^{-1}$: it steps in the direction of steepest descent in the space of probability distributions (invariant to parameter reparameterization) rather than in Euclidean parameter space. This gives faster convergence when the parameter space has non-uniform curvature — connections to Adam (which approximates the diagonal of $F$).

---

*See also: [[backpropagation]] · [[backpropagation-advanced]] · [[linear-algebra-fundamentals]] · [[eigenvalues-pca]] · [[lagrangian-optimization]] · [[activation-softmax]] · [[loss-cross-entropy]] · [[fisher-information]]*
