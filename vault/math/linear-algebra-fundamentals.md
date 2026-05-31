---
title: Linear Algebra Fundamentals
tags: [linear-algebra, math, foundations, rank, norms, condition-number]
aliases: [vectors, matrices, rank, null space, norms, determinant, trace, condition number, projection matrix]
difficulty: 1
status: complete
related: [eigenvalues-pca, matrix-calculus, math-svd, distributions-gaussian, regularization-weight-decay]
---

# Linear Algebra Fundamentals

---

## Fundamental

### Vectors and Subspaces

A **vector space** over $\mathbb{R}$ is a set closed under addition and scalar multiplication. Vectors in $\mathbb{R}^n$ are elements.

**Span:** $\text{span}(v_1,\ldots,v_k) = \{c_1 v_1 + \cdots + c_k v_k : c_i \in \mathbb{R}\}$ — all linear combinations.

**Linear independence:** $v_1,\ldots,v_k$ are linearly independent iff $\sum c_i v_i = 0 \Rightarrow c_i = 0$ for all $i$.

**Basis:** a linearly independent set that spans the space. Any basis for $\mathbb{R}^n$ has exactly $n$ vectors.

### The Four Subspaces

For $A \in \mathbb{R}^{m \times n}$ with rank $r$:

| Subspace | Definition | Dimension | Lives in |
|---|---|---|---|
| Column space $\mathcal{C}(A)$ | $\{Ax : x \in \mathbb{R}^n\}$ | $r$ | $\mathbb{R}^m$ |
| Row space $\mathcal{C}(A^\top)$ | $\{A^\top y : y \in \mathbb{R}^m\}$ | $r$ | $\mathbb{R}^n$ |
| Null space $\mathcal{N}(A)$ | $\{x : Ax = 0\}$ | $n - r$ | $\mathbb{R}^n$ |
| Left null space $\mathcal{N}(A^\top)$ | $\{y : A^\top y = 0\}$ | $m - r$ | $\mathbb{R}^m$ |

**Rank-Nullity Theorem:** $\text{rank}(A) + \text{nullity}(A) = n$.

**Orthogonal complements:** $\mathcal{C}(A^\top) \perp \mathcal{N}(A)$ and $\mathcal{C}(A) \perp \mathcal{N}(A^\top)$.

**Worked example.** $A = \begin{bmatrix}1&2&3\\2&4&6\end{bmatrix}$ (row 2 = 2 × row 1, so rank = 1).

$\mathcal{N}(A)$: solve $x_1 + 2x_2 + 3x_3 = 0$. Two free variables: $\mathcal{N}(A) = \text{span}([-2,1,0]^\top, [-3,0,1]^\top)$. Dimension $= 2 = 3-1$ ✓.

$\mathcal{C}(A) = \text{span}([1,2]^\top)$. Dimension $= 1$ ✓.

### Vector Norms

$$\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$$

| Norm | Formula | Geometry | ML Use |
|---|---|---|---|
| $L_1$ | $\sum|x_i|$ | Manhattan distance | Lasso, sparsity |
| $L_2$ | $\sqrt{\sum x_i^2}$ | Euclidean distance | Ridge, gradient clipping |
| $L_\infty$ | $\max|x_i|$ | Chebyshev distance | Adversarial robustness |

Ordering: $\|x\|_\infty \leq \|x\|_2 \leq \|x\|_1 \leq \sqrt{n}\|x\|_2$.

Example: $v = [3, 4]$: $\|v\|_1 = 7$, $\|v\|_2 = 5$, $\|v\|_\infty = 4$.

---

## Intermediate

### Matrix Norms

| Norm | Formula | ML Use |
|---|---|---|
| Frobenius | $\|A\|_F = \sqrt{\text{tr}(A^\top A)} = \sqrt{\sum_{ij} A_{ij}^2}$ | Weight decay |
| Spectral ($L_2$) | $\|A\|_2 = \sigma_{\max}(A)$ | Discriminator Lipschitz constraint |
| Nuclear | $\|A\|_* = \sum_i \sigma_i$ | Low-rank regularization |

Inequalities: $\|A\|_2 \leq \|A\|_F \leq \sqrt{\min(m,n)}\|A\|_2$.

**Spectral norm as the Lipschitz constant:** $\|Ax\|_2 \leq \|A\|_2 \|x\|_2$ — the spectral norm is the maximum amplification factor. Spectral normalization in GAN discriminators divides each weight matrix by its spectral norm, enforcing a 1-Lipschitz constraint on the discriminator (used in WGAN-GP).

### Trace

$\text{tr}(A) = \sum_i A_{ii} = \sum_i \lambda_i$ (sum of eigenvalues).

**Key properties:**
- **Cyclic:** $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$
- $\|A\|_F^2 = \text{tr}(A^\top A)$
- **Trace trick:** $x^\top Ax = \text{tr}(x^\top Ax) = \text{tr}(Axx^\top)$ — converts a scalar to a trace for matrix differentiation
- $\nabla_A \text{tr}(AB) = B^\top$ (used constantly in backpropagation)

ML appearances: Gaussian entropy $\propto \text{tr}(\ln\Sigma)$; KL divergence between Gaussians contains $\text{tr}(\Sigma_1^{-1}\Sigma_0)$; normalizing flows use $\text{tr}(J)$ for the instantaneous log-density change.

### Determinant

$\det(A) = \prod_i \lambda_i$ (product of eigenvalues).

**Geometric meaning:** $|\det(A)|$ is the volume scaling factor of the linear map $x \mapsto Ax$. $\det(A) = 0 \Leftrightarrow A$ is singular $\Leftrightarrow$ rank-deficient.

Properties: $\det(AB) = \det(A)\det(B)$; $\det(A^{-1}) = 1/\det(A)$; $\det(A^\top) = \det(A)$; $\det(cA) = c^n\det(A)$.

ML appearances: Gaussian normalizing constant $\propto |\Sigma|^{-1/2}$; normalizing flows use $|\det J|$ for the change-of-variables formula; $\ln\det\Sigma = \text{tr}(\ln\Sigma) = \sum_i \ln\lambda_i$ (computed via Cholesky for $O(n^3)$ stability).

### Condition Number and Convergence

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\lambda_{\max}}{\lambda_{\min}} \quad\text{(for positive definite matrices)}$$

**Gradient descent convergence on quadratics:** for loss $f(x) = \frac{1}{2}x^\top Hx - b^\top x$ with optimal step size, error at step $t$:
$$\|x_t - x^*\|_H \leq \left(\frac{\kappa-1}{\kappa+1}\right)^t \|x_0 - x^*\|_H$$

| $\kappa$ | Steps to reduce error by 99% |
|---|---|
| 1 | 1 |
| 10 | ~28 |
| 100 | ~275 |
| 1000 | ~2750 |

**Squaring in normal equations:** $\kappa(X^\top X) = \kappa(X)^2$. Computing normal equations $(X^\top X)^{-1}X^\top y$ directly squares the condition number — use QR decomposition instead. This is why `sklearn.LinearRegression` uses `scipy.linalg.lstsq` (QR-based), not explicit matrix inversion.

### Projection Matrix

Orthogonal projection of $b$ onto $\mathcal{C}(A)$:

$$\hat{b} = A(A^\top A)^{-1}A^\top b = Pb$$

$P = A(A^\top A)^{-1}A^\top$ is the projection matrix. Properties: $P^2 = P$ (idempotent), $P^\top = P$ (symmetric), eigenvalues are 0 or 1.

**Normal equation derivation:** minimize $\|Ax - b\|^2$. Set gradient $2A^\top(Ax - b) = 0$ to get $A^\top Ax = A^\top b$. The solution $\hat{b} = A\hat{x}$ is the projection of $b$ onto $\mathcal{C}(A)$.

---

## Advanced

### Pseudoinverse and the Minimum-Norm Solution

When $A$ is not square or is rank-deficient, the system $Ax = b$ either has no solution or infinitely many. The **Moore-Penrose pseudoinverse** $A^+$ gives the minimum-norm least-squares solution $x^+ = A^+ b$.

Via SVD $A = U\Sigma V^\top$: $A^+ = V\Sigma^+ U^\top$ where $(\Sigma^+)_{ii} = 1/\sigma_i$ for $\sigma_i > 0$, $0$ otherwise.

**Properties:**
- If $A$ has full column rank: $A^+ = (A^\top A)^{-1}A^\top$ (left pseudoinverse)
- If $A$ has full row rank: $A^+ = A^\top(AA^\top)^{-1}$ (right pseudoinverse)
- $A^+ A$ is the projection onto $\mathcal{C}(A^\top)$ (row space)
- $AA^+$ is the projection onto $\mathcal{C}(A)$ (column space)

In neural networks, the pseudoinverse appears in the theoretical analysis of linear networks: the minimum-norm interpolating solution found by gradient descent from zero initialization equals $A^+ b$.

### LU, QR, and Cholesky Decompositions

**LU decomposition** ($A = PLU$, $P$ = permutation, $L$ = lower triangular, $U$ = upper triangular): solves $Ax = b$ in $O(n^3)$ via forward/back substitution. Used for general square matrices.

**QR decomposition** ($A = QR$, $Q$ orthogonal, $R$ upper triangular): solves least-squares without squaring the condition number. Forward pass: $O(mn^2)$ for $A \in \mathbb{R}^{m\times n}$, $m > n$. **Gram-Schmidt process** computes QR but is numerically unstable; **Householder reflections** give a stable $O(mn^2)$ QR.

**Cholesky decomposition** ($A = LL^\top$ for positive definite $A$): twice as fast as LU for symmetric positive definite matrices. Required for: sampling multivariate Gaussians ($x = \mu + Lz$, $z \sim \mathcal{N}(0,I)$), computing $\ln\det A = 2\sum_i \ln L_{ii}$, and solving linear systems involving covariance matrices in Gaussian processes.

**Numerical stability note:** for ill-conditioned $A$ (large $\kappa$), LU gives poor solutions; QR is better; Cholesky + regularization (adding $\epsilon I$) is best for covariance matrices. This is why Gaussian process libraries (GPyTorch) use jitter ($\epsilon I$) before Cholesky.

### Rank Facts and the SVD Connection

$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$: multiplying by a low-rank matrix cannot increase rank.

$\text{rank}(A) = \text{rank}(A^\top) = \text{rank}(A^\top A) = \text{rank}(AA^\top)$: rank is preserved under these operations.

$\text{rank}(A+B) \leq \text{rank}(A) + \text{rank}(B)$: adding low-rank matrices gives a low-rank result (if both are rank $r$, sum is at most rank $2r$ — the basis for LoRA's initialization strategy).

**Effective rank in pretrained models:** weight matrices $W$ in transformers have nominal rank $\min(m,n)$ but stable rank $\|W\|_F^2/\|W\|_2^2 = \sum\sigma_i^2/\sigma_1^2 \ll \min(m,n)$. Fine-tuning moves mostly within this low stable-rank subspace — the justification for LoRA (see [[math-svd]]).

---

*See also: [[eigenvalues-pca]] · [[math-svd]] · [[matrix-calculus]] · [[regularization-weight-decay]]*
