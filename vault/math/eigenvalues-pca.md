---
title: Eigenvalues, Eigenvectors, and PCA
tags: [linear-algebra, pca, dimensionality-reduction, math, spectral-theorem]
aliases: [eigenvalues, eigenvectors, PCA, principal component analysis, spectral theorem, power iteration]
difficulty: 2
status: complete
depends_on: [linear-algebra-fundamentals, matrix-calculus]
related: [linear-algebra-fundamentals, math-svd, matrix-calculus, lagrangian-optimization, distributions-gaussian, feature-preprocessing]
---

# Eigenvalues, Eigenvectors, and PCA

---

## Fundamental

**Eigenvectors** are the "natural axes" of a matrix — the directions a linear transformation stretches or compresses without rotating. **Eigenvalues** say how much stretching happens along each axis. They underlie PCA (finding directions of maximum variance), graph algorithms, spectral clustering, and stability analysis of optimization. If you've ever seen a covariance ellipse or a scree plot, you've seen eigenvalues in action.

A vector $v \neq 0$ is an **eigenvector** of a square matrix $A \in \mathbb{R}^{n\times n}$ with **eigenvalue** $\lambda$ if:

$$Av = \lambda v$$

where $A \in \mathbb{R}^{n \times n}$ = the square matrix (e.g., a covariance matrix), $v \in \mathbb{R}^n$ = the eigenvector (a non-zero direction that is not rotated by $A$, only stretched), and $\lambda \in \mathbb{R}$ = the corresponding eigenvalue (the stretch factor — positive means same direction, negative means flipped, zero means collapsed to zero).

Geometrically: $A$ maps $v$ to a scalar multiple of itself — $v$ is a fixed direction under the transformation, stretched or compressed by factor $\lambda$. All other vectors rotate when multiplied by $A$.

**Finding eigenvalues:** rearrange as $(A - \lambda I)v = 0$, which has a non-zero solution iff $\det(A - \lambda I) = 0$. The characteristic polynomial $p(\lambda) = \det(A - \lambda I)$ has degree $n$, yielding at most $n$ eigenvalues.

**Worked example.** $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$.

Characteristic polynomial: $(3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda-4)(\lambda-2) = 0 \Rightarrow \lambda_1 = 4,\; \lambda_2 = 2$.

Eigenvector for $\lambda_1 = 4$: $(A-4I)v = \begin{bmatrix}-1&1\\1&-1\end{bmatrix}v = 0 \Rightarrow v_1 = \tfrac{1}{\sqrt{2}}[1,1]^\top$.

Eigenvector for $\lambda_2 = 2$: $v_2 = \tfrac{1}{\sqrt{2}}[1,-1]^\top$.

Check: $Av_1 = [3/\sqrt{2}+1/\sqrt{2},\;1/\sqrt{2}+3/\sqrt{2}]^\top = 4\cdot v_1\; ✓$

Geometric picture: $A$ stretches space by 4 along $[1,1]$ and by 2 along $[1,-1]$. The ellipse $\{x : x^\top A x = 1\}$ has axes aligned with these eigenvectors and half-lengths $1/\sqrt{\lambda_i}$.

**Spectral Theorem.** Every real symmetric matrix $A = A^\top$ has:
1. All eigenvalues **real**
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
3. Orthonormal eigenbasis always exists: $A = Q\Lambda Q^\top$

**Key consequences:** $\det(A) = \prod_i \lambda_i$; $\text{tr}(A) = \sum_i \lambda_i$; $A^k = Q\Lambda^k Q^\top$; $A^{-1} = Q\Lambda^{-1}Q^\top$ (invert by inverting eigenvalues); $f(A) = Qf(\Lambda)Q^\top$ for any scalar function $f$.

---

## Intermediate

### PCA: Derivation from First Principles

Given centered data matrix $X \in \mathbb{R}^{N \times d}$ (zero mean), the sample covariance is $\Sigma = X^\top X / N$.

**Goal:** find unit vector $w$ that maximizes projected variance $w^\top \Sigma w$.

**[[lagrangian-optimization|Lagrangian]]:** $\mathcal{L}(w, \lambda) = w^\top \Sigma w - \lambda(w^\top w - 1)$.

$\partial \mathcal{L}/\partial w = 0$: $2\Sigma w = 2\lambda w$, so $\Sigma w = \lambda w$.

The variance-maximizing direction is the **eigenvector of $\Sigma$ with largest eigenvalue $\lambda_1$**. Maximum variance achieved $= \lambda_1$.

**Subsequent components:** the $k$-th PC is the eigenvector $q_k$ orthogonal to $q_1,\ldots,q_{k-1}$ with $k$-th largest eigenvalue. Variance explained by PC $k$:

$$\frac{\lambda_k}{\sum_{i=1}^d \lambda_i} = \frac{\lambda_k}{\text{tr}(\Sigma)}$$

**Reconstruction.** Project onto top-$k$ components: $Z = XQ_k \in \mathbb{R}^{N\times k}$. Reconstruct: $\hat{X} = ZQ_k^\top$. Frobenius reconstruction error $= \sum_{i=k+1}^d \lambda_i \cdot N$ — exactly the discarded eigenvalue energy.

### PCA via SVD (Numerically Preferred)

[[math-svd|SVD]] of the data matrix: $X = U\Sigma V^\top$. Then $\frac{1}{N}X^\top X = V\frac{\Sigma^2}{N}V^\top$, so:
- **Principal directions** = right singular vectors $V$
- **Eigenvalues** = $\sigma_i^2 / N$
- **Scores** = $U\Sigma = XV$

SVD is numerically superior to forming $X^\top X$ because squaring the matrix worsens the [[linear-algebra-fundamentals|condition number]]: $\kappa(X^\top X) = \kappa(X)^2$. A 4-digit accurate $X$ becomes an 8-digit problem after squaring, potentially losing all significant digits in small eigenvalues.

### Whitening

Transform data to have identity covariance: $\tilde{x} = \Lambda^{-1/2}Q^\top x$ (PCA whitening) or $\tilde{x} = Q\Lambda^{-1/2}Q^\top x$ (ZCA whitening).

ZCA whitening minimizes $\mathbb{E}[\|\tilde{x} - x\|^2]$ over all whitening transforms — the whitened image looks as similar to the original as possible while satisfying $\text{Cov}(\tilde{X}) = I$.

**Regularization:** replace $\lambda_i$ with $\max(\lambda_i, \epsilon)$ to prevent amplification of near-zero-variance directions.

---

## Advanced

### Kernel PCA

Standard PCA finds linear structure. **Kernel PCA** implicitly maps to a feature space $\phi: \mathbb{R}^d \to \mathcal{H}$ and performs PCA there via the [[kernel-methods|kernel trick]] $k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$.

**Centering in feature space:** $\tilde{K} = K - \mathbf{1}_N K - K\mathbf{1}_N + \mathbf{1}_N K\mathbf{1}_N$ where $(\mathbf{1}_N)_{ij} = 1/N$.

Principal components: eigenvectors of $\tilde{K}$ (scaled by $1/\sqrt{N\lambda}$). Projecting test point $x_*$: compute $k(x_*, x_i)$ for all training points, center, project.

With RBF kernel $k(x,y) = \exp(-\|x-y\|^2/2\sigma^2)$, kernel PCA finds **nonlinear manifolds** in input space — a Swiss roll unfolds under kernel PCA with appropriate $\sigma$. The kernel replaces an intractable infinite-dimensional eigendecomposition with an $N \times N$ problem.

### Power Iteration and Convergence

**Power iteration:** start random $v_0$, iterate $v_{k+1} = Av_k / \|Av_k\|$ until convergence.

**Convergence rate:** $\|v_t - q_1\| = O\!\left(\left|\frac{\lambda_2}{\lambda_1}\right|^t\right)$ — geometric convergence determined by the **spectral gap** $|\lambda_1 - \lambda_2|$. If $\lambda_1 = \lambda_2$, power iteration diverges into the degenerate eigenspace.

**Deflation** for top-$k$ eigenvectors: after finding $q_1$, set $A \leftarrow A - \lambda_1 q_1 q_1^\top$ and repeat.

**Lanczos algorithm** builds a Krylov subspace $\{b, Ab, A^2b, \ldots, A^{k-1}b\}$ and extracts the best rank-$k$ approximation. Cost: $O(k \cdot \text{nnz}(A))$ instead of $O(n^3)$. Standard for PCA on large sparse matrices.

**Randomized SVD (Halko et al., 2011):** generate random matrix $\Omega \in \mathbb{R}^{n \times (k+p)}$; compute $Y = A\Omega$; QR-decompose $Y = QR$; compute small SVD of $Q^\top A \in \mathbb{R}^{(k+p)\times n}$. Total cost $O(mn\log k + k^2 n)$ for rank-$k$ approximation vs $O(mn\min(m,n))$ for full SVD. Oversampling $p \approx 10$ makes failure probability exponentially small. This is `sklearn.utils.extmath.randomized_svd` used by `TruncatedSVD` and `RandomizedPCA`.

### Probabilistic PCA

Model: $x = Wz + \mu + \epsilon$, $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$.

MLE: the columns of $W$ span the same subspace as the top-$k$ PCA eigenvectors; $\hat{\sigma}^2 = \frac{1}{d-k}\sum_{i=k+1}^d \lambda_i$ (average of discarded eigenvalues, interpretable as reconstruction noise).

As $\sigma^2 \to 0$: probabilistic PCA recovers standard PCA exactly. As $\sigma^2 \to \infty$: the posterior over $z$ becomes isotropic (no information from $x$). Probabilistic PCA enables **missing data imputation** via EM and principled comparison of models with different $k$ via marginal likelihood.

---

## Links

- [[linear-algebra-fundamentals]] — eigenvalues and eigenvectors are properties of square matrices; they rely on the foundation of linear maps and basis transformations
- [[matrix-calculus]] — PCA's principal components are found by computing the gradient of variance and setting it to zero; eigendecomposition is the closed-form solution
- [[math-svd]] — SVD generalizes eigendecomposition to non-square matrices; $A = U\Sigma V^\top$ with singular values = square roots of eigenvalues of $A^\top A$
- [[lagrangian-optimization]] — PCA's constrained variance-maximization problem is solved via Lagrange multipliers; the Lagrangian gives the eigenvalue equation directly
- [[distributions-gaussian]] — PCA finds the axes of maximum variance in a Gaussian; the covariance matrix's eigenvectors are the principal components
- [[kernel-methods]] — kernel PCA replaces the $d \times d$ covariance matrix with the $n \times n$ kernel matrix, enabling nonlinear dimensionality reduction
- [[feature-preprocessing]] — PCA is a standard preprocessing step to reduce collinearity; features should be standardized before applying PCA
