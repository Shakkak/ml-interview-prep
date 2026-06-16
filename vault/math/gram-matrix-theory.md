---
title: Gram Matrix
tags: [gram-matrix, gram-schmidt, inner-product, kernel, style-transfer]
aliases: [Gram matrix, Gramian, Gram-Schmidt orthogonalization, feature correlation matrix]
difficulty: 1
status: complete
depends_on: [linear-algebra-fundamentals, eigenvalues-pca]
related: [kernel-methods, linear-algebra-fundamentals, math-svd, eigenvalues-pca, neural-tangent-kernel]
---

# Gram Matrix

---

## Fundamental

The **Gram matrix** encodes all pairwise similarities between a set of vectors — each entry is the dot product between two vectors. It appears in kernel methods (the kernel matrix is a Gram matrix computed via the kernel trick), neural style transfer (matching texture by matching feature correlations), and the theory of neural network training dynamics (the NTK is a Gram matrix of gradients).

### Definition

Given a matrix $X \in \mathbb{R}^{n \times d}$ (rows are vectors), the **Gram matrix** is:

$$G = X X^\top \in \mathbb{R}^{n \times n}, \quad G_{ij} = \mathbf{x}_i^\top \mathbf{x}_j$$

Each entry $G_{ij}$ is the inner product (dot product) between vectors $i$ and $j$.

**Properties:**
- Symmetric: $G = G^\top$
- Positive semidefinite: $\mathbf{v}^\top G \mathbf{v} = \|X^\top \mathbf{v}\|^2 \geq 0$
- Rank = rank of $X$ (at most $\min(n, d)$)
- Eigenvalues of $G$ = squared singular values of $X$ = eigenvalues of $X^\top X$

### Gram Matrix vs Covariance Matrix

$X^\top X \in \mathbb{R}^{d \times d}$ is the **feature covariance** (up to scaling) — $d \times d$, captures feature-feature relationships.

$X X^\top \in \mathbb{R}^{n \times n}$ is the **Gram matrix** — $n \times n$, captures sample-sample similarities.

They share the same nonzero eigenvalues (by the SVD relationship $X = U\Sigma V^\top$).

---

## Intermediate

### Gram-Schmidt Orthogonalization

Given $n$ linearly independent vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$, Gram-Schmidt produces orthonormal vectors $\{\mathbf{u}_1, \ldots, \mathbf{u}_n\}$ spanning the same space:

$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \frac{\mathbf{v}_k^\top \mathbf{u}_j}{\mathbf{u}_j^\top \mathbf{u}_j} \mathbf{u}_j, \quad \text{then normalize}$$

This is the foundation of QR decomposition: $X = QR$ where $Q$ has orthonormal columns (from Gram-Schmidt) and $R$ is upper triangular.

**Numerical stability:** classical Gram-Schmidt accumulates floating-point errors; **modified Gram-Schmidt** or Householder reflections are used in practice.

### Gram Matrix in Neural Style Transfer

Gatys et al. (2015) represented the **style** of an image as the Gram matrix of feature activations at each CNN layer:

$$G^{(l)} = \frac{1}{h_l w_l} F^{(l)} (F^{(l)})^\top$$

where $F^{(l)} \in \mathbb{R}^{C_l \times (h_l w_l)}$ stacks spatially flattened feature maps. The Gram matrix captures **feature co-occurrence statistics** — which features tend to activate together — encoding texture and style without spatial structure.

Style transfer minimizes the Frobenius distance between Gram matrices of the generated image and the style image.

---

## Advanced

### Gram Matrix as a Kernel Matrix

If $\mathbf{x}_i, \mathbf{x}_j$ are feature vectors and $G_{ij} = \mathbf{x}_i^\top \mathbf{x}_j$, the Gram matrix is the **kernel matrix** for the linear kernel. More generally, for any kernel $k(\cdot, \cdot)$:

$$K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$$

is positive semidefinite by Mercer's theorem (see [[kernel-methods]]). The linear Gram matrix is the simplest kernel matrix.

**Kernel trick connection:** computing $K = XX^\top$ in $O(n^2 d)$ is equivalent to computing all pairwise dot products. For the RBF kernel, $K_{ij} = \exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma^2)$, which can be computed from the Gram matrix via $\|x_i - x_j\|^2 = G_{ii} + G_{jj} - 2G_{ij}$.

### Neural Tangent Kernel and the Gram Matrix

For wide neural networks at initialization, the kernel matrix of the network (the **Neural Tangent Kernel** matrix, see [[neural-tangent-kernel]]) behaves like a Gram matrix in the feature space induced by the gradients. Training dynamics of wide networks are governed by the eigenvalues of this kernel Gram matrix.

## Links

- [[linear-algebra-fundamentals]] — the Gram matrix $G_{ij} = \langle x_i, x_j \rangle$ is the $n\times n$ matrix of pairwise inner products; it is positive semi-definite by construction
- [[eigenvalues-pca]] — the Gram matrix's eigenvalues equal those of $XX^\top$; kernel PCA eigendecomposes the Gram matrix
- [[kernel-methods]] — the kernel matrix is a Gram matrix in the feature space; every valid kernel matrix is a Gram matrix for some feature map (Mercer's theorem)
- [[math-svd]] — if $X = U\Sigma V^\top$, then the Gram matrix $XX^\top = U\Sigma^2 U^\top$; the eigenvalues of the Gram matrix are the squared singular values of $X$
- [[neural-tangent-kernel]] — the NTK is the Gram matrix of gradients: $K_{ij} = \nabla_\theta f(x_i) \cdot \nabla_\theta f(x_j)$; it governs the training dynamics of wide networks
