---
title: Kronecker Products
tags: [kronecker-product, matrix-operations, vec-operator, structured-matrices, kfac]
aliases: [Kronecker product, Kronecker factorization, vec operator, mixed-product property]
difficulty: 2
status: complete
related: [linear-algebra-fundamentals, matrix-calculus, math-svd, information-geometry, second-order-optimization]
---

# Kronecker Products

---

## Fundamental

The **Kronecker product** builds a large block matrix from two smaller ones — each entry of the first matrix multiplies the entire second matrix. In ML, it appears in K-FAC (an efficient approximation of the Fisher information matrix for second-order optimization), tensorized neural network layers, and the vec trick that converts matrix equations into vector equations. It is mainly a tool for structured matrix algebra that makes otherwise expensive operations tractable.

### Definition

For matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, the **Kronecker product** $A \otimes B \in \mathbb{R}^{mp \times nq}$ is the block matrix:

$$A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{pmatrix}$$

Each element of $A$ multiplies the entire matrix $B$.

**Key properties:**
- $(A \otimes B)^\top = A^\top \otimes B^\top$
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ (if both inverses exist)
- Eigenvalues of $A \otimes B$: all products $\lambda_i(A) \cdot \mu_j(B)$
- $\text{rank}(A \otimes B) = \text{rank}(A) \cdot \text{rank}(B)$

### Mixed Product Property

The most useful identity:

$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

whenever the dimensions are compatible. This means matrix multiplications can be factored across Kronecker structure — crucial for efficient computation.

---

## Intermediate

### Vec Operator

The **vec operator** $\text{vec}(A)$ stacks the columns of $A$ into a single vector:

$$A = \begin{pmatrix}a & c \\ b & d\end{pmatrix} \Rightarrow \text{vec}(A) = (a, b, c, d)^\top$$

**Key identity:** for matrices $A, X, B$ of compatible dimensions:

$$\text{vec}(AXB) = (B^\top \otimes A) \text{vec}(X)$$

This converts matrix equations into vector equations: if $AXB = C$, then $(B^\top \otimes A)\text{vec}(X) = \text{vec}(C)$.

**Application:** solving linear matrix equations $AXB = C$ reduces to solving $(B^\top \otimes A)\text{vec}(X) = \text{vec}(C)$ — a standard linear system.

### K-FAC and Kronecker Approximations

The Fisher information matrix $F$ for a neural network layer with weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is $F \in \mathbb{R}^{d_\text{in}d_\text{out} \times d_\text{in}d_\text{out}}$ — enormous.

**K-FAC (Kronecker-Factored Approximate Curvature, Martens & Grosse 2015)** approximates:

$$F \approx \hat{A} \otimes \hat{G}$$

where $\hat{A} = \mathbb{E}[a a^\top] \in \mathbb{R}^{d_\text{in} \times d_\text{in}}$ (input activation covariance) and $\hat{G} = \mathbb{E}[g g^\top] \in \mathbb{R}^{d_\text{out} \times d_\text{out}}$ (output gradient covariance).

**Efficient inverse:** $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$ — two small matrix inversions instead of one huge one. For $d_\text{in} = d_\text{out} = 1024$: invert two $1024 \times 1024$ matrices instead of one $10^6 \times 10^6$ matrix.

---

## Advanced

### Structured Weight Matrices

Kronecker products define structured weight matrices that are compact yet expressive:

**Monarch matrices (Dao et al., 2022):** products of permuted block-diagonal matrices, expressible as Kronecker products. Enable $O(n\sqrt{n})$ matrix-vector products for $n \times n$ weight matrices, reducing parameter count from $n^2$ to $n\sqrt{n}$.

**Butterfly matrices:** recursive Kronecker structure from the Fast Fourier Transform: $F_n = F_2 \otimes I_{n/2} \cdot D_n$ — the entire FFT is a product of sparse Kronecker-structured matrices. Butterfly networks parameterize such structures for learned sparse-matrix transformations.

**LoRA connection:** LoRA decomposes $\Delta W = AB$ (rank-$r$ factorization). For multiple layers, stacking these factorizations approximates a Kronecker structure in the parameter update space.

*See also: [[linear-algebra-fundamentals]] · [[matrix-calculus]] · [[second-order-optimization]] · [[information-geometry]]*
