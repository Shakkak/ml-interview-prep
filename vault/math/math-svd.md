---
title: Singular Value Decomposition (SVD)
tags: [linear-algebra, math, dimensionality-reduction, low-rank, svd]
aliases: [SVD, singular value decomposition, low-rank approximation, stable rank, Eckart-Young]
difficulty: 2
status: complete
related: [lora-quantization, math-convexity-jensen, backpropagation-advanced, eigenvalues-pca, linear-algebra-fundamentals]
---

# Singular Value Decomposition (SVD)

---

## Fundamental

Every matrix $A \in \mathbb{R}^{m \times n}$ has a **singular value decomposition**:

$$A = U \Sigma V^\top$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors, orthonormal columns ($U^\top U = I$)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal, entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$ (singular values)
- $V \in \mathbb{R}^{n \times n}$: right singular vectors, orthonormal columns ($V^\top V = I$)

$r = \text{rank}(A)$ = number of non-zero singular values.

**Geometric interpretation:** any linear map $A: \mathbb{R}^n \to \mathbb{R}^m$ transforms the unit sphere into an ellipsoid. SVD decomposes this into:
1. $V^\top$: rotate the input (align input axes with principal directions)
2. $\Sigma$: stretch/compress along $r$ coordinate axes by factors $\sigma_i$
3. $U$: rotate the output (align stretched axes with output directions)

The singular values $\sigma_i$ are the half-lengths of the ellipsoid axes.

**Worked numerical example.** $A = \begin{bmatrix}3&1\\1&3\end{bmatrix}$ (symmetric, so $U = V$).

$A^\top A = \begin{bmatrix}10&6\\6&10\end{bmatrix}$. Eigenvalues: $\lambda^2 - 20\lambda + 64 = 0 \Rightarrow \lambda_1 = 16$, $\lambda_2 = 4$.

Singular values: $\sigma_1 = \sqrt{16} = 4$, $\sigma_2 = \sqrt{4} = 2$.

Eigenvectors (columns of $V = U$): $v_1 = [1/\sqrt{2}, 1/\sqrt{2}]^\top$, $v_2 = [-1/\sqrt{2}, 1/\sqrt{2}]^\top$.

$$A = \begin{bmatrix}1/\sqrt{2} & -1/\sqrt{2}\\1/\sqrt{2} & 1/\sqrt{2}\end{bmatrix} \begin{bmatrix}4&0\\0&2\end{bmatrix} \begin{bmatrix}1/\sqrt{2} & 1/\sqrt{2}\\-1/\sqrt{2} & 1/\sqrt{2}\end{bmatrix}$$

**Thin SVD:** for $m > n$, keep only the first $n$ columns of $U$ (size $m \times n$) and make $\Sigma$ square ($n \times n$). Gives the same $A = U_n \Sigma_n V^\top$ with less storage.

---

## Intermediate

### Best Low-Rank Approximation (Eckart-Young Theorem)

The best rank-$k$ approximation of $A$ in Frobenius norm is:

$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top = U_k \Sigma_k V_k^\top$$

**Eckart-Young-Mirsky theorem (1936):** for any rank-$k$ matrix $B$:
$$\|A - A_k\|_F \leq \|A - B\|_F, \qquad \|A - A_k\|_2 \leq \|A - B\|_2$$

The approximation error in Frobenius norm:
$$\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

**Verification with example:** rank-1 approximation: $A_1 = 4 \cdot v_1 v_1^\top = \begin{bmatrix}2&2\\2&2\end{bmatrix}$.

$\|A - A_1\|_F^2 = (3-2)^2 + (1-2)^2 + (1-2)^2 + (3-2)^2 = 4 = \sigma_2^2$ ✓

### Applications in ML

**PCA connection:** the right singular vectors $V$ are the principal directions. Project data onto top-$k$: multiply by $V_k$ (first $k$ columns). Variance explained by component $i$ = $\sigma_i^2 / \sum_j \sigma_j^2$ (computed from the data matrix $X$, not the covariance matrix, saving an $O(d^3)$ eigendecomposition).

**LoRA fine-tuning:** pretrained weight matrix $W_0 \in \mathbb{R}^{d \times d}$. Instead of full fine-tuning (updating $d^2$ parameters), add a low-rank update: $W = W_0 + \Delta W$, $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$. Parameter count: $2dr \ll d^2$. The constraint that $\Delta W$ is rank-$r$ is justified by the empirical observation that weight updates during fine-tuning concentrate in a low-rank subspace (Aghajanyan et al., 2020 on intrinsic dimensionality).

**Matrix compression:** replace $W$ with $W_k = U_k \Sigma_k V_k^\top$. Storage: $k(m+n)$ vs $mn$ original. Compression ratio $= mn / k(m+n)$. For a square matrix, ratio $= n/(2k)$ — rank-50 approximation of a $1000 \times 1000$ matrix saves 90% of storage.

**Semantic search (LSA):** SVD of a term-document matrix $M \in \mathbb{R}^{\text{terms} \times \text{docs}}$. Top-$k$ components capture "latent semantic concepts" — document vectors in $k$-dimensional concept space. Documents with similar meanings cluster even if they share no words.

### Relationship to Eigendecomposition

For symmetric positive semi-definite $A = A^\top$ ($\lambda_i \geq 0$): $A = Q\Lambda Q^\top = U\Sigma V^\top$ with $U = V = Q$ and $\sigma_i = \lambda_i$.

For general $A$: $A^\top A = V\Sigma^2 V^\top$ (eigendecomposition of the Gram matrix, eigenvalues $= \sigma_i^2$). SVD generalizes eigendecomposition to rectangular matrices.

---

## Advanced

### Stable Rank and Effective Low Rank

The nominal rank of $W \in \mathbb{R}^{m \times n}$ is $\min(m,n)$, but most weight matrices in neural networks behave like much lower-rank matrices. The **stable rank**:

$$\text{sr}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2}$$

If $\text{sr}(A) \ll \min(m,n)$: most energy is concentrated in the top singular values. The matrix behaves like a low-rank matrix for practical purposes.

**Empirical finding (Aghajanyan et al., 2020):** fine-tuning directions for BERT on NLP tasks have intrinsic dimension $\sim 200$ even though the full parameter space has $\sim 110M$ dimensions. The fine-tuning gradient trajectories lie in a low-dimensional subspace of parameter space — justifying LoRA's rank-$r$ constraint.

**Singular value spectrum diagnostics:**
- Flat spectrum ($\sigma_i \approx$ const): near-random matrix (e.g., random initialization). No low-rank structure.
- Steep decay (few large $\sigma_i$, rest near 0): strong low-rank structure. Common in well-trained representations.
- Power-law decay ($\sigma_i \propto 1/i^\alpha$): intermediate, often seen in real-world datasets and pretrained embeddings.

### Randomized SVD Algorithm

For large matrices where exact SVD is too slow, **randomized SVD (Halko, Martinsson, Tropp, 2011):**

1. Draw random Gaussian matrix $\Omega \in \mathbb{R}^{n \times (k+p)}$ ($k$ = target rank, $p \approx 10$ = oversampling)
2. Compute $Y = A\Omega \in \mathbb{R}^{m \times (k+p)}$ — sample random projections
3. QR-decompose: $Y = QR$ to get an orthonormal basis $Q$ for the range of $A$
4. Form small matrix $B = Q^\top A \in \mathbb{R}^{(k+p) \times n}$
5. Compute exact SVD of $B = \tilde{U}\Sigma V^\top$; set $U = Q\tilde{U}$

Total cost: $O(mn\log k + k^2 n)$ — major speedup over $O(m^2 n)$ exact SVD. Error: $\mathbb{E}\|A - U_k\Sigma_k V_k^\top\|_F \leq (1+4\sqrt{(k+p)\cdot r}/p)\cdot\sigma_{k+1}$ — essentially optimal with $p \approx 10$.

Used in `sklearn.utils.extmath.randomized_svd`, which powers `TruncatedSVD`, `RandomizedPCA`, and `NMF`.

### Power Method and Convergence for Top Singular Vectors

The **power method for SVD** iterates:

$$Q_{t+1}, \_ = \text{QR}(A\Omega_t), \qquad \text{then } \Omega_{t+1} = A^\top Q_{t+1}$$

Convergence rate for the top singular vector: $O((\sigma_2/\sigma_1)^t)$ — exponential in the **singular gap** $\sigma_1 - \sigma_2$. When $\sigma_1 \approx \sigma_2$, convergence is slow.

**Spectral norm computation** (required for WGAN-GP Lipschitz constraint): the spectral norm $\|W\|_2 = \sigma_{\max}$ can be computed approximately by a single power iteration step per gradient step, making spectral normalization computationally cheap — $O(d)$ per layer per step rather than $O(d^2)$.

---

*See also: [[eigenvalues-pca]] · [[linear-algebra-fundamentals]] · [[lora-quantization]] · [[backpropagation-advanced]]*
