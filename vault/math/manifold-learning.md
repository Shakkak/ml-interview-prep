---
title: Manifold Learning and Dimensionality Reduction
tags: [manifold-learning, dimensionality-reduction, tsne, umap, intrinsic-dimension]
aliases: [manifold hypothesis, t-SNE, UMAP, Isomap, intrinsic dimensionality]
difficulty: 2
status: complete
related: [eigenvalues-pca, curse-of-dimensionality, kernel-methods, spectral-bias, word-embeddings]
depends_on: [eigenvalues-pca, kernel-methods, linear-algebra-fundamentals]
---

# Manifold Learning and Dimensionality Reduction

---

## Fundamental

### The Manifold Hypothesis

High-dimensional data (images, text, audio) does not fill $\mathbb{R}^d$ uniformly. Instead, it lies on or near a **low-dimensional manifold** embedded in $\mathbb{R}^d$:

- A $28\times28$ image lives in $\mathbb{R}^{784}$, but natural images form a manifold of much lower **intrinsic dimension** $k \ll 784$
- Interpolating between two face images in embedding space produces valid faces — the manifold is smooth
- "Most" of $\mathbb{R}^{784}$ contains noise images, not natural ones

**Implication for ML:** learning is tractable because the effective search space is $k$-dimensional, not $d$-dimensional. Models implicitly learn to navigate the data manifold.

### PCA as Linear Dimensionality Reduction

PCA (see [[eigenvalues-pca]]) finds the linear subspace of dimension $k$ that maximizes variance. The $k$ principal components are the top-$k$ eigenvectors of the covariance matrix.

**Limitation:** PCA can only capture linear structure. If the manifold is curved (e.g., the Swiss Roll: a 2D surface spiraling in 3D), PCA will not "unroll" it.

---

## Intermediate

### Nonlinear Methods

**Isomap:** approximate geodesic distances on the manifold by shortest paths through a $k$-nearest-neighbor graph. Then apply classical MDS (multidimensional scaling) using these graph distances. Unrolls smooth manifolds but fails with holes or non-convex geometry.

**Locally Linear Embedding (LLE):** each point is reconstructed as a weighted sum of its neighbors (local linearity). The low-dimensional embedding preserves these reconstruction weights. Good for unrolling; sensitive to parameter choice.

**t-SNE (van der Maaten & Hinton, 2008):** models pairwise similarities as probabilities (Gaussian kernel in high-D, Student-$t$ in low-D) and minimizes KL divergence between the two distributions:

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

where $P$ = joint probability distribution over pairs in high-dimensional space (Gaussian kernel similarities), $Q$ = joint distribution in low-dimensional space (Student-$t$ kernel similarities), $p_{ij}$ = probability that point $i$ would pick point $j$ as its neighbor in high-D, and $q_{ij}$ = the analogous probability in low-D. Minimizing this KL pushes the low-D layout to reproduce the high-D neighborhood structure.

Using a heavy-tailed $t$-distribution in low-D prevents crowding — dissimilar points get pushed far apart. t-SNE excels at revealing cluster structure in visualizations (2D/3D). **Limitations:**
- Non-parametric: can't embed new points without re-running
- Not globally meaningful: distances between clusters are not interpretable
- Sensitive to perplexity hyperparameter (typically 5–50)

### UMAP

UMAP (McInnes et al., 2018) builds a fuzzy topological representation of the data and minimizes cross-entropy between high-D and low-D fuzzy sets. Compared to t-SNE:

| Property | t-SNE | UMAP |
|----------|-------|------|
| Speed | $O(N \log N)$ with Barnes-Hut | Faster, better scales |
| Global structure | Poor | Better preserved |
| Cluster separation | Strong | Good |
| New point embedding | No (need parametric extension) | Yes (approximate) |
| Theoretical grounding | Probabilistic | Topological (Riemannian geometry) |

UMAP is now the default for visualization of learned representations (embeddings from BERT, ViT, etc.).

---

## Advanced

### Intrinsic Dimension Estimation

The **intrinsic dimension** $k$ is the smallest $k$ such that the data approximately lies on a $k$-manifold. Estimation methods:

**Correlation dimension:** count how many points fall within radius $r$; the count scales as $r^k$. Fit $k$ from the slope of $\log(\text{count})$ vs $\log(r)$.

**Two-NN estimator (Facco et al., 2017):** uses only the ratio of 1st and 2nd nearest-neighbor distances: $\mu_i = d_{i,2}/d_{i,1}$. Under the manifold assumption, $\mu$ follows a Pareto distribution with parameter $k$.

**Practical finding:** ImageNet images have intrinsic dimension $\approx 40$–$60$ despite $d = 150{,}528$ pixels; MNIST $\approx 13$.

### Manifold Learning in Deep Networks

Deep networks implicitly learn manifold structure:
- **Progressive disentanglement:** successive layers "unfold" the manifold, making it more linearly separable
- **Neural collapse:** at the final layer of a well-trained classifier, class means form a **simplex ETF (equiangular tight frame)** — a configuration where all $C$ class prototype vectors have equal pairwise angles ($\cos\theta = -1/(C-1)$), maximally spread on the unit sphere. This is the highest-symmetry arrangement possible for $C$ points, making the last-layer geometry independent of the training data distribution.
- **Latent traversals:** in VAEs and GANs, interpolating in latent space produces smooth transitions because the generator maps a simple latent manifold to the data manifold

**Spectral bias** (see [[spectral-bias]]): networks learn low-frequency (smooth, manifold-like) features first — consistent with the manifold hypothesis that low-frequency structure captures most variance.

## Links

- [[eigenvalues-pca]] — PCA finds a linear manifold (affine subspace); manifold learning (t-SNE, UMAP, Isomap) finds non-linear low-dimensional structure that PCA cannot capture
- [[kernel-methods]] — kernel PCA applies the kernel trick to find non-linear projections; Isomap uses geodesic distances (shortest manifold paths) instead of Euclidean distances
- [[linear-algebra-fundamentals]] — spectral embedding methods (Laplacian Eigenmaps, diffusion maps) build graph Laplacians and eigendecompose them to find manifold coordinates
- [[curse-of-dimensionality]] — manifold learning is motivated by the manifold hypothesis: high-dimensional data lies near a low-dimensional manifold, escaping the curse
- [[variational-autoencoders]] — VAE latent spaces are designed to approximate a smooth low-dimensional manifold; manifold learning provides the theoretical grounding
- [[word-embeddings]] — word embeddings are learned manifold coordinates; the geometry of the embedding space reflects semantic relationships
