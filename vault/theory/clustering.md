---
title: Clustering
tags: [clustering, k-means, dbscan, hierarchical-clustering, unsupervised-learning, gmm, silhouette]
aliases: [clustering, k-means, DBSCAN, hierarchical clustering, Gaussian mixture model, soft clustering, unsupervised clustering]
difficulty: 2
status: complete
related: [distributions-gaussian, eigenvalues-pca, curse-of-dimensionality, evaluation-metrics-guide, bayesian-inference, distributions-overview, ensemble-methods]
depends_on: [distributions-gaussian, linear-algebra-fundamentals, statistical-inference-mle]
---

# Clustering

---

## Fundamental

Clustering partitions a dataset into groups (**clusters**) such that points in the same group are more similar to each other than to points in other groups. Unlike classification, there are no labels — the algorithm discovers structure purely from the data geometry.

### K-Means

Partition $n$ points into $k$ clusters by minimizing **within-cluster sum of squares (WCSS)**, also called inertia:

$$\text{WCSS} = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

where $\mu_j$ is the centroid of cluster $C_j$.

**Algorithm (Lloyd's algorithm):**
1. Initialize $k$ centroids (e.g., randomly or via k-means++).
2. **Assignment step:** assign each point to its nearest centroid.
3. **Update step:** recompute each centroid as the mean of its assigned points.
4. Repeat steps 2–3 until assignments stop changing.

**Worked example** (1D, $k=2$, points $\{1, 2, 8, 9\}$, initial centroids $\mu_1=2$, $\mu_2=8$):

| Iteration | Assignments | New centroids |
|-----------|-------------|---------------|
| 1 | $C_1=\{1,2\}$, $C_2=\{8,9\}$ | $\mu_1=1.5$, $\mu_2=8.5$ |
| 2 | $C_1=\{1,2\}$, $C_2=\{8,9\}$ | unchanged — converged |

**Convergence:** WCSS is non-increasing at every step (both assignment and update reduce or preserve it). The algorithm always converges, but may reach a local minimum. Run multiple restarts and take the best result.

> [!warning] k-means++ initialization matters
> Random centroid initialization often converges to poor local minima. **k-means++** selects each new centroid with probability proportional to $\|x - \text{nearest existing centroid}\|^2$, spreading initial centroids across the data. In practice this reduces the number of restarts needed by ~10× and consistently produces lower WCSS.

---

## Intermediate

### Choosing K

**Elbow method:** plot WCSS vs. $k$. WCSS always decreases as $k$ increases (more clusters = smaller variance within each). The "elbow" — the point where the rate of decrease sharply flattens — suggests the natural number of clusters. Subjective; the elbow is often ambiguous in real data.

**Silhouette score:** for each point $i$, let $a(i)$ = mean distance to points in the same cluster, $b(i)$ = mean distance to points in the nearest other cluster:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))} \in [-1, 1]$$

where $a(i)$ = mean distance from point $i$ to all points in its own cluster, $b(i)$ = mean distance to points in the nearest other cluster, $s(i) \in [-1,1]$ = silhouette coefficient (higher = better).

$s(i) \approx 1$: well-clustered; $s(i) \approx 0$: on a boundary; $s(i) < 0$: likely mis-assigned. Average silhouette over all points peaks at the best $k$.

> [!tip] Why silhouette is more reliable than the elbow ([[evaluation-metrics-guide]])
> WCSS is an internal measure that always improves with more clusters — it can never detect "too many clusters." Silhouette compares within-cluster cohesion to between-cluster separation: it will drop if $k$ is too large (clusters split cohesive groups) or too small (clusters merge distinct groups). This gives a genuine optimum, not just a change in slope.

### DBSCAN

**Density-Based Spatial Clustering of Applications with Noise** identifies clusters as dense regions separated by low-density space. Unlike k-means, it does not require specifying $k$ and naturally handles noise.

Parameters: $\varepsilon$ (neighborhood radius), $\text{minPts}$ (minimum neighbors to form a core point).

**Point types:**
- **Core point:** has $\geq \text{minPts}$ neighbors within $\varepsilon$.
- **Border point:** within $\varepsilon$ of a core point but not itself a core point.
- **Noise point (outlier):** not within $\varepsilon$ of any core point.

**Algorithm:** expand clusters by recursively adding all density-reachable points from each unvisited core point. Noise points are left unclustered (label $-1$).

**Advantages over k-means:**
- Finds arbitrarily shaped clusters (not constrained to convex/spherical).
- Detects outliers explicitly.
- Does not require specifying $k$.

**Disadvantages:**
- Requires tuning $\varepsilon$ and $\text{minPts}$ (typically set $\text{minPts} = 2 \times d$ for $d$-dimensional data, then use a k-distance plot to find $\varepsilon$).
- Struggles with varying-density clusters (HDBSCAN addresses this by varying $\varepsilon$).
- $O(n^2)$ naively; with a spatial index $O(n \log n)$.

### Hierarchical Clustering

Builds a **dendrogram** — a binary tree of successive merges (agglomerative) or splits (divisive). Most common: **agglomerative** (bottom-up): start with each point as its own cluster, merge the two closest clusters at each step.

The merge criterion depends on the **linkage function**:

| Linkage | Distance between clusters $A$, $B$ | Shape bias |
|---------|------------------------------------|-----------|
| Single | $\min_{a \in A, b \in B} d(a, b)$ | Chains elongated clusters; sensitive to outliers |
| Complete | $\max_{a \in A, b \in B} d(a, b)$ | Compact, equal-size clusters |
| Average (UPGMA) | Mean pairwise distance | Balance between single and complete |
| Ward | Increase in WCSS from merging | Spherical, roughly equal clusters (similar to k-means) |

**Cutting the dendrogram** at a height $h$ produces clusters; the number of clusters is implicit in the cut height. No need to specify $k$ in advance — you can inspect the dendrogram and decide post-hoc.

---

## Advanced

### Gaussian Mixture Models (GMM)

K-means assigns each point to exactly one cluster (**hard assignment**). A **GMM** models the data as a mixture of $k$ Gaussian components and produces **soft assignments** (probabilities):

$$p(x) = \sum_{j=1}^k \pi_j\, \mathcal{N}(x;\, \mu_j,\, \Sigma_j)$$

where $\pi_j$ are mixing weights ($\sum_j \pi_j = 1$). The responsibility of component $j$ for point $x_i$:

$$r_{ij} = \frac{\pi_j\, \mathcal{N}(x_i;\, \mu_j, \Sigma_j)}{\sum_{l=1}^k \pi_l\, \mathcal{N}(x_i;\, \mu_l, \Sigma_l)}$$

where $r_{ij}$ = posterior probability ("responsibility") that component $j$ generated point $x_i$, numerator = weighted likelihood of $x_i$ under component $j$, denominator = total likelihood (normalizer).

Fit via **Expectation-Maximization (EM)**:
- **E-step:** compute responsibilities $r_{ij}$ using current parameters.
- **M-step:** update $\mu_j$, $\Sigma_j$, $\pi_j$ as weighted averages over all points.

**GMM vs. k-means:**

| Property | K-Means | GMM |
|----------|---------|-----|
| Assignment | Hard (one cluster) | Soft (probabilities) |
| Cluster shape | Spherical (Euclidean distance) | Ellipsoidal (full $\Sigma_j$) |
| Interpretability | Simple | Probabilistic; model selection via BIC/AIC |
| Convergence | Guaranteed (monotone WCSS) | EM converges to local maximum of log-likelihood |

K-means is a special case of GMM with spherical $\Sigma_j = \sigma^2 I$ and hard assignments in the limit $\sigma \to 0$.

### Spectral Clustering

Standard clustering algorithms fail when clusters are non-convex and not separated by low density (e.g., two concentric rings). **Spectral clustering** maps points into a space where clusters are linearly separable, then applies k-means.

**Algorithm:**
1. Build affinity matrix $W$ (e.g., Gaussian kernel $W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2)$).
2. Compute normalized graph Laplacian $L = I - D^{-1/2} W D^{-1/2}$ (where $D_{ii} = \sum_j W_{ij}$).
3. Take the $k$ smallest eigenvectors of $L$ — stack into matrix $U \in \mathbb{R}^{n \times k}$.
4. Cluster the rows of $U$ with k-means.

**Why eigenvectors?** The $k$ smallest eigenvectors of $L$ encode connected components in the affinity graph — points strongly connected by the kernel map to nearby rows in $U$, regardless of their original geometry. This is the [[eigenvalues-pca|spectral]] approach applied to graph structure rather than variance.

### High-Dimensional Clustering

As $d \to \infty$, Euclidean distances concentrate: $\max_i d(x, x_i) / \min_i d(x, x_i) \to 1$ ([[curse-of-dimensionality]]). All points become approximately equidistant — k-means loses discriminative power and silhouette scores degrade toward 0.

**Practical strategies:**
- **Dimensionality reduction first:** PCA or [[eigenvalues-pca|PCA]] to retain 95% of variance, then cluster in low-$d$ space. Works well when structure is in a low-dimensional subspace.
- **Deep clustering:** learn a low-dimensional embedding jointly with cluster assignments (e.g., DEC — Deep Embedded Clustering).
- **Subspace clustering:** assume each cluster lies in a different low-dimensional subspace; fit a union-of-subspaces model.
- **Feature selection:** use sparse methods to identify the subset of dimensions that actually contain cluster structure.

---

## Links

- [[distributions-gaussian]] — Gaussian mixture models are soft-assignment clustering; k-means is hard-assignment GMM with isotropic equal-variance components
- [[linear-algebra-fundamentals]] — spectral clustering (Laplacian Eigenmaps) eigendecomposes the graph Laplacian matrix; the top-$k$ eigenvectors define the $k$-dimensional embedding for k-means
- [[statistical-inference-mle]] — the silhouette score and within-cluster SSE are model selection criteria for choosing $k$; there is no likelihood to maximize in hard k-means
- [[eigenvalues-pca]] — dimensionality reduction (PCA) before clustering mitigates the curse of dimensionality; the top principal components capture most cluster structure
- [[curse-of-dimensionality]] — distance-based clustering (k-means, DBSCAN) degrades in high dimensions as all pairwise distances converge; dimensionality reduction is a prerequisite
- [[evaluation-metrics-guide]] — clustering evaluation uses internal metrics (silhouette, Davies-Bouldin) when labels are unavailable, or ARI/NMI when labels are known
- [[anomaly-detection]] — DBSCAN naturally identifies outliers as points with fewer than `min_samples` neighbors within `eps`; these are labeled as noise, not assigned to a cluster
