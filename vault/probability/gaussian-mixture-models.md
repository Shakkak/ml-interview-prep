---
title: Gaussian Mixture Models
tags: [gaussian-mixture-models, gmm, clustering, density-estimation, em-algorithm]
aliases: [GMM, Gaussian mixture, mixture model, mixture of Gaussians]
difficulty: 2
status: complete
related: [distributions-gaussian, expectation-maximization, clustering, variational-inference, bayesian-inference, distributions-overview, sampling-methods]
---

# Gaussian Mixture Models

---

## Fundamental

### Definition

A Gaussian Mixture Model is a **weighted sum of Gaussian distributions**. The joint distribution is:

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \Sigma_k)$$

where $\pi_k \geq 0$ are **mixing weights** (summing to 1), $\boldsymbol{\mu}_k$ are component means, and $\Sigma_k$ are component covariance matrices.

**Generative story:**
1. Choose a component: $z \sim \text{Categorical}(\pi_1, \ldots, \pi_K)$
2. Sample from that component: $\mathbf{x} | z = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$

The variable $z$ is **latent** — we observe $\mathbf{x}$ but not which component generated it.

**Why GMMs?** Gaussians have closed-form everything, but a single Gaussian can't model multimodal, skewed, or heavy-tailed distributions. A mixture of $K$ Gaussians can approximate any smooth density arbitrarily well with enough components (universal density approximator).

### Parameter Estimation via EM

GMMs are fit by maximizing the log-likelihood using the **EM algorithm** (see [[expectation-maximization]]):

**E-step:** compute the responsibility (posterior probability that component $k$ generated $\mathbf{x}_n$):
$$r_{nk} = \frac{\pi_k \mathcal{N}(\mathbf{x}_n; \mu_k, \Sigma_k)}{\sum_{j} \pi_j \mathcal{N}(\mathbf{x}_n; \mu_j, \Sigma_j)}$$

**M-step:** update parameters using weighted statistics:
$$\mu_k = \frac{\sum_n r_{nk} \mathbf{x}_n}{\sum_n r_{nk}}, \qquad \Sigma_k = \frac{\sum_n r_{nk}(\mathbf{x}_n - \mu_k)(\mathbf{x}_n - \mu_k)^\top}{\sum_n r_{nk}}, \qquad \pi_k = \frac{\sum_n r_{nk}}{N}$$

Guaranteed to non-decrease the log-likelihood at each iteration. Typically initialized with k-means.

---

## Intermediate

### GMM vs K-Means

K-means is hard EM with isotropic covariances:

| Property | GMM | K-Means |
|----------|-----|---------|
| Assignment | Soft (responsibility $r_{nk}$) | Hard (argmax) |
| Cluster shape | Arbitrary ellipsoids | Spherical (isotropic $\Sigma = \sigma^2 I$) |
| Output | Full posterior + density | Labels only |
| Objective | Log-likelihood (ELBO) | Reconstruction error $\sum_n \|x_n - \mu_{z_n}\|^2$ |
| Outlier sensitivity | Moderate | High |

**In practice:** k-means is faster and good for initialization. GMM is richer but slower and sensitive to initialization. Use k-means to initialize GMM for robust convergence.

### Covariance Structures

Fitting a full $d \times d$ covariance matrix per component requires $O(Kd^2)$ parameters — expensive and prone to overfitting. Common constraints:

| Type | Structure | Parameters per component | Notes |
|------|-----------|-------------------------|-------|
| Full | Arbitrary $\Sigma_k$ | $d(d+1)/2$ | Most flexible, can overfit |
| Tied | All $\Sigma_k = \Sigma$ | $d(d+1)/2$ (shared) | Good for similar-shaped clusters |
| Diagonal | $\Sigma_k = \text{diag}(\sigma_{k,1}^2, \ldots)$ | $d$ | Independent features (Naive Bayes) |
| Spherical | $\Sigma_k = \sigma_k^2 I$ | 1 | K-means limit |

**Degenerate solutions:** a component can collapse onto a single data point — $\boldsymbol{\mu}_k = \mathbf{x}_n$ for some $n$, with $\Sigma_k \to 0$, making the likelihood $\to \infty$. Fix: add a regularization term $\epsilon I$ to covariances, or use a Bayesian GMM with an Inverse-Wishart prior.

### Model Selection: Choosing K

GMM log-likelihood always increases with more components — we need regularization:

**BIC (Bayesian Information Criterion):**
$$\text{BIC} = -2\log p(\mathcal{D}|\hat\theta) + p\log N$$

where $p$ = number of free parameters, $N$ = number of samples. Penalizes model complexity.

**AIC (Akaike Information Criterion):**
$$\text{AIC} = -2\log p(\mathcal{D}|\hat\theta) + 2p$$

AIC penalizes less — tends to choose more components. BIC is consistent (chooses correct $K$ as $N \to \infty$ if the model is correct); AIC is not.

**Dirichlet Process GMM:** Bayesian non-parametric alternative — automatically infers $K$ from data. Uses a DP prior over the number of components. Inference via variational EM or Gibbs sampling.

### Applications

**Density estimation:** model $p(\mathbf{x})$ as a GMM and evaluate likelihood for new samples. Used for anomaly detection — low $p(\mathbf{x})$ indicates anomaly (see [[anomaly-detection]]).

**Soft clustering:** $r_{nk}$ gives cluster membership probabilities rather than hard labels. Useful when data points plausibly belong to multiple clusters.

**Generative modeling:** once fit, sample by (1) drawing $k \sim \text{Categorical}(\pi)$, (2) drawing $\mathbf{x} \sim \mathcal{N}(\mu_k, \Sigma_k)$. GMMs are used as priors in some diffusion model guidance methods.

**Speech recognition:** GMMs historically modeled acoustic features in HMM-GMM systems before deep learning.

---

## Advanced

### Variational Bayesian GMM

The maximum-likelihood GMM has multiple local optima and degenerate solutions. A **Variational Bayesian GMM** places conjugate priors over all parameters:

- $\pi \sim \text{Dirichlet}(\alpha_0)$ — symmetric prior over mixture weights
- $\mu_k | \Sigma_k \sim \mathcal{N}(m_0, \beta_0^{-1}\Sigma_k)$ — Gaussian prior on means
- $\Sigma_k \sim \text{Wishart}(W_0, \nu_0)$ — Wishart prior on precision matrices

The ELBO over the variational posterior $q(\mathbf{Z}, \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Lambda}) = q(\mathbf{Z}) q(\boldsymbol{\pi}) \prod_k q(\mu_k, \Lambda_k)$ is maximized by coordinate ascent. Components with low effective $N_k$ are automatically pruned — the Dirichlet prior drives $\pi_k \to 0$ for unused components. This **effectively performs model selection** without cross-validation.

### GMMs and the Riemannian Manifold

GMMs live on the statistical manifold of positive-definite matrices plus the simplex for mixing weights. The natural gradient (Fisher information metric, see [[fisher-information]]) of the GMM log-likelihood is much more efficient than Euclidean gradient descent — it accounts for the curvature of the probability space. Natural gradient EM converges faster than standard EM in high dimensions.

---

*See also: [[distributions-gaussian]] · [[expectation-maximization]] · [[clustering]] · [[variational-inference]] · [[bayesian-inference]] · [[anomaly-detection]] · [[distributions-overview]]*
