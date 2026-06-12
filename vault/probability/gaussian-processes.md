---
title: Gaussian Processes
tags: [gaussian-processes, bayesian, kernel, non-parametric, regression]
aliases: [GP, GP regression, Gaussian process regression, GPR, Kriging]
difficulty: 3
status: complete
related: [bayesian-inference, kernel-methods, distributions-gaussian, variational-inference, sampling-methods, statistical-inference-mle, neural-tangent-kernel]
---

# Gaussian Processes

---

## Fundamental

### Definition

A **Gaussian process** is a probability distribution over functions: any finite collection of function values $\{f(x_1), \ldots, f(x_n)\}$ is jointly Gaussian. Formally:

$$f \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

where $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$ is the mean function (often $m \equiv 0$) and $k(\mathbf{x}, \mathbf{x}')$ is the **covariance function** (kernel) that encodes assumptions about function smoothness.

**Intuition:** instead of parameterizing a function by weights, a GP places a prior directly over the function space. The kernel $k(\mathbf{x}, \mathbf{x}')$ encodes "how correlated should $f(\mathbf{x})$ and $f(\mathbf{x}')$ be?" — nearby points should behave similarly.

**As a non-parametric method:** a GP has infinitely many "parameters" (one per function evaluation), but inference collapses to finite-dimensional Gaussian algebra at training points. See [[parametric-nonparametric]].

### GP Regression

Given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ with $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$:

**Prior:** $\mathbf{f} = [f(x_1), \ldots, f(x_n)]^\top \sim \mathcal{N}(\mathbf{0}, K_{XX})$

where $K_{XX}$ is the $n \times n$ **Gram matrix** with $[K_{XX}]_{ij} = k(x_i, x_j)$.

**Posterior** at test points $X_*$: conditioning a multivariate Gaussian gives:

$$f(X_*)|\mathcal{D} \sim \mathcal{N}(\bar{f}_*, \text{Cov}_*)$$

$$\bar{f}_* = K_{X_*X}(K_{XX} + \sigma_n^2 I)^{-1}\mathbf{y}$$
$$\text{Cov}_* = K_{X_*X_*} - K_{X_*X}(K_{XX} + \sigma_n^2 I)^{-1} K_{XX_*}$$

**Key insight:** the predictive mean $\bar{f}_*$ is a linear combination of training observations — a kernel smoother. The predictive variance $\text{Cov}_*$ quantifies uncertainty: it is large far from training points and small near them.

**Computational cost:** requires inverting $K_{XX} + \sigma_n^2 I$ — $O(n^3)$ time and $O(n^2)$ memory. The bottleneck for large datasets.

---

## Intermediate

### Common Kernels

The choice of kernel encodes beliefs about the function's properties:

| Kernel | Formula | Properties |
|--------|---------|-----------|
| Squared exponential (RBF) | $\exp(-\|x-x'\|^2 / 2\ell^2)$ | Infinitely differentiable, very smooth |
| Matérn $\nu=3/2$ | $(1 + \sqrt{3}r/\ell)\exp(-\sqrt{3}r/\ell)$ | Once differentiable |
| Matérn $\nu=5/2$ | $(1 + \sqrt{5}r/\ell + 5r^2/3\ell^2)\exp(-\sqrt{5}r/\ell)$ | Twice differentiable |
| Periodic | $\exp(-2\sin^2(\pi r / p) / \ell^2)$ | Captures periodicity |
| Linear | $\sigma_b^2 + \sigma_v^2 (x - c)(x' - c)$ | Bayesian linear regression as special case |

where $r = \|x - x'\|$ and $\ell$ is the length-scale hyperparameter.

**Matérn kernels** are preferred over RBF in practice because RBF's infinite smoothness is unrealistic for real-world data. Physical processes are typically only finitely differentiable.

**Kernel composition:** kernels can be added (modeling multiple length scales) or multiplied (joint smoothness requirements). A GP with a polynomial kernel is Bayesian polynomial regression. A GP with an RBF + linear kernel models a smooth trend with a linear component.

### Marginal Likelihood and Hyperparameter Learning

GP hyperparameters (length-scale $\ell$, signal variance $\sigma_f^2$, noise variance $\sigma_n^2$) are optimized by maximizing the **log marginal likelihood**:

$$\log p(\mathbf{y}|X, \theta) = -\frac{1}{2}\mathbf{y}^\top(K + \sigma_n^2 I)^{-1}\mathbf{y} - \frac{1}{2}\log|K + \sigma_n^2 I| - \frac{n}{2}\log 2\pi$$

- First term: data fit (penalizes poor fit)
- Second term: complexity penalty (penalizes over-fit via large $\log|K|$)
- This is **automatic Occam's razor** — the marginal likelihood balances fit against model complexity without a separate validation set.

Optimize with L-BFGS or Adam using automatic differentiation (GPyTorch, GPflow). Multiple restarts are needed because the marginal likelihood is non-convex in $\theta$.

### GP Classification

For classification, the likelihood $p(y|f)$ is non-Gaussian (e.g., Bernoulli with sigmoid). The posterior $p(\mathbf{f}|\mathbf{y})$ is no longer Gaussian — inference requires approximation:

- **Laplace approximation:** find the mode of $p(\mathbf{f}|\mathbf{y})$ using Newton's method, then approximate with a Gaussian at the mode. $O(n^3)$ cost.
- **Expectation Propagation (EP):** approximate non-Gaussian factors with Gaussians iteratively. More accurate than Laplace.
- **Variational methods:** lower-bound the marginal likelihood and optimize (see [[variational-inference]]).

---

## Advanced

### Sparse GPs and Inducing Points

$O(n^3)$ cost is prohibitive for $n > 10^4$. Sparse GP approximations introduce $m \ll n$ **inducing points** $Z = \{z_1, \ldots, z_m\}$ and approximate the full GP:

$$p(\mathbf{f}) \approx \int p(\mathbf{f}|\mathbf{u}) q(\mathbf{u})\, d\mathbf{u}$$

where $\mathbf{u} = f(Z)$ are function values at inducing points.

**FITC (Fully Independent Training Conditional):** the posterior $p(f_i|\mathbf{u})$ is independent given $\mathbf{u}$. Cost: $O(nm^2 + m^3)$.

**SVGP (Stochastic Variational GP, Hensman et al., 2013):** optimize inducing point locations and variational parameters jointly using mini-batch SGD. Scales to millions of data points. The ELBO decomposes over data points when using mini-batches.

### GP and Neural Networks

**NNGP (Neal, 1994; later Matthews et al., 2018):** as the width of a single-hidden-layer network $\to \infty$ with i.i.d. weight initialization, the network prior converges to a GP with a specific kernel (the NNGP kernel). This means infinitely wide neural networks ARE Gaussian processes at initialization.

**Deep kernel learning:** use a deep neural network $\phi(\mathbf{x})$ as the feature map and apply a GP on top: $k(\mathbf{x}, \mathbf{x}') = k_0(\phi(\mathbf{x}), \phi(\mathbf{x}'))$. Learns representations while maintaining GP uncertainty quantification. The end-to-end training is the Gaussian process posterior conditioned on the NN's learned representations.

**Neural Tangent Kernel (NTK):** the training dynamics of infinitely wide networks (see [[neural-tangent-kernel]]) are exactly described by a GP regression with the NTK — predictions at any test point are linear in the initial network outputs, with the NTK as the kernel.

### Deep GPs

A deep GP is a composition of GPs: $f^{(l+1)} = f^{(l)} \circ f^{(l-1)} \circ \cdots$ where each $f^{(l)}$ is a GP. Unlike a shallow GP, deep GPs can model non-stationary functions and adaptively allocate uncertainty. Inference is intractable — approximated using doubly stochastic variational inference (Salimbeni & Deisenroth, 2017), which uses inducing point approximations at each layer.

---

*See also: [[bayesian-inference]] · [[kernel-methods]] · [[distributions-gaussian]] · [[variational-inference]] · [[neural-tangent-kernel]] · [[statistical-inference-mle]] · [[sampling-methods]]*
