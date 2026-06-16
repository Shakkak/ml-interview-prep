---
title: Normalizing Flows
tags: [normalizing-flows, change-of-variables, log-det-jacobian, realnvp, glow, invertible-networks, exact-likelihood]
aliases: [normalizing flows, flow-based model, RealNVP, Glow, invertible network, change of variables, Jacobian determinant]
difficulty: 3
status: complete
depends_on: [variational-autoencoders, distributions-gaussian, loss-kl-divergence]
related: [variational-autoencoders, generative-adversarial-networks, diffusion-models, math-svd, loss-kl-divergence]
---

# Normalizing Flows

---

## Fundamental

[[variational-autoencoders|VAEs]] optimize a **lower bound** on the likelihood (ELBO, not exact). [[generative-adversarial-networks|GANs]] don't optimize likelihood at all — there is no way to evaluate $p(x)$ for a trained GAN. **Normalizing flows** achieve **exact, tractable likelihood** by constructing $p(x)$ as a sequence of invertible transformations from a simple base distribution.

### Change of Variables

Let $z \sim p_Z(z)$ be a simple distribution (e.g., [[distributions-gaussian|$\mathcal{N}(0,I)$]]) and $x = f(z)$ where $f$ is invertible and differentiable. The density of $x$:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left|\det\frac{\partial f^{-1}}{\partial x}\right|$$

$$\log p_X(x) = \log p_Z(z) - \log\left|\det\frac{\partial f}{\partial z}\right|$$

where $p_Z(z)$ = density of the simple base distribution at $z = f^{-1}(x)$, $\frac{\partial f}{\partial z}$ = the Jacobian matrix of $f$ (all partial derivatives of each output w.r.t. each input dimension), $\det(\cdot)$ = determinant (measures the volume scaling factor), $|\cdot|$ = absolute value (density is always positive), and $\log|\det J|$ = log-Jacobian (the correction for how much $f$ stretches or compresses space — subtracted because stretching space decreases density).

**Intuition:** the Jacobian determinant measures how much $f$ locally stretches or compresses volume. If $f$ stretches a region by factor 2 (det = 2), density decreases by factor 2. The log-det-Jacobian is the volume-correction term.

**1D worked example:** $z \sim \mathcal{N}(0,1)$, $f(z) = 2z+1$ (scale and shift). Then $x \sim \mathcal{N}(1, 4)$. Log-det-Jacobian $= \log|2| = \log 2$. Stretching the distribution by 2 decreases density by 2.

### Composing Flows

Stack $K$ invertible transformations $z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} \cdots \xrightarrow{f_K} x$:

$$\log p_X(x) = \log p_Z(z_0) - \sum_{k=1}^K \log\left|\det J_{f_k}\right|$$

where $z_0$ = sample from the base distribution, $K$ = number of flow transformations stacked in sequence, $J_{f_k} = \frac{\partial f_k}{\partial z_{k-1}}$ = Jacobian matrix of the $k$-th transformation, and $\sum_{k=1}^K \log|\det J_{f_k}|$ = total volume change accumulated across all $K$ transformations.

**Training:** maximize $\log p_X(x)$ directly — exact MLE, no approximation. **Sampling:** sample $z_0 \sim p_Z$, apply $f_1, \ldots, f_K$. **Density evaluation:** invert $f_K, \ldots, f_1$ to get $z_0$, accumulate log-dets.

---

## Intermediate

### The Log-Det-Jacobian Bottleneck

For a general $d$-dimensional transformation, computing $\det(J)$ costs $O(d^3)$. For $d = 256 \times 256 \times 3 \approx 200k$ (an image), this is completely intractable.

Flow architectures are designed so the Jacobian is **triangular**, making the determinant a product of diagonal entries:

$$\det(J) = \prod_i \frac{\partial f_i}{\partial z_i}$$

This is $O(d)$ instead of $O(d^3)$. The art of normalizing flows is constructing expressive invertible transformations with triangular Jacobians.

### RealNVP — Affine Coupling Layers

Split input $z$ into two halves $z_{1:d/2}$ and $z_{d/2+1:d}$:

$$y_{1:d/2} = z_{1:d/2}$$
$$y_{d/2+1:d} = z_{d/2+1:d} \odot \exp(s(z_{1:d/2})) + t(z_{1:d/2})$$

where $s(\cdot)$ and $t(\cdot)$ are arbitrary neural networks (scale and translation). The Jacobian is **lower triangular** (top-left block is identity; bottom-right block is diagonal $\text{diag}(\exp(s(\cdot)))$):

$$\log|\det J| = \sum_i s_i(z_{1:d/2})$$

**Inverse (free):**
$$z_{1:d/2} = y_{1:d/2}, \quad z_{d/2+1:d} = (y_{d/2+1:d} - t(y_{1:d/2})) \odot \exp(-s(y_{1:d/2}))$$

The inverse requires only evaluating $s$ and $t$ in the forward direction — no matrix inversion needed. Stack multiple coupling layers with **alternating masks** (which half is "passed through") to transform all dimensions.

### Glow: Invertible 1×1 Convolutions

RealNVP uses fixed channel shuffles between coupling layers. Glow (Kingma & Dhariwal, 2018) replaces this with a **learned $1 \times 1$ convolution** $W \in \mathbb{R}^{c \times c}$:

$$y = Wz, \qquad \log|\det J| = h \cdot w \cdot \log|\det W|$$

where $W \in \mathbb{R}^{c \times c}$ = learned $1\times1$ convolution weight matrix (mixes channels), $h$ and $w$ = spatial height and width of the feature map (the $1\times1$ conv is applied independently at each of $h \times w$ spatial positions), and $h \cdot w \cdot \log|\det W|$ = total log-Jacobian (the same $c\times c$ mixing is applied at every spatial location, so the per-location log-det is multiplied by the number of locations). To compute $\det W$ efficiently, decompose $W = PLU$ (LU decomposition with permutation): $\det(W) = \det(U) = \prod_i U_{ii}$. This costs $O(c^3)$ once and $O(c)$ per forward pass.

Glow also adds **ActNorm** (per-channel affine transform initialized from the first batch, replacing [[normalization-layers|batch norm]]). Together: stable training, high-quality 256×256 image synthesis, smooth latent interpolation.

---

## Advanced

### Continuous Normalizing Flows and Flow Matching

Discrete flows apply $K$ fixed transformations. **Continuous normalizing flows (CNFs)** define a continuous-time transformation via an ODE:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t), \quad z(0) = z_0 \sim p_Z, \quad z(1) = x$$

The change in log-density follows the instantaneous change of variables theorem:

$$\frac{d\log p(z(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial z(t)}\right)$$

Computing the full Jacobian trace costs $O(d^2)$ — but can be estimated unbiasedly in $O(d)$ using the Hutchinson trace estimator: $\text{tr}(A) \approx \epsilon^T A \epsilon$ for random $\epsilon$.

> [!tip] Why $\epsilon^T A \epsilon$ is an unbiased estimate of $\text{tr}(A)$ ([[matrix-calculus]])
> For any random vector $\epsilon$ with $\mathbb{E}[\epsilon\epsilon^T] = I$ (e.g., $\epsilon \sim \mathcal{N}(0,I)$ or $\epsilon_i \sim \pm 1$ Rademacher):
> $\mathbb{E}[\epsilon^T A \epsilon] = \mathbb{E}\!\left[\sum_{ij} \epsilon_i A_{ij} \epsilon_j\right] = \sum_{ij} A_{ij}\,\mathbb{E}[\epsilon_i\epsilon_j] = \sum_{ij} A_{ij}\,\delta_{ij} = \sum_i A_{ii} = \text{tr}(A)$.
> Computing $A\epsilon$ costs $O(d^2)$ for a dense $A$ — but here $A = \partial f/\partial z$ and the matrix-vector product can be computed via one JVP call without materialising $A$, making the full estimator $O(d)$.

**Flow Matching** (Lipman et al., 2022) provides a simpler training objective that avoids simulating the ODE during training: directly regress the vector field $f_\theta$ on analytically computable conditional vector fields $u_t(x|x_0)$. This yields training as simple as DDPM but produces a continuous flow that can be simulated exactly at test time.

### Comparison to Other Generative Models

| Property | VAE | GAN | Diffusion | Normalizing Flow |
|---|:-:|:-:|:-:|:-:|
| Exact likelihood | ✗ (ELBO) | ✗ | ✗ (ELBO) | ✓ |
| Stable training | ✓ | ✗ | ✓ | ✓ |
| Sample quality | Medium | High | Very high | Medium |
| Exact latent inference | Approx | ✗ | Via DDIM | ✓ |
| Architecture constraint | Encoder-decoder | $G$/$D$ pair | U-Net | Invertible only |

**When to choose flows:** exact likelihoods for anomaly detection, density estimation, or Bayesian inference over latents. For image generation, [[diffusion-models|diffusion]] is currently superior. Flows as **priors in VAEs** (replace $\mathcal{N}(0,I)$ with a flow-based prior) improve VAE sample quality with exact likelihood and tractable inference.

### The Expressiveness-Tractability Tradeoff

The fundamental tension: more expressive transformations → better density estimates, but the Jacobian must remain tractable (triangular or structured). This creates three regimes:

1. **Autoregressive flows** (MAF, IAF): fully autoregressive transformations with lower-triangular Jacobians. Very expressive (one pass per element). MAF: fast training/density estimation, slow sampling; IAF: fast sampling, slow training.
2. **Coupling flows** (RealNVP, Glow): split-and-transform. Moderate expressiveness; fast in both directions.
3. **CNFs with trace estimator:** full flexibility (any ODE vector field), $O(d)$ Hutchinson estimator, but slow due to ODE solving.

Flow Matching + CNF is currently the most active research direction, achieving strong generation quality with simple MSE-like training.

---

## Links

- [[variational-autoencoders]] — VAEs use an ELBO lower bound on log-likelihood; flows compute exact log-likelihood via the change-of-variables formula
- [[diffusion-models]] — diffusion models are an implicit multi-step flow; normalizing flows are explicit bijections computed in a single forward pass
- [[generative-adversarial-networks]] — GANs learn implicit densities but cannot evaluate log-likelihoods; flows trade flexibility for tractability
- [[loss-kl-divergence]] — training maximizes log-likelihood, equivalent to minimizing KL from model to data distribution
- [[linear-algebra-fundamentals]] — the change-of-variables formula requires a Jacobian determinant; efficient flow architectures make this $O(d)$
- [[distributions-gaussian]] — the base distribution is typically $\mathcal{N}(0,I)$; the flow transforms it into the data distribution
- [[normalization-layers]] — ActNorm (used in Glow) is a flow-based normalization layer with data-dependent initialization
- [[anomaly-detection]] — flows assign exact log-likelihoods, making them natural density estimators for anomaly scoring
