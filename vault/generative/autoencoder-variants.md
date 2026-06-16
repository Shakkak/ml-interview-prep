---
title: Autoencoder Variants
tags: [autoencoder, denoising-autoencoder, sparse-autoencoder, contractive-autoencoder, representation-learning]
aliases: [autoencoder, denoising AE, sparse AE, contractive AE, undercomplete autoencoder]
difficulty: 1
status: complete
depends_on: [backpropagation, linear-algebra-fundamentals]
related: [variational-autoencoders, self-supervised-overview, regularization-dropout, energy-based-models, masked-autoencoders]
---

# Autoencoder Variants

---

## Fundamental

### Basic Autoencoder

An **autoencoder** learns to compress data into a low-dimensional representation and reconstruct it:

$$z = f_\theta(x) \quad \text{(encoder)}, \qquad \hat{x} = g_\phi(z) \quad \text{(decoder)}$$

$$\mathcal{L} = \|x - \hat{x}\|^2$$

where $x \in \mathbb{R}^d$ = input data point, $z \in \mathbb{R}^k$ = latent code (with $k \ll d$), $f_\theta$ = encoder (parameterized by $\theta$), $g_\phi$ = decoder (parameterized by $\phi$), and $\hat{x} = g_\phi(f_\theta(x))$ = reconstructed input. The bottleneck forces the model to learn a compact representation capturing the most important structure. The encoder learns a nonlinear dimensionality reduction; the decoder learns to invert it.

**Undercomplete AE:** $k < d$ — bottleneck forces compression. Overcomplete ($k > d$) autoencoders can learn the identity unless regularized.

---

## Intermediate

### Denoising Autoencoder (DAE)

**DAE (Vincent et al., 2008):** corrupt the input before encoding, reconstruct the clean original:

$$\tilde{x} = x + \epsilon \text{ (corruption)}, \qquad \mathcal{L} = \|x - g_\phi(f_\theta(\tilde{x}))\|^2$$

where $\tilde{x}$ = corrupted input, $\epsilon$ = corruption noise or mask, and $g_\phi(f_\theta(\tilde{x}))$ = reconstruction of the clean input from the corrupted version.

Corruption types: Gaussian noise, masking (set pixels to zero), salt-and-pepper.

**Why it works better:** reconstructing from corrupted input forces the encoder to learn robust, stable representations — not just memorize the identity. The model must learn which input features are stable (signal) vs unstable (noise).

**Connection to score matching:** a DAE with Gaussian noise implicitly learns the **score function** $\nabla_x \log p(x)$ — the gradient of the log data density, pointing toward regions of higher probability — directly connecting to [[diffusion-models|diffusion models]] and [[energy-based-models]].

**MAE as extreme DAE:** Masked Autoencoders (see [[masked-autoencoders]]) are denoising autoencoders with 75% masking — extreme corruption requiring semantic reconstruction.

### Sparse Autoencoder (SAE)

**SAE** adds an $\ell_1$ penalty on activations to encourage sparse representations:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \|z\|_1$$

Or uses a KL divergence penalty to keep average activation $\hat{\rho}_j$ close to a target sparsity $\rho$:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \beta \sum_j \text{KL}(\rho \| \hat{\rho}_j)$$

Sparse representations are more interpretable (few active features per input), more generalizable (less overfitting), and suitable for overcomplete dictionaries ($k > d$) where the bottleneck is replaced by sparsity.

**Modern SAEs (Anthropic, 2024):** train sparse autoencoders on internal LLM activations to find interpretable features — individual neurons in the SAE correspond to human-interpretable concepts ("bananas", "DNA sequences", "sycophancy").

### Contractive Autoencoder (CAE)

**CAE (Rifai et al., 2011):** penalizes the Frobenius norm of the encoder Jacobian:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \|J_f(x)\|_F^2 = \|x - \hat{x}\|^2 + \lambda \sum_{ij}\left(\frac{\partial f_j}{\partial x_i}\right)^2$$

where $J_f(x) = \left[\frac{\partial f_j}{\partial x_i}\right]$ = the Jacobian matrix of the encoder $f$ (all partial derivatives of each latent dimension $j$ w.r.t. each input dimension $i$), $\|\cdot\|_F^2$ = sum of squared entries (Frobenius norm squared), and $\lambda$ = regularization strength.

This encourages the representation to be **insensitive to small input perturbations** — the encoder is locally contracted. Small changes in input → small changes in code.

**Geometry:** CAE learns codes that lie on a low-dimensional manifold — the Jacobian penalty forces the encoder to be flat (low-rank Jacobian) in directions perpendicular to the data manifold.

---

## Advanced

### Comparison Table

| Variant | Regularization | Main property | Use case |
|---------|---------------|---------------|----------|
| Undercomplete AE | Bottleneck dimension | Compression | Dimensionality reduction |
| DAE | Input corruption | Robust features | Pretraining, noise robustness |
| SAE | $\ell_1$ on $z$ | Sparse, interpretable codes | Dictionary learning, interpretability |
| CAE | Jacobian Frobenius norm | Smooth, manifold-aligned | Manifold learning |
| VAE | KL on $q(z\|x)$ | Generative, probabilistic | Generation, latent space arithmetic |

### Overcomplete SAEs for Interpretability

The intuition for large overcomplete SAEs ($k \gg d$, e.g., $k = 16d$) on LLM activations: neural networks exhibit **superposition** — a single neuron encodes multiple unrelated concepts by using almost-orthogonal directions in activation space. An overcomplete SAE finds these hidden directions, decomposing each neuron's activity into interpretable components.

## Links

- [[backpropagation]] — autoencoders are trained by backpropagating reconstruction loss through the encoder-decoder; the bottleneck forces compact representations
- [[linear-algebra-fundamentals]] — a linear autoencoder with MSE loss is equivalent to PCA; the encoder weight matrix spans the same subspace as the top principal components
- [[variational-autoencoders]] — VAEs add a probabilistic latent space with KL regularization; autoencoders have no such constraint and can memorize rather than generalize
- [[masked-autoencoders]] — MAE is a denoising autoencoder applied to image patches; the masking corruption forces the encoder to learn robust global representations
- [[energy-based-models]] — contractive autoencoders minimize the Frobenius norm of the encoder Jacobian, connecting AE regularization to energy-based flatness criteria
- [[self-supervised-overview]] — denoising autoencoders are a self-supervised pretraining method; the corruption-reconstruction objective is a form of SSL without labels
