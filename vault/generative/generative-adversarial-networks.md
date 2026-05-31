---
title: Generative Adversarial Networks
tags: [generative-models, gans, adversarial-training, deep-learning, image-generation, wgan, stylegan, fid]
aliases: [GAN, GANs, adversarial training, generator discriminator, WGAN, StyleGAN, mode collapse]
difficulty: 2
status: complete
related: [diffusion-models, variational-autoencoders, normalization-layers, loss-cross-entropy]
---

# Generative Adversarial Networks

---

## Fundamental

Two networks in adversarial competition:
- **Generator** $G(z; \theta_G)$: maps random noise $z \sim p_z = \mathcal{N}(0,I)$ to data space. Wants to produce samples the discriminator cannot tell from real data.
- **Discriminator** $D(x; \theta_D)$: classifies inputs as real or fake. Wants to correctly identify real vs. generated.

**Minimax objective:**
$$\min_G \max_D\;V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

**Optimal discriminator** (given fixed $G$): at each $x$, the optimal $D$ is
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

When $p_g = p_{\text{data}}$: $D^*(x) = \frac{1}{2}$ everywhere — the discriminator cannot distinguish real from generated. This is the Nash equilibrium.

**Substituting $D^*$ into $V$:** the generator's objective at optimal $D$ equals $2\,D_{JS}(p_{\text{data}} \| p_g) - \log 4$. Training the GAN minimizes Jensen-Shannon divergence between the data and generator distributions.

---

## Intermediate

### Training Instabilities

**Vanishing gradient (saturation):** early in training, $D$ is strong and $D(G(z)) \approx 0$. The gradient of $\log(1-D(G(z)))$ vanishes. **Fix:** use the non-saturating generator loss:
$$L_G^{\text{non-sat}} = -\mathbb{E}_z[\log D(G(z))]$$

When $D(G(z))=0.01$: saturating gradient $\approx -0.01$ vs non-saturating gradient $\approx 4.6$ — 460× larger.

**Mode collapse:** the generator finds one or a few outputs that reliably fool the discriminator and ignores the rest of the data distribution. Causes: the game dynamics cycle — $G$ finds a mode, $D$ adapts, $G$ jumps to another mode, indefinitely. Partial fixes: minibatch discrimination (allow $D$ to compare samples within a batch), unrolled GANs, historical averaging of parameters.

**Training oscillation:** the minimax game has no guaranteed convergence. The discriminator and generator can cycle without reaching equilibrium.

### DCGAN: Making GANs Work on Images

Four architectural rules that made GANs reliably train on images:
1. Replace pooling with **strided convolutions** (D) and **transposed convolutions** (G).
2. **Batch normalization** everywhere (except $G$'s output layer and $D$'s input layer).
3. No fully connected hidden layers.
4. **ReLU** in $G$ (Tanh at output), **LeakyReLU** in $D$.

### Evaluation: FID and Inception Score

**FID (Fréchet Inception Distance):** extract Inception-v3 features from real and generated samples; fit Gaussians $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$; compute:
$$\text{FID} = \|\mu_r-\mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g})$$

Lower is better. Captures both **fidelity** (individual samples look real) and **diversity** (all modes of the distribution are covered). Requires ~50k samples for stable estimates.

**Inception Score (IS):** $\text{IS} = \exp(\mathbb{E}_x[D_{KL}(p(y|x)\|p(y))])$. Rewards sharp, diverse class predictions. Does not compare to real data — can be gamed. FID is the standard today.

---

## Advanced

### Wasserstein GAN: Fixing the Gradient Problem

**Why JS divergence fails:** when $p_{\text{data}}$ and $p_g$ have non-overlapping support (common at the start of training), $D_{JS} = \log 2$ regardless of how far apart they are. Zero gradient to the generator.

**Wasserstein (Earth Mover) Distance:**
$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$$

$W$ is the minimum cost of transporting mass from $p_g$ to $p_{\text{data}}$. Unlike JS, $W$ is continuous and differentiable even for non-overlapping distributions.

**Kantorovich-Rubinstein duality** transforms $W$ into an objective:
$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1}\mathbb{E}_{p_{\text{data}}}[f(x)] - \mathbb{E}_{p_g}[f(G(z))]$$

where the supremum is over 1-Lipschitz functions. Replace the discriminator with a **critic** $f_w$ and enforce the Lipschitz constraint via:

- **Weight clipping** (original WGAN): clip all $w_i \in [-c, c]$. Works but can cause gradient flow issues.
- **Gradient penalty** (WGAN-GP): add $\lambda\,\mathbb{E}_{\hat x}[(\|\nabla_{\hat x} f_w(\hat x)\|_2 - 1)^2]$ where $\hat x = \epsilon x + (1-\epsilon)G(z)$.

**Benefits of WGAN:** stable training, no mode collapse, critic loss correlates with sample quality (useful signal during training).

### StyleGAN: Disentangled Generation

Progressive training (4×4 → 8×8 → … → 1024×1024) stabilizes high-resolution GAN training. StyleGAN (Karras et al., 2019) adds disentanglement:

**Mapping network:** 8-layer MLP maps $z \sim \mathcal{N}(0,I)$ to intermediate latent $w$. The $\mathcal{W}$-space is far more disentangled than $\mathcal{Z}$-space because the mapping network can untangle the (unconstrained) distribution.

**Adaptive Instance Normalization (AdaIN):** inject $w$ at each resolution level:
$$\text{AdaIN}(x_i, y) = y_{s,i}\frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$
where $y = \text{linear}(w)$. Coarse styles (pose, face shape) controlled at low resolutions; fine styles (color, texture) at high resolutions.

**StyleGAN2** removes AdaIN artifacts (droplet artifacts) via **weight demodulation**: instead of normalizing feature statistics, demodulate the conv weights themselves before each forward pass. Clean high-resolution samples with FID < 3 on FFHQ.

---

*See also: [[diffusion-models]] · [[variational-autoencoders]] · [[normalization-layers]]*
