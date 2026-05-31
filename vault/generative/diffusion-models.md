---
title: Diffusion Models
tags: [generative-models, deep-learning, probabilistic-modeling, score-matching, ddpm, ddim, stable-diffusion]
aliases: [DDPM, score matching, denoising diffusion, DDIM, latent diffusion, classifier-free guidance]
difficulty: 2
status: complete
related: [variational-autoencoders, bayesian-inference, unet, normalizing-flows, activation-gelu-swish]
---

# Diffusion Models

---

## Fundamental

The key insight behind diffusion: **it is easy to destroy an image** (add Gaussian noise step by step until it becomes pure noise). **Learning to reverse this** — predict the noise added at each step — is a tractable supervised regression problem. Chain many small reverse steps together to generate from pure noise.

### The Forward Process

Given a data point $x_0 \sim q(x_0)$, define a Markov chain that gradually adds Gaussian noise:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)$$

$\beta_t$ is the noise schedule (small, increases over time). After $T=1000$ steps, $x_T \approx \mathcal{N}(0,I)$.

**Closed-form sampling at any timestep:** define $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$:

$$\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)}$$

This allows sampling $x_t$ directly from $x_0$ without simulating all $t$ intermediate steps — critical for efficient training.

### Training Objective (DDPM Simplified Loss)

Train a U-Net $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ that was added:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,x_0,\,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\, t)\|^2\right]}$$

**Algorithm:** (1) sample $x_0$ from data; (2) sample random $t$ and noise $\epsilon$; (3) compute $x_t$; (4) predict $\hat\epsilon = \epsilon_\theta(x_t, t)$; (5) backprop on MSE. Just supervised regression — no adversarial training, no posterior approximation.

---

## Intermediate

### Noise Schedules

**Linear schedule (DDPM):** $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T-\beta_1)$, with $\beta_1=10^{-4}$, $\beta_T=0.02$. Destroys information rapidly early in the schedule — many easy timesteps near $t=0$ waste training capacity.

**Cosine schedule (Improved DDPM):** defines $\bar\alpha_t = \cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)/\cos^2\!\left(\frac{s}{1+s}\cdot\frac{\pi}{2}\right)$. Decreases signal-to-noise ratio more gradually, distributing learning difficulty evenly → better sample quality.

### Sampling: DDPM vs DDIM

**DDPM reverse step:**
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\tilde\beta_t}\,z, \quad z \sim \mathcal{N}(0,I)$$

Requires 1000 steps — slow.

**DDIM (Song et al., 2020):** deterministic non-Markovian update that uses the same trained $\epsilon_\theta$:
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{"predicted }x_0\text{"}} + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta$$

50 DDIM steps rival 1000-step DDPM in quality. Key property: DDIM is **invertible** — encode a real image by running the forward DDIM steps; this enables image editing in noise space.

### Conditioning: Classifier-Free Guidance (CFG)

Train the same model jointly on conditioned and unconditioned inputs (10% of training examples, drop the condition $c$ and replace with null token $\emptyset$). At inference, linearly extrapolate:

$$\hat\epsilon = \epsilon_\theta(x_t, \emptyset) + w\!\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\right)$$

Guidance scale $w > 1$ amplifies the conditioning signal — trades diversity for fidelity. In practice $w \in [5, 12]$ for text-to-image. Single model, no separate classifier needed.

---

## Advanced

### Score Matching Equivalence

**Score function** of a distribution: $\nabla_x \log p(x)$ — points toward regions of higher probability. **Score matching** trains $s_\theta(x) \approx \nabla_x \log p(x)$.

The connection to diffusion: predicting the noise $\epsilon$ is equivalent to predicting the score of the noisy distribution:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$$

This unification, formalized via Stochastic Differential Equations (Song et al., 2021), expresses the forward process as an SDE $dx = f(x,t)dt + g(t)dW$ and the reverse as a reverse-time SDE. This framework allows using any off-the-shelf numerical SDE/ODE solver for faster sampling — including higher-order solvers that achieve quality comparable to 1000 DDPM steps in 10–25 function evaluations.

### Latent Diffusion Models (LDM / Stable Diffusion)

Running diffusion in pixel space at 512×512 costs $\approx 512^2 \times T$ operations per training step. **LDM** (Rombach et al., 2022) runs diffusion in the **latent space of a pre-trained VAE**:

1. Train a VAE encoder $\mathcal{E}$ and decoder $\mathcal{D}$ on images.
2. Compress image: $z = \mathcal{E}(x)$, $z \in \mathbb{R}^{64 \times 64 \times 4}$ (8× spatial compression).
3. Train diffusion model entirely in latent space: $\epsilon_\theta(z_t, t, c)$.

Compute cost reduction: $(512/64)^2 = 64\times$ fewer spatial positions. The diffusion model never sees pixels — only the 64×64 latent code. The VAE decoder reconstructs the final image from the denoised latent.

Stable Diffusion's U-Net has 860M parameters and operates on 64×64 latent features. Text conditioning is injected via cross-attention (CLIP ViT-L/14 text embeddings). This architecture became the standard for all subsequent open-source text-to-image models.

### Why Diffusion Dominates GANs

| Property | GAN | Diffusion |
|---|---|---|
| Training stability | Fragile (adversarial, mode collapse) | Stable (denoising regression) |
| Mode coverage | Often misses modes | Covers full distribution |
| Training recipe | Delicate (balance G/D, spectral norm) | Simple MSE loss |
| Sample quality | High but artifacts | State-of-the-art |
| Inference speed | Single forward pass | 20–1000 steps (DDIM: 20–50) |
| Latent editing | Difficult | Natural via DDIM inversion |

The only remaining advantage of GANs is single-step inference. Consistency Models (Song et al., 2023) train a model to directly map any noisy $x_t$ to $x_0$, achieving comparable quality in 1–2 steps — potentially closing this gap.

---

*See also: [[variational-autoencoders]] · [[unet]] · [[normalizing-flows]] · [[bayesian-inference]]*
