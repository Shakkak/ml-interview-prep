---
title: Diffusion Models
tags: [generative-models, deep-learning, probabilistic-modeling, score-matching, ddpm, ddim, stable-diffusion]
aliases: [DDPM, score matching, denoising diffusion, DDIM, latent diffusion, classifier-free guidance]
difficulty: 2
status: complete
depends_on: [variational-autoencoders, distributions-gaussian, loss-mse]
related: [variational-autoencoders, bayesian-inference, unet, normalizing-flows, activation-gelu-swish]
---

# Diffusion Models

---

## Fundamental

The key insight behind diffusion: **it is easy to destroy an image** (add [[distributions-gaussian|Gaussian]] noise step by step until it becomes pure noise). **Learning to reverse this** — predict the noise added at each step — is a tractable supervised regression problem. Chain many small reverse steps together to generate from pure noise.

### The Forward Process

Given a data point $x_0 \sim q(x_0)$, define a Markov chain that gradually adds Gaussian noise:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\,x_{t-1},\, \beta_t I\right)$$

where $x_0$ = original clean data point, $x_t$ = noisy version after $t$ steps, $\beta_t \in (0,1)$ = noise schedule at step $t$ (controls how much noise is added at each step — small at first, increasing over time), $\sqrt{1-\beta_t}$ = scaling factor that shrinks the signal so that total energy stays constant, and $\beta_t I$ = variance of the added Gaussian noise (isotropic, same in all dimensions).

$\beta_t$ is the noise schedule (small, increases over time). After $T=1000$ steps, $x_T \approx \mathcal{N}(0,I)$.

**Closed-form sampling at any timestep:** define $\alpha_t = 1-\beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$:

$$\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)}$$

where $\alpha_t = 1 - \beta_t$ (signal retention at step $t$), $\bar\alpha_t = \prod_{s=1}^t \alpha_s$ (cumulative product — fraction of original signal remaining after $t$ steps), $\sqrt{\bar\alpha_t}$ = how much of the clean image $x_0$ survives, $\sqrt{1-\bar\alpha_t}$ = noise magnitude at step $t$, and $\epsilon \sim \mathcal{N}(0,I)$ = a single standard Gaussian sample (the combined noise from all $t$ steps telescopes into one).

> [!tip] Why the noise telescopes to $\sqrt{1-\bar\alpha_t}$ ([[distributions-gaussian]])
> Each step: $x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$.
> After two steps: $x_2 = \sqrt{\alpha_1\alpha_2}\,x_0 + \underbrace{\sqrt{\alpha_2(1-\alpha_1)}\,\epsilon_1 + \sqrt{1-\alpha_2}\,\epsilon_2}_{\text{two independent Gaussians}}$.
> Two independent Gaussians with variances $a$ and $b$ sum to one Gaussian with variance $a + b$ (closure under addition).
> Variance of the combined noise: $\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar\alpha_2$.
> By induction, after $t$ steps: noise variance $= 1 - \bar\alpha_t$, giving $\epsilon_{\text{combined}} = \sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$.

This allows sampling $x_t$ directly from $x_0$ without simulating all $t$ intermediate steps — critical for efficient training.

### Training Objective (DDPM Simplified Loss)

Train a [[unet|U-Net]] $\epsilon_\theta(x_t, t)$ to predict the noise $\epsilon$ that was added:

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\,x_0,\,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\, t)\|^2\right]}$$

where $\epsilon$ = the noise actually added (ground truth), $\epsilon_\theta(\cdot, t)$ = the neural network's prediction of that noise given the noisy image $x_t$ and timestep $t$, and $\|\cdot\|^2$ = squared L2 norm (mean squared error). The expectation is over random timesteps $t$, data points $x_0$, and noise samples $\epsilon$.

**Algorithm:** (1) sample $x_0$ from data; (2) sample random $t$ and noise $\epsilon$; (3) compute $x_t$; (4) predict $\hat\epsilon = \epsilon_\theta(x_t, t)$; (5) backprop on [[loss-mse|MSE]]. Just supervised regression — no adversarial training, no [[bayesian-inference|posterior]] approximation.

---

## Intermediate

### Noise Schedules

**Linear schedule (DDPM):** $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T-\beta_1)$, with $\beta_1=10^{-4}$, $\beta_T=0.02$. Destroys information rapidly early in the schedule — many easy timesteps near $t=0$ waste training capacity.

**Cosine schedule (Improved DDPM):** defines $\bar\alpha_t = \cos^2\!\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)/\cos^2\!\left(\frac{s}{1+s}\cdot\frac{\pi}{2}\right)$. Decreases signal-to-noise ratio more gradually, distributing learning difficulty evenly → better sample quality.

### Sampling: DDPM vs DDIM

**DDPM reverse step:**
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sqrt{\tilde\beta_t}\,z, \quad z \sim \mathcal{N}(0,I)$$

where $\alpha_t = 1 - \beta_t$, $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ = posterior variance (computed from the schedule), and $z \sim \mathcal{N}(0,I)$ = fresh noise injected at each step (makes the reverse process stochastic). The $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$ coefficient rescales the predicted noise to correct the signal.

Requires 1000 steps — slow.

**DDIM (Song et al., 2020):** deterministic non-Markovian update that uses the same trained $\epsilon_\theta$:
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}\right)}_{\text{"predicted }x_0\text{"}} + \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta$$

50 DDIM steps rival 1000-step DDPM in quality. Key property: DDIM is **invertible** — encode a real image by running the forward DDIM steps; this enables image editing in noise space.

### Conditioning: Classifier-Free Guidance (CFG)

Train the same model jointly on conditioned and unconditioned inputs (10% of training examples, drop the condition $c$ and replace with null token $\emptyset$). At inference, linearly extrapolate:

$$\hat\epsilon = \epsilon_\theta(x_t, \emptyset) + w\!\left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\right)$$

where $\epsilon_\theta(x_t, c)$ = conditional noise prediction (given the text/class condition $c$), $\epsilon_\theta(x_t, \emptyset)$ = unconditional noise prediction (condition replaced with null token), $w$ = guidance scale (how strongly to follow the condition), and $\hat\epsilon$ = the guided noise prediction used for the reverse step. When $w=1$, reverts to pure conditional; when $w=0$, reverts to unconditional.

Guidance scale $w > 1$ amplifies the conditioning signal — trades diversity for fidelity. In practice $w \in [5, 12]$ for text-to-image. Single model, no separate classifier needed.

---

## Advanced

### Score Matching Equivalence

**Score function** of a distribution: $\nabla_x \log p(x)$ — points toward regions of higher probability. **Score matching** trains $s_\theta(x) \approx \nabla_x \log p(x)$.

The connection to diffusion: predicting the noise $\epsilon$ is equivalent to predicting the score of the noisy distribution:

$$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$$

This unification, formalized via Stochastic Differential Equations (Song et al., 2021), expresses the forward process as an SDE $dx = f(x,t)dt + g(t)dW$ and the reverse as a reverse-time SDE. This framework allows using any off-the-shelf numerical SDE/ODE solver for faster sampling — including higher-order solvers that achieve quality comparable to 1000 DDPM steps in 10–25 function evaluations.

### Latent Diffusion Models (LDM / Stable Diffusion)

Running diffusion in pixel space at 512×512 costs $\approx 512^2 \times T$ operations per training step. **LDM** (Rombach et al., 2022) runs diffusion in the **latent space of a pre-trained [[variational-autoencoders|VAE]]**:

1. Train a VAE encoder $\mathcal{E}$ and decoder $\mathcal{D}$ on images.
2. Compress image: $z = \mathcal{E}(x)$, $z \in \mathbb{R}^{64 \times 64 \times 4}$ (8× spatial compression).
3. Train diffusion model entirely in latent space: $\epsilon_\theta(z_t, t, c)$.

Compute cost reduction: $(512/64)^2 = 64\times$ fewer spatial positions. The diffusion model never sees pixels — only the 64×64 latent code. The VAE decoder reconstructs the final image from the denoised latent.

Stable Diffusion's U-Net has 860M parameters and operates on 64×64 latent features. Text conditioning is injected via [[attention-mechanism|cross-attention]] ([[clip|CLIP]] ViT-L/14 text embeddings). This architecture became the standard for all subsequent open-source text-to-image models.

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

## Links

- [[variational-autoencoders]] — VAEs also learn latent representations but use a single-step ELBO; diffusion models use a T-step Markov chain of noising
- [[distributions-gaussian]] — the forward noising process adds Gaussian noise at each step; the reverse process predicts Gaussian noise
- [[loss-mse]] — diffusion training minimizes MSE between predicted and actual noise; the simplified objective discards time-step weighting
- [[normalizing-flows]] — flows compute exact log-likelihood; diffusion models compute a variational lower bound analogous to the VAE ELBO
- [[bayesian-inference]] — the reverse denoising process is a learned posterior; classifier-free guidance is Bayesian conditioning on the class label
- [[unet]] — the standard noise-prediction backbone; skip connections preserve spatial structure across scales at each denoising step
- [[attention-mechanism]] — U-Net variants add self-attention at lower spatial resolutions for long-range coherence in generated images
- [[clip]] — text conditioning in Stable Diffusion is done via CLIP embeddings injected into the U-Net through cross-attention
