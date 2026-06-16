---
title: Diffusion Guidance
tags: [diffusion-guidance, cfg, classifier-free-guidance, negative-prompting, score-matching]
aliases: [classifier-free guidance, CFG, classifier guidance, negative prompting, guidance scale]
difficulty: 2
status: complete
depends_on: [diffusion-models, bayesian-inference]
related: [diffusion-models, variational-autoencoders, contrastive-learning, loss-kl-divergence, clip]
---

# Diffusion Guidance

---

## Fundamental

### The Conditional Generation Problem

Diffusion models learn $p(x)$ (unconditional). For useful generation we want $p(x | c)$ for a condition $c$ (text prompt, class label, image). Guidance methods steer the denoising process toward the condition without retraining a fully conditional model from scratch.

### Classifier Guidance

**Dhariwal & Nichol (2021)** showed that a pre-trained classifier $p_\phi(c | x_t)$ (trained on noisy images at each noise level $t$) can guide diffusion sampling:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t) - \sqrt{1 - \bar{\alpha}_t} \cdot \gamma \nabla_{x_t} \log p_\phi(c | x_t)$$

where $\epsilon_\theta(x_t)$ = unconditional noise prediction (no class label), $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$ = cumulative signal retention at step $t$ (defined in [[diffusion-models]]), $\gamma \geq 0$ = guidance scale (how strongly to follow the class), $\nabla_{x_t} \log p_\phi(c | x_t)$ = gradient of the log-classifier w.r.t. the current noisy image $x_t$ (points in the direction that increases the probability of class $c$), and $\tilde{\epsilon}$ = the guided noise estimate used for the reverse step.

The gradient of the log-classifier pushes the denoised sample toward the class $c$. Scale $\gamma$ controls guidance strength — larger $\gamma$ = more class-consistent but less diverse.

**Score function interpretation:** in score-based diffusion, $\nabla_x \log p(x)$ is the score. Classifier guidance adds a conditioning term: $\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x)$.

**Drawback:** requires a separate noise-robust classifier at every noise level — expensive to train.

---

## Intermediate

### Classifier-Free Guidance (CFG)

**Ho & Salimans (2022)** eliminated the separate classifier by jointly training a conditional and unconditional model:

During training: randomly drop condition $c$ with probability $p_\text{uncond}$ (typically 10–20%), training the model on both $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \varnothing)$ (where $\varnothing$ is a null/empty condition).

At sampling, extrapolate away from the unconditional prediction:

$$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$$

where $\epsilon_\theta(x_t, c)$ = noise prediction conditioned on prompt $c$, $\epsilon_\theta(x_t, \varnothing)$ = unconditional noise prediction (null/empty condition), and $w \geq 1$ = the **guidance scale** (how strongly to follow the condition). This is equivalent to sampling from $p(x|c)^{1+w} / p(x)^w$ — a sharpened conditional distribution.

**Effect of guidance scale $w$:**
- $w = 0$: unconditional generation
- $w = 1$: standard conditional generation
- $w > 1$: over-conditioned — high prompt adherence, reduced diversity
- $w \approx 7$–$15$: typical sweet spot for text-to-image models (Stable Diffusion default: 7.5)

**Only one model needed** — both conditional and unconditional calls use the same weights. This is why CFG became the standard.

### Negative Prompting

**Negative prompts** use CFG with a non-null unconditional condition — instead of $\varnothing$, use a negative text embedding $c_\text{neg}$:

$$\tilde{\epsilon} = \epsilon_\theta(x_t, c_\text{neg}) + w \cdot (\epsilon_\theta(x_t, c_\text{pos}) - \epsilon_\theta(x_t, c_\text{neg}))$$

The model moves away from $c_\text{neg}$ and toward $c_\text{pos}$. Examples: negative prompt "blurry, low quality, extra fingers" steers away from common failure modes.

This is not removing content from the scene but rather using the model's learned negative space to guide the direction of the conditional score.

---

## Advanced

### Guidance Scale and Quality-Diversity Tradeoff

High guidance scale $w$ increases **Fréchet Inception Distance (FID)** on diversity metrics but improves **CLIP score** (prompt adherence). This is the fundamental tradeoff: more conditioning = less randomness = less diversity.

**Guidance Distillation (Meng et al., LCM):** consistency models and flow matching approaches bake guidance into the model at distillation time, reducing inference steps from 50 to 4 while maintaining quality.

### Variants of CFG

**Perturbed-Attention Guidance (PAG):** instead of conditioning on null, degrade the attention map (self-attention with identity) as the unconditional. Improves structural coherence.

**Auto-guidance / Self-guidance:** use an intermediate-quality model output (early denoising steps) as the unconditional reference instead of a null embedding.

**Semantic guidance (SEGA):** decompose guidance into multiple semantic directions simultaneously — separately guide for multiple attributes (style, content, composition) in one sampling trajectory.

### CFG in Video and 3D

CFG extends naturally to video diffusion (condition = text + first frame, null = text only), 3D generation (SDS — Score Distillation Sampling in DreamFusion uses CFG guidance to optimize a NeRF), and audio generation (AudioLDM, MusicGen).

## Links

- [[diffusion-models]] — guidance modifies the diffusion reverse process; classifier-free guidance scales the difference between conditional and unconditional noise predictions
- [[bayesian-inference]] — classifier-free guidance is implicit Bayesian conditioning: scaling the score function is equivalent to raising the posterior temperature
- [[clip]] — CLIP embeddings are the text condition in Stable Diffusion; CFG scale controls how strongly the generated image follows the CLIP text embedding
- [[variational-autoencoders]] — latent diffusion models (Stable Diffusion) run the diffusion process in VAE latent space; guidance operates in that latent space
- [[contrastive-learning]] — CLIP's contrastive pretraining creates the text-image alignment that makes text-to-image guidance possible
