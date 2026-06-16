---
title: Variational Autoencoders (VAEs)
tags: [generative-models, latent-space, bayesian, deep-learning, elbo, reparameterization, beta-vae]
aliases: [VAE, ELBO, variational inference, reparameterization trick, beta-VAE, evidence lower bound]
difficulty: 2
status: complete
depends_on: [bayesian-inference, distributions-gaussian, loss-kl-divergence]
related: [bayesian-inference, diffusion-models, normalizing-flows, backpropagation, distributions-gaussian]
---

# Variational Autoencoders (VAEs)

---

## Fundamental

We want to model a complex data distribution $p(x)$ (e.g., natural images) and generate new samples from it. Directly parameterizing $p(x)$ over millions of pixel dimensions is intractable. The solution: posit a **low-dimensional latent variable** $z$ that generates $x$:

$$z \sim p(z) = \mathcal{N}(0, I), \quad x \sim p_\theta(x \mid z)$$

The marginal $p_\theta(x) = \int p_\theta(x \mid z) p(z)\,dz$ can be complex even with a simple prior — the decoder neural network creates the complexity.

**Why vanilla autoencoders fail for generation:** a deterministic encoder maps each $x$ to a single point in $z$-space. The latent space has no structure — large empty regions produce garbage when decoded. There is no probability model, so you cannot sample from it.

### The ELBO

We want to maximize $\log p_\theta(x)$, but the integral over $z$ is intractable. Introduce an approximate [[bayesian-inference|posterior]] $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x) I)$. By [[math-convexity-jensen|Jensen's inequality]] ($\log$ is concave):

> [!tip] Which direction Jensen's inequality points here ([[math-convexity-jensen]])
> For any concave function $f$ (like $\log$) and random variable $X$: $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$.
> Rearranging with $f = \log$ and $X = p(x,z)/q_\phi(z|x)$:
> $\log \mathbb{E}_q[X] \geq \mathbb{E}_q[\log X]$
> So moving $\log$ inside the expectation gives a **lower bound** — that lower bound is the ELBO.
> The gap equals $D_{KL}(q_\phi(z|x) \,\|\, p_\theta(z|x)) \geq 0$, so the ELBO is tight exactly when $q$ perfectly approximates the true posterior.

$$\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z \mid x)}\!\left[\frac{p_\theta(x \mid z)\,p(z)}{q_\phi(z \mid x)}\right] \geq \mathbb{E}_{q_\phi}\!\left[\log \frac{p_\theta(x \mid z)\,p(z)}{q_\phi(z \mid x)}\right]$$

Expanding:

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z \mid x) \| p(z))}_{\text{regularization}}$$

This is the **ELBO** (Evidence Lower BOund). The gap to the true log-likelihood is exactly $D_{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x)) \geq 0$. Maximizing the ELBO simultaneously maximizes the likelihood lower bound and minimizes the approximation error.

---

## Intermediate

### The Two ELBO Terms

**Reconstruction term:** $\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]$ — maximize log-likelihood of the input under the decoded distribution. For [[distributions-gaussian|Gaussian]] decoder: equivalent to [[loss-mse|MSE loss]]. For Bernoulli decoder (binary pixels): [[loss-cross-entropy|binary cross-entropy]].

**[[loss-kl-divergence|KL divergence]] term:** $D_{KL}(q_\phi(z \mid x) \| p(z))$ — push the encoder's posterior toward the standard Gaussian prior. For diagonal Gaussian:

$$D_{KL}(\mathcal{N}(\mu, \sigma^2 I) \| \mathcal{N}(0,I)) = \frac{1}{2}\sum_{j=1}^d (\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$$

where $d$ = dimensionality of the latent space $z$, $j$ = index over latent dimensions, $\mu_j$ = mean of the encoder's output distribution for dimension $j$ (how far the latent is from the prior mean 0), and $\sigma_j^2$ = variance for dimension $j$ (the $-\log\sigma_j^2$ and $\sigma_j^2$ terms together penalize distributions that are too narrow or too wide relative to the unit Gaussian prior).

This has a **closed-form expression** — no sampling needed for the KL term, only for the reconstruction term.

**Tension between the two terms:**
- Reconstruction only → vanilla autoencoder (unstructured $z$-space, can't generate)
- KL only → encoder ignores input, always outputs $\mathcal{N}(0,I)$ (generates but doesn't reconstruct)
- Both balanced → encoder maps similar inputs to nearby, overlapping Gaussian regions with $\mathcal{N}(0,I)$ aggregate posterior

### The Reparameterization Trick

To [[backpropagation|backpropagate]] through the sampling step $z \sim q_\phi(z \mid x) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$, rewrite as:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Now $z$ is a **deterministic function** of $\phi$ and a noise variable $\epsilon$ that is independent of $\phi$. Gradients flow through $\mu_\phi$ and $\sigma_\phi$ normally. Without this trick, backpropagation stops at the sampling step.

**Training algorithm:**
```
For each batch x:
  1. Encode: (μ, log σ²) = Encoder_φ(x)
  2. Sample: ε ~ N(0, I)
  3. z = μ + exp(log σ²/2) * ε        ← differentiable w.r.t. μ, σ
  4. x̂ = Decoder_θ(z)
  5. L = recon_loss(x, x̂) + β * KL(μ, σ²)
  6. Backpropagate through steps 4→3→1
  7. Update θ and φ
```

---

## Advanced

### $\beta$-VAE and Disentanglement

Increase the weight on the KL term: $\mathcal{L}_\beta = \mathbb{E}[\log p_\theta(x \mid z)] - \beta D_{KL}(q_\phi(z \mid x) \| p(z))$ for $\beta > 1$.

Higher $\beta$ **forces stronger disentanglement**: each dimension of $z$ is pushed harder toward $\mathcal{N}(0,1)$, making dimensions statistically independent. Empirically, at $\beta \approx 4$–10, the latent dimensions specialize — one dimension controls face pose, another controls lighting, another controls smile, etc. — without any labels.

**Tradeoff:** higher $\beta$ → worse reconstruction quality (more information is discarded to satisfy the stricter KL constraint). The optimal $\beta$ balances disentanglement with reconstruction fidelity for the downstream task.

### ELBO Gap and Posterior Collapse

The ELBO is tight when $q_\phi(z \mid x) = p_\theta(z \mid x)$. In practice, the encoder is too simple to perfectly approximate the true posterior. The residual gap manifests as a lower bound that can be far from the true likelihood.

**Posterior collapse:** a pathological training outcome where the encoder ignores the input and outputs $q_\phi(z \mid x) \approx p(z) = \mathcal{N}(0,I)$ for all $x$. Causes: the KL term dominates; the decoder is expressive enough to model $p(x)$ without the latent code. Common in VAEs for text generation (decoder is an [[rnn-lstm|RNN]] that learns its own context).

**Fixes:** KL annealing (start with $\beta=0$, gradually increase to 1 during training); Free bits ($\beta$-VAE variant with a minimum target KL per dimension to prevent full collapse); using a [[normalizing-flows|flow-based]] posterior to match the true posterior more closely.

### VAE as Hierarchical VAE / Connection to Diffusion

The [[diffusion-models|DDPM]] forward process can be viewed as a **hierarchical VAE** with a fixed encoder:
- $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ — fixed (not learned)
- $p_\theta(x_{t-1} \mid x_t)$ — learned decoder (the denoising network)

The DDPM ELBO is the sum of $T$ KL terms $D_{KL}(q(x_{t-1} \mid x_t, x_0) \| p_\theta(x_{t-1} \mid x_t))$ — exactly the latent hierarchy structure of a hierarchical VAE, but with $T=1000$ levels and a fixed (non-learned) encoder. This connection illuminates why diffusion works: the fixed forward process avoids posterior collapse while the deep denoiser handles the complex reverse mapping.

---

## Links

- [[bayesian-inference]] — VAEs are the deep learning instantiation of Bayesian inference; the encoder posterior $q(z|x)$ approximates the true posterior $p(z|x)$
- [[distributions-gaussian]] — the prior $p(z) = \mathcal{N}(0,I)$ and the reparameterization trick both rely on Gaussian properties
- [[loss-kl-divergence]] — the ELBO contains a KL term that regularizes the latent space toward the prior
- [[loss-cross-entropy]] — the reconstruction term is cross-entropy (or MSE) measuring how well the decoder recovers the input
- [[normalizing-flows]] — flows achieve exact log-likelihood without the ELBO approximation; complementary approach to latent-variable generation
- [[diffusion-models]] — diffusion models supersede VAEs for image quality while sharing the latent-variable framing
- [[math-convexity-jensen]] — Jensen's inequality is the key step in deriving the ELBO lower bound on log-likelihood
- [[backpropagation]] — the reparameterization trick makes the stochastic latent variable differentiable, enabling gradient-based training
