---
title: Energy-Based Models
tags: [energy-based-models, ebm, score-matching, contrastive-divergence, unnormalized-density]
aliases: [EBM, energy based model, score matching, contrastive divergence, Langevin dynamics]
difficulty: 3
status: complete
depends_on: [statistical-inference-mle, entropy-mutual-info, distributions-overview]
related: [diffusion-models, sampling-methods, variational-inference, normalizing-flows, generative-adversarial-networks]
---

# Energy-Based Models

---

## Fundamental

### The Energy Function

An **Energy-Based Model (EBM)** defines a probability distribution via an energy function $E_\theta: \mathcal{X} \to \mathbb{R}$:

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad Z_\theta = \int \exp(-E_\theta(x)) \, dx$$

Low energy = high probability. The model is flexible — any neural network can define $E_\theta$. The challenge is the **partition function** $Z_\theta$: a normalizing constant that requires integrating over all $x$, which is intractable for high-dimensional data.

**Comparison to other generative models:**
- Flow/VAE/GAN: require architectural constraints to be tractable
- EBM: no architectural constraints — any function works; but $Z_\theta$ is intractable

---

## Intermediate

### Learning EBMs: Contrastive Divergence

Maximum likelihood training: $\max_\theta \mathbb{E}_{x \sim p_\text{data}}[\log p_\theta(x)] = -\mathbb{E}_{x \sim p_\text{data}}[E_\theta(x)] - \log Z_\theta$

The gradient of $\log Z_\theta$ requires computing $\mathbb{E}_{p_\theta}[\nabla_\theta E_\theta(x)]$ — an expectation under the model, estimated by sampling.

**Contrastive Divergence (CD, Hinton 2002):** instead of fully converged MCMC samples, run only $k$ steps of MCMC (Gibbs or MCMC) from data points, then use those short-chain samples as "negative" examples:

$$\nabla_\theta \mathcal{L}_\text{CD} \approx \mathbb{E}_{x^+ \sim p_\text{data}}[\nabla_\theta E_\theta(x^+)] - \mathbb{E}_{x^- \sim \text{MCMC}_k}[\nabla_\theta E_\theta(x^-)]$$

**Positive phase:** lower energy of data points (make them more probable).
**Negative phase:** raise energy of model samples (make everything else less probable).

CD-1 (1 MCMC step) is cheap but biased. CD-$k$ for larger $k$ is less biased but more expensive.

### Score Matching

**Score matching (Hyvärinen, 2005)** avoids $Z_\theta$ entirely by matching the **score function** $\nabla_x \log p(x)$:

$$\mathcal{L}_\text{SM} = \mathbb{E}_{p_\text{data}}\!\left[\sum_i \partial_i s_\theta(x)_i + \frac{1}{2}\|s_\theta(x)\|^2\right]$$

where $s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$. The partition function cancels in the score!

**Denoising Score Matching (DSM):** easier to compute — add noise to data, train score to point toward clean data:
$$\mathcal{L}_\text{DSM} = \mathbb{E}_{x, \tilde{x}}\left[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x)\|^2\right]$$

where $x$ = clean data point from $p_\text{data}$, $\tilde{x} = x + \sigma\eta$ = noisy version of $x$ (Gaussian corruption), $s_\theta(\tilde{x}) = -\nabla_{\tilde{x}} E_\theta(\tilde{x})$ = predicted score (points toward clean $x$), and $\nabla_{\tilde{x}} \log p(\tilde{x}|x) = -({\tilde{x} - x})/{\sigma^2}$ = the analytical target score (points directly toward $x$, computable without $Z$).

DSM connects directly to **diffusion models** — the diffusion model's denoising objective is exactly DSM at multiple noise levels.

---

## Advanced

### Langevin Dynamics for Sampling

Given a trained EBM (or score function), generate samples via **Langevin Monte Carlo**:

$$x_{t+1} = x_t - \frac{\epsilon}{2} \nabla_x E_\theta(x_t) + \sqrt{\epsilon} \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)$$

where $\epsilon > 0$ = step size (both the gradient coefficient $\epsilon/2$ and noise scale $\sqrt\epsilon$ depend on it — this relationship ensures the chain converges to $p_\theta$ as $\epsilon \to 0$), $\nabla_x E_\theta(x_t)$ = gradient of the energy function (points toward higher energy, so negated by the minus sign to move toward lower energy), and $\eta_t \sim \mathcal{N}(0,I)$ = fresh isotropic Gaussian noise at each step (prevents collapse to a single mode).

The gradient term pushes toward low-energy (high-probability) regions; noise ensures exploration. As $\epsilon \to 0$ and $t \to \infty$, this converges to $p_\theta(x)$.

**MCMC for EBMs:** Langevin dynamics is used as the negative-phase sampler in contrastive divergence and in SGLD (Stochastic Gradient Langevin Dynamics).

### Connection to Diffusion Models

Diffusion models (see [[diffusion-models]]) can be viewed as EBMs at multiple noise levels. The score function $s(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ is the noisy EBM score at noise level $t$. Diffusion training (DDPM) is equivalent to multi-scale DSM. Sampling via reverse diffusion is Langevin dynamics with time-varying noise schedule.

### Joint Energy-Based Models (JEM)

**JEM (Grathwohl et al., 2020):** reinterpret a standard classifier $f_\theta(x) \in \mathbb{R}^K$ as an EBM for $p(x)$:

$$p_\theta(x) = \frac{\exp(\text{LogSumExp}(f_\theta(x)))}{Z_\theta}$$

The same network is simultaneously a classifier (via $p(y|x) = \text{softmax}(f_\theta(x))_y$) and a generative model. Training with both classification loss and contrastive divergence yields better calibration and out-of-distribution detection.

## Links

- [[statistical-inference-mle]] — training EBMs via MLE requires computing the partition function $Z$, which is intractable; contrastive divergence approximates the MLE gradient
- [[entropy-mutual-info]] — the free energy $F = E - TS$ connects EBMs to thermodynamics; entropy plays the role of temperature-scaled regularization
- [[distributions-overview]] — EBMs define unnormalized densities $p(x) \propto e^{-E(x)}$; the partition function is the normalizing constant that makes this a valid distribution
- [[diffusion-models]] — diffusion models can be viewed as EBMs with a score function $\nabla_x \log p(x)$; score matching connects the two frameworks
- [[sampling-methods]] — Langevin dynamics (SGLD) samples from EBMs by following the score function with noise; Markov chain convergence requires careful step size tuning
- [[variational-inference]] — ELBO-based training of EBMs uses a variational approximation to the partition function; VAEs can be interpreted as EBMs with learned samplers
- [[normalizing-flows]] — flows compute exact log-likelihoods; EBMs learn unnormalized densities; they are complementary approaches to density modeling
