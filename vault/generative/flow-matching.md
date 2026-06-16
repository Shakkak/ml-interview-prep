---
title: Flow Matching
tags: [flow-matching, continuous-normalizing-flows, ode, rectified-flow, stochastic-interpolants]
aliases: [flow matching, continuous normalizing flows, CNF, rectified flow, stochastic interpolants, FM]
difficulty: 3
status: complete
depends_on: [normalizing-flows, diffusion-models, distributions-gaussian]
related: [normalizing-flows, diffusion-models, variational-autoencoders, sampling-methods, change-of-variables]
---

# Flow Matching

---

## Fundamental

### Continuous Normalizing Flows (CNFs)

A **Continuous Normalizing Flow** defines a time-dependent ODE that transforms a simple distribution $p_0$ (e.g., Gaussian) into a complex data distribution $p_1$:

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) \sim p_0, \quad x(1) \sim p_1$$

The vector field $v_\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is parameterized by a neural network. Sampling: integrate the ODE forward from $x \sim p_0$ using an ODE solver (Euler, Runge-Kutta).

**Classical CNFs (FFJORD, Chen et al., 2018):** train by maximizing likelihood using the instantaneous change of variables formula. Requires solving the ODE twice (forward and backward) and computing a trace of the Jacobian — expensive.

### Flow Matching Objective

**Flow Matching (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022)** sidesteps likelihood training by directly regressing onto a **target vector field**:

$$\mathcal{L}_\text{FM} = \mathbb{E}_{t, p_t(x)}\!\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

where $t \in [0,1]$ = continuous time variable, $p_t(x)$ = marginal distribution of the flow at time $t$ (interpolates from $p_0$ to $p_1$), $v_\theta(x, t) \in \mathbb{R}^d$ = the network's predicted vector field (direction and speed to move $x$ at time $t$), and $u_t(x)$ = the target vector field that generates $p_t$ (the interpolant distribution at time $t$). The loss avoids computing the trace of the Jacobian entirely — just MSE on the vector field.

**Key insight:** instead of optimizing the marginal vector field directly (hard), condition on individual sample pairs $(x_0, x_1)$ from $p_0 \times p_1$ and regress on the **conditional** vector field (easy):

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t, x_0, x_1}\!\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$$

where $x_0 \sim p_0$ = source noise sample, $x_1 \sim p_1$ = target data sample, $x_t = (1-t)x_0 + tx_1$ = linear interpolant between the pair at time $t$, and $(x_1 - x_0)$ = constant velocity target (the direction the model should predict to move from $x_0$ to $x_1$ in one unit of time).

---

## Intermediate

### Rectified Flow

**Rectified Flow (Liu et al., 2022):** a special case of flow matching using:
- Source: $x_0 \sim \mathcal{N}(0, I)$
- Target: $x_1 \sim p_\text{data}$
- Interpolant: $x_t = (1-t)x_0 + tx_1$ (straight line)
- Target field: $v_t = x_1 - x_0$ (constant, pointing from noise to data)

The model learns to follow straight-line trajectories from noise to data. **Why straight lines?** Straight paths are shortest geodesics in flat space — fewer ODE steps needed to integrate accurately. Diffusion (DDPM) uses curved paths; rectified flow uses straight paths.

**Reflow:** after training, sample (noise, generated image) pairs from the trained model, then retrain on these to make paths even straighter. After 2–3 reflow iterations, the model converges well enough for **1-step sampling**.

### Comparison to Diffusion

| Property | Diffusion (DDPM) | Flow Matching (FM) |
|----------|-----------------|-------------------|
| Path type | Curved (stochastic) | Straight (or custom) |
| Training target | Noise prediction / score | Vector field |
| Steps needed | 20–1000 | 1–20 |
| FID quality | Excellent | Comparable |
| Theoretical framework | SDE | ODE |
| Flexible path design | Via noise schedule | Direct interpolant design |

---

## Advanced

### Stochastic Interpolants

**Stochastic interpolants (Albergo et al., 2023)** generalize flow matching to any interpolant between $p_0$ and $p_1$, including stochastic interpolants:

$$x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Different choices of $(\alpha, \beta, \gamma)$ recover: DDPM ($\alpha = \sqrt{1-\sigma_t^2}$, $\beta = 0$, $\gamma = \sigma_t$), rectified flow ($\alpha = 1-t$, $\beta = t$, $\gamma = 0$), and new interpolants.

**Mini-batch optimal transport (OT-CFM):** instead of pairing $x_0$ and $x_1$ randomly, use optimal transport to pair them (minimize expected transport cost). Straighter paths → fewer ODE steps → better 1-step generation.

### Practical Adoption

Flow matching (under various names) has rapidly replaced diffusion in state-of-the-art models:
- **Stable Diffusion 3 (2024):** uses rectified flow matching
- **Flux (Black Forest Labs):** uses flow matching
- **Meta Voicebox:** uses flow matching for audio generation
- **SiT (Scalable Interpolant Transformers):** ViT-based flow matching for image generation

## Links

- [[normalizing-flows]] — normalizing flows use discrete bijective transformations; flow matching learns continuous vector fields that generate the same transport
- [[diffusion-models]] — diffusion models add and then remove noise along a fixed Markov process; flow matching learns a direct straight-line transport (rectified flow) that is faster at inference
- [[distributions-gaussian]] — the source distribution for flow matching is typically $\mathcal{N}(0,I)$; the learned vector field transports it to the data distribution
- [[sampling-methods]] — flow matching samples by integrating the learned ODE from $t=0$ to $t=1$; fewer integration steps = faster inference, trading accuracy for speed
- [[change-of-variables]] — the change-of-variables formula for continuous-time flows involves the instantaneous change-of-variables via the trace of the Jacobian
