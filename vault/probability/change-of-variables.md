---
title: Change of Variables & Probability Transformations
tags: [probability, change-of-variables, jacobian, normalizing-flows, transformation]
aliases: [change of variables, transformation of random variables, probability density transformation]
difficulty: 2
status: complete
related: [normalizing-flows, variational-autoencoders, distributions-gaussian, matrix-calculus, statistical-inference-mle, sampling-methods, taylor-series]
---

# Change of Variables & Probability Transformations

---

## Fundamental

When you apply a function to a random variable, it changes the distribution — but probability must still add up to 1, so the density must stretch or compress to compensate. The **change of variables formula** tells you exactly how the density transforms: you multiply by the Jacobian, which measures how much the function stretches space. This is the foundation of normalizing flows (which turn simple distributions into complex ones by stacking transformations) and appears whenever you reparameterize a model.

### Scalar Change of Variables

Given a random variable $X$ with density $p_X(x)$ and a **monotone differentiable** transformation $Y = g(X)$, the density of $Y$ is:

$$p_Y(y) = p_X(g^{-1}(y)) \cdot \left|\frac{dg^{-1}}{dy}\right| = p_X(x) \cdot \left|\frac{dx}{dy}\right|$$

**Intuition:** probability must be conserved — the probability in the interval $(x, x+dx)$ must equal the probability in $(y, y+dy)$. Since $p_X(x)|dx| = p_Y(y)|dy|$, we have $p_Y(y) = p_X(x) / |dy/dx| = p_X(x) \cdot |dx/dy|$.

**Worked example:** $X \sim \text{Uniform}(0, 1)$, $Y = -\log(X)$.

Inverse: $x = e^{-y}$, derivative: $dx/dy = -e^{-y}$, absolute value: $e^{-y}$.

$$p_Y(y) = p_X(e^{-y}) \cdot e^{-y} = 1 \cdot e^{-y} = e^{-y} \quad (y > 0)$$

This is the $\text{Exponential}(1)$ distribution. **Practical use:** since uniform samples are easy to generate, this gives a simple way to sample from the exponential distribution: generate $u \sim U(0,1)$, output $-\log(u)$. This is the **inverse CDF method** (or inversion sampling).

**The CDF method generalized:** for any distribution with invertible CDF $F_X$: if $U \sim \text{Uniform}(0,1)$, then $X = F_X^{-1}(U)$ has distribution $F_X$. The change of variables proof: $P(X \leq x) = P(F_X^{-1}(U) \leq x) = P(U \leq F_X(x)) = F_X(x)$.

### Multivariate Change of Variables

For a vector transformation $\mathbf{y} = g(\mathbf{x})$ where $g : \mathbb{R}^d \to \mathbb{R}^d$ is differentiable and bijective:

$$p_Y(\mathbf{y}) = p_X(g^{-1}(\mathbf{y})) \cdot \left|\det J_{g^{-1}}(\mathbf{y})\right|$$

where $J_{g^{-1}}$ is the **Jacobian matrix** of the inverse transformation:

$$J_{g^{-1}}(\mathbf{y}) = \frac{\partial g^{-1}(\mathbf{y})}{\partial \mathbf{y}} = \begin{pmatrix} \partial x_1/\partial y_1 & \cdots & \partial x_1/\partial y_d \\ \vdots & & \vdots \\ \partial x_d/\partial y_1 & \cdots & \partial x_d/\partial y_d \end{pmatrix}$$

**Equivalently** (expressing in terms of the forward Jacobian):

$$p_Y(\mathbf{y}) = p_X(\mathbf{x}) \cdot \left|\det J_g(\mathbf{x})\right|^{-1}$$

The determinant $|\det J_g|$ is the "volume scaling factor" — how much the transformation $g$ stretches or compresses an infinitesimal volume at $\mathbf{x}$.

---

## Intermediate

### The Log-Probability Form and Normalizing Flows

Taking the log of the change-of-variables formula:

$$\log p_Y(\mathbf{y}) = \log p_X(g^{-1}(\mathbf{y})) + \log\left|\det J_{g^{-1}}(\mathbf{y})\right|$$

This is the fundamental equation of **normalizing flows** (see [[normalizing-flows]]). A normalizing flow is a composition of invertible transformations $f = f_K \circ \cdots \circ f_1$ with tractable Jacobian determinants:

$$\log p_Y(\mathbf{y}) = \log p_X(\mathbf{x}) + \sum_{k=1}^K \log\left|\det J_{f_k^{-1}}\right|$$

The goal: start with a simple base distribution $p_X$ (e.g., standard Gaussian), apply a sequence of learnable bijections, and arrive at a complex distribution $p_Y$ that matches the data. The Jacobian determinant terms are the "price" of the transformation — the correction for the volume change.

### Computing the Jacobian Determinant

For general $d \times d$ Jacobians, computing $\det J$ costs $O(d^3)$ — prohibitive for high-dimensional data ($d = 10^4$ for images). Flow architectures are designed around making $\det J$ cheap to compute:

**Triangular Jacobians:** if $g$ has a triangular Jacobian (lower or upper), $\det J = \prod_i J_{ii}$ — the product of diagonal elements. Autoregressive flows (MADE, PixelCNN) achieve this: each output $y_i$ depends only on inputs $x_1, \ldots, x_{i-1}$, making $J$ strictly lower triangular. Determinant computation: $O(d)$ instead of $O(d^3)$.

**Coupling layers (RealNVP, Glow):** split $\mathbf{x}$ into two halves; transform one half using a function of the other:

$$\mathbf{y}_{1:d/2} = \mathbf{x}_{1:d/2}$$
$$\mathbf{y}_{d/2+1:d} = \mathbf{x}_{d/2+1:d} \odot \exp(s(\mathbf{x}_{1:d/2})) + t(\mathbf{x}_{1:d/2})$$

The Jacobian is block triangular (identity in upper-left, something in lower-right), and the determinant is $\prod \exp(s_i) = \exp(\sum s_i)$ — computable in $O(d)$.

**Residual flows:** $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ where $F$ is a contraction ($\text{Lip}(F) < 1$). The Jacobian is $I + J_F$; its determinant can be estimated using the infinite series $\log\det(I + J_F) = \text{tr}(J_F) - \text{tr}(J_F^2)/2 + \cdots$ (Skilling, 1989), truncated with Hutchinson's estimator.

### Gaussian Transformations and Linear Algebra

The most common change of variables in ML: linear transformations of Gaussians.

If $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \Sigma_x)$ and $\mathbf{y} = A\mathbf{x} + \mathbf{b}$:

$$\mathbf{y} \sim \mathcal{N}(A\boldsymbol{\mu}_x + \mathbf{b},\; A\Sigma_x A^\top)$$

The Jacobian is $|det(A^{-1})| = 1/|\det A|$ — for orthogonal $A$ (rotations), $|\det A| = 1$, so rotations preserve the density entirely.

**Reparameterization trick (VAE):** sample $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ by: sample $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$, then $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$. This is a simple linear change of variables that makes sampling differentiable: $\partial \mathbf{z}/\partial \boldsymbol{\mu} = I$ and $\partial \mathbf{z}/\partial \boldsymbol{\sigma} = \text{diag}(\boldsymbol{\epsilon})$. The Jacobian determinant is $\prod \sigma_i$ (diagonal Jacobian), but for the VAE it's not needed since we're differentiating the sample path, not transforming a density.

### Moment Computation via Transformations

**The delta method** (also covered in [[central-limit-theorem]]): for a nonlinear function $g(X)$, apply first-order Taylor expansion around $\mu = \mathbb{E}[X]$:

$$g(X) \approx g(\mu) + g'(\mu)(X - \mu)$$

$$\text{Var}(g(X)) \approx [g'(\mu)]^2 \cdot \text{Var}(X)$$

For the multivariate case: $\text{Var}(g(\mathbf{X})) \approx \nabla g(\boldsymbol{\mu})^\top \Sigma \nabla g(\boldsymbol{\mu})$.

This is how you propagate uncertainty through a differentiable function — a linearized version of the change-of-variables formula. Used in Kalman filters, EKF (Extended Kalman Filter), Laplace approximations, and uncertainty quantification for neural networks.

---

## Advanced

### Change of Variables in Variational Inference

The ELBO objective in [[variational-autoencoders|VAEs]] requires computing expectations under $q_\phi(\mathbf{z}|\mathbf{x})$:

$$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The challenge: how to take gradients through the expectation $\mathbb{E}_{q_\phi}[\cdot]$ with respect to $\phi$ (the distribution parameters)?

**REINFORCE (log derivative trick):**
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{q_\phi}[f(\mathbf{z})\nabla_\phi \log q_\phi(\mathbf{z})]$$

High variance — many samples needed.

**Reparameterization trick (change of variables approach):**
Express $\mathbf{z} = g_\phi(\boldsymbol{\epsilon})$ where $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ (independent of $\phi$). Then:
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})}[\nabla_\phi f(g_\phi(\boldsymbol{\epsilon}))]$$

Gradients flow through the transformation $g_\phi$, not through the sampling operation. For $q_\phi = \mathcal{N}(\mu, \sigma^2)$: $g_\phi(\epsilon) = \mu + \sigma\epsilon$, $\nabla_\mu g_\phi = 1$, $\nabla_\sigma g_\phi = \epsilon$. The reparameterization is a direct application of the change-of-variables theorem for differentiating through a probability transformation.

### Diffusion Models as Hierarchical Change of Variables

A diffusion model (see [[diffusion-models]]) applies $T$ noise-adding steps:
$$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t, \qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(0, I)$$

Each step is a linear change of variables on $(\mathbf{x}_{t-1}, \boldsymbol{\epsilon}_t)$ to produce $\mathbf{x}_t$. By the change-of-variables formula applied repeatedly:

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar\alpha_t}\mathbf{x}_0,\; (1-\bar\alpha_t)I), \qquad \bar\alpha_t = \prod_{s=1}^t \alpha_s$$

This closed form for the forward process is possible precisely because Gaussian transformations compose analytically. The denoising score matching loss is then a change-of-variables problem in the reverse direction: the reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ is Gaussian when $p(\mathbf{x}_0)$ is Gaussian or when the step size is small, following directly from the Bayesian change-of-variables formula for conditional densities.

### Score Functions and the Stein Identity

The **score function** of a distribution is $\nabla_\mathbf{x} \log p(\mathbf{x})$. The Stein identity states that for any smooth $f$ with $\mathbb{E}[f(x)] < \infty$:

$$\mathbb{E}_p[\nabla_\mathbf{x} \log p(\mathbf{x}) \cdot f(\mathbf{x}) + \nabla_\mathbf{x} f(\mathbf{x})] = 0$$

This follows from integration by parts (a form of change of variables in the continuous setting):
$$\int p(x) \nabla_x \log p(x) f(x) dx = \int \nabla_x p(x) f(x) dx = -\int p(x) \nabla_x f(x) dx$$

**Applications in ML:** the score function is directly learned in score-based generative models (diffusion models, NCSN). Stein variational gradient descent (SVGD) uses the Stein identity to derive particle updates that transform a set of particles to match a target distribution, without requiring the change-of-variables Jacobian determinant. The Fisher information matrix (see [[fisher-information]]) is $I(\theta) = \mathbb{E}[\nabla_\theta \log p \cdot (\nabla_\theta \log p)^\top]$ — the covariance of the score function.

---

*See also: [[normalizing-flows]] · [[variational-autoencoders]] · [[diffusion-models]] · [[distributions-gaussian]] · [[sampling-methods]] · [[fisher-information]] · [[central-limit-theorem]]*
