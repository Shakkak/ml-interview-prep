---
title: Information Geometry
tags: [information-geometry, fisher-metric, natural-gradient, riemannian, statistical-manifold]
aliases: [statistical manifold, Fisher-Rao metric, natural gradient geometry, Amari geometry]
difficulty: 3
status: complete
related: [fisher-information, exponential-family, variational-inference, statistical-inference-mle, matrix-calculus]
---

# Information Geometry

---

## Fundamental

Ordinary calculus measures distances in Euclidean space. When your objects are probability distributions rather than vectors, the natural notion of distance is not Euclidean — it depends on how different two distributions are statistically. **Information geometry** formalizes this by treating a family of distributions as a curved space (manifold) where distances are measured by Fisher information. The payoff: it explains why natural gradient descent converges faster, and gives a geometric interpretation of KL divergence and the ELBO.

### Statistical Manifolds

**Information geometry** treats families of probability distributions as **Riemannian manifolds**, where each point is a distribution $p_\theta$ and distances are measured by the Fisher information metric.

For a parametric family $\{p_\theta : \theta \in \Theta \subseteq \mathbb{R}^k\}$, the **Fisher information matrix** defines a Riemannian metric on $\Theta$:

$$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[\frac{\partial \log p_\theta}{\partial \theta_i} \frac{\partial \log p_\theta}{\partial \theta_j}\right]$$

This metric captures the local geometry of the distribution family — how quickly distributions change as parameters change.

**Invariance:** the Fisher metric is the unique (up to scaling) Riemannian metric on the space of distributions that is invariant under reparameterization. Distances and volumes are intrinsic to the family of distributions, not to the particular parameterization.

---

## Intermediate

### The Natural Gradient

Ordinary gradient descent moves in parameter space: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. But parameter space is not uniform — a unit step in $\theta$ can mean very different changes to the distribution depending on the local geometry.

**Natural gradient** (Amari, 1998) moves in distribution space by preconditioning with the inverse Fisher metric:

$$\tilde{\nabla} \mathcal{L} = G(\theta)^{-1} \nabla_\theta \mathcal{L}$$

$$\theta \leftarrow \theta - \eta \tilde{\nabla} \mathcal{L}$$

This is the steepest descent in the KL-divergence sense: the natural gradient update minimizes $\mathcal{L}(\theta + d\theta)$ subject to $\text{KL}(p_\theta \| p_{\theta + d\theta}) \leq \epsilon$.

**Properties:**
- Invariant to reparameterization — the update is the same regardless of how $\theta$ is parameterized
- Converges in $O(1)$ steps for exponential families (vs $O(\kappa)$ for gradient descent where $\kappa$ is the condition number)
- Computing $G^{-1}$ exactly is $O(k^3)$ — impractical for large networks

### Connections to Exponential Families

For exponential families (see [[exponential-family]]), the Fisher metric is the Hessian of the log-partition function:

$$G(\eta) = \nabla^2_\eta A(\eta) = \text{Cov}[T(x)]$$

Natural gradient updates for exponential families reduce to Newton's method in natural parameters — explaining why Newton's method is geometrically natural for these models.

---

## Advanced

### Dual Connections and Divergences

Information geometry reveals that the KL divergence is not symmetric, and the asymmetry has geometric meaning:

- **$e$-connection** ($\alpha = +1$): exponential family mixture geodesics; forward KL
- **$m$-connection** ($\alpha = -1$): moment-matching geodesics; reverse KL
- At $\alpha = 0$: Riemannian (Levi-Civita) connection

**Pythagorean theorem for KL:** for exponential families, the KL divergence satisfies an orthogonality relation analogous to the Pythagorean theorem — the $m$-projection (moment-matching) and $e$-projection (KL minimization) are "dual" projections that satisfy $\text{KL}(p\|r) = \text{KL}(p\|q) + \text{KL}(q\|r)$ when $q$ is the $e$-projection of $p$ onto an $m$-flat subspace containing $r$.

### Applications in ML

**K-FAC (Kronecker-Factored Approximate Curvature):** approximates $G^{-1}$ layer-wise as a Kronecker product $A \otimes B$, making natural gradient practical for deep networks.

**TRPO/PPO:** policy gradient methods that constrain policy updates to be small in KL-divergence terms — a discretized natural gradient step with a trust region.

**Variational inference:** ELBO maximization can be viewed as a natural gradient step in the space of variational distributions.

*See also: [[fisher-information]] · [[exponential-family]] · [[variational-inference]] · [[matrix-calculus]]*
