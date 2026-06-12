---
title: Second-Order Optimization Methods
tags: [second-order-optimization, newton-method, bfgs, kfac, hessian-free]
aliases: [Newton's method, quasi-Newton, L-BFGS, BFGS, Hessian-free optimization, K-FAC]
difficulty: 3
status: complete
related: [hessian-curvature, optimizer-adam, optimizer-sgd-momentum, lagrangian-optimization, numerical-methods, fisher-information]
---

# Second-Order Optimization Methods

---

## Fundamental

### Newton's Method

Gradient descent takes a step proportional to the gradient. **Newton's method** takes a step proportional to the gradient preconditioned by the inverse Hessian:

$$\theta \leftarrow \theta - H^{-1}(\theta) \nabla \mathcal{L}(\theta)$$

**Why it's better:** gradient descent treats all parameters equally (assumes isotropic curvature). Newton's method accounts for curvature — large steps in flat directions, small steps in steep directions.

**Convergence:**
- Gradient descent: $O(\kappa)$ steps (condition number)
- Newton's method: quadratic convergence near optimum (1 step to optimum for quadratics)

**Cost:** computing $H^{-1}$ requires $O(d^3)$ for inversion or $O(d^2)$ to store — infeasible for $d = 10^7$ (large neural networks).

---

## Intermediate

### Quasi-Newton Methods

Quasi-Newton methods approximate $H^{-1}$ using gradient history, avoiding full Hessian computation.

**BFGS (Broyden-Fletcher-Goldfarb-Shanno):**
Updates an approximation $B_k \approx H_k^{-1}$ using the secant condition:
$$B_{k+1}(g_{k+1} - g_k) = \theta_{k+1} - \theta_k$$

The approximation is positive definite by construction. Full BFGS stores the dense $d \times d$ matrix — still $O(d^2)$ memory.

**L-BFGS (Limited-memory BFGS):** stores only the last $m = 10$–$20$ gradient/step pairs $\{(s_i, y_i)\}$ and reconstructs the $H^{-1}$ vector product on the fly in $O(md)$ time. Standard for machine learning problems with $d < 10^6$. Used in SciPy, PyTorch LBFGS optimizer.

**Convergence:** L-BFGS typically converges in $O(\kappa)$ to $O(1)$ steps (superlinear convergence) — much faster than gradient descent for well-conditioned problems.

### Natural Gradient

**Natural gradient** (see [[fisher-information]]) uses the Fisher information matrix $F$ as the preconditioner instead of the Hessian:

$$\tilde{\nabla} \mathcal{L} = F^{-1} \nabla \mathcal{L}$$

For exponential family log-likelihoods, $F = -\mathbb{E}[H]$ — the Fisher equals the expected Hessian. The natural gradient is geometrically invariant to parameterization, converging in $O(1)$ steps for exponential families.

---

## Advanced

### K-FAC: Kronecker-Factored Approximate Curvature

**K-FAC (Martens & Grosse, 2015)** makes natural gradient practical for deep networks by approximating the Fisher as block-diagonal, with each block approximated as a Kronecker product:

$$F_l \approx A_{l-1} \otimes G_l$$

where $A_{l-1} = \mathbb{E}[a_{l-1} a_{l-1}^\top]$ (input activation covariance) and $G_l = \mathbb{E}[g_l g_l^\top]$ (gradient covariance for layer $l$).

The inverse: $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$ — invert two small matrices instead of one large one. $O(d^{3/2})$ vs $O(d^3)$.

K-FAC converges 5–10× faster than Adam in training steps, but each step is more expensive. Used in large-scale training experiments (Shampoo optimizer generalizes K-FAC to higher-order Kronecker factors).

### Hessian-Free Optimization

**Hessian-free (HF) optimization (Martens, 2010):** uses conjugate gradient (CG) to solve $H d = -g$ for the Newton step $d$, using only Hessian-vector products $Hv$ (computed via automatic differentiation in $O(d)$).

Algorithm:
1. Compute gradient $g$
2. Run CG using $Hv$ oracle to solve $Hd \approx -g$
3. Take step $\theta \leftarrow \theta + \alpha d$ (with line search or trust region)

CG with $k$ iterations gives an $O(k)$-accurate approximation to the Newton step. Each CG iteration costs one Hessian-vector product ($O(d)$). For deep networks: effective with Tikhonov damping ($H + \lambda I$) to handle saddle points.

*See also: [[hessian-curvature]] · [[optimizer-adam]] · [[numerical-methods]] · [[fisher-information]] · [[lagrangian-optimization]]*
