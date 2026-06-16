---
title: Hessian Matrix and Curvature
tags: [hessian, curvature, second-order, optimization, loss-landscape]
aliases: [Hessian, second-order derivatives, curvature, saddle points]
difficulty: 2
status: complete
depends_on: [matrix-calculus, eigenvalues-pca]
related: [loss-landscape, matrix-calculus, lagrangian-optimization, numerical-methods, eigenvalues-pca]
---

# Hessian Matrix and Curvature

---

## Fundamental

While the gradient tells you which direction to go to reduce a function, the **Hessian** tells you how curved the function is — whether a step in some direction will decrease it a little or a lot. In ML it appears in second-order optimization (Newton's method, K-FAC), loss landscape analysis (flat vs. sharp minima and generalization), and explains why poorly-conditioned loss surfaces make gradient descent slow.

### The Hessian

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** $H \in \mathbb{R}^{n \times n}$ is the matrix of second partial derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

where $f: \mathbb{R}^n \to \mathbb{R}$ = scalar function (e.g., the training loss), $x_i, x_j$ = the $i$-th and $j$-th parameters, and $H_{ij}$ = how the gradient in direction $i$ changes as we move in direction $j$.

For smooth $f$, the Hessian is symmetric ($H = H^\top$) by Schwarz's theorem.

**Second-order Taylor expansion:** near a point $\mathbf{x}_0$:

$$f(\mathbf{x}_0 + \mathbf{d}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top H(\mathbf{x}_0) \mathbf{d}$$

The Hessian captures how the gradient changes — i.e., the local curvature in every direction.

### Curvature and Critical Points

At a critical point $\nabla f = 0$, the Hessian determines the type:

| Hessian eigenvalues | Point type |
|---------------------|------------|
| All positive (PD) | Local minimum |
| All negative (ND) | Local maximum |
| Mixed signs | Saddle point |
| Some zero | Degenerate (indeterminate) |

**Curvature in direction $\mathbf{d}$:** $\kappa = \mathbf{d}^\top H \mathbf{d} / \|\mathbf{d}\|^2$, where $\mathbf{d} \in \mathbb{R}^n$ = a unit direction vector and $\kappa$ = the second-order rate of increase of $f$ in that direction. The eigenvalues of $H$ are the principal curvatures — maximum curvature = $\lambda_\text{max}$, minimum = $\lambda_\text{min}$.

---

## Intermediate

### Condition Number and Optimization Difficulty

The **condition number** $\kappa(H) = \lambda_\text{max} / \lambda_\text{min}$ governs convergence of gradient descent:

- Well-conditioned ($\kappa \approx 1$): gradients point toward the minimum, fast convergence
- Ill-conditioned ($\kappa \gg 1$): elongated loss surface, gradients nearly perpendicular to the minimum — gradient descent zigzags

**SGD convergence rate** for quadratic objectives: $O(\kappa)$ steps to reduce error by a constant factor. Adam and adaptive methods implicitly precondition by the diagonal of $H$ (or an approximation).

**Newton's method** uses the full Hessian to precondition: $\mathbf{x} \leftarrow \mathbf{x} - H^{-1} \nabla f$. This is curvature-aware gradient descent — one step reaches the exact minimum for quadratics. Cost: $O(n^3)$ to invert $H$; impractical for deep networks with millions of parameters.

### Hessian in Deep Learning

**Loss surface of neural networks:** at a trained minimum, the Hessian has:
- A small number of large positive eigenvalues (a "bulk" corresponding to sharp directions)
- A large number of near-zero eigenvalues (flat directions — weight symmetries, overparameterization)

This means neural network loss surfaces are nearly flat in most directions — many local minima of equal loss quality exist. The sharpness of the few large eigenvalues determines generalization:

**Flat minima generalize better:** if a minimum has small $\lambda_\text{max}$, small perturbations of the weights don't increase the loss much — the solution is robust. Sharp minima (large $\lambda_\text{max}$) are sensitive to weight perturbations, correlating with worse generalization (Hochreiter & Schmidhuber, 1997).

**Sharpness-aware minimization (SAM):** explicitly minimizes the worst-case loss in a small ball $\|\epsilon\|_2 \leq \rho$ around the current weights, finding flatter minima. Requires two forward-backward passes per step.

---

## Advanced

### Hessian-Vector Products

Computing the full $n \times n$ Hessian is $O(n^2)$ in memory. For large networks, one instead computes **Hessian-vector products** $H\mathbf{v}$ in $O(n)$ via the **Pearlmutter trick** (forward-over-backward AD):

$$H\mathbf{v} = \nabla(\nabla f \cdot \mathbf{v})$$

where $\mathbf{v} \in \mathbb{R}^n$ = an arbitrary direction vector, $\nabla f$ = the gradient vector, $\nabla f \cdot \mathbf{v}$ = a scalar (directional derivative of $f$ in direction $\mathbf{v}$), and taking the gradient of that scalar gives the Hessian-vector product. One forward + one backward pass suffices — no materialization of the full $n\times n$ matrix.

This enables iterative eigenvalue methods (Lanczos) to estimate the spectrum, and Hessian-free optimization (conjugate gradient with $H\mathbf{v}$ oracle).

### Fisher Information and the Hessian

For a log-likelihood $\ell(\theta) = \log p(x|\theta)$, the **Fisher information matrix** $F = \mathbb{E}[\nabla \ell \nabla \ell^\top]$ equals the expected Hessian under mild conditions:

$$F = -\mathbb{E}\left[\frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j}\right]$$

The natural gradient (see [[fisher-information]]) uses $F^{-1}$ as a preconditioning matrix, equivalent to second-order optimization in the space of distributions rather than parameters.

### Practical Hessian Approximations

| Method | Approximation | Cost |
|--------|--------------|------|
| Diagonal | $\text{diag}(H)$ only | $O(n)$ extra memory |
| K-FAC | Kronecker-factored $H$ per layer | $O(d^2)$ per layer |
| L-BFGS | Low-rank update from gradient history | $O(mn)$, $m$ history steps |
| Empirical Fisher | $\sum_i \nabla \ell_i \nabla \ell_i^\top$ | $O(n)$ per batch |

## Links

- [[matrix-calculus]] — the Hessian is the matrix of second-order partial derivatives; computing it requires the chain rule applied twice
- [[eigenvalues-pca]] — the Hessian's eigenvalues determine curvature: positive eigenvalues = convex direction, zero = flat direction, negative = saddle; largest eigenvalue = sharpness
- [[loss-landscape]] — the Hessian characterizes the local geometry of the loss landscape; sharpness (largest eigenvalue) correlates with poor generalization
- [[fisher-information]] — the Fisher information matrix is equal to the expected Hessian of the negative log-likelihood for exponential family models
- [[numerical-methods]] — the Hessian is $O(p^2)$ to store and $O(p^3)$ to invert; quasi-Newton methods (L-BFGS) approximate the inverse Hessian implicitly
