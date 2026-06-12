---
title: Coordinate Descent
tags: [coordinate-descent, block-coordinate, alternating-minimization, admm, optimization]
aliases: [coordinate descent, block coordinate descent, alternating minimization, ADMM, Gauss-Seidel]
difficulty: 2
status: complete
related: [optimizer-sgd-momentum, lagrangian-optimization, second-order-optimization, numerical-methods, support-vector-machines]
---

# Coordinate Descent

---

## Fundamental

### Algorithm

**Coordinate descent** minimizes $f(\theta_1, \ldots, \theta_n)$ by cycling through variables and minimizing over each one while holding the others fixed:

$$\theta_j^{(t+1)} = \arg\min_{\theta_j} f(\theta_1^{(t+1)}, \ldots, \theta_{j-1}^{(t+1)}, \theta_j, \theta_{j+1}^{(t)}, \ldots, \theta_n^{(t)})$$

Each 1D subproblem is solved exactly (or approximately). The process cycles through $j = 1, \ldots, n$ repeatedly.

**Convergence:** for convex functions with a unique minimizer, CD converges. For differentiable strictly convex functions, it converges to the global minimum. For non-smooth functions (LASSO), it converges if each subproblem has a unique solution.

### When CD is Useful

- Each subproblem has a closed-form solution (much cheaper than gradient steps)
- The function is separable or has sparse coupling structure
- The gradient is expensive but per-coordinate updates are cheap
- Distributed optimization where variables are split across machines

---

## Intermediate

### LASSO via Coordinate Descent

The LASSO problem $\min_w \|y - Xw\|^2 + \lambda\|w\|_1$ has no closed-form solution globally, but has a closed-form per-coordinate update (soft thresholding):

$$w_j \leftarrow \text{sign}(r_j^{-j}) \cdot \max(0, |r_j^{-j}| - \lambda/2)$$

where $r_j^{-j} = \langle x_j, y - X_{-j}w_{-j}\rangle / \|x_j\|^2$ is the partial residual for variable $j$.

This is the core of **GLMNET** — the most widely used LASSO solver — which cycles through coordinates and applies soft thresholding, converging in $O(np/\epsilon)$ operations.

### Block Coordinate Descent (BCD)

**BCD** partitions variables into blocks and minimizes over each block simultaneously:

$$\theta_B^{(t+1)} = \arg\min_{\theta_B} f(\theta_B, \theta_{\bar{B}}^{(t)})$$

**Examples:**
- **Alternating least squares (ALS)** for matrix factorization: fix $U$, solve for $V$ (linear system); fix $V$, solve for $U$ — each block is a linear system
- **EM algorithm** (see [[expectation-maximization]]): alternating between the E-step (update latent variables) and M-step (update parameters)
- **K-means**: alternating between cluster assignment (argmin) and centroid update (mean)

All of these are instances of block coordinate descent.

### ADMM: Alternating Direction Method of Multipliers

**ADMM** solves constrained problems by alternating between primal and dual variable updates:

$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c$$

**ADMM iterations:**
$$x^{k+1} = \arg\min_x \left(f(x) + \frac{\rho}{2}\|Ax + Bz^k - c + u^k\|^2\right)$$
$$z^{k+1} = \arg\min_z \left(g(z) + \frac{\rho}{2}\|Ax^{k+1} + Bz - c + u^k\|^2\right)$$
$$u^{k+1} = u^k + Ax^{k+1} + Bz^{k+1} - c$$

ADMM is widely used for distributed optimization (split variables across machines, coordinate via dual variable $u$) and problems where $f$ and $g$ are each tractable separately but not jointly (e.g., one is smooth, one is a proximal operator like $\ell_1$).

---

## Advanced

### Randomized and Asynchronous CD

**Randomized CD:** select coordinate $j$ randomly at each step (instead of cyclically). For convex functions, converges as fast as full gradient descent in terms of iterations but $n\times$ cheaper per step.

**Asynchronous CD (Hogwild!):** in parallel settings, multiple workers update different coordinates simultaneously without locking. For sparse problems (few coordinate collisions), this is nearly equivalent to synchronized updates. Used in large-scale logistic regression and embedding training.

### CD for SVM (SMO)

Sequential Minimal Optimization (see [[support-vector-machines]]) is coordinate descent on the SVM dual, updating 2 dual variables per step (the minimum block size that satisfies the equality constraint $\sum_i \alpha_i y_i = 0$). Each 2D subproblem has a closed-form solution.

*See also: [[lagrangian-optimization]] · [[second-order-optimization]] · [[support-vector-machines]] · [[expectation-maximization]] · [[numerical-methods]]*
