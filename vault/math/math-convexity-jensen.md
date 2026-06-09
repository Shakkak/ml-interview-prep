---
title: Convexity and Jensen's Inequality
tags: [math, optimization, probability, convexity, jensen, strong-convexity]
aliases: [convexity, Jensen's inequality, convex function, concave function, strong convexity, ELBO]
difficulty: 2
status: complete
related: [loss-kl-divergence, variational-autoencoders, bayesian-inference, entropy-mutual-info, lagrangian-optimization]
---

# Convexity and Jensen's Inequality

---

## Fundamental

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for any $x, y$ and $\lambda \in [0,1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

Geometrically: the chord connecting any two points on the graph lies above or on the graph ("bowl up"). **Concave** functions reverse the inequality ("bowl down").

**Second-order characterization:** for twice-differentiable $f$, convex $\Leftrightarrow$ $\nabla^2 f \succeq 0$ (Hessian is positive semi-definite everywhere).

Common examples:

| Function | Convex? | Domain |
|---|---|---|
| $x^2$ | ✓ (strongly) | $\mathbb{R}$ |
| $e^x$ | ✓ | $\mathbb{R}$ |
| $-\log x$ | ✓ | $x > 0$ |
| $|x|$ | ✓ | $\mathbb{R}$ |
| $\log x$ | ✗ (concave) | $x > 0$ |
| $\sqrt{x}$ | ✗ (concave) | $x \geq 0$ |
| $x^2 - x^4$ | ✗ (neither) | $\mathbb{R}$ |

### Jensen's Inequality

For a **convex** function $f$ and a random variable $X$:

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

For a **concave** $f$: the inequality reverses. The function of the average $\leq$ the average of the function.

**Finite form:** $f\!\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)$ for any weights $\lambda_i \geq 0$ summing to 1.

**Worked example.** $f(x) = x^2$ (convex), $X \in \{1, 3\}$ with equal probability.

$f(\mathbb{E}[X]) = f(2) = 4$. $\mathbb{E}[f(X)] = (f(1) + f(3))/2 = (1+9)/2 = 5$.

$4 \leq 5$ ✓. Geometric picture: the midpoint of the parabola at $x=2$ (value 4) lies below the chord from $(1,1)$ to $(3,9)$ evaluated at $x=2$ (value 5).

---

## Intermediate

### KL Divergence is Non-Negative (Gibbs Inequality)

$$D_{KL}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = -\mathbb{E}_p\!\left[\log\frac{q(X)}{p(X)}\right]$$

Apply Jensen to $f = -\log$ (convex since $\log$ is concave):

$$-\mathbb{E}_p\!\left[\log\frac{q}{p}\right] \geq -\log\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log 1 = 0$$

Therefore $D_{KL}(p \| q) \geq 0$ with equality iff $q = p$ everywhere. This single inequality, proved by Jensen in one line, underlies information theory, statistical physics, and variational inference.

### ELBO Derivation via Jensen

The log-likelihood of data $x$ under model $p_\theta(x) = \int p_\theta(x|z)p(z)\,dz$:

$$\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z|x)}\!\left[\frac{p_\theta(x,z)}{q_\phi(z|x)}\right]$$

Apply Jensen ($\log$ is concave, so $\log\mathbb{E}[Y] \geq \mathbb{E}[\log Y]$):

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi}\!\left[\log\frac{p_\theta(x,z)}{q_\phi(z|x)}\right] = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}} = \text{ELBO}$$

Jensen converts an intractable log-integral into a tractable lower bound. The gap equals $D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) \geq 0$. Maximizing the [[variational-autoencoders|ELBO]] simultaneously tightens the lower bound (pushes $q_\phi$ toward the true [[bayesian-inference|posterior]]) and maximizes the likelihood lower bound.

### Strong Convexity and Convergence Rates

$f$ is **$\mu$-strongly convex** if:
$$f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2$$

Equivalently, $\nabla^2 f \succeq \mu I$ everywhere. Strong convexity implies a unique minimum.

**Gradient descent convergence** for $\mu$-strongly convex, $L$-smooth $f$ (Lipschitz gradient: $\nabla^2 f \preceq LI$) with step size $\eta = 1/L$:
$$f(x_t) - f^* \leq \left(1 - \frac{\mu}{L}\right)^t (f(x_0) - f^*)$$

The convergence factor $(1 - \mu/L) = (1 - 1/\kappa) < 1$ where $\kappa = L/\mu$ is the condition number. High $\kappa$ → slow convergence. This is why preconditioning ([[optimizer-adam|Adam]], quasi-Newton) and normalizing inputs ([[normalization-layers|batchnorm, layer norm]]) dramatically speed up training — they reduce the effective $\kappa$.

**Comparison of convergence rates:**

| Method | Rate | Oracle calls to $\epsilon$-accuracy |
|---|---|---|
| Gradient descent | Linear: $(1-1/\kappa)^t$ | $O(\kappa \log 1/\epsilon)$ |
| [[optimizer-sgd-momentum\|Nesterov momentum]] | Linear: $(1-1/\sqrt{\kappa})^t$ | $O(\sqrt{\kappa}\log 1/\epsilon)$ |
| Newton's method | Quadratic near optimum | $O(\log\log 1/\epsilon)$ |

---

## Advanced

### Log-Sum Inequality and Entropy Proofs

**Log-sum inequality:** for $a_i, b_i > 0$:
$$\sum_i a_i \log\frac{a_i}{b_i} \geq \left(\sum_i a_i\right)\log\frac{\sum_i a_i}{\sum_i b_i}$$

This generalizes Jensen (the KL case is $b_i = \sum_j a_j$) and is used to prove subadditivity of [[entropy-mutual-info|entropy]]:

$$H(X, Y) \leq H(X) + H(Y)$$

with equality iff $X \perp Y$. This follows from $D_{KL}(p_{XY} \| p_X p_Y) \geq 0$ — the [[loss-kl-divergence|KL divergence]] between the joint and the product of marginals is the mutual information $I(X;Y) \geq 0$.

### AM-GM from Jensen

For positive numbers $x_1, \ldots, x_n$, apply Jensen to $f(x) = -\log x$ (convex, concave $\log$ reversed):

$$-\log\!\left(\frac{1}{n}\sum_i x_i\right) \leq \frac{1}{n}\sum_i (-\log x_i) = -\log\!\left(\prod_i x_i^{1/n}\right)$$

Negating: $\log\!\left(\frac{\sum x_i}{n}\right) \geq \log\!\left(\prod x_i^{1/n}\right)$. Exponentiating:

$$\frac{x_1 + \cdots + x_n}{n} \geq (x_1 \cdots x_n)^{1/n}$$

The classical AM-GM inequality is a special case of Jensen's inequality applied to the concave function $\log$.

### Non-Convex Landscape of Neural Networks

The composition of convex functions need not be convex. $\text{ReLU}(x) = \max(0, x)$ is convex, but $\text{ReLU}(W_2 \text{ReLU}(W_1 x))$ is not. Neural network loss landscapes are highly non-convex with:
- **Saddle points** (gradient = 0, indefinite Hessian): exponentially more common than local minima in high dimensions. First-order methods (SGD) escape saddle points with random perturbations; second-order methods (Newton) are attracted to them.
- **Local minima** that are not global: exist but empirically have similar loss to global minima for overparameterized networks (Goodfellow et al., 2015).
- **Flat directions (zero curvature):** parameter redundancies. A permutation of hidden units gives identical function — the loss is constant along these directions.

**Why SGD still works:** (1) the loss landscape of overparameterized networks is "almost convex" near global minima — many descent directions everywhere; (2) SGD noise prevents convergence to sharp minima, preferring flat minima that generalize better (Hochreiter & Schmidhuber, 1997); (3) the implicit regularization of gradient descent (moving minimum-norm direction) biases toward solutions with good generalization.

**Sharp vs flat minima and generalization:** a minimum is flat if the Hessian has small eigenvalues ($\mu$ small, $\kappa$ large but with a flat floor). Flat minima generalize better empirically — small perturbations to weights don't increase the loss, suggesting the model is not memorizing fine details. SAM (Sharpness-Aware Minimization, Foret et al., 2021) explicitly minimizes the loss at the worst-case perturbation within a ball, converging to flatter minima with improved generalization.

---

*See also: [[loss-kl-divergence]] · [[variational-autoencoders]] · [[entropy-mutual-info]] · [[lagrangian-optimization]] · [[optimizer-adam]] · [[normalization-layers]] · [[optimizer-sgd-momentum]]*
