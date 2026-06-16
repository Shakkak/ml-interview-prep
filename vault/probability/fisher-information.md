---
title: Fisher Information and Natural Gradient
tags: [statistics, optimization, fisher-information, natural-gradient, information-theory]
aliases: [Fisher information, Fisher matrix, natural gradient, Cramér-Rao, KFAC]
difficulty: 3
status: complete
depends_on: [statistical-inference-mle, matrix-calculus, entropy-mutual-info]
related: [statistical-inference-mle, matrix-calculus, optimizer-adam, entropy-mutual-info, bayesian-inference]
---

# Fisher Information and Natural Gradient

---

## Fundamental

**Fisher information** measures how much information a single data observation carries about a model parameter — specifically, how sharply the likelihood peaks near the true parameter value. A high Fisher information means small data changes shift the likelihood a lot, so the parameter is easy to estimate. It sets the fundamental lower bound on estimator variance (Cramér-Rao bound) and is the metric that natural gradient methods use instead of Euclidean distance in parameter space.

### Fisher Information (Scalar Case)

For a model with parameter $\theta$ and log-likelihood $\ell(\theta; x) = \log p(x; \theta)$:

$$I(\theta) = \mathbb{E}_x\!\left[\left(\frac{\partial}{\partial\theta}\log p(x;\theta)\right)^2\right] = -\mathbb{E}_x\!\left[\frac{\partial^2}{\partial\theta^2}\log p(x;\theta)\right]$$

where $\theta$ = the model parameter we want to estimate, $p(x;\theta)$ = the probability of observation $x$ under parameter $\theta$, $\frac{\partial}{\partial\theta}\log p(x;\theta)$ = the score function (how fast the log-likelihood changes with $\theta$ at a given $x$), $\mathbb{E}_x[\cdot]$ = expectation over all possible observations $x$, and $I(\theta)$ = the expected squared score (high $I(\theta)$ means observations are highly informative about $\theta$).

The two expressions are equal (score identity). $I(\theta) \geq 0$ always.

**Interpretation:** $I(\theta)$ measures how much information a single observation carries about $\theta$. High Fisher information = likelihood changes rapidly with $\theta$ = $\theta$ is easy to estimate from data.

### Cramér-Rao Lower Bound

For any **unbiased** estimator $\hat\theta$:

$$\text{Var}(\hat\theta) \geq \frac{1}{I(\theta)}$$

where $\text{Var}(\hat\theta)$ = variance of any unbiased estimator, $I(\theta)$ = Fisher information at the true $\theta$, and $1/I(\theta)$ = the minimum achievable variance (the lower bound). The bound says: the more information a sample carries about $\theta$, the lower we can make the estimator's variance.

No unbiased estimator can have variance smaller than $1/I(\theta)$. For $n$ i.i.d. observations: $\text{Var}(\hat\theta) \geq 1/(nI(\theta))$.

**Example — Bernoulli($p$):**

| $p$ | $I(p) = 1/(p(1-p))$ | Meaning |
|-----|---------------------|---------|
| 0.5 | 4 | Least informative — most uncertainty |
| 0.1 | 11.1 | More informative |
| 0.01 | 101 | Very informative — extreme $p$ easier to estimate |

For $p=0.5$, $n=100$: $\text{Var}(\hat{p}) \geq 0.0025$, SD $\geq 0.05$.

**[[statistical-inference-mle|MLE]] achieves this bound asymptotically** — MLE is the most efficient unbiased estimator. As $n \to \infty$: $\sqrt{n}(\hat\theta_{MLE} - \theta^*) \to \mathcal{N}(0, I(\theta^*)^{-1})$.

### Fisher Information for Gaussians

For $p(x;\mu,\sigma^2) = \mathcal{N}(\mu,\sigma^2)$:

$$I(\mu) = \frac{1}{\sigma^2}, \quad I(\sigma^2) = \frac{1}{2\sigma^4}$$

The Cramér-Rao bound gives $\text{Var}(\hat\mu) \geq \sigma^2/n$ — matching the sample mean variance, which achieves the bound.

---

## Intermediate

### Fisher Information Matrix (Multivariate)

For $\theta \in \mathbb{R}^d$:

$$F(\theta) = \mathbb{E}_x\!\left[\nabla_\theta \log p(x;\theta)\; \nabla_\theta \log p(x;\theta)^\top\right] = -\mathbb{E}_x\!\left[\nabla^2_\theta \log p(x;\theta)\right]$$

where $\nabla_\theta \log p(x;\theta) \in \mathbb{R}^d$ = the score vector (gradient of the log-likelihood w.r.t. all $d$ parameters), $\nabla_\theta \log p \cdot (\nabla_\theta \log p)^\top$ = the outer product of the score vector with itself (a $d\times d$ rank-1 matrix), and $\nabla^2_\theta \log p(x;\theta)$ = the Hessian of the log-likelihood (the second expression is the expected negative Hessian).

$F$ is a $d \times d$ positive semi-definite matrix. It equals both the expected outer product of the score vector and the expected negative [[matrix-calculus|Hessian]] of the log-likelihood.

### The Natural Gradient

**Problem with standard gradient descent:** moving in Euclidean parameter space with step $\|\Delta\theta\|_2 = \epsilon$ can cause very different changes to the model's distribution depending on local curvature.

**Natural gradient descent** moves a fixed size in terms of KL divergence between old and new distribution:

$$\theta \leftarrow \theta - \eta\, F(\theta)^{-1} g$$

where $g = \nabla_\theta L$ = the ordinary gradient of the loss, $F(\theta)$ = the Fisher information matrix (provides the geometry of the parameter space), $F(\theta)^{-1} g$ = the natural gradient (ordinary gradient rotated and scaled to account for distribution curvature), and $\eta$ = learning rate.

**Key property:** the natural gradient is invariant to parameterization — reparameterizing the model does not change the update direction (unlike ordinary gradient).

**Why:** natural gradient solves $\min_{\delta\theta} g^\top \delta\theta$ subject to $D_{KL}(p_\theta \| p_{\theta+\delta\theta}) \leq \epsilon$ — it takes the steepest descent step in [[loss-kl-divergence|distribution space]], not parameter space.

### Fisher Information and KL Divergence

The KL divergence between nearby distributions is:

$$\text{KL}(p(\cdot;\theta) \| p(\cdot;\theta+\delta)) \approx \frac{1}{2}\delta^\top F(\theta)\,\delta$$

This second-order Taylor expansion shows the Fisher matrix is the Riemannian metric on the manifold of probability distributions — the core idea of **information geometry**.

### Empirical Fisher

In practice, compute:

$$\hat{F} = \frac{1}{n}\sum_{i=1}^n g_i g_i^\top \quad \text{where } g_i = \nabla_\theta \log p(x_i; \theta)$$

Using observed data instead of the true expectation. This is what Adam approximates (diagonal of $\hat{F}$).

---

## Advanced

### Why We Don't Use Natural Gradient Directly

Computing and inverting $F \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ storage and $O(d^3)$ inversion — infeasible for neural networks with millions of parameters.

### Practical Approximations: Diagonal Fisher (Adam)

[[optimizer-adam|Adam]]'s second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ is an exponential moving average of $g_i^2$ — a running estimate of the diagonal Fisher. Adam's update $g_t / \sqrt{v_t}$ approximates $F^{-1}g$. This is a first-order diagonal approximation that ignores all off-diagonal curvature.

### K-FAC (Kronecker-Factored Approximate Curvature)

Martens & Grosse (2015) proposed approximating the layer-wise Fisher block as a Kronecker product:

$$F_\ell \approx A_\ell \otimes G_\ell$$

where $A_\ell = \mathbb{E}[a_\ell a_\ell^\top]$ is the covariance of layer inputs, $a_\ell \in \mathbb{R}^{d_{in}}$ = activations fed into layer $\ell$, $G_\ell = \mathbb{E}[\nabla_{s_\ell}\mathcal{L}\, (\nabla_{s_\ell}\mathcal{L})^\top]$ is the covariance of backpropagated gradients, and $s_\ell \in \mathbb{R}^{d_{out}}$ = pre-activations (output of the linear map $W_\ell a_\ell$, before the nonlinearity). Inversion: $(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}$, reducing cost from $O(d_\ell^3)$ to $O(d_{in}^3 + d_{out}^3)$ per layer.

**Extensions:** EKFAC (George et al., 2018) improves K-FAC by rescaling eigenvalues; FOOF (Benzing, 2022) uses only the input covariance $A_\ell$ for a cheaper approximation. In practice, K-FAC requires careful damping ($F + \lambda I$) and periodic Fisher updates to remain stable.

### Natural Gradient and Trust Region Methods

The natural gradient update is equivalent to the Trust Region Policy Optimization (TRPO) constraint in RL:

$$\max_\theta E[r(\theta)] \quad \text{s.t.} \quad D_{KL}(p_{\theta_{old}} \| p_\theta) \leq \delta$$

The first-order approximation to this constraint gives $\delta \approx \frac{1}{2}\Delta\theta^\top F \Delta\theta$, so the optimal step is $\Delta\theta = F^{-1}g \cdot \sqrt{2\delta/g^\top F^{-1}g}$. Proximal Policy Optimization (PPO) approximates this with a clipped objective, avoiding the matrix inversion entirely at the cost of a weaker geometric guarantee.

### Laplace Approximation and Observed Fisher

The observed Fisher information at the MLE:

$$\hat{F} = -\nabla^2_\theta \log p(D|\hat\theta_{MLE})$$

is the curvature of the negative log-likelihood at the mode. The Laplace approximation to the [[bayesian-inference|Bayesian posterior]] uses this as the precision of a Gaussian approximation: $p(\theta|D) \approx \mathcal{N}(\hat\theta_{MLE}, \hat{F}^{-1})$. This is the basis for post-hoc Bayesian deep learning (Ritter et al., 2018) and for Bayesian optimization acquisition functions that require predictive variance.

---

## Links

- [[statistical-inference-mle]] — the Cramér-Rao bound says no unbiased estimator can have variance less than $1/I(\theta)$; MLE achieves this bound asymptotically
- [[matrix-calculus]] — the Fisher matrix is the expected outer product of the score; computing and inverting it requires Hessian and gradient machinery
- [[entropy-mutual-info]] — Fisher information is the second derivative of KL divergence at zero; both measure informativeness of a distribution
- [[bayesian-inference]] — the Jeffreys prior $p(\theta) \propto \sqrt{\det F(\theta)}$ is invariant under reparameterization; posterior uncertainty shrinks as $1/\sqrt{nI(\theta)}$
- [[loss-kl-divergence]] — the Fisher information matrix is the Hessian of $D_{KL}(p_\theta \| p_{\theta+d\theta})$ at $d\theta = 0$
- [[optimizer-adam]] — natural gradient preconditions by $F^{-1}$; K-FAC approximates the Fisher for tractable natural gradient descent in neural networks
