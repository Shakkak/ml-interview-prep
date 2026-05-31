---
title: SGD and Momentum
tags: [optimizer, training, gradient-descent]
aliases: [SGD, stochastic gradient descent, momentum, mini-batch SGD]
difficulty: 1
status: complete
related: [optimizer-adam, backpropagation, bias-variance-double-descent]
---

# SGD and Momentum

---

## Fundamental

**Gradient descent** on full dataset:
$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

Too slow for large datasets — one pass through all data per update.

**Stochastic Gradient Descent (SGD)** uses a mini-batch of $B$ samples per update:
$$\theta \leftarrow \theta - \frac{\eta}{B}\sum_{i \in \text{batch}} \nabla_\theta L_i(\theta)$$

**Why noise helps:** the noisy gradient estimate acts as regularization — helps escape sharp minima and saddle points. Smaller batches → more noise → stronger implicit regularization.

**Worked example:** minimize $L(\theta) = (\theta_1 - 3)^2 + (\theta_2 - 2)^2$ (minimum at $[3,2]$). Start at $[0,0]$, $\eta=0.3$.

Step 1: $\nabla L = [-6,-4]$, $\theta \leftarrow [0+1.8, 0+1.2] = [1.8, 1.2]$

Step 2: $\nabla L = [-2.4,-1.6]$, $\theta \leftarrow [2.52, 1.68]$

Converging toward $[3,2]$. Each step reduces distance by factor $(1-2\eta) = 0.4$.

**The oscillation problem:** for a loss with very different curvatures in different directions (e.g., $L = 10\theta_1^2 + \theta_2^2$, elongated bowl), the high-curvature direction has large gradients → large oscillations; the low-curvature direction has small gradients → slow progress.

**Momentum** maintains a running average of past gradients (velocity):
$$v_t \leftarrow \beta v_{t-1} + \nabla_\theta L_t, \qquad \theta \leftarrow \theta - \eta v_t$$

Typical $\beta = 0.9$. Consistent gradient directions accumulate (accelerate); flipping directions cancel out (dampen oscillations).

---

## Intermediate

**Momentum numerical example** — bowl example with $\beta=0.9$, starting from $[0,0]$:

Step 1: $g_1=[-6,-4]$, $v_1=[-6,-4]$, $\theta_1=[1.8, 1.2]$

Step 2: $g_2=[-2.4,-1.6]$, $v_2=0.9[-6,-4]+[-2.4,-1.6]=[-7.8,-5.2]$, $\theta_2=[4.14, 2.76]$

Note: we overshot the optimum. Momentum accumulates velocity and initially oscillates, then converges faster. The steady-state speed (terminal velocity) is $v_\infty = g/(1-\beta)$ — momentum amplifies the effective step size by $1/(1-\beta) = 10$ in the direction of persistent gradient.

**Nesterov Momentum (look-ahead):**
$$v_t \leftarrow \beta v_{t-1} + \nabla_\theta L(\theta - \eta \beta v_{t-1}), \qquad \theta \leftarrow \theta - \eta v_t$$

Computes gradient at "where we're headed" rather than "where we are." Better convergence on convex problems (Nesterov's accelerated gradient achieves $O(1/t^2)$ vs $O(1/t)$ for standard gradient descent on smooth convex objectives). Less sensitive to oscillations because the gradient is evaluated at the look-ahead point, not the current position.

**SGD vs Adam:**

| Property | SGD+Momentum | Adam |
|----------|-------------|------|
| Learning rate | Same for all parameters | Per-parameter adaptive |
| Hyperparameters | $\eta$, $\beta$ | $\eta$, $\beta_1$, $\beta_2$, $\epsilon$ |
| CV/ImageNet | Often better (with tuning) | Slightly worse |
| NLP/Transformers | Worse | Much better |
| Flat minima | More likely (noise) | Less so |
| Memory | 1 extra per param | 2 extra per param |

**Why SGD+momentum finds flatter minima:** the gradient noise from small batches and the momentum-dampened oscillations together prevent the optimizer from converging into narrow, sharp minima. This is the fundamental reason SGD with proper learning rate schedules often generalizes better than Adam on vision tasks (Wilson et al., 2017).

---

## Advanced

**Polyak-Ruppert averaging:** instead of using the final weight iterate, average all weights over the last $K$ epochs: $\bar{\theta} = \frac{1}{K}\sum_{t=T-K}^T \theta_t$. For SGD, this provably improves convergence rate on convex objectives from $O(1/\sqrt{t})$ to the optimal $O(1/t)$. In practice, stochastic weight averaging (SWA, Izmailov et al., 2018) applies this principle to deep learning: averaging multiple checkpoints from the end of training (or from cyclic LR restarts) consistently improves generalization by 0.5–1% on ImageNet and 1–2% on CIFAR, at zero inference cost.

**The implicit regularization of SGD toward flat minima:** the continuous-time Langevin dynamics view shows SGD with step size $\eta$ and batch size $B$ performs a stochastic process with noise temperature $T_{eff} = \eta/B$. Stationary distributions of Langevin dynamics favor flat regions of the loss landscape — regions with lower curvature. This provides a formal theory for why large learning rates and small batches find better-generalizing solutions (Chaudhari et al., 2017, "Entropy-SGD"). The noise explicitly maximizes entropy of the parameter distribution within the loss surface.

**Momentum schedule: the heavyball problem:** classical Polyak heavy-ball (constant $\beta$) converges on convex quadratics with rate depending on the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ of the Hessian. The optimal convergence rate is $O((1-1/\sqrt{\kappa})^t)$, achieved with $\beta^* = ((1-1/\sqrt{\kappa})/(1+1/\sqrt{\kappa}))^2$. For ill-conditioned problems ($\kappa \gg 1$), this means $\beta^* \approx 1$ — very high momentum, essentially taking many small steps in the same direction. This is the theoretical motivation for $\beta = 0.9$–$0.95$ in practice.

**SGDW and coupling weight decay:** the same L2 vs weight decay decoupling issue that motivates AdamW (see [[optimizer-adam]]) also applies to SGD: L2 regularization in the loss is algebraically equivalent to weight decay in SGD ($\theta \leftarrow (1-2\eta\lambda)\theta - \eta g$), so there is no difference for SGD. For SGD with momentum, however, the coupling of L2 regularization with the velocity term can cause regularization to be inconsistent across different momentum values. SGDW (Loshchilov & Hutter, 2019) decouples weight decay from the momentum update for cleaner regularization.

---

*See also: [[optimizer-adam]] · [[backpropagation]] · [[large-batch-training]]*
