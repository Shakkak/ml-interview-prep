---
title: SGD and Momentum
tags: [optimizer, training, gradient-descent]
aliases: [SGD, stochastic gradient descent, momentum, mini-batch SGD]
difficulty: 1
status: complete
depends_on: [backpropagation, loss-mse]
related: [optimizer-adam, backpropagation, bias-variance-double-descent]
---

# SGD and Momentum

---

## Fundamental

**The problem:** a neural network has millions of parameters. We want to find the values that minimize the training loss. There is no closed-form solution for most architectures, so we must iterate. The gradient of the loss tells us the direction of steepest increase — subtracting it moves us toward a minimum.

**Gradient descent** minimizes $L(\theta)$ by repeatedly moving in the direction opposite to the gradient:

$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

where $\theta$ = model parameters (all weights and biases), $\eta$ = learning rate (step size, e.g., $10^{-3}$), $L(\theta)$ = the loss function, and $\nabla_\theta L(\theta)$ = gradient of the loss w.r.t. all parameters (pointing uphill; we subtract to go downhill).

**Geometric intuition:** think of a ball placed on a hilly landscape. The gradient is the slope at the current position. Gradient descent rolls the ball downhill one small step at a time until it reaches a valley (local minimum).

This requires computing the gradient over the entire dataset — one pass per update. For a dataset of 1M examples, that is one gradient step per epoch. Too slow.

**Stochastic Gradient Descent (SGD)** uses a mini-batch of $B$ samples per update:
$$\theta \leftarrow \theta - \frac{\eta}{B}\sum_{i \in \text{batch}} \nabla_\theta L_i(\theta)$$

where $B$ = batch size (number of samples per update, e.g., 32 or 256) and $L_i(\theta)$ = loss on the $i$-th sample; averaging over the batch gives an unbiased estimate of the full-dataset gradient.

**Why noise helps:** the noisy gradient estimate acts as regularization — helps escape sharp minima and saddle points. Smaller batches → more noise → stronger implicit regularization.

**Worked example:** minimize $L(\theta) = (\theta_1 - 3)^2 + (\theta_2 - 2)^2$ (minimum at $[3,2]$). Start at $[0,0]$, $\eta=0.3$.

Step 1: $\nabla L = [-6,-4]$, $\theta \leftarrow [0+1.8, 0+1.2] = [1.8, 1.2]$

Step 2: $\nabla L = [-2.4,-1.6]$, $\theta \leftarrow [2.52, 1.68]$

Converging toward $[3,2]$. Each step reduces distance by factor $(1-2\eta) = 0.4$.

**The oscillation problem:** for a loss with very different curvatures in different directions (e.g., $L = 10\theta_1^2 + \theta_2^2$, elongated bowl), the high-curvature direction has large gradients → large oscillations; the low-curvature direction has small gradients → slow progress.

**Momentum** maintains a running average of past gradients (velocity):
$$v_t \leftarrow \beta v_{t-1} + \nabla_\theta L_t, \qquad \theta \leftarrow \theta - \eta v_t$$

where $v_t$ = velocity (accumulated gradient direction at step $t$), $\beta$ = momentum coefficient (typically 0.9, meaning each step retains 90% of previous velocity), and $\nabla_\theta L_t$ = gradient at current step $t$. The update moves in the direction of velocity $v_t$, not just the current gradient.

Typical $\beta = 0.9$. Consistent gradient directions accumulate (accelerate); flipping directions cancel out (dampen oscillations).

---

## Intermediate

**Momentum numerical example** — bowl example with $\beta=0.9$, starting from $[0,0]$:

Step 1: $g_1=[-6,-4]$, $v_1=[-6,-4]$, $\theta_1=[1.8, 1.2]$

Step 2: $g_2=[-2.4,-1.6]$, $v_2=0.9[-6,-4]+[-2.4,-1.6]=[-7.8,-5.2]$, $\theta_2=[4.14, 2.76]$

Note: we overshot the optimum. Momentum accumulates velocity and initially oscillates, then converges faster. The steady-state speed (terminal velocity) is $v_\infty = g/(1-\beta)$ — momentum amplifies the effective step size by $1/(1-\beta) = 10$ in the direction of persistent gradient.

**Nesterov Momentum (look-ahead):**
$$v_t \leftarrow \beta v_{t-1} + \nabla_\theta L(\theta - \eta \beta v_{t-1}), \qquad \theta \leftarrow \theta - \eta v_t$$

where $\theta - \eta \beta v_{t-1}$ is the "look-ahead position" — where momentum would carry us next step. We evaluate the gradient there, not at the current position $\theta$. All other variables as above.

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

## Links

- [[backpropagation]] — provides the gradient $\nabla_\theta L$ that SGD applies to update parameters
- [[loss-mse]] — the loss landscape that gradient descent descends; loss function choice determines the gradient shape
- [[optimizer-adam]] — Adam extends SGD with per-parameter adaptive rates; understanding SGD is prerequisite
- [[large-batch-training]] — batch size directly controls the noise level in SGD's gradient estimates
- [[bias-variance-double-descent]] — SGD's noise-induced implicit regularization connects to why flat minima generalize better
