---
title: Adam Optimizer
tags: [optimizer, training, adaptive-learning-rate]
aliases: [Adam, AdamW, adaptive moment estimation]
difficulty: 2
status: complete
depends_on: [optimizer-sgd-momentum, backpropagation]
related: [optimizer-sgd-momentum, optimizer-rmsprop-adagrad, optimizer-lr-schedules]
---

# Adam Optimizer

---

## Fundamental

[[optimizer-sgd-momentum|SGD]] with momentum adapts direction (accumulates gradient history) but uses the same learning rate for every parameter. Adam adds **per-parameter adaptive learning rates**: parameters with historically large gradients get smaller updates; sparse/small-gradient parameters get larger updates.

**Algorithm:** maintain two running averages per parameter:
$$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(first moment — exponential moving average of gradients)}$$
$$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(second moment — exponential moving average of squared gradients)}$$

where $g_t = \nabla_\theta L_t$ is the gradient at step $t$, $\beta_1$ controls how fast the first moment forgets old gradients (typical: 0.9 = 90% memory), and $\beta_2$ controls the second moment decay (typical: 0.999 = 99.9% memory).

**Bias correction** (corrects for both moments being initialized at zero):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

where $1 - \beta_1^t$ and $1 - \beta_2^t$ are correction factors that shrink toward 1 as $t$ grows (after ~50 steps, $(1-0.9^{50}) \approx 1$ and correction becomes negligible).

**Parameter update:**
$$\theta_t \leftarrow \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

where $\eta$ = global learning rate, $\hat{m}_t$ = bias-corrected gradient estimate (direction), $\sqrt{\hat{v}_t}$ = root-mean-square of historical gradients (scale), and $\epsilon$ = small constant for numerical stability (prevents division by zero when $\hat{v}_t \approx 0$).

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

**Why bias correction?** At $t=1$: $m_1 = 0.1 g_1$. Without correction, the first moment is $10\times$ too small — a tiny initial step. $\hat{m}_1 = 0.1 g_1 / (1-0.9) = g_1$. After $\sim 50$ steps, $(1-\beta^t) \approx 1$ and correction becomes negligible.

---

## Intermediate

**Worked numerical example** (one parameter, $\eta=0.001$, gradients $g_1=0.5$, $g_2=0.8$, $g_3=-0.3$):

Step 1 ($t=1$):
$$m_1 = 0.05,\; v_1 = 0.00025,\; \hat{m}_1 = 0.5,\; \hat{v}_1 = 0.25$$
$$\Delta\theta_1 = -\frac{0.001}{\sqrt{0.25}+10^{-8}}(0.5) = -0.001$$

Step 3 (gradient reverses to $-0.3$): $m_3 = 0.0825$, $v_3 \approx 0.000979$. The sign reversal slows $m_3$ (less gradient accumulated) but $v_3$ stays high from previous large gradients — protecting against large oscillatory steps.

**Adam vs SGD+Momentum:**

| | SGD+Momentum | Adam |
|---|---|---|
| Per-parameter LR | No | Yes |
| Memory | 1 extra per param | 2 extra per param |
| Sparse gradients | Poor | Good (NLP embeddings) |
| CV/ImageNet | Often better with tuning | Slightly worse |
| Transformers | Suboptimal | Much better |
| Flat minima tendency | More likely to escape | Less so |

**AdamW: decoupled weight decay** — L2 regularization in the loss adds $\lambda w$ to the gradient, which then gets divided by $\sqrt{\hat{v}} + \epsilon$. Parameters with large gradient history get their regularization scaled down — unintended behavior. AdamW separates weight decay from the gradient:
$$\theta_t \leftarrow \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

where $\lambda$ is the weight decay coefficient (e.g., $10^{-2}$). The $\lambda \theta_t$ term directly shrinks each parameter by a fraction $\eta\lambda$ per step, independent of gradient history — unlike L2 regularization in the loss which would also get divided by $\sqrt{\hat{v}_t}$.

Every parameter decays by the same fraction regardless of gradient history. **AdamW is almost always preferred over Adam for transformers and large vision models.**

**Common failure modes:**
- $\beta_2$ too low (e.g., 0.9): $v_t$ forgets quickly → noisy step sizes
- $\epsilon$ too small: numerical instability when $\hat{v}_t \approx 0$ for a parameter not yet seen
- LR too large: even with adaptivity, $10\times$ too-large LR diverges

---

## Advanced

**Adam's generalization gap — why SGD sometimes wins:** Wilson et al. (2017) showed that adaptive methods like Adam converge to **sharper minima** than SGD with momentum, leading to worse generalization even when training loss is lower. The mechanism: Adam divides the gradient by $\sqrt{\hat{v}_t}$, making the effective step size $\eta_\text{eff} = \eta / \sqrt{\hat{v}_t}$. Parameters with consistently small gradients get large effective steps; this allows the optimizer to enter and stay in narrow, sharp regions of the loss landscape that have small gradient curvature *in gradient magnitude* but large curvature *in loss value*. SGD, with its uniform learning rate and gradient noise from small batches, is more likely to escape these narrow basins.

The practical implication: for tasks where generalization matters most (image classification, language generation with many runs), SGD+momentum with a carefully tuned LR often outperforms Adam by 1–2% top-1. AdamW partially closes this gap by decoupling weight decay (which penalizes large weights uniformly, indirectly discouraging sharp minima), but the gap persists for large-scale vision tasks. The recommendation: use Adam/AdamW for transformers and sparse-gradient tasks (NLP embeddings); use SGD+momentum for CNNs with sufficient hyperparameter tuning budget.

**Adam's convergence theory is non-trivial:** Adam does not converge in general for all problems. Reddi et al. (2018) showed Adam can fail to converge even on convex problems — they constructed an adversarial example where $v_t$ forgets past large gradients, causing the effective LR to increase at exactly the wrong time. Their fix, AMSGrad, uses a running maximum $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ instead of EMA. In practice, AMSGrad often doesn't outperform Adam on standard benchmarks, suggesting Adam's non-convergence is not a practical issue for common ML problems but is a theoretical concern.

**Shampoo optimizer (Gupta et al., 2018):** approximates the full-matrix [[optimizer-rmsprop-adagrad|AdaGrad]] update using Kronecker-factored preconditioners. For a matrix parameter $W \in \mathbb{R}^{m \times n}$, Shampoo maintains $L = \sum_t G_t G_t^\top \in \mathbb{R}^{m \times m}$ and $R = \sum_t G_t^\top G_t \in \mathbb{R}^{n \times n}$ and preconditions by $L^{-1/4} G R^{-1/4}$. This approximates the natural gradient and significantly outperforms Adam on language model training per step (but requires matrix decompositions at each step, partially offsetting the advantage). Used at scale at Google for large LLM training.

**The $\beta_2 = 0.999$ choice and its implications:** the $\beta_2$ EMA window covers approximately $1/(1-\beta_2) = 1000$ gradient steps. This means the effective learning rate adapts slowly to changes in gradient scale. For learning rate schedules with warmup + decay, Adam's adaptivity means the effective update size at the end of training (low LR + stable $v_t$) is similar to the beginning of [[optimizer-lr-schedules|cosine decay]] — Adam doesn't naturally slow down as LR decays. This is why LR decay schedules are important for Adam even though the adaptive scaling already controls per-parameter learning rates.

**Adan (Adaptive Nesterov Momentum, Xie et al., 2022):** extends Adam to use a three-term estimate incorporating first, second, and difference of gradients (approximating Nesterov acceleration): $m_t = (1-\beta_1)g_t + \beta_1 m_{t-1}$, $v_t = (1-\beta_2)(g_t - g_{t-1}) + \beta_2 v_{t-1}$, $n_t = (1-\beta_3)(g_t + (1-\beta_2)(g_t-g_{t-1}))^2 + \beta_3 n_{t-1}$. Reported improvements of 1–2% on ImageNet and faster convergence on LLM pretraining compared to AdamW, with similar hyperparameter sensitivity.

---

## Links

- [[optimizer-sgd-momentum]] — Adam extends SGD+momentum by adding per-parameter adaptive learning rates; read this first
- [[backpropagation]] — provides the gradients $g_t$ that Adam's moment estimates are built from
- [[optimizer-rmsprop-adagrad]] — RMSProp is Adam's direct predecessor; Adam adds bias correction and a first-moment term
- [[optimizer-lr-schedules]] — Adam still benefits from LR warmup/decay because adaptive scaling does not replace scheduling
- [[regularization-weight-decay]] — AdamW decouples weight decay from the gradient update to fix the unintended interaction in Adam
