---
title: Weight Decay (L1 / L2 Regularization)
tags: [regularization, training, overfitting]
aliases: [weight decay, L2 regularization, L1 regularization, ridge regression, lasso]
difficulty: 1
status: complete
related: [regularization-dropout, optimizer-adam, loss-cross-entropy, bias-variance-double-descent, regularization-early-stopping, bayesian-inference]
---

# Weight Decay (L1 / L2 Regularization)

---

## Fundamental

A model with too many parameters relative to training data memorizes noise. Regularization penalizes large weights, encouraging simpler solutions.

**L2 regularization (ridge / weight decay):**

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2}\|w\|_2^2$$

Gradient update:

$$w \leftarrow w - \eta\frac{\partial \mathcal{L}}{\partial w} - \eta\lambda w = (1-\eta\lambda)w - \eta\frac{\partial \mathcal{L}}{\partial w}$$

The factor $(1 - \eta\lambda)$ shrinks every weight by a constant fraction per step — hence "weight decay." Small and large weights are both reduced proportionally; no weight reaches exactly zero.

**L1 regularization (lasso):**

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{task}} + \lambda\|w\|_1$$

Gradient:

$$w \leftarrow w - \eta\frac{\partial \mathcal{L}}{\partial w} - \eta\lambda\cdot\text{sign}(w)$$

L1 applies *constant* shrinkage regardless of weight magnitude. Small weights cross zero and are set to exactly 0 → **sparse solutions**.

**Numerical example** ($w = 0.1$, $\lambda = 0.5$, $\eta = 1$, no task gradient):

| Method | Update | Result |
|---|---|---|
| L2 | $(1-0.5)(0.1)$ | $0.05$ — halved, never reaches 0 |
| L1 | $0.1 - 0.5 = -0.4$ → clip to 0 | $0$ — zeroed in one step |

**When to use which:**
- L2: default for neural networks; smooth, stable training
- L1: when sparse weights are desired (feature selection, embedded systems)
- Elastic net: $\alpha\|w\|_1 + (1-\alpha)\|w\|_2^2$ — sparsity with L2 stability for correlated features

---

## Intermediate

### Geometric Interpretation

$\min \mathcal{L}_{\text{task}} + \lambda R(w)$ is equivalent to $\min \mathcal{L}_{\text{task}}$ subject to $R(w) \leq c$:

- **L2:** constraint is a sphere $\|w\|_2 \leq c$. Smooth surface — solutions land anywhere; no sparsity.
- **L1:** constraint is a diamond (L1 ball) with corners on the axes. Optimal solutions prefer corners → many weights = 0.

### AdamW vs Adam + L2

With standard [[optimizer-adam|Adam]] + L2 in the loss, the gradient $\nabla \mathcal{L} + \lambda w$ passes through Adam's adaptive scaling — divided by $\sqrt{\hat{v}} + \epsilon$. Parameters with large gradient history receive *less* L2 push. This is unintended.

**AdamW** decouples weight decay from the gradient:

$$w_{t+1} = (1 - \eta\lambda)w_t - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Every parameter is decayed uniformly, independent of gradient history. **AdamW is almost always preferred over Adam + L2 for transformers.**

### Choosing $\lambda$

| Setting | Typical $\lambda$ |
|---|---|
| Vision CNNs (SGD) | $10^{-4}$ |
| Transformers (AdamW) | $10^{-2}$ |
| Fine-tuning | $10^{-3}$ to $10^{-2}$ |
| Very small datasets | $10^{-3}$ to $10^{-2}$ |

Cross-validate on a log scale: $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$.

---

## Advanced

### Bayesian Interpretation

L2 and L1 regularization are [[bayesian-inference|MAP]] estimation with specific priors over weights:

- **L2 (ridge):** $w_i \sim \mathcal{N}(0, 1/\lambda)$. The Gaussian prior penalizes large deviations from zero quadratically. The MAP objective $-\log p(w|X,y) = \mathcal{L}_{\text{CE}} + \frac{\lambda}{2}\|w\|^2$.
- **L1 (lasso):** $w_i \sim \text{Laplace}(0, 1/\lambda)$. The Laplace prior has heavier tails and a sharp peak at zero, which induces sparsity — the prior assigns finite probability mass to exactly-zero weights.

This Bayesian framing makes the choice of regularizer a choice of prior belief about weight distributions. When training data is limited, the prior dominates — strong regularization is equivalent to saying "I believe weights should be small."

### Implicit Regularization of Gradient Descent

Even without explicit $\lambda$, gradient descent with small step size and initialization near zero exhibits **implicit L2 regularization**. For underdetermined linear systems (more parameters than equations — the typical case in deep learning), gradient descent from $w_0 = 0$ converges to the **minimum-L2-norm solution**:

$$w^* = X^\top (XX^\top)^{-1} y$$

This is the pseudoinverse solution (see [[math-svd]]). No explicit regularization needed — the optimization geometry itself prefers small-norm solutions. This implicit bias is one reason large overparameterized neural networks generalize without requiring aggressive explicit regularization.

### Equivalence to Early Stopping for Linear Models

For linear regression trained with gradient descent, after $t$ steps from $w_0 = 0$:

$$w(t) \approx (X^\top X + \frac{1}{\eta t}I)^{-1}X^\top y$$

This is the ridge regression solution with $\lambda_{\text{eff}} = 1/(\eta t)$. Fewer training steps → larger effective $\lambda$ → stronger regularization. This is the formal equivalence between [[regularization-early-stopping|early stopping]] and weight decay in the linear case (Goodfellow et al., §7.8). In neural networks, the relationship is approximate but qualitatively holds.

---

*See also: [[regularization-dropout]] · [[regularization-early-stopping]] · [[optimizer-adam]] · [[bias-variance-double-descent]] · [[bayesian-inference]]*
