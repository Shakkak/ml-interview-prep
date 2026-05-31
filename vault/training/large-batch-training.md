---
title: Large-Batch Training
tags: [optimization, training, batch-size, learning-rate, generalization]
aliases: [large batch, batch size, linear scaling rule, warmup, sharp minima, flat minima]
difficulty: 2
status: complete
related: [optimizer-adam, optimizer-sgd-momentum, optimizer-lr-schedules, regularization-dropout, normalization-layers]
---

# Large-Batch Training

---

## Fundamental

| Batch size | Gradient estimate | Steps per epoch | Memory | Speed |
|-----------|-----------------|----------------|--------|-------|
| Small (32–256) | Noisy | Many | Low | Slower per step |
| Large (1k–64k) | Accurate | Few | High | Faster per step |

Gradient variance: $\text{Var}(\hat{g}) = \sigma^2_\text{true} / B$ — doubling $B$ halves gradient variance.

**Linear Scaling Rule** (Goyal et al., 2017): when increasing batch size by $k$, scale the learning rate by $k$:
$$\eta_\text{new} = k \cdot \eta_\text{base}$$

*Why it works:* SGD update on $B$ examples is $\theta \leftarrow \theta - \eta \hat{g}_B$. Summing $k$ such updates approximates one update on $kB$ examples with learning rate $k\eta$:
$$\theta \leftarrow \theta - k\eta \cdot \frac{1}{k}\sum_{i=1}^k \hat{g}_{B_i} \approx \theta - k\eta \cdot g_\text{true}$$

**Learning rate warmup:** at the start of training, weights are random and gradients are large and noisy. Linearly increase LR from 0 to target over the first $T_\text{warmup}$ steps:
$$\eta(t) = \frac{t}{T_\text{warmup}} \cdot \eta_\text{target} \quad \text{for } t \leq T_\text{warmup}$$

Warmup is more critical at large batches because the scaled LR is higher and more dangerous early. Also important for Adam, whose second-moment estimate $v_t$ is unreliable for the first $\sim 50$ steps.

---

## Intermediate

**The generalization gap — flat vs sharp minima:**

A minimum $\theta^*$ is **flat** if $L(\theta^* + \delta) \approx L(\theta^*)$ for large $\|\delta\|$ (small Hessian eigenvalues). It is **sharp** if the loss rises quickly in a small neighborhood (large Hessian eigenvalues).

Keskar et al. (2017) showed large-batch SGD converges to sharp minima while small-batch SGD finds flat minima. Flat minima generalize better: small perturbations of weights (from noise, quantization, or distribution shift) do not hurt them, while sharp minima are sensitive to any deviation from the exact optimum.

*Why small-batch SGD finds flat minima:* gradient noise from small batches acts as implicit regularization. The noise kicks the optimizer out of narrow sharp valleys, which have small basins of attraction, and allows it to settle in wide flat basins. *Why large-batch SGD finds sharp minima:* accurate gradients allow precise convergence into the nearest local minimum without noise to escape it.

Typical generalization gap: large-batch (4k–32k) ImageNet top-1 accuracy is 1–2% below small-batch (256) with the same total compute.

**Techniques to close the generalization gap:**

*Ghost Batch Normalization:* compute BatchNorm statistics on sub-batches of size $B_\text{ghost}$ even when the physical batch is $B \gg B_\text{ghost}$. Preserves the noise in BN statistics that small-batch training provides.

*LARS (Layer-wise Adaptive Rate Scaling):* different layers have very different gradient magnitudes. LARS adapts the learning rate per layer:
$$\eta_\ell = \eta \cdot \frac{\|\theta_\ell\|}{\|\nabla_\ell L\|}$$
This keeps the ratio of update to weight magnitude consistent across layers, enabling stable training at batch sizes 32k+.

*LAMB (Layer-wise Adaptive Moments):* LARS + Adam-style second-moment normalization. Used to train BERT with batch sizes up to 65536 (You et al., 2020), reducing BERT pre-training from 3 days to 76 minutes.

*Sharpness-Aware Minimization (SAM):* instead of minimizing $L(\theta)$, minimize the worst-case loss in a neighborhood:
$$L_\text{SAM}(\theta) = \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$
The inner maximization finds the sharpest perturbation; the outer minimization flattens it. Requires two forward passes per step but significantly improves generalization for large batches.

---

## Advanced

**SGD noise as a stochastic differential equation:** small-batch gradient noise can be modeled as:
$$d\theta = -g\, dt + \sqrt{\eta/B}\, C^{1/2}\, dW$$
where $C$ is the gradient covariance matrix and $W$ is Brownian motion. The noise term drives the system toward lower-curvature regions — equivalently, toward higher-entropy flat minima. The noise scale $\eta/B$ is the fundamental quantity: increasing $\eta$ or decreasing $B$ increases noise. This is why the linear scaling rule $\eta \propto B$ *conserves* the noise scale, explaining why it works mathematically.

**The linear scaling rule breaks at the critical batch size $B^*$:** Hoffer et al. (2017) and Shallue et al. (2019) showed empirically that training efficiency (test accuracy per example processed) degrades for $B > B^*$, where $B^* \approx 512$–$8192$ for typical vision tasks. Below $B^*$, doubling $B$ and $\eta$ together preserves training efficiency. Above $B^*$, the gradient noise drops too low, the model converges to progressively sharper minima, and the generalization penalty exceeds the compute savings. This defines the practical frontier of data parallelism.

**Gradient accumulation as budget-constrained large-batch training:** when GPU memory limits the physical batch size, gradient accumulation performs $k$ micro-batch forward-backward passes before one optimizer step. This achieves an effective batch of $k \times B_\text{micro}$ at the cost of $k$ serial forward passes (no throughput gain, but memory-efficient). The math is identical to training on the full batch; BN statistics require care (use SyncBN or LayerNorm).

**SAM's connection to PAC-Bayes sharpness:** SAM (Foret et al., 2021) is motivated by PAC-Bayes generalization bounds of the form $\text{gen-error} \lesssim \mathbb{E}_{|\epsilon|<\rho}[L(\theta+\epsilon)] - L(\theta_\text{train})$, which bound generalization error by the maximum loss in a neighborhood. Minimizing this bound directly is equivalent to minimizing the sharpest loss in the neighborhood — exactly what SAM does. The choice of perturbation norm $\rho$ controls the scale of the flatness penalty; values around $0.05$–$0.1$ work across diverse tasks.

**Practical guidelines:**

| Batch size | Guidance |
|-----------|----------|
| 32–512 | Default; no special tricks needed |
| 512–4096 | Linear scaling rule + warmup |
| 4k–32k | LARS/LAMB + warmup + ghost BN |
| 32k+ | SAM + significant engineering; research territory |
| Fine-tuning | Use small batches (32–256); generalization gap is minor |

---

*See also: [[optimizer-sgd-momentum]] · [[optimizer-adam]] · [[optimizer-lr-schedules]] · [[normalization-layers]]*
