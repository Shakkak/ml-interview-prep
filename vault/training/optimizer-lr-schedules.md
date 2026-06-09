---
title: Learning Rate Schedules
tags: [optimizer, training, learning-rate]
aliases: [LR schedule, warmup, cosine annealing, cyclical LR, LR decay]
difficulty: 2
status: complete
related: [optimizer-adam, optimizer-sgd-momentum, regularization-early-stopping]
---

# Learning Rate Schedules

---

## Fundamental

A fixed learning rate forces a compromise: too large → unstable early training; too small → slow convergence, stuck in poor local minima. Schedules adapt LR over time: explore aggressively early, refine carefully later.

**Warmup:** start small, linearly increase to target over $T_w$ steps:
$$\eta_t = \eta_{\max} \cdot \frac{t}{T_w}, \quad t \leq T_w$$

*Why:* early batches have high loss → large gradients → large random weight updates without warmup. Critical for transformers where Adam's $v_t$ is not yet calibrated (see [[optimizer-adam]]).

**Cosine annealing:** decay LR following a half-cosine from $\eta_{\max}$ to $\eta_{\min}$:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Smooth decay; gentle near the minimum (the cosine is nearly flat at its trough).

**Warmup + cosine decay (standard transformer schedule):**
$$\eta_t = \begin{cases} \eta_{\max} \cdot t / T_w & t \leq T_w \\ \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi(t-T_w)}{T-T_w}\right) & t > T_w \end{cases}$$

Used in [[bert-mlm|BERT]], GPT, [[vision-transformer|ViT]], and most modern transformers. Typical warmup: 4% of total training steps (Devlin et al., 2019).

**Step decay:** multiply LR by $\gamma$ every $k$ epochs:
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

Classic for CNNs: $\gamma = 0.1$ every 30 epochs (original ResNet). Simple and interpretable, but discontinuous drops can destabilize training.

---

## Intermediate

**Numerical example — warmup + cosine:**
Setup: $T_w=100$, $T=1000$, $\eta_{\max}=0.001$, $\eta_{\min}=10^{-5}$.

| Step | Phase | $\eta_t$ |
|------|-------|----------|
| 50 | Warmup | 0.00050 |
| 100 | Warmup (peak) | 0.00100 |
| 500 | Cosine | ~0.000654 (cos(0.444π)/2 + 0.5) |
| 1000 | Cosine (end) | ~$10^{-5}$ |

At the midpoint of the cosine phase ($t=550$), $\eta \approx 0.0005$ — exactly halfway between peak and floor. At the end, $\cos(\pi) = -1$, so $\eta = \eta_{\min}$.

**Cyclical Learning Rates (CLR):** cycle LR linearly between $\eta_{\min}$ and $\eta_{\max}$ over $2c$ steps:
$$\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \max\left(0, 1 - \left|\frac{t}{c} - 2\left\lfloor\frac{t}{2c}\right\rfloor + 1\right|\right)$$

High LR phases help escape saddle points and sharp minima; low LR phases allow fine-grained optimization.

**LR Range Test (Smith, 2017):** train for 100–1000 steps with LR increasing exponentially. Plot loss vs LR: the steepest descent zone → optimal $\eta_{\max}$; divergence zone → too large. Implemented in fastai, PyTorch Lightning, HuggingFace Trainer.

**1-cycle policy:** one full LR cycle for the entire training run — ramp up to $\eta_{\max}$, then cosine down to $\eta_{\min}$, with simultaneous momentum scheduling (high momentum when LR is high, low when LR is low). Achieves strong results in fewer epochs.

**Schedule comparison:**

| Schedule | Shape | Best For |
|----------|-------|----------|
| Constant | Flat | Quick experiments |
| Step decay | Staircase | Classic CNN training (ResNet-style) |
| Cosine | Smooth S-curve | Transformers, general DL |
| Warmup+cosine | Rise then smooth decay | Pre-training (BERT, GPT) |
| Cyclical | Triangular wave | Short training runs, saddle point escape |
| 1cycle | Rise to max, cosine down | Fast convergence (fastai) |

---

## Advanced

**Why LR schedules matter more for Adam than [[optimizer-sgd-momentum|SGD]]:** SGD's update $\theta \leftarrow \theta - \eta g$ scales directly with $\eta$ — the schedule directly controls step size. Adam's update $\theta \leftarrow \theta - \eta \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ is already normalized by the second moment. At initialization, $\hat{m}/\sqrt{\hat{v}} \approx \text{sign}(g)$ (Adam acts like sign gradient descent early). As training progresses, $\hat{v}$ stabilizes and captures gradient scale. The schedule controls *when* Adam commits to a solution (stops exploring). Without decay, Adam keeps taking steps of similar effective size indefinitely — it doesn't naturally converge. Without warmup, early steps are unpredictable because $\hat{v}$ is estimated from only a few gradients.

**Cosine with warm restarts (SGDR, Loshchilov & Hutter, 2017):** periodically reset LR to $\eta_{\max}$ with period doubling (1 → 2 → 4 → 8 epochs). Each restart helps escape the current local minimum. The model snapshots at each restart's minimum can be ensembled for better predictions than any single model — "snapshot ensembles." SGDR with $T_0=10, T_{mult}=2$ produces approximate ensemble benefits at the cost of a single training run.

**Inverse square root schedule:** used in the original transformer (Vaswani et al., 2017):
$$\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot T_w^{-1.5})$$

This combines warmup (linear phase) with an infinite decay ($t^{-0.5}$). Unlike cosine decay, it has no predetermined endpoint — useful for training without a fixed budget. The $d_{\text{model}}^{-0.5}$ factor scales LR inversely with model size, making the optimal LR roughly model-size independent.

**Grokking and the late-training LR regime (Power et al., 2022):** a surprising phenomenon where models first memorize training data (perfect training accuracy, chance-level validation accuracy), then after many more gradient steps "generalize" (near-perfect validation accuracy). This transition is sharp and occurs long after the model has converged to near-zero training loss. Extended training with a decaying LR causes the network to move from a memorizing solution to a generalizing one. This challenges the conventional wisdom that once training loss converges, further training is wasted — LR schedule endpoint matters even for well-converged models.

---

*See also: [[optimizer-adam]] · [[optimizer-sgd-momentum]] · [[regularization-early-stopping]] · [[large-batch-training]] · [[bert-mlm]] · [[vision-transformer]]*
