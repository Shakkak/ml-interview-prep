---
title: Cosine Annealing and LR Schedules
tags: [cosine-annealing, lr-schedule, warm-restarts, sgdr, snapshot-ensembles]
aliases: [cosine annealing, SGDR, warm restarts, cosine decay, OneCycleLR, snapshot ensemble]
difficulty: 1
status: complete
related: [optimizer-lr-schedules, optimizer-adam, optimizer-sgd-momentum, large-batch-training, loss-landscape]
---

# Cosine Annealing and LR Schedules

---

## Fundamental

A constant learning rate forces a compromise: high enough to make progress early, but low enough to converge precisely later — and no single value does both well. **Learning rate schedules** solve this by varying the rate during training. Cosine annealing is the most widely used schedule: it starts high for fast initial progress, decays smoothly following a cosine curve, and reaches near-zero at the end for fine-grained convergence. It consistently outperforms step-decay and is the default in modern LLM and ViT training.

### Cosine Decay

**Cosine annealing** decays the learning rate from $\eta_\text{max}$ to $\eta_\text{min}$ following a cosine curve over $T$ steps:

$$\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

Properties:
- Smooth decay (no sudden drops like step decay)
- Starts fast (large $\eta$ for long time), ends slow
- Spends most time near $\eta_\text{min}$ — final convergence is careful

Compared to **linear decay** ($\eta_t = \eta_\text{max}(1 - t/T)$): cosine spends more time at higher LR early (faster progress) and more time at lower LR late (better convergence). In practice, cosine slightly outperforms linear on most tasks.

---

## Intermediate

### SGDR: Warm Restarts

**SGDR (Loshchilov & Hutter, 2017):** periodically restart the learning rate to $\eta_\text{max}$ at intervals $T_i$, with intervals growing over time:

$$T_i = T_0 \cdot T_\text{mult}^i$$

For $T_0 = 10$, $T_\text{mult} = 2$: restart at steps 10, 30, 70, 150 (each cycle twice as long).

**Why restarts help:** the model escapes the current local minimum by temporarily receiving large LR updates, potentially finding a flatter, better-generalizing minimum. Each restart is an "escape attempt."

**Snapshot ensembles:** save the model at each restart minimum. The saved models are at diverse loss basin points — ensemble them for free accuracy improvement without extra training cost.

### OneCycleLR

**OneCycleLR (Smith & Touvron, 2019):** one cycle of increase then decrease, designed for super-convergence:

1. **Warmup:** LR increases from $\eta_\text{min}$ to $\eta_\text{max}$ over first 30% of steps (linear or cosine)
2. **Decay:** LR decreases from $\eta_\text{max}$ to $\eta_\text{min}/100$ over remaining 70% (cosine)

**Momentum** is simultaneously annealed in reverse (high momentum at low LR, low momentum at high LR). This allows aggressive LR ($\eta_\text{max}$ can be 10–100× higher than conventional) with stable training.

OneCycleLR often matches training-to-convergence quality in 5–10× fewer epochs. Used as default in fast.ai.

### Warmup

**Learning rate warmup:** start at a small LR (e.g., 1e-6) and linearly increase to target LR over the first $W$ steps (typically 1–10% of total training).

**Why warmup:** early in training, gradients are large and the second-moment estimate in Adam is poorly initialized. A large initial LR causes instability. Warmup allows $v_t$ to stabilize before LR reaches full strength.

Critical for transformers: without warmup, training often diverges in the first few hundred steps.

---

## Advanced

### Cooldown and Restart Interaction

For transformers trained long-term (e.g., LLMs), the standard recipe is:
1. Linear or cosine warmup (1–2K steps)
2. Cosine decay over the full training run to 10% of peak LR
3. Optional final cooldown to nearly zero LR for the last 5% of steps

This single-cycle approach (no restarts) works well because LLM training runs are too long to benefit from multiple restart cycles, and the dataset is shuffled, ensuring diverse gradients throughout.

### Cyclical Learning Rates

**CLR (Smith, 2017):** triangular oscillation between $\eta_\text{min}$ and $\eta_\text{max}$ with fixed period $2T$:
$$\eta_t = \eta_\text{min} + (\eta_\text{max} - \eta_\text{min})\max(0, 1 - |\text{cycle} - 1|)$$

Simpler than SGDR, no growing periods. The periodic high-LR phases provide implicit exploration of the loss landscape.

**Finding the right LR range (LR range test):** train for a few epochs while linearly increasing LR from very small to very large; plot loss vs LR; choose $\eta_\text{max}$ where loss starts increasing, $\eta_\text{min}$ about 10× smaller.

*See also: [[optimizer-lr-schedules]] · [[optimizer-adam]] · [[large-batch-training]] · [[loss-landscape]]*
