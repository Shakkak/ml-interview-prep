---
title: Early Stopping
tags: [regularization, training, overfitting]
aliases: [early stopping, validation loss, patience]
difficulty: 1
status: complete
related: [regularization-dropout, regularization-weight-decay, optimizer-lr-schedules, bias-variance-double-descent, spectral-bias]
---

# Early Stopping

---

## Fundamental

Monitor validation loss during training. Stop when validation loss stops improving — even if training loss keeps decreasing. Restore the checkpoint from the best validation epoch.

**The algorithm:**

```
best_val_loss = ∞
patience_counter = 0
PATIENCE = 10

for epoch in training:
    train_one_epoch()
    val_loss = evaluate(val_set)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break

load_checkpoint(best_model)
```

**Numerical example:**

| Epoch | Train Loss | Val Loss | Action |
|---|---|---|---|
| 30 | 0.22 | 0.50 | new best — save |
| 35 | 0.18 | 0.51 | patience = 1 |
| 45 | 0.12 | 0.55 | patience = 3 |
| 70 | 0.08 | 0.62 | stop → restore epoch 30 |

Training loss keeps falling; validation diverges from epoch 30. With patience = 10, stop at epoch 70 and restore the epoch-30 checkpoint.

**Key parameters:**
- **Patience:** epochs to wait after last improvement. Too small → stops prematurely (high bias). Too large → wastes compute. Typical: 5–20 for CV; 1–3 for LLM fine-tuning.
- **min_delta:** minimum change to count as improvement — prevents stopping on plateau noise.
- **Monitor metric:** validation loss is default; use mAP or F1 when accuracy is the task objective.

---

## Intermediate

### Bias-Variance View

Early in training: high bias (underfits). Late in training: high variance (overfits). Early stopping finds the sweet spot.

This is not just heuristic — for linear models with gradient descent, early stopping and L2 regularization select the **same effective parameter subspace** (Goodfellow et al., *Deep Learning* §7.8). The number of gradient steps is the implicit regularization budget, analogous to $1/\lambda$.

### Comparison with Explicit Regularization

| | Early Stopping | L2 / Dropout |
|---|---|---|
| Mechanism | Stops optimization | Modifies loss or architecture |
| Cost | Free (monitoring only) | Hyperparameter $\lambda$ to tune |
| Val set quality | Noisy signal if val is small | More stable |
| When to combine | Always — they are complementary |

Weight decay + dropout + early stopping are not redundant; each attacks overfitting from a different angle.

### Interaction with Learning Rate Schedules

With **cosine annealing** ending at step $T$: LR decays to $\eta_{\min}$ — training naturally slows near the end. Early stopping is less critical but checkpoint selection still matters.

With **constant LR**: validation loss divergence is the primary guard. Monitor closely.

Common transformer practice: fixed warmup + cosine schedule for a preset budget; select the best checkpoint by validation metric rather than stopping early.

---

## Advanced

### Equivalence to L2 Regularization for Linear Models

For a linear model trained with gradient descent from $w_0 = 0$, after $t$ steps the solution lies in the span of the first $t$ gradient directions — a low-dimensional subspace. This is equivalent to the L2-regularized solution:

$$w_{\text{early-stop}}(t) \approx (X^\top X + \frac{1}{\eta t} I)^{-1} X^\top y$$

where $\lambda_{\text{eff}} = 1/(\eta t)$. More training steps → smaller effective regularization → more overfitting. This is the formal sense in which "training duration = regularization strength."

### Connection to Spectral Bias

Early stopping is also a **frequency-domain low-pass filter** (see [[spectral-bias]]). Because gradient descent learns low-frequency components of the target first, stopping before full convergence means high-frequency noise components (which fit training-specific irregularities) are never learned. The checkpoint captures the smooth low-frequency signal. This gives a theoretical account of *why* early stopping improves generalization beyond the bias-variance framing.

### Data Contamination Risk

If you tune patience, $\lambda$, or architecture choices by running many early-stopped experiments on the **same** validation set, you overfit the validation set. The reported validation loss becomes optimistically biased — it no longer estimates test performance.

Fix:
1. Use a proper held-out test set for final evaluation.
2. Use nested cross-validation for hyperparameter search.
3. Use a separate calibration split distinct from the hyperparameter-tuning split.

---

*See also: [[regularization-dropout]] · [[regularization-weight-decay]] · [[optimizer-lr-schedules]] · [[bias-variance-double-descent]] · [[spectral-bias]]*
