---
title: BYOL and SimSiam
tags: [byol, simsiam, self-supervised, bootstrap, collapse-prevention, negative-free]
aliases: [BYOL, SimSiam, Bootstrap Your Own Latent, self-distillation, non-contrastive SSL]
difficulty: 2
status: complete
related: [contrastive-learning, dino-dinov2, self-supervised-overview, knowledge-distillation, normalization-layers]
---

# BYOL and SimSiam

---

## Fundamental

### The Contrastive Learning Problem

SimCLR and MoCo use **negative pairs** to prevent collapse — the model can't output the same representation for everything because negative pairs are penalized. But:
- Negatives require large batches or memory banks
- Designing good augmentations is critical and domain-specific
- The loss is sensitive to the negative sampling distribution

**BYOL and SimSiam** demonstrate that negatives are not necessary — they prevent collapse by other means.

### BYOL (Bootstrap Your Own Latent)

**BYOL (Grill et al., 2020)** uses two networks:
- **Online network** $f_\theta$ + projector $g_\theta$ + **predictor** $q_\theta$
- **Target network** $f_\xi$ + projector $g_\xi$ (no predictor)

Loss: minimize MSE between the online prediction and the target projection (both $\ell_2$-normalized):

$$\mathcal{L} = \|\bar{q}_\theta(z_\theta) - \bar{z}'_\xi\|^2$$

The target network is a **momentum encoder** (EMA of online): $\xi \leftarrow \tau\xi + (1-\tau)\theta$, $\tau \approx 0.996$.

**No negatives, no contrastive loss.** Two augmented views of the same image → online predicts target's representation.

---

## Intermediate

### Why BYOL Doesn't Collapse

The trivial solution $f \equiv c$ (constant output) would give zero loss — why doesn't this happen?

Several mechanisms prevent collapse:
1. **Asymmetry:** online has a predictor, target does not. The predictor must predict a moving target (EMA) — collapse would require the predictor to constantly adjust to a collapsing target.
2. **Batch normalization:** normalizing features within a batch prevents all representations from becoming identical.
3. **EMA dynamics:** the target updates slowly, providing a stable prediction target. Online network must predict something non-trivial.

**Theoretical analysis:** BYOL implicitly implements a form of whitening or decorrelation through the predictor + BN combination.

### SimSiam (Simple Siamese)

**SimSiam (Chen & He, 2021)** shows EMA is also not necessary — even simpler:
- Two views → same encoder + projector → predictor on one side only (asymmetric)
- Stop gradient on the target branch

$$\mathcal{L} = -\frac{1}{2}\left[\frac{p_1}{\|p_1\|} \cdot \text{sg}(z_2) + \frac{p_2}{\|p_2\|} \cdot \text{sg}(z_1)\right]$$

`sg()` = stop-gradient (detach). Without stop-gradient, collapse occurs immediately.

**Why stop-gradient prevents collapse:** the stop-gradient breaks the gradient flow that would allow the model to find the trivially-constant fixed point. With sg, the model can't optimize both sides simultaneously — it must predict a stable target.

**SimSiam as EM:** stop-gradient implements alternating minimization (EM): E-step fixes target (sg), M-step updates predictor. The iterative algorithm converges to a non-trivial solution.

---

## Advanced

### BYOL vs SimCLR vs SimSiam

| Property | SimCLR | BYOL | SimSiam |
|----------|--------|------|---------|
| Negatives | Yes (large batch) | No | No |
| Momentum encoder | No | Yes ($\tau = 0.996$) | No |
| Predictor | No | Yes (online only) | Yes (one side) |
| Stop gradient | No | No (but EMA is slow) | Yes (explicit) |
| Batch size sensitivity | Very high | Moderate | Low |
| Performance (linear probe) | ~93% (b=8192) | ~74% (b=256 ok) | ~71% (b=256 ok) |

BYOL and SimSiam are more practical for smaller compute budgets — they don't need huge batches.

### Connection to DINO

DINO (see [[dino-dinov2]]) combines insights from BYOL (momentum teacher) and contrastive learning (centering + sharpening instead of explicit negatives). DINO can be seen as a generalization that:
- Uses a momentum teacher like BYOL
- Uses multi-crop + Sinkhorn normalization / centering instead of explicit negatives
- Produces features with strong semantic localization properties

*See also: [[contrastive-learning]] · [[dino-dinov2]] · [[self-supervised-overview]] · [[knowledge-distillation]] · [[normalization-layers]]*
