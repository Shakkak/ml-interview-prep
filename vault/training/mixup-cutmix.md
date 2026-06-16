---
title: Mixup and CutMix
tags: [mixup, cutmix, data-augmentation, label-smoothing, interpolation-regularization]
aliases: [Mixup, CutMix, manifold mixup, FMix, label interpolation]
difficulty: 1
status: complete
related: [data-augmentation, regularization-label-smoothing, vit-training-recipe, bias-variance-double-descent, loss-cross-entropy]
depends_on: [data-augmentation, loss-cross-entropy, regularization-label-smoothing]
---

# Mixup and CutMix

---

## Fundamental

### Mixup

**Mixup (Zhang et al., 2018)** trains on convex combinations of training examples and their labels:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ with $\alpha \in [0.2, 1.0]$.

**Effect:**
- Creates "soft" training examples between classes
- Forces the model to produce linear interpolations of predictions in the input space
- Equivalent to data augmentation that encourages linear behavior between training points

**Intuition:** the model should be "uncertain" between two classes when shown a blend of two examples. This discourages overconfident, sharp decision boundaries.

### CutMix

**CutMix (Yun et al., 2019)** replaces a rectangular region of one image with the corresponding region from another, with labels mixed proportionally to the pixel area:

$$\tilde{x} = M \odot x_i + (1-M) \odot x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

where $M$ is a binary mask (1 = keep from $x_i$, 0 = replace from $x_j$), $\lambda = $ area of kept region.

**Advantage over Mixup:** preserves spatial coherence — the pasted regions are still recognizable objects, not ghostly overlaps. The model must locate and classify object parts even when they're incomplete, improving localization.

---

## Intermediate

### Training Dynamics

**Mixup prevents overconfidence:** training with soft labels $\tilde{y}$ is equivalent to a form of label smoothing, but input-adaptive. The model cannot produce probability 1 on any training point (since $\tilde{y}$ is never one-hot).

**Calibration:** models trained with Mixup tend to be better calibrated (predicted probabilities match observed frequencies) because they never train with hard targets.

**Loss surface smoothing:** Mixup implicitly regularizes the loss landscape — regions between training points are constrained. This reduces sharp loss surface features that correlate with poor generalization.

### Variants

**Manifold Mixup (Verma et al., 2019):** apply mixing in the hidden representation space (intermediate layers) rather than input space. Creates smoother class boundaries in representation space. Generally stronger regularization than input-space Mixup.

**FMix (Harris et al., 2020):** use Fourier-space masks instead of rectangles for CutMix-style mixing. Produces more varied cut shapes.

**SmoothMix:** apply smooth blending at the cut boundary (soft edge rather than hard rectangle).

**GridMix:** split image into grid cells and randomly mix cells from two images.

---

## Advanced

### Mixup in the ViT Training Recipe

Mixup and CutMix are essential components of the standard ViT training recipe (see [[vit-training-recipe]]). The DeiT paper showed that without strong augmentation (including Mixup/CutMix), ViTs underfit dramatically on ImageNet without JFT-scale pretraining.

**Typical recipe:** CutMix probability 1.0 (always active), $\alpha = 1.0$ for both Mixup and CutMix, randomly switch between the two. Combined with RandAugment, label smoothing 0.1, and stochastic depth.

### Optimal Transport Interpretation

Mixup can be viewed as sampling along the straight-line interpolant between two data points. Under the manifold hypothesis, this interpolant may not stay on the data manifold — it passes through "off-manifold" space.

**Optimal transport Mixup:** mix along the geodesic of the learned data distribution (optimal transport map), staying on the data manifold. Theoretically cleaner but computationally expensive.

### Mixup for NLP

For text, Mixup can be applied in embedding space:
- Embed both sequences, mix embeddings, decode (for generation) or classify (for classification)
- **Token-level Mixup:** mix at each token position independently
- Useful for data augmentation in low-resource settings

Sequence-level mixing is complicated by variable length — padding alignment and attention masks must be handled carefully.

## Links

- [[data-augmentation]] — Mixup and CutMix are data augmentation methods that operate in the sample space; unlike geometric augmentations, they create entirely new training examples by interpolating
- [[loss-cross-entropy]] — Mixup requires soft labels $\tilde{y} = \lambda y_a + (1-\lambda)y_b$; the cross-entropy is computed against these mixed targets: $H(\tilde{y}, f(x)) = \lambda H(y_a, f(x)) + (1-\lambda)H(y_b, f(x))$
- [[regularization-label-smoothing]] — Mixup's soft labels implicitly smooth the training targets; label smoothing is a special case of Mixup where the second label is the uniform distribution
- [[vit-training-recipe]] — DeiT's training recipe combines Mixup, CutMix, RandAugment, and stochastic depth; these augmentations together allow ViT to train without more data than ResNet
