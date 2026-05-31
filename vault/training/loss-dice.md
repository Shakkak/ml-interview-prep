---
title: Dice Loss
tags: [loss, segmentation, class-imbalance]
aliases: [Dice loss, F1 loss, soft Dice, Sørensen–Dice]
difficulty: 2
status: complete
related: [loss-focal, loss-cross-entropy, evaluation-metrics-guide]
---

# Dice Loss

---

## Fundamental

The Dice coefficient (Sørensen–Dice) measures overlap between two sets $A$ and $B$:
$$\text{Dice}(A, B) = \frac{2|A \cap B|}{|A| + |B|}$$

Range: 0 (no overlap) to 1 (perfect overlap). Equivalent to the F1 score for binary classification.

**Why not cross-entropy for segmentation?** Pixel-wise cross-entropy treats each pixel independently and equally. For a segmentation map where 99% of pixels are background, cross-entropy is dominated by background pixels. A model that labels everything as background gets 99% pixel accuracy and near-zero cross-entropy loss — completely useless.

Dice evaluates the overlap of the entire mask directly:

*Example:* 10,000-pixel image, 100 foreground pixels (1% prevalence).
- Model A predicts all background: CE ≈ 0 (near-perfect per-pixel accuracy), Dice = 0 (no overlap).
- Model B correctly predicts foreground with some FP: CE slightly higher, Dice ≈ 0.8.
- Dice selects Model B; cross-entropy selects Model A.

**Soft Dice loss** for predicted probabilities $\hat{p} \in [0,1]^N$ and binary ground truth $y \in \{0,1\}^N$:
$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_i y_i \hat{p}_i}{\sum_i y_i + \sum_i \hat{p}_i + \epsilon}$$

The $\epsilon$ (typically $10^{-6}$) prevents division by zero for empty masks. Minimizing Dice loss maximizes the Dice coefficient.

---

## Intermediate

**Gradient behavior:** the gradient of Dice loss is:
$$\frac{\partial \mathcal{L}_{Dice}}{\partial \hat{p}_i} = -\frac{2y_i(\sum_j y_j + \sum_j \hat{p}_j) - 2\sum_j y_j \hat{p}_j}{(\sum_j y_j + \sum_j \hat{p}_j)^2}$$

Key property: **gradient is normalized by total predicted/true volume**. A prediction error on a large object has the same gradient scale as an error on a tiny object. Cross-entropy gradient on a 1-pixel object is $10,000\times$ smaller than on a 10,000-pixel object (just from counting pixels) — Dice avoids this.

**Tversky loss:** a generalization that separately weights false positives (FP) and false negatives (FN):
$$T(\alpha, \beta) = \frac{\sum y\hat{p}}{\sum y\hat{p} + \alpha \sum(1-y)\hat{p} + \beta \sum y(1-\hat{p})}$$
With $\alpha = \beta = 0.5$: standard Dice. With $\beta > \alpha$: penalizes FN more heavily (higher recall). Used in medical imaging where missing a lesion (FN) is more dangerous than a false alarm (FP).

**Generalized Dice Loss** (Sudre et al., 2017): weights each class by inverse squared volume $w_c = 1/(\sum_i y_{ic})^2$, equalizing gradient contribution across small and large structures. Essential for multi-class segmentation with highly variable object sizes (e.g., liver vs. gallbladder vs. bile duct).

**Combined loss — cross-entropy + Dice:** in practice, the most effective approach combines both:
$$\mathcal{L} = \lambda \mathcal{L}_{CE} + (1-\lambda)\mathcal{L}_{Dice}$$
Cross-entropy provides stable pixel-level gradient signal everywhere; Dice enforces global mask overlap. $\lambda \in [0.4, 0.6]$ is common; tuned on the validation set.

**When to use Dice alone vs combined:**
- Extreme imbalance (tumor < 1% of volume): Dice alone, or Dice + weighted CE
- Mild imbalance: combined loss
- Multi-class with variable size classes: Generalized Dice

---

## Advanced

**Focal Dice loss** (Wang et al., 2020): combines focal modulation with Dice structure by applying a $(1 - \text{Dice})^\gamma$ modulating factor to down-weight easy (well-segmented) samples and focus training on hard cases. Extends focal loss ideas from detection to segmentation by operating at the image level rather than the pixel level.

**Boundary-aware Dice loss:** standard Dice treats all foreground pixels equally. For tasks where boundary precision matters (surgical instrument segmentation, cell boundary detection), boundary-weighted variants apply higher loss near the predicted or ground-truth boundary: $\mathcal{L}_{BD} = \mathcal{L}_{Dice} + \lambda \mathcal{L}_{Dice,boundary}$ where the boundary loss is computed only on a dilated boundary mask. Reduces Hausdorff distance without sacrificing volumetric Dice.

**Clough et al. (2020) — topological Dice:** standard Dice is topology-blind. A segmentation can achieve high Dice while having the wrong number of connected components (e.g., missing a hole in a donut structure). Topological loss functions penalize differences in Betti numbers (number of connected components, holes, voids) computed from persistent homology, preserving topological structure in the segmentation at the cost of higher computational complexity.

**Dice loss instability with small structures and empty masks:** when the ground truth mask is empty ($\sum_i y_i = 0$), the Dice loss is undefined without $\epsilon$. Even with $\epsilon$, the gradient signal for empty-mask images is weak and may cause training instability. Practical solutions: (1) stratified batching (always include at least one positive example per batch); (2) use cross-entropy on empty-mask images and Dice only on non-empty ones; (3) increase $\epsilon$ during early training when predictions are unstable.

---

*See also: [[loss-focal]] · [[loss-cross-entropy]] · [[evaluation-metrics-guide]]*
