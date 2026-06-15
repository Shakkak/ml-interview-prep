---
title: Evaluation Metrics — Complete Guide
tags: [evaluation, metrics, classification, detection, segmentation, calibration, map, auc, f1]
aliases: [precision recall, mAP, AUC, ROC, F1, calibration, IoU, mIoU, ECE]
difficulty: 3
status: complete
related: [feature-pyramid-networks, bayesian-inference, bias-variance-double-descent, model-calibration]
---

# Evaluation Metrics — Complete Guide

---

## Fundamental

How do you know if a model is good? Accuracy alone is misleading — a model that predicts "no cancer" for every patient achieves 99% accuracy on a dataset that is 99% healthy. **Evaluation metrics** capture the right tradeoffs for each problem type: precision and recall for classification with class imbalance, mAP for object detection, mIoU for segmentation, and calibration for probabilistic models. Choosing the wrong metric is one of the most common sources of misleading evaluation results.

For binary classification with threshold $t$ (predict positive if score ≥ $t$):

|  | Predicted Positive | Predicted Negative |
|--|:-:|:-:|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

$$\text{Precision} = \frac{TP}{TP+FP} \quad \text{Recall} = \frac{TP}{TP+FN} \quad \text{F1} = \frac{2 \cdot P \cdot R}{P + R}$$

**F1 uses [[math-means|harmonic mean]]** — not arithmetic. Arithmetic mean of P=1.0, R=0.01 is 0.505 (looks fine). Harmonic mean is 0.020 (correctly penalizes the near-zero component). The harmonic mean is dominated by the smaller value: $\text{HM}(a, b) \leq \min(a, b)$.

> [!tip] Why harmonic mean — not arithmetic — for F1
> For P = 1.0, R = 0.01:
> - Arithmetic mean = (1.0 + 0.01)/2 = **0.505** — hides the near-zero recall
> - Harmonic mean = 2/(1/1.0 + 1/0.01) = 2/101 ≈ **0.020** — correctly low
>
> Key property: $\text{HM}(a,b) \leq \min(a,b)$ — a near-zero component drags it all the way down.
> F1 only stays high when *both* P and R are high, not just one. See math-means for the full picture.

**$F_\beta$ score:** $F_\beta = (1+\beta^2)\frac{P \cdot R}{\beta^2 P + R}$. $\beta > 1$ weights recall more (FN is costly: cancer screening). $\beta < 1$ weights precision more (FP is costly: spam filter).

### IoU — Spatial Prediction Quality

For bounding boxes and segmentation masks:
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

For axis-aligned boxes: intersection = max(0, min(x2) − max(x1)) × max(0, min(y2) − max(y1)). Union = area(A) + area(B) − intersection.

**mIoU** (semantic segmentation): compute IoU per class, average over classes. Penalizes false positives and false negatives equally per class — much more informative than pixel accuracy on imbalanced classes.

---

## Intermediate

### The Precision-Recall Curve and AP

Varying the decision threshold $t$ from high to low traces a curve in (Recall, Precision) space. High $t$: very few positives predicted → high precision, low recall. Low $t$: nearly everything predicted positive → low precision, high recall.

**Average Precision (AP)** — area under the P-R curve:

*All-points interpolation (COCO):*
$$AP = \sum_n (R_n - R_{n-1}) P_n$$
summed over every threshold where recall changes.

*11-point interpolation (VOC):*
$$AP = \frac{1}{11}\sum_{r \in \{0, 0.1, \ldots, 1.0\}} \max_{r' \geq r} P(r')$$

**mAP** (mean AP) = average AP over all object classes.

### mAP Worked Example

5 ground-truth "cat" boxes; detector outputs 6 predictions sorted by confidence:

| Rank | Match? | Cumul TP | Cumul FP | Prec | Recall |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | TP (IoU=0.82) | 1 | 0 | 1.00 | 0.20 |
| 2 | TP (IoU=0.74) | 2 | 0 | 1.00 | 0.40 |
| 3 | TP (IoU=0.91) | 3 | 0 | 1.00 | 0.60 |
| 4 | FP (IoU=0.32) | 3 | 1 | 0.75 | 0.60 |
| 5 | TP (IoU=0.61) | 4 | 1 | 0.80 | 0.80 |
| 6 | FP (duplicate) | 4 | 2 | 0.67 | 0.80 |

$$AP = (0.2)(1.0) + (0.2)(1.0) + (0.2)(1.0) + (0)(0.75) + (0.2)(0.80) = 0.76$$

**COCO mAP variants:**
- AP@[.50:.05:.95]: primary metric — average over IoU thresholds {0.50, 0.55, …, 0.95}
- AP50: lenient (any reasonable detection counts)
- AP75: strict (requires tight localization)
- AP$_S$, AP$_M$, AP$_L$: small (<32²px), medium, large objects

### ROC / AUC

As threshold varies, plot (FPR = FP/(FP+TN), TPR = Recall). Diagonal = random classifier (AUC = 0.5). AUC has a probabilistic interpretation:

$$\text{AUC} = P(\text{score}(x^+) > \text{score}(x^-))$$

**P-R vs ROC for imbalanced data:** on a dataset with 1% positives, a classifier can achieve FPR=0.09, TPR=0.8 — the ROC point (0.09, 0.8) looks excellent. But precision = TP/(TP+FP) = 0.8P/(0.8P + 0.09×99P) ≈ 8% — only 1 in 12 predicted positives is real. The P-R curve reveals this; the ROC curve hides it because it normalizes FP by TN (many negatives make FPR look small even when absolute FPs are large).

---

## Advanced

### Calibration: When Probabilities Mean What They Say

A model outputting $p(y=1 \mid x) = 0.9$ should be correct 90% of the time among all such predictions. Modern deep networks are typically **overconfident**: they output 0.9 when actual accuracy is 0.7.

**Reliability diagram:** bin predictions by confidence; plot (mean confidence, fraction correct) per bin. A perfectly [[model-calibration|calibrated]] model produces points on the diagonal.

**Expected Calibration Error:**
$$ECE = \sum_{b=1}^B \frac{|B_b|}{n}\left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

**Temperature scaling** — simplest fix: divide all logits by $T > 1$ before softmax. $T$ is a single scalar optimized on a validation set by minimizing NLL. It does not change accuracy (argmax is unchanged) but softens the probability distribution. Guo et al. (2017) showed temperature scaling is remarkably effective for modern ResNets and DenseNets.

### Metrics for Generative Models

| Task | Metric | What it measures |
|---|---|---|
| Image generation | FID (Fréchet Inception Distance) | [[loss-kl-divergence\|KL divergence]] between real/fake feature distributions |
| Image quality | SSIM | Structural similarity (luminance × contrast × structure) |
| Text generation | BLEU | $n$-gram overlap between hypothesis and references |
| Text generation | [[bert-mlm\|BERTScore]] | Soft token matching using contextual embeddings |
| Retrieval | nDCG | Normalized discounted cumulative gain (position-weighted recall) |

**FID:** compute Inception-v3 features for real and generated images; fit [[distributions-gaussian|Gaussians]] $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$; FID = $\|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$. Lower is better. Requires ~50k samples for stable estimates. FID is known to correlate with human judgments of image quality better than Inception Score.

> [!tip] What the FID formula is computing
> FID is the squared **Fréchet distance** (Wasserstein-2 distance) between two multivariate Gaussians.
> $\|\mu_r - \mu_g\|^2$ measures mean shift; $\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ measures covariance mismatch.
> It equals zero iff the two Gaussians are identical — i.e., the generated images' feature distribution perfectly matches real images.

### Choosing the Right Metric

| Task | Do | Don't |
|---|---|---|
| Imbalanced classification | Macro F1, AUC-PR | Accuracy |
| Object detection | COCO mAP@[.5:.95] | AP50 alone |
| Semantic segmentation | mIoU | Pixel accuracy |
| Calibration | ECE, reliability diagram | Accuracy alone |
| Regression | RMSE + MAE (both) | R² alone (hides scale) |

Report a **learning curve** (performance vs. labeled set size) for active learning experiments — not a single accuracy number. A single number at the end of training doesn't show how efficiently the method uses labels.

---

*See also: [[feature-pyramid-networks]] · [[bayesian-inference]] · [[model-calibration]] · [[bias-variance-double-descent]] · [[math-means]] · [[distributions-gaussian]] · [[loss-kl-divergence]]*
