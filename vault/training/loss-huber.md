---
title: Huber Loss / Smooth L1
tags: [loss, regression, robust, object-detection]
aliases: [Huber loss, smooth L1, robust regression]
difficulty: 2
status: complete
related: [loss-mse, loss-cross-entropy]
---

# Huber Loss / Smooth L1

---

## Fundamental

**The problem with MSE:** MSE penalizes errors quadratically — a 10× larger error produces 100× the loss. A single outlier can dominate the entire gradient update.

**L1 (absolute error)** is robust to outliers (linear growth) but its gradient is constant ($\pm 1$) near zero, making convergence slow for small errors.

**Huber loss combines both:**

$$L_\delta(a) = \begin{cases} \frac{1}{2}a^2 & |a| \leq \delta \\ \delta\left(|a| - \frac{1}{2}\delta\right) & |a| > \delta \end{cases}$$

where $a = y - \hat{y}$ is the residual.

- For **small errors** ($|a| \leq \delta$): quadratic penalty like MSE — smooth, fast convergence.
- For **large errors** ($|a| > \delta$): linear penalty like L1 — outlier-robust.

**Gradient:**
$$\frac{\partial L_\delta}{\partial \hat{y}} = \begin{cases} -(y - \hat{y}) & |y - \hat{y}| \leq \delta \\ -\delta \cdot \text{sign}(y - \hat{y}) & |y - \hat{y}| > \delta \end{cases}$$

The gradient is bounded — outliers can push the gradient by at most $\pm\delta$.

**Smooth L1** (Faster R-CNN variant): Huber with $\delta = 1$:
$$\text{smooth}_{L1}(a) = \begin{cases} 0.5a^2 & |a| < 1 \\ |a| - 0.5 & |a| \geq 1 \end{cases}$$

**As $\delta \to \infty$:** Huber → MSE. As $\delta \to 0$: Huber → scaled L1.

**When to use:** bounding box regression in detection (SSD, Faster R-CNN, YOLO), general regression with suspected outliers, when smoother gradients than L1 are needed.

---

## Intermediate

**Choosing $\delta$:** $\delta$ defines the boundary between "small" and "large" errors. Set $\delta$ at the scale where errors above it should be considered outliers. For bounding box regression in pixels, $\delta = 1$ is typical (offsets normalized by anchor size are usually in this range). For general regression, cross-validate on a log scale.

**Object detection application:** predicting bounding box coordinates $(\Delta x, \Delta y, \Delta w, \Delta h)$ as offsets from anchor boxes. Most predictions will have small offsets (accurate anchor assignment). A few will have large offsets (wrong anchor matches or hard examples). Smooth L1 fits the majority well (quadratic) while not being destroyed by outlier boxes (linear cap).

**Comparison of regression losses:**

| Loss | Small error behavior | Large error behavior | Gradient bound |
|------|-------------------|--------------------|---------------|
| MSE | Quadratic | Quadratic | No |
| L1 | Slow (constant gradient) | Linear | Yes (±1) |
| Huber | Quadratic (fast) | Linear | Yes (±δ) |

**Log-cosh loss** as a smooth alternative: $L(a) = \log(\cosh(a)) \approx \frac{a^2}{2}$ for small $|a|$ and $\approx |a| - \log 2$ for large $|a|$. Fully differentiable everywhere (unlike Huber at $|a| = \delta$), with similar outlier robustness properties. Used in Keras and some regression frameworks as a drop-in replacement.

**Quantile regression loss** (pinball loss): a generalization that allows predicting different quantiles of the output distribution:
$$L_q(a) = \begin{cases} q \cdot a & a \geq 0 \\ (q-1) \cdot a & a < 0 \end{cases}$$
At $q=0.5$: standard L1 (median regression). Predicting multiple quantiles (e.g., 10th, 50th, 90th percentile) gives prediction intervals. Widely used in demand forecasting and weather prediction.

---

## Advanced

**Huber loss and M-estimators in robust statistics:** Huber loss is a specific M-estimator, a class of regression estimators that replace the least-squares criterion $\sum \rho(r_i)$ with a robust $\rho$ function. The Tukey biweight, Hampel, and Welsh estimators provide even stronger outlier resistance than Huber by eventually having zero gradient for very large outliers (redescending estimators). Huber (1964) proved that the Huber M-estimator is the minimax optimal estimator over a contamination neighborhood of the Gaussian — it minimizes the worst-case asymptotic variance for contamination levels up to some fraction $\varepsilon$.

**Dynamic $\delta$ scheduling:** the optimal $\delta$ for bounding box regression changes during training. Early in training, predictions are far from ground truth, so large $\delta$ (more MSE-like) provides stronger gradient signal. Late in training, most errors are small, so small $\delta$ (more L1-like) provides outlier robustness. Adaptive Huber loss (Holland & Welsch, 1977) dynamically adjusts $\delta$ based on the median absolute deviation of residuals — a principled approach used in the deep learning context by Wang et al. (2021) for 3D object detection.

**GIoU / DIoU / CIoU losses for bounding box regression:** Smooth L1 on box coordinates treats $x, y, w, h$ independently, which doesn't directly optimize the final metric (IoU). IoU-based losses directly optimize the overlap ratio:
$$L_{GIoU} = 1 - IoU + \frac{|C \setminus (A \cup B)|}{|C|}$$
where $C$ is the smallest enclosing box. GIoU (Rezatofighi et al., 2019) handles non-overlapping boxes better. DIoU adds a distance term for center point alignment; CIoU adds aspect ratio consistency. These consistently outperform Smooth L1 on COCO by 1–2 AP points and have largely replaced it in modern detectors (YOLOv5, DETR variants).

---

*See also: [[loss-mse]] · [[feature-pyramid-networks]]*
