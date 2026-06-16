---
title: Anchor-Free Object Detection
tags: [anchor-free, fcos, centernet, detection, object-detection]
aliases: [anchor-free detection, FCOS, CenterNet, DETR, anchor-based vs anchor-free]
difficulty: 2
status: complete
depends_on: [feature-pyramid-networks, convolution-math]
related: [feature-pyramid-networks, arch-roi-align, iou-nms, open-vocabulary-detection, attention-mechanism]
---

# Anchor-Free Object Detection

---

## Fundamental

**The problem:** object detection requires predicting bounding boxes. The dominant approach (Faster R-CNN, YOLO) tiles the image with thousands of pre-defined "anchor" boxes at multiple scales and aspect ratios, then predicts offsets from anchors to ground truths. This works, but anchors are a hand-engineered prior: their sizes, ratios, and strides must be tuned per dataset, and the vast majority of anchors are just background noise (extreme class imbalance).

**Intuition behind anchor-free:** if you trust the backbone features enough, why not just predict box edges directly from each feature map pixel? Instead of "which anchor here is closest to this object," ask "what are the distances from this pixel to the object's four edges?" This removes the anchor hyperparameter entirely.

### Anchor-Based vs Anchor-Free

**Anchor-based detectors** (Faster R-CNN, SSD, YOLO v1–v4) tile the feature map with pre-defined anchor boxes at multiple scales and aspect ratios, then predict offsets from these anchors to the ground truth boxes. Problems:
- Hyperparameter-sensitive: anchor sizes, ratios, strides must be manually tuned per dataset
- Many anchors are negative (background) — extreme class imbalance
- Anchors constrain the object shapes the model can detect

**Anchor-free detectors** directly predict box coordinates (or keypoints) from feature map locations without pre-defined anchors.

---

## Intermediate

### FCOS: Fully Convolutional One-Stage Detector

**FCOS (Tian et al., 2019)** treats detection as a per-pixel regression problem:

For each spatial location $(x, y)$ in the feature map:
- **Classification head:** $K$-class probability vector
- **Regression head:** 4 distances to the box edges $(l, t, r, b)$ — left, top, right, bottom from the current location
- **Centerness head:** scalar measuring how close the location is to the object center:

$$\text{centerness} = \sqrt{\frac{\min(l,r)}{\max(l,r)} \cdot \frac{\min(t,b)}{\max(t,b)}}$$

Centerness $\approx 1$ at the box center, $\approx 0$ at corners. Multiplying classification score by centerness suppresses low-quality predictions far from centers, reducing the need for NMS.

**Multi-scale assignment:** small objects assigned to shallow FPN levels (high resolution), large objects to deep levels — prevents ambiguity when multiple objects overlap at the same location.

### CenterNet: Objects as Points

**CenterNet (Zhou et al., 2019)** represents each object as a single point — its center — in a heatmap:

- **Heatmap head:** for each class, a Gaussian blob at the object center: $Y_{xyc} = \exp(-((x-\tilde{x})^2 + (y-\tilde{y})^2)/(2\sigma^2))$
- **Size head:** predicted width and height at each center
- **Offset head:** sub-pixel correction for the discretized center location

**Inference:** find peaks in the heatmap (local maxima), read off size and offset — no NMS on boxes needed (one peak = one detection).

**CenterPoint (3D):** extends CenterNet to 3D point clouds for LiDAR detection in autonomous driving.

---

## Advanced

### DETR: Detection Transformer

**DETR (Carion et al., 2020)** reformulates detection as a set prediction problem — no anchors, no NMS:

1. CNN backbone + transformer encoder processes the image into features
2. A fixed set of $N = 100$ **object queries** (learnable embeddings) attend over image features via cross-attention
3. Each query independently predicts one (class, box) or "no object"
4. **Bipartite matching loss:** at training, use the Hungarian algorithm to optimally match predictions to ground truths (1-to-1), then apply class + box loss only on matched pairs. The **Hungarian algorithm** solves the assignment problem: given $N$ predictions and $M$ ground truths, find the minimum-cost 1-to-1 matching (one prediction per ground truth, allowing unmatched predictions to be marked "no object"). It runs in $O(N^3)$ — tractable for $N = 100$ but not for thousands of anchors.

DETR removes all hand-engineered components (anchors, NMS, region proposals). But it trains slowly (convergence requires 500 epochs vs ~12 for Faster R-CNN) because the attention must learn to specialize each query to a spatial region.

**Deformable DETR:** adds deformable attention (each query attends to a small set of sampled points rather than all spatial locations), reducing convergence to ~50 epochs.

### Anchor-Free vs Anchor-Based Summary

| Property | Anchor-based | Anchor-free (FCOS/CenterNet) | DETR |
|----------|-------------|---------------------------|------|
| Anchor design | Required | None | None |
| NMS | Required | Required (FCOS) / None (CenterNet) | None |
| Training epochs | ~12 | ~12 | ~500 (vanilla) |
| Small objects | Good with small anchors | FPN handles | Weak |
| Dense scenes | Can struggle | FCOS centerness helps | Set limit ($N$) |

## Links

- [[feature-pyramid-networks]] — anchor-free detectors use FPN features at each scale; different pyramid levels handle different object size ranges
- [[convolution-math]] — the per-pixel prediction heads are $1\times 1$ convolutions applied to each FPN level
- [[iou-nms]] — anchor-free detectors still post-process predictions with NMS to remove duplicates; IoU thresholds determine suppression
- [[arch-roi-align]] — transformer-based detectors (DETR) use ROI-like cross-attention to extract features from the backbone; anchor-free methods bypass explicit ROI extraction
- [[open-vocabulary-detection]] — anchor-free heads adapt naturally to open-vocabulary settings; DINO-style detectors use anchor-free predictions with text-conditional classification
- [[attention-mechanism]] — DETR replaces the anchor + NMS pipeline entirely with cross-attention between object queries and image features
