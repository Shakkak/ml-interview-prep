---
title: IoU, NMS, and Bounding Box Operations
tags: [iou, nms, non-maximum-suppression, bounding-box, detection]
aliases: [IoU, intersection over union, NMS, non-maximum suppression, soft-NMS, GIoU, DIoU]
difficulty: 1
status: complete
depends_on: [feature-pyramid-networks, statistical-inference-mle]
related: [arch-roi-align, feature-pyramid-networks, open-vocabulary-detection, evaluation-metrics-guide]
---

# IoU, NMS, and Bounding Box Operations

---

## Fundamental

Object detectors produce thousands of candidate boxes — far more than the number of actual objects. Two core operations decide which boxes to keep: **IoU** measures how much two boxes overlap (used for matching predictions to ground truth and for filtering duplicates), and **NMS** suppresses duplicate detections of the same object.

**Intuition for NMS:** if two boxes both claim to detect the same object, keep the more confident one and discard anything that heavily overlaps it. Repeat until no two remaining boxes overlap above the threshold — one box per object.

### Intersection over Union (IoU)

For two bounding boxes $A$ and $B$:

$$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

IoU ranges from 0 (no overlap) to 1 (perfect overlap). It is the standard metric for bounding box similarity in object detection.

**Common thresholds:**
- IoU $\geq 0.5$: standard match threshold (PASCAL VOC)
- IoU $\geq 0.75$: strict match (COCO AP@0.75)
- IoU 0.5:0.95: mean across thresholds (COCO mAP)

**Efficient computation:** represent boxes as $(x_1, y_1, x_2, y_2)$ (top-left, bottom-right). Intersection: $\max(x_1^A, x_1^B), \max(y_1^A, y_1^B), \min(x_2^A, x_2^B), \min(y_2^A, y_2^B)$.

---

## Intermediate

### Non-Maximum Suppression (NMS)

Object detectors produce many overlapping predictions. **NMS** keeps only the most confident detection per object.

**Standard NMS algorithm:**
1. Sort all detections by confidence score (descending)
2. Select the detection with highest score → output it
3. Remove all other detections with IoU $\geq \theta_\text{NMS}$ (typically 0.5)
4. Repeat from step 2 on remaining detections

**Greedy NMS is fast** ($O(n \log n)$ sort + $O(n^2)$ scan) but has failure modes:
- Two overlapping objects of the same class → one is suppressed
- Very dense crowds → many missed detections

**Soft-NMS (Bodla et al., 2017):** instead of hard removal, decay the score of overlapping detections:
$$s_i \leftarrow s_i \cdot f(\text{IoU}(x_k, x_i))$$
where $f$ is linear ($1 - \text{IoU}$) or Gaussian decay. Overlapping detections get lower scores and may survive if sufficiently confident.

### Generalized IoU (GIoU) and DIoU

Standard IoU has a gradient problem: when two boxes don't overlap at all, IoU = 0 regardless of how far apart they are (no gradient signal for regression training).

**GIoU (Rezatofighi et al., 2019):**
$$\text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$
where $C$ is the smallest enclosing box. GIoU $\in [-1, 1]$; when boxes don't overlap, it measures relative distance to the enclosing box — provides gradient even for non-overlapping boxes.

**DIoU (Distance IoU):** adds a term penalizing the center distance between boxes:
$$\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}$$
where $\rho$ is center distance, $c$ is diagonal of enclosing box. Converges faster by directly minimizing center mismatch.

**CIoU (Complete IoU):** adds aspect ratio consistency to DIoU. Used as the training loss in YOLOv5/v8.

---

## Advanced

### Anchor Matching Strategies

During training, each ground truth box must be assigned to one or more anchor boxes (or predicted boxes) to compute the regression and classification losses.

**IoU-based matching (Faster R-CNN):**
- Anchors with IoU $\geq 0.7$ → positive (foreground)
- Anchors with IoU $\leq 0.3$ → negative (background)
- Anchors between → ignored

**ATSS (Adaptive Training Sample Selection):** for each ground truth, compute mean and std of IoU between the top-$k$ anchors (by center distance) and the GT box. Threshold = mean + std. Adaptive and reduces sensitivity to anchor design.

**Task-Aligned Learning (TAL, TOOD):** jointly consider classification score and IoU in the assignment: $t_i = s_i^\alpha \cdot u_i^\beta$ where $s_i$ is classification score and $u_i$ is IoU. High-quality matches have both high confidence and high overlap.

## Links

- [[feature-pyramid-networks]] — FPN generates candidate boxes at each pyramid level; IoU-based NMS is the post-processing step that deduplicates them
- [[statistical-inference-mle]] — IoU is a similarity measure used as the assignment criterion in object detection; threshold choices mirror hypothesis testing cutoffs
- [[arch-roi-align]] — ROI Align extracts features from boxes selected after NMS or matched by IoU threshold during training
- [[open-vocabulary-detection]] — open-vocabulary detectors use generalized IoU and class-agnostic NMS; DETR-based models replace NMS with learned one-to-one matching
- [[evaluation-metrics-guide]] — mAP is computed by integrating precision-recall at IoU thresholds 0.5:0.05:0.95; IoU is the foundation of COCO evaluation
