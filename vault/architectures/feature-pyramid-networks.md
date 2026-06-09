---
title: Feature Pyramid Networks and Multi-Scale Detection
tags: [object-detection, fpn, multi-scale, cnn, segmentation]
aliases: [FPN, feature pyramid, top-down pathway, anchor assignment]
difficulty: 3
status: complete
related: [cnn-architectures-guide, evaluation-metrics-guide, backpropagation-advanced]
---

# Feature Pyramid Networks and Multi-Scale Detection

---

## Fundamental

Objects appear at dramatically different scales — the same person can occupy 400×200 pixels (nearby) or 20×10 pixels (distant). Standard CNN feature maps at a single scale cannot detect both: a 7×7 feature map (stride 32) has resolution too low to detect small objects.

**FPN** (Lin et al., 2017) uses the existing feature hierarchy inside a CNN backbone and adds top-down connections to combine deep (semantic) features with shallow (high-resolution) features — producing a multi-scale pyramid at essentially no extra cost.

### Before FPN

**Image pyramid:** run the detector at multiple input scales. 3 scales = 3× inference cost.

**SSD (single feature map per scale):** use different backbone layers for different object sizes.
- Problem: early layers have high spatial resolution but poor semantics. Small objects detected at early layers lack the context to be recognized.

```
Semantics:  Low  ──────────────────► High
Resolution: High ──────────────────► Low
```
Each backbone layer is good at one but not both.

---

## Intermediate

### FPN Architecture

**Bottom-up pathway:** standard [[arch-residual-block|ResNet]] forward pass:

```
C2: 200×200, 256ch (stride 4)   ← fine spatial, low semantic
C3: 100×100, 512ch (stride 8)
C4:  50×50, 1024ch (stride 16)
C5:  25×25, 2048ch (stride 32)  ← coarse spatial, high semantic
```

**Top-down pathway:** start from the coarsest level, progressively merge with finer levels:

$$P_l = \text{Conv}_{3\times3}\!\left(\text{Upsample}_{\times2}(P_{l+1}) + \text{Conv}_{1\times1}(C_l)\right)$$

1×1 convs project all backbone levels to 256 channels (lateral connections).
2× nearest-neighbor upsample brings the coarser level to the finer resolution.
3×3 conv after addition reduces aliasing from upsampling.

Each $P_l$ fuses: **upsampled $P_{l+1}$** (deep semantics) + **projected $C_l$** (fine spatial detail). Both problems solved simultaneously.

### Anchor Assignment

| FPN Level | Stride | Responsible Scale |
|-----------|--------|------------------|
| P2 | 4 | ~32px objects |
| P3 | 8 | ~64px objects |
| P4 | 16 | ~128px objects |
| P5 | 32 | ~256px objects |
| P6 | 64 (maxpool of P5) | ~512px objects |

Assignment rule for a ground-truth box with area $wh$:

$$l = \left\lfloor l_0 + \log_2\!\left(\frac{\sqrt{wh}}{224}\right) \right\rfloor, \quad l_0 = 4$$

**Why this enables small object detection:** P2 has stride 4 → 200×200 spatial resolution. A 32px object maps to $32/4 = 8$ pixels in P2 — 8×8 = 64 anchor positions. The same object maps to <1 pixel in a stride-32 feature map. And P2 now carries semantic information propagated from P5 through the top-down pathway.

---

## Advanced

### FPN in Different Architectures

**Faster R-CNN + FPN:** RPN runs on each FPN level with level-specific anchors. [[arch-roi-align|ROI Align]] extracts features from the appropriate level per proposal (scale-aware). The detection head is shared across all levels.

**RetinaNet:** single-stage detector with FPN. Classification and regression heads (shared weights) run on all FPN levels. [[loss-focal|Focal Loss]] handles anchor class imbalance.

**Mask R-CNN + FPN:** adds instance segmentation mask head. FPN is critical for small-object masks — high-resolution P2/P3 features are essential for accurate pixel-level segmentation of small instances.

### BiFPN: Weighted Bidirectional FPN (EfficientDet)

Standard FPN: only top-down pathway. BiFPN adds:
1. **Bottom-up pathway after top-down:** a second round of connections flowing from fine to coarse, allowing fine-level information to propagate upward.
2. **Weighted fusion** instead of equal addition:

$$P_l^{out} = \frac{w_1 P_l + w_2\,\text{Resize}(P_{l-1}^{out}) + w_3\,\text{Resize}(P_{l+1}^{out})}{w_1 + w_2 + w_3 + \epsilon}$$

Weights $w_i \geq 0$ are learned; the normalization keeps them as a convex combination. Bidirectional flow allows both semantic-to-spatial (top-down) and detail-to-semantic (bottom-up) information propagation. EfficientDet uses BiFPN stacked 3–7× for state-of-the-art detection efficiency.

### FPN vs U-Net vs ASPP

All three solve multi-scale representation but differ in mechanism:

| | FPN | [[unet\|U-Net]] | [[dilated-convolution\|ASPP]] |
|--|:---:|:---:|:---:|
| Multi-scale mechanism | Top-down feature merging | Encoder-decoder skip | Parallel dilated convs |
| Resolution | Multiple pyramid levels | Encoder resolution levels | Single resolution |
| Best for | Detection (scale anchors) | Segmentation (dense, medical) | Segmentation (single-level) |
| Original domain | Natural image detection | Medical image segmentation | Semantic segmentation |

---

*See also: [[cnn-architectures-guide]] · [[dilated-convolution]] · [[unet]] · [[arch-roi-align]] · [[arch-residual-block]] · [[loss-focal]]*
