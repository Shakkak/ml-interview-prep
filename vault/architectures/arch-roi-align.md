---
title: ROI Align
tags: [architecture, object-detection, segmentation]
aliases: [ROI Align, ROI Pooling, RoIAlign, region of interest]
difficulty: 2
status: complete
related: [feature-pyramid-networks, cnn-architectures-guide]
---

# ROI Align

---

## Fundamental

**The problem:** Faster R-CNN uses ROI Pooling to extract fixed-size ($7 \times 7$) features from arbitrary-sized region proposals. This requires two quantization steps:

1. Map proposal coordinates to the feature map: $250.3 \div 32 = 7.82 \to 7$ (rounded).
2. Divide the feature-map region into $7 \times 7$ bins with integer boundaries (rounded).

Rounding errors of 0.5–1 pixels on the feature map correspond to **16–32 pixels in the input image** (stride 32). For detection this is tolerable; for instance segmentation or keypoints, it causes visible misalignment.

**ROI Align (He et al., Mask R-CNN, 2017):** eliminate both quantization steps using bilinear interpolation.

1. Map proposal to feature map *without rounding*: $(x_1/s, y_1/s, x_2/s, y_2/s)$ in float.
2. Divide into $k \times k$ bins (float boundaries).
3. Sample 4 points per bin on a $2 \times 2$ grid within the bin.
4. Compute feature values at non-integer sample locations via **bilinear interpolation**.
5. Average samples within each bin → one value per bin.

---

## Intermediate

### Bilinear Interpolation

For a non-integer sample point $(x, y)$ with integer neighbors $x_0 = \lfloor x \rfloor$, $x_1 = x_0+1$, $y_0 = \lfloor y \rfloor$, $y_1 = y_0+1$, and weights $\alpha = x-x_0$, $\beta = y-y_0$:

$$f(x, y) = (1-\alpha)(1-\beta) f_{00} + \alpha(1-\beta) f_{10} + (1-\alpha)\beta f_{01} + \alpha\beta f_{11}$$

**Worked example.** Feature map values: $f_{00}=1, f_{10}=2, f_{01}=4, f_{11}=5$. Sample point $(0.5, 0.75)$, so $\alpha=0.5$, $\beta=0.75$:

$$f = 0.5 \times 0.25 \times 1 + 0.5 \times 0.25 \times 2 + 0.5 \times 0.75 \times 4 + 0.5 \times 0.75 \times 5$$
$$= 0.125 + 0.250 + 1.500 + 1.875 = 3.75$$

Gradients of $f$ w.r.t. the 4 feature map pixels are just the bilinear coefficients — smooth and differentiable, enabling end-to-end training.

### Performance Impact

| | ROI Pooling | ROI Align |
|---|---|---|
| Quantization | Yes (2 rounds) | None |
| Object detection AP | Baseline | +~0.3 mAP |
| Instance segmentation mask AP | Poor | **+~3 mask AP** |
| Keypoint AP | Poor | Large improvement |

The mask AP improvement is dramatic because segmentation requires precise spatial alignment at the pixel level — a 16-pixel misalignment in ROI Pooling directly corrupts mask predictions at object boundaries.

### FPN Level Assignment

In Mask R-CNN, ROI Align is applied to [[feature-pyramid-networks|FPN]] feature maps. Each proposal is assigned to a pyramid level based on its area:

$$\text{level} = \left\lfloor k_0 + \log_2\left(\frac{\sqrt{wh}}{224}\right) \right\rfloor$$

Large objects → coarser levels (P5, low resolution). Small objects → finer levels (P2, high resolution). ROI Align then extracts from the assigned level, ensuring features are sampled at an appropriate spatial scale.

---

## Advanced

### Gradient Through Bilinear Interpolation

During backprop, the gradient of the loss w.r.t. feature map value $f_{ij}$ at position $(x_0+i, y_0+j)$ for a sample at $(x, y)$ is the bilinear weight:

$$\frac{\partial L}{\partial f_{00}} = \frac{\partial L}{\partial f(x,y)} \cdot (1-\alpha)(1-\beta)$$

For each feature map pixel, gradients accumulate from all sample points that used it for interpolation. This makes ROI Align fully differentiable — all backbone weights are updated based on the aligned region features.

### Deformable ROI Pooling

An extension (Dai et al., 2017): learn spatial offsets for the sampling grid. Instead of a regular $n \times n$ grid within each bin, the model learns a displacement $(\Delta x_{ij}, \Delta y_{ij})$ per bin per proposal, enabling the pooling region to adapt to object shape. Applied in Deformable ConvNets and later models for dense prediction.

### Exact vs Approximate ROI Align

With 1 sample per bin: fast but lower quality. With 4 samples ($2 \times 2$) per bin: standard. The number of samples trades computation for accuracy — for segmentation tasks, 4 samples is standard; for detection only, 1 sample suffices.

---

*See also: [[feature-pyramid-networks]] · [[cnn-architectures-guide]]*
