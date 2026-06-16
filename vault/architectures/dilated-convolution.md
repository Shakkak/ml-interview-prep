---
title: Dilated (Atrous) Convolution and ASPP
tags: [dilated-convolution, atrous-convolution, aspp, deeplab, semantic-segmentation, receptive-field]
aliases: [dilated convolution, atrous convolution, ASPP, atrous spatial pyramid pooling, DeepLab, dilation rate]
difficulty: 2
status: complete
depends_on: [convolution-math, arch-residual-block]
related: [cnn-architectures-guide, feature-pyramid-networks, unet, arch-residual-block, arch-bottleneck-1x1]
---

# Dilated (Atrous) Convolution and ASPP

---

## Fundamental

Standard CNN backbones aggressively downsample via strided convolutions — ResNet-50 produces a 7×7 feature map from 224×224 input (stride 32). For classification this is fine. For **dense prediction** (segmentation, depth, optical flow), you need both a large receptive field *and* high spatial resolution — these goals are in direct tension with strided downsampling.

**Intuition:** imagine a 3×3 filter that, instead of looking at 3 consecutive pixels, looks at every 2nd pixel — covering a $5 \times 5$ area with the same 9 weights. That is dilation rate 2. Increase to rate 4 and the same 9 weights cover a $9 \times 9$ area. The network "sees further" without needing more parameters or losing spatial resolution.

**Dilated convolution** inserts gaps between kernel elements, expanding the receptive field without downsampling:

$$y[i] = \sum_k x[i + d \cdot k] \cdot w[k]$$

where $y[i]$ = output at position $i$, $x$ = input signal, $d$ = dilation rate (gap between sampled positions), $k$ = kernel element index (0 to $K-1$ for a size-$K$ kernel), and $w[k]$ = kernel weight at position $k$. For a 3×3 kernel with dilation rate $d$:
- $d=1$: standard conv, covers $3 \times 3$ area.
- $d=2$: covers $5 \times 5$ area (skipping every other element).
- $d=4$: covers $9 \times 9$ area.

**Key property:** the number of parameters stays the same (3×3 = 9 weights regardless of $d$). Receptive field grows without additional parameters or resolution loss.

---

## Intermediate

### Effective Receptive Field with Stacked Dilations

Stacking layers with increasing dilation rates gives exponentially growing receptive fields:

| Layer | Dilation | Cumulative RF |
|-------|----------|---------------|
| Conv 1 | 1 | 3×3 |
| Conv 2 | 2 | 7×7 |
| Conv 3 | 4 | 15×15 |
| Conv 4 | 8 | 31×31 |
| Conv 5 | 16 | 63×63 |

After 5 layers with rates $\{1,2,4,8,16\}$: 63×63 receptive field with only 9 parameters per layer. A standard 3×3 conv would need 31 layers for the same coverage.

### DeepLab Family

DeepLab modifies a [[arch-residual-block|ResNet]] backbone: the **last two stages' strided convolutions are replaced with dilated convolutions** (stride=1, dilation=2 and 4). Output stride changes from 32 to 8:
- ResNet final feature map: 28×28 instead of 7×7 for a 224×224 input.
- 16× more spatial detail for the segmentation head.
- Same receptive field preserved by the dilation values.

| Version | Backbone | Post-processing | Key innovation |
|---------|----------|----------------|----------------|
| v1/v2 | Dilated ResNet | CRF | Dilated conv for segmentation |
| v3 | Dilated ResNet | None | ASPP (multi-scale) |
| v3+ | Dilated ResNet | [[unet\|U-Net]] decoder | Sharp boundaries via skip connections |

### Dilated vs Strided vs Pooling

| | Strided conv / pooling | Dilated conv |
|--|:---:|:---:|
| Spatial resolution | ↓ Reduced | Unchanged |
| Receptive field | Grows with depth | Grows without resolution loss |
| Computation | Fewer ops (smaller FM) | More ops (full FM persists) |
| Best for | Classification backbones | Dense prediction |

---

## Advanced

### ASPP — Atrous Spatial Pyramid Pooling

A single dilation rate captures context at one scale. ASPP applies multiple rates in parallel and concatenates the results:

```
Feature map (H×W×C)
  ├─ 1×1 conv (rate=1)        ─────────────────────────────┐
  ├─ 3×3 atrous (rate=6)      ─────────────────────────────┤
  ├─ 3×3 atrous (rate=12)     ─────────────────────────────┤ → concat → 1×1 conv → output
  ├─ 3×3 atrous (rate=18)     ─────────────────────────────┤
  └─ Global average pool + 1×1 conv + upsample ────────────┘
```

The global pooling branch captures image-level context (what objects exist in the scene at all). The 1×1 conv fuses all scales. Rates $\{6, 12, 18\}$ are chosen for a stride-8 backbone to roughly tile the receptive field at coarse, medium, and fine scales.

### Gridding Artifacts and Fixes

With the same dilation rate $d$ for multiple consecutive layers: each output pixel depends on only every $d$-th input pixel. For $d=2$ over 3 layers, only $\frac{1}{4}$ of input pixels are seen — a regular gap pattern.

**Fix 1 — Increasing rates $\{1, 2, 4, 8\}$:** each layer's gap pattern offsets the previous, providing coverage. Ensure $\gcd(d_1, d_2, \ldots) = 1$ — e.g., avoid all-even rates.

**Fix 2 — Hybrid Dilated Convolutions (HDC):** design the rate sequence so the union of all sampling patterns covers every input pixel with equal density. This is verified by computing the "coverage matrix" for the rate sequence.

**Fix 3 — ASPP with multiple rates:** inherently samples at multiple offsets, diluting the impact of any single grid.

### FPN vs ASPP for Multi-Scale

Both FPN and ASPP solve the same multi-scale representation problem, but differently:

| | FPN | ASPP |
|--|:---:|:---:|
| Multi-scale mechanism | Top-down feature merging across backbone levels | Parallel dilated convs at one level |
| Resolution | Multiple resolutions (P2–P5) | Single resolution |
| Best for | Object detection (scale-specific anchors) | Dense segmentation (uniform spatial prediction) |

DeepLab v3+ combines both: ASPP for multi-scale context + a U-Net-style decoder for spatial detail recovery.

---

## Links

- [[convolution-math]] — dilation is a modification to the standard convolution kernel; inserting $r-1$ zeros between kernel elements is equivalent to multiplying the effective kernel size
- [[arch-residual-block]] — dilated convolutions replace standard convolutions inside residual blocks in WaveNet, DeepLab, and dilated ResNets
- [[cnn-architectures-guide]] — dilated convolutions are the alternative to pooling + upsampling for maintaining spatial resolution in semantic segmentation
- [[feature-pyramid-networks]] — FPN aggregates multi-scale context using pooling; ASPP achieves similar coverage via parallel dilated convolutions at different rates
- [[unet]] — U-Net uses pooling + upsampling for multi-scale context; dilated convolutions provide the same without losing spatial resolution
- [[arch-bottleneck-1x1]] — atrous bottleneck blocks combine dilated $3\times 3$ convolutions with $1\times 1$ channel reduction
