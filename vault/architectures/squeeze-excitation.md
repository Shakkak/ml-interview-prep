---
title: Squeeze-and-Excitation Networks (SE Blocks)
tags: [squeeze-excitation, channel-attention, se-block, efficientnet, recalibration, cbam]
aliases: [SE block, Squeeze-and-Excitation, channel attention, SENet, recalibration, CBAM]
difficulty: 2
status: complete
related: [arch-residual-block, cnn-architectures-guide, arch-bottleneck-1x1, attention-mechanism, normalization-layers]
---

# Squeeze-and-Excitation Networks (SE Blocks)

---

## Fundamental

Standard CNNs treat all channels equally after a convolution. But different channels represent different features (edges, textures, object parts), and their relative importance varies with input content. An edge detector channel should be down-weighted when processing a uniform background; a texture channel should be emphasized when processing fabric.

**SE blocks** (Hu et al., 2018) add a lightweight mechanism that:
1. **Squeezes** — globally summarizes what each channel contains (global average pooling).
2. **Excites** — computes channel importance weights via two FC layers.
3. **Scales** — multiplies each channel by its learned weight.

### The Three Operations

Given feature map $X \in \mathbb{R}^{H \times W \times C}$:

**Squeeze (global average pool):**
$$z_c = \frac{1}{HW} \sum_{i,j} X_{ijc} \in \mathbb{R}^C$$

$z_c$ summarizes "how active is channel $c$ on average for this image?"

**Excitation (two FC layers + sigmoid):**
$$s = \sigma\!\left(W_2 \cdot \text{ReLU}(W_1 z)\right), \quad W_1 \in \mathbb{R}^{C/r \times C},\ W_2 \in \mathbb{R}^{C \times C/r}$$

> [!tip] Why sigmoid, not softmax?
> Softmax weights sum to 1 — boosting one channel forces down-weighting others (zero-sum). Sigmoid lets each channel weight be set independently in (0,1): you can suppress several channels simultaneously without redistributing their "budget" to others.

Reduction ratio $r=16$ by default. Parameters: $2C^2/r$.

**Scale:**
$$\tilde{X}_{ijc} = s_c \cdot X_{ijc}$$

---

## Intermediate

### Integration with ResNet

In SE-ResNet, the SE block is inserted after the last conv of each [[arch-residual-block|residual block]], before the residual addition:

```
x ──── Conv block ──── SE block ──── + ──── y
│                                    ↑
└────────────────────────────────────┘ (skip)
```

### Parameter Overhead and Performance Gain

For $C=256$, $r=16$: SE parameters = $2 \times 256^2/16 = 8{,}192$. This is tiny compared to a 3×3 conv (256→256 = 590K params).

| Model | ImageNet Top-1 | Params |
|-------|:---:|:---:|
| ResNet-50 | 75.2% | 25.6M |
| SE-ResNet-50 | **76.7%** | 28.1M (+10%) |
| ResNet-101 | 76.4% | 44.5M |
| SE-ResNet-101 | **77.6%** | 49.3M (+11%) |

~1.5% absolute improvement at ~10% parameter overhead — one of the best parameter-efficiency tradeoffs in the literature. SE-Net won ILSVRC 2017 classification.

### SE in EfficientNet

EfficientNet's [[arch-depthwise-separable|MBConv block]] uses SE with reduction ratio $r=4$ (applied to the *expanded* channel dimension):

```
Input → 1×1 Expand → 3×3 Depthwise → SE block → 1×1 Project → Output
```

The SE block sees the expanded channel count (6× input channels), so $r=4$ on the expanded dimension. Every EfficientNet variant (B0–B7) includes SE; it is a key reason EfficientNet outperforms other efficient CNNs.

---

## Advanced

### Why Global Average Pooling for Squeeze?

Global average pooling aggregates each channel into a single number: $z_c = \frac{1}{HW}\sum_{i,j} X_{ijc}$. This captures the mean activation of each semantic detector across the entire image.

An alternative is global max pooling ($z_c = \max_{i,j} X_{ijc}$), which captures whether the detector fired *at all*. In practice, both work similarly; CBAM uses both.

The key insight: global spatial information is crucial for channel recalibration. A "dog ear" channel should be up-weighted if ears appear anywhere in the image — a local receptive field would miss them if the dog is in a different region.

### CBAM: Adding Spatial Attention

CBAM (Woo et al., 2018) extends SE with a **spatial attention module** applied after channel attention:

**Channel attention (extended SE):** uses both avg and max pool:
$$M_c = \sigma\!\left(W_1(\text{AvgPool}(F)) + W_1(\text{MaxPool}(F))\right)$$

**Spatial attention:** pooling across the channel dimension produces spatial maps:
$$M_s = \sigma\!\left(\text{Conv}_{7\times7}\!\left[\text{AvgPool}_c(F);\ \text{MaxPool}_c(F)\right]\right)$$

$\text{AvgPool}_c$ and $\text{MaxPool}_c$ collapse the channel dimension to a single value per spatial location, producing $H \times W$ maps. The conv fuses them into a spatial importance map.

Applied sequentially: $F' = M_c(F) \odot F$, then $F'' = M_s(F') \odot F'$.

CBAM slightly outperforms SE at comparable parameter count. It addresses a limitation of SE: SE recalibrates *what* features (channels) are important, but not *where* in the image.

### SE as Learned Feature Selection

From a representation learning perspective, SE blocks implement **dynamic, input-dependent feature selection**. [[normalization-layers|BatchNorm]] learns a fixed affine rescaling $(\gamma, \beta)$ per channel — the same scale regardless of input. SE learns to compute $s_c$ as a function of the input, enabling the network to select which feature detectors are relevant for each specific input. This is a form of sample-wise attention that is computationally cheap.

---

*See also: [[arch-residual-block]] · [[cnn-architectures-guide]] · [[attention-mechanism]] · [[normalization-layers]] · [[arch-depthwise-separable]]*
