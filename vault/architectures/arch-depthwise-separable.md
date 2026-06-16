---
title: Depthwise Separable Convolution
tags: [architecture, cnn, efficiency, mobilenet]
aliases: [depthwise separable conv, depthwise conv, MobileNet, separable convolution]
difficulty: 2
status: complete
depends_on: [convolution-math, arch-bottleneck-1x1]
related: [arch-bottleneck-1x1, cnn-architectures-guide]
---

# Depthwise Separable Convolution

---

## Fundamental

**The problem:** a standard convolution combines spatial filtering (how pixels near each other relate) and channel mixing (how features from different channels combine) in a single operation. This coupling is expensive: every input channel talks to every output filter at every spatial location.

**The insight:** these two operations can be separated. Apply spatial filtering per-channel first (depthwise), then mix channels with a cheap $1 \times 1$ convolution (pointwise). The result is nearly identical representation power at a fraction of the cost.

A standard $k \times k$ conv on $H \times W \times C_{in}$ to $C_{out}$ channels costs:

$$\text{FLOPs} = H \cdot W \cdot C_{in} \cdot C_{out} \cdot k^2$$

where $H, W$ = spatial height and width of the feature map, $C_{in}$ = number of input channels, $C_{out}$ = number of output channels, $k$ = kernel size (e.g., 3 for a $3 \times 3$ filter).

The $C_{in} \cdot C_{out}$ term is expensive — every input channel combines with every output filter.

**Factorization into two steps:**

**Step 1 — Depthwise conv:** one $k \times k$ filter per input channel (no cross-channel mixing):
$$\text{FLOPs}_{\text{DW}} = H \cdot W \cdot C \cdot k^2 \qquad \text{(output: } H \times W \times C\text{)}$$

where $C = C_{in}$ (same channel count in and out — one filter per channel, no mixing).

**Step 2 — Pointwise conv (1×1):** mix information across channels:
$$\text{FLOPs}_{\text{PW}} = H \cdot W \cdot C_{in} \cdot C_{out} \qquad \text{(output: } H \times W \times C_{out}\text{)}$$

**Cost ratio:**
$$\frac{\text{FLOPs}_{\text{DW+PW}}}{\text{FLOPs}_{\text{standard}}} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

For $k=3$, $C_{out}=256$: ratio $\approx \frac{1}{9} \approx 0.11$ → **~8–9× cheaper**.

---

## Intermediate

### Numerical Example

Input: $14 \times 14 \times 64$, target: $128$ output channels with $3 \times 3$ filters.

| Method | FLOPs |
|--------|-------|
| Standard 3×3 conv | $14^2 \times 64 \times 128 \times 9 = 14.45 \times 10^6$ |
| Depthwise (3×3, 64 ch) | $14^2 \times 64 \times 9 = 112{,}896$ |
| Pointwise (1×1, 64→128) | $14^2 \times 64 \times 128 = 1{,}605{,}632$ |
| **DW+PW total** | $\mathbf{1.72 \times 10^6}$ |
| **Reduction** | **8.4×** ✓ |

### MobileNetV1

Replace all standard 3×3 convs in a VGG-style network with depthwise separable convs (Howard et al., 2017). Result: 8–9× fewer FLOPs, ~3–4% top-1 accuracy drop on ImageNet. Runs in real-time on mobile CPU.

**Width multiplier $\alpha$:** scale all channel counts by $\alpha$ (e.g., 0.75 → 75% of all channels). Reduces FLOPs by $\alpha^2$.
**Resolution multiplier $\rho$:** scale input resolution. FLOPs scale as $\rho^2$.

These two multipliers provide an accuracy-efficiency dial for deployment constraints.

### MobileNetV2: Inverted Residual

Standard bottleneck: wide → narrow (bottleneck) → wide.
MobileNetV2 **inverts**: narrow → wide → narrow.

```
32 ch → 1×1 expand (→192, factor 6) → 3×3 depthwise → 1×1 project (→32) + skip
```

Why inverted: depthwise conv has limited expressive power per channel. Expanding first gives more features for spatial filtering. [[activation-relu-variants|ReLU6]] (capped at 6) used after expansion for numerical stability at low precision. **No activation after the projection** — the bottleneck embedding stays in a low-dimensional manifold without information loss from non-linearity.

---

## Advanced

### Xception: Extreme Inception

Inception modules use 1×1 convs to mix channels before different-sized spatial filters. Xception (Chollet, 2017) takes this to the extreme: replace every standard conv with depthwise separable conv, add [[arch-residual-block|residual connections]] between blocks.

**The key theoretical claim:** "the mapping of cross-channel correlations and spatial correlations in the feature maps of convolutional neural networks can be entirely decoupled." This is the hypothesis that depthwise separable conv tests. Xception outperforms Inception-V3 on ImageNet and JFT-300M with the same number of parameters, empirically supporting the hypothesis.

### EfficientNet: Compound Scaling

EfficientNet (Tan & Le, 2019) makes three observations:
1. Depth scaling (more layers), width scaling (more channels), and resolution scaling each improve accuracy but hit diminishing returns alone.
2. **Compound scaling** all three together under a fixed FLOP budget outperforms scaling any one dimension.
3. The optimal MBConv block (= MobileNetV2 inverted residual with [[squeeze-excitation|Squeeze-and-Excitation]]) is found via neural architecture search.

Scaling rule: $d = \alpha^\phi$, $w = \beta^\phi$, $r = \gamma^\phi$ where $\phi$ is the compound coefficient controlling overall scale, $\alpha, \beta, \gamma$ are the per-dimension scaling factors found by grid search, and the constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ ensures FLOPs roughly double with each unit increase in $\phi$ — since depth scales linearly with FLOPs while width and resolution each scale quadratically. EfficientNet-B7 achieves 84.4% ImageNet top-1 at 37B FLOPs — higher accuracy than GPipe (557B FLOPs) at 15× lower cost.

---

## Links

- [[convolution-math]] — standard convolution mixes spatial and channel dimensions simultaneously; depthwise separable convolution factors this into two separate steps
- [[arch-bottleneck-1x1]] — the pointwise ($1\times 1$) step in depthwise separable convolution is identical to a bottleneck $1\times 1$ conv that remixes channels
- [[cnn-architectures-guide]] — MobileNet and EfficientNet are built on depthwise separable convolutions; the FLOPs reduction makes them deployable on mobile hardware
- [[squeeze-excitation]] — SE blocks add channel-wise attention on top of depthwise separable convolutions in MobileNetV3
- [[arch-residual-block]] — Xception combines depthwise separable convolutions with residual connections; this is the design used in EfficientNet's inverted residuals
- [[activation-relu-variants]] — ReLU6 (clipped ReLU) is used after depthwise separable convolutions in MobileNet for quantization compatibility
