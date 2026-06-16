---
title: 1×1 Convolution and Bottleneck Design
tags: [architecture, cnn, convolution]
aliases: [1x1 conv, pointwise conv, network-in-network, bottleneck]
difficulty: 1
status: complete
depends_on: [convolution-math, arch-residual-block]
related: [arch-residual-block, arch-depthwise-separable, cnn-architectures-guide, feature-pyramid-networks]
---

# 1×1 Convolution and Bottleneck Design

---

## Fundamental

**The problem:** standard $3 \times 3$ convolutions mix channels and spatial information together. When channel counts are large (256, 512, 1024), the channel-mixing part is the dominant cost. A $3 \times 3$ conv on 256 channels connecting to 256 output channels requires $256 \times 256 \times 9 \approx 589K$ operations per spatial location.

**The solution:** separate channel mixing from spatial filtering. Use $1 \times 1$ convolutions to cheaply change channel counts, then apply expensive spatial operations only on the reduced representation.

A $1 \times 1$ convolution applies a learned linear combination across *channels* at each spatial location independently.

For input $H \times W \times C_{\text{in}}$, a $1 \times 1$ conv with $C_{\text{out}}$ filters:
- **Output:** $H \times W \times C_{\text{out}}$
- **Parameters:** $C_{\text{in}} \times C_{\text{out}}$ (no spatial extent)
- **Spatial footprint:** 1 pixel — no spatial mixing

Equivalent to applying the same fully-connected layer to every spatial position. It is a channel mixer with no spatial dependencies.

**Three primary uses:**
1. Channel reduction (bottleneck) — reduce before expensive 3×3 conv
2. Channel expansion — grow capacity ([[arch-depthwise-separable|MobileNet]] inverted bottleneck)
3. Cross-channel mixing — after depthwise conv (which is per-channel only)

---

## Intermediate

### Channel Reduction: Bottleneck

Reduce channels before an expensive 3×3 conv, restore after:

```
256 channels → 1×1 (→64) → 3×3 (→64) → 1×1 (→256)
```

**FLOPs comparison** on $56 \times 56$ feature map, 256 input channels:

| Approach | FLOPs |
|----------|-------|
| Direct 3×3 conv (256→256) | $56^2 \times 256^2 \times 9 \approx 1.85 \times 10^9$ |
| Bottleneck (256→64→64→256) | $56^2(256 \times 64 + 64^2 \times 9 + 64 \times 256) \approx 2.18 \times 10^8$ |
| **Speedup** | **~8.5×** |

Same input/output dimensions, 8.5× cheaper. The cost is reduced intermediate capacity (64 channels instead of 256 in the spatial filter).

### Inception Parameter Savings

Input: $28 \times 28 \times 192$ channels. Two branches:

| Branch | Without 1×1 | With 1×1 (dim reduction to 96) |
|--------|-------------|---------------------------|
| 3×3 (192→128) | 221K params | $192\times96 + 96\times128\times9 = 128K$ |
| 5×5 (192→32) | 154K params | $192\times16 + 16\times32\times25 = 16K$ |

Total: 375K → 144K. The full Inception network achieves ~12× computation savings with 1×1 reduction.

### FPN Lateral Connections

[[feature-pyramid-networks|Feature Pyramid Networks]] use 1×1 convs to align channel counts before adding top-down features:

```
C5 (2048 ch) → 1×1 → P5 (256 ch)
C4 (1024 ch) → 1×1 →   + → P4 (256 ch)
C3 (512 ch)  → 1×1 →   + → P3 (256 ch)
```

The 1×1 conv unifies disparate channel widths to a common 256-channel representation.

---

## Advanced

### Factorized View: Standard = Depthwise + Pointwise

| Convolution type | Spatial mixing | Channel mixing | Cost |
|-----------------|:---:|:---:|------|
| Standard $k \times k$ | Yes | Yes | $HWC_{in}C_{out}k^2$ |
| Depthwise $k \times k$ | Yes | No | $HWCk^2$ |
| Pointwise $1 \times 1$ | No | Yes | $HWC_{in}C_{out}$ |

Standard convolution = depthwise (spatial) + pointwise (channel) approximately. This decomposition is the foundation of MobileNet. The key theoretical insight (Chollet, Xception): spatial correlations and cross-channel correlations can be fully decoupled without meaningful loss of representational capacity.

### Inverted Bottleneck (MobileNetV2)

Standard bottleneck: wide → narrow → wide (compress in the spatial filter).
MobileNetV2 **inverts** this: narrow → wide → narrow.

```
32 ch → 1×1 expand (→192) → 3×3 depthwise (→192) → 1×1 project (→32) + skip
```

Rationale: depthwise convolution has limited expressive power per channel. Expanding first (factor 6) gives more feature channels for the depthwise filter, then the pointwise projection compresses back. [[activation-relu-variants|ReLU]] is applied after the expansion but **not** after the projection — information is not discarded at the narrow bottleneck.

### Network-in-Network and Global Average Pooling

The $1 \times 1$ conv was introduced in Network-in-Network (Lin et al., 2013) as a micro-network at each spatial location. The same paper proposed global average pooling (GAP) as a structural regularizer replacing fully-connected layers. GAP computes the spatial average of each channel: $H \times W \times C \to 1 \times 1 \times C$. This eliminated millions of FC parameters and provided geometric transformation robustness. Both ideas became standard in virtually every modern CNN.

---

## Links

- [[convolution-math]] — $1\times 1$ convolution is a special case of convolution with kernel size 1; it mixes channel information without spatial aggregation
- [[arch-residual-block]] — the bottleneck residual block uses $1\times 1$ convolutions to reduce then restore channel depth around the $3\times 3$ conv
- [[arch-depthwise-separable]] — depthwise separable convolutions use a $1\times 1$ pointwise conv to mix channels after the spatial depthwise step
- [[feature-pyramid-networks]] — FPN uses $1\times 1$ lateral connections to align channel depths across pyramid levels before merging
- [[cnn-architectures-guide]] — bottleneck blocks are the core building block of ResNet-50+, ResNeXt, and EfficientNet
- [[activation-relu-variants]] — the $1\times 1$ conv is typically followed by BN + ReLU to introduce nonlinearity at the channel-mixing step
