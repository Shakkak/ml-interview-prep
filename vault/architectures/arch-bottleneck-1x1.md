---
title: 1×1 Convolution and Bottleneck Design
tags: [architecture, cnn, convolution]
aliases: [1x1 conv, pointwise conv, network-in-network, bottleneck]
difficulty: 1
status: complete
related: [arch-residual-block, arch-depthwise-separable, cnn-architectures-guide, feature-pyramid-networks]
---

# 1×1 Convolution and Bottleneck Design

---

## Fundamental

A $1 \times 1$ convolution applies a learned linear combination across *channels* at each spatial location independently.

For input $H \times W \times C_{\text{in}}$, a $1 \times 1$ conv with $C_{\text{out}}$ filters:
- **Output:** $H \times W \times C_{\text{out}}$
- **Parameters:** $C_{\text{in}} \times C_{\text{out}}$ (no spatial extent)
- **Spatial footprint:** 1 pixel — no spatial mixing

Equivalent to applying the same fully-connected layer to every spatial position. It is a channel mixer with no spatial dependencies.

**Three primary uses:**
1. Channel reduction (bottleneck) — reduce before expensive 3×3 conv
2. Channel expansion — grow capacity (MobileNet inverted bottleneck)
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

Feature Pyramid Networks use 1×1 convs to align channel counts before adding top-down features:

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

Rationale: depthwise convolution has limited expressive power per channel. Expanding first (factor 6) gives more feature channels for the depthwise filter, then the pointwise projection compresses back. ReLU is applied after the expansion but **not** after the projection — information is not discarded at the narrow bottleneck.

### Network-in-Network and Global Average Pooling

The $1 \times 1$ conv was introduced in Network-in-Network (Lin et al., 2013) as a micro-network at each spatial location. The same paper proposed global average pooling (GAP) as a structural regularizer replacing fully-connected layers. GAP computes the spatial average of each channel: $H \times W \times C \to 1 \times 1 \times C$. This eliminated millions of FC parameters and provided geometric transformation robustness. Both ideas became standard in virtually every modern CNN.

---

*See also: [[arch-residual-block]] · [[arch-depthwise-separable]] · [[feature-pyramid-networks]] · [[cnn-architectures-guide]]*
