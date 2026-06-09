---
title: Residual Block
tags: [architecture, cnn, resnet, skip-connection]
aliases: [residual block, skip connection, shortcut connection, ResNet block]
difficulty: 1
status: complete
related: [cnn-architectures-guide, arch-bottleneck-1x1, normalization-layers, backpropagation-advanced]
---

# Residual Block

---

## Fundamental

Adding more layers to a plain CNN *hurts* performance — not just overfitting, but training accuracy decreases. Deeper networks are harder to optimize, even though theoretically the extra layers could just learn identity and match the shallower network.

**The solution:** let layers learn the *residual* $F(x)$ relative to the input, then add back:

$$y = F(x, \{W_i\}) + x$$

If the optimal function is near identity, $F(x) \to 0$ is trivially easy (zero out the weights). Learning $F(x) = x$ from scratch is much harder.

**Basic block (ResNet-18/34):**

```
x ─────────────────────────────────────────────────────► + ──► y
│                                                         ↑
└──► Conv3×3 → BN → ReLU → Conv3×3 → BN ─────────────────┘
```

([[normalization-layers|BN]] = Batch Normalization)

**Projection shortcut** (when dimensions change — stride 2 or channel change): $y = F(x) + W_s x$ where $W_s$ is a 1×1 conv matching dimensions. Used only at transition layers.

---

## Intermediate

### Why Residuals Enable Deep Networks: Gradient Analysis

[[backpropagation|Backpropagating]] through a residual block:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(1 + \frac{\partial F}{\partial x}\right)$$

The $+1$ term means gradients flow directly back through the shortcut path, bypassing the non-linear layers. Even if $\partial F/\partial x \approx 0$ (saturated layers), the gradient is still $\partial L/\partial y$ — no vanishing.

For a network with $N$ residual blocks, the gradient at block 1 involves a sum of $2^N$ paths (all subsets of shortcuts and residual branches). At least $2^{N-1}$ paths pass through at least one shortcut and thus propagate non-zero gradients.

### Bottleneck Block (ResNet-50/101/152)

```
x ─────────────────────────────────────────────────── + ──► y
│                                                      ↑
└──► Conv1×1(64)→BN→ReLU → Conv3×3(64)→BN→ReLU → Conv1×1(256)→BN ─┘
```

For input $C=256$, bottleneck $c=64$:

| Layer | Params |
|-------|--------|
| 1×1 (256→64) | $256 \times 64 = 16{,}384$ |
| 3×3 (64→64) | $64 \times 64 \times 9 = 36{,}864$ |
| 1×1 (64→256) | $64 \times 256 = 16{,}384$ |
| **Total** | **~70K** |

vs. two 3×3 convs (256→256): $2 \times 256^2 \times 9 = 1.18M$ params. **17× fewer parameters** for similar receptive field.

### Numerical Forward Pass

$x = [1.0, -0.5, 0.8]$. Suppose $F(x) = 0.1x$ (simplified):

$$y = 0.1x + x = 1.1x = [1.10, -0.55, 0.88]$$

Near initialization with small weights, $F(x) \approx 0$, so $y \approx x$ — the block behaves like identity, giving stable training from the start of optimization.

---

## Advanced

### Pre-activation ResNet (He et al., ResNet-v2)

Original order: Conv → BN → [[activation-relu-variants|ReLU]] → (add shortcut).
Pre-activation: BN → ReLU → Conv.

The shortcut path in pre-activation ResNet is completely clean — no BN or ReLU between blocks. This means the identity signal flows through without any modification, which improves gradient flow through very deep networks (1000+ layers).

Additionally, pre-activation places the activation *before* the weight layer, so the weight layer sees normalized inputs at every depth. This is theoretically cleaner and enables training networks up to 1001 layers.

### Wide ResNets vs Deep ResNets

WRN (Zagoruyko & Komodakis, 2016): increase channel width by factor $k$ instead of adding more layers. A WRN-28-10 (28 layers, 10× width) outperforms ResNet-1001 (1001 layers) on CIFAR-10 while being 8× faster to train.

**Why:** very deep residual networks waste parameters in the non-residual branches — many blocks contribute trivially small $F(x)$. Wider networks have more capacity per block and avoid the optimization difficulties of extreme depth.

### ResNeXt: Aggregated Transformations

ResNeXt replaces the 3×3 conv in the bottleneck with $C$ parallel 3×3 convs over low-dimensional embeddings (grouped convolution), then adds their outputs. This introduces "cardinality" (number of groups) as a design dimension alongside width and depth.

$$F(x) = \sum_{i=1}^{C} \mathcal{T}_i(x)$$

Empirically, increasing cardinality at fixed parameter count outperforms increasing width or depth. ResNeXt-101 32×8d outperforms ResNet-101 by 1.7% on ImageNet.

---

*See also: [[cnn-architectures-guide]] · [[arch-bottleneck-1x1]] · [[normalization-layers]] · [[backpropagation-advanced]] · [[backpropagation]] · [[activation-relu-variants]]*
