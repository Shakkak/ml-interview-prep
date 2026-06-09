---
title: CNN Architectures — Complete Guide
tags: [cnn, deep-learning, architectures, resnet, efficientnet, vgg, mobilenet, convnext]
aliases: [ResNet, EfficientNet, VGG, MobileNet, ConvNeXt, bottleneck, AlexNet, GoogLeNet]
difficulty: 3
status: complete
related: [arch-residual-block, arch-bottleneck-1x1, arch-depthwise-separable, feature-pyramid-networks, normalization-layers, squeeze-excitation]
---

# CNN Architectures — Complete Guide

---

## Fundamental

A 2D convolution of input $x \in \mathbb{R}^{C_{in} \times H \times W}$ with kernel $w \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$ produces:
- **Parameters:** $C_{out} \times C_{in} \times K^2 + C_{out}$
- **FLOPs:** $C_{out} \times C_{in} \times K^2 \times H' \times W'$
- **Output size:** $H' = \lfloor(H + 2p - K)/s\rfloor + 1$

**Receptive field** grows with every layer. For a network of layers with kernel sizes $k_i$ and strides $s_j$:

$$\text{RF}_n = 1 + \sum_{i=1}^{n}(k_i - 1)\prod_{j=1}^{i-1}s_j$$

Early strides multiply the RF of all subsequent layers. A stride-2 layer in layer 1 doubles the effective coverage of every downstream layer — this is why stem layers use aggressive strides without wasting much information.

### The Key Architectures at a Glance

| Architecture | Year | Top-1 | Params | Key Innovation |
|---|:-:|:-:|:-:|---|
| AlexNet | 2012 | 63.3% | 60M | [[activation-relu-variants\|ReLU]], [[regularization-dropout\|dropout]], GPU |
| VGG-16 | 2014 | 74.4% | 138M | Systematic depth with 3×3 |
| ResNet-50 | 2015 | 76.1% | 25M | [[arch-residual-block\|Residual connections]] |
| Inception-v3 | 2016 | 78.8% | 24M | Multi-scale parallel paths |
| MobileNetV2 | 2018 | 72.0% | 3.4M | Inverted residual, linear bottleneck |
| EfficientNet-B7 | 2019 | 84.3% | 66M | Compound scaling |
| ConvNeXt-L | 2022 | 86.6% | 197M | Transformer recipe on CNN |

---

## Intermediate

### VGG: Why Only 3×3?

Three stacked 3×3 convolutions have the same receptive field (7×7) as a single 7×7, but cost $3 \times 9C^2 = 27C^2$ vs $49C^2$ parameters — **45% fewer** — plus two extra nonlinearities for richer representations. This was a major empirical discovery: smaller kernels stacked deep are strictly better than large kernels.

VGG-16: 13 conv layers of 3×3, doubling channels at each downsampling (64→128→256→512→512), ending in two FC-4096 layers. 138M parameters — 96% in the FC layers. This motivated replacing FC layers with global average pooling (GoogLeNet, ResNet) as the field matured.

### ResNet: The Residual Learning Hypothesis

Deep networks degrade in training accuracy before overfitting. The paradox: a 56-layer network should perform at least as well as 34-layer because extra layers can approximate identity. It doesn't — optimization fails.

**Hypothesis:** instead of learning $H(x)$ directly, learn the residual $F(x) = H(x) - x$ so $\text{output} = F(x) + x$. If identity is optimal, $F(x) = 0$ — trivially achievable by driving weights toward zero.

Gradient analysis makes this precise. With a residual connection, the gradient of loss $L$ with respect to early activation $x_l$:

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L}\left(1 + \sum_{i=l}^{L-1}\frac{\partial F_i}{\partial x_l}\right)$$

The $+1$ term guarantees gradient reaches $x_l$ directly, regardless of how small the $F_i$ terms become.

**Bottleneck block (ResNet-50/101/152):** 1×1 → 3×3 → 1×1. Reduces channels by 4× before the expensive 3×3, then expands back. A 256-channel bottleneck costs 70K FLOPs/position vs 1.18M for a plain 3×3 block — 17× cheaper.

### MobileNet: Factored Convolutions for Mobile

Standard 3×3 conv on $C_{in}$ input, $C_{out}$ output: $C_{in} \times C_{out} \times 9$ parameters and FLOPs.

Depthwise separable factorization:
1. **Depthwise:** $C_{in} \times 9$ (one 3×3 per channel, no cross-channel mixing)
2. **Pointwise:** $C_{in} \times C_{out}$ (1×1 mixing)

Cost ratio: $\frac{1}{C_{out}} + \frac{1}{9} \approx \frac{1}{9}$ for large $C_{out}$. MobileNetV1 achieves 8-9× FLOP reduction at ~1% accuracy loss.

**MobileNetV2 — inverted residual:** standard bottleneck narrows then widens (256→64→256). Inverted residual widens then narrows (24→144→24 with expansion factor 6). The expansion happens where depthwise conv operates, so channels are wide there at low per-channel cost. Linear (no-activation) final projection preserves the information-theoretic manifold.

### EfficientNet: Compound Scaling

CNNs have three scaling dimensions — depth $d$, width $w$, resolution $r$ — and scaling one while fixing the others saturates quickly. EfficientNet finds the optimal balance:

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi, \quad \text{subject to } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

The constraint ensures doubling $\phi$ (overall scale) roughly doubles FLOPs. Constants $\alpha=1.2, \beta=1.1, \gamma=1.15$ are found by grid search on the small baseline (B0). B0-B7 are obtained by increasing $\phi = 0, 1, \ldots, 7$.

EfficientNet-B7 achieves 84.3% ImageNet top-1 with 8.4× fewer parameters than the previous state of the art at the same accuracy. Every variant includes [[squeeze-excitation|SE blocks]] (reduction $r=4$), making channel attention a core architectural component.

---

## Advanced

### ConvNeXt: Closing the Gap to Vision Transformers

The 2022 ConvNeXt paper (Liu et al.) asked: can a pure CNN match Swin Transformer if we apply every modern design choice from the transformer world? The answer was yes — and the analysis reveals why [[vision-transformer|ViTs]] outperformed ResNets: it was architecture design, not the attention mechanism.

Cumulative improvement from ResNet-50 baseline → ConvNeXt:

| Modification | Change from transformer recipe | ΔTop-1 |
|---|---|:-:|
| Training recipe | AdamW, cosine LR, RandAugment, Mixup | +2.7% |
| Stage compute ratio | 3:4:6:3 → 3:3:9:3 (more in late stages) | +0.6% |
| Patchify stem | 7×7 stride-2 → 4×4 stride-4 non-overlap conv | +0.3% |
| Depthwise conv | 3×3 conv → depthwise 3×3 | +0.1% |
| Large kernel | 3×3 DW → **7×7 DW** (matching ViT attention span) | +0.3% |
| Inverted bottleneck | Standard → wide-narrow-wide | +0.1% |
| Normalization | BatchNorm → [[normalization-layers\|**LayerNorm**]] (1 per block) | +0.1% |
| Fewer activations | Three ReLUs → **one [[activation-gelu-swish\|GELU]]** per block | +0.1% |

ConvNeXt-T matches Swin-T at identical FLOPs. The critical changes were large depthwise kernels (expanding local context) and LayerNorm (which avoids BN's batch-size dependency and dead neuron issues in small batches).

### Receptive Field Theory: Why Stride Placement Matters

The receptive field formula $\text{RF}_n = 1 + \sum_{i=1}^n (k_i - 1)\prod_{j<i} s_j$ shows that stride in layer $j$ multiplies every subsequent layer's contribution. A stride-2 at layer 1 doubles all later RF contributions; at layer 10, it only doubles the contributions of layers 11 onward.

This implies early striding is highly efficient: the classic 7×7 stride-2 stem (ResNet) gives each layer-1 output unit a 7×7 receptive field, and the subsequent stride-2 pooling expands it to 14×14 — all before any 3×3 convolutions. By contrast, ViT's 16×16 stride-16 patchify stem immediately gives each token a 16×16 field, trading fine spatial resolution for extreme efficiency.

### Architecture Search and the Efficiency Frontier

EfficientNet's baseline (B0) was found by Neural Architecture Search (NAS) on a mobile-size problem. NAS optimized a mixed objective (accuracy × FLOPs^{-0.5}). The key NAS insight is that the building block matters more than the macro-structure: MBConv (inverted residual + SE + depthwise) in B0 was the result.

Later work (EfficientNetV2, 2021) found that early layers should use MBConv (efficient for small feature maps) and later layers should use Fused-MBConv (replaces DW+PW with a single 3×3 to reduce memory access overhead). Training efficiency also matters: large-resolution inputs in early epochs waste time — progressive resizing during training gives 4× speedup at no accuracy cost.

### Revisiting the ResNet vs ViT Debate

At the same dataset size and FLOPs, ResNet variants (particularly ResNet-RS and ResNetV2 with modern training) match or exceed ViT-B/16 trained from scratch on ImageNet-1k. ViT's advantage emerges with:
1. Much larger datasets (ImageNet-21k, JFT-300M)
2. Self-supervised pretraining ([[self-supervised-overview|MAE, DINO]]) which exploits ViT's global attention for better features
3. Tasks requiring global reasoning (VQA, long-range segmentation)

ConvNeXt demonstrated that the right comparison is not CNN vs. transformer but rather: for a given compute budget and dataset size, what architecture family gives the best features? The answer is still task-dependent.

---

*See also: [[arch-residual-block]] · [[arch-bottleneck-1x1]] · [[arch-depthwise-separable]] · [[squeeze-excitation]] · [[feature-pyramid-networks]] · [[vision-transformer]] · [[normalization-layers]] · [[activation-relu-variants]] · [[regularization-dropout]] · [[activation-gelu-swish]] · [[self-supervised-overview]]*
