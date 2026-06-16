---
title: U-Net — Encoder-Decoder with Skip Connections
tags: [unet, encoder-decoder, skip-connections, segmentation, upsampling, transposed-conv, diffusion]
aliases: [U-Net, UNet, encoder decoder, skip connections, contracting path, expanding path]
difficulty: 2
status: complete
depends_on: [arch-residual-block, feature-pyramid-networks, convolution-math]
related: [arch-residual-block, feature-pyramid-networks, diffusion-models, variational-autoencoders, arch-bottleneck-1x1, vision-transformer]
---

# U-Net — Encoder-Decoder with Skip Connections

---

## Fundamental

Semantic segmentation requires a prediction at every pixel. A standard classification CNN progressively downsamples: 224×224 → 112×112 → 56×56 → 7×7 (stride 32). The final 7×7 feature map contains rich semantic information but has discarded all spatial detail. Recovering pixel-level boundaries from a 7×7 representation requires the network to hallucinate spatial information it deliberately threw away.

**U-Net** (Ronneberger et al., 2015) solves this by preserving spatial information throughout the network with an encoder-decoder structure and skip connections:

```
Encoder (contracting)                     Decoder (expanding)
──────────────────────                    ───────────────────
Input 572×572
  Conv+Conv → features_1 ──────────────→ CONCAT → Conv+Conv → Output
  MaxPool/2
  Conv+Conv → features_2 ──────────→ CONCAT → Up
  MaxPool/2
  Conv+Conv → features_3 ──────→ CONCAT → Up
  MaxPool/2
  Conv+Conv → features_4 ──→ CONCAT → Up
  MaxPool/2
         Bottleneck ──────────────→ Up
```

**Three components:**
1. **Encoder** — repeated [Conv 3×3 → [[activation-relu-variants|ReLU]] → Conv 3×3 → ReLU → MaxPool 2×2], doubling channels and halving spatial resolution at each level.
2. **Bottleneck** — deepest level, highest channel count, lowest resolution.
3. **Decoder** — repeated upsampling with skip connection concatenation, halving channels and doubling spatial resolution.

---

## Intermediate

### Skip Connections: Concatenation, Not Addition

At each resolution level, the encoder's feature map is **concatenated** to the decoder's upsampled output:

$$\text{decoder input at level } l = \text{CONCAT}\!\left[\text{upsample}(\text{decoder}_{l+1}),\ \text{encoder}_l\right]$$

**Why concatenation, not addition?**
- Addition (as in [[arch-residual-block|ResNets]] and [[feature-pyramid-networks|FPN]]) requires matching channel counts — constraining the architecture.
- Concatenation preserves both representations independently. The subsequent 3×3 convolutions learn to combine semantic context (from the deep decoder) with fine spatial detail (from the shallow encoder).

**Information asymmetry:** encoder features at early levels carry high spatial frequency (edges, corners, texture boundaries) that vanish in the bottleneck. Decoder features carry low spatial frequency semantic context. The skip connection is the only pathway for high-frequency information to survive — without it, the decoder reconstructs boundaries by pattern completion, not by reading them.

### Upsampling Methods

**Transposed convolution (learned upsampling):** maps each input activation to a $k \times k$ output region via learned weights, summing overlapping regions. Stride $s$ gives $H \times W \to sH \times sW$. Expressive — the network learns task-specific upsampling. **Checkerboard artifacts** appear when the kernel size is not divisible by stride (e.g., 3×3 kernel, stride 2): overlapping regions receive contributions from different numbers of input activations, producing a regular grid pattern in the output.

**Bilinear upsampling + conv (preferred):** upsample with fixed bilinear interpolation (no learned parameters), then apply a standard 3×3 conv for learned refinement. Avoids checkerboard artifacts. Slightly less expressive but far more stable. DeepLab, FPN, and most modern segmentation heads use this.

### U-Net Variants

| Variant | Key change | Use case |
|---|---|---|
| Original U-Net (2015) | No padding → output smaller than input (572→388) | Biomedical, few training images |
| Modern U-Net | Same padding, BatchNorm → input/output same size | General segmentation |
| U-Net++ | Dense nested skip connections at every decoder level | Maximum accuracy, higher params |
| ResU-Net | ResBlocks replace plain conv stacks | Gradient flow, very deep versions |
| SegFormer | [[vision-transformer\|ViT encoder]] + lightweight MLP decoder | State-of-the-art semantic seg |

---

## Advanced

### U-Net in Diffusion Models

Denoising diffusion probabilistic models (DDPM, Ho et al., 2020) use a U-Net as their noise predictor $\epsilon_\theta(x_t, t)$ — trained to predict the noise added to $x_0$ to produce $x_t$ at timestep $t$. The U-Net architecture is particularly suited because:

1. **Multi-scale processing:** diffusion must denoise at all spatial frequencies simultaneously — coarse structures (low frequency) and fine textures (high frequency) need different treatment at different resolutions. U-Net's skip connections naturally implement this.

2. **Time conditioning:** the scalar timestep $t$ is embedded via [[arch-positional-encoding|sinusoidal encoding]] followed by a 2-layer MLP. The embedding is injected into every residual block via AdaIN-style affine transform — scale $\gamma(t)$ and shift $\beta(t)$ are applied after [[normalization-layers|GroupNorm]]:
   $$\text{output} = \gamma(t) \cdot \text{GroupNorm}(h) + \beta(t)$$

> [!tip] AdaIN-style affine transform
> Adaptive Instance Normalization (AdaIN) applies a learned scale and shift after normalizing — the same mechanism used in StyleGAN for style conditioning. Here the conditioning signal is the timestep embedding instead of a style vector, letting the denoiser adapt its behavior at each noise level.

3. **Cross-attention for text conditioning:** in text-to-image models (Stable Diffusion), text embeddings from a frozen [[clip|CLIP]]/T5 encoder are injected via [[attention-mechanism|cross-attention]] in the bottleneck and decoder blocks. The U-Net's spatial features serve as queries; the text tokens provide keys and values.

   The AdaIN-style time injection formula is: $\text{output} = \gamma(t) \cdot \text{GroupNorm}(h) + \beta(t)$, where $h$ = intermediate feature map, $\text{GroupNorm}(h)$ = the normalized feature map (zero mean, unit variance per group), $\gamma(t), \beta(t) \in \mathbb{R}^C$ = scale and shift vectors computed from the timestep embedding $t$ via a learned 2-layer MLP, and $C$ = number of channels (applied channel-wise to broadcast across spatial dimensions).

Stable Diffusion's U-Net operates in the 64×64 latent space of a [[variational-autoencoders|VAE]] (not pixel space), has 860M parameters, and uses ResBlocks + self-attention at the 16×16 and 8×8 levels + cross-attention for text.

### Why Skip Connections Prevent Hallucination

Consider what the decoder must do without skip connections: recover fine spatial detail from the bottleneck alone. The bottleneck of a 5-level U-Net is $7 \times 7$ for a $224 \times 224$ input. Each bottleneck spatial cell covers a $32 \times 32$ pixel region. Precise object boundaries — which can be sub-pixel — cannot be reconstructed from this coarse grid.

With skip connections, the decoder at the finest level receives direct access to encoder features with stride 1. The question becomes not "where is the boundary?" (the encoder knows) but "does this boundary separate foreground from background in the current semantic context?" (the decoder knows). The two streams of information are complementary; the convolution after concatenation learns to fuse them.

This is why U-Net was so successful in biomedical imaging where training data is scarce: the skip connections provide a strong structural prior that constrains the decoder to produce spatially coherent predictions.

### U-Net vs FPN: Architectural Comparison

| | U-Net | FPN |
|--|---|---|
| Skip connection mechanism | Concatenation (preserves both representations) | Addition (after 1×1 alignment conv) |
| Decoder depth | Full symmetric decoder (many conv layers) | Lightweight top-down pathway |
| Output | Single resolution (full or near-full) | Multi-scale pyramid (P2–P5) |
| Downsampling in decoder | None — output is full resolution | Not applicable (multi-scale outputs) |
| Primary task | Dense prediction per pixel | Scale-specific detection anchors |
| Information flow | Bidirectional per level | Top-down only |

FPN's addition-based merging is more parameter-efficient; U-Net's concatenation-based merging is more expressive. DeepLabV3+ combines both: FPN-style top-down pathway for multi-scale context (ASPP) + U-Net-style decoder for sharp boundaries at fine resolution.

---

## Links

- [[arch-residual-block]] — modern U-Nets use residual blocks instead of plain convolutions in both the encoder and decoder paths
- [[feature-pyramid-networks]] — FPN and U-Net share the same top-down + lateral connection design; FPN is optimized for detection, U-Net for dense pixel-level prediction
- [[convolution-math]] — U-Net uses transposed convolutions (deconvolutions) in the decoder path for spatial upsampling; the operation is the transpose of a strided conv
- [[diffusion-models]] — the denoising U-Net is the standard noise-prediction backbone; its skip connections preserve spatial structure across noise levels
- [[variational-autoencoders]] — Latent Diffusion Models (Stable Diffusion) use a U-Net denoiser operating in VAE latent space, not pixel space
- [[attention-mechanism]] — attention-augmented U-Nets insert self-attention at low-resolution bottleneck levels for long-range spatial coherence
- [[normalization-layers]] — Group Normalization replaced Batch Normalization in medical image U-Nets where small batch sizes make BN statistics noisy
- [[activation-relu-variants]] — diffusion U-Nets use SiLU (Swish) instead of ReLU; the smooth non-linearity improves gradient flow through the deep encoder-decoder
