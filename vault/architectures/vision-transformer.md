---
title: Vision Transformer (ViT)
tags: [vision-transformer, vit, patch-embedding, class-token, self-supervised, image-classification, dino, mae, swin]
aliases: [ViT, Vision Transformer, patch tokenization, image patches, DeiT, Swin Transformer]
difficulty: 2
status: complete
depends_on: [attention-mechanism, arch-positional-encoding, linear-algebra-fundamentals]
related: [attention-mechanism, arch-positional-encoding, cnn-architectures-guide, knowledge-distillation, clip, self-supervised-overview, contrastive-learning]
---

# Vision Transformer (ViT)

---

## Fundamental

ViT (Dosovitskiy et al., 2020) applies the standard NLP Transformer **unchanged** to sequences of image patches — no convolutions, no image-specific inductive biases. The central hypothesis: with enough data and scale, a general-purpose architecture learns the spatial structure that CNNs have hard-coded.

### Patch Tokenization

Given an image $x \in \mathbb{R}^{H \times W \times C}$, divide it into $N$ non-overlapping patches of size $P \times P$:

$$N = \frac{HW}{P^2}$$

Each patch (flattened to $P^2 C$ values) is linearly projected to the model dimension $d$:

$$z_i = E \cdot \text{flatten}(\text{patch}_i), \quad E \in \mathbb{R}^{d \times P^2C}$$

**ViT-B/16 on 224×224:** $N = 224^2/16^2 = 196$ patches. Each patch contains $16 \times 16 \times 3 = 768$ values projected to $d = 768$. The sequence length is 196 — dramatically shorter than the 50,176 individual pixels, making [[attention-mechanism|transformer attention]] tractable.

### Class Token and Positional Embeddings

A learnable **[CLS] token** is prepended to the patch sequence and attends to all patches through all $L$ layers. Its final representation is fed to the classification head — global image information aggregated without any privileged spatial position.

Positional embeddings $e_i^{pos}$ (1D learned, one per position) are added to patch embeddings before the first layer. Without them, the transformer is permutation-equivariant and cannot distinguish spatial layout.

### Full Forward Pass

```
Image (H×W×C)
  → N patches of P×P×C
  → Linear projection → [N × d]
  → Prepend [CLS] → [(N+1) × d]
  → Add positional embeddings
  → L × Transformer blocks (LN → MHA → residual → LN → FFN → residual)
  → CLS token output → MLP head → logits
```

---

## Intermediate

### ViT Variants

| Model | Layers | Hidden $d$ | Heads | Patch | Params | Notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| ViT-S/16 | 12 | 384 | 6 | 16×16 | 22M | Small baseline |
| ViT-B/16 | 12 | 768 | 12 | 16×16 | 86M | Standard benchmark |
| ViT-L/16 | 24 | 1024 | 16 | 16×16 | 307M | Large, needs 21k |
| ViT-H/14 | 32 | 1280 | 16 | 14×14 | 632M | CLIP image encoder |

Larger patch → fewer tokens → faster but less fine-grained. ViT-B/32 uses 32×32 patches giving $N=49$ — fast but poor at fine-grained tasks.

### Inductive Bias: What ViT Lacks

CNNs have three built-in inductive biases that act as implicit regularization on small datasets:

| Bias | CNN | ViT |
|---|---|---|
| Translation equivariance | Built in (shared weights) | Must be learned |
| Local connectivity | 3×3 kernels — local by design | Global attention from layer 1 |
| Spatial weight sharing | Identical filter at every position | Separate learned weight per position |

On ImageNet-1k (1.28M images), ViT-B/16 trained from scratch underperforms ResNet-50. The biases CNN bakes in are genuinely useful when data is limited — they constrain the function space to image-plausible solutions.

**DeiT** (Touvron et al., 2021) showed that adding a **[[knowledge-distillation|distillation]] token** (trained to mimic a CNN teacher's hard prediction) and aggressive augmentation ([[data-augmentation|MixUp, CutMix, RandAugment]]) lets ViT-B/16 match ResNet on ImageNet-1k alone — bridging the data gap without more data.

### Swin Transformer: Hierarchical ViT

Standard ViT produces a single-resolution feature map (no spatial downsampling through the layers) — unsuitable for dense prediction tasks (detection, segmentation) that need multi-scale features.

Swin (Liu et al., 2021) introduces:
1. **Shifted window attention:** compute attention within non-overlapping 7×7 windows of patches ($O(N)$ complexity vs $O(N^2)$ for global). Adjacent layers use windows shifted by (3,3) patches to allow cross-window information flow.
2. **Hierarchical stages:** patch merging (concatenate 2×2 neighboring patches, then project) halves spatial resolution and doubles channels — exactly like CNN stages. Produces C3–C5-like feature maps suitable for FPN.

Swin-L achieves 86.4% on ImageNet and state-of-the-art on COCO/ADE20K, matching or exceeding CNN-based methods at equivalent compute.

---

## Advanced

### Attention Patterns: What ViT Learns

Early ViT layers exhibit **locally-focused attention** — [CLS] and patch tokens primarily attend to spatially nearby patches, mimicking CNN's local receptive fields. Later layers show **global semantic attention** — [CLS] attends to the most discriminative patches for the class.

**[[self-supervised-overview|DINO]]** (Caron et al., 2021) self-supervised training reveals a striking emergent behavior: without any labels, different attention heads specialize to different object parts. Head 3 might segment foreground from background; head 7 might localize the head of an animal; head 11 might track limb boundaries. The attention maps produce high-quality object segmentation masks despite zero segmentation supervision — evidence that ViT's global attention naturally learns semantic grouping when trained with appropriate self-supervised objectives.

**Why does DINO work?** The self-distillation objective (student's output matches a slow-moving teacher) combined with multi-crop training forces the model to predict consistent global features from local patches — implicitly training the model to associate patches belonging to the same object.

### MAE: Masked Autoencoder Pretraining

MAE (He et al., 2021) pretrains ViT by masking 75% of patches and asking the encoder-decoder model to reconstruct the masked pixel values. Key design decisions:
- **75% mask ratio** (much higher than [[bert-mlm|BERT]]'s 15%): image patches are highly redundant — predicting missing patches from 25% of context is still easy, so a high mask ratio is necessary to create a hard task.
- **Asymmetric architecture:** the encoder operates only on the 25% visible patches (fast). The lightweight decoder reconstructs all 196 patches. The encoder sees far fewer tokens than a standard ViT — training is 3× faster.
- **Pixel space prediction:** predict normalized pixel values directly, not discrete tokens.

MAE-pretrained ViT-H/14 achieves 87.8% on ImageNet with linear probing, matching supervised ConvNets. Fine-tuning reaches 90.0% — the pretraining task is hard enough to learn strong representations.

### Scaling Laws for Vision

ViT scales better than CNNs with both parameters and data:

$$\text{Performance} \approx a - b \cdot N_{\text{data}}^{-\alpha}$$

where $a$ = asymptotic best performance (ceiling), $b$ = a positive constant controlling how far below the ceiling we start, $N_{\text{data}}$ = number of training examples, and $\alpha$ = the scaling exponent (larger $\alpha$ = steeper improvement with more data; ViT's $\alpha$ exceeds CNN's). On JFT-3B (3 billion image-label pairs), ViT-G/14 (1.8B params) achieves 90.4% ImageNet top-1 with fine-tuning — a ceiling CNNs appear unable to reach.

The reason: CNN inductive biases act as a prior that is helpful when data is limited but becomes a **constraint** when data is abundant. ViT can leverage arbitrary relational structure in the image that the CNN's local, translation-equivariant prior would ignore. At 300M+ training examples, the CNN's assumptions are increasingly unnecessary and begin to hurt.

**Practical implication:** for ImageNet-scale fine-tuning, ViT-B/16 pretrained on ImageNet-21k is the standard starting point (86M params, competitive with ResNet-152). For the highest accuracy at any parameter budget, MAE-pretrained ViT-L or ViT-H are currently optimal.

### Positional Embedding Interpolation

ViT learns 1D positional embeddings for $N = 196$ positions during pretraining on 224×224. Fine-tuning at higher resolution (384×384 gives $N = 576$) requires 380 additional position embeddings not seen during training. Standard approach: bicubic interpolation of the 196 learned embeddings to 576 positions, then fine-tune. This works because nearby position embeddings are similar (the model learned a smooth function of position) — the interpolation stays within the learned manifold.

RoPE and Swin's relative position bias handle this more elegantly: they are defined by **relative** patch positions, so they generalize automatically to different resolutions without interpolation.

---

## Links

- [[attention-mechanism]] — ViT applies standard transformer self-attention to patch tokens; each layer mixes information across all patches
- [[arch-positional-encoding]] — ViT uses 2D learned position embeddings; without them, the model cannot distinguish spatial arrangement of patches
- [[linear-algebra-fundamentals]] — patch embedding is a learned linear projection of flattened pixel patches; the class token aggregation is a learned linear operation
- [[cnn-architectures-guide]] — ViT replaces convolutions with attention; CNNs have translation equivariance built in while ViTs learn it from data
- [[knowledge-distillation]] — DeiT trains ViT efficiently using a distillation token that learns from a CNN teacher
- [[clip]] — CLIP trains a ViT image encoder jointly with a text encoder via contrastive loss on image-text pairs
- [[self-supervised-overview]] — DINO and MAE are self-supervised ViT training methods that avoid label requirements
- [[data-augmentation]] — ViTs require stronger data augmentation than CNNs because they lack the inductive biases that make CNNs sample-efficient
- [[bert-mlm]] — Masked Autoencoder (MAE) applies masked prediction to image patches, directly analogous to BERT's masked language modeling
