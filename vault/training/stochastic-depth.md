---
title: Stochastic Depth and DropPath
tags: [stochastic-depth, droppath, residual-networks, regularization, drop-path]
aliases: [stochastic depth, DropPath, layer dropout, survival probability]
difficulty: 1
status: complete
related: [regularization-dropout, arch-residual-block, vit-training-recipe, normalization-layers, bias-variance-double-descent]
depends_on: [regularization-dropout, arch-residual-block, backpropagation]
---

# Stochastic Depth and DropPath

---

## Fundamental

### The Idea

**Stochastic depth (Huang et al., 2016)** randomly drops entire residual blocks during training — a form of dropout at the layer level rather than the neuron level.

For a residual block with input $x$ and output $F(x)$:
$$\tilde{H}(x) = \begin{cases} x + F(x) & \text{with probability } p_l \\ x & \text{with probability } 1 - p_l \end{cases}$$

where $x$ = block input, $F(x)$ = residual transformation, $p_l$ = survival probability of layer $l$ (probability the block is active), $1-p_l$ = drop probability.

When a block is dropped, the input passes through unchanged via the skip connection. The network effectively becomes shallower by a random number of layers each training step.

At inference, all blocks are active (like dropout inference mode), typically with no rescaling needed since the skip connection is always present.

### Survival Probability Schedule

A common choice: **linear decay** — early layers are dropped less, later layers more:

$$p_l = 1 - \frac{l}{L}(1 - p_L)$$

where $L$ is total depth and $p_L$ is the final layer's survival probability (typically 0.7–0.9). This reflects the intuition that early layers learn fundamental features that should be stable, while later layers are more specialized and can be skipped more.

---

## Intermediate

### Why It Helps

**Implicit ensemble:** each training step trains a different sub-network (with different layers dropped). At inference the full network is used, effectively ensembling over all these sub-networks — similar to dropout's ensemble interpretation.

**Gradient flow:** dropping later layers means gradients flow directly from the loss to early layers without passing through all the intermediate layers. This reduces vanishing gradient effects in very deep networks.

**Regularization:** reduces over-reliance on any specific layer or path through the network. Layers must be individually useful even when later layers are absent.

**Training efficiency:** dropped layers mean fewer FLOPs per training step — training is faster. For a 50-layer ResNet with $p_L = 0.5$: expected active layers $\approx 37.5$ (25% speedup).

### DropPath vs Stochastic Depth

**DropPath** is the implementation name in modern frameworks (timm library) — it drops entire sample paths in a batch independently, rather than dropping the same set of layers for all samples in a batch.

**Batch behavior:**
- Stochastic depth (original): same layers dropped for all samples in a batch
- DropPath: each sample in the batch has independently dropped paths

DropPath is strictly more varied (more independence between samples) and is the version used in DeiT, ViT, Swin Transformer.

---

## Advanced

### DropPath Rate in ViT Training

Standard ViT-L with DeiT training recipe uses DropPath rate 0.1–0.4, increasing with model size and depth. For ViT-H, DropPath rates up to 0.5 are used.

The DropPath rate is the probability that a block is **dropped** (survival probability = 1 - DropPath rate). Higher rates = stronger regularization = needed for larger models with limited data.

**Interaction with other regularizers:** DropPath and Mixup/CutMix (see [[mixup-cutmix]]) are complementary:
- DropPath regularizes depth (model learns to work with varying depth)
- Mixup regularizes the input distribution (model learns interpolated decision boundaries)

Both are necessary components of the strong ViT training recipe.

### Relationship to Dropout

| Property | Dropout | DropPath / Stochastic Depth |
|----------|---------|---------------------------|
| What's dropped | Individual neurons | Entire residual blocks |
| Granularity | Fine (per neuron) | Coarse (per layer) |
| Skip connection | None (can't bypass) | Yes (residual passes through) |
| Works without skip | Yes | No (input would be zeros) |
| Main use case | Dense layers, embeddings | Residual networks, transformers |

## Links

- [[regularization-dropout]] — stochastic depth is dropout applied at the layer level; instead of dropping individual neurons, entire residual blocks are dropped with probability $1-p_l$
- [[arch-residual-block]] — stochastic depth requires the residual connection: when a block is dropped, only the skip connection $x$ passes through; this is only possible with residual networks
- [[backpropagation]] — dropped blocks contribute no gradient; survival probability $p_l$ creates an implicit ensemble of networks of varying depth
- [[vit-training-recipe]] — DeiT and DINOv2 use stochastic depth (DropPath) with rate 0.1–0.4; it is the dominant regularizer for training ViTs alongside Mixup/CutMix
- [[data-augmentation]] — stochastic depth is an architectural augmentation (varies the effective network depth); it is complementary to input-level augmentation (Mixup, CutMix)
