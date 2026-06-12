---
title: Masked Autoencoders (MAE)
tags: [mae, masked-autoencoder, self-supervised, vision-transformer, masked-image-modeling]
aliases: [MAE, masked autoencoder, masked image modeling, MIM, BEiT]
difficulty: 2
status: complete
related: [vision-transformer, bert-mlm, self-supervised-overview, dino-dinov2, contrastive-learning, vit-training-recipe]
---

# Masked Autoencoders (MAE)

---

## Fundamental

### Core Idea

**MAE (He et al., 2022)** adapts BERT-style masked prediction to images. Random patches of the input image are masked, and the model learns to reconstruct the missing pixels.

**Architecture:**
- **Encoder (ViT):** sees only the unmasked patches ($\sim$25% of the image), processes them with full self-attention. No mask tokens in the encoder — this keeps the encoder computationally cheap.
- **Decoder (shallow transformer):** takes encoded unmasked patches + learnable mask tokens at masked positions, predicts raw pixel values for all masked patches.

At inference, only the encoder is used — the decoder is discarded.

### Why High Masking Ratio?

MAE uses a **75% masking ratio** — 3× higher than BERT's 15%. This is key:

- Low masking (15%): task is too easy, the model can inpaint from nearby context without semantic understanding
- High masking (75%): forces the model to learn holistic structure — can't reconstruct a masked eye from the face without understanding "face"

The high ratio also makes encoding cheap (only 25% of patches pass through the encoder) and makes the task non-trivially redundancy-eliminating.

---

## Intermediate

### Reconstruction Target

MAE predicts raw **pixel values** (not discrete tokens or features). Each masked patch's pixel values are normalized by the patch's mean and standard deviation. The loss is MSE over masked patches only:

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|p_i - \hat{p}_i\|^2$$

**BEiT (Bao et al., 2022)** predicts discrete visual tokens (DALL-E tokenizer) instead of pixels — similar performance but requires a separate tokenizer.

**MAE vs BEiT:**
- MAE is simpler (no separate tokenizer)
- Both achieve similar downstream accuracy
- MAE features may be less semantically localized; BEiT features follow discrete token semantics

### MAE Training Properties

**Asymmetric encoder-decoder:** the encoder is large (ViT-L/16), the decoder is small (4–8 transformer layers, 512 dim). The encoder doesn't see mask tokens → the encoder processes 4× fewer tokens → training is ~3× faster than full-image ViT.

**Pre-training recipe (ImageNet):** 1600 epochs, batch 4096, AdamW, cosine LR schedule with warmup. Linear probe: ~75% top-1 (ViT-L). Fine-tuned: ~85% top-1 — comparable to supervised ViT-L with far less supervision.

### Comparison to Contrastive Methods

| Property | MAE | SimCLR/DINO |
|----------|-----|-------------|
| Augmentation design | Minimal (just masking) | Extensive (crop, flip, color jitter) |
| Architecture | Encoder + discarded decoder | Twin encoders / teacher-student |
| Batch size | 4096 (flexible) | 4096+ (needs diverse negatives/views) |
| Training time | Fast (sparse encoder) | Slower |
| Feature quality (linear probe) | Lower | Higher |
| Feature quality (fine-tuned) | Equal or better | Equal |

*Contrastive methods tend to produce better features for frozen linear probing; MAE features are better after fine-tuning on small datasets.*

---

## Advanced

### Extensions

**VideoMAE (Tong et al., 2022):** apply MAE to video with tube masking (mask the same spatial patch across frames). 90% masking ratio is needed because video has high temporal redundancy. Strong results for action recognition.

**AudioMAE:** mask mel spectrogram patches. The high redundancy of audio benefits from the same approach.

**Point-MAE:** mask point cloud groups for 3D understanding.

**MAE with semantic targets (MaskFeat):** predict HOG (Histogram of Oriented Gradients) features instead of pixels — a middle ground between pixel reconstruction and discrete tokens.

### Why MAE Works: Feature Analysis

Visualization reveals:
- **Shallow MAE features:** capture low-level texture (edges, colors)
- **Deep MAE features:** capture semantic content (object parts, scene structure)
- **Compared to supervised ViT:** MAE features are slightly more texture-biased; supervised features are slightly more shape-biased

**Reconstruction as a regularizer:** the pixel reconstruction objective prevents representation collapse (unlike contrastive methods without careful design) and encourages local-to-global coherence.

*See also: [[vision-transformer]] · [[bert-mlm]] · [[self-supervised-overview]] · [[dino-dinov2]] · [[vit-training-recipe]]*
