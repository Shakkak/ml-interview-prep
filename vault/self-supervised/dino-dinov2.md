---
title: DINO and DINOv2 — Self-Distillation for Visual Representations
tags: [dino, dinov2, self-distillation, self-supervised, vision-transformer, multi-crop, ema, register-tokens]
aliases: [DINO, DINOv2, self-distillation, DINO ViT, DINOv2 backbone, register tokens, knowledge self-distillation]
difficulty: 2
status: complete
related: [vision-transformer, self-supervised-overview, contrastive-learning, knowledge-distillation, data-augmentation, clip]
depends_on: [vision-transformer, contrastive-learning, knowledge-distillation]
---

# DINO and DINOv2 — Self-Distillation for Visual Representations

---

## Fundamental

DINO (Self-**Di**stillation with **No** labels, Caron et al. 2021) trains a [[vision-transformer|ViT]] student to match a momentum-averaged teacher's output — with no explicit negative pairs, no labels, and no reconstruction target. The result is an unexpected emergent property: the learned attention maps segment objects without any segmentation supervision.

### Architecture: Student and Teacher

Both student and teacher share the same ViT architecture. The teacher is never directly trained — it is an **exponential moving average (EMA)** of the student:

$$\theta_{\text{teacher}} \leftarrow m\,\theta_{\text{teacher}} + (1-m)\,\theta_{\text{student}}, \quad m \approx 0.996$$

Each network has a projection head $g$ appended after the [CLS] token. The student minimizes the cross-entropy between its softmax output and the teacher's softmax output:

$$\mathcal{L} = -\sum_v \sum_{v' \neq v} p_{\text{teacher}}(x^{v'}) \log p_{\text{student}}(x^v)$$

where $v, v'$ range over augmented views of the same image.

### Multi-Crop Training

DINO uses **multi-crop augmentation**: for each training image, create $2$ large (global) crops and $N$ small (local) crops.

- **Global crops:** scale $s \in [0.4, 1.0]$, resized to $224 \times 224$ — contain most of the object.
- **Local crops:** scale $s \in [0.05, 0.4]$, resized to $96 \times 96$ — contain small fragments.

**Training rule:** the teacher only sees global crops; the student predicts the teacher's output from both global and local crops. This forces the student to predict global object-level semantics from a tiny local patch — impossible without understanding what object the patch belongs to.

> [!tip] Why local-to-global prediction forces object-level understanding ([[contrastive-learning]])
> A local crop of 5% of the image cannot be matched to the teacher's global view by low-level texture matching — the fragment could look similar to patches from entirely different images.
> The only consistent signal that survives across views is the *identity of the object*.
> The model must learn "this fragment is part of object X" to correctly predict the teacher's global representation, which encodes "this image contains object X."
> This implicit constraint acts like the augmentation-invariance training of SimCLR but is far more aggressive: it requires semantic understanding rather than just augmentation robustness.

### Collapse Prevention: Centering

Without safeguards, the teacher and student can collapse to constant output (all tokens map to the same representation). DINO prevents this via **centering**: subtract a running mean from the teacher's output before the softmax:

$$g_{\text{teacher}}(x) \leftarrow g_{\text{teacher}}(x) - c, \quad c \leftarrow m\,c + (1-m)\,\text{mean\_batch}(g_{\text{teacher}})$$

Centering prevents one dimension from dominating but does not prevent uniform output. The combination of centering + sharpening (low softmax temperature $\tau = 0.04$ for teacher, $0.1$ for student) is what keeps the representation informative and non-collapsed.

---

## Intermediate

### Emergent Object Segmentation

The striking result: without any segmentation annotation, DINO ViT attention heads spontaneously specialize. When visualizing the [CLS] token's attention weights to all patch positions:

- Different attention heads focus on semantically distinct object parts (head for background separation, head for limb tracking, head for boundary detection).
- The union of attention maps produces a mask closely aligned with the ground-truth object boundary.

This does not happen with supervised ViT trained on ImageNet labels — the supervision signal collapses all head diversity toward class-discriminative patches. It does not happen with CNN architectures at all — the local receptive field prevents any single channel from computing a global semantic map.

**Why multi-crop is the key ingredient:** ablations show that removing multi-crop and keeping everything else (EMA, centering) eliminates the segmentation property. The local-to-global prediction objective is what forces the model to build spatially coherent object representations rather than texture statistics.

### DINO vs. Other SSL Methods

| Property | SimCLR / MoCo | BYOL | MAE | DINO |
|----------|:---:|:---:|:---:|:---:|
| Needs negatives | ✓ | ✗ | ✗ | ✗ |
| Reconstruction target | ✗ | ✗ | pixels | ✗ |
| Linear probe quality | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ |
| Dense prediction (seg/depth) | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ |
| Emergent segmentation | ✗ | ✗ | ✗ | ✓ |

DINO achieves the best linear probing accuracy among SSL methods before DINOv2 — the [CLS] token is a high-quality global descriptor directly usable with a linear head, no fine-tuning needed.

### DINO Attention as a Feature

The self-attention maps of a DINO-pretrained ViT are directly usable as zero-shot segmentation cues:

```python
# Zero-shot foreground segmentation with DINO
features = dino_model.get_last_selfattention(img)  # [heads, N+1, N+1]
attn_cls = features[:, 0, 1:]  # [CLS] attention to all patches: [heads, N]
attn_map = attn_cls.mean(0).reshape(h_patches, w_patches)
# Threshold → binary foreground mask
```

This zero-shot segmentation outperforms many supervised methods on PASCAL VOC when only 5–10 labelled examples are used for threshold calibration.

---

## Advanced

### DINOv2: Scaling Up with Curated Data

DINOv2 (Oquab et al., 2023) identifies data quality as the primary bottleneck in DINO. Internet-scraped data contains near-duplicates, low-information frames, and domain-biased images that waste compute and produce uneven representations.

**LVD-142M dataset curation:**
1. Start with a seed set of high-quality curated images.
2. Embed all candidate images with a preliminary DINO model.
3. Retrieve the $k$-nearest neighbors of seed images from the candidate pool.
4. Deduplicate via copy-detection to remove near-duplicates.

Result: LVD-142M — 142 million curated images that outperform 1.2 billion uncurated images for representation quality.

**Training changes from DINO:**
- Larger models: ViT-S, B, L, and G (1.1B params).
- Combines DINO (self-distillation) with iBOT (masked patch prediction) as a secondary objective — forces both global (from [CLS]) and local (from patch tokens) representations to be informative.
- Stronger augmentation and longer training.

### Register Tokens: Fixing ViT Attention Artifacts

A known problem in ViT attention maps: certain "background" patches develop **extremely high attention norms** and attract [CLS] attention disproportionately. These high-norm tokens act as "garbage collectors" — patch tokens that encode no local information but carry global contextual information needed elsewhere in the model.

**DINOv2 register tokens:** add $R$ learnable register tokens to the input sequence alongside patches. These tokens can absorb the global information that would otherwise hijack background patch representations. With registers:
- Background patch tokens recover meaningful local features.
- Attention maps become cleaner and spatially coherent.
- Dense prediction tasks improve because patch features now actually reflect local image content.

> [!tip] Why high-norm tokens arise without registers ([[attention-mechanism]])
> In standard ViT, every token must serve two roles: encode local patch information and participate in global information routing.
> For patches in flat background regions (low local information), the model "repurposes" the token as a global memory cell — it discards local information and accumulates global context from other tokens via self-attention.
> Register tokens provide dedicated global memory, so background patches no longer need to be hijacked. This is the same principle as [CLS]: a dedicated token for a specific role prevents other tokens from being repurposed.

### DINOv2 vs CLIP vs MAE as a Frozen Backbone

The choice of pretrained ViT backbone for a downstream task depends on what representation each method optimizes:

| Criterion | DINOv2 | CLIP | MAE |
|-----------|:-------:|:----:|:---:|
| Dense prediction (seg, depth, detection) | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Zero-shot classification | ★★★☆☆ | ★★★★★ | ★☆☆☆☆ |
| Few-shot (linear probe) | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Cross-modal retrieval (image↔text) | ✗ | ✓ | ✗ |
| Fine-tuning on limited data | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Out-of-domain robustness | ★★★★☆ | ★★★★★ | ★★★☆☆ |

**Rules of thumb:**
- **CLIP** if you need text-image alignment, zero-shot classification, or open-vocabulary tasks.
- **DINOv2** if you need a frozen visual backbone for dense prediction (depth estimation, semantic segmentation, instance segmentation) or the best linear probe without any fine-tuning.
- **MAE** if you have labeled data and want to fine-tune — MAE representations are richer in low-level detail (the reconstruction task preserves all spatial information), so fine-tuning on a specific task extracts what is needed.

At sufficient scale, DINOv2-ViT-G/14 with a frozen backbone + linear head achieves competitive performance with full fine-tuning of smaller supervised models on semantic segmentation — a milestone that closed the gap between self-supervised and supervised visual representations.

---

## Links

- [[vision-transformer]] — DINO and DINOv2 use ViT as the backbone; DINO's self-attention maps produce sharp semantic segmentation without any segmentation labels
- [[contrastive-learning]] — DINO avoids explicit negatives by using centering and sharpening; DINOv2 adds iBOT (masked patch token prediction) alongside the self-distillation objective
- [[knowledge-distillation]] — DINO is self-distillation: the student network learns to match the EMA teacher's class token output; this is pseudo-labeling from your own network's predictions
- [[data-augmentation]] — DINO's multi-crop augmentation creates one global view and multiple local views; the student processes local crops and must predict the teacher's global view representation
- [[clip]] — DINOv2 produces better dense prediction features than CLIP; CLIP excels at zero-shot classification while DINOv2 excels at segmentation and depth estimation
- [[attention-mechanism]] — DINO's attention heads develop semantic segmentation spontaneously; the CLS token attends to semantically coherent regions without explicit supervision
- [[arch-positional-encoding]] — DINOv2 registers tokens (special non-semantic tokens) reduce artifacts in attention maps caused by high-norm outlier tokens at specific positions
