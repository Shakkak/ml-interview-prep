---
title: "Open-Vocabulary Detection and Segmentation"
tags: [open-vocabulary-detection, grounding-dino, owl-vit, sam, glip, ovdet, zero-shot-detection, language-grounding]
aliases: [Open-Vocabulary Object Detection, OVD, Language-Grounded Detection, Grounding DINO, OWL-ViT, SAM]
---

Open-vocabulary detection lets you detect, localize, and segment objects specified by arbitrary text prompts at test time — without retraining. This file covers the core models from first principles: what problem they solve, how they work, and their strengths and weaknesses for ecological applications.

## 1. The Problem: Closed vs Open Vocabulary

**Classical object detection** (YOLO, Faster R-CNN, DETR) trains on a fixed set of categories defined at training time. COCO has 80 categories. If you want to detect "Quercus robur" (English oak) at test time, you cannot — unless oak was in the training set.

**Open-vocabulary detection (OVD)** removes this constraint. At test time you provide:
- An image
- A text prompt: "a plant with elongated leaves" or "an individual oak tree"

The model localises objects matching the description, using visual-language alignment learned from large-scale image-text data.

**Why this matters for ecology**: species descriptions (or simply species names) become queries. No need to retrain for new species or describe novel phenotypes.

---

## 2. GLIP: Grounding Language in Image Pairs

**Paper**: "GLIP: Grounded Language-Image Pre-Training" (Li et al., 2022, Microsoft)

### 2.1 Core Insight: Unify Detection and Phrase Grounding

Detection and phrase grounding (linking noun phrases to image regions) are the same task at different levels of abstraction. GLIP treats both as *image-text alignment at the region level*.

### 2.2 Architecture

```
Image → Feature Extractor (Swin Transformer) → region features {r_i}
Text → [[bert-mlm|BERT-style encoder]] → phrase embeddings {t_j}

Grounding score: S_{ij} = r_i · t_j  (dot product)
Loss: phrase-region contrastive loss + detection localization loss
```

For each proposal region $r_i$ and text phrase $t_j$:
$$s_{ij} = \frac{r_i \cdot t_j}{\|r_i\| \|t_j\|}$$

**Contrastive grounding loss**: for a ground-truth (region $i$, phrase $j$) pair, maximize $s_{ij}$ and minimize $s_{ij'}$ for all non-matching phrases $j'$.

### 2.3 Training at Scale

GLIP is pretrained on:
- Detection data (COCO, Objects365) — with category names as text
- Phrase grounding data (Flickr30K Entities) — noun phrases linked to bounding boxes
- Image-caption data (CC3M) — weakly supervised

Total: ~27M image-region-text pairs. This teaches the model to associate region appearances with free-form language.

### 2.4 GLIPv2

Extends GLIP to also produce segmentation masks alongside the bounding boxes (similar to mask R-CNN head on top). Also improves the cross-modal fusion.

---

## 3. Grounding DINO

**Paper**: "Grounding DINO: Marrying DINO with Grounded Pre-Training" (Liu et al., 2023)

### 3.1 DINO Background

DETR-style detectors (DINO, DN-DETR) use:
- [[attention-mechanism|Transformer encoder]] (cross-attention between image tokens)
- A set of learned object queries
- Transformer decoder: queries attend to image tokens to predict box + class

DINO (2022) is state-of-the-art for closed-vocabulary detection. Grounding DINO adapts it for open-vocabulary.

### 3.2 Architecture

```
Image → Swin Transformer backbone → multi-scale image features
Text  → BERT encoder → text features

Cross-Modal Fusion (key innovation):
  Image features ↔ Text features via cross-attention (repeated K times)
  → fused features align visual and linguistic information

Object Queries:
  - Initialized from text tokens (feature-enhanced query selection)
  - Queries attend to fused image features in the decoder

Output per query:
  - Bounding box (cx, cy, w, h)
  - Grounding score: cosine similarity between query embedding and text tokens
```

### 3.3 Cross-Modal Fusion — the Key Innovation

Grounding DINO inserts cross-attention between image and text features at multiple stages:

```
Image encoder layer → image features I
Text encoder layer → text features T

Feature enhancer block:
  I' = I + CrossAttn(Q=I, KV=T)   # image features attend to text
  T' = T + CrossAttn(Q=T, KV=I)   # text features attend to image
```

This bidirectional fusion means the image features are conditioned on the text and vice versa. The visual representation changes depending on what you're looking for.

**Intuition**: if you search for "healthy oak tree", the text "oak" + "healthy" conditions the visual features to attend to appropriate textures and shapes. If you search for "diseased leaf", the same image produces different visual representations.

### 3.4 Open-Vocabulary Detection at Inference

```python
from groundingdino.util.inference import load_model, predict

model = load_model(config_path, weights_path)
boxes, logits, phrases = predict(
    model=model,
    image=image_tensor,
    caption="individual plant . leaf . flower",  # dot-separated prompts
    box_threshold=0.35,
    text_threshold=0.25
)
```

Returns bounding boxes and which phrase each box matches.

---

## 4. OWL-ViT: Simple Open-Vocabulary Detection with Vision Transformers

**Paper**: "Simple Open-Vocabulary Object Detection with Vision Transformers" (Minderer et al., 2022, Google)

### 4.1 Architecture

OWL-ViT is conceptually the simplest open-vocabulary detector:

```
Image → CLIP ViT encoder → patch tokens + [CLS] token
Text prompts → CLIP text encoder → text embeddings {t_j}

For each image patch token p_i:
  Detection score for class j: s_ij = p_i · t_j  (cosine similarity)
  Box prediction: linear layer on p_i → (cx, cy, w, h)
```

That's it. Take [[clip|CLIP]], remove the pooling, add a detection head per patch token, compute similarity against text embeddings.

### 4.2 Strengths and Weaknesses

**Strengths**: very simple, no cross-modal fusion needed (CLIP already aligned), fast.

**Weaknesses**: CLIP was trained for image-level understanding, not localization — patch tokens are not optimized for object detection. Performance lags behind Grounding DINO on standard benchmarks.

**OWLv2** (2023): scales the approach to much larger datasets. Competitive with Grounding DINO on COCO zero-shot benchmarks.

---

## 5. SAM: Segment Anything Model

**Paper**: "Segment Anything" (Kirillov et al., 2023, Meta AI)

### 5.1 What SAM Does

SAM does not detect objects — it *segments* them given a prompt. Prompts can be:
- Points (click on the object you want segmented)
- Bounding boxes (the rectangle is the "roughly where it is" signal)
- Text (experimental, via SAM + CLIP)
- A previous mask

SAM outputs a high-quality binary mask for the prompted region.

### 5.2 Architecture

```
Image → Image Encoder (MAE-pretrained ViT-H, 632M params) → image embedding
Prompt (points / box / mask) → Prompt Encoder → prompt embedding

Lightweight Mask Decoder:
  Cross-attention between prompt tokens and image embedding
  Outputs: 3 candidate masks (handles ambiguity) + confidence scores
```

The image encoder runs once per image (expensive: ~15 GB VRAM for [[vision-transformer|ViT-H]]). The decoder is fast (runs in milliseconds per prompt). This design enables interactive use: encode once, prompt many times.

### 5.3 SA-1B Dataset

SAM was trained on a dataset of 1.1 billion masks collected by:
1. Train SAM on expert-annotated masks
2. Use trained SAM to propose masks on new images (automated)
3. Human annotators correct/accept the proposals (semi-automated)
4. Iterate → 1.1B masks on 11M images

This scale explains SAM's remarkable generalization.

### 5.4 SAM2 (2024)

Extends SAM to video:
- Adds a streaming memory module to track objects across frames
- Handles occlusion by maintaining a memory of object appearance

Useful for drone video: annotate a plant in frame 1, track it through the entire clip.

### 5.5 SAM + Language: Segment and Name

SAM itself has no semantic understanding. To get "what is this?", combine:

1. **SAM + CLIP**: SAM generates all masks in an image → crop each mask → pass to CLIP → score against text descriptions of species.
2. **SAM + Grounding DINO**: Grounding DINO detects objects from text prompt → use predicted boxes as SAM prompts → get precise masks.
3. **Segment Anything with Text (SAT)**: fine-tune SAM with text-conditioned prompts.

This combination is powerful for ecological imagery: Grounding DINO detects "individual plant with visible flower", SAM segments the precise plant mask, then classify via CLIP.

---

## 6. DETIC: Detecting Twenty-Thousand Classes

**Paper**: "Detecting Twenty-Thousand Classes Using Image-Level Supervision" (Zhou et al., 2022)

DETIC addresses a practical problem: most detection datasets have <200 categories, but image classification datasets (ImageNet-21K) have 21,000 categories.

**Key idea**: for categories that appear in image-level labels but not in detection annotations, use the max-size proposal (the largest detected region) as a pseudo ground truth bounding box. Train the classification head on both detection data (with boxes) and classification data (with max-proposal).

This allows training a detector on 21K ImageNet categories without needing 21K manually annotated bounding boxes. The resulting classifier is then an open-vocabulary classifier (text embedding space).

---

## 7. Open-Vocabulary Segmentation

### OvSeg

**Paper**: "Open-Vocabulary Semantic Segmentation with Mask-Adapted CLIP" (Liang et al., 2022)

Problem: CLIP classifies whole images. For semantic segmentation, you need pixel-level labels.

Approach:
1. Use class-agnostic mask proposals (from Mask2Former trained on COCO or any segmentation dataset)
2. Crop each proposed mask region from the image
3. Pass each crop to CLIP → class score for each text category
4. Keep highest-scoring (mask, category) pairs

The key insight: CLIP's image-level features are misaligned with masked region crops (background leaks into the crop). OvSeg fine-tunes CLIP on masked images with random background removed → "mask-adapted CLIP".

---

## 8. Practical Comparison for Ecological Applications

| Model | Input | Output | Open-Vocab? | Typical Use |
|-------|-------|--------|-------------|-------------|
| **Grounding DINO** | Image + text query | Bounding boxes | ✅ | "Detect all individual oak trees" |
| **OWL-ViT** | Image + text list | Bounding boxes | ✅ | Rapid multi-class detection |
| **SAM** | Image + point/box | Masks | ❌ (no text) | Precise plant segmentation |
| **SAM + GDINO** | Image + text | Masks + labels | ✅ | Full pipeline: detect + segment |
| **DETIC** | Image + text vocab | Boxes + labels | ✅ | Large vocabulary detection |
| **OvSeg** | Image + class list | Semantic seg. | ✅ | Land cover semantic segmentation |

**Recommended pipeline for plant/species detection from drone imagery:**
```
1. Grounding DINO("individual plant") → bounding boxes
2. SAM(image, boxes as prompts) → instance masks
3. CLIP(cropped instances, species text gallery) → species labels
```

---

## 9. Key Failure Modes

1. **Fine-grained confusion**: "oak tree" vs "beech tree" — similar visual appearance, language grounding can confuse them. Need fine-tuning on fine-grained data.
2. **Scale sensitivity**: SAM and detection models expect objects of "typical" size. Satellite imagery has objects at completely different scales — need scale-aware approaches.
3. **Aerial perspective**: models trained on ground-level images may fail on top-down drone imagery. OWL-ViT inherits CLIP's bias toward eye-level images.
4. **Text prompt engineering**: model performance varies significantly with prompt wording. "a leaf" vs "leaves" vs "plant foliage" can produce very different results.
5. **Overlapping instances**: dense vegetation patches — individual plant segmentation is hard when plants overlap heavily.

---

## See Also

[[vlm-architectures]], [[clip]], [[vision-transformer]], [[biodiversity-ml]], [[remote-sensing-foundation-models]]
