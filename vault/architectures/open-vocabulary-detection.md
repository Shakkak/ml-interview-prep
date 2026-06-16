---
title: "Open-Vocabulary Detection and Segmentation"
tags: [open-vocabulary-detection, grounding-dino, owl-vit, sam, glip, ovdet, zero-shot-detection, language-grounding]
aliases: [Open-Vocabulary Object Detection, OVD, Language-Grounded Detection, Grounding DINO, OWL-ViT, SAM, GLIP, DETIC, OvSeg]
difficulty: 3
status: complete
depends_on: [vision-transformer, attention-mechanism, clip]
related: [feature-pyramid-networks, iou-nms, vlm-architectures, anchor-free-detection]
---

# Open-Vocabulary Detection and Segmentation

---

## Fundamental

**The problem:** classical object detection (YOLO, Faster R-CNN, DETR) trains on a fixed set of categories. COCO has 80. If you want to detect "Quercus robur" (English oak) at test time, you cannot — unless oak was in the training set. Every new category requires re-annotation and retraining.

**Open-vocabulary detection (OVD)** removes this constraint. At test time you provide an image and a text prompt ("a plant with elongated leaves"); the model localizes objects matching the description, using visual-language alignment learned from large-scale image-text data.

**Intuition:** CLIP (see [[clip]]) already knows how to score image patches against text. Open-vocabulary detection is CLIP with localization added — instead of scoring the whole image, score each candidate region separately, then output boxes for the regions whose text-similarity exceeds a threshold.

### GLIP: Region-Level Vision-Language Alignment

**GLIP (Li et al., 2022)** reframes detection as phrase grounding: if you have pairs of (region, noun phrase), you can train with contrastive loss to align them — no fixed category list needed.

Architecture: image features (Swin Transformer) and phrase embeddings ([[bert-mlm|BERT]]-style encoder) interact via cross-attention, then each proposal region $r_i$ and phrase $t_j$ are scored:

$$s_{ij} = \frac{r_i \cdot t_j}{\|r_i\| \|t_j\|}$$

where $r_i \in \mathbb{R}^d$ = visual feature vector for proposal region $i$ (extracted by the image backbone and cross-attended with text), $t_j \in \mathbb{R}^d$ = text embedding for noun phrase $j$ (from the language encoder), and $s_{ij} \in [-1, 1]$ = cosine similarity used as the grounding score. The contrastive grounding loss maximizes $s_{ij}$ for matching (region, phrase) pairs and minimizes it for all mismatches. Pretrained on ~27M image-region-text pairs (COCO, Objects365, Flickr30K, CC3M).

---

## Intermediate

### Grounding DINO: Cross-Modal Fusion for Detection

**Grounding DINO (Liu et al., 2023)** marries a DINO DETR-style detector with deep text-image fusion. The key innovation is bidirectional cross-attention between image and text features at every encoder stage:

```
Feature enhancer block:
  I' = I + CrossAttn(Q=I, KV=T)   # image features attend to text
  T' = T + CrossAttn(Q=T, KV=I)   # text features attend to image
```

Object queries are initialized from text tokens — so the decoder already knows what to look for. Output: one bounding box per object query, scored by cosine similarity between the query embedding and text tokens.

**Intuition:** the visual representation is conditioned on what you are searching for. "Healthy oak tree" and "diseased leaf" produce different visual features from the same image. This conditioning is absent in OWL-ViT, which uses a frozen CLIP encoder.

### OWL-ViT: Simplest Open-Vocabulary Detector

**OWL-ViT (Minderer et al., 2022):** take a [[clip|CLIP]] ViT, remove the global pooling, add a detection head per patch token, and compute detection scores as cosine similarity between patch tokens and text embeddings.

```
Image → CLIP ViT → patch tokens {p_i}
Text prompts → CLIP text encoder → {t_j}
Score: s_ij = p_i · t_j   (per patch, per class)
Box:   linear layer on p_i → (cx, cy, w, h)
```

**Strengths:** very simple — no cross-modal fusion layer needed since CLIP already aligned the spaces. Fast at inference.

**Weakness:** CLIP was trained for image-level understanding, not localization — patch tokens are not optimized to represent individual objects. Lags behind Grounding DINO on standard benchmarks. OWLv2 (2023) scales the dataset and closes most of this gap.

### SAM: Segment Anything Given a Prompt

**SAM (Kirillov et al., 2023)** does not detect objects — it segments a prompted region at high quality. Prompts: points, bounding boxes, or masks.

**Architecture:**
- Image encoder: MAE-pretrained ViT-H (632M params), runs once per image
- Prompt encoder: lightweight, encodes points/boxes/masks into prompt tokens
- Mask decoder: cross-attention between prompt tokens and image embedding → 3 candidate masks + confidence scores

**Intuition:** by encoding the image once and making the decoder cheap, SAM enables interactive use — encode once, prompt thousands of times at millisecond speed.

**SAM2 (2024)** extends to video via a streaming memory module that tracks objects across frames.

---

## Advanced

### DETIC: 21K-Category Detection from Image-Level Labels

**DETIC (Zhou et al., 2022)** solves the annotation bottleneck. ImageNet-21K has 21K categories with image-level labels but no boxes. Key idea: use the largest detected proposal region as a pseudo bounding box for image-level categories. Train the classifier jointly on detection data (with real boxes) and classification data (with max-size-proposal pseudo boxes). Result: a detector for 21K+ categories without 21K annotation campaigns.

### OvSeg: Open-Vocabulary Semantic Segmentation

**Problem:** CLIP classifies whole images. Segmentation needs pixel-level classification, but masking a region before CLIP creates background leakage that confuses the image encoder.

**OvSeg (Liang et al., 2022):**
1. Generate class-agnostic mask proposals (Mask2Former on any segmentation dataset)
2. Crop each proposed mask region, removing background
3. Fine-tune CLIP on masked images ("mask-adapted CLIP") to fix background leakage
4. Score each crop against text category embeddings → keep highest-scoring (mask, category) pair

### Combining the Models

Full pipeline for species detection from drone imagery:

```
1. Grounding DINO("individual plant") → bounding boxes
2. SAM(image, boxes as SAM prompts) → precise instance masks
3. CLIP(cropped instances, species text gallery) → species labels
```

**Failure modes:**
- **Fine-grained confusion:** "oak" vs "beech" — similar visual texture; language grounding fails without fine-grained training data
- **Scale mismatch:** models trained on ground-level images underperform on top-down aerial views
- **Prompt sensitivity:** "a leaf" vs "leaves" vs "plant foliage" produce different results; treat prompt wording as a hyperparameter

| Model | Input | Output | Open-Vocab | Key Strength |
|-------|-------|--------|:---:|------|
| Grounding DINO | Image + text | Boxes | ✓ | Best zero-shot detection accuracy |
| OWL-ViT | Image + text list | Boxes | ✓ | Simplest, no fine-tuning needed |
| SAM | Image + point/box | Masks | ✗ (no text) | Precise, interactive segmentation |
| DETIC | Image + vocab | Boxes | ✓ | 21K category support |
| OvSeg | Image + class list | Semantic masks | ✓ | Dense prediction |

---

## Links

- [[vision-transformer]] — OWL-ViT and Grounding DINO use ViT as the image backbone; patch tokens provide per-region features for scoring against text
- [[clip]] — CLIP provides the text-image alignment that enables zero-shot category generalization; most open-vocabulary detectors build on or initialize from CLIP
- [[attention-mechanism]] — cross-attention between text embeddings and image patch tokens is the core mechanism in Grounding DINO for aligning language concepts with spatial regions
- [[vlm-architectures]] — open-vocabulary detectors are specialized VLMs: they consume text category queries and produce bounding boxes rather than free-form text
- [[feature-pyramid-networks]] — Grounding DINO uses a Deformable DETR backbone with FPN-style multi-scale feature extraction
- [[iou-nms]] — most open-vocabulary detectors still use IoU-based NMS for post-processing; DETR-based detectors replace NMS with learned bipartite matching
- [[anchor-free-detection]] — DETR-based open-vocabulary detectors (Grounding DINO) use anchor-free object queries rather than anchor boxes
