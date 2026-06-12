---
title: "Remote Sensing Foundation Models"
tags: [remote-sensing, foundation-models, remote-clip, prithvi, geochat, satmae, scale-mae, geospatial-fm, earth-observation-ml]
aliases: [EO Foundation Models, Geospatial Foundation Models, RemoteCLIP, Prithvi, GeoChat]
status: complete
---

Foundation models are large neural networks pretrained on broad data that can be fine-tuned for many downstream tasks. In Earth observation (EO), standard ImageNet-pretrained models underperform because satellite imagery differs fundamentally from natural photos. This file covers why domain-specific pretraining matters and the key models built for geospatial data.

## 1. Why Standard Foundation Models Fail at EO

### 1.1 Distribution Shift

ImageNet contains photos taken from eye level, with typical subjects (animals, objects, people) in natural lighting. Satellite imagery is:
- Top-down perspective
- Multi-spectral (more than RGB)
- Objects at very different scales (individual plants, cities, continents in the same dataset)
- Contains physical signals (reflectance, radar backscatter) rather than perceptual color

A [[clip|CLIP]] [[vision-transformer|ViT-L]] pretrained on 400M image-text pairs from the internet knows what "forest" looks like from a hiking trail photo. Its representation of a Sentinel-2 false-color composite of a boreal forest is far weaker.

### 1.2 Spectral Mismatch

CLIP processes 3-channel RGB. Sentinel-2 has 13 bands including NIR and SWIR, which carry most of the ecologically relevant information (vegetation health, moisture, mineral content). Standard models cannot ingest these channels without modification.

### 1.3 Temporal Structure

Satellite data has a temporal dimension — the same location imaged weekly for years. This temporal structure is invisible to standard image models. Phenological monitoring, crop yield forecasting, and forest disturbance detection require temporal representations.

---

## 2. Pretraining Strategies for EO Foundation Models

### 2.1 Masked Image Modeling (MAE-based)

Masked Autoencoders (MAE) randomly mask patches of an image and reconstruct them. For EO:
- Works on raw multispectral patches — no need for text captions
- [[self-supervised-overview|Self-supervised]]: uses only the unlabeled satellite imagery
- Billions of satellite patches are freely available (Sentinel-2, Landsat)

The encoder learns representations that capture spectral-spatial patterns useful for downstream tasks.

### 2.2 Contrastive Learning for EO

Adapted from CLIP: instead of image-text pairs, use:
- Same location at different times (temporal augmentation)
- Same location in different modalities (optical + SAR)
- Image + geolocation text ("a satellite image of a temperate broadleaf forest in central Europe")

### 2.3 Scale-Aware Pretraining

Scale is a unique challenge in EO: a tree at 10 m resolution looks like a single pixel. Scale-MAE explicitly conditions on the ground sample distance (GSD) — the physical size of each pixel — during pretraining, so the model understands scale context.

---

## 3. Key Models

### 3.1 SatMAE (Satellite Masked Autoencoders)

**Paper**: "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery" (Cong et al., 2022)

**What it does**: MAE pretraining adapted for satellite imagery with two key modifications:
1. **Temporal encoding**: position encodings include a time index — the model sees the same location at multiple dates and must reconstruct masked patches
2. **Spectral tokenization**: rather than concatenating all bands as channels, group bands by spectral similarity and tokenize each group separately (spectral-group tokens)

**Training data**: fMoW (Functional Map of the World) — satellite images of 84 land use categories across the globe.

**Results**: fine-tuning SatMAE on downstream tasks outperforms ImageNet-pretrained ViT by significant margins on geospatial benchmarks.

### 3.2 Scale-MAE

**Paper**: "Scale-MAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning" (Reed et al., 2023)

**Core idea**: in satellite imagery, the same type of object (e.g., a crop field) appears at dramatically different sizes depending on sensor resolution. Models must be scale-aware.

**Architecture change**: the image encoder receives the Ground Sample Distance (GSD) as an additional input. Position encodings are scaled by GSD:
$$PE(pos) = PE_{base}(pos \cdot GSD)$$

This allows the model to understand "this pixel represents 10 m × 10 m" vs "this pixel represents 1 m × 1 m".

**Decoder**: uses a Laplacian pyramid loss that explicitly reconstructs at multiple scales.

**Results**: state-of-the-art on cross-resolution transfer — a model pretrained on 10 m Sentinel-2 can fine-tune on 30 cm aerial imagery without catastrophic forgetting.

### 3.3 RemoteCLIP

**Paper**: "RemoteCLIP: A Vision-Language Foundation Model for Remote Sensing" (Liu et al., 2023)

**What it does**: [[clip|CLIP]] adapted to remote sensing via [[contrastive-learning|contrastive pretraining]] on remote sensing image-text pairs.

**Data**: collected from:
- Remote sensing image captioning datasets (UCMD, RSICD, RSITMD)
- Remote sensing question-answering data
- Synthetically generated captions from detection/segmentation annotations (converting category labels + bounding boxes into descriptive sentences)

**Key innovation**: the dataset creation pipeline — taking detection-annotated RS datasets and generating rich textual descriptions. A COCO-style annotation "three aircraft on a runway" becomes "The image shows an airport with multiple large aircraft parked on the tarmac. Three aircraft are clearly visible in the center."

**Results**: significantly outperforms CLIP on remote sensing zero-shot classification and image-text retrieval. Particularly strong on fine-grained RS tasks like airport type identification.

**Limitation**: trained on 223K image-text pairs vs CLIP's 400M — still limited by data scale.

### 3.4 SatCLIP

**Paper**: "SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery" (Klemmer et al., 2023)

**What it does**: learns location-conditioned representations by contrastive learning between satellite image patches and geographic coordinates.

**Training signal**: image patch at location $(lat, lon)$ must be aligned with the location encoding of $(lat, lon)$. The location encoder uses spherical harmonics to represent position on the globe.

**Use case**: meta-learning for geospatial tasks. Given a new location, retrieve similar training locations using location embeddings. Enables few-shot learning for [[biodiversity-ml|species distribution modelling]]: find locations with similar environments → use observations from those locations.

**Key insight**: geographic location is itself rich information. A point in the Amazon rainforest and one in the Sahara will have completely different ecological contexts even if their images look similar due to cloud cover.

### 3.5 Prithvi (IBM + NASA)

**Paper**: "Foundation Models for Generalist Geospatial Artificial Intelligence" (Jakubik et al., 2023)

**What it does**: a 100M-parameter [[vision-transformer|ViT]] foundation model for geospatial data, pretrained by IBM and NASA.

**Architecture**: ViT with 3D patch embedding — treats time as a third dimension alongside height and width:
- Input: (T=3, C=6, H=224, W=224) — 3 time steps, 6 Sentinel-2 bands, 224×224 spatial
- 3D patches: (t=1, h=16, w=16) patches — each patch spans one time step and one 16×16 spatial region
- Masked autoencoder objective: mask ~75% of patches and reconstruct

**Training data**: 250 GB of Harmonized Landsat Sentinel (HLS) data — coregistered Landsat 8 + Sentinel-2 data at 30 m resolution.

**Downstream applications**: 
- Flood mapping — detects flood extent from multi-temporal SAR/optical
- Wildfire scar mapping
- Crop type mapping

**Fine-tuning**: Prithvi ships as an encoder; attach a segmentation head (UperNet) and fine-tune on task-specific data.

```python
# Prithvi usage pattern
from prithvi import PrithviEncoder
encoder = PrithviEncoder.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
# Input: (B, T, C, H, W)
features = encoder(time_series_tensor)  # → (B, num_patches, embed_dim)
# Attach UperNet decoder and fine-tune
```

### 3.6 GeoChat

**Paper**: "GeoChat: Grounded Large Vision-Language Model for Remote Sensing" (Kuckreja et al., 2023)

**What it does**: a [[vlm-architectures|VLM]] specifically designed for remote sensing images, supporting:
- Image-level QA ("What is the land use in this image?")
- Region-level QA ("Describe the vegetation at [350, 180, 420, 240]")
- Grounded detection ("Where are the buildings?")

**Architecture**: builds on LLaVA framework:
1. SigLIP visual encoder (better than standard CLIP for EO due to sigmoid loss, trained on EO data)
2. MLP projection into Vicuna-7B language model
3. **Grounded instruction tuning**: training data includes box annotations + language, so the model learns to output `[x1, y1, x2, y2]` coordinates in text

**Training data**: GeoChat-Instruct dataset — 318K instruction-following examples generated from 8 existing RS datasets (RSICD, UCM, DIOR, etc.) using GPT-4 as caption generator.

**Grounding capability**: GeoChat can be asked "Where is the forest in this image?" and outputs both a text description and bounding box coordinates.

---

## 4. GeoCLIP: Location-Based Contrastive Learning

**Paper**: "GeoCLIP: Clip-Inspired Alignment Between Locations and Images for Effective Worldwide Geo-Localization" (Vivanco et al., 2023)

**What it does**: trains a CLIP-style model where:
- Image encoder: standard ViT
- Location encoder: MLP on GPS coordinates (encoded with Fourier features)
- Contrastive loss: match image to its geographic location

**Use case**: geolocalization — given an image, predict where in the world it was taken. Also useful as a geographic prior for species distribution modelling.

---

## 5. Using EO Foundation Models in Practice

### 5.1 Zero-Shot Inference

Use RemoteCLIP or Prithvi directly without fine-tuning:
- **RemoteCLIP**: pass satellite chip + text prompt → similarity score for classification/retrieval
- **Prithvi**: extract patch features → train lightweight linear probe for your task

### 5.2 Linear Probing

Freeze the foundation model backbone; train only a linear classifier on top:
```python
with torch.no_grad():
    features = model.encode(images)  # frozen
linear = nn.Linear(embed_dim, num_classes)
# train only `linear`
```
Very data-efficient. Works well with 100–1000 labelled examples.

### 5.3 Fine-Tuning with LoRA

For more complex tasks or limited data, apply LoRA to the foundation model (see [[lora-quantization]]). Adds only 0.1–1% additional parameters while enabling meaningful adaptation.

### 5.4 Temporal Fine-Tuning

When your downstream task requires temporal reasoning (crop monitoring, phenology), Prithvi's 3D temporal architecture is strongly preferred. Standard 2D image models must process timestamps independently and miss temporal dependencies.

---

## 6. Benchmark Comparison

| Model | Architecture | Pretraining | Multi-spectral | Temporal | Text-conditioned |
|-------|------------|-------------|---------------|---------|-----------------|
| CLIP | ViT | Image-text pairs | ❌ (RGB only) | ❌ | ✅ |
| RemoteCLIP | ViT | RS image-text | ❌ | ❌ | ✅ |
| SatCLIP | ViT | Image-location | ❌ | ❌ | ✅ (via location) |
| SatMAE | ViT | MAE on RS | ✅ (13 bands) | ✅ | ❌ |
| Scale-MAE | ViT | Scale-aware MAE | ✅ | ❌ | ❌ |
| Prithvi | ViT-3D | MAE (HLS) | ✅ (6 bands) | ✅ (3 timesteps) | ❌ |
| GeoChat | ViT + LLM | RS instruction | ✅ | ❌ | ✅ (QA + grounding) |

---

## See Also

[[earth-observation-fundamentals]], [[satellite-imagery-preprocessing]], [[clip]], [[blip]], [[vlm-architectures]], [[biodiversity-ml]], [[self-supervised-overview]]
