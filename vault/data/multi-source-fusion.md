---
title: "Multi-Source Data Fusion for Earth Observation"
tags: [multi-source-fusion, data-fusion, drone-satellite, early-fusion, late-fusion, mid-fusion, multimodal-eo, sensor-fusion]
aliases: [Multi-Modal Fusion EO, Sensor Fusion Remote Sensing, Drone Satellite Fusion, Multi-Source EO]
difficulty: 3
status: complete
depends_on: [earth-observation-fundamentals, attention-mechanism]
related: [satellite-imagery-preprocessing, vlm-architectures, clip, knowledge-distillation]
---

# Multi-Source Data Fusion for Earth Observation

---

## Fundamental

**The problem:** each EO data source has unique strengths and blind spots. Satellite imagery (Sentinel-2) covers the whole landscape every 5 days but at 10 m resolution. Drone imagery reveals individual plants sub-centimeter but covers only a few hectares. In-situ ground sensors give direct physical measurements at single points.

**Intuition:** combine what each source knows well and compensate for what it cannot see. The satellite gives regional context; the drone gives detail; in-situ gives truth. A model trained on all three can predict fine-grained traits at landscape scale — impossible from any single source alone.

| Source | Strengths | Limitations |
|---|---|---|
| Satellite (Sentinel-2, Landsat) | Global coverage, free, multispectral, daily–weekly revisit | 10–30 m resolution, cloud-limited, temporal lag |
| Drone (UAV) | Sub-cm resolution, flexible timing, RGB+NIR+thermal | Small area coverage, expensive per ha, weather dependent |
| In-situ sensors | Direct physical measurement (soil moisture, species count, plant traits) | Point data, sparse, labor intensive |
| Hyperspectral (airborne) | 100+ narrow bands, sub-meter resolution | Very expensive, limited coverage |

### Core Fusion Challenges

**Spatial scale mismatch:** a Sentinel-2 pixel is 100 m² (10×10 m). A single satellite pixel covers thousands to millions of drone pixels. Aggregating drone measurements to satellite scale requires handling heterogeneous land cover and mixed pixels.

**Temporal mismatch:** Sentinel-2 revisits every 5 days; drone surveys happen once per season; in-situ measurements may be monthly. Vegetation phenology changes significantly over 2–4 weeks — a satellite composite from June may not match a drone survey from late August.

**Geometric misalignment:** satellite imagery has ~10 m absolute geolocation accuracy; drone imagery needs ground control points (GCPs) for sub-meter accuracy; standard GPS has ±3 m accuracy. Without careful coregistration, fusion adds noise rather than information.

**Modality gap:** satellite = 13-channel reflectance tensors; drone = 3-channel normalized intensities; in-situ = structured tabular data. Naive concatenation causes one modality to dominate due to scale differences.

---

## Intermediate

### Three Fusion Architectures

**Early Fusion (input-level):** concatenate all modalities before any model sees them.

```
Satellite (13 bands, 10 m) + Drone (3 bands, 0.1 m→resampled to 10 m) + In-situ (tabular→interpolated)
  → stacked tensor → single encoder → prediction
```

Best when: all modalities always available, similar spatial resolution after resampling. Loss: spatial resampling of drone to satellite scale discards fine-grained information; harder training when modality quality varies.

**Late Fusion (decision-level):** train separate models per modality, combine predictions.

```
Satellite → Model A → score_A
Drone     → Model B → score_B
In-situ   → Model C → score_C
  → f(score_A, score_B, score_C) = weighted average, stacking, or learned combiner
```

Best when: prototyping, modalities arrive at different times, or missing modality tolerance is needed. Loss: no cross-modality feature-level complementarity.

**Mid-Level Fusion (feature-level):** each modality has its own encoder; features are fused at an intermediate layer.

```
Satellite → Encoder_S → f_S ∈ ℝ^{D_S}
Drone     → Encoder_D → f_D ∈ ℝ^{D_D}   →  Fusion module → task head → prediction
In-situ   → Encoder_I → f_I ∈ ℝ^{D_I}
```

**Mid-level fusion modules:**

*Concatenation + MLP:* $f_\text{fused} = \text{MLP}([f_S; f_D; f_I])$ — simplest; learns linear interactions after concat.

*[[attention-mechanism|Cross-attention]] (Transformer fusion):* satellite features as queries; drone + in-situ as keys/values — the satellite representation learns which parts of the drone to attend to. Best for fine-grained trait prediction.

*FiLM conditioning (Feature-wise Linear Modulation):* $f_S^\text{conditioned} = \gamma(f_I) \odot f_S + \beta(f_I)$ — in-situ measurements condition the satellite representation by learned channel-wise scale $\gamma$ and shift $\beta$.

**Hierarchical Fusion:** exploit the natural scale hierarchy for drone-satellite pairing:
1. Encode drone imagery at native resolution → fine-grained features (N_drone patches per satellite pixel)
2. Aggregate drone features spatially using attention pooling (satellite queries over drone patches)
3. Concatenate aggregated drone features with satellite backbone features
4. Predict at satellite pixel scale

Preserves drone fine-grained information without naive resampling.

---

## Advanced

### Drone + Satellite Preprocessing Requirements

Before any fusion, data must be coregistered (align drone orthomosaic to satellite CRS using GCPs to sub-pixel accuracy), resampled to a common grid if fusing at satellite scale (use area-weighted average, not nearest neighbor), radiometrically calibrated (satellite: atmospheric correction to surface reflectance; drone: calibration panel → reflectance), and temporally matched (use satellite composite from the same date/week as drone survey).

### Training Strategy: Semi-Supervised Distillation

The typical scenario: drone data covers 50 field sites × 1 ha each = 50 ha; satellite data covers the full landscape (100,000 ha). 

**Approach:** train a fusion model (drone + satellite) at field sites → use it as a teacher to generate soft labels at landscape scale → train a satellite-only student model on the full landscape using those soft labels. This is [[knowledge-distillation|knowledge distillation]] across modalities: the multi-sensor teacher trains the single-sensor student.

### Self-Supervised Pretraining for Multi-Source EO

**SeCo (Seasonal Contrast):** positive pairs = same location at different seasons; negative pairs = different locations. Forces the model to learn location-invariant representations that separate land cover types, not seasonal appearance.

**Cross-Modal Contrastive Learning:** treat optical (Sentinel-2) and SAR (Sentinel-1) images of the same location as positive pairs. Forces representations invariant to the sensing modality — "forest" from SAR and "forest" from optical map to nearby embeddings.

### When to Use Each Strategy

| Strategy | Best scenario |
|---|---|
| Early fusion | All modalities always available; same spatial scale |
| Late fusion | Prototyping; modalities sometimes missing |
| Mid-level cross-attention | Model needs to select which parts of one modality are relevant to another |
| Hierarchical | Very high-resolution drone + lower-resolution satellite; preserve both scales |

---

## Links

- [[earth-observation-fundamentals]] — understanding each sensor's spatial, spectral, and temporal resolution is the prerequisite for fusion; physical differences drive design choices
- [[attention-mechanism]] — cross-attention is the standard deep fusion mechanism: queries from one modality, keys/values from another; it learns which parts to attend to
- [[satellite-imagery-preprocessing]] — each sensor requires different preprocessing before fusion; misaligned preprocessing choices are a common failure mode
- [[vlm-architectures]] — multimodal VLMs use identical fusion architectures (projection or cross-attention) as multi-source EO models
- [[clip]] — CLIP-based alignment enables text-guided fusion; text descriptions can guide which sensor features to emphasize
- [[knowledge-distillation]] — multi-source distillation trains a single-sensor student to mimic a multi-sensor teacher; enables deployment without all sensors at inference
