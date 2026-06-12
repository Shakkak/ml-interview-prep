---
title: "Multi-Source Data Fusion for Earth Observation"
tags: [multi-source-fusion, data-fusion, drone-satellite, early-fusion, late-fusion, mid-fusion, multimodal-eo, sensor-fusion]
aliases: [Multi-Modal Fusion EO, Sensor Fusion Remote Sensing, Drone Satellite Fusion]
status: complete
---

Multi-source fusion combines data from different sensors, platforms, or time points to produce richer information than any single source. In Earth observation and ecology, the three main sources are: satellite imagery, drone (UAV) imagery, and in-situ ground measurements. This file covers the technical challenges and strategies for fusing them.

## 1. Why Multi-Source Fusion?

Each data source has unique strengths and limitations:

| Source | Strengths | Limitations |
|--------|-----------|------------|
| **Satellite** (Sentinel-2, Landsat) | Global coverage, free, daily–weekly revisit, multispectral | 10–30 m resolution, clouded, temporal lag |
| **Drone (UAV)** | Sub-cm to dm resolution, flexible timing, RGB+NIR+thermal | Small area coverage, expensive per ha, weather dependent |
| **In-situ sensors** | Direct measurement (soil moisture, temperature, species count) | Point data, sparse, labor intensive, biased by access |
| **Hyperspectral (airborne)** | 100+ narrow bands, sub-meter resolution | Very expensive, limited coverage |

**Why combine?**
- Satellite gives context (land cover, seasonal trend) for the whole landscape
- Drone gives fine-grained detail (individual plants, species counts) at sample sites
- In-situ provides ground truth (actual species presence, measured plant traits)

The combination enables predictions at landscape scale (from satellite) calibrated with fine-grained detail (from drone) and validated by ground truth (from in-situ).

---

## 2. The Core Challenges

### 2.1 Spatial Scale Mismatch

- A Sentinel-2 pixel is 100 m² (10×10 m)
- A drone pixel at 10 m flight height is 0.25 cm² (0.005×0.005 m at 1 m flight height, ~1 cm at typical ecology surveys)
- A single satellite pixel covers thousands to millions of drone pixels

Aggregating drone measurements to satellite pixel scale requires careful attention to:
- Spatial variability within a pixel (heterogeneous land cover)
- Mixed pixel problem: a satellite pixel over an ecotone contains both forest and grassland
- Representativeness of the drone sample area for the satellite pixel

### 2.2 Temporal Mismatch

- Sentinel-2 captures a snapshot every 5 days (weather permitting)
- Drone surveys are typically conducted once per season (spring, summer, autumn)
- In-situ measurements may be weekly, monthly, or annual

Temporal interpolation is needed to align all sources to a common time reference. Vegetation phenology changes significantly over 2–4 weeks — a satellite composite from June may not match a drone survey from late August.

### 2.3 Geometric Misalignment

Even with GPS-tagged data from all sources:
- Satellite imagery has ~10 m absolute geolocation accuracy
- Drone imagery requires ground control points (GCPs) for sub-meter accuracy
- In-situ GPS measurements have ±3 m accuracy with standard receivers, ±cm with RTK-GPS

Without careful coregistration, a "fusion" model actually trains on misaligned data — adding noise rather than information.

### 2.4 Modality Gap

Each modality lives in a different feature space:
- Satellite: 13-channel reflectance tensors (physical units)
- Drone RGB: 3-channel normalized intensities
- In-situ: structured tabular data (temperature, species counts, soil pH)

A naive concatenation of these representations may cause one modality to dominate due to scale differences.

---

## 3. Fusion Strategies

### 3.1 Early Fusion (Input-Level)

Concatenate all modalities at the input level and feed to a single model.

```
Satellite features (13 bands, 10 m resolution)
Drone features (3 bands, 0.1 m resolution) → resample to 10 m → stack
In-situ tabular features → interpolate to grid → stack

Model input: concatenated tensor → single encoder → prediction
```

**Advantages**:
- Model learns cross-modality interactions from the start
- Single forward pass, efficient at inference

**Disadvantages**:
- Requires all modalities at inference time (no missing modality handling)
- Spatial resampling of drone → satellite scale loses fine-grained information
- Hard to train: gradients for rare modalities (in-situ) may be overwhelmed by common modalities (satellite)

**Best for**: all modalities always available, same spatial resolution after resampling.

### 3.2 Late Fusion (Decision-Level)

Train separate models on each modality independently; combine predictions at the output level.

```
Satellite data → Model A → score_A
Drone data     → Model B → score_B
In-situ data   → Model C → score_C

Ensemble: prediction = f(score_A, score_B, score_C)
where f is weighted average, stacking, or a learned combiner
```

**Advantages**:
- Easy to implement: train models independently
- Missing modalities: just skip that branch
- Can use different model architectures for each modality

**Disadvantages**:
- Loses cross-modality complementarity at feature level
- Ensemble weights may be wrong if modality quality varies by location

**Best for**: prototyping, when modalities arrive at different times, or when modalities have very different architectures.

### 3.3 Mid-Level Fusion (Feature-Level)

Each modality has its own encoder; features are fused at an intermediate representation layer.

```
Satellite data → Encoder_S → features f_S ∈ ℝ^{D_S}
Drone data     → Encoder_D → features f_D ∈ ℝ^{D_D}
In-situ data   → Encoder_I → features f_I ∈ ℝ^{D_I}

Fusion module → fused features → task head → prediction
```

**Fusion modules:**

**Concatenation + MLP**: simplest, just concatenate and pass through MLP.
$$f_{fused} = \text{MLP}([f_S; f_D; f_I])$$

**[[attention-mechanism|Cross-attention]] (Transformer fusion)**:
Satellite features as queries; drone + in-situ as keys and values. The satellite tokens attend to fine-grained drone information.
```
Q = W_Q f_S   (satellite as query — what context do I need?)
K = W_K f_D   (drone as key — what fine-grained info is available?)
V = W_V f_D
f_fused = softmax(QK^T/√d) V
```

**FiLM conditioning (Feature-wise Linear Modulation)**:
Use one modality to condition another by scaling and shifting:
$$f_S^{conditioned} = \gamma(f_I) \odot f_S + \beta(f_I)$$
where $\gamma, \beta$ are learned from in-situ features. In-situ measurements condition the satellite representation.

### 3.4 Hierarchical Fusion

For the drone-satellite case, exploit the natural scale hierarchy:
1. Process drone imagery at native resolution → fine-grained features
2. Aggregate drone features spatially to satellite pixel scale (mean/max pooling)
3. Concatenate aggregated drone features with satellite features
4. Predict at satellite pixel scale

This preserves fine-grained drone information without losing spatial resolution to a naive resampling.

```python
# Hierarchical fusion pseudocode
drone_features = drone_encoder(drone_patches)  # (N_drone, D)
# N_drone = number of drone patches within this satellite pixel

# Aggregate: attention pooling over drone patches
attn = softmax(Q_sat @ drone_features.T)  # satellite queries drone patches
aggregated_drone = attn @ drone_features   # (D,)

# Fuse with satellite features
satellite_feature = satellite_encoder(sat_chip)   # (D_sat,)
fused = MLP(cat(satellite_feature, aggregated_drone))
```

---

## 4. Drone + Satellite Fusion in Practice

### 4.1 Preprocessing Steps Required

Before any fusion, all data must be:

1. **Coregistered**: align drone orthomosaic to satellite CRS.
   - Compute GCPs (Ground Control Points) manually or with automatic feature matching
   - Apply affine or polynomial warp to drone imagery
   - Target: sub-pixel alignment (< 0.5 satellite pixels)

2. **Resampled to common grid (if needed)**: if fusing at satellite pixel scale.
   - Drone 5 cm → Sentinel-2 10 m: 200× aggregation
   - Use area-weighted average (not nearest neighbor) to preserve reflectance statistics

3. **Radiometrically calibrated**:
   - Satellite: [[satellite-imagery-preprocessing|atmospheric correction]] (DN → surface reflectance)
   - Drone: calibration panel measurements → reflectance conversion

4. **Temporally matched**: use satellite composite from same date/week as drone survey.

### 4.2 Training Data Strategy

You likely have drone data for a small subset of the area (e.g., 50 field sites × 1 ha each = 50 ha total) and satellite data for the entire landscape (e.g., 100,000 ha).

**Semi-supervised approach**:
1. Use drone + satellite together at field sites → train fusion model → get fine-grained predictions at those sites
2. Use satellite alone at landscape scale → predict landscape-scale trait/habitat maps
3. Use drone-calibrated site predictions as "soft labels" for satellite-only training

This is analogous to **[[knowledge-distillation|teacher-student learning]]**: the drone+satellite fusion model (teacher) trains the satellite-only model (student) at landscape scale.

---

## 5. In-Situ + Satellite Fusion

### 5.1 Spatial Matching

An in-situ observation (e.g., a plant trait measured at GPS coordinate [lat, lon]) must be matched to a satellite pixel:

```python
# Find satellite pixel containing the GPS point
row, col = ~transform * (lon, lat)  # rasterio inverse affine transform
satellite_features = raster[:, int(row), int(col)]  # (bands,)
```

But a single satellite pixel (10×10 m) contains many plants — the in-situ measurement at one location within the pixel is not representative of the whole pixel.

**Strategy**: average multiple in-situ measurements within the same satellite pixel. Or use the closest pixel center + a distance weight.

### 5.2 Temporal Interpolation

The satellite composite may be from a different date than the in-situ measurement. If vegetation changes rapidly (e.g., phenological peak), this introduces noise.

**Options**:
- Use the satellite image closest in time to the in-situ measurement
- Fit a phenological curve (double logistic) to the satellite time series; evaluate the curve at the in-situ measurement date
- Include time-of-year as an additional feature in the model

---

## 6. Self-Supervised Pretraining for Multi-Source EO

### 6.1 SeCo (Seasonal Contrast)

**Paper**: "Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data" (Mañas et al., 2021)

Uses temporal self-supervision: images of the same location at different seasons should be similar (they show the same underlying land cover), while images from different locations should be different.

**Positive pairs**: same location, different season.
**Negative pairs**: different locations.

This teaches the model to produce location-invariant representations that separate land cover types, not seasonal appearance.

### 6.2 TiCo (Time-Invariant Contrastive Learning)

Similar to SeCo but explicitly minimizes the variance of representations across time for the same location while maximizing variance across locations.

### 6.3 Cross-Modal Contrastive Learning

Treat optical (Sentinel-2) and SAR (Sentinel-1) images of the same location as positive pairs:
- Optical and SAR see the same land cover from different physical perspectives
- Training forces the model to learn representations invariant to the sensing modality
- Result: a unified embedding space where "forest" from SAR and "forest" from optical are similar

---

## 7. Practical Guidelines

**When to use early fusion**: you have plenty of training data and all modalities are always available. Good for: crop type mapping with multitemporal Sentinel-1 + Sentinel-2.

**When to use late fusion**: you're prototyping or modalities are sometimes missing. Good for: species detection where drone imagery is only available for some sites.

**When to use mid-level (cross-attention) fusion**: you want the model to learn which parts of one modality are relevant to another. Best for: fine-grained trait prediction where drone close-ups should inform which satellite spectral region to focus on.

**When to use hierarchical fusion**: you have very high-resolution drone data and lower-resolution satellite data and want to keep both scales. Best for: individual plant counting from drone + landscape-scale habitat mapping from satellite.

---

## See Also

[[earth-observation-fundamentals]], [[satellite-imagery-preprocessing]], [[vlm-architectures]], [[biodiversity-ml]], [[remote-sensing-foundation-models]], [[attention-mechanism]], [[knowledge-distillation]], [[clip]]
