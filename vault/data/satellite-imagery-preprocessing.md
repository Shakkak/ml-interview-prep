---
title: "Satellite Imagery Preprocessing"
tags: [satellite-preprocessing, radiometric-calibration, cloud-masking, normalization, temporal-compositing, geospatial-ml, vlm-preprocessing]
aliases: [Remote Sensing Preprocessing, EO Data Preprocessing, Satellite Data Preparation, Satellite Preprocessing]
difficulty: 2
status: complete
depends_on: [earth-observation-fundamentals, feature-preprocessing]
related: [multi-source-fusion, clip, eigenvalues-pca, data-augmentation]
---

# Satellite Imagery Preprocessing

---

## Fundamental

Raw satellite data cannot be fed directly to an ML model. The full pipeline transforms raw digital numbers to model-ready tensors:

```
Raw DN values
  ↓ 1. Radiometric calibration (DN → TOA reflectance)
  ↓ 2. Atmospheric correction (TOA → surface reflectance)
  ↓ 3. Cloud and shadow masking
  ↓ 4. Geometric correction / coregistration
  ↓ 5. Temporal compositing
  ↓ 6. Band normalization
  ↓ 7. Chip extraction (tiling)
  → Ready for model
```

Skipping step 2 (atmospheric correction) means a model trained in summer fails in winter — atmospheric haze varies with season. Skipping step 3 means cloud pixels corrupt training labels.

### Radiometric Calibration: DN → Reflectance

Satellites record **digital numbers (DN)**: integers proportional to energy hitting the detector. Different sensors have incompatible DN scales. TOA (Top-of-Atmosphere) reflectance is the physically meaningful quantity:

$$\rho_\text{TOA}(\lambda) = \frac{\pi \cdot L(\lambda) \cdot d^2}{ESUN(\lambda) \cdot \cos(\theta_s)}$$

where $L(\lambda)$ = radiance measured by sensor at wavelength $\lambda$ (W/m²/sr/μm), $d$ = Earth–Sun distance in AU (varies ±3% annually), $ESUN(\lambda)$ = mean exo-atmospheric solar irradiance at $\lambda$, and $\theta_s$ = solar zenith angle. For Sentinel-2 specifically: $\rho_\text{TOA} = DN / 10000$ (the manufacturer pre-scales to reflectance × 10,000).

### Atmospheric Correction: TOA → Surface Reflectance

TOA reflectance still contains atmospheric scattering (Rayleigh) and aerosol/water-vapor absorption — all scene-dependent and season-varying. A model training on mixed TOA data learns atmospheric confounders rather than surface properties.

**Sen2Cor** (Sentinel-2 official, free): estimates aerosol optical depth and water vapor from the image itself → produces Level-2A (surface reflectance). **Practical recommendation:** download Sentinel-2 Level-2A products from Copernicus directly — already Sen2Cor corrected.

Alternative methods: **6S** (physics-based, very accurate when separate aerosol measurements available), **Dark Object Subtraction** (subtract darkest pixel as a proxy for atmospheric path radiance — fast but approximate).

---

## Intermediate

### Cloud and Shadow Masking

Unmasked cloud pixels appear as bright white surfaces — catastrophic for vegetation analysis. Sentinel-2 Level-2A includes a **Scene Classification Layer (SCL)** with per-pixel classes. Standard practice: mask SCL values 1 (saturated), 2–3 (cloud shadows), 8–9 (cloud), 10 (thin cirrus), 11 (snow/ice). Keep only vegetation (4), not-vegetated (5), water (6), unclassified (7).

For higher accuracy: **s2cloudless** (ML-based cloud probability map) or **MAJA** (multi-temporal correction that uses image history to detect clouds by appearance change over time).

**Shadow masking:** use solar geometry to project predicted cloud positions along sun angle → find shadow footprint. Also use spectral detection (shadows are dark across all bands with low NDVI).

**Why imperfect masking hurts:** if 5% of "vegetation" training pixels are actually cloud edges, the model learns cloud spectral signatures. At inference on clear-sky scenes, these examples inject noise. For time series models, one contaminated timestamp corrupts temporal features.

### Geometric Correction and Temporal Compositing

**Orthorectification:** corrects geometric distortions from Earth's curvature, terrain elevation, and off-nadir viewing using a DEM + sensor orbit parameters. Sentinel-2 L2A is already orthorectified.

**Coregistration:** even orthorectified images from different dates can be misaligned by 1–3 pixels. Sub-pixel coregistration (e.g., AROSICS) computes cross-correlation at tie points and applies correction shifts. Critical when computing vegetation index time series — a 1-pixel shift creates false change signals.

**Temporal compositing:** any single image may have cloud cover. A composite combines multiple acquisitions over a time window:

- **Median composite:** take the median value per pixel across all acquisitions in a time window (e.g., 3-month season). Robust to clouds and outliers. Most commonly used.
- **Max-NDVI composite:** select the acquisition with highest NDVI per pixel — picks the greenest/most vegetation-active observation; useful for growing season characterization.
- **Percentile composites:** 10th/90th percentile captures dry/wet extremes.

For deep learning: create seasonal composites (4 per year), stack as multi-temporal tensor shape (T, C, H, W) where T = time steps.

---

## Advanced

### Band Normalization for Deep Learning

Satellite pixel values are not ImageNet pixels (0–255). Options:

**Per-dataset statistics (recommended):** compute mean $\mu_c$ and std $\sigma_c$ per band $c$ across the training set, then $x_\text{norm}^c = (x^c - \mu_c) / \sigma_c$. Published statistics for BigEarthNet and EuroSAT allow normalization consistent with pretrained models.

**Percentile clipping + min-max:** clip to 2nd–98th percentile, then normalize to [0, 1]. More robust than global min-max. Standard for visualization.

**Per-image min-max:** stretch each image to [0, 1] using its own min/max per band. Sensitive to cloud outliers — apply only after cloud masking.

### RGB Composites for VLM Adaptation

[[vlm-architectures|Vision-language models]] like CLIP were trained on RGB images; satellite data has 13 bands.

| Composite | Bands used | Best for |
|---|---|---|
| True color | B4 (Red), B3 (Green), B2 (Blue) | Visually interpretable; CLIP recognizes familiar patterns |
| False color (vegetation) | B8 (NIR), B4 (Red), B3 (Green) | Vegetation appears bright red; emphasizes health |
| Agriculture / SWIR | B11 (SWIR1), B8 (NIR), B4 (Red) | Distinguishes crops, bare soil, and water |

For fine-tuned VLMs: (1) early fusion — 13-channel patch embedding instead of 3-channel; (2) multiple composites — encode true-color + false-color separately → combine embeddings; (3) [[eigenvalues-pca|PCA]] reduction — reduce 13 bands to 3 principal components.

**Practical recommendation for CLIP inference:** true-color RGB, normalized to ImageNet statistics. Fine-tune on domain data when budget allows.

### Chip Extraction and Spatial Train/Test Split

Satellite scenes are large (e.g., 10,980 × 10,980 pixels for a Sentinel-2 tile). Tile into N×N patches (e.g., 224×224 or 512×512). Use overlapping tiling (stride < chip size) for dense prediction to ensure objects near boundaries appear in at least one full chip.

Always store geographic coordinates (origin lat/lon + CRS) with each chip to geo-reference predictions and enable spatial splitting.

**Spatial train/test split:** random chip splitting is wrong — adjacent chips overlap spatially and have identical land cover context. Assign entire geographic blocks to train/val/test. This is the spatial analogue of temporal splitting for time series data.

### Data Augmentation for EO

| Augmentation | Safe? | Notes |
|---|:---:|---|
| Random horizontal / vertical flip | ✓ | No canonical orientation in satellite imagery |
| Random rotation (any angle) | ✓ | Satellite imagery is north-up but CNNs don't need it |
| Color jitter (RGB only) | ⚠ | Can destroy spectral relationships; be conservative |
| CutMix / MixUp | ✓ | Works well for multi-label land cover |
| Band dropout | ✓ | Simulates missing bands; improves robustness |

Do **not** use aggressive brightness/contrast augmentation on physical reflectance values — it breaks spectral index relationships.

---

## Links

- [[earth-observation-fundamentals]] — sensor physics (radiance, reflectance, DN values) covered there are the prerequisites for understanding preprocessing choices here
- [[feature-preprocessing]] — radiometric calibration and atmospheric correction are EO-specific preprocessing steps that precede standard feature scaling
- [[multi-source-fusion]] — correctly preprocessed and georeferenced imagery is the prerequisite for multi-source fusion; alignment errors compound across modalities
- [[clip]] — CLIP expects ImageNet-style normalized inputs; satellite imagery requires separate normalization decisions before CLIP fine-tuning
- [[eigenvalues-pca]] — PCA on hyperspectral or multispectral stacks reduces band redundancy and noise before feeding to ML models
- [[data-augmentation]] — EO-specific augmentations (random rotation, spectral jitter, cloud simulation) are applied after preprocessing
