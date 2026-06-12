---
title: "Satellite Imagery Preprocessing"
tags: [satellite-preprocessing, radiometric-calibration, cloud-masking, normalization, temporal-compositing, geospatial-ml, vlm-preprocessing]
aliases: [Remote Sensing Preprocessing, EO Data Preprocessing, Satellite Data Preparation]
status: complete
---

This file covers every preprocessing step needed to transform raw satellite data into model-ready tensors — from raw digital numbers to normalized patches ready for a [[vlm-architectures|vision-language model]]. Written for someone starting from zero with satellite data.

## 1. The Preprocessing Pipeline at a Glance

Raw satellite data undergoes several steps before an ML model can use it:

```
Raw DN values
    ↓ 1. Radiometric calibration (DN → TOA reflectance)
    ↓ 2. Atmospheric correction (TOA → BOA / surface reflectance)
    ↓ 3. Cloud and shadow masking
    ↓ 4. Geometric correction / coregistration
    ↓ 5. Band selection and compositing
    ↓ 6. Normalisation and standardisation
    ↓ 7. Chip extraction (tiling into model-sized patches)
    ↓ Ready for model
```

Each step matters. Skipping step 2 (atmospheric correction) means a model trained in summer in Germany will fail on the same area in winter because atmospheric haze varies. Skipping step 3 means cloud pixels corrupt your training labels.

---

## 2. Radiometric Calibration — DN to Reflectance

### 2.1 What are DN Values?

Satellites record *digital numbers* (DN): integer values proportional to the energy hitting the detector. They are **not** physical reflectance values. Different sensors have different gain/offset settings, so DN from Sentinel-2 and Landsat are on completely different scales.

**Top-of-Atmosphere (TOA) reflectance** accounts for sun angle and sensor gain:
$$\rho_{TOA}(\lambda) = \frac{\pi \cdot L(\lambda) \cdot d^2}{ESUN(\lambda) \cdot \cos(\theta_s)}$$

Where:
- $L(\lambda)$ = radiance measured by sensor (W/m²/sr/μm)
- $d$ = Earth–Sun distance in AU (varies ±3% through the year)
- $ESUN(\lambda)$ = mean exo-atmospheric solar irradiance at wavelength $\lambda$
- $\theta_s$ = solar zenith angle

For Sentinel-2, the manufacturer provides a simpler conversion:
$$\rho_{TOA} = \frac{DN}{10000}$$
(the quantification value is 10,000, and the raw data is already scaled to reflectance × 10,000)

### 2.2 Why TOA is Not Enough

Even after TOA conversion, the signal still includes:
- Atmospheric scattering (Rayleigh scattering: short wavelengths scatter more → makes blue sky blue)
- Aerosol absorption and scattering
- Water vapour absorption

These effects are **scene-dependent** and vary by season, location, and atmospheric conditions. A model training on mixed TOA data will learn confounding atmospheric signals instead of surface properties.

---

## 3. Atmospheric Correction — TOA to Surface Reflectance

**Surface reflectance** (also called Bottom-of-Atmosphere, BOA reflectance) is what the land surface actually reflects, stripped of atmospheric effects.

### 3.1 Methods

**Sen2Cor (Sentinel-2 official)**
The standard free tool from ESA. Takes Level-1C (TOA) Sentinel-2 data → Level-2A (BOA).
Estimates aerosol optical depth (AOD) and water vapour from the image itself.
Works well in most conditions; can fail in extreme smoke/dust events.

**6S (Second Simulation of the Satellite Signal in the Solar Spectrum)**
Physics-based radiative transfer model. Very accurate when you have separate aerosol measurements (e.g., from AERONET ground stations).
More complex to run — requires atmospheric parameters as inputs.

**Dark Object Subtraction (DOS)**
Simple heuristic: find the darkest pixel in the image (assumed to be zero surface reflectance — e.g., deep shadow or very dark water). Subtract its value from all pixels.
Fast but approximate. Assumes no adjacency effects and ignores aerosol wavelength dependence.

**iCOR**
Atmospheric correction for Sentinel-2 that also handles inland water bodies well. Better than Sen2Cor over water.

### 3.2 Practical Recommendation

For most ecology applications: download Sentinel-2 **Level-2A** products from Copernicus — these are already atmospherically corrected by Sen2Cor. No need to run it yourself.

---

## 4. Cloud and Shadow Masking

Clouds are the biggest nuisance in optical satellite imagery. Unmasked cloud pixels look like bright white surfaces — terrible for vegetation analysis.

### 4.1 The SCL Layer (Sentinel-2)

Sentinel-2 Level-2A products include a **Scene Classification Layer (SCL)** with per-pixel classes:

| SCL value | Class |
|-----------|-------|
| 1 | Saturated / Defective |
| 2 | Dark areas / Cloud shadows |
| 3 | Cloud shadows |
| 4 | Vegetation |
| 5 | Not vegetated |
| 6 | Water |
| 7 | Unclassified |
| 8 | Cloud medium probability |
| 9 | Cloud high probability |
| 10 | Thin cirrus |
| 11 | Snow or Ice |

Standard practice: mask out SCL values 1, 2, 3, 8, 9, 10, 11. Keep only 4, 5, 6, 7 for analysis.

### 4.2 Advanced Cloud Masking

**s2cloudless**: a machine learning-based cloud detector for Sentinel-2. Produces a cloud probability map (0–100%) per pixel. More accurate than SCL for thin/broken cloud.

**MAJA (CNES/DLR)**: multi-temporal atmospheric correction that uses image history to detect clouds by their appearance change over time. Very accurate but requires a time series.

**Fmask**: used for Landsat; function of mask using spectral cloud detection rules.

### 4.3 Cloud Shadow Masking

Shadows appear dark and can be mistaken for water or bare soil. Shadow detection uses:
1. Solar geometry: project predicted cloud positions along sun angle to find shadow footprint
2. Spectral detection: shadows are dark across all bands and have low NDVI

### 4.4 Why Imperfect Masking Hurts ML

If 5% of your "vegetation" training pixels are actually clouds or cloud edges, your model learns cloud spectral signatures. At inference on a clear-sky scene, these misclassified training examples inject noise that degrades accuracy. For time series models, a single contaminated timestamp can corrupt temporal features.

---

## 5. Geometric Correction and Coregistration

### 5.1 Orthorectification

Raw satellite images have geometric distortions from:
- Earth's curvature
- Terrain elevation: a mountain top is displaced from its true location
- Sensor view angle: off-nadir viewing shifts objects

Orthorectification uses a Digital Elevation Model (DEM) + sensor orbit parameters to correct these. Sentinel-2 Level-2A is already orthorectified.

### 5.2 Coregistration

When combining images from different dates or sensors, pixel-level alignment is needed.
Even "orthorectified" images can be misaligned by 1–3 pixels due to different processing pipelines.

**Sub-pixel coregistration** (e.g., AROSICS tool) computes cross-correlation between images at tie points and applies a correction shift. Critical when:
- Fusing drone imagery with satellite data
- Computing vegetation index time series (a 1-pixel shift creates false change signal)

---

## 6. Temporal Compositing

### 6.1 Why Compositing?

Any single satellite image may have cloud cover over your area. A *temporal composite* combines multiple images over a time window to produce a clean, cloud-free mosaic.

### 6.2 Compositing Methods

**Median composite**: take the median value of each pixel across all acquisitions in a time window (e.g., a 3-month season). Robust to clouds and outliers. The most commonly used method.

```python
# Google Earth Engine pseudocode
s2_collection = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2023-06-01', '2023-09-01')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
composite = s2_collection.median()
```

**Max-NDVI composite**: for each pixel, select the acquisition with the highest NDVI value. Picks the greenest/most vegetation-active observation — useful for growing season characterisation.

**Seasonal composites**: divide year into quarters or phenological seasons; produce one image per period. Creates a multi-temporal tensor: shape (T, C, H, W) where T = number of seasons.

**Percentile composites**: use 10th/90th percentile instead of median to capture dry/wet extremes.

### 6.3 Temporal Compositing for Deep Learning

For training:
- 3-month quarterly composites → 4 composite images per year per location
- Each composite: shape (13, H, W) for Sentinel-2 with all bands
- Stack into time series: (4, 13, H, W) — then treat time as additional channels or use temporal encoder

For inference on new data: ensure the same compositing window is used as at training time.

---

## 7. Band Normalization for Deep Learning

Satellite pixel values are not like ImageNet pixels (0–255). They must be normalized before feeding to a neural network.

### 7.1 Per-Dataset Statistics Normalization (Recommended)

Compute mean $\mu_c$ and standard deviation $\sigma_c$ per band $c$ across the entire training set:
$$x_{norm}^{c} = \frac{x^{c} - \mu_c}{\sigma_c}$$

This is the same as ImageNet normalization but with satellite-specific per-band statistics. Published statistics exist for BigEarthNet and EuroSAT — use them so your model generalizes to other datasets.

**BigEarthNet Sentinel-2 statistics (13 bands):**
```python
MEANS = [429.9, 614.2, 669.6, 728.9, 1049.8, 1755.5, 1993.1, 
         2066.9, 2168.9, 634.5, 14.4, 1783.9, 1117.8]  # ordered B1..B12
STDS  = [572.4, 590.5, 642.5, 668.6, 802.8,  964.5, 1089.4, 
         1082.0, 1067.2, 565.0, 67.7, 1043.0, 853.5]
```

### 7.2 Per-Image Min-Max Normalization

Stretch each image to [0, 1] using its own min/max per band:
$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Simple but sensitive to outliers (bright cloud pixels will compress everything else to near 0). Use only after cloud masking.

### 7.3 Percentile Clipping

Clip to the 2nd–98th percentile before min-max normalizing:
$$x_{clipped} = \text{clip}(x, P_2, P_{98})$$

More robust than global min-max. Standard practice for visualization and some models.

### 7.4 TOA vs Surface Reflectance Normalization

Surface reflectance (BOA) is already physically bounded to [0, 1] in theory (though values slightly outside this range occur). Dividing by 10,000 (for Sentinel-2 L2A) gives fractional reflectance. Then apply per-band standardization.

---

## 8. RGB Composites — Adapting Satellite Imagery for VLMs

Vision-language models like CLIP were trained on RGB images. Multispectral satellite data has 13 bands. You must select or combine bands to produce an RGB image.

### 8.1 True Color Composite

Use B4 (Red), B3 (Green), B2 (Blue). Looks like a natural photograph.
- Pros: visually interpretable, VLMs recognize familiar landscape patterns
- Cons: discards spectral richness (NIR, Red Edge, SWIR)

### 8.2 False Color Composite (Vegetation Emphasis)

Use B8 (NIR), B4 (Red), B3 (Green) → displayed as R, G, B.
Vegetation appears bright red/magenta. Non-vegetated areas appear in shades of blue/grey.
Very common in ecology for highlighting vegetation.

### 8.3 Agriculture / SWIR Composite

Use B11 (SWIR1), B8 (NIR), B4 (Red).
Highlights differences between healthy crops, bare soil, and water.

### 8.4 Custom Band-to-RGB Strategies for VLMs

For fine-tuned VLMs:
1. **Early fusion**: concatenate all 13 bands as a 13-channel tensor; use a patch embedding that accepts 13 channels instead of 3. Requires modifying the vision encoder.
2. **Late fusion**: encode RGB separately + encode NDVI separately + combine embeddings downstream.
3. **Multiple RGB composites**: create true-color + false-color → pass both through CLIP separately → combine embeddings.
4. **[[eigenvalues-pca|PCA]] reduction**: reduce 13 bands to 3 principal components. Maximizes variance but components have no intuitive spectral meaning.

**Practical recommendation for [[clip|CLIP]] inference**: true-color RGB, normalized to ImageNet statistics. Fine-tune on domain data if budget allows.

---

## 9. Chip Extraction (Tiling)

Satellite scenes are large (e.g., 10,980 × 10,980 pixels for a Sentinel-2 tile). ML models operate on small, fixed-size patches.

### 9.1 Non-Overlapping Tiling

Divide scene into N×N pixel chips (e.g., 224×224 or 512×512):
```python
for row in range(0, height, chip_size):
    for col in range(0, width, chip_size):
        chip = image[:, row:row+chip_size, col:col+chip_size]
```

- Fast and simple
- Chips at tile edges may be smaller → pad to full size

### 9.2 Overlapping Tiling (Sliding Window)

Use stride < chip_size for overlap. Useful for:
- Ensuring objects near chip boundaries appear in at least one full chip
- Dense prediction tasks (segmentation, detection)

### 9.3 Tracking Chip Coordinates

Always store the geographic coordinates of each chip (origin lat/lon + CRS) alongside the tensor. This lets you:
- Geo-reference predictions
- Sample geographically diverse training sets
- Avoid train/test spatial leakage (adjacent chips may be identical — split by spatial block, not random)

### 9.4 Spatial Train/Test Split

Random chip splitting is wrong for satellite data. Adjacent chips overlap spatially and have identical land cover context. Use **spatial block splitting**: divide the region into large blocks and assign entire blocks to train/val/test.

---

## 10. Data Augmentation for Satellite Imagery

Standard augmentations from natural image ML remain valid, with some satellite-specific considerations:

| Augmentation | Safe for EO? | Notes |
|-------------|-------------|-------|
| Random horizontal / vertical flip | ✅ | No canonical orientation in satellite imagery |
| Random rotation (any angle) | ✅ | Satellite imagery is north-up but CNNs don't need it |
| Colour jitter (on RGB only) | ⚠️ | Can destroy spectral relationships; be conservative |
| [[data-augmentation\|CutMix / MixUp]] | ✅ | Works well for multi-label land cover |
| Random crop | ✅ | Useful if chips are larger than model input |
| Temporal jitter | ✅ | Use slightly different time window for compositing |
| Band dropout | ✅ | Simulate missing bands; improves robustness |

**Do not** use aggressive brightness/contrast augmentation on physical reflectance values — it breaks the relationship between spectral indices and what they measure.

---

## 11. Special Case: Hyperspectral Data

Hyperspectral sensors (e.g., AVIRIS, EnMAP, DESIS) capture hundreds of narrow bands (e.g., 242 bands at ~10 nm spacing).

Preprocessing additions:
- **Spectral smoothing**: remove noise bands at atmospheric absorption windows (e.g., around 1400 nm and 1900 nm)
- **Dimensionality reduction**: PCA to 10–50 components, or use bands known to carry information
- **Continuum removal**: normalize to the convex hull to isolate absorption features

Hyperspectral ML models often use 1D CNN across the spectral dimension first, then 2D spatial processing.

---

## See Also

[[earth-observation-fundamentals]], [[multi-source-fusion]], [[remote-sensing-foundation-models]], [[vlm-architectures]], [[clip]], [[eigenvalues-pca]], [[data-augmentation]]
