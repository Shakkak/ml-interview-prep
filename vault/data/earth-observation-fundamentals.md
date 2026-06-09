---
title: "Earth Observation Fundamentals"
tags: [earth-observation, remote-sensing, satellite-imagery, multispectral, sar, geotiff, ndvi, sentinel-2, landsat]
aliases: [Remote Sensing Basics, Satellite Imagery Fundamentals, EO Fundamentals]
---

Earth observation (EO) uses satellite and airborne sensors to acquire information about Earth's surface. This file covers everything from raw sensor physics to datasets — written so a complete beginner can follow, while still being precise enough for an ML practitioner.

## 1. Why Earth Observation Matters for ML

Traditional ML datasets (ImageNet, COCO) contain everyday photos taken by consumer cameras in RGB. Earth observation imagery is fundamentally different:

- **Sensor diversity**: satellites carry optical, radar, thermal, and hyperspectral sensors
- **Spectral richness**: data extends far beyond red-green-blue (R-G-B) visible wavelengths
- **Spatial scale**: a single pixel might cover 10 m × 10 m to 1 km × 1 km of ground
- **Temporal depth**: the same location is imaged repeatedly (every 5 days for Sentinel-2)
- **Physical meaning**: pixel values encode physical quantities (reflectance, backscatter) not perceptual ones

For biodiversity and ecology applications, satellite data lets us monitor:
- vegetation health and structure (NDVI, LAI, canopy height)
- land cover and land use change
- phenological cycles (greening, peak biomass, senescence)
- habitat mapping at continental scale

---

## 2. Key Satellite Missions and Their Sensors

### 2.1 Sentinel-2 (ESA, launched 2015/2017)

The workhorse of open-access EO for vegetation monitoring.

| Property | Value |
|----------|-------|
| Operator | ESA (European Space Agency) |
| Revisit time | 5 days (with both satellites) |
| Spatial resolution | 10 m (RGB + NIR), 20 m (Red Edge, SWIR), 60 m (atmospheric bands) |
| Swath width | 290 km |
| Free access | Yes (Copernicus Open Access Hub) |

**13 spectral bands:**

| Band | Wavelength (nm) | Resolution | Common Use |
|------|----------------|------------|------------|
| B1 | 443 (Coastal aerosol) | 60 m | Aerosol correction |
| B2 | 490 (Blue) | 10 m | True color, water |
| B3 | 560 (Green) | 10 m | True color, vegetation |
| B4 | 665 (Red) | 10 m | True color, chlorophyll |
| B5 | 705 (Red Edge 1) | 20 m | Canopy chlorophyll |
| B6 | 740 (Red Edge 2) | 20 m | Canopy structure |
| B7 | 783 (Red Edge 3) | 20 m | Leaf area index |
| B8 | 842 (NIR broad) | 10 m | Vegetation vigour |
| B8A | 865 (NIR narrow) | 20 m | Water vapour correction |
| B9 | 945 (Water vapour) | 60 m | Atmospheric correction |
| B10 | 1375 (Cirrus) | 60 m | Cloud detection |
| B11 | 1610 (SWIR 1) | 20 m | Soil moisture, snow |
| B12 | 2190 (SWIR 2) | 20 m | Geology, dry vegetation |

### 2.2 Landsat (USGS/NASA, since 1972)

The longest continuous EO record — 50+ years of comparable observations.

| Property | Value |
|----------|-------|
| Current satellite | Landsat 8 (2013) and Landsat 9 (2021) |
| Revisit time | 16 days |
| Spatial resolution | 30 m (multispectral), 15 m (panchromatic), 100 m (thermal) |
| Bands | 11 (similar to Sentinel-2 but coarser) |

Key advantage over Sentinel-2: decades of historical data enable long-term trend analysis.

### 2.3 MODIS (NASA, 2000–present)

Low resolution (250 m – 1 km), but near-daily global coverage.
Used for large-scale phenology monitoring, fire detection, ocean color.

### 2.4 Planet Labs / SkySat (Commercial)

Planet Dove constellation: 3–5 m resolution, daily revisit, RGB+NIR.
SkySat: sub-meter resolution. Licensed access; increasingly open for research.

### 2.5 WorldView (Maxar, Commercial)

Sub-meter resolution (30–50 cm). Extremely detailed but expensive. Used for individual tree mapping, infrastructure inspection.

---

## 3. Understanding Spectral Bands

### 3.1 Why Different Bands?

Every material reflects and absorbs electromagnetic radiation differently. The "spectral signature" of a material — how much it reflects at each wavelength — is like a fingerprint.

**Key spectral regions:**

```
UV      Visible      NIR          SWIR         TIR (Thermal)
|-------|------------|------------|-------------|-------------|
 200nm   400  700nm  700-1400nm  1400-3000nm   8000-14000nm
```

**Vegetation spectral signature (the "red edge"):**
- Absorbs **red** (665 nm) strongly → used for photosynthesis (chlorophyll a, b)
- Reflects **green** (560 nm) moderately → why leaves look green
- Reflects **NIR** (700-1300 nm) very strongly → cell structure scatters NIR internally
- Transition from red-absorbing to NIR-reflecting happens around 700-730 nm = "Red Edge"

The steep slope in the red edge region is uniquely diagnostic of photosynthetically active vegetation. Stressed, senescent, or dead vegetation loses this signature.

### 3.2 Key Spectral Indices

**NDVI (Normalized Difference Vegetation Index)**
$$\text{NDVI} = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}} \in [-1, 1]$$

- Dense healthy vegetation: NDVI 0.6–0.9
- Sparse vegetation / grassland: 0.2–0.5
- Bare soil: 0.1–0.2
- Water / cloud / snow: < 0
- Saturates at high biomass (LAI > 3) — EVI was designed to fix this

**EVI (Enhanced Vegetation Index)**
$$\text{EVI} = 2.5 \cdot \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + 6\rho_{Red} - 7.5\rho_{Blue} + 1}$$

Less saturated at high biomass. Uses blue band to correct aerosol contamination.

**NDWI (Normalized Difference Water Index)**
$$\text{NDWI} = \frac{\rho_{Green} - \rho_{NIR}}{\rho_{Green} + \rho_{NIR}}$$

Detects open water (NDWI > 0) and leaf water content (variant uses SWIR).

**NDSI (Normalized Difference Snow Index)**
$$\text{NDSI} = \frac{\rho_{Green} - \rho_{SWIR}}{\rho_{Green} + \rho_{SWIR}}$$

Snow is bright in Green but dark in SWIR; vegetation is the opposite.

**Red Edge indices for plant health:**
$$\text{Red Edge NDVI} = \frac{\rho_{NIR} - \rho_{RedEdge}}{\rho_{NIR} + \rho_{RedEdge}}$$

More sensitive to subtle chlorophyll changes than standard NDVI. Critical for crop stress monitoring.

---

## 4. SAR (Synthetic Aperture Radar)

### 4.1 What is SAR and Why Does It Matter?

Optical sensors (Sentinel-2, Landsat) only see surface reflectance when the sky is clear. In cloudy tropical or boreal regions, months of data can be lost to cloud cover.

**SAR is an active microwave sensor** — it transmits its own radar pulses and records the backscatter. Microwaves penetrate clouds, smoke, and (partially) dense canopy. SAR works day and night.

**Key SAR missions:**
- Sentinel-1 (ESA): C-band (5.4 GHz), free access, 10 m resolution, 12-day revisit
- ALOS PALSAR-2 (JAXA): L-band (1.2 GHz), penetrates canopy better, useful for biomass
- TerraSAR-X: X-band, very high resolution, commercial

### 4.2 SAR Polarization

SAR transmits and receives horizontally (H) or vertically (V) polarized pulses.

| Polarization | What it measures |
|-------------|-----------------|
| VV | Vertical–vertical: sensitive to surface roughness, soil moisture |
| VH | Vertical–horizontal (cross-pol): sensitive to volume scattering (vegetation, forest) |
| HH | Horizontal–horizontal: penetrates into canopy, used for biomass |

Sentinel-1 acquires VV and VH dual-polarization in most regions.

### 4.3 SAR Backscatter Physics

- Smooth surfaces (calm water) reflect away from sensor → low backscatter (dark)
- Rough surfaces (waves, urban, forest) scatter back → high backscatter (bright)
- Volume scatterers (forest canopy, crops) scatter from multiple layers → VH high
- Double-bounce (vertical structures + ground) → very high HH in urban areas

### 4.4 SAR Challenges for ML

- Speckle noise: coherent imaging creates multiplicative noise — must be filtered
- Geometric distortions: layover (tall structures lean toward sensor), foreshortening
- Units: raw SAR is in linear power or amplitude; convert to dB (logarithmic): $\sigma^0_{dB} = 10 \log_{10}(\sigma^0)$
- Radiometric terrain correction (RTC) needed in mountainous areas

---

## 5. Spatial, Spectral, Temporal, and Radiometric Resolution

These four "resolutions" are often traded off against each other — no sensor is best on all four:

| Resolution type | Definition | Example |
|----------------|------------|---------|
| **Spatial** | Ground pixel size | Sentinel-2: 10 m; WorldView: 0.5 m |
| **Spectral** | Number and width of spectral bands | Sentinel-2: 13 bands; Hyperion: 242 bands |
| **Temporal** | Repeat cycle (revisit time) | Sentinel-2: 5 days; Landsat: 16 days |
| **Radiometric** | Bit depth / sensitivity | Landsat 8: 16-bit; older sensors: 8-bit |

**Practical trade-off example:**
- High spatial resolution (WorldView, 0.5 m) → small swath, rare revisit, expensive
- Moderate spatial resolution (Sentinel-2, 10 m) → wide swath, frequent revisit, free
- Coarse spatial resolution (MODIS, 250 m–1 km) → daily global coverage

For biodiversity monitoring: Sentinel-2 is the sweet spot — 10 m allows individual tree canopy mapping with temporal frequency for phenology.

---

## 6. Coordinate Reference Systems and GeoTIFF

### 6.1 Why CRS Matters

Every EO image must declare *where* it is on Earth. The coordinate reference system (CRS) defines:
1. The geodetic datum (shape model of Earth): typically WGS84
2. The projection (how 3D sphere maps to 2D): UTM, Lambert, Plate Carrée, etc.

**WGS84** is the global standard (GPS uses it). Coordinates are latitude/longitude in degrees.
**UTM (Universal Transverse Mercator)** divides Earth into 60 zones; coordinates are in metres within each zone. Better for measuring distances.

### 6.2 GeoTIFF Format

A GeoTIFF is a TIFF image file with embedded geospatial metadata:

```
GeoTIFF metadata:
  - CRS (EPSG code, e.g., EPSG:32632 = WGS84/UTM Zone 32N)
  - Affine transform: maps pixel (row, col) → (x, y) in CRS coordinates
      x = x_origin + col * pixel_width
      y = y_origin + row * pixel_height
  - No-data value: sentinel value for missing pixels (often 0 or -9999)
  - Bands: multiple spectral bands in a single file
```

Reading a GeoTIFF in Python (rasterio):
```python
import rasterio
with rasterio.open("sentinel2_scene.tif") as src:
    data = src.read()            # shape: (bands, height, width)
    transform = src.transform    # affine: pixel → coordinate
    crs = src.crs               # CRS object (EPSG code)
    bounds = src.bounds          # (left, bottom, right, top)
```

### 6.3 Common Issues

- **CRS mismatch**: drone in WGS84 + satellite in UTM → must reproject before fusing
- **Pixel alignment**: even same CRS can have offset grids → need coregistration
- **Cloud-masked pixels**: stored as 0 or no-data; must be handled before feeding to ML

---

## 7. Key Open Datasets for EO Machine Learning

| Dataset | Size | Task | Resolution | Sensor |
|---------|------|------|------------|--------|
| **BigEarthNet** | 590,326 patches | Multi-label land cover | 120×120 px at 10–60 m | Sentinel-2 |
| **EuroSAT** | 27,000 patches | Land use classification | 64×64 px at 10 m | Sentinel-2 |
| **DeepGlobe** | 1,146 images | Road/building/land cover seg. | 50 cm | Satellite |
| **SpaceNet** | City-scale | Building footprint extraction | 30–50 cm | WorldView |
| **DOTA** | 2,806 images | Object detection (aerial) | Variable | Aerial |
| **iSAID** | 655,451 instances | Instance segmentation | Variable | Aerial |
| **SEN12MS** | 180,662 patches | Multi-modal classification | 10 m | Sentinel-1 + 2 |
| **TreeSatAI** | 50,000 patches | Forest species classification | 20 m + 0.2 m aerial | Sentinel-2 + aerial |

**Data portals:**
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) — all Sentinel data, free
- [USGS EarthExplorer](https://earthexplorer.usgs.gov/) — Landsat, free
- [Google Earth Engine](https://earthengine.google.com/) — petabyte-scale cloud computing on satellite data

---

## 8. Common Failure Modes in EO ML

1. **Spectral shift between sensors**: a model trained on Sentinel-2 will fail on Landsat without [[domain-adaptation]] — different bands, different sensor response functions
2. **Seasonal distribution shift**: training on summer imagery and testing on winter imagery gives very different vegetation appearance
3. **Geographic bias**: most labelled EO data is from Europe and North America — models may fail in tropical forests or African savanna
4. **Cloud contamination**: pixels under thin cloud still have cloud signal; not all cloud masks are perfect
5. **Topographic effects**: slopes facing away from sun appear darker — must correct if using reflectance for classification

---

## See Also

[[satellite-imagery-preprocessing]], [[multi-source-fusion]], [[remote-sensing-foundation-models]], [[biodiversity-ml]], [[clip]], [[blip]], [[domain-adaptation]]
