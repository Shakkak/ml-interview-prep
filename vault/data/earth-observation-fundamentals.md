---
title: "Earth Observation Fundamentals"
tags: [earth-observation, remote-sensing, satellite-imagery, multispectral, sar, geotiff, ndvi, sentinel-2, landsat]
aliases: [Remote Sensing Basics, Satellite Imagery Fundamentals, EO Fundamentals, Earth Observation, EO, Remote Sensing]
difficulty: 2
status: complete
depends_on: [feature-preprocessing, linear-algebra-fundamentals]
related: [satellite-imagery-preprocessing, multi-source-fusion, domain-adaptation, clip]
---

# Earth Observation Fundamentals

---

## Fundamental

**Why EO differs from standard ML imagery:** ordinary photos (ImageNet, COCO) contain everyday RGB images at human-eye wavelengths. Earth observation imagery is fundamentally different:

- **Spectral richness:** data extends far beyond red-green-blue (13 bands in Sentinel-2, hundreds in hyperspectral sensors)
- **Physical meaning:** pixel values encode physical quantities (reflectance, radar backscatter), not perceptual ones
- **Spatial scale:** a single pixel can cover 10 m × 10 m (Sentinel-2) to 1 km × 1 km (MODIS) of ground
- **Temporal depth:** the same location is imaged repeatedly — every 5 days for Sentinel-2

For biodiversity and ecology: satellite data monitors vegetation health (NDVI, LAI, canopy height), land cover change, phenological cycles (greening, peak biomass, senescence), and habitat mapping at continental scale.

**Intuition for spectral bands:** every material reflects electromagnetic radiation differently — this "spectral signature" is like a fingerprint. Healthy vegetation absorbs red (chlorophyll uses it for photosynthesis), reflects green (why leaves look green), and strongly reflects near-infrared (NIR) via leaf cell structure. This signature vanishes in stressed or dead vegetation, making it diagnostic.

### Key Spectral Indices

**NDVI (Normalized Difference Vegetation Index)**:
$$\text{NDVI} = \frac{\rho_\text{NIR} - \rho_\text{Red}}{\rho_\text{NIR} + \rho_\text{Red}} \in [-1, 1]$$

where $\rho_\text{NIR}$ = surface reflectance in the near-infrared band (e.g., Sentinel-2 B8, ~842 nm) and $\rho_\text{Red}$ = surface reflectance in the red band (~665 nm). The normalization by the sum makes the index robust to illumination differences. Dense healthy vegetation: NDVI 0.6–0.9; sparse vegetation: 0.2–0.5; bare soil: 0.1–0.2; water or cloud: < 0. NDVI saturates at high biomass (LAI > 3).

**EVI (Enhanced Vegetation Index)**:
$$\text{EVI} = 2.5 \cdot \frac{\rho_\text{NIR} - \rho_\text{Red}}{\rho_\text{NIR} + 6\rho_\text{Red} - 7.5\rho_\text{Blue} + 1}$$

Less saturated at high biomass. Uses the blue band to correct aerosol contamination. Standard constants (2.5, 6, 7.5, 1) are empirically calibrated for MODIS/Sentinel-2.

**Other key indices:**
- **NDWI** = $(ρ_\text{Green} - ρ_\text{NIR}) / (ρ_\text{Green} + ρ_\text{NIR})$ — detects open water (> 0) and leaf water content
- **Red Edge NDVI** = $(ρ_\text{NIR} - ρ_\text{RedEdge}) / (ρ_\text{NIR} + ρ_\text{RedEdge})$ — more sensitive to subtle chlorophyll changes than standard NDVI; critical for crop stress monitoring

---

## Intermediate

### Key Satellite Missions

**Sentinel-2 (ESA)** — the workhorse of open-access EO for vegetation:
- Revisit: 5 days (two satellites); Swath: 290 km; Free access
- 13 spectral bands: 10 m (RGB + NIR), 20 m (Red Edge, SWIR), 60 m (atmospheric correction)
- Red Edge bands (B5 705nm, B6 740nm, B7 783nm) are absent from Landsat — unique to Sentinel-2 for plant health

**Landsat (USGS/NASA)** — the longest continuous EO record (1972–present):
- 16-day revisit; 30 m spatial resolution; 11 bands similar to Sentinel-2
- Key advantage: 50+ years of comparable observations for long-term trend analysis

**Planet / SkySat (Commercial)** — 3–5 m resolution, daily revisit (Planet Dove); sub-meter (SkySat). Enables individual tree canopy mapping.

### SAR: Radar That Sees Through Clouds

Optical sensors (Sentinel-2) only work under clear skies. **SAR (Synthetic Aperture Radar)** is an active microwave sensor — it transmits radar pulses and records backscatter. Microwaves penetrate clouds, smoke, and (partially) dense canopy. Works day and night.

**Key missions:** Sentinel-1 (C-band, 5.4 GHz, 10 m, free); ALOS PALSAR-2 (L-band, 1.2 GHz, penetrates canopy, useful for biomass).

**SAR polarization:**

| Polarization | What it captures |
|---|---|
| VV | Surface roughness, soil moisture |
| VH (cross-pol) | Volume scattering — sensitive to vegetation structure, forests |
| HH | Penetrates canopy; used for above-ground biomass estimation |

**Physics:** smooth surfaces (calm water) reflect away from sensor → low backscatter. Rough surfaces (forest, crops) scatter back → high backscatter. Urban double-bounce (wall + ground) → very high HH.

**SAR preprocessing requirements:** speckle filtering (multiplicative coherent noise), radiometric terrain correction (RTC) in mountainous areas, conversion to dB: $\sigma^0_{dB} = 10 \log_{10}(\sigma^0)$ where $\sigma^0$ = linear backscatter coefficient.

### Four Resolutions: The Core Trade-Off

| Resolution | Definition | Example |
|---|---|---|
| Spatial | Ground pixel size | Sentinel-2: 10 m; WorldView: 0.5 m |
| Spectral | Number of bands | Sentinel-2: 13 bands; Hyperion: 242 bands |
| Temporal | Revisit cycle | Sentinel-2: 5 days; Landsat: 16 days |
| Radiometric | Bit depth / sensitivity | Landsat 8: 16-bit; older sensors: 8-bit |

High spatial resolution (WorldView, 0.5 m) → small swath, rare revisit, expensive. Sentinel-2 (10 m) is the sweet spot for biodiversity monitoring: individual tree canopy resolution + frequent revisit for phenology.

---

## Advanced

### Coordinate Reference Systems and GeoTIFF

Every EO image declares *where* it is on Earth via a CRS (Coordinate Reference System):
- **WGS84:** global standard; GPS coordinates; latitude/longitude in degrees
- **UTM:** divides Earth into 60 zones; coordinates in metres — better for measuring distances

A GeoTIFF is a TIFF with embedded geospatial metadata: CRS, affine transform (maps pixel (row, col) to (x, y) coordinates), no-data value, and multiple spectral bands.

```python
import rasterio
with rasterio.open("sentinel2_scene.tif") as src:
    data = src.read()            # shape: (bands, height, width)
    transform = src.transform    # affine: pixel → coordinate
    crs = src.crs               # e.g., EPSG:32632 = WGS84/UTM Zone 32N
```

**Common issues:** CRS mismatch (drone in WGS84 + satellite in UTM → must reproject before fusing); pixel grid offset (same CRS but different origins → need coregistration); cloud-masked pixels (stored as 0 or no-data, must be excluded before ML).

### Key Open Datasets

| Dataset | Size | Task | Resolution | Sensor |
|---|---|---|---|---|
| BigEarthNet | 590K patches | Multi-label land cover | 120×120 px at 10–60 m | Sentinel-2 |
| EuroSAT | 27K patches | Land use classification | 64×64 px at 10 m | Sentinel-2 |
| SEN12MS | 180K patches | Multi-modal classification | 10 m | Sentinel-1 + 2 |
| TreeSatAI | 50K patches | Forest species classification | 20 m + 0.2 m aerial | Sentinel-2 + aerial |
| SpaceNet | City-scale | Building footprint extraction | 30–50 cm | WorldView |

**Data portals:** Copernicus Open Access Hub (all Sentinel), USGS EarthExplorer (Landsat), Google Earth Engine (petabyte-scale cloud computing on satellite data).

### Common EO Failure Modes

1. **Spectral shift between sensors:** a model trained on Sentinel-2 fails on Landsat without [[domain-adaptation]] — different bands, different sensor response functions
2. **Seasonal distribution shift:** summer → winter imagery gives very different vegetation appearance; models overfit to seasonal signal if trained on single-season data
3. **Geographic bias:** most labelled EO data is from Europe and North America — models fail in tropical forests or African savanna
4. **Cloud contamination:** pixels under thin cloud still have cloud signal; not all cloud masks are perfect — check cloud probability layer before use
5. **Topographic effects:** slopes facing away from sun appear darker — must apply topographic correction if using reflectance for classification in mountainous areas

---

## Links

- [[feature-preprocessing]] — EO data requires sensor-specific preprocessing (radiometric calibration, atmospheric correction) before generic ML preprocessing steps apply
- [[linear-algebra-fundamentals]] — spectral bands are vectors; band ratios, PCA on hyperspectral cubes, and change detection all use linear algebra on the band dimension
- [[satellite-imagery-preprocessing]] — follow-up covering the full preprocessing pipeline from raw digital numbers to model-ready tensors
- [[multi-source-fusion]] — fusing satellite, drone, and in-situ data requires understanding each sensor's spatial/spectral/temporal properties
- [[clip]] — CLIP-pretrained models transfer to satellite imagery; understanding EO sensor characteristics informs which image transformations preserve CLIP's image statistics
- [[domain-adaptation]] — EO models trained on one sensor or season often fail on another; physical differences between sensors guide domain adaptation strategies
