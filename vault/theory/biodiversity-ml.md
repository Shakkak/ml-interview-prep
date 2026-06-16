---
title: "Biodiversity and Ecology Machine Learning"
tags: [biodiversity-ml, species-distribution-modelling, maxent, plant-traits, phenology, long-tail, gbif, inaturalist, bioclip, fine-grained-recognition]
aliases: [Ecology ML, SDM, Species Distribution Modelling, Biodiversity Informatics ML]
status: complete
depends_on: [clip, earth-observation-fundamentals, statistical-inference-mle]
---

This file covers machine learning methods for biodiversity and ecology — from species distribution modelling to plant trait prediction, phenology monitoring, and fine-grained species recognition. Written to be self-contained for someone coming from ML without ecology background.

## 1. Why Biodiversity ML is Different

Biodiversity data has properties that make it challenging for standard ML pipelines:

1. **Extreme class imbalance**: millions of observations for common species (house sparrow, dandelion) but only a handful for rare species. Power-law distribution.
2. **Presence-only data**: most biodiversity observations record *where* a species was seen, but not *where it wasn't* (pseudo-absence problem).
3. **Observer bias**: amateur naturalists concentrate in cities, parks, and accessible areas. The map of observations reflects human activity as much as species distribution.
4. **Taxonomic complexity**: species boundaries are uncertain (cryptic species, subspecies). Labels in databases may contain historical errors.
5. **High intraclass variability**: the same species looks very different across life stages (juvenile/adult), seasons (summer plumage vs winter plumage), and geographic populations.

---

## 2. Species Distribution Modelling (SDM)

### 2.1 What is SDM?

SDM (also called *habitat modelling* or *ecological niche modelling*) estimates the probability of a species occurring at a location, given environmental variables (climate, elevation, soil type, vegetation).

**Goal**: given occurrence observations of species X, predict where else in the world species X is likely to occur.

**Applications**:
- Prioritizing areas for conservation
- Predicting range shifts under climate change
- Identifying new sites for survey campaigns
- Mapping potential invasive species spread

### 2.2 Environmental Predictors

Models use layers of environmental data as predictors:

| Variable | Example | Source |
|----------|---------|--------|
| Temperature | Mean annual temperature, temp of coldest quarter | WorldClim, CHELSA |
| Precipitation | Annual rainfall, precipitation seasonality | WorldClim |
| Topography | Elevation, slope, aspect | SRTM DEM |
| Vegetation | NDVI, land cover type | Sentinel-2, ESA CCI |
| Soil | pH, organic carbon, texture | SoilGrids |

WorldClim provides 19 bioclimatic variables (BIO1–BIO19) at 1 km resolution, globally, derived from temperature and precipitation data. These are the standard inputs to most SDM models.

### 2.3 MaxEnt: Maximum Entropy Modelling

**What is MaxEnt?** The most widely used SDM method, based on the *principle of maximum entropy*: among all probability distributions consistent with the observed data, choose the one with maximum entropy (most uniform, least assumptions).

**Intuition**: imagine you've observed a species at 100 locations. You want to predict the probability of occurrence everywhere else. MaxEnt says: make the simplest assumption consistent with what you've seen. Don't assume the species prefers conditions it hasn't been found in; don't rule out conditions it hasn't been found in.

**Mathematical form**: fit a probability distribution $q(x)$ over geographic locations $x$ that:
1. Matches the feature expectations observed at presence locations: $\mathbb{E}_q[f_k(x)] = \mathbb{E}_{presence}[f_k(x)]$ for each environmental feature $f_k$
2. Maximizes entropy: $H(q) = -\sum_x q(x) \log q(x)$

The solution is a Gibbs distribution:
$$q(x) \propto \exp\left(\sum_k \lambda_k f_k(x)\right)$$

The $\lambda_k$ are learned to match the observed feature expectations at presence points.

**What MaxEnt needs**:
- Presence records (GPS coordinates of species observations)
- Environmental rasters (one layer per predictor)
- A background sample (random points from the study area, representing the "available" landscape)

**Key limitation**: MaxEnt produces a *relative probability of presence*, not an absolute probability. It tells you "this location is 3× more suitable than average", not "probability of occurrence is 0.7".

### 2.4 BIOCLIM

An older, simpler SDM: define the species' environmental niche as the rectangular "envelope" (range) of each environmental variable at presence locations. Predict presence where all variables fall within the envelope.

Faster than MaxEnt but much less flexible — assumes no interactions between variables and rectangular niche boundaries.

### 2.5 Presence-Only vs Presence-Absence Data

**Presence-only (PO) data**: most biodiversity databases (GBIF, iNaturalist) only record where a species was observed, not where it wasn't. This makes modelling harder because absence could mean "species doesn't occur here" or "species wasn't observed here".

**Presence-absence (PA) data**: systematic surveys that also record confirmed absences. Much rarer, but statistically more tractable. With PA data, you can use standard logistic regression or random forests.

**Pseudo-absence**: a common workaround for PO data — randomly sample locations from the study area and treat them as "absences". The interpretation is "background" vs "target group". MaxEnt uses this implicitly.

### 2.6 Deep Learning for SDM

Recent approaches train CNNs or transformers on:
- Remote sensing imagery patches at occurrence locations (multi-spectral, seasonal stacks)
- Environmental variable rasters
- Satellite time series (temporal phenology signals)

**SatBird / DeepBioClim**: train models on Sentinel-2 + bioclimatic variables to predict species assemblages. The model learns which image textures correspond to which habitat types.

**SINR (Spatially Implicit Neural Representations)**: learns a neural field $f_\theta(lat, lon)$ that predicts species occurrence probabilities everywhere. Position encoded with spherical harmonics; trained on GBIF observations.

---

## 3. GBIF and iNaturalist

### 3.1 GBIF (Global Biodiversity Information Facility)

GBIF is the world's largest aggregator of biodiversity occurrence data:
- **>2 billion occurrence records** from 1.7 million species
- Data from museum specimens, citizen science, research surveys, government monitoring
- Open access, standardised format (Darwin Core)

**Issues for ML**:
- **Taxonomic synonyms**: the same species may be referenced under multiple scientific names
- **Coordinate uncertainty**: older records from museum specimens may have very imprecise GPS
- **Observer bias**: 80% of records come from Europe and North America
- **Temporal coverage**: museum specimens from 1800s mixed with recent observations

### 3.2 iNaturalist

Citizen science platform with ~170M observations (as of 2024), primarily 2010s–present.
- Rich metadata: photo, GPS, observer ID, timestamp
- Taxonomic ID verified by AI (Seek app) and community voting
- **iNaturalist Research Grade**: confirmed by at least 2 independent identifications
- Used in competitions (iNat 2017, 2018, 2019, 2021) — standard benchmarks for fine-grained recognition

### 3.3 Observer Bias

Urban areas and accessible parks are heavily oversampled. A species seen in 10,000 observations near cities but 5 observations deep in a national park is almost certainly more common than the count ratio suggests.

**Correction methods**:
- **Spatial thinning**: thin observations so no two are closer than X km. Removes spatial clustering.
- **Target group background**: use records of all species in the same taxonomic group as background points. If observers go to a region, they observe many species there — so background point density reflects observer effort.
- **Bias offset in MaxEnt**: provide an observer effort layer (total number of observations per grid cell) as an offset to the background sampling.

---

## 4. Long-Tail Recognition in Biodiversity

### 4.1 The Power-Law Problem

Species observation frequencies follow a power law: a handful of species have millions of observations, while most species have only tens.

Example from iNat 2021 (10,000 species):
- Top 10 species: >50,000 observations each
- Median species: <500 observations
- Bottom 10%: <50 observations

Standard training on this data produces models that are excellent at common species and terrible at rare ones — which are often the most conservation-relevant.

### 4.2 Techniques for Long-Tail Recognition

**Class-balanced sampling**: oversample rare classes and undersample common ones during training. Simple but effective baseline.

**[[loss-focal|Focal Loss]]**: reduces the loss weight for easy (well-classified) examples, forcing the model to focus on hard (typically rare) classes:
$$FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$$
where $\gamma > 0$ (typically 2) down-weights easy examples and $\alpha_t$ balances class frequencies.

**Class-conditional features**: rather than one classifier, learn class-specific feature prototypes. Few-shot learning approaches (prototypical networks, matching networks) work naturally with limited observations per class.

**Large-vocabulary contrastive pretraining**: BioCLIP (below) pretrained on all taxa in iNaturalist achieves excellent few-shot recognition because it learns a rich embedding space that separates even rare species.

**Self-supervised pretraining on unlabeled images**: use all plant/animal photos from the web without labels → better backbone → better few-shot on rare species.

### 4.3 BioCLIP

**Paper**: "BioCLIP: A Vision Foundation Model for the Tree of Life" (Stevens et al., 2024)

Standard [[clip|CLIP]] was trained on internet text — biological descriptions in that corpus are sparse and colloquial. BioCLIP is CLIP retrained specifically on biological images:

- **Training data**: 450K TreeOfLife-10M images (10M images from iNaturalist, spanning all life kingdoms)
- **Text prompts**: hierarchical taxonomic names at multiple levels: kingdom → phylum → class → order → family → genus → species. Each image generates multiple text labels.
- **CLIP training** on (image, taxonomic text) pairs

**Intuition**: training on the full taxonomic hierarchy allows BioCLIP to learn that a "Quercus robur" image is similar to other "Quercus" species images (sharing genus-level features) and also similar to "Fagaceae" family images (sharing family-level features). The embedding space respects taxonomic structure.

**Results**: BioCLIP significantly outperforms CLIP on fine-grained species classification across birds, fungi, insects, plants — especially for rare species with few observations.

---

## 5. Plant Functional Traits

### 5.1 What Are Plant Functional Traits?

Plant functional traits are measurable properties of plants that link plant physiology and ecology. They help predict how plants grow, compete, and respond to environmental change.

**Key traits**:

| Trait | Abbreviation | What it measures |
|-------|-------------|-----------------|
| Specific Leaf Area | SLA | Leaf area per unit dry mass (cm²/g). High SLA = fast-growing, resource-rich environments |
| Leaf Nitrogen Content | LNC | % nitrogen per dry mass. Proxy for photosynthetic capacity |
| Leaf Chlorophyll | CHL | Chlorophyll a+b concentration. Related to photosynthetic rate |
| Leaf Dry Matter Content | LDMC | Dry mass / fresh mass. Low LDMC = stress-tolerant species |
| Plant Height | H | Maximum height. Proxy for competitive ability for light |
| Stem Wood Density | WD | Biomass per volume of stem. High WD = slow-growing, durable |

**TRY Database**: global repository of plant trait measurements from 1M+ records across 280,000+ species. Standard reference for SDM + trait modelling.

### 5.2 Predicting Traits from Imagery

Traits correlate with spectral signatures because plant biochemistry controls how leaves interact with light. High chlorophyll plants strongly absorb red light. High water content plants absorb in the SWIR.

**From hyperspectral data**: spectral bands at 400–2500 nm contain direct biochemical information. Partial Least Squares Regression (PLSR) can predict LNC, SLA, CHL from hyperspectral spectra with R² > 0.8 in controlled conditions.

**From RGB drone imagery**: less direct. Deep learning can learn correlations between leaf colour, texture, and traits — but performance drops significantly compared to hyperspectral.

**From VLMs**: a [[vlm-architectures|VLM]] can be prompted "What is the leaf texture and colour? Is the plant stressed?" → these linguistic observations correlate with traits even without explicit numerical prediction. More useful for qualitative trait assignment than quantitative prediction.

**Practical pipeline for trait mapping**:
1. Fly drone over study area (RGB + possibly multispectral)
2. Run plant detection ([[open-vocabulary-detection|Grounding DINO]] → SAM) to get individual plant masks
3. Crop and feed each plant to a trait prediction CNN (or VLM)
4. Geo-reference predictions back to field coordinates
5. Combine with Sentinel-2 time series for landscape-scale trait maps

---

## 6. Phenology Monitoring

### 6.1 What is Phenology?

Phenology is the study of cyclic and seasonal natural phenomena — when plants bud, bloom, reach peak biomass, and senesce. Climate change is altering phenological timing, with cascading effects on biodiversity.

**Key phenological stages** (for vegetation):
- **Green-up / bud burst**: onset of spring leaf emergence
- **Peak greening**: maximum LAI / NDVI
- **Maturity**: peak photosynthetic activity
- **Senescence**: yellowing and leaf fall
- **Dormancy**: bare/brown land surface

### 6.2 Satellite-Based Phenology

Vegetation indices (NDVI, EVI) form time series that reveal phenological cycles:

```
NDVI
 1.0 |          ****
 0.8 |        **    **
 0.6 |      **        **
 0.4 |     *            *
 0.2 |   **              ***
 0.0 |**                    ****
     Jan   Mar   May   Jul   Sep   Nov  Dec
```

**Start of Season (SOS)**: estimated as the date when NDVI crosses a threshold (e.g., 0.3) on the ascending limb, or the inflection point of a fitted logistic curve.

**Double logistic fitting**: the NDVI time series is typically modelled as the sum of two logistic functions (one for green-up, one for senescence):
$$NDVI(t) = V_{min} + (V_{max} - V_{min}) \left[\frac{1}{1+e^{-m_1(t-t_1)}} - \frac{1}{1+e^{-m_2(t-t_2)}}\right]$$

**Tools**: TIMESAT software fits these curves to MODIS/Landsat/Sentinel-2 time series globally.

### 6.3 Derived Ecological Variables from Vegetation Indices

| Variable | Definition | Ecological Use |
|----------|-----------|---------------|
| **NDVI** | (NIR-Red)/(NIR+Red) | General vegetation greenness |
| **EVI** | Enhanced, less saturation | Better for dense canopy |
| **LAI** | Leaf Area Index: total leaf area per ground area | Canopy structure, GPP estimation |
| **FAPAR** | Fraction of Absorbed PAR | Photosynthetically active radiation absorption |
| **GPP** | Gross Primary Production | Carbon assimilation by vegetation |

**NDVI vs EVI vs LAI comparison**:
- NDVI saturates above LAI ≈ 3 (dense canopy looks the same as moderate canopy)
- EVI corrects this using the blue band to remove aerosol effects; better for tropical forests
- LAI is a physical quantity (m² leaf per m² ground), not a reflectance ratio; measured in situ or inverted from multi-angle optical data; ranges from 0 (bare soil) to ~12 (dense conifer forest)

---

## 7. Fine-Grained Visual Recognition (FGVC) in Biology

Fine-grained recognition distinguishes between visually similar subcategories (e.g., 100 oak species, 300 bird species in a single family).

### 7.1 Why it's Hard

- High inter-class similarity: many species differ by subtle color patterns, leaf shape, or flower morphology
- High intra-class variability: juvenile vs adult, seasonal variation, geographic morphs
- Background clutter: wild photos have complex backgrounds that can confuse models
- Scale variation: a full tree and a close-up leaf of the same species look completely different

### 7.2 Benchmark Datasets

| Dataset | Categories | Images | Domain |
|---------|-----------|--------|--------|
| iNat 2021 | 10,000 species | 2.7M | Multi-kingdom |
| CUB-200-2011 | 200 bird species | 11,788 | Birds |
| NABirds | 555 bird species | 48K | N. American birds |
| PlantNet-300K | 306 plant species | 306K | Plants |
| Herbarium 2021 | 15,501 species | 2.5M | Herbarium specimens |
| FungiCLEF | ~1,500 fungi | 295K | Fungi |

### 7.3 Approaches

**Part-based models**: detect key semantic parts (bird head, wings, tail) and classify from part crops.

**Attention mechanisms**: learned spatial attention focuses on discriminative regions (DINO self-supervised attention reliably identifies object parts without labels).

**Hierarchical classification**: first classify at family level, then within-family. Reduces confusion among unrelated species.

**BioCLIP embedding**: use BioCLIP embeddings + k-NN classifier. Zero-shot and few-shot classification with competitive accuracy even for rare species.

**Test-time augmentation**: average predictions across multiple crops, flips, scales.

---

## 8. Common Mistakes in Ecology ML

1. **Spatial autocorrelation in train/test split**: if you split by random record, train and test overlap spatially → inflated performance. Always split by geographic region (hold-out 20% of spatial blocks).

2. **Ignoring temporal shift**: a model trained on 2010–2018 phenology may fail on 2020+ data due to climate-induced shifts. Always evaluate on held-out years.

3. **Confusing observer effort with species richness**: an area with 1,000 observations of 50 species may simply be heavily surveyed, not more biodiverse than a remote area with 10 observations.

4. **Using NDVI from raw DN**: always atmospherically correct before computing vegetation indices. TOA vs BOA NDVI can differ by 0.15–0.3 units in conditions with aerosol contamination.

5. **Ignoring taxonomic synonymy**: the same species may appear under multiple names in GBIF — deduplicate by backbone taxonomy (GBIF taxonomic ID, not raw scientific name string).

---

## Links

- [[clip]] — BioCLIP adapts CLIP to fine-grained species classification; the contrastive pretraining on scientific name-image pairs aligns visual and taxonomic representations
- [[earth-observation-fundamentals]] — biodiversity monitoring increasingly uses satellite EO data; species distribution models (SDMs) are being coupled with temporal remote sensing time series
- [[statistical-inference-mle]] — MaxEnt (Maximum Entropy) species distribution modeling is equivalent to MLE in an exponential family model; GBIF presence-only data requires presence-background sampling
- [[satellite-imagery-preprocessing]] — multi-spectral and temporal satellite data are inputs to EO-based biodiversity models; band normalization and cloud masking are preprocessing prerequisites
- [[vlm-architectures]] — VLMs like GeoChat are used for biodiversity monitoring; they combine visual encoders with language models for multimodal ecological question answering
- [[open-vocabulary-detection]] — open-vocabulary detection enables zero-shot species detection without training on every species; CLIP embeddings provide the text-image alignment
- [[vlm-explainability]] — biodiversity applications require explainability for scientific trust; Grad-CAM and attention rollout show which image regions drive species predictions
- [[remote-sensing-foundation-models]] — EO foundation models pretrained on satellite data provide stronger representations for biodiversity tasks than ImageNet-pretrained models
