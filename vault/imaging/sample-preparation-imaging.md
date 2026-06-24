---
title: "Sample Preparation in Imaging"
tags: [sample-preparation, staining, contrast-agents, fluorescent-labels, imaging-bias]
aliases: [staining, contrast agents, fluorescent dyes, sample prep, immunofluorescence]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview]
related: [image-artifacts, sensor-technology, modality-selection, data-augmentation]
---

## Fundamental

### Why Sample Preparation Matters for AI

In many imaging modalities — fluorescence microscopy, histology, and contrast-enhanced CT/MRI — the image you see is not purely the object as it exists. It is the object after deliberate modification to make certain structures visible. This modification introduces:

- **Biases**: structures that stain more intensely appear more prominent, regardless of their actual biological importance.
- **Artefacts**: preparation steps can damage, distort, or introduce foreign material into the sample.
- **Variability**: different labs use different protocols, dye concentrations, and fixation methods — producing visually different images of the same cell type.

An AI model trained on images from one preparation protocol may fail entirely on images from a slightly different protocol, even if the underlying biology is identical.

### Staining in Histology

Histological (tissue) samples are fixed, sliced into thin sections (3–10 µm), and stained with chemical dyes to make cell structures visible under a bright-field microscope.

**H&E staining** (Haematoxylin and Eosin) is the clinical standard:
- **Haematoxylin**: stains nuclei blue–purple (binds to DNA)
- **Eosin**: stains cytoplasm and extracellular matrix pink–red

Other common stains:
- **PAS** (Periodic Acid-Schiff): stains glycogen and carbohydrates magenta
- **Masson's Trichrome**: collagen (blue), muscle (red), cytoplasm (pink)
- **IHC** (immunohistochemistry): antibody-linked dye highlights a specific protein

**AI implication**: H&E stain colour varies between labs due to dye concentration, staining duration, and scanner calibration. **Stain normalisation** (e.g. Macenko, Reinhard, or deep learning methods) is essential preprocessing before training pathology AI. A model trained on one hospital's staining protocol will often degrade at another hospital without normalisation.

### Fluorescent Labels in Microscopy

Fluorescence microscopy requires that target structures emit light. Two strategies:

1. **Organic fluorescent dyes** (e.g. DAPI for nuclei, phalloidin for actin): chemical dyes that bind specific molecular targets. Applied externally after fixation. Require the cell to be dead (fixed).

2. **Fluorescent proteins** (e.g. GFP — Green Fluorescent Protein): genetically encoded; cells produce the fluorescent protein themselves. Can be used in living cells.

Each fluorophore has specific **excitation** and **emission** wavelength ranges. The imaging system must use a matching excitation light source and emission filter.

**Common fluorophores and their colours:**

| Fluorophore | Excitation (nm) | Emission (nm) | Typical target |
|-------------|-----------------|---------------|----------------|
| DAPI | 360 | 460 (blue) | Nuclei |
| GFP | 488 | 510 (green) | Genetically encoded |
| Cy3 | 550 | 570 (yellow) | Antibody conjugate |
| Cy5 | 650 | 670 (red) | Antibody conjugate |
| mCherry | 587 | 610 (red) | Genetically encoded |

**AI implication**: the channel in a multi-channel fluorescence image encodes **which molecule was labelled**, not just a colour. Each channel must be treated independently — stacking them as RGB and applying a natural-image CNN is incorrect. Models should be designed for $C$-channel input where each channel corresponds to a biological target.

### Contrast Agents in CT and MRI

**CT contrast agents**: typically iodine-based (e.g. Omnipaque). Iodine strongly absorbs X-rays, making blood vessels and vascularised structures appear brighter (higher HU). Injected intravenously; imaged at different time phases:
- **Arterial phase**: arteries enhanced
- **Portal venous phase**: liver parenchyma and veins enhanced
- **Delayed phase**: kidneys and some tumours enhanced

**MRI contrast agents**: typically gadolinium-based (e.g. Gadovist). Gadolinium shortens T1 relaxation time, making vascularised structures appear brighter on T1-weighted images. Blood–brain barrier breakdown (tumours, inflammation) causes gadolinium to accumulate in brain lesions.

**AI implication**: a model trained on contrast-enhanced CT will fail on non-enhanced CT of the same anatomy — vascular structures look entirely different. Training data must be labelled by contrast phase, and models should either be trained separately per phase or explicitly conditioned on whether contrast was used.

---

## Intermediate

### Fixation and Its Artefacts

Before staining, tissue is fixed (chemically preserved) to prevent degradation. The most common fixative is formalin (FFPE: Formalin-Fixed Paraffin-Embedded tissue).

Fixation artefacts:
- **Shrinkage**: formalin causes tissue to shrink (up to 20%), distorting morphology.
- **Antigen masking**: formalin cross-links proteins, blocking some antibody targets. Antigen retrieval steps partially reverse this.
- **Autofluorescence**: FFPE tissue has elevated background fluorescence at most excitation wavelengths, reducing SNR.

**AI implication**: fixation-induced shrinkage changes cell size and shape measurements. Models that rely on absolute morphological measurements (e.g. nuclear area) must be calibrated against the fixation protocol. Autofluorescence in FFPE fluorescence imaging adds a spatially non-uniform background that must be subtracted before analysis.

### Spectral Unmixing and Cross-Talk

When multiple fluorophores are used simultaneously, their emission spectra often overlap. Photons from fluorophore A are detected in the channel intended for fluorophore B. This is called **spectral cross-talk** (or bleed-through).

Spectral unmixing separates the contributions:

$$\mathbf{I}_\text{observed} = \mathbf{M} \cdot \mathbf{I}_\text{true}$$

where $\mathbf{M}$ is the mixing matrix of cross-talk coefficients between channels. Solving for $\mathbf{I}_\text{true}$ requires knowing $\mathbf{M}$ (measured from single-stain control images).

**AI implication**: if spectral unmixing is not applied, co-localisation analysis of two proteins will be confounded by cross-talk (apparent co-localisation that is actually a measurement artefact). Deep learning models that learn directly from uncorrected channels will fit the cross-talk pattern as if it were biological signal.

### Batch Effects from Protocol Variation

Even when following the same protocol, staining results vary across:
- Different lots of the same dye (dye concentration, purity)
- Different technicians (manual steps: pipetting, washing times)
- Different staining machines
- Different storage times of prepared slides

This produces **batch effects**: systematic differences between groups of images processed at different times or locations, unrelated to the biological variable of interest.

**AI implication**: batch effects are among the most common failure modes in computational pathology and quantitative microscopy. Models trained on one batch can dramatically fail on another. Standard mitigations: batch normalisation in preprocessing (not in the model), mixed-batch training, domain adaptation, and prospective standardisation of protocols.

### Contrast Timing and Wash-Out

For CT contrast studies, the timing of image acquisition after contrast injection determines which structures are enhanced. Scan too early → only arteries enhanced; scan too late → contrast washed out. Multi-phase protocols acquire several scans at different delay times after injection.

**AI implication**: models trained on single-phase data will see only part of the contrast information. Multi-phase fusion models (e.g. treating each phase as a separate input channel) leverage the complementary enhancement patterns. Metadata about contrast phase must be preserved through the data pipeline.

---

## Advanced

### Label Efficiency Trade-offs

Sample preparation that makes structures visible also limits what structures can be simultaneously observed. In fluorescence microscopy, typically only 3–5 channels are available per experiment (limited by non-overlapping fluorophore spectra). Choosing which proteins to label is a scientific decision that permanently constrains what an AI model can learn.

Emerging technologies (cyclic immunofluorescence — CyCIF, CODEX) allow sequential staining of 40+ targets on the same tissue section by stripping and re-staining. These create extremely high-dimensional per-cell feature vectors, enabling spatial biology AI that models cellular neighbourhood and tissue architecture.

**AI implication**: spatial biology datasets have very different structure from standard fluorescence images — they are more like point clouds or graph-structured data than regular images. Graph neural networks and spatial statistics are increasingly used instead of CNNs.

---

## Links

- [[imaging-modalities-overview]] — fluorescence microscopy, histology, and contrast CT/MRI all depend on sample preparation
- [[image-artifacts]] — photobleaching, cross-talk, and fixation artefacts originate in the preparation process
- [[sensor-technology]] — the sensor's spectral sensitivity must match the fluorophore emission spectrum
- [[modality-selection]] — sample preparation requirements (fixation, staining, contrast injection) constrain modality choice
- [[data-augmentation]] — stain augmentation (random H&E colour perturbation) is a common strategy to handle staining variability
