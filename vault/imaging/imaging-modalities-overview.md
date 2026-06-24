---
title: "Imaging Modalities Overview"
tags: [imaging-modalities, ct, mri, pet, ultrasound, fluorescence-microscopy, hyperspectral]
aliases: [imaging modalities, medical imaging, scientific imaging, image modalities]
difficulty: 1
status: complete
depends_on: []
related: [image-quality-metrics, image-artifacts, modality-selection, fourier-transform, unet]
---

## Fundamental

### What Problem Do Imaging Modalities Solve?

Different physical phenomena carry different kinds of information about an object. X-rays pass through soft tissue but are absorbed by bone. Radio waves reveal hydrogen atom density in water-rich tissue. Visible light reflected from a stained cell slide tells you about molecular structure. Each imaging modality exploits a different physical interaction to extract information that the others cannot see.

For an AI system, the modality determines the **nature of the input data**: its dimensionality, resolution, noise statistics, and what features are semantically meaningful. A model trained on CT volumes cannot be naively applied to MRI volumes, even for the same anatomy — the pixel values mean entirely different things.

### The Core Table

| Modality | Physical principle | Output data type | Typical resolution | Key AI concern |
|----------|--------------------|------------------|--------------------|----------------|
| CT (Computed Tomography) | X-ray attenuation | 3D grayscale volume (HU values) | ~0.5–1 mm isotropic | High contrast bone/air, low soft tissue contrast |
| MRI (Magnetic Resonance Imaging) | Nuclear spin relaxation (hydrogen) | 3D volume; multiple contrast weightings (T1, T2, FLAIR…) | ~1 mm isotropic | Slow acquisition, rich soft tissue contrast |
| PET (Positron Emission Tomography) | Gamma-ray coincidence from radiotracer | 3D functional volume | ~4–6 mm (low resolution) | Metabolic signal, always fused with CT/MRI |
| Ultrasound | Acoustic reflection | 2D or 3D grayscale, real-time | ~0.1–1 mm | Speckle noise, operator-dependent, real-time |
| Fluorescence microscopy | Fluorophore excitation/emission | 2D or 3D multi-channel | Sub-micron | Multi-channel, sample preparation artefacts |
| Hyperspectral imaging | Reflectance across hundreds of spectral bands | 3D data cube (x, y, wavelength) | Spatial: cm–mm; spectral: nm | Very high-dimensional, band selection needed |

### CT — 3D Grayscale Volumes

A CT scanner rotates an X-ray source around the patient and measures attenuation at many angles. Reconstruction algorithms (filtered backprojection or iterative methods) produce a 3D volume where each voxel's value is a **Hounsfield Unit (HU)**: air = −1000 HU, water = 0 HU, bone = +1000 HU.

AI implication: the grayscale value has an absolute physical meaning, making CT data relatively consistent across scanners. Segmentation and detection tasks (lung nodules, vertebrae, ribs) can exploit this predictable intensity range.

### MRI — Multi-Contrast 3D Volumes

MRI measures how hydrogen atoms in water relax after a radiofrequency pulse. Different pulse sequences (T1, T2, T2*, FLAIR, DWI…) produce different tissue contrasts from the same anatomy. The pixel values are **relative**, not absolute — they depend on scanner field strength, coil, and acquisition parameters.

AI implication: MRI images from different scanners are not directly comparable in intensity. Normalisation and harmonisation are essential preprocessing steps. The multi-contrast nature is both an opportunity (different sequences reveal different pathologies) and a challenge (which sequences are available varies per site).

### PET — Low-Resolution Functional Images

PET detects pairs of gamma rays emitted when a positron from a radiotracer annihilates with an electron. Tracers like FDG (fluorodeoxyglucose) accumulate in metabolically active tissue (e.g. tumours). The resulting image reflects **metabolic activity**, not anatomy.

AI implication: PET is always used alongside CT or MRI for anatomical localisation (PET/CT or PET/MRI). Resolution is low (~4–6 mm) and images are noisy. Models must handle the misalignment between functional and structural images, and the sparsity of signal.

### Ultrasound — Real-Time, Noisy 2D/3D

Ultrasound emits acoustic pulses and records reflections at tissue boundaries. Images are acquired in real time, making ultrasound the only modality usable for dynamic processes (cardiac motion, blood flow, needle guidance).

AI implication: ultrasound images have characteristic **speckle noise** (coherent interference pattern) and depend heavily on probe angle and operator skill. Features are less consistent than CT/MRI. Augmentation strategies must account for speckle and orientation variability.

### Fluorescence Microscopy — Multi-Channel Sub-Cellular Images

Biological samples are stained with fluorescent dyes or genetically engineered to express fluorescent proteins. Each channel corresponds to a different fluorophore targeting a specific cellular structure (nucleus, membrane, cytoskeleton).

AI implication: multi-channel images are not RGB photographs. Each channel is an independent grayscale signal. Cell segmentation (e.g. with [[unet]]) must handle overlapping cells, varying cell density, and photobleaching artefacts from prolonged exposure.

### Hyperspectral Imaging — High-Dimensional Cubes

Rather than 3 colour channels (RGB), a hyperspectral sensor captures hundreds of spectral bands at each pixel. The result is a data cube with dimensions (height × width × bands). Used in remote sensing, agriculture, and food inspection.

AI implication: the spectral dimension is high (100–400 bands), but bands are highly correlated. Dimensionality reduction (PCA, band selection) is typically required before feeding into ML models. See [[earth-observation-fundamentals]] for remote sensing context.

---

## Intermediate

### Resolution Trade-offs Across Modalities

Every modality faces a fundamental trade-off between spatial resolution, temporal resolution, spectral resolution, and signal-to-noise ratio (SNR). Improving one typically degrades another.

- **CT**: good spatial resolution, fast acquisition (seconds), no spectral resolution, ionising radiation dose limits repeat scans.
- **MRI**: similar spatial resolution, slow acquisition (minutes), multiple contrast weightings available, no radiation.
- **PET**: poor spatial resolution, slow acquisition, provides functional/metabolic information unavailable from anatomy-only modalities.
- **Ultrasound**: variable spatial resolution (frequency-dependent), real-time temporal resolution, operator-dependent.
- **Fluorescence microscopy**: sub-micron spatial resolution, limited to transparent or thin samples, photobleaching limits temporal sampling.
- **Hyperspectral**: rich spectral information, usually at the cost of spatial resolution or acquisition time.

### Data Formats and Dimensions

| Modality | Typical shape | Common format |
|----------|--------------|---------------|
| CT | (512, 512, N_slices) | DICOM, NIfTI |
| MRI | (256, 256, N_slices) per sequence | DICOM, NIfTI |
| PET | (128, 128, N_slices) | DICOM |
| Ultrasound | (H, W) or (H, W, T) for cine | DICOM, AVI |
| Fluorescence microscopy | (C, Z, H, W) | TIFF, OME-TIFF |
| Hyperspectral | (H, W, B) | ENVI, GeoTIFF |

### How Modality Choice Affects ML Architecture

The dimensionality and channel structure of input data directly constrains model architecture:

- **3D volumes** (CT, MRI): 3D CNNs or 2.5D slice-stacking strategies. Memory is the main bottleneck.
- **Multi-contrast MRI**: early or late fusion of sequences; or a single network with multiple input channels.
- **Multi-channel microscopy**: each channel as an independent input; avoid treating as RGB.
- **Hyperspectral cubes**: spectral attention mechanisms, band selection layers, or 1D convolutions along the spectral axis.
- **Real-time ultrasound**: temporal models (RNNs, 3D CNNs over time) for dynamic analysis.

---

## Advanced

### Partial Volume Effect

When a voxel straddles two tissue types, its value is a mixture of both signals. This is especially problematic at tissue boundaries in CT and MRI, and causes boundary voxels to appear as intermediate intensities rather than clean class labels.

AI implication: segmentation models near boundaries face inherently ambiguous labels. Probabilistic outputs (soft segmentation) and boundary-aware loss functions (e.g. [[loss-dice]] + boundary loss) help, but the underlying physical ambiguity cannot be fully resolved.

### Domain Shift Between Scanners

Even within one modality (e.g. MRI), images from different scanner manufacturers, field strengths (1.5 T vs 3 T), or acquisition protocols differ significantly in appearance. A model trained on data from one site can degrade substantially at another site.

AI implication: domain adaptation techniques, intensity normalisation, and federated learning strategies are essential for real-world deployment. Multi-site training or test-time adaptation are active research directions.

### Multimodal Fusion

Clinical decisions often integrate multiple modalities simultaneously (e.g. PET/CT for oncology staging, T1+T2+FLAIR for brain tumour grading). Fusion strategies include:

- **Early fusion**: concatenate all modality images as input channels.
- **Late fusion**: run separate encoders per modality, combine at the feature level.
- **Cross-attention fusion**: modality-specific tokens attend to each other (common in transformer-based models).

The choice depends on whether the modalities are spatially registered and whether one modality dominates the decision.

---

## Links

- [[image-quality-metrics]] — each modality has distinct SNR, resolution, and dynamic range characteristics
- [[image-artifacts]] — each modality introduces its own characteristic artefacts (speckle, motion blur, k-space errors)
- [[modality-selection]] — framework for choosing the right modality for a given AI task
- [[sampling-aliasing]] — Nyquist limits apply to spatial resolution in CT and k-space sampling in MRI
- [[sensor-technology]] — detector physics underlies the noise model of CT and fluorescence imaging
- [[sample-preparation-imaging]] — fluorescence microscopy requires staining; contrast agents used in CT/MRI
- [[fourier-transform]] — CT reconstruction and MRI k-space both rely on Fourier analysis
- [[unet]] — the dominant architecture for cell and organ segmentation across modalities
- [[earth-observation-fundamentals]] — hyperspectral imaging bridges medical and remote sensing contexts
