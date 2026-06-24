---
title: "Modality Selection for AI Tasks"
tags: [modality-selection, imaging-tradeoffs, medical-imaging-ai, computer-vision-applications]
aliases: [choosing imaging modality, modality comparison, which imaging modality]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, image-quality-metrics]
related: [image-artifacts, unet, earth-observation-fundamentals, sample-preparation-imaging]
---

## Fundamental

### The Core Question

Given an AI task (classify, detect, segment, track), which imaging modality should you use? The answer depends on three things:

1. **What must be visible**: can the target structure be seen in this modality? Does it have sufficient contrast and resolution?
2. **What constraints exist**: cost, availability, acquisition time, patient safety (radiation), sample preparation, real-time requirements.
3. **What the AI pipeline requires**: 2D vs 3D, labelling feasibility, dataset size, inference speed.

No single modality is universally best. The goal is to match modality characteristics to task requirements.

### The Modality Trade-off Table

| Modality | Soft tissue contrast | Bone/hard tissue | Functional info | Resolution | Acquisition time | Real-time | Radiation | Cost |
|----------|---------------------|-----------------|-----------------|------------|-----------------|-----------|-----------|------|
| CT | Low–Medium | Excellent | No | High | Seconds | No | Yes | Medium |
| MRI | Excellent | Poor | Partial (fMRI, DWI) | High | Minutes | No | No | High |
| PET | None | None | Excellent | Low | 10–30 min | No | Yes | Very high |
| Ultrasound | Medium | Poor (shadowing) | Partial (Doppler) | Medium | Real-time | Yes | No | Low |
| Fluorescence microscopy | Excellent (labelled) | — | Partial | Sub-µm | Seconds–minutes | Partial | No | Medium |
| Hyperspectral | Spectral | Spectral | Spectral | Spatial: low | Seconds–minutes | No | No | Medium–high |

---

## Intermediate

### Worked Examples: Matching Task to Modality

**Brain tumour grading**
- Need: soft tissue contrast, 3D volume, multiple tissue types
- Answer: **MRI (T1+T2+FLAIR)** → excellent soft tissue contrast, no radiation, multi-contrast enables tumour sub-region segmentation (oedema, enhancing core, necrosis)
- AI approach: 3D [[unet]] with early fusion of 4 MRI channels
- Wrong choice: CT (insufficient soft tissue contrast for white/grey matter distinction)

**Lung nodule screening**
- Need: high spatial resolution 3D volume, wide dynamic range (air to tissue)
- Answer: **CT** → fast, high resolution, lung nodules appear as high-density spots against low-density air
- AI approach: 3D CNN or 2.5D sliding-window detector
- Wrong choice: MRI (motion from breathing degrades quality; slow acquisition; expensive)

**Cell counting and segmentation (in vitro)**
- Need: sub-cellular resolution, cell-type specificity, multi-structure discrimination
- Answer: **Fluorescence microscopy** → each cell structure labelled with a specific fluorophore; sub-micron resolution
- AI approach: [[unet]]-based instance segmentation per channel
- Wrong choice: bright-field only (insufficient contrast for individual organelles)

**Precision agriculture — crop disease detection**
- Need: large field coverage, spectral discrimination between healthy and diseased tissue
- Answer: **Hyperspectral drone imaging** → disease changes spectral reflectance profile before visible colour changes appear
- AI approach: 1D CNN or SVM on spectral features after PCA reduction; see [[earth-observation-fundamentals]]
- Wrong choice: RGB camera (too few spectral bands to detect early-stage disease)

**Cardiac wall motion assessment**
- Need: real-time imaging, temporal resolution to capture cardiac cycle, no breath-hold
- Answer: **Ultrasound (echocardiography)** → real-time, 30–100 fps, portable, no radiation
- AI approach: optical flow or recurrent network on cine sequences
- Wrong choice: MRI (gated cardiac MRI is possible but slow; requires patient to hold breath)

**Oncology staging — tumour metabolic activity**
- Need: functional/metabolic information about tumour viability, whole-body coverage
- Answer: **PET/CT** → PET provides metabolic signal (FDG uptake), CT provides anatomical localisation
- AI approach: multimodal fusion network (PET + CT channels)
- Wrong choice: CT alone (cannot distinguish metabolically active from necrotic tumour)

### The Resolution–Contrast–Time Triangle

Three quantities constantly trade off:

- **Higher resolution** → more data to acquire → longer scan → more motion blur
- **Better contrast** (lower noise) → more signal averaging → longer scan
- **Faster acquisition** → fewer projections/k-space lines → lower resolution or higher noise

Every modality optimises this triangle differently for its physics. AI practitioners must understand which corner of the triangle a given dataset occupies, because it determines which artefacts are present and what spatial scale of features are learnable.

### Key Questions Before Choosing a Modality for an AI Task

1. **Can the target be seen?** Check that the structure has sufficient contrast ($\text{CNR} > 1$ as a rule of thumb) in the candidate modality.
2. **What resolution is needed?** The smallest structure of interest must be at least 2× larger than the pixel size (Nyquist).
3. **Is acquisition time compatible with motion?** If the object moves faster than the acquisition time, motion artefacts will corrupt the image.
4. **Is the modality available at scale?** Deep learning requires thousands of labelled examples. PET/CT datasets are rare; CT and ultrasound datasets are more common.
5. **Is radiation acceptable?** CT and PET involve ionising radiation — repeat scanning for longitudinal studies may be restricted.
6. **Is sample preparation feasible?** Fluorescence microscopy requires staining (see [[sample-preparation-imaging]]); contrast agents in CT/MRI require IV access and may be contraindicated.

---

## Advanced

### Multi-Modality Fusion Decisions

When multiple modalities are available, the decision is no longer which to use but **how to combine them**:

- **PET/CT**: PET provides metabolic signal; CT provides anatomy. Always fuse — PET alone is uninterpretable. Rigid registration is usually sufficient (brain); deformable registration needed for thorax/abdomen.
- **MRI multi-sequence**: T1, T2, FLAIR, DWI carry complementary pathological information. Early fusion (stack as channels) is standard for segmentation. Late fusion can be used when sequences are missing at test time.
- **Hyperspectral + RGB**: spectral data complements spatial detail; RGB is cheaper and more available. Spectral attention modules can select the most informative spectral bands given a spatial query.

### Dataset Availability vs. Modality Optimality

The theoretically best modality for a task may not be the right choice for AI development if labelled datasets are too small. Factors:

- **CT and ultrasound** have the largest public datasets (NLST, LUNA16, TCGA-CT, CheXpert for X-ray).
- **MRI** has moderate datasets (BraTS, ACDC, UK Biobank).
- **PET/MRI fusion** and **hyperspectral** datasets are scarce; transfer learning from CT/RGB backbones is common.
- **Fluorescence microscopy** has cell-specific datasets (Cell Tracking Challenge, Broad Bioimage Benchmark).

When the optimal modality has insufficient labelled data, domain adaptation from a data-rich modality is an active research area.

---

## Links

- [[imaging-modalities-overview]] — reference table of modality characteristics
- [[image-quality-metrics]] — SNR, CNR, and resolution determine whether a structure is visible in a given modality
- [[image-artifacts]] — modality-specific artefact profiles affect which preprocessing steps are required
- [[unet]] — the standard architecture for segmentation tasks across most modalities
- [[earth-observation-fundamentals]] — hyperspectral and SAR modalities in remote sensing context
- [[sample-preparation-imaging]] — preparation requirements for fluorescence microscopy and contrast-enhanced CT/MRI
