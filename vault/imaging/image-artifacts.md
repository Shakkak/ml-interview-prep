---
title: "Image Artifacts"
tags: [image-artifacts, noise, motion-artifacts, speckle, aliasing, blur]
aliases: [imaging artefacts, image distortions, MRI artifacts, CT artifacts]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, image-quality-metrics]
related: [sampling-aliasing, fourier-transform, data-augmentation]
---

## Fundamental

### Why Artifacts Matter for AI

An artifact is any feature in an image that does not correspond to actual structure in the object being imaged. It is produced by the imaging process itself — physics, hardware, acquisition choices, or patient behaviour.

From an AI perspective, artifacts are dangerous in two ways:
1. **They create false positives**: a model may classify an artifact as pathology.
2. **They corrupt features**: true structures are obscured, causing false negatives.

Artifact-aware preprocessing and training data that includes artifact examples are essential for robust models.

### Noise

Noise is random, unwanted variation superimposed on the true signal. Unlike artifacts from deterministic causes, noise is stochastic — different on every acquisition even if the object is identical.

**Common noise types by modality:**

| Modality | Dominant noise type | Cause |
|----------|---------------------|-------|
| CT | Quantum (Poisson) noise | Finite photon count |
| MRI | Thermal (Rician) noise | Thermal fluctuations in electronics |
| PET | Poisson noise | Finite radiotracer decay events |
| Ultrasound | Speckle (coherent) noise | Interference of scattered waves |
| Fluorescence microscopy | Shot noise + read noise | Photon statistics + detector electronics |

**Poisson noise** scales with signal: $\sigma \propto \sqrt{S}$, so SNR $\propto \sqrt{S}$. Doubling the dose (CT) or exposure time (microscopy) improves SNR by $\sqrt{2}$, not 2×.

**AI implication**: noise augmentation during training (adding synthetic Gaussian or Poisson noise) improves robustness but must match the real noise statistics. See [[data-augmentation]].

### Motion Artifacts

Motion during image acquisition blurs or distorts the image. The effect depends on the acquisition time relative to the motion period.

**CT**: sub-second acquisition; cardiac and respiratory motion cause blurring and streaking at tissue/air interfaces. Gating (synchronising acquisition to heartbeat or breathing) reduces this.

**MRI**: minutes-long acquisition; patient movement between k-space lines causes **ghosting** — repeated copies of the moving structure displaced in the phase-encode direction.

**Fluorescence microscopy**: stage drift or cell movement during z-stack acquisition causes misregistration between slices.

**AI implication**: motion artifacts can mimic pathology (blurring simulates mass lesions) and corrupt ground-truth annotations. Motion-corrupted images should be flagged in the dataset or augmented with synthetic motion blur to encourage robustness.

### Speckle (Ultrasound)

Speckle is not noise in the classical sense — it is a **deterministic but complex** interference pattern caused by coherent scattering from sub-resolution structures. It gives ultrasound images their characteristic grainy texture.

Speckle is correlated: adjacent pixels are not independent. Its spatial pattern changes if the probe angle or position changes, even slightly.

**AI implication**: speckle makes ultrasound images harder to segment than CT/MRI. Classical filters (median, Lee, anisotropic diffusion) reduce speckle at the cost of blurring edges. Deep learning models trained on speckle-corrupted data can learn to ignore it, but require diverse probe-angle examples to generalise.

---

## Intermediate

### Aliasing (Wrap-Around)

Aliasing occurs when the sampling rate is too low relative to the signal's bandwidth. In MRI, if the field of view (FOV) is smaller than the imaged object, tissue outside the FOV wraps around and overlaps with tissue inside — called **wrap-around** or **fold-over**.

In CT, aliasing from insufficient angular or detector sampling appears as **streaking** artefacts.

Full treatment in [[sampling-aliasing]]. Key point for AI: aliasing creates features (overlapping structures) that do not exist in the patient, causing the model to encounter inputs that violate its training distribution.

### Metal and Beam-Hardening Artifacts (CT)

**Metal artifacts**: metal implants (screws, prostheses) absorb X-rays completely, creating dark streaks and bright star-patterns radiating from the metal. The reconstruction algorithm cannot distinguish the implant from the projection data correctly.

**Beam-hardening**: a polychromatic X-ray beam loses its low-energy photons first as it passes through dense tissue. The remaining beam is "hardened" (higher mean energy), causing underestimation of attenuation in the centre of dense objects — dark bands between two dense structures ("cupping artefact").

**AI implication**: metal artefacts are severe enough that standard segmentation models completely fail near implants. Metal artefact reduction (MAR) preprocessing algorithms exist and should be applied before model inference. Training data must include examples with and without metal.

### Ringing / Gibbs Artifact

Ringing (Gibbs phenomenon) appears as oscillating bands near sharp edges. It is caused by truncating the frequency spectrum during reconstruction — the finite bandwidth of the imaging system cannot perfectly represent a step edge, producing the oscillation familiar from Fourier analysis of discontinuities.

Common in MRI near interfaces between CSF and brain tissue.

**AI implication**: ringing creates false signal variations near edges. Edge-based features and boundary-loss segmentation objectives can be confused by these oscillations.

### MRI-Specific Artifacts

**B1 inhomogeneity**: the radiofrequency excitation field is not uniform across large objects, causing spatial variation in image intensity (brighter centre, darker periphery at 3 T). Bias field correction (N4ITK) is standard preprocessing.

**Chemical shift**: fat and water have slightly different resonant frequencies. In frequency-encode direction, fat pixels are displaced relative to water, creating a bright rim on one side and dark rim on the other at fat–water interfaces.

**Susceptibility artefacts**: air–tissue interfaces (sinuses, ear canals) create local magnetic field distortions, causing geometric distortion and signal loss. EPI sequences (used in fMRI and diffusion MRI) are especially susceptible.

**AI implication**: MRI preprocessing pipelines must include bias field correction and, for EPI, geometric distortion correction. Models that skip these steps will encounter systematic spatial intensity patterns that corrupt learned features.

### Photobleaching (Fluorescence Microscopy)

Fluorescent dyes are permanently destroyed by the excitation light. In time-lapse or z-stack acquisitions, signal decreases over time as the fluorophore bleaches. This creates artificial intensity gradients in the z or time dimension.

**AI implication**: bleaching introduces a systematic z-dependent bias. Normalisation per slice or per time point can mitigate it, but deconvolution and photobleaching correction are preferred.

---

## Advanced

### Artifact-Augmented Training

A principled approach to artifact robustness:

1. **Identify** the dominant artifact class for the deployment modality.
2. **Model** the artifact physically (e.g. simulate Poisson noise, synthetic MRI ghosting, random metal insertion in CT).
3. **Augment** training data with synthetic versions.
4. **Evaluate** on held-out artifact-corrupted images to confirm robustness.

This is especially important in medical imaging where artifact prevalence varies between clinical sites.

### Artifact vs. Pathology Confusion

Some artifacts closely mimic pathological findings:

| Artifact | Mimics |
|----------|--------|
| CT streak near metal | Bone lesion or calcification |
| MRI ghosting from pulsatile flow | Vessel duplication or lesion |
| Ultrasound shadowing behind calcification | Deep tissue absence |
| Fluorescence cross-talk between channels | Co-localisation of two proteins |

Models trained without artifact examples will assign pathology probability to these regions. Including radiologist-annotated artifacts as a separate class in the training data is one mitigation.

---

## Links

- [[imaging-modalities-overview]] — each modality has its own characteristic artifact profile
- [[image-quality-metrics]] — SNR and dynamic range determine susceptibility to noise and saturation artifacts
- [[sampling-aliasing]] — aliasing and Gibbs ringing share Fourier-domain roots
- [[fourier-transform]] — beam-hardening, Gibbs ringing, and k-space truncation are best understood in the frequency domain
- [[data-augmentation]] — artifact simulation is a form of data augmentation for robustness
- [[sample-preparation-imaging]] — photobleaching and cross-talk arise from the sample preparation process
