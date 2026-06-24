---
title: "Image Quality Metrics"
tags: [image-quality, snr, dynamic-range, spatial-resolution, contrast, psf]
aliases: [SNR, signal-to-noise ratio, image resolution, dynamic range, point spread function]
difficulty: 1
status: complete
depends_on: [imaging-modalities-overview]
related: [image-artifacts, sampling-aliasing, fourier-transform]
---

## Fundamental

### Why Image Quality Metrics Matter for AI

A deep learning model sees exactly what the sensor provides — no more. If the input image has low SNR, the model must either learn to ignore noise or it will learn to fit it. If resolution is insufficient to resolve the structure of interest, no amount of model capacity can recover that information. Understanding image quality is prerequisite to understanding why a model fails on a given modality.

### Signal-to-Noise Ratio (SNR)

**SNR** measures how much of the measured signal is true information versus unwanted random variation (noise).

$$\text{SNR} = \frac{\mu_s}{\sigma_n}$$

where $\mu_s$ is the mean signal intensity in a region of interest and $\sigma_n$ is the standard deviation of the noise (measured in a uniform background region).

Intuition: an SNR of 10 means the signal is 10× larger than the noise fluctuations. An SNR of 1 means signal and noise are indistinguishable.

**AI implication**: low SNR forces models to average over many examples to learn reliable features. Data augmentation that adds noise during training can improve robustness, but cannot substitute for sufficient SNR in the test data.

### Contrast

Contrast describes the difference in signal intensity between two adjacent regions (e.g. tumour vs. surrounding tissue):

$$C = \frac{S_1 - S_2}{S_1 + S_2}$$

or simply $\Delta S = S_1 - S_2$ for additive comparisons.

A structure may be large and well-resolved but invisible if its contrast against the background is zero. Contrast agents (iodine in CT, gadolinium in MRI) are injected to artificially boost contrast of target structures.

**AI implication**: models trained with contrast-enhanced images may fail on non-enhanced acquisitions of the same anatomy, and vice versa. Dataset curation must track contrast agent use.

### Dynamic Range

**Dynamic range** is the ratio between the maximum and minimum signal that the sensor can meaningfully record:

$$\text{DR} = \frac{I_\text{max}}{I_\text{min}}$$

A sensor with high dynamic range can capture bright and dark regions simultaneously without saturating or losing low-signal detail. CT has a very wide dynamic range (covering air to metal implants). Optical microscopes have a narrower range — bright fluorescent spots can saturate the detector while dim regions are lost in noise.

**AI implication**: images acquired outside the sensor's dynamic range lose information at either end. Clipping (saturation) is irreversible. Histogram normalisation in preprocessing must account for the actual dynamic range used.

---

## Intermediate

### Spatial Resolution and the Point Spread Function (PSF)

Spatial resolution describes the smallest resolvable detail. No real imaging system is perfect — a point source (infinitely small object) is recorded not as a single pixel but as a blurred blob. This blur is characterised by the **Point Spread Function (PSF)**.

In the Fourier domain, blurring is multiplication by the **Modulation Transfer Function (MTF)**:

$$I_\text{observed} = I_\text{true} * \text{PSF}$$

equivalently (in frequency space):

$$\hat{I}_\text{observed}(\mathbf{k}) = \hat{I}_\text{true}(\mathbf{k}) \cdot \text{MTF}(\mathbf{k})$$

The MTF describes how faithfully the system transfers each spatial frequency from the true scene to the image. High spatial frequencies (fine detail) are attenuated by the PSF before they are recorded.

**Practical values:**

| Modality | Approximate spatial resolution |
|----------|-------------------------------|
| CT (clinical) | 0.5–1 mm in-plane |
| MRI (clinical) | 0.5–2 mm in-plane |
| PET | 4–6 mm FWHM |
| Ultrasound | 0.1–1 mm (frequency-dependent) |
| Fluorescence microscopy (confocal) | 0.2–0.5 µm lateral |
| Hyperspectral (drone) | cm–m spatial |

**AI implication**: structures smaller than the PSF FWHM cannot be resolved and appear merged. Super-resolution models attempt to recover sub-pixel detail, but they can only do so if the blurring is known and the training distribution is matched.

### Temporal Resolution

Temporal resolution is the shortest time interval between two acquired images. It is critical whenever the imaged object moves:

- **Cardiac MRI**: must capture full cardiac cycle in ≤ 1 s per phase.
- **Echocardiography (ultrasound)**: 30–100 frames/second; sufficient for motion.
- **CT**: single rotation ~0.3 s; gated CT captures cardiac phase.
- **Fluorescence microscopy (live imaging)**: limited by photobleaching rate.

**AI implication**: motion during acquisition creates blurring or ghosting artefacts (see [[image-artifacts]]). Models designed for static anatomy may fail on images with motion blur. Temporal models must match the frame rate of the acquisition.

### Spectral Resolution

Spectral resolution describes how finely the sensor discriminates wavelengths (or energies):

- **Grayscale CT/MRI/ultrasound**: single channel — no spectral resolution.
- **RGB camera**: 3 broad channels.
- **Multispectral (e.g. Sentinel-2)**: 10–13 discrete bands.
- **Hyperspectral**: 100–400+ narrow bands (~10 nm bandwidth).
- **Dual-energy CT**: two X-ray energy levels, enabling material decomposition.

**AI implication**: richer spectral information enables finer discrimination (e.g. tumour subtype by spectral signature) but at the cost of higher-dimensional inputs and greater demand for labelled training data. See [[fourier-transform]] for spectral analysis tools.

### Contrast-to-Noise Ratio (CNR)

A more diagnostically useful metric than SNR alone:

$$\text{CNR} = \frac{|S_1 - S_2|}{\sigma_n}$$

where $S_1$ and $S_2$ are the mean signals in two regions of interest and $\sigma_n$ is the noise standard deviation. CNR combines both contrast and noise into a single detectability measure.

**AI implication**: CNR is a better predictor of model performance on detection tasks than SNR alone, because it captures whether the target structure can be distinguished from background, not just whether signal exceeds noise.

---

## Advanced

### Resolution–Dose Trade-off in CT

In CT, spatial resolution and radiation dose are coupled. Higher dose → more photons → lower noise → better low-contrast detectability. Manufacturers use iterative reconstruction and deep learning reconstruction (e.g. GE's AIR Recon DL) to achieve lower noise at reduced dose, effectively improving the resolution–dose trade-off.

**AI implication**: low-dose CT images have different noise texture than standard-dose images. A model trained on standard-dose CT can fail on low-dose CT from the same scanner. Domain-aware training or paired denoising networks are required.

### Noise Power Spectrum

Rather than characterising noise by a single number ($\sigma_n$), the Noise Power Spectrum (NPS) characterises noise as a function of spatial frequency:

$$\text{NPS}(f) = |\hat{n}(f)|^2$$

where $\hat{n}(f)$ is the Fourier transform of the noise. Structured noise (e.g. MRI ghosting from periodic motion) produces peaks at specific frequencies; white noise is flat.

**AI implication**: convolutional models are sensitive to the spatial structure of noise, not just its amplitude. A model trained on white noise may be brittle to structured (correlated) noise from a different scanner reconstruction kernel.

### Isotropy and Anisotropic Voxels

Clinical 3D images are often **anisotropic**: the slice thickness (z-direction) is larger than the in-plane pixel size. For example, CT may have 0.5 mm in-plane but 3 mm slice thickness. This means the volume has lower resolution in one axis.

**AI implication**: 3D CNNs applied to anisotropic volumes must use anisotropic kernels or resample to isotropic resolution first. Resampling introduces interpolation artefacts and must be handled consistently between training and inference.

---

## Links

- [[imaging-modalities-overview]] — each modality has distinct SNR, dynamic range, and resolution characteristics
- [[image-artifacts]] — low SNR, PSF blur, and dynamic range limits are the physical root of most imaging artefacts
- [[sampling-aliasing]] — spatial resolution is bounded by the Nyquist limit; aliasing occurs when sampling is coarser than the PSF allows
- [[fourier-transform]] — MTF and NPS are defined in the frequency domain; spatial resolution analysis uses Fourier tools
- [[modality-selection]] — quality metrics determine which modality is feasible for a given task
