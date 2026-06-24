---
title: "Sampling and Aliasing in Imaging"
tags: [sampling, aliasing, nyquist, k-space, fourier-slice-theorem, reconstruction]
aliases: [Nyquist theorem, aliasing, k-space, Fourier slice theorem, wrap-around artefact]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, fourier-transform]
related: [image-artifacts, image-quality-metrics, imaging-modalities-overview]
---

## Fundamental

### The Sampling Problem

When a continuous physical signal (e.g. X-ray attenuation through tissue) is converted to a digital image, it must be sampled at discrete spatial positions. The central question is: **how densely must we sample to faithfully represent the original signal?**

Too sparse, and high-frequency details are lost or misrepresented — this is **aliasing**. Too dense, and we acquire more data than necessary, increasing acquisition time, dose, and storage.

### The Nyquist–Shannon Sampling Theorem

> A signal whose highest frequency component is $f_\text{max}$ can be perfectly reconstructed from samples taken at a rate $f_s \geq 2 f_\text{max}$.

In imaging, "sampling rate" corresponds to the pixel/voxel size relative to the finest detail in the image. For a pixel size $\Delta x$:

$$f_\text{max,\ representable} = \frac{1}{2\Delta x}$$

This is the **Nyquist frequency**. Any spatial frequency above this limit cannot be correctly represented.

**Intuition**: to represent a sine wave, you need at least two samples per cycle — one at the peak and one at the trough. Fewer than two samples per cycle and the waveform looks like a lower-frequency wave — it has been aliased.

### Aliasing in Images

When spatial frequencies above the Nyquist limit are present in the scene but the sampling rate is insufficient, those frequencies **fold back** into the representable range and appear as spurious low-frequency patterns.

Classic aliasing examples:
- Wagon wheel appearing to spin backward in film (temporal aliasing).
- Moiré patterns when photographing fine fabric or grids.
- Wrap-around in MRI when the imaged object extends outside the field of view.

**AI implication**: if training images are acquired at one resolution and test images at another, the aliasing artefact pattern will differ — creating a domain shift that hurts generalisation.

---

## Intermediate

### Anti-Aliasing: Low-Pass Filtering Before Sampling

The correct solution is to apply a **low-pass filter** (anti-aliasing filter) before sampling, removing all frequencies above $f_\text{max}$ that cannot be correctly represented. This ensures the Nyquist theorem's conditions are met and aliasing is eliminated.

In digital cameras, this is done optically by a slightly blurring filter in front of the sensor. In MRI and CT, the analog signal is band-limited before the ADC (analog-to-digital converter).

**AI implication**: resampling images during preprocessing (e.g. downsampling CT to a coarser resolution) must always be preceded by a low-pass filter (e.g. Gaussian blur) to avoid introducing aliasing. `torch.nn.functional.interpolate` with `antialias=True` handles this correctly.

### k-Space in MRI

MRI does not acquire images directly. The scanner measures the **Fourier transform of the image** — called **k-space**. Each readout line fills one row (or line) of k-space, corresponding to a specific spatial frequency.

The relationship is:

$$k_x = \frac{\gamma}{2\pi} G_x t, \quad k_y = \frac{\gamma}{2\pi} G_y t_\text{phase}$$

where $\gamma$ is the gyromagnetic ratio, $G$ is the applied gradient, and $t$ is time.

The image is recovered by inverse 2D Fourier transform:

$$I(x, y) = \mathcal{F}^{-1}\{S(k_x, k_y)\}$$

**Key insight**: the **centre of k-space** contains low spatial frequencies — contrast, overall brightness. The **periphery of k-space** contains high spatial frequencies — edges, fine detail.

**AI implication for k-space undersampling**: acquiring only a fraction of k-space lines reduces scan time but introduces aliasing. Compressed sensing (CS) MRI and deep learning reconstruction (e.g. E2E-VarNet) recover images from undersampled k-space by exploiting image sparsity priors. These methods must understand which k-space lines are most informative.

### MRI Field of View and Wrap-Around

In MRI, the field of view (FOV) determines the sampling interval in k-space via:

$$\text{FOV} = \frac{1}{\Delta k}$$

If the FOV is smaller than the patient, tissue outside the FOV aliases back inside it — producing **wrap-around** (the side of the patient appears on the opposite edge of the image). This is managed by oversampling or saturation bands.

**AI implication**: wrap-around creates anatomically impossible features (tissue in wrong location). Preprocessing pipelines should check for and crop or suppress wrap-around before inference.

### The Fourier Slice Theorem (CT Reconstruction)

CT reconstruction relies on a fundamental relationship between projection data and the 2D Fourier transform of the object:

> **Fourier Slice Theorem**: the 1D Fourier transform of a projection at angle $\theta$ equals a radial slice through the 2D Fourier transform of the object at the same angle.

Formally: if $p_\theta(t)$ is the projection (line integral of attenuation) at angle $\theta$, then:

$$\mathcal{F}_1\{p_\theta(t)\}(\omega) = \hat{f}(\omega \cos\theta,\ \omega \sin\theta)$$

where $\hat{f}$ is the 2D Fourier transform of the attenuation map $f(x, y)$.

**Consequence**: CT reconstruction is achievable by collecting projections at many angles, computing their 1D Fourier transforms, placing them as radial slices in 2D Fourier space, then applying inverse 2D FFT. In practice, **filtered backprojection (FBP)** implements this efficiently, and iterative methods improve on FBP for sparse or noisy data.

**AI implication**: deep learning CT reconstruction methods operate either in sinogram space (raw projections) or image space (post-FBP), and must be aware of which representation they receive. Models trained on FBP images will encounter different noise textures than models trained on iterative reconstruction outputs.

---

## Advanced

### Compressed Sensing MRI

Compressed sensing (CS) theory shows that if a signal is **sparse** in some basis (e.g. wavelet domain), it can be exactly recovered from far fewer measurements than Nyquist would require, provided:

1. The sampling is **incoherent** (random-like in k-space).
2. A **sparsity-promoting** reconstruction is used (e.g. $\ell_1$ minimisation).

CS MRI reduces scan times by 4–10× by acquiring a random subset of k-space lines and solving the reconstruction as an optimisation problem.

**AI implication**: deep learning unrolled networks (e.g. MoDL, E2E-VarNet) combine the k-space data consistency of CS with learned priors, achieving better reconstruction than classical CS at the same undersampling factor. Training such networks requires paired fully-sampled / undersampled k-space data.

### Partial Fourier Acquisition

MRI exploits the **Hermitian symmetry** of k-space: for a real-valued image, $S(-k) = S^*(k)$. This means only slightly more than half of k-space needs to be acquired — the rest can be estimated by conjugate symmetry. This halves acquisition time at the cost of slight SNR reduction.

**AI implication**: partial Fourier images have subtly different noise and phase characteristics from full-Fourier acquisitions. A reconstruction network trained on full-Fourier data may not generalise to partial-Fourier inputs without retraining.

---

## Links

- [[fourier-transform]] — k-space, Fourier slice theorem, and anti-aliasing filters are all rooted in Fourier analysis
- [[imaging-modalities-overview]] — CT and MRI both depend on Fourier-domain sampling; aliasing appears differently in each
- [[image-artifacts]] — aliasing, wrap-around, and Gibbs ringing are the primary Fourier-domain artefacts
- [[image-quality-metrics]] — spatial resolution is bounded by the Nyquist limit; PSF blurring is a consequence of finite k-space extent
