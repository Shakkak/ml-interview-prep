---
title: "Sensor Technology for Imaging"
tags: [sensor-technology, ccd, cmos, pixel-size, dynamic-range, detector-physics]
aliases: [CCD vs CMOS, image sensor, detector, camera sensor, photodetector]
difficulty: 2
status: complete
depends_on: [imaging-modalities-overview, image-quality-metrics]
related: [image-artifacts, sample-preparation-imaging]
---

## Fundamental

### Why Sensor Technology Matters for AI

The sensor converts physical energy (photons, X-rays, acoustic waves) into digital numbers. Its properties directly determine the noise statistics, dynamic range, and spatial resolution of the resulting image — properties that propagate into every downstream AI model.

A model trained on images from a CCD sensor will see different noise characteristics than one trained on CMOS data from the same scene. Understanding detector physics helps anticipate these differences and design appropriate preprocessing.

### CCD vs CMOS: The Dominant Sensor Technologies

Both CCD (Charge-Coupled Device) and CMOS (Complementary Metal-Oxide-Semiconductor) sensors convert photons to electrons via the photoelectric effect, then quantise the resulting charge. They differ in how charge is read out.

| Property | CCD | CMOS |
|----------|-----|------|
| Readout | Charge shifted across chip to single amplifier | Each pixel has its own amplifier |
| Read noise | Very low (single amplifier) | Higher (per-pixel amplifier variation) |
| Dynamic range | High | Moderate–high (improving rapidly) |
| Speed | Slower (serial readout) | Fast (parallel readout) |
| Power consumption | High | Low |
| Rolling vs global shutter | Global | Rolling (usually) or global (expensive) |
| On-chip processing | No | Yes (ADC, correction on chip) |
| Cost | Higher | Lower |
| Dominant use | Scientific microscopy, astronomy | Consumer cameras, medical endoscopes, mobile |

**Rolling shutter** (most CMOS): rows are read out sequentially, so a fast-moving object is captured at slightly different times per row — causing a "jelly" distortion in the image.

**Global shutter**: all pixels captured simultaneously — no rolling distortion. Required for high-speed imaging.

**AI implication**: rolling shutter distortion is a systematic artefact in CMOS-based video data. Detection or tracking models applied to sports or surgical video must be trained or augmented with rolling shutter examples.

### Pixel Size and Its Trade-offs

Pixel size $\Delta x$ (e.g. 5 µm × 5 µm in a fluorescence microscope camera) determines:

- **Spatial resolution**: smaller pixels can sample finer detail, but only if the optics (PSF) support it.
- **Photon collection**: larger pixels collect more photons → higher SNR per pixel. Smaller pixels → noisier per pixel.
- **Dynamic range**: a larger pixel holds more electrons before saturating (higher well depth).

**Oversampling**: pixel size much smaller than the optical PSF. Wastes pixels but oversampled images can be binned to increase SNR.  
**Undersampling**: pixel size larger than the optical PSF. Loses spatial information irreversibly (aliasing).

The Nyquist criterion applied to pixel size: $\Delta x \leq \frac{1}{2} \cdot \text{PSF FWHM}$ (see [[sampling-aliasing]]).

### Exposure and Saturation

**Exposure time** controls how many photons are collected per pixel. Longer exposure → more signal → better SNR — but also more motion blur and risk of saturation.

**Saturation** (pixel full well capacity exceeded): all photons above the saturation level register as the same maximum value, clipping the high end of the dynamic range. Saturation is irreversible — the actual signal is lost.

**AI implication**: saturated pixels break intensity-based features. Dataset inspection should check histogram tails for evidence of clipping. Training images with saturation artefacts can confuse models unless they are explicitly labelled or augmented.

---

## Intermediate

### Quantum Efficiency (QE)

Quantum efficiency is the fraction of incident photons that produce a detected photoelectron:

$$\text{QE}(\lambda) = \frac{\text{photoelectrons detected}}{\text{photons incident}}$$

QE is wavelength-dependent — sensors are more sensitive to some colours than others. Scientific CMOS sensors can achieve QE > 95% at peak wavelength; standard CMOS sensors peak around 50–70%.

**AI implication**: QE curves vary between cameras. An image of the same scene taken with two different sensors will have different channel brightness ratios. Multi-camera datasets require radiometric normalisation before training across sensors.

### Read Noise, Dark Current, and Shot Noise

Three independent noise sources add in quadrature:

$$\sigma_\text{total}^2 = \sigma_\text{read}^2 + \sigma_\text{dark}^2 + \sigma_\text{shot}^2$$

- **Read noise** ($\sigma_\text{read}$): electronic noise introduced during analog-to-digital conversion. Independent of exposure time. Dominant at very low light levels. Lower in CCD than standard CMOS.
- **Dark current** ($\sigma_\text{dark}$): thermally generated electrons even with no light. Scales with exposure time and temperature. Cooled cameras (scientific CCD/sCMOS) reduce dark current dramatically.
- **Shot noise** ($\sigma_\text{shot} = \sqrt{N}$ where $N$ is photon count): quantum noise from the discrete nature of photons. Unavoidable. Dominant at moderate to high light levels.

**AI implication**: the noise model determines which denoising approach is appropriate. Shot noise (Poisson) is signal-dependent; read noise (Gaussian) is signal-independent. Deep denoising networks (e.g. Noise2Void) must match the assumed noise model.

### X-ray and CT Detectors

CT scanners use scintillator-based detectors: X-rays hit a scintillator crystal (e.g. GOS, CsI), which emits visible light, which is then detected by a photodiode array. Each detector element (bin) integrates photons over the acquisition time.

**Energy-integrating detectors** (standard): sum the energy of all detected X-ray photons. Higher-energy photons contribute more signal, introducing beam-hardening effects (see [[image-artifacts]]).

**Photon-counting detectors** (emerging): count individual photons and record their energy. Enables spectral CT (multiple energy bins), eliminates electronic noise, improves low-contrast detectability.

**AI implication**: models trained on energy-integrating CT data may not transfer to photon-counting CT data without retraining, because noise statistics and spatial resolution characteristics differ fundamentally.

### Ultrasound Transducers

Ultrasound uses piezoelectric transducers that convert electrical pulses to acoustic waves and vice versa. The frequency of the transducer determines the resolution–penetration trade-off:

- **High frequency** (10–15 MHz): fine spatial resolution (~0.1 mm), shallow penetration (~3–4 cm). Used for superficial structures (thyroid, breast, skin).
- **Low frequency** (2–5 MHz): coarser resolution (~1 mm), deep penetration (~15–20 cm). Used for abdominal organs.

**AI implication**: ultrasound images from different transducer frequencies have different spatial scales of features. A model trained on high-frequency superficial images will not generalise to low-frequency abdominal images. Frequency must be recorded as metadata.

---

## Advanced

### Fixed-Pattern Noise and Flat-Field Correction

CMOS sensors have **fixed-pattern noise (FPN)**: systematic pixel-to-pixel sensitivity variations due to manufacturing differences in per-pixel amplifiers. This appears as a faint fixed grid or striping pattern in images.

Correction: flat-field correction divides each image by a uniformly-illuminated reference image (flat-field), removing pixel-to-pixel sensitivity variation:

$$I_\text{corrected}(x, y) = \frac{I_\text{raw}(x, y) - I_\text{dark}(x, y)}{I_\text{flat}(x, y) - I_\text{dark}(x, y)}$$

**AI implication**: raw microscopy images that have not been flat-field corrected contain systematic spatial intensity gradients. Training a model on uncorrected data will cause it to learn position-dependent features — the pixel value at a given spatial location is partly a function of where in the frame it falls, not just what object is there.

---

## Links

- [[imaging-modalities-overview]] — each modality uses different sensor physics; sensor type determines noise model
- [[image-quality-metrics]] — SNR, dynamic range, and spatial resolution are directly set by sensor properties
- [[image-artifacts]] — rolling shutter, fixed-pattern noise, saturation, and dark current are sensor-origin artefacts
- [[sampling-aliasing]] — pixel size relative to PSF determines whether the system is oversampled or undersampled
- [[sample-preparation-imaging]] — fluorescence microscopy sensors must match the emission spectra of the fluorophores used
