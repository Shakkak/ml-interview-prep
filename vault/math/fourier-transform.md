---
title: Fourier Transform for Computer Vision
tags: [signal-processing, frequency-domain, image-processing, mathematics, fft, convolution-theorem]
aliases: [DFT, FFT, frequency analysis, 2D Fourier, convolution theorem, power spectrum, Gabor filter]
difficulty: 1
status: complete
related: [attention-mechanism, backpropagation-advanced, spectral-bias, dilated-convolution, feature-preprocessing]
---

# Fourier Transform for Computer Vision

---

## Fundamental

Any periodic signal can be decomposed into a sum of sinusoids at integer multiples of a base frequency. The **Discrete Fourier Transform (DFT)** makes this precise for finite signals.

**1D DFT:** given signal $f[n]$ for $n = 0, \ldots, N-1$:

$$F[k] = \sum_{n=0}^{N-1} f[n]\, e^{-2\pi ikn/N}, \quad k = 0, 1, \ldots, N-1$$

$F[k]$ is the complex amplitude at frequency $k/N$ cycles per sample. $|F[k]|$ is the energy at that frequency; $\angle F[k]$ is the phase.

**Inverse DFT:**
$$f[n] = \frac{1}{N}\sum_{k=0}^{N-1} F[k]\, e^{2\pi ikn/N}$$

**Parseval's theorem:** $\sum_n |f[n]|^2 = \frac{1}{N}\sum_k |F[k]|^2$ — total energy is preserved. The DFT is an isometric (energy-preserving) change of basis.

**Worked example:** $f = [1, 2, 3, 4]$, $N=4$.

$F[0] = 1+2+3+4 = 10$ (DC = sum of signal).
$F[1] = 1 + 2e^{-i\pi/2} + 3e^{-i\pi} + 4e^{-3i\pi/2} = 1 - 2i - 3 + 4i = -2 + 2i$.
$F[2] = 1 - 2 + 3 - 4 = -2$ (alternating pattern = Nyquist frequency).
$F[3] = \overline{F[1]} = -2 - 2i$ (conjugate symmetry for real signals).

### The Convolution Theorem

$$h = f * g \quad \Longleftrightarrow \quad H[k] = F[k] \cdot G[k]$$

**Convolution in space/time equals pointwise multiplication in frequency.** This is the most important property: an $O(N^2)$ spatial convolution becomes $O(N)$ pointwise multiplication after the FFT. Large blurs (wide [[distributions-gaussian|Gaussian]] kernel, motion blur) are always computed via FFT in practice.

---

## Intermediate

### 2D DFT and the $1/f^2$ Power Law

The 2D DFT of an image $f[m,n] \in \mathbb{R}^{M \times N}$:

$$F[k, l] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f[m,n]\, e^{-2\pi i(km/M + ln/N)}$$

After `fftshift` (center the DC component):
- **Center** = DC (average brightness)
- **Distance from center** = spatial frequency (farther = faster spatial variation)
- **Orientation** = perpendicular to image structure (vertical edges → horizontal lines in spectrum)

**Empirical law of natural images:** power spectral density $P(f) \propto 1/f^2$. Low frequencies carry far more energy than high. Consequences:
- A random pixel image has flat spectrum ($P \propto f^0$) — looks nothing like a photo
- JPEG compression (discarding high-frequency DCT coefficients) works because those coefficients are near zero for natural images
- CNNs capture most image information efficiently with local filters — their RF sizes are matched to where image energy lives

**Frequency-domain filtering:**
| Filter | Effect | Spatial equivalent |
|---|---|---|
| Gaussian (low-pass) | Attenuates high frequencies | Smooth/blur |
| High-pass | Preserves edges, removes DC | Sharpening |
| Band-pass | Selects a frequency ring | Gabor filter |
| Notch | Removes specific frequency | Periodic noise removal |

Gaussian blur: $g(x) = e^{-x^2/2\sigma^2}$ has DFT $G(f) \propto e^{-2\pi^2\sigma^2 f^2}$ — also a Gaussian. Wider spatial Gaussian $\uparrow$ $\sigma$ → narrower frequency Gaussian → more aggressive low-pass. This is the **spatial-frequency duality (uncertainty principle)**: you cannot simultaneously have a filter that is narrow in both space and frequency.

### The Fast Fourier Transform (FFT)

Naive DFT: $O(N^2)$ — 16M operations for $N = 4096$.

**Cooley-Tukey FFT (1965):** $O(N\log N)$ via divide and conquer. Split into even/odd indexed halves:

$$F[k] = \underbrace{\sum_{n=0}^{N/2-1} f[2n]\, e^{-2\pi ikn/(N/2)}}_{E[k]} + e^{-2\pi ik/N} \underbrace{\sum_{n=0}^{N/2-1} f[2n+1]\, e^{-2\pi ikn/(N/2)}}_{O[k]}$$

**Butterfly operation:** $F[k] = E[k] + W^k O[k]$ and $F[k+N/2] = E[k] - W^k O[k]$ where $W = e^{-2\pi i/N}$.

Recursively apply: $T(N) = 2T(N/2) + O(N) \Rightarrow T(N) = O(N\log N)$.

For $N = 4096$: from $\sim 16M$ multiplications to $\sim 49K$ — a $\sim 330\times$ speedup. The butterfly diagram has $\log_2 N$ stages, each with $N/2$ butterfly operations.

### Circular Convolution and Zero-Padding

The DFT assumes the signal is **periodic** — the end wraps to the beginning. FFT-based multiplication of DFTs gives **circular convolution**, not linear convolution.

For signal length $M$ and kernel length $K$, linear convolution has length $M + K - 1$. **Fix:** zero-pad both $f$ and $g$ to length $\geq M + K - 1$ before taking DFTs. This ensures wrap-around contributions land outside the valid region.

Without zero-padding: applying a Gaussian blur via FFT produces faint edge artifacts where opposite edges bleed into each other — especially visible when there's strong intensity contrast at image boundaries.

---

## Advanced

### Gabor Filters and the Heisenberg-Gabor Limit

A Gabor filter is a sinusoid modulated by a Gaussian envelope:
$$g(x, y; f_0, \theta, \sigma) = \exp\!\left(-\frac{x_\theta^2 + \gamma^2 y_\theta^2}{2\sigma^2}\right)\cos(2\pi f_0 x_\theta + \psi)$$

where $(x_\theta, y_\theta)$ are coordinates rotated by angle $\theta$.

**Why Gabor achieves the fundamental optimum:** the Heisenberg-Gabor uncertainty principle states:
$$\Delta x \cdot \Delta f \geq \frac{1}{4\pi}$$

No filter can have smaller joint position-frequency uncertainty. The Gabor filter achieves **exact equality** — it is the unique filter minimizing $\Delta x \cdot \Delta f$. This makes Gabor filters optimal for detecting oriented edges at specific spatial frequencies.

**Biological and deep learning convergence:** V1 simple cells in the visual cortex have Gabor-like receptive fields — modeled since Marcelja (1980). AlexNet's first conv layer spontaneously learns Gabor-like filters from data, with no explicit inductive bias for this structure. The visual system and CNNs independently converge on the same optimal representation, because the $1/f^2$ natural image statistics reward exactly the joint space-frequency tuning that Gabors provide.

### DCT and JPEG Compression

The Discrete Cosine Transform (DCT-II):
$$C[k] = \sum_{n=0}^{N-1} f[n]\cos\!\left(\frac{\pi k(2n+1)}{2N}\right)$$

JPEG applies the 2D DCT on $8 \times 8$ blocks:
1. Subtract 128 (center around zero)
2. 2D DCT-II → 64 frequency coefficients
3. Quantize by dividing by a quantization matrix (larger divisors for high frequencies → more lossy)
4. Zig-zag scan + run-length encode + Huffman code

**Why DCT works for compression:** the $1/f^2$ power law means high-frequency DCT coefficients are near zero for natural images and can be quantized coarsely with minimal perceptual loss. JPEG at 10:1 compression discards ~90% of the information while remaining visually acceptable.

**DCT vs DFT:** the DFT assumes periodicity, creating Gibbs ringing at $8\times8$ block boundaries (JPEG blocking artifacts at high compression). The DCT implicitly applies even-symmetric extension, reducing spectral leakage within blocks. DCT also produces purely real coefficients, avoiding the redundancy of storing both real and imaginary parts.

### Spectrogram and Mel Frequency Representation

The DFT gives global frequency content — it cannot tell *when* a frequency occurs. The **Short-Time Fourier Transform (STFT):**

$$S[m, k] = \sum_n f[n]\, w[n - mH]\, e^{-2\pi ikn/N}$$

where $w$ is a window function (Hann reduces spectral leakage) and $H$ is the hop size.

**Mel spectrogram:** apply triangular filterbanks spaced on the mel scale $m = 2595\log_{10}(1 + f/700)$. The mel scale is perceptually uniform — equal steps on the mel scale correspond to equal steps in perceived pitch. At low frequencies (< 500 Hz), mel spacing is nearly linear; at high frequencies (> 4 kHz), it is logarithmic.

Modern audio deep learning (Whisper, wav2vec 2.0, AudioMAE) uses 80-channel log-mel spectrograms as inputs. The mel compression reduces the 80,000-dimensional raw audio (1 second at 16 kHz) to a 80×100 matrix (80 mel bins × 100 time frames) — a 10:1 compression that discards frequencies the ear cannot distinguish.

**Connection to [[spectral-bias]]:** neural networks trained with gradient descent learn low-frequency components of the target function first (neural tangent kernel theory). This is the Fourier-domain interpretation of why networks overfit slowly and why early stopping works — it stops training before high-frequency noise components are fit.

---

*See also: [[spectral-bias]] · [[dilated-convolution]] · [[feature-preprocessing]] · [[attention-mechanism]] · [[distributions-gaussian]]*
